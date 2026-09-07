#include "gemm2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static void usage(void)
{
    printf(
"gemm2d — distributed generic GEMM with 2D / 2.5D decomposition\n"
"\n"
"  --M --N --K <int>       matrix dimensions (default 512)\n"
"  --alpha --beta <float>  C := alpha*A*B + beta*C (default 1, 0)\n"
"  --engine <name>         naive1d | cannon | summa | summa25d (default summa)\n"
"  --bcast <name>          blocking | ibcast | shm      (default blocking)\n"
"  --kernel <name>         naive | blocked | packed     (default packed)\n"
"  --gridmap <name>        linear | nodeaware           (default linear)\n"
"  --Pr --Pc <int>         process grid (default: aspect-matched)\n"
"  --c <int>               2.5D replication depth       (default 1)\n"
"  --b <int>               SUMMA panel width            (default 128)\n"
"  --lookahead <0|1>       one step of lookahead (ibcast only)\n"
"  --plan                  let the cost model choose Pr, Pc, c, b\n"
"  --plan-report           print the top configurations and exit\n"
"  --plan-P <int>          hypothetical P for --plan-report\n"
"  --calibrate             measure alpha, beta, gamma before planning\n"
"  --mem-gb <float>        usable memory per rank for the planner (2.0)\n"
"  --verify <name>         none | ref | freivalds       (default none)\n"
"  --trials <int>          Freivalds trials             (default 2)\n"
"  --reps <int>            timed repetitions            (default 3)\n"
"  --warmup <int>          untimed repetitions          (default 1)\n"
"  --seed <int>            generator seed               (default 1)\n"
"  --csv <file>            append one row per repetition\n"
"  --tag <string>          free-form label copied into the CSV\n"
"  --quiet                 suppress the human-readable summary\n");
}

static engine_t parse_engine(const char *s)
{
    if (!strcmp(s, "naive1d"))  return ENG_NAIVE1D;
    if (!strcmp(s, "cannon"))   return ENG_CANNON;
    if (!strcmp(s, "summa25d")) return ENG_SUMMA25D;
    return ENG_SUMMA;
}
static bcast_t parse_bcast(const char *s)
{
    if (!strcmp(s, "ibcast")) return BC_IBCAST;
    if (!strcmp(s, "shm"))    return BC_SHM;
    return BC_BLOCKING;
}
static kernel_t parse_kernel(const char *s)
{
    if (!strcmp(s, "naive"))   return KRN_NAIVE;
    if (!strcmp(s, "blocked")) return KRN_BLOCKED;
    return KRN_PACKED;
}
static verify_t parse_verify(const char *s)
{
    if (!strcmp(s, "ref"))       return VER_REF;
    if (!strcmp(s, "freivalds")) return VER_FREIVALDS;
    return VER_NONE;
}

/* Open the CSV and write the header if the file is new. */
static FILE *csv_open(const char *path)
{
    FILE *f = fopen(path, "a");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    if (!ftell(f))
        fprintf(f, "tag,engine,bcast,kernel,gridmap,plan,M,N,K,P,Pr,Pc,c,b,"
                   "threads,lookahead,rep,time_s,gflops,wire_gb,rank_gb,"
                   "t_comm,t_comp,err\n");
    return f;
}

int main(int argc, char **argv)
{
    int P, wrank, i;
    int M = 512, N = 512, K = 512;
    int Pr = 0, Pc = 0, c = 1, b = 128, lookahead = 0;
    int reps = 3, warmup = 1, trials = 2, quiet = 0;
    int do_plan = 0, do_planreport = 0, do_cal = 0, plan_P = 0;
    uint64_t seed = 1;
    scalar_t alpha = 1.0, beta = 0.0;
    engine_t eng = ENG_SUMMA;
    bcast_t  bc  = BC_BLOCKING;
    kernel_t krn = KRN_PACKED;
    gridmap_t map = MAP_LINEAR;
    verify_t ver = VER_NONE;
    const char *csv = NULL, *tag = "run";
    machine_t mm;
    pgrid_t g;
    gemm_opts_t o;
    dmat_t A, B, Cm;
    int Koff, Klen, mloc, nloc, ka, kb, r0, c0, a0, b0;
    double err = -1.0, best_t = 0.0;
    scalar_t *v0 = NULL;
    int threads = 1, mapfallback = 0;
    /* One buffered row per timed repetition. The CSV is written only after
     * verification has run, so the err column carries the measured residual
     * instead of the -1 placeholder. */
    struct { double t, gf, wire, rank, comm, comp; } *rows = NULL;
    FILE *cf = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &wrank);
    machine_defaults(&mm);

    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
#define ARGI(name, var) if (!strcmp(a, name) && i + 1 < argc) { var = atoi(argv[++i]); continue; }
#define ARGD(name, var) if (!strcmp(a, name) && i + 1 < argc) { var = atof(argv[++i]); continue; }
#define ARGS(name, var) if (!strcmp(a, name) && i + 1 < argc) { var = argv[++i]; continue; }
        ARGI("--M", M) ARGI("--N", N) ARGI("--K", K)
        ARGI("--Pr", Pr) ARGI("--Pc", Pc) ARGI("--c", c) ARGI("--b", b)
        ARGI("--lookahead", lookahead) ARGI("--reps", reps) ARGI("--plan-P", plan_P)
        ARGI("--warmup", warmup) ARGI("--trials", trials)
        ARGD("--alpha", alpha) ARGD("--beta", beta) ARGD("--mem-gb", mm.mem_bytes)
        ARGS("--csv", csv) ARGS("--tag", tag)
        if (!strcmp(a, "--seed") && i + 1 < argc) { seed = (uint64_t)atoll(argv[++i]); continue; }
        if (!strcmp(a, "--engine")  && i + 1 < argc) { eng = parse_engine(argv[++i]); continue; }
        if (!strcmp(a, "--bcast")   && i + 1 < argc) { bc  = parse_bcast(argv[++i]);  continue; }
        if (!strcmp(a, "--verify")  && i + 1 < argc) { ver = parse_verify(argv[++i]); continue; }
        if (!strcmp(a, "--kernel")  && i + 1 < argc) { krn = parse_kernel(argv[++i]); continue; }
        if (!strcmp(a, "--gridmap") && i + 1 < argc) { map = strcmp(argv[++i], "nodeaware") ? MAP_LINEAR : MAP_NODEAWARE; continue; }
        if (!strcmp(a, "--plan"))        { do_plan = 1; continue; }
        if (!strcmp(a, "--plan-report")) { do_planreport = 1; continue; }
        if (!strcmp(a, "--calibrate"))   { do_cal = 1; continue; }
        if (!strcmp(a, "--quiet"))       { quiet = 1; continue; }
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            if (!wrank) usage();
            MPI_Finalize();
            return 0;
        }
        if (!wrank) fprintf(stderr, "unknown option: %s\n", a);
#undef ARGI
#undef ARGD
#undef ARGS
    }
    if (mm.mem_bytes < 1e6) mm.mem_bytes *= 1e9;   /* --mem-gb given in GB */
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif

    if (do_cal) machine_calibrate(MPI_COMM_WORLD, &mm, krn, !wrank && !quiet);

    if (do_planreport) {
        plan_t best, top[5];
        if (!wrank) {
            int Pq = plan_P > 0 ? plan_P : P;
            plan_choose(&mm, M, N, K, Pq, 1, &best, top, 5);
            P = Pq;
            printf("# plan report for M=%d N=%d K=%d P=%d\n", M, N, K, P);
            printf("# %4s %4s %4s %5s  %12s %12s %12s\n",
                   "Pr", "Pc", "c", "b", "t_model[s]", "t_comm[s]", "t_comp[s]");
            for (i = 0; i < 5 && top[i].t_model > 0; i++)
                printf("  %4d %4d %4d %5d  %12.6f %12.6f %12.6f\n",
                       top[i].Pr, top[i].Pc, top[i].c, top[i].b,
                       top[i].t_model, top[i].t_comm, top[i].t_comp);
        }
        MPI_Finalize();
        return 0;
    }

    /* ---------------- naive 1D baseline: its own layout ------------- */
    if (eng == ENG_NAIVE1D) {
        int rep;
        o.engine = eng; o.bcast = bc; o.kernel = krn; o.b = b; o.c = 1;
        o.lookahead = 0; o.alpha = alpha; o.beta = beta;
        rows = calloc((size_t)(reps > 0 ? reps : 1), sizeof(*rows));
        for (rep = -warmup; rep < reps; rep++) {
            double t0, t1, tmax, e = -1.0;
            counters_reset();
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = MPI_Wtime();
            gemm_naive1d(MPI_COMM_WORLD, M, N, K, &o, seed,
                         (rep == reps - 1) ? &e : NULL,
                         (rep == reps - 1) ? ver : VER_NONE);
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = MPI_Wtime();
            t0 = t1 - t0;
            MPI_Reduce(&t0, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rep == reps - 1) err = e;
            if (rep >= 0 && !wrank) {
                double gf = 2.0 * M * N * K / (tmax * 1e9);
                rows[rep].t = tmax; rows[rep].gf = gf;
                if (!quiet)
                    printf("naive1d P=%d rep=%d t=%.6f s  %.2f Gflop/s\n",
                           P, rep, tmax, gf);
            }
        }
        if (!wrank && ver != VER_NONE)
            printf("verify(reference): relative error = %.3e  [%s]\n",
                   err, (err >= 0 && err < 1e-10) ? "OK" : (err < 0 ? "skipped" : "FAIL"));
        if (csv && !wrank && (cf = csv_open(csv)) != NULL) {
            for (rep = 0; rep < reps; rep++)
                fprintf(cf, "%s,naive1d,-,%s,-,0,%d,%d,%d,%d,%d,1,1,0,%d,0,%d,"
                            "%.6f,%.3f,0,0,0,0,%.3e\n",
                        tag, kernel_name(krn), M, N, K, P, P, threads, rep,
                        rows[rep].t, rows[rep].gf, err);
            fclose(cf);
        }
        free(rows);
        MPI_Finalize();
        return 0;
    }

    /* ---------------- grid selection --------------------------------- */
    if (do_plan) {
        plan_t best, top[5];
        int allow = (eng == ENG_SUMMA25D);
        if (plan_choose(&mm, M, N, K, P, allow, &best, top, 5) == 0) {
            Pr = best.Pr; Pc = best.Pc; c = best.c; b = best.b;
            if (!wrank && !quiet)
                printf("# planner chose Pr=%d Pc=%d c=%d b=%d (modeled %.6f s)\n",
                       Pr, Pc, c, b, best.t_model);
        }
    }
    if (eng == ENG_SUMMA)  c = 1;
    if (eng == ENG_CANNON) {
        int q = (int)(sqrt((double)P) + 0.5);
        c = 1;
        if (q * q != P) {
            if (!wrank) fprintf(stderr,
                "cannon needs a square process grid; P=%d is not a perfect square\n", P);
            MPI_Finalize();
            return 2;
        }
        Pr = Pc = q;
    }
    if (Pr <= 0 || Pc <= 0 || (long)Pr * Pc * c != P)
        pgrid_default_shape(P, c, M, N, &Pr, &Pc);
    if ((long)Pr * Pc * c != P) {
        if (!wrank) fprintf(stderr, "Pr*Pc*c (%d*%d*%d) != P (%d)\n", Pr, Pc, c, P);
        MPI_Finalize();
        return 2;
    }
    if (b <= 0) b = 128;
    if (bc != BC_IBCAST) lookahead = 0;

    mapfallback = pgrid_create(MPI_COMM_WORLD, Pr, Pc, c, map, &g);
    if (mapfallback < 0) { MPI_Finalize(); return 2; }
    if (mapfallback == 1 && !wrank && !quiet)
        printf("# node-aware map not applicable here, using linear\n");

    /* ---------------- distributed operands --------------------------- */
    Koff = blk_off(K, c, g.my_layer);
    Klen = blk_size(K, c, g.my_layer);
    mloc = blk_size(M, Pr, g.my_row); r0 = blk_off(M, Pr, g.my_row);
    nloc = blk_size(N, Pc, g.my_col); c0 = blk_off(N, Pc, g.my_col);
    ka   = blk_size(Klen, Pc, g.my_col); a0 = Koff + blk_off(Klen, Pc, g.my_col);
    kb   = blk_size(Klen, Pr, g.my_row); b0 = Koff + blk_off(Klen, Pr, g.my_row);

    dmat_alloc(&A,  M, K, mloc, ka,  r0, a0);
    dmat_alloc(&B,  K, N, kb,  nloc, b0, c0);
    dmat_alloc(&Cm, M, N, mloc, nloc, r0, c0);
    dmat_fill(&A, seed);
    dmat_fill(&B, seed + 1);

    o.engine = eng; o.bcast = bc; o.kernel = krn; o.b = b; o.c = c;
    o.lookahead = lookahead; o.alpha = alpha; o.beta = beta;

    /* C0 * r, needed by Freivalds when beta != 0 */
    if (ver == VER_FREIVALDS && beta != 0.0) {
        int tr;
        if (g.my_layer == 0) dmat_fill(&Cm, seed + 2); else dmat_zero(&Cm);
        v0 = (scalar_t *)calloc((size_t)trials * (mloc ? mloc : 1), sizeof(scalar_t));
        for (tr = 0; tr < trials; tr++) {
            scalar_t *t = freivalds_cr(&g, &Cm, N, seed + 100, tr);
            if (mloc > 0) memcpy(v0 + (size_t)tr * mloc, t, (size_t)mloc * sizeof(scalar_t));
            free(t);
        }
    }

    /* ---------------- timed repetitions ------------------------------ */
    rows = calloc((size_t)(reps > 0 ? reps : 1), sizeof(*rows));
    if (!rows) { MPI_Finalize(); return 2; }
    for (i = -warmup; i < reps; i++) {
        double t0, t1, tmax, cw[2], cwmax[2], bs[2], bsum[2];
        int rc;

        /* only layer 0 carries beta*C, so the depth reduce yields
         * alpha*A*B + beta*C exactly once */
        if (beta != 0.0 && g.my_layer == 0) {
            dmat_fill(&Cm, seed + 2);
            dmat_scale(&Cm, beta);
        } else {
            dmat_zero(&Cm);
        }

        counters_reset();
        MPI_Barrier(g.grid);
        t0 = MPI_Wtime();
        switch (eng) {
        case ENG_CANNON:   rc = gemm_cannon(&g, &A, &B, &Cm, &o, K);       break;
        case ENG_SUMMA25D: rc = gemm_summa25d(&g, &A, &B, &Cm, &o, K);     break;
        default:           rc = gemm_summa(&g, &A, &B, &Cm, &o, 0, K);     break;
        }
        MPI_Barrier(g.grid);
        t1 = MPI_Wtime();
        if (rc) {
            if (!wrank) fprintf(stderr, "engine %s refused this configuration (rc=%d)\n",
                                engine_name(eng), rc);
            MPI_Finalize();
            return 3;
        }
        t0 = t1 - t0;
        MPI_Reduce(&t0, &tmax, 1, MPI_DOUBLE, MPI_MAX, 0, g.grid);
        cw[0] = g_cnt.t_comm; cw[1] = g_cnt.t_comp;
        MPI_Reduce(cw, cwmax, 2, MPI_DOUBLE, MPI_MAX, 0, g.grid);
        bs[0] = g_cnt.bytes_offnode; bs[1] = g_cnt.bytes_rank;
        MPI_Reduce(bs, bsum, 2, MPI_DOUBLE, MPI_SUM, 0, g.grid);

        if (i >= 0 && !wrank) {
            double gf = 2.0 * (double)M * N * K / (tmax * 1e9);
            best_t = (i == 0 || tmax < best_t) ? tmax : best_t;
            if (!quiet)
                printf("%s/%s P=%d %dx%dx%d b=%d rep=%d  t=%.6f s  %.2f Gflop/s"
                       "  comm=%.4f comp=%.4f  offnode=%.3f GB\n",
                       engine_name(eng), bcast_name(bc), P, Pr, Pc, c, b, i,
                       tmax, gf, cwmax[0], cwmax[1], bsum[0] / 1e9);
            rows[i].t = tmax; rows[i].gf = gf;
            rows[i].wire = bsum[0] / 1e9; rows[i].rank = bsum[1] / 1e9;
            rows[i].comm = cwmax[0];      rows[i].comp = cwmax[1];
        }
    }

    /* ---------------- verification ----------------------------------- */
    if (ver == VER_REF)
        err = verify_reference(&g, &Cm, M, N, K, alpha, beta,
                               seed, seed + 1, seed + 2, 4000000L);
    else if (ver == VER_FREIVALDS)
        err = verify_freivalds(&g, &A, &B, &Cm, M, N, K, Koff, Klen,
                               alpha, beta, v0, seed + 100, trials);

    if (!wrank && ver != VER_NONE) {
        const char *what = (ver == VER_REF) ? "reference" : "freivalds";
        if (err < 0)
            printf("verify(%s): skipped (problem too large for the exact path)\n", what);
        else
            printf("verify(%s): relative error = %.3e  [%s]\n", what, err,
                   err < 1e-10 ? "OK" : "FAIL");
    }

    if (csv && !wrank && (cf = csv_open(csv)) != NULL) {
        for (i = 0; i < reps; i++)
            fprintf(cf, "%s,%s,%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
                        "%.6f,%.3f,%.6f,%.6f,%.6f,%.6f,%.3e\n",
                    tag, engine_name(eng), bcast_name(bc), kernel_name(krn),
                    gridmap_name(g.map), do_plan, M, N, K, P, Pr, Pc, c, b,
                    threads, lookahead, i, rows[i].t, rows[i].gf,
                    rows[i].wire, rows[i].rank, rows[i].comm, rows[i].comp, err);
        fclose(cf);
    }

    if (!wrank && !quiet)
        printf("# nodes=%d ranks/node=%d threads=%d best=%.6f s\n",
               g.n_nodes, g.node_size, threads, best_t);

    free(rows);
    free(v0);
    dmat_free(&A); dmat_free(&B); dmat_free(&Cm);
    pgrid_free(&g);
    MPI_Finalize();
    return (ver != VER_NONE && err >= 0 && err > 1e-10) ? 1 : 0;
}
