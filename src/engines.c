#include "gemm2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

typedef struct { int k0, kb, ownc, ownr; } sstep_t;

/* Panel boundaries never straddle an owner boundary. A's k-dimension is
 * split across Pc and B's across Pr, and those two partitions differ, so
 * each step is clipped to the intersection of both owners' ranges. That
 * is what makes arbitrary K, Pr, Pc work with no divisibility assumption. */
static int summa_schedule(int Klen, int Pr, int Pc, int b, sstep_t **out)
{
    int cap = 16, ns = 0, k0 = 0;
    sstep_t *s = (sstep_t *)malloc((size_t)cap * sizeof(sstep_t));
    if (!s) return -1;
    while (k0 < Klen) {
        int ownc = blk_owner(Klen, Pc, k0);
        int ownr = blk_owner(Klen, Pr, k0);
        int endc = blk_off(Klen, Pc, ownc) + blk_size(Klen, Pc, ownc);
        int endr = blk_off(Klen, Pr, ownr) + blk_size(Klen, Pr, ownr);
        int end  = k0 + b;
        if (end > endc) end = endc;
        if (end > endr) end = endr;
        if (end > Klen) end = Klen;
        if (end <= k0) { k0++; continue; }          /* zero-width owner */
        if (ns == cap) {
            cap *= 2;
            s = (sstep_t *)realloc(s, (size_t)cap * sizeof(sstep_t));
            if (!s) return -1;
        }
        s[ns].k0 = k0; s[ns].kb = end - k0;
        s[ns].ownc = ownc; s[ns].ownr = ownr;
        ns++;
        k0 = end;
    }
    *out = s;
    return ns;
}

static void pack_A(const dmat_t *A, int loc_k0, int kb, scalar_t *dst)
{
    double t0 = MPI_Wtime();
    int i;
    for (i = 0; i < A->m; i++)
        memcpy(dst + (size_t)i * kb,
               A->data + (size_t)i * A->ld + loc_k0,
               (size_t)kb * sizeof(scalar_t));
    g_cnt.t_pack += MPI_Wtime() - t0;
}

static void pack_B(const dmat_t *B, int loc_k0, int kb, scalar_t *dst)
{
    double t0 = MPI_Wtime();
    /* B is stored row-major with ld == n, so the rows of a panel are
     * already contiguous: one memcpy. */
    memcpy(dst, B->data + (size_t)loc_k0 * B->ld,
           (size_t)kb * (size_t)B->n * sizeof(scalar_t));
    g_cnt.t_pack += MPI_Wtime() - t0;
}

/* --------------------------------------------------------------------
 * SUMMA over the k-range [Koff, Koff+Klen) held by this layer.
 * For plain 2D, Koff = 0 and Klen = K.
 * -------------------------------------------------------------------- */
int gemm_summa(const pgrid_t *g, const dmat_t *A, const dmat_t *B, dmat_t *C,
               const gemm_opts_t *o, int Koff, int Klen)
{
    sstep_t *st = NULL;
    int ns, s, nbuf, rc = 0;
    int aco = blk_off(Klen, g->Pc, g->my_col);   /* slab-local col start of my A */
    int bro = blk_off(Klen, g->Pr, g->my_row);   /* slab-local row start of my B */
    chan_t chA[2], chB[2];
    int i;
    (void)Koff;

    ns = summa_schedule(Klen, g->Pr, g->Pc, o->b, &st);
    if (ns < 0) return -1;

    nbuf = (o->bcast == BC_IBCAST && o->lookahead) ? 2 : 1;
    for (i = 0; i < nbuf; i++) {
        if (chan_create(g->row, o->bcast, (size_t)(C->m > 0 ? C->m : 1) * o->b, &chA[i]) ||
            chan_create(g->col, o->bcast, (size_t)o->b * (C->n > 0 ? C->n : 1), &chB[i])) {
            free(st);
            return -1;
        }
    }

#define POST_STEP(idx, buf)                                                  \
    do {                                                                     \
        int _kb = st[idx].kb;                                                \
        chan_begin(&chA[buf]);                                               \
        chan_begin(&chB[buf]);                                               \
        if (g->my_col == st[idx].ownc && C->m > 0)                           \
            pack_A(A, st[idx].k0 - aco, _kb, chan_wbuf(&chA[buf]));          \
        chan_post(&chA[buf], C->m * _kb, st[idx].ownc);                      \
        if (g->my_row == st[idx].ownr && C->n > 0)                           \
            pack_B(B, st[idx].k0 - bro, _kb, chan_wbuf(&chB[buf]));          \
        chan_post(&chB[buf], _kb * C->n, st[idx].ownr);                      \
    } while (0)

    if (nbuf == 2) {
        if (ns > 0) POST_STEP(0, 0);
        for (s = 0; s < ns; s++) {
            int cur = s & 1, nxt = (s + 1) & 1;
            if (s + 1 < ns) POST_STEP(s + 1, nxt);
            chan_wait(&chA[cur]);
            chan_wait(&chB[cur]);
            kernel_gemm_acc(C->m, C->n, st[s].kb, o->alpha,
                            chan_rbuf(&chA[cur]), st[s].kb,
                            chan_rbuf(&chB[cur]), C->n,
                            C->data, C->ld, o->kernel);
        }
    } else {
        for (s = 0; s < ns; s++) {
            POST_STEP(s, 0);
            chan_wait(&chA[0]);
            chan_wait(&chB[0]);
            kernel_gemm_acc(C->m, C->n, st[s].kb, o->alpha,
                            chan_rbuf(&chA[0]), st[s].kb,
                            chan_rbuf(&chB[0]), C->n,
                            C->data, C->ld, o->kernel);
        }
    }
#undef POST_STEP

    for (i = 0; i < nbuf; i++) { chan_free(&chA[i]); chan_free(&chB[i]); }
    free(st);
    return rc;
}

/* --------------------------------------------------------------------
 * 2.5D: each layer owns a k-slab, runs a full SUMMA on it, then the
 * partial C blocks are summed across the depth communicator.
 * Extra memory is c copies of C; extra communication is one reduce.
 * -------------------------------------------------------------------- */
int gemm_summa25d(const pgrid_t *g, const dmat_t *A, const dmat_t *B, dmat_t *C,
                  const gemm_opts_t *o, int K)
{
    int Koff = blk_off(K, g->C, g->my_layer);
    int Klen = blk_size(K, g->C, g->my_layer);
    int rc, cnt = C->m * C->n;
    double t0;

    rc = gemm_summa(g, A, B, C, o, Koff, Klen);
    if (rc) return rc;

    if (g->C > 1) {
        t0 = MPI_Wtime();
        if (g->my_layer == 0)
            MPI_Reduce(MPI_IN_PLACE, C->data, cnt, MPI_SCALAR, MPI_SUM, 0, g->depth);
        else
            MPI_Reduce(C->data, NULL, cnt, MPI_SCALAR, MPI_SUM, 0, g->depth);
        g_cnt.t_comm += MPI_Wtime() - t0;
        if (g->my_layer != 0) g_cnt.bytes_rank += (double)cnt * sizeof(scalar_t);
        g_cnt.bytes_offnode += (double)cnt * sizeof(scalar_t);   /* modeled */
    }
    return 0;
}

/* --------------------------------------------------------------------
 * Cannon. Deliberately kept restrictive: square grid, K/M/N divisible by
 * q. The failure of this engine on generic inputs is a result, not a
 * defect -- it is the argument for choosing SUMMA as the base.
 * -------------------------------------------------------------------- */
int gemm_cannon(const pgrid_t *g, const dmat_t *A, const dmat_t *B, dmat_t *C,
                const gemm_opts_t *o, int K)
{
    int q = g->Pr, s;
    int m = C->m, n = C->n, kloc;
    int *rnode = NULL, *cnode = NULL;
    scalar_t *Ab, *Bb;
    double t0;

    if (g->Pr != g->Pc || g->C != 1) return -2;
    if (C->gm % q || C->gn % q || K % q) return -2;

    kloc = K / q;
    m = C->gm / q; n = C->gn / q;
    Ab = A->data; Bb = B->data;

    rnode = (int *)malloc((size_t)q * sizeof(int));
    cnode = (int *)malloc((size_t)q * sizeof(int));
    MPI_Allgather(&g->node_id, 1, MPI_INT, rnode, 1, MPI_INT, g->row);
    MPI_Allgather(&g->node_id, 1, MPI_INT, cnode, 1, MPI_INT, g->col);

#define SHIFT(buf, cnt, comm, dst, src, tbl)                                  \
    do {                                                                      \
        t0 = MPI_Wtime();                                                     \
        MPI_Sendrecv_replace(buf, cnt, MPI_SCALAR, dst, 77, src, 77,          \
                             comm, MPI_STATUS_IGNORE);                        \
        g_cnt.t_comm += MPI_Wtime() - t0;                                     \
        g_cnt.n_bcast++;                                                      \
        g_cnt.bytes_rank += (double)(cnt) * sizeof(scalar_t);                 \
        if ((tbl)[src] != g->node_id)                                         \
            g_cnt.bytes_offnode += (double)(cnt) * sizeof(scalar_t);          \
    } while (0)

    /* initial skew */
    if (g->my_row) SHIFT(Ab, m * kloc, g->row, (g->my_col - g->my_row + q) % q,
                         (g->my_col + g->my_row) % q, rnode);
    if (g->my_col) SHIFT(Bb, kloc * n, g->col, (g->my_row - g->my_col + q) % q,
                         (g->my_row + g->my_col) % q, cnode);

    for (s = 0; s < q; s++) {
        kernel_gemm_acc(m, n, kloc, o->alpha, Ab, kloc, Bb, n,
                        C->data, C->ld, o->kernel);
        if (s + 1 < q) {
            SHIFT(Ab, m * kloc, g->row, (g->my_col - 1 + q) % q,
                  (g->my_col + 1) % q, rnode);
            SHIFT(Bb, kloc * n, g->col, (g->my_row - 1 + q) % q,
                  (g->my_row + 1) % q, cnode);
        }
    }
#undef SHIFT
    free(rnode); free(cnode);
    return 0;
}

/* --------------------------------------------------------------------
 * Naive 1D row decomposition: A and C split by rows, B gathered whole on
 * every rank. Per-rank volume is K*N words regardless of P, so it cannot
 * scale. Kept as the honest "why bother with 2D" baseline.
 * -------------------------------------------------------------------- */
int gemm_naive1d(MPI_Comm comm, int M, int N, int K, const gemm_opts_t *o,
                 uint64_t seed, double *err_out, verify_t ver)
{
    int P, r, i, j, t;
    int m, k, r0, k0;
    int *cnts = NULL, *disp = NULL;
    scalar_t *Aloc, *Bloc, *Bfull, *Cloc;
    double t0, err = -1.0;

    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &r);
    m  = blk_size(M, P, r); r0 = blk_off(M, P, r);
    k  = blk_size(K, P, r); k0 = blk_off(K, P, r);

    Aloc  = (scalar_t *)calloc((size_t)(m ? m : 1) * K, sizeof(scalar_t));
    Bloc  = (scalar_t *)calloc((size_t)(k ? k : 1) * N, sizeof(scalar_t));
    Bfull = (scalar_t *)calloc((size_t)K * N, sizeof(scalar_t));
    Cloc  = (scalar_t *)calloc((size_t)(m ? m : 1) * N, sizeof(scalar_t));
    if (!Aloc || !Bloc || !Bfull || !Cloc) return -1;

    for (i = 0; i < m; i++)
        for (t = 0; t < K; t++) Aloc[(size_t)i * K + t] = gen_elem(seed, r0 + i, t);
    for (i = 0; i < k; i++)
        for (j = 0; j < N; j++) Bloc[(size_t)i * N + j] = gen_elem(seed + 1, k0 + i, j);

    cnts = (int *)malloc((size_t)P * sizeof(int));
    disp = (int *)malloc((size_t)P * sizeof(int));
    for (i = 0; i < P; i++) {
        cnts[i] = blk_size(K, P, i) * N;
        disp[i] = blk_off(K, P, i) * N;
    }

    t0 = MPI_Wtime();
    MPI_Allgatherv(Bloc, k * N, MPI_SCALAR, Bfull, cnts, disp, MPI_SCALAR, comm);
    g_cnt.t_comm += MPI_Wtime() - t0;
    g_cnt.n_bcast++;
    g_cnt.bytes_rank    += (double)((size_t)K * N - (size_t)k * N) * sizeof(scalar_t);
    g_cnt.bytes_offnode += (double)((size_t)K * N - (size_t)k * N) * sizeof(scalar_t);

    kernel_gemm_acc(m, N, K, o->alpha, Aloc, K, Bfull, N, Cloc, N, o->kernel);

    if (ver == VER_REF && (long)M * N <= 4000000L) {
        double loc = 0.0, ref = 0.0, num = 0.0, den = 0.0;
        for (i = 0; i < m; i++)
            for (j = 0; j < N; j++) {
                double acc = 0.0;
                for (t = 0; t < K; t++)
                    acc += gen_elem(seed, r0 + i, t) * gen_elem(seed + 1, t, j);
                acc *= o->alpha;
                ref = acc;
                loc = Cloc[(size_t)i * N + j];
                num += (loc - ref) * (loc - ref);
                den += ref * ref;
            }
        MPI_Allreduce(MPI_IN_PLACE, &num, 1, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, &den, 1, MPI_DOUBLE, MPI_SUM, comm);
        err = (den > 0.0) ? sqrt(num / den) : sqrt(num);
    }
    if (err_out) *err_out = err;

    free(Aloc); free(Bloc); free(Bfull); free(Cloc); free(cnts); free(disp);
    return 0;
}
