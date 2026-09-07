/* planner.c — pick (Pr, Pc, c, b) from a calibrated cost model instead of
 * hardcoding a square grid. This is the component that makes the code
 * "generic": for square matrices it returns the usual near-square grid,
 * for short-fat matrices it discovers k-parallelism (c > 1).
 */
#include "gemm2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

void machine_defaults(machine_t *mm)
{
    mm->alpha_lat  = 2.0e-6;    /* 2 us per message      */
    mm->beta_bw    = 1.0e-10;   /* 10 GB/s               */
    mm->gamma_flop = 2.0e-10;   /* 5 Gflop/s per rank    */
    mm->mem_bytes  = 2.0e9;     /* 2 GB usable per rank  */
}

/* Ping-pong between the first and last rank (most likely on different
 * nodes) gives alpha and beta; a local kernel run gives gamma. */
void machine_calibrate(MPI_Comm comm, machine_t *mm, int verbose)
{
    int P, r, i, rep;
    const int nsz = 2;
    size_t sizes[2] = { 8, 1 << 20 };
    double tt[2] = { 0, 0 };
    scalar_t *buf;
    int peer;

    MPI_Comm_size(comm, &P);
    MPI_Comm_rank(comm, &r);
    peer = P - 1;
    buf = (scalar_t *)calloc(sizes[1] / sizeof(scalar_t) + 1, sizeof(scalar_t));

    if (P > 1) {
        for (i = 0; i < nsz; i++) {
            int cnt = (int)(sizes[i] / sizeof(scalar_t));
            double t0;
            MPI_Barrier(comm);
            t0 = MPI_Wtime();
            for (rep = 0; rep < 20; rep++) {
                if (r == 0) {
                    MPI_Send(buf, cnt, MPI_SCALAR, peer, 1, comm);
                    MPI_Recv(buf, cnt, MPI_SCALAR, peer, 2, comm, MPI_STATUS_IGNORE);
                } else if (r == peer) {
                    MPI_Recv(buf, cnt, MPI_SCALAR, 0, 1, comm, MPI_STATUS_IGNORE);
                    MPI_Send(buf, cnt, MPI_SCALAR, 0, 2, comm);
                }
            }
            if (r == 0) tt[i] = (MPI_Wtime() - t0) / (2.0 * 20.0);
            MPI_Bcast(&tt[i], 1, MPI_DOUBLE, 0, comm);
        }
        if (tt[1] > tt[0] && tt[0] > 0) {
            mm->alpha_lat = tt[0];
            mm->beta_bw   = (tt[1] - tt[0]) / (double)(sizes[1] - sizes[0]);
        }
    }

    /* gamma: time a local rank-k update of a cache-resident block */
    {
        int m = 128, n = 128, k = 128;
        scalar_t *A = (scalar_t *)calloc((size_t)m * k, sizeof(scalar_t));
        scalar_t *B = (scalar_t *)calloc((size_t)k * n, sizeof(scalar_t));
        scalar_t *C = (scalar_t *)calloc((size_t)m * n, sizeof(scalar_t));
        double t0, el, save_comp = g_cnt.t_comp;
        for (i = 0; i < m * k; i++) A[i] = 1.0 / (i + 1);
        for (i = 0; i < k * n; i++) B[i] = 1.0 / (i + 2);
        t0 = MPI_Wtime();
        for (rep = 0; rep < 20; rep++)
            kernel_gemm_acc(m, n, k, 1.0, A, k, B, n, C, n, KRN_BLOCKED);
        el = (MPI_Wtime() - t0) / 20.0;
        g_cnt.t_comp = save_comp;
        mm->gamma_flop = el / (2.0 * m * n * k);
        MPI_Allreduce(MPI_IN_PLACE, &mm->gamma_flop, 1, MPI_DOUBLE, MPI_MAX, comm);
        free(A); free(B); free(C);
    }

    if (verbose && r == 0)
        fprintf(stderr,
                "# calibrated: alpha=%.3e s  beta=%.3e s/B (%.2f GB/s)  gamma=%.3e s/flop (%.2f Gflop/s)\n",
                mm->alpha_lat, mm->beta_bw, 1.0 / (mm->beta_bw * 1e9),
                mm->gamma_flop, 1.0 / (mm->gamma_flop * 1e9));
    free(buf);
}

static double lg2(int x) { return (x > 1) ? log2((double)x) : 0.0; }

double plan_cost(const machine_t *mm, int M, int N, int K, int P,
                 int Pr, int Pc, int c, int b, double *t_comm, double *t_comp)
{
    double w = (double)sizeof(scalar_t);
    double Kl = ceil((double)K / c);
    double steps = ceil(Kl / (double)b);
    double lat = mm->alpha_lat * (steps * (lg2(Pr) + lg2(Pc)) + lg2(c));
    double bw  = mm->beta_bw * w *
                 (Kl * ((double)M / Pr + (double)N / Pc) +
                  (double)c * (double)M * (double)N / P);
    double comp = mm->gamma_flop * 2.0 * (double)M * (double)N * (double)K / P;
    if (t_comm) *t_comm = lat + bw;
    if (t_comp) *t_comp = comp;
    return lat + bw + comp;
}

static double plan_mem(int M, int N, int K, int P, int Pr, int Pc, int c, int b)
{
    double w = (double)sizeof(scalar_t);
    return w * ((double)M * K / P + (double)K * N / P +
                (double)c * M * N / P +
                2.0 * b * ((double)M / Pr + (double)N / Pc));
}

static void insert_top(plan_t *top, int ntop, const plan_t *p)
{
    int i, j;
    for (i = 0; i < ntop; i++) {
        if (top[i].t_model < 0 || p->t_model < top[i].t_model) {
            for (j = ntop - 1; j > i; j--) top[j] = top[j - 1];
            top[i] = *p;
            return;
        }
    }
}

int plan_choose(const machine_t *mm, int M, int N, int K, int P,
                int allow_25d, plan_t *best, plan_t *top, int ntop)
{
    int c, Pr, b, i, found = 0;
    static const int bs[] = { 32, 64, 128, 256, 512, 1024 };
    const int nb = (int)(sizeof(bs) / sizeof(bs[0]));

    best->t_model = -1.0;
    for (i = 0; i < ntop; i++) top[i].t_model = -1.0;

    for (c = 1; c <= P; c++) {
        if (P % c) continue;
        if (!allow_25d && c > 1) break;
        for (Pr = 1; Pr <= P / c; Pr++) {
            int Pl = P / c, Pc;
            if (Pl % Pr) continue;
            Pc = Pl / Pr;
            for (b = 0; b < nb; b++) {
                plan_t p;
                if (bs[b] > K && bs[b] > 32) continue;
                if (plan_mem(M, N, K, P, Pr, Pc, c, bs[b]) > mm->mem_bytes) continue;
                p.Pr = Pr; p.Pc = Pc; p.c = c; p.b = bs[b];
                p.t_model = plan_cost(mm, M, N, K, P, Pr, Pc, c, bs[b],
                                      &p.t_comm, &p.t_comp);
                if (best->t_model < 0 || p.t_model < best->t_model) *best = p;
                insert_top(top, ntop, &p);
                found = 1;
            }
        }
    }
    return found ? 0 : -1;
}
