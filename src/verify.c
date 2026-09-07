#include "gemm2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

/* --------------------------------------------------------------------
 * Exact check against a sequential reference. O(M*N*K) on one rank, so
 * only for the small correctness tests (prime P, non-divisible sizes).
 * -------------------------------------------------------------------- */
double verify_reference(const pgrid_t *g, const dmat_t *C, int M, int N, int K,
                        scalar_t alpha, scalar_t beta, uint64_t sA, uint64_t sB,
                        uint64_t sC, long max_elems)
{
    double err = -1.0;

    if ((long)M * N <= max_elems && g->layer0 != MPI_COMM_NULL) {
        int lr, ls, p;
        MPI_Comm_rank(g->layer0, &lr);
        MPI_Comm_size(g->layer0, &ls);

        if (lr == 0) {
            scalar_t *full = (scalar_t *)calloc((size_t)M * N, sizeof(scalar_t));
            double num = 0.0, den = 0.0;
            int i, j, t, hdr[4];

            for (i = 0; i < C->m; i++)
                for (j = 0; j < C->n; j++)
                    full[(size_t)(C->row0 + i) * N + C->col0 + j] =
                        C->data[(size_t)i * C->ld + j];

            for (p = 1; p < ls; p++) {
                MPI_Recv(hdr, 4, MPI_INT, p, 10, g->layer0, MPI_STATUS_IGNORE);
                if (hdr[2] > 0 && hdr[3] > 0) {
                    scalar_t *tmp = (scalar_t *)malloc((size_t)hdr[2] * hdr[3] * sizeof(scalar_t));
                    MPI_Recv(tmp, hdr[2] * hdr[3], MPI_SCALAR, p, 11, g->layer0,
                             MPI_STATUS_IGNORE);
                    for (i = 0; i < hdr[2]; i++)
                        for (j = 0; j < hdr[3]; j++)
                            full[(size_t)(hdr[0] + i) * N + hdr[1] + j] =
                                tmp[(size_t)i * hdr[3] + j];
                    free(tmp);
                }
            }
            for (i = 0; i < M; i++)
                for (j = 0; j < N; j++) {
                    double acc = 0.0, ref, got;
                    for (t = 0; t < K; t++)
                        acc += gen_elem(sA, i, t) * gen_elem(sB, t, j);
                    ref = alpha * acc + beta * gen_elem(sC, i, j);
                    got = full[(size_t)i * N + j];
                    num += (got - ref) * (got - ref);
                    den += ref * ref;
                }
            err = (den > 0.0) ? sqrt(num / den) : sqrt(num);
            free(full);
        } else {
            int hdr[4];
            hdr[0] = C->row0; hdr[1] = C->col0; hdr[2] = C->m; hdr[3] = C->n;
            MPI_Send(hdr, 4, MPI_INT, 0, 10, g->layer0);
            if (C->m > 0 && C->n > 0) {
                scalar_t *tmp = (scalar_t *)malloc((size_t)C->m * C->n * sizeof(scalar_t));
                int i, j;
                for (i = 0; i < C->m; i++)
                    for (j = 0; j < C->n; j++)
                        tmp[(size_t)i * C->n + j] = C->data[(size_t)i * C->ld + j];
                MPI_Send(tmp, C->m * C->n, MPI_SCALAR, 0, 11, g->layer0);
                free(tmp);
            }
        }
    }
    MPI_Bcast(&err, 1, MPI_DOUBLE, 0, g->grid);
    return err;
}

/* --------------------------------------------------------------------
 * Distributed Freivalds. Cost is O(n^2/P) instead of O(n^3), so it can
 * be run on every large job. Vectors are O(n) words, so they are simply
 * replicated instead of redistributed.
 * -------------------------------------------------------------------- */
scalar_t *freivalds_cr(const pgrid_t *g, const dmat_t *C, int N,
                       uint64_t rseed, int trial)
{
    scalar_t *u = (scalar_t *)calloc((size_t)(C->m ? C->m : 1), sizeof(scalar_t));
    int i, j;
    for (i = 0; i < C->m; i++) {
        double acc = 0.0;
        for (j = 0; j < C->n; j++)
            acc += C->data[(size_t)i * C->ld + j] *
                   gen_elem(rseed + (uint64_t)trial, 0, C->col0 + j);
        u[i] = acc;
    }
    (void)N;
    if (C->m > 0) MPI_Allreduce(MPI_IN_PLACE, u, C->m, MPI_SCALAR, MPI_SUM, g->row);
    return u;
}

double verify_freivalds(const pgrid_t *g, const dmat_t *A, const dmat_t *B,
                        const dmat_t *C, int M, int N, int K, int Koff, int Klen,
                        scalar_t alpha, scalar_t beta, const scalar_t *v0,
                        uint64_t rseed, int trials)
{
    double worst = 0.0;
    int aco = blk_off(Klen, g->Pc, g->my_col);
    int tr, i, j;
    int *cnts = (int *)malloc((size_t)g->Pr * sizeof(int));
    int *disp = (int *)malloc((size_t)g->Pr * sizeof(int));
    scalar_t *z = (scalar_t *)calloc((size_t)(Klen ? Klen : 1), sizeof(scalar_t));
    scalar_t *zp, *w, *u;
    (void)M; (void)Koff;

    for (i = 0; i < g->Pr; i++) {
        cnts[i] = blk_size(Klen, g->Pr, i);
        disp[i] = blk_off(Klen, g->Pr, i);
    }
    zp = (scalar_t *)calloc((size_t)(B->m ? B->m : 1), sizeof(scalar_t));
    w  = (scalar_t *)calloc((size_t)(C->m ? C->m : 1), sizeof(scalar_t));

    for (tr = 0; tr < trials; tr++) {
        double num = 0.0, den = 0.0;

        /* z = B * r, over this layer's slab */
        for (i = 0; i < B->m; i++) {
            double acc = 0.0;
            for (j = 0; j < B->n; j++)
                acc += B->data[(size_t)i * B->ld + j] *
                       gen_elem(rseed + (uint64_t)tr, 0, B->col0 + j);
            zp[i] = acc;
        }
        if (B->m > 0)
            MPI_Allreduce(MPI_IN_PLACE, zp, B->m, MPI_SCALAR, MPI_SUM, g->row);
        MPI_Allgatherv(zp, B->m, MPI_SCALAR, z, cnts, disp, MPI_SCALAR, g->col);

        /* w = A * z, summed over column slices and then over layers */
        for (i = 0; i < C->m; i++) {
            double acc = 0.0;
            for (j = 0; j < A->n; j++)
                acc += A->data[(size_t)i * A->ld + j] * z[aco + j];
            w[i] = acc;
        }
        if (C->m > 0) {
            MPI_Allreduce(MPI_IN_PLACE, w, C->m, MPI_SCALAR, MPI_SUM, g->row);
            MPI_Allreduce(MPI_IN_PLACE, w, C->m, MPI_SCALAR, MPI_SUM, g->depth);
        }

        /* u = C * r, and compare against alpha*w + beta*(C0*r) */
        u = freivalds_cr(g, C, N, rseed, tr);
        if (g->my_layer == 0) {
            for (i = 0; i < C->m; i++) {
                double want = alpha * w[i] +
                              (v0 ? beta * v0[(size_t)tr * C->m + i] : 0.0);
                double d = want - u[i];
                num += d * d;
                den += want * want;
            }
            MPI_Allreduce(MPI_IN_PLACE, &num, 1, MPI_DOUBLE, MPI_SUM, g->col);
            MPI_Allreduce(MPI_IN_PLACE, &den, 1, MPI_DOUBLE, MPI_SUM, g->col);
            {
                double e = (den > 0.0) ? sqrt(num / den) : sqrt(num);
                if (e > worst) worst = e;
            }
        }
        free(u);
    }
    MPI_Bcast(&worst, 1, MPI_DOUBLE, 0, g->grid);

    free(cnts); free(disp); free(z); free(zp); free(w);
    return worst;
}
