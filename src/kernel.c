/* kernel.c — local rank-k update, C += alpha * A * B.
 *
 * Row-major with i-k-j ordering: B[k][j] and C[i][j] are unit-stride in
 * the innermost loop and A[i][k] is a scalar broadcast. That is the same
 * lesson as the row-wise vs column-wise cache example from the course,
 * one level up.
 *
 * The blocked variant tiles (MC, KC, NC) so that the B panel stays in L2
 * and the A block in L1, and threads over i-tiles only: each thread owns
 * whole rows of C, so there is no reduction and no false sharing.
 */
#include "gemm2d.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef MC
#define MC 64
#endif
#ifndef KC
#define KC 256
#endif
#ifndef NC
#define NC 512
#endif

static void gemm_naive(int m, int n, int k, scalar_t alpha,
                       const scalar_t *A, int lda,
                       const scalar_t *B, int ldb,
                       scalar_t *C, int ldc)
{
    int i, kk, j;
#ifdef _OPENMP
#pragma omp parallel for private(kk, j) schedule(static)
#endif
    for (i = 0; i < m; i++)
        for (kk = 0; kk < k; kk++) {
            scalar_t a = alpha * A[(size_t)i * lda + kk];
            if (a == (scalar_t)0.0) continue;
            for (j = 0; j < n; j++)
                C[(size_t)i * ldc + j] += a * B[(size_t)kk * ldb + j];
        }
}

static void gemm_blocked(int m, int n, int k, scalar_t alpha,
                         const scalar_t *A, int lda,
                         const scalar_t *B, int ldb,
                         scalar_t *C, int ldc)
{
    int ii;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (ii = 0; ii < m; ii += MC) {
        int imax = (ii + MC < m) ? ii + MC : m;
        int kk, jj, i, t, j;
        for (kk = 0; kk < k; kk += KC) {
            int kmax = (kk + KC < k) ? kk + KC : k;
            for (jj = 0; jj < n; jj += NC) {
                int jmax = (jj + NC < n) ? jj + NC : n;
                for (i = ii; i < imax; i++) {
                    scalar_t *crow = C + (size_t)i * ldc;
                    const scalar_t *arow = A + (size_t)i * lda;
                    for (t = kk; t < kmax; t++) {
                        scalar_t a = alpha * arow[t];
                        const scalar_t *brow = B + (size_t)t * ldb;
                        if (a == (scalar_t)0.0) continue;
                        for (j = jj; j < jmax; j++)
                            crow[j] += a * brow[j];
                    }
                }
            }
        }
    }
}

void kernel_gemm_acc(int m, int n, int k, scalar_t alpha,
                     const scalar_t *A, int lda,
                     const scalar_t *B, int ldb,
                     scalar_t *C, int ldc, kernel_t which)
{
    double t0;
    if (m <= 0 || n <= 0 || k <= 0) return;
    t0 = MPI_Wtime();
    if (which == KRN_NAIVE) gemm_naive(m, n, k, alpha, A, lda, B, ldb, C, ldc);
    else                    gemm_blocked(m, n, k, alpha, A, lda, B, ldb, C, ldc);
    g_cnt.t_comp += MPI_Wtime() - t0;
}
