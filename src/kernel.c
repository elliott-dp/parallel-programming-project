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
#include <stdlib.h>
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

/* ---- tunables (overridable from the command line for parameter sweeps) ---- */
#ifndef KMR
#define KMR 8              /* register tile rows    */
#endif
#ifndef KNR
#define KNR 8              /* register tile columns */
#endif
#ifndef KMC
#define KMC 256            /* A block rows  (L2)    */
#endif
#ifndef KKC
#define KKC 256            /* depth block           */
#endif
#ifndef KNC
#define KNC 1024           /* B block cols  (L3)    */
#endif

/* Fused multiply-add.  GCC/Clang in strict ISO mode (-std=c11) default to
 * -ffp-contract=off, so a plain "x += a*b" compiles to separate vmulpd+vaddpd
 * and costs half the achievable flops.  Ask for the fused form explicitly when
 * the target has a fast hardware FMA; otherwise fall back to plain arithmetic
 * (still correct, just slower).                                            */
#if defined(__FP_FAST_FMA) && (defined(__GNUC__) || defined(__clang__))
#  define KFMA(a,b,c) __builtin_fma((a), (b), (c))
#else
#  define KFMA(a,b,c) ((a) * (b) + (c))
#endif

#define CEILDIV(a,b) (((a)+(b)-1)/(b))
#define ROUNDUP(a,b) (CEILDIV(a,b)*(b))
#define KMAXTHR 256        /* cap on threads used inside one call */

/* ------------------------------------------------------------------ packing */

/* A block (mc x kc, row-major, ld=lda) -> row panels of KMR rows.
 * Panel ir holds, for p = 0..kc-1, KMR consecutive scalar_ts a[0..KMR-1][p].
 * alpha is folded in here so the C update is a bare accumulate.            */
static void pack_A(int mc, int kc, scalar_t alpha,
                   const scalar_t *restrict A, int lda,
                   scalar_t *restrict dst)
{
    for (int i0 = 0; i0 < mc; i0 += KMR) {
        int mrb = mc - i0;  if (mrb > KMR) mrb = KMR;
        const scalar_t *src = A + (size_t)i0 * lda;
        if (mrb == KMR) {
            for (int p = 0; p < kc; p++) {
                for (int i = 0; i < KMR; i++)
                    dst[i] = alpha * src[(size_t)i * lda + p];
                dst += KMR;
            }
        } else {
            for (int p = 0; p < kc; p++) {
                int i = 0;
                for (; i < mrb; i++) dst[i] = alpha * src[(size_t)i * lda + p];
                for (; i < KMR; i++) dst[i] = 0.0;
                dst += KMR;
            }
        }
    }
}

/* B block (kc x nc, row-major, ld=ldb) -> column panels of KNR columns.
 * Panel jr holds, for p = 0..kc-1, KNR consecutive scalar_ts b[p][0..KNR-1]. */
static void pack_B(int kc, int nc,
                   const scalar_t *restrict B, int ldb,
                   scalar_t *restrict dst)
{
    for (int j0 = 0; j0 < nc; j0 += KNR) {
        int nrb = nc - j0;  if (nrb > KNR) nrb = KNR;
        const scalar_t *src = B + j0;
        if (nrb == KNR) {
            for (int p = 0; p < kc; p++) {
                const scalar_t *s = src + (size_t)p * ldb;
                for (int j = 0; j < KNR; j++) dst[j] = s[j];
                dst += KNR;
            }
        } else {
            for (int p = 0; p < kc; p++) {
                const scalar_t *s = src + (size_t)p * ldb;
                int j = 0;
                for (; j < nrb; j++) dst[j] = s[j];
                for (; j < KNR; j++) dst[j] = 0.0;
                dst += KNR;
            }
        }
    }
}

/* ------------------------------------------------------------- micro-kernel */
/* Full KMR x KNR register tile.  Both packed panels are zero padded, so the
 * arithmetic is always done on the full tile and only the live mr x nr corner
 * is written back -- no edge-case code in the hot loop, no OOB touch of C.
 *
 * The row index of the accumulator tile is a compile-time constant in every
 * statement (macro-unrolled below).  That is what lets a plain autovectoriser
 * keep the whole tile in vector registers across the k loop instead of
 * spilling it to the stack. */

#define KROW(i) ab[i][j] = KFMA(a[i], bj, ab[i][j]);

#if   KMR == 4
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)
#elif KMR == 6
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)KROW(4)KROW(5)
#elif KMR == 8
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)KROW(4)KROW(5)KROW(6)KROW(7)
#elif KMR == 10
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)KROW(4)KROW(5)KROW(6)KROW(7)\
                 KROW(8)KROW(9)
#elif KMR == 12
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)KROW(4)KROW(5)KROW(6)KROW(7)\
                 KROW(8)KROW(9)KROW(10)KROW(11)
#elif KMR == 14
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)KROW(4)KROW(5)KROW(6)KROW(7)\
                 KROW(8)KROW(9)KROW(10)KROW(11)KROW(12)KROW(13)
#elif KMR == 16
#define KALLROWS KROW(0)KROW(1)KROW(2)KROW(3)KROW(4)KROW(5)KROW(6)KROW(7)\
                 KROW(8)KROW(9)KROW(10)KROW(11)KROW(12)KROW(13)KROW(14)KROW(15)
#else
#error "unsupported KMR"
#endif

static void micro_kernel(int kc,
                         const scalar_t *restrict Ap, const scalar_t *restrict Bp,
                         scalar_t *restrict C, int ldc, int mr, int nr)
{
    scalar_t ab[KMR][KNR];
    for (int i = 0; i < KMR; i++)
        for (int j = 0; j < KNR; j++) ab[i][j] = 0.0;

    for (int p = 0; p < kc; p++) {
        const scalar_t *restrict a = Ap + (size_t)p * KMR;
        const scalar_t *restrict b = Bp + (size_t)p * KNR;
        for (int j = 0; j < KNR; j++) {
            scalar_t bj = b[j];
            KALLROWS
        }
    }

    if (mr == KMR && nr == KNR) {
        for (int i = 0; i < KMR; i++) {
            scalar_t *restrict c = C + (size_t)i * ldc;
            for (int j = 0; j < KNR; j++) c[j] += ab[i][j];
        }
    } else {
        for (int i = 0; i < mr; i++) {
            scalar_t *restrict c = C + (size_t)i * ldc;
            for (int j = 0; j < nr; j++) c[j] += ab[i][j];
        }
    }
}

/* -------------------------------------------------------------- macro-kernel */
static void macro_kernel(int mc, int nc, int kc,
                         const scalar_t *restrict Ap, const scalar_t *restrict Bp,
                         scalar_t *restrict C, int ldc)
{
    for (int j0 = 0; j0 < nc; j0 += KNR) {
        int nr = nc - j0;  if (nr > KNR) nr = KNR;
        const scalar_t *bp = Bp + (size_t)(j0 / KNR) * (size_t)kc * KNR;
        for (int i0 = 0; i0 < mc; i0 += KMR) {
            int mr = mc - i0;  if (mr > KMR) mr = KMR;
            const scalar_t *ap = Ap + (size_t)(i0 / KMR) * (size_t)kc * KMR;
            micro_kernel(kc, ap, bp, C + (size_t)i0 * ldc + j0, ldc, mr, nr);
        }
    }
}

/* C11 aligned_alloc, matching util.c. The size must be a multiple of the
 * alignment, so round up. */
static void *aalloc(size_t bytes)
{
    if (bytes == 0) bytes = 64;
    return aligned_alloc(64, ((bytes + 63) / 64) * 64);
}

/* ------------------------------------------------------------------ fallback */
/* Used only if a scratch allocation fails: correct, just slow. */
static void gemm_ref(int m, int n, int k, scalar_t alpha,
                     const scalar_t *A, int lda, const scalar_t *B, int ldb,
                     scalar_t *C, int ldc)
{
    for (int i = 0; i < m; i++)
        for (int p = 0; p < k; p++) {
            scalar_t a = alpha * A[(size_t)i * lda + p];
            const scalar_t *b = B + (size_t)p * ldb;
            scalar_t *c = C + (size_t)i * ldc;
            for (int j = 0; j < n; j++) c[j] += a * b[j];
        }
}

/* ------------------------------------------------------------------- driver */
static void gemm_packed(int m, int n, int k, scalar_t alpha,
                   const scalar_t *A, int lda, const scalar_t *B, int ldb,
                   scalar_t *C, int ldc)
{
    if (m <= 0 || n <= 0 || k <= 0) return;

    const int kc_max = (k < KKC) ? k : KKC;
    const int mc_max = (m < KMC) ? m : KMC;
    const int nc_max = (n < KNC) ? n : KNC;

    const size_t bsz = (size_t)kc_max * (size_t)ROUNDUP(nc_max, KNR);
    const size_t asz = (size_t)kc_max * (size_t)ROUNDUP(mc_max, KMR);

    int nthr = 1;
#ifdef _OPENMP
    {
        int mblocks = CEILDIV(m, KMC);
        nthr = omp_get_max_threads();
        if (nthr > mblocks) nthr = mblocks;
        if (nthr < 1) nthr = 1;
        if (nthr > KMAXTHR) nthr = KMAXTHR;
        if (omp_in_parallel()) nthr = 1;   /* never nest */
    }
#endif

    /* One B buffer (shared, read-only in the parallel region) plus one private
     * A buffer per thread.  All obtained up front: if anything fails we take
     * the plain triple loop and no partial work has happened yet.           */
    scalar_t *Bpack = (scalar_t *)aalloc(bsz * sizeof(scalar_t));
    scalar_t *Apack[KMAXTHR];
    int ok = (Bpack != NULL);
    for (int t = 0; t < nthr; t++) Apack[t] = NULL;
    for (int t = 0; ok && t < nthr; t++) {
        Apack[t] = (scalar_t *)aalloc(asz * sizeof(scalar_t));
        if (!Apack[t]) ok = 0;
    }
    if (!ok) {
        for (int t = 0; t < nthr; t++) free(Apack[t]);
        free(Bpack);
        gemm_ref(m, n, k, alpha, A, lda, B, ldb, C, ldc);
        return;
    }

    for (int j0 = 0; j0 < n; j0 += KNC) {
        int nc = n - j0;  if (nc > KNC) nc = KNC;
        for (int p0 = 0; p0 < k; p0 += KKC) {
            int kc = k - p0;  if (kc > KKC) kc = KKC;

            pack_B(kc, nc, B + (size_t)p0 * ldb + j0, ldb, Bpack);

#ifdef _OPENMP
            #pragma omp parallel for num_threads(nthr) schedule(dynamic,1) \
                    if (nthr > 1)
#endif
            for (int i0 = 0; i0 < m; i0 += KMC) {
                int mc = m - i0;  if (mc > KMC) mc = KMC;
#ifdef _OPENMP
                scalar_t *ap = Apack[omp_get_thread_num() % nthr];
#else
                scalar_t *ap = Apack[0];
#endif
                pack_A(mc, kc, alpha, A + (size_t)i0 * lda + p0, lda, ap);
                macro_kernel(mc, nc, kc, ap, Bpack,
                             C + (size_t)i0 * ldc + j0, ldc);
            }
        }
    }

    for (int t = 0; t < nthr; t++) free(Apack[t]);
    free(Bpack);
}

void kernel_gemm_acc(int m, int n, int k, scalar_t alpha,
                     const scalar_t *A, int lda,
                     const scalar_t *B, int ldb,
                     scalar_t *C, int ldc, kernel_t which)
{
    double t0;
    if (m <= 0 || n <= 0 || k <= 0) return;
    t0 = MPI_Wtime();
    switch (which) {
    case KRN_NAIVE:   gemm_naive  (m, n, k, alpha, A, lda, B, ldb, C, ldc); break;
    case KRN_BLOCKED: gemm_blocked(m, n, k, alpha, A, lda, B, ldb, C, ldc); break;
    default:          gemm_packed (m, n, k, alpha, A, lda, B, ldb, C, ldc); break;
    }
    g_cnt.t_comp += MPI_Wtime() - t0;
}
