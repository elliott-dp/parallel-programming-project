#include "gemm2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

counters_t g_cnt;

void counters_reset(void) { memset(&g_cnt, 0, sizeof(g_cnt)); }

double wtime(void) { return MPI_Wtime(); }

int ipow2_floor(int x)
{
    int p = 1;
    while (p * 2 <= x) p *= 2;
    return p;
}

int blk_owner(int G, int P, int k)
{
    int q, r;
    if (k < 0)  k = 0;
    if (k >= G) k = G - 1;
    q = G / P;
    r = G % P;
    if (q == 0) return k;                 /* P > G: first G parts hold 1 each */
    if (k < r * (q + 1)) return k / (q + 1);
    return r + (k - r * (q + 1)) / q;
}

const char *engine_name(engine_t e)
{
    switch (e) {
    case ENG_NAIVE1D:   return "naive1d";
    case ENG_CANNON:    return "cannon";
    case ENG_SUMMA:     return "summa";
    case ENG_SUMMA25D:  return "summa25d";
    }
    return "?";
}
const char *bcast_name(bcast_t b)
{
    switch (b) {
    case BC_BLOCKING: return "blocking";
    case BC_IBCAST:   return "ibcast";
    case BC_SHM:      return "shm";
    }
    return "?";
}
const char *kernel_name(kernel_t k)
{
    return k == KRN_NAIVE ? "naive" : "blocked";
}
const char *gridmap_name(gridmap_t g)
{
    return g == MAP_LINEAR ? "linear" : "nodeaware";
}

/* --------------------------------------------------------------------
 * Deterministic, layout-independent element generator.
 * The value of a global element depends only on (seed, i, j), so every
 * process can create its own block with no communication, and results
 * are bit-comparable across different values of P and different grids.
 * -------------------------------------------------------------------- */
scalar_t gen_elem(uint64_t seed, int i, int j)
{
    uint64_t x = seed;
    x ^= 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
    x ^= 0xC2B2AE3D27D4EB4FULL * (uint64_t)(j + 1);
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31;
    /* map to [-1, 1) */
    return ((double)(x >> 11) / 9007199254740992.0) * 2.0 - 1.0;
}

int dmat_alloc(dmat_t *A, int gm, int gn, int m, int n, int row0, int col0)
{
    size_t nel = (size_t)(m > 0 ? m : 1) * (size_t)(n > 0 ? n : 1);
    A->gm = gm; A->gn = gn;
    A->m = m;   A->n = n;   A->ld = n;
    A->row0 = row0; A->col0 = col0;
    A->data = (scalar_t *)aligned_alloc(64, ((nel * sizeof(scalar_t) + 63) / 64) * 64);
    if (!A->data) return -1;
    memset(A->data, 0, nel * sizeof(scalar_t));
    return 0;
}

void dmat_free(dmat_t *A)
{
    free(A->data);
    A->data = NULL;
}

void dmat_zero(dmat_t *A)
{
    if (A->m > 0 && A->n > 0)
        memset(A->data, 0, (size_t)A->m * A->ld * sizeof(scalar_t));
}

void dmat_scale(dmat_t *A, scalar_t s)
{
    int i, j;
    if (s == (scalar_t)1.0) return;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            A->data[(size_t)i * A->ld + j] *= s;
}

void dmat_fill(dmat_t *A, uint64_t seed)
{
    int i, j;
    for (i = 0; i < A->m; i++)
        for (j = 0; j < A->n; j++)
            A->data[(size_t)i * A->ld + j] = gen_elem(seed, A->row0 + i, A->col0 + j);
}

/* --------------------------------------------------------------------
 * Node decomposition of a communicator.
 *
 * Normally this is MPI_COMM_TYPE_SHARED. Setting GEMM2D_FAKE_PPN=k
 * pretends that every k consecutive ranks form a node, which lets the
 * shared-memory broadcast path be exercised and validated on a single
 * machine before moving to the cluster. The window is still a real
 * shared-memory window, only the grouping is synthetic.
 * -------------------------------------------------------------------- */
void node_split(MPI_Comm comm, MPI_Comm *out)
{
    const char *e = getenv("GEMM2D_FAKE_PPN");
    int ppn = e ? atoi(e) : 0;
    if (ppn > 0) {
        int wr;
        /* node identity must be a property of the physical rank, not of
         * the sub-communicator, so that row and column communicators
         * agree about who shares a node */
        MPI_Comm_rank(MPI_COMM_WORLD, &wr);
        MPI_Comm_split(comm, wr / ppn, wr, out);
    } else {
        int r;
        MPI_Comm_rank(comm, &r);
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, r, MPI_INFO_NULL, out);
    }
}
