/* gemm2d.h — distributed-memory generic GEMM with 2D / 2.5D decomposition
 *
 * C := alpha * A * B + beta * C,  A: MxK, B: KxN, C: MxN
 * Arbitrary M, N, K, arbitrary P, arbitrary Pr x Pc x c process grid.
 */
#ifndef GEMM2D_H
#define GEMM2D_H

#include <mpi.h>
#include <stddef.h>
#include <stdint.h>

typedef double scalar_t;
#define MPI_SCALAR MPI_DOUBLE

/* ------------------------------------------------------------------ */
/* Balanced block partition of G items over P parts.                   */
/* Part i has blk_size() items starting at blk_off(); sizes differ by  */
/* at most one, which is what makes non-divisible dimensions work.     */
/* ------------------------------------------------------------------ */
static inline int blk_size(int G, int P, int i)
{
    return G / P + (i < (G % P) ? 1 : 0);
}
static inline int blk_off(int G, int P, int i)
{
    int q = G / P, r = G % P;
    return i * q + (i < r ? i : r);
}
int blk_owner(int G, int P, int k);   /* inverse of blk_off */

/* ------------------------------------------------------------------ */
/* Enums                                                               */
/* ------------------------------------------------------------------ */
typedef enum { ENG_NAIVE1D = 0, ENG_CANNON, ENG_SUMMA, ENG_SUMMA25D } engine_t;
typedef enum { BC_BLOCKING = 0, BC_IBCAST, BC_SHM }                  bcast_t;
typedef enum { KRN_NAIVE = 0, KRN_BLOCKED }                          kernel_t;
typedef enum { MAP_LINEAR = 0, MAP_NODEAWARE }                       gridmap_t;
typedef enum { VER_NONE = 0, VER_REF, VER_FREIVALDS }                verify_t;

const char *engine_name(engine_t e);
const char *bcast_name(bcast_t b);
const char *kernel_name(kernel_t k);
const char *gridmap_name(gridmap_t g);

/* ------------------------------------------------------------------ */
/* Instrumentation counters (per rank, reset each repetition)          */
/* ------------------------------------------------------------------ */
typedef struct {
    double t_comm;        /* seconds inside panel broadcasts           */
    double t_comp;        /* seconds inside the local kernel           */
    double t_pack;        /* seconds packing panels                    */
    double bytes_offnode; /* modeled panel bytes crossing a node edge  */
    double bytes_rank;    /* panel bytes delivered to this rank        */
    long   n_bcast;
} counters_t;

extern counters_t g_cnt;
void counters_reset(void);

/* ------------------------------------------------------------------ */
/* Process grid: Pr x Pc x c                                           */
/* grid rank = (layer*Pr + row)*Pc + col                               */
/* ------------------------------------------------------------------ */
typedef struct {
    MPI_Comm world;
    MPI_Comm grid;       /* world, permuted according to gridmap       */
    MPI_Comm row;        /* Pc ranks: same layer, same row             */
    MPI_Comm col;        /* Pr ranks: same layer, same col             */
    MPI_Comm depth;      /* c  ranks: same (row,col), all layers       */
    MPI_Comm layer0;     /* ranks with my_layer == 0 (else COMM_NULL)  */
    int  P, world_rank, grid_rank;
    int  Pr, Pc, C;
    int  my_row, my_col, my_layer;
    int  node_id, node_rank, node_size, n_nodes;
    gridmap_t map;
} pgrid_t;

int  pgrid_create(MPI_Comm world, int Pr, int Pc, int c, gridmap_t map, pgrid_t *g);
void pgrid_free(pgrid_t *g);
/* Aspect-matched default factorisation: Pr ~ sqrt(P/c * M/N). */
void pgrid_default_shape(int P, int c, int M, int N, int *Pr, int *Pc);

/* ------------------------------------------------------------------ */
/* Distributed matrix (a ScaLAPACK-lite descriptor)                    */
/* row0/col0 are TRUE GLOBAL indices of local element (0,0), so the    */
/* deterministic generator and the verifier need no special cases.     */
/* Local storage is row-major with ld == n.                            */
/* ------------------------------------------------------------------ */
typedef struct {
    scalar_t *data;
    int gm, gn;        /* dims of the full global matrix        */
    int m, n, ld;      /* local block                           */
    int row0, col0;    /* global index of local (0,0)           */
} dmat_t;

int  dmat_alloc(dmat_t *A, int gm, int gn, int m, int n, int row0, int col0);
void dmat_free(dmat_t *A);
void dmat_fill(dmat_t *A, uint64_t seed);   /* deterministic, layout-independent */
void dmat_zero(dmat_t *A);
void dmat_scale(dmat_t *A, scalar_t s);

/* Deterministic element generator: value depends only on (seed,i,j). */
scalar_t gen_elem(uint64_t seed, int i, int j);

/* ------------------------------------------------------------------ */
/* Panel broadcast channel                                             */
/*                                                                     */
/* One interface, three policies. The engine always does:              */
/*     if (I am root) pack into chan_wbuf();                           */
/*     chan_post(c, cnt, root);  chan_wait(c);                         */
/*     read chan_rbuf();                                               */
/*                                                                     */
/* BC_SHM is the headline technique: the panel lives in an MPI-3       */
/* shared-memory window, only one leader per node takes part in the    */
/* inter-node broadcast, and every other rank on the node reads the    */
/* panel straight out of shared memory. Off-node bytes drop by a       */
/* factor of (ranks per node within the communicator).                 */
/* ------------------------------------------------------------------ */
typedef struct {
    bcast_t   policy;
    MPI_Comm  comm;
    int       rank, size;

    /* shared-memory path */
    MPI_Comm  shm;          /* ranks of comm that share memory   */
    MPI_Comm  leaders;      /* one rank per node (COMM_NULL else)*/
    int       shm_rank, shm_size, is_leader, n_nodes;
    int      *leader_of;    /* [size] -> rank within leaders     */
    MPI_Win   win;
    scalar_t *sbuf;         /* base of rank-0-of-node segment    */

    /* plain path */
    scalar_t *lbuf;
    MPI_Request req;
    int       pending;

    size_t    cap;          /* capacity in elements              */
} chan_t;

int   chan_create(MPI_Comm comm, bcast_t policy, size_t cap, chan_t *c);
void  chan_free(chan_t *c);
scalar_t *chan_wbuf(chan_t *c);
scalar_t *chan_rbuf(chan_t *c);
void  chan_begin(chan_t *c);
void  chan_post(chan_t *c, int cnt, int root);
void  chan_wait(chan_t *c);

/* ------------------------------------------------------------------ */
/* Local kernel                                                        */
/* ------------------------------------------------------------------ */
void kernel_gemm_acc(int m, int n, int k, scalar_t alpha,
                     const scalar_t *A, int lda,
                     const scalar_t *B, int ldb,
                     scalar_t *C, int ldc, kernel_t which);

/* ------------------------------------------------------------------ */
/* Engines — all share one signature                                   */
/* ------------------------------------------------------------------ */
typedef struct {
    engine_t  engine;
    bcast_t   bcast;
    kernel_t  kernel;
    int       b;          /* SUMMA panel width          */
    int       c;          /* replication depth          */
    int       lookahead;  /* 0 or 1 (BC_IBCAST only)    */
    scalar_t  alpha, beta;
} gemm_opts_t;

int gemm_summa(const pgrid_t *g, const dmat_t *A, const dmat_t *B, dmat_t *C,
               const gemm_opts_t *o, int Koff, int Klen);
int gemm_summa25d(const pgrid_t *g, const dmat_t *A, const dmat_t *B, dmat_t *C,
                  const gemm_opts_t *o, int K);
int gemm_cannon(const pgrid_t *g, const dmat_t *A, const dmat_t *B, dmat_t *C,
                const gemm_opts_t *o, int K);
int gemm_naive1d(MPI_Comm comm, int M, int N, int K, const gemm_opts_t *o,
                 uint64_t seed, double *err_out, verify_t ver);

/* ------------------------------------------------------------------ */
/* Verification                                                        */
/* ------------------------------------------------------------------ */
double verify_reference(const pgrid_t *g, const dmat_t *C, int M, int N, int K,
                        scalar_t alpha, scalar_t beta, uint64_t sA, uint64_t sB,
                        uint64_t sC, long max_elems);
/* Freivalds: O(n^2) randomised check, usable at full scale.
 * v0 must hold trials consecutive vectors (C_initial * r_t), one per
 * trial, each of length C->m, when beta != 0; otherwise NULL. */
double verify_freivalds(const pgrid_t *g, const dmat_t *A, const dmat_t *B,
                        const dmat_t *C, int M, int N, int K, int Koff, int Klen,
                        scalar_t alpha, scalar_t beta, const scalar_t *v0,
                        uint64_t rseed, int trials);
scalar_t *freivalds_cr(const pgrid_t *g, const dmat_t *C, int N,
                       uint64_t rseed, int trial);

/* ------------------------------------------------------------------ */
/* Machine model + planner                                             */
/* ------------------------------------------------------------------ */
typedef struct {
    double alpha_lat;   /* s per message   */
    double beta_bw;     /* s per byte      */
    double gamma_flop;  /* s per flop      */
    double mem_bytes;   /* usable bytes per rank */
} machine_t;

typedef struct {
    int Pr, Pc, c, b;
    double t_model, t_comm, t_comp;
} plan_t;

void machine_defaults(machine_t *mm);
void machine_calibrate(MPI_Comm comm, machine_t *mm, int verbose);
double plan_cost(const machine_t *mm, int M, int N, int K, int P,
                 int Pr, int Pc, int c, int b, double *t_comm, double *t_comp);
int  plan_choose(const machine_t *mm, int M, int N, int K, int P,
                 int allow_25d, plan_t *best, plan_t *top, int ntop);

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */
void   node_split(MPI_Comm comm, MPI_Comm *out);
double wtime(void);
int    ipow2_floor(int x);

#endif /* GEMM2D_H */
