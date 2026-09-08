#include "gemm2d.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* Aspect-matched default grid: minimising K*(M/Pr + N/Pc) under
 * Pr*Pc = P gives Pr = sqrt(P * M/N).  We snap to the nearest divisor. */
void pgrid_default_shape(int P, int c, int M, int N, int *Pr, int *Pc)
{
    int Pl = P / c, best = 1, r;
    double target = sqrt((double)Pl * (double)M / (double)(N > 0 ? N : 1));
    double bestd = 1e30;
    for (r = 1; r <= Pl; r++) {
        if (Pl % r) continue;
        double d = fabs((double)r - target);
        if (d < bestd) { bestd = d; best = r; }
    }
    *Pr = best;
    *Pc = Pl / best;
}

/* Split ppn (ranks per node) into a pr_n x pc_n tile whose aspect ratio
 * is as close as possible to Pr:Pc, subject to divisibility of the grid. */
static int node_tile(int ppn, int Pr, int Pc, int *pr_n, int *pc_n)
{
    int r, bestr = 0, bestc = 0;
    double target = (double)Pr / (double)Pc, bestd = 1e30;
    for (r = 1; r <= ppn; r++) {
        if (ppn % r) continue;
        int cc = ppn / r;
        if (Pr % r || Pc % cc) continue;
        double d = fabs((double)r / (double)cc - target);
        if (d < bestd) { bestd = d; bestr = r; bestc = cc; }
    }
    if (!bestr) return -1;
    *pr_n = bestr;
    *pc_n = bestc;
    return 0;
}

int pgrid_create(MPI_Comm world, int Pr, int Pc, int c, gridmap_t map, pgrid_t *g)
{
    int P, wr, key, ok = 1;
    MPI_Comm shm_probe;
    int ppn;

    MPI_Comm_size(world, &P);
    MPI_Comm_rank(world, &wr);
    if ((long)Pr * Pc * c != P) return -1;

    g->world = world; g->P = P; g->world_rank = wr;
    g->Pr = Pr; g->Pc = Pc; g->C = c; g->map = map;

    /* how many ranks share a node? (used by the node-aware map and by
     * the shared-memory broadcast) */
    node_split(world, &shm_probe);
    MPI_Comm_size(shm_probe, &ppn);

    key = wr;   /* MAP_LINEAR: identity permutation */

    if (map == MAP_NODEAWARE) {
        int pr_n, pc_n, uniform = 1, minppn, maxppn;
        MPI_Allreduce(&ppn, &minppn, 1, MPI_INT, MPI_MIN, world);
        MPI_Allreduce(&ppn, &maxppn, 1, MPI_INT, MPI_MAX, world);
        if (minppn != maxppn || P % ppn) uniform = 0;
        if (uniform && node_tile(ppn, Pr, Pc, &pr_n, &pc_n) == 0) {
            int ngr = Pr / pr_n, ngc = Pc / pc_n;
            int nodes_per_layer = ngr * ngc;
            int node = wr / ppn, local = wr % ppn;
            int layer = node / nodes_per_layer;
            int nil   = node % nodes_per_layer;
            int nr = nil / ngc, nc = nil % ngc;
            int lr = local / pc_n, lc = local % pc_n;
            int row = nr * pr_n + lr, col = nc * pc_n + lc;
            key = (layer * Pr + row) * Pc + col;
        } else {
            ok = 0;   /* fall back to linear, reported by the caller */
        }
    }
    MPI_Comm_free(&shm_probe);

    MPI_Comm_split(world, 0, key, &g->grid);
    MPI_Comm_rank(g->grid, &g->grid_rank);

    g->my_layer = g->grid_rank / (Pr * Pc);
    g->my_row   = (g->grid_rank % (Pr * Pc)) / Pc;
    g->my_col   = g->grid_rank % Pc;

    /* row  = same layer, same row  (Pc ranks) */
    MPI_Comm_split(g->grid, g->my_layer * Pr + g->my_row, g->my_col, &g->row);
    /* col  = same layer, same col  (Pr ranks) */
    MPI_Comm_split(g->grid, g->my_layer * Pc + g->my_col, g->my_row, &g->col);
    /* depth = same (row,col), across layers (c ranks) */
    MPI_Comm_split(g->grid, g->my_row * Pc + g->my_col, g->my_layer, &g->depth);
    /* layer 0 only */
    MPI_Comm_split(g->grid, g->my_layer == 0 ? 0 : MPI_UNDEFINED,
                   g->grid_rank, &g->layer0);

    /* node identity, for reporting */
    {
        MPI_Comm shm;
        int leader;
        node_split(g->grid, &shm);
        MPI_Comm_rank(shm, &g->node_rank);
        MPI_Comm_size(shm, &g->node_size);
        leader = (g->node_rank == 0) ? 1 : 0;
        MPI_Allreduce(&leader, &g->n_nodes, 1, MPI_INT, MPI_SUM, g->grid);
        g->node_id = g->grid_rank / (g->node_size > 0 ? g->node_size : 1);
        MPI_Comm_free(&shm);
    }
    return ok ? 0 : 1;   /* 1 = node-aware requested but fell back */
}

void pgrid_free(pgrid_t *g)
{
    if (g->row    != MPI_COMM_NULL) MPI_Comm_free(&g->row);
    if (g->col    != MPI_COMM_NULL) MPI_Comm_free(&g->col);
    if (g->depth  != MPI_COMM_NULL) MPI_Comm_free(&g->depth);
    if (g->layer0 != MPI_COMM_NULL) MPI_Comm_free(&g->layer0);
    if (g->grid   != MPI_COMM_NULL) MPI_Comm_free(&g->grid);
}
