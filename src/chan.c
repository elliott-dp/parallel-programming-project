/* chan.c — panel broadcast channel.
 *
 * Three interchangeable policies behind one interface:
 *
 *   BC_BLOCKING : MPI_Bcast on the full row/column communicator.
 *   BC_IBCAST   : MPI_Ibcast, so the engine can run one step of lookahead.
 *   BC_SHM      : node-aware two-level broadcast (the technique).
 *
 * BC_SHM in one paragraph: in SUMMA every rank of a process row receives
 * the *same* A-panel, so with several ranks per node the identical bytes
 * are pulled onto that node once per rank. Here the panel lives in an
 * MPI-3 shared-memory window owned jointly by the ranks of the row that
 * happen to share a node. Only one leader per node takes part in the
 * inter-node MPI_Bcast; everyone else reads the panel directly out of
 * shared memory. Off-node panel traffic falls by a factor equal to the
 * number of ranks per node inside that communicator, and the per-node
 * panel footprint falls from (ranks per node) copies to one.
 */
#include "gemm2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int chan_create(MPI_Comm comm, bcast_t policy, size_t cap, chan_t *c)
{
    int leader, myleader = -1;

    memset(c, 0, sizeof(*c));
    c->policy = policy;
    c->comm   = comm;
    c->cap    = cap ? cap : 1;
    MPI_Comm_rank(comm, &c->rank);
    MPI_Comm_size(comm, &c->size);

    /* The node decomposition is always built: BC_SHM needs it to work,
     * the other policies need it so that the off-node byte counter is
     * measured the same way for every policy. */
    node_split(comm, &c->shm);
    MPI_Comm_rank(c->shm, &c->shm_rank);
    MPI_Comm_size(c->shm, &c->shm_size);
    c->is_leader = (c->shm_rank == 0);

    MPI_Comm_split(comm, c->is_leader ? 0 : MPI_UNDEFINED, c->rank, &c->leaders);
    if (c->leaders != MPI_COMM_NULL) MPI_Comm_rank(c->leaders, &myleader);
    MPI_Bcast(&myleader, 1, MPI_INT, 0, c->shm);      /* share within the node */

    leader = c->is_leader ? 1 : 0;
    MPI_Allreduce(&leader, &c->n_nodes, 1, MPI_INT, MPI_SUM, comm);

    c->leader_of = (int *)malloc((size_t)c->size * sizeof(int));
    if (!c->leader_of) return -1;
    MPI_Allgather(&myleader, 1, MPI_INT, c->leader_of, 1, MPI_INT, comm);

    if (policy == BC_SHM) {
        MPI_Aint sz = (c->shm_rank == 0)
                    ? (MPI_Aint)(c->cap * sizeof(scalar_t)) : 0;
        void *ptr = NULL;
        MPI_Aint qsz; int qdisp;
        MPI_Win_allocate_shared(sz, (int)sizeof(scalar_t), MPI_INFO_NULL,
                                c->shm, &ptr, &c->win);
        /* every rank addresses the single segment owned by node-rank 0 */
        MPI_Win_shared_query(c->win, 0, &qsz, &qdisp, &c->sbuf);
        MPI_Win_lock_all(MPI_MODE_NOCHECK, c->win);
    } else {
        c->lbuf = (scalar_t *)aligned_alloc(64,
                     ((c->cap * sizeof(scalar_t) + 63) / 64) * 64);
        if (!c->lbuf) return -1;
    }
    return 0;
}

void chan_free(chan_t *c)
{
    if (c->policy == BC_SHM) {
        MPI_Win_unlock_all(c->win);
        MPI_Win_free(&c->win);
    } else {
        free(c->lbuf);
    }
    free(c->leader_of);
    if (c->leaders != MPI_COMM_NULL) MPI_Comm_free(&c->leaders);
    MPI_Comm_free(&c->shm);
    memset(c, 0, sizeof(*c));
}

scalar_t *chan_wbuf(chan_t *c) { return c->policy == BC_SHM ? c->sbuf : c->lbuf; }
scalar_t *chan_rbuf(chan_t *c) { return c->policy == BC_SHM ? c->sbuf : c->lbuf; }

/* Modeled bytes: what the panel costs this rank, and what it costs the
 * node boundary. Counted identically for every policy so the ablation is
 * apples-to-apples. */
static void account(chan_t *c, int cnt, int root)
{
    double bytes = (double)cnt * (double)sizeof(scalar_t);
    g_cnt.n_bcast++;
    if (c->rank != root) g_cnt.bytes_rank += bytes;

    if (c->policy == BC_SHM) {
        if (c->is_leader && c->leader_of[c->rank] != c->leader_of[root])
            g_cnt.bytes_offnode += bytes;
    } else {
        if (c->leader_of[c->rank] != c->leader_of[root])
            g_cnt.bytes_offnode += bytes;
    }
}

/* Must be called by every rank of the communicator BEFORE the root
 * writes its panel into the shared buffer.
 *
 * Without this, a rank that finishes its local update early races ahead
 * to the next step and overwrites the shared panel while a slower rank
 * on the same node is still reading it. With a private receive buffer
 * per rank (the blocking and Ibcast paths) that race cannot happen, so
 * it is specific to the shared-memory optimisation: the panel is no
 * longer private, so the reuse of the buffer has to be synchronised. */
void chan_begin(chan_t *c)
{
    double t0;
    if (c->policy != BC_SHM || c->shm_size <= 1) return;
    t0 = MPI_Wtime();
    MPI_Win_sync(c->win);
    MPI_Barrier(c->shm);
    MPI_Win_sync(c->win);
    g_cnt.t_comm += MPI_Wtime() - t0;
}

void chan_post(chan_t *c, int cnt, int root)
{
    double t0 = MPI_Wtime();

    account(c, cnt, root);

    switch (c->policy) {
    case BC_BLOCKING:
        MPI_Bcast(c->lbuf, cnt, MPI_SCALAR, root, c->comm);
        c->pending = 0;
        break;

    case BC_IBCAST:
        MPI_Ibcast(c->lbuf, cnt, MPI_SCALAR, root, c->comm, &c->req);
        c->pending = 1;
        break;

    case BC_SHM:
        /* 1. make the root's write to the shared segment visible */
        MPI_Win_sync(c->win);
        MPI_Barrier(c->shm);
        MPI_Win_sync(c->win);
        /* 2. one message per node instead of one per rank */
        if (c->n_nodes > 1) {
            if (c->is_leader)
                MPI_Bcast(c->sbuf, cnt, MPI_SCALAR, c->leader_of[root], c->leaders);
            /* 3. publish the leader's write to the rest of the node */
            MPI_Win_sync(c->win);
            MPI_Barrier(c->shm);
            MPI_Win_sync(c->win);
        }
        c->pending = 0;
        break;
    }
    g_cnt.t_comm += MPI_Wtime() - t0;
}

void chan_wait(chan_t *c)
{
    double t0;
    if (!c->pending) return;
    t0 = MPI_Wtime();
    MPI_Wait(&c->req, MPI_STATUS_IGNORE);
    c->pending = 0;
    g_cnt.t_comm += MPI_Wtime() - t0;
}
