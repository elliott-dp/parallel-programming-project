# Distributed-memory generic matrix multiplication with 2D decomposition
## Architecture and project guide

Target: ParCo final project / D-F deliverable — 4-page IEEE report + git repository + 10–15 min presentation.
Platform assumed: the UniTN HPC cluster (PBS, `qsub`, `short_cpuQ`, `module load gcc91`, Open MPI 5.x = MPI-4.0), C or C++ with MPI + OpenMP.

---

## 1. What "generic" should mean here (and why it is the whole project)

Most student implementations of 2D matmul silently assume `M = N = K`, `P` a perfect square, and `n` divisible by `√P`. That is a *demo*, not a GEMM. The word "generic" in the title is the opportunity: it is where the originality points live, and it is exactly what the classical literature (SUMMA, COSMA) was written to solve.

Define genericity on three axes and state them in the report's introduction:

| Axis | Requirement | Where it bites |
|---|---|---|
| **G1 — shape** | Arbitrary `M, N, K` (square, tall-skinny, short-fat), `α·A·B + β·C`, optional transposes | Choice of process grid; a fixed `√P × √P` grid is badly wrong for skewed shapes |
| **G2 — machine** | Arbitrary `P` (including primes), non-divisible dimensions, arbitrary node/core counts | Uneven block sizes, v-collectives, grid factorisation |
| **G3 — type** | `float`, `double`, `int32`, (`_Float16` optional) through one code path | Templates / X-macros + MPI datatype traits |

**Recommendation:** make G1 and G2 mandatory (they carry the science), and G3 a clearly scoped bonus (it costs ~1 day and gives you a free mixed-precision experiment that is very visible on a roofline plot).

Non-goals, state them explicitly so the report is honest: sparse matrices, Strassen-class fast algorithms, GPU offload (unless you get time), fault tolerance, out-of-core.

---

## 2. Design principles

1. **One data layout, several algorithms.** All engines operate on the same distributed block layout so you can A/B them at runtime with a flag. That is what makes the experiments cheap.
2. **Separate policy from mechanism.** *Which* grid and panel width to use is a decision made once by a planner; *how* to move a panel is a swappable communication policy. Ablations then become one CLI flag each, not three forks of the code.
3. **Correctness before performance** (this is also the course's stated workflow). Ship a working naive path first, keep it forever as the reference.
4. **Every design choice must produce a plot.** If a component cannot be ablated, it does not belong in a 4-page report.
5. **Measure with a model in hand.** The course explicitly distinguishes benchmarking from performance modelling — do both, and put measured-vs-model on the same axes.

---

## 3. Layered architecture

```
L5  driver / harness      main.c, args, generator, verifier, timing, CSV emitter
L4  planner               cost model -> (Pr, Pc, c, b) and engine choice
L3  engines               summa | summa25d | cannon | naive1d   (same signature)
L2  communication         grid + comms, panel pack/unpack, collective policy
L1  local kernel          cache-blocked, OpenMP-threaded rank-b update
L0  runtime               MPI-4, OpenMP, gcc; PBS job scripts
```

### Module contracts (C sketch)

```c
/* --- L2: grid ------------------------------------------------------ */
typedef struct {
    MPI_Comm world, grid;         /* grid = world, possibly reordered   */
    MPI_Comm row, col, depth;     /* from MPI_Comm_split on the grid    */
    int Pr, Pc, Pc_layers;        /* Pr*Pc*Pc_layers == P               */
    int my_row, my_col, my_layer;
} pgrid_t;

int pgrid_create(MPI_Comm world, int Pr, int Pc, int c, pgrid_t *g);

/* --- L2: distributed matrix (a mini "descriptor", ScaLAPACK-lite) --- */
typedef struct {
    scalar_t *data;               /* local block, row-major             */
    int  gm, gn;                  /* global dims                        */
    int  m,  n,  ld;              /* local dims + leading dimension     */
    int  row0, col0;              /* global index of local element (0,0)*/
    const pgrid_t *g;
} dmat_t;

int  dmat_alloc (const pgrid_t *g, int gm, int gn, dmat_t *A);
void dmat_fill  (dmat_t *A, uint64_t seed);      /* deterministic, rank-independent */
int  dmat_gather(const dmat_t *A, scalar_t *out_root); /* small sizes only, for tests */

/* --- L3: one signature for every engine ---------------------------- */
typedef struct {
    int  panel_b;                 /* SUMMA panel width                  */
    int  c;                       /* replication depth (1 = plain 2D)   */
    enum { COMM_BLOCKING, COMM_IBCAST, COMM_PERSISTENT } comm_policy;
    int  lookahead;               /* 0 = none, 1 = double buffer, ...    */
} gemm_opts_t;

int gemm_summa   (scalar_t alpha, const dmat_t *A, const dmat_t *B,
                  scalar_t beta,  dmat_t *C, const gemm_opts_t *o);
int gemm_summa25d(/* same */);
int gemm_cannon  (/* same */);

/* --- L1: local kernel ---------------------------------------------- */
void kernel_gemm(int m, int n, int k, scalar_t alpha,
                 const scalar_t *A, int lda, const scalar_t *B, int ldb,
                 scalar_t beta, scalar_t *C, int ldc);
```

Keeping `dmat_t` as an explicit descriptor (global dims + local dims + global offset) is the single most valuable structural decision: every "generic" edge case (non-divisible sizes, ragged last block, uneven grids) collapses into computing `row0/col0/m/n` once, in one function, instead of being scattered across every loop.

---

## 4. Data distribution

### 4.1 Block vs block-cyclic

Use **plain 2D block** distribution (each process owns one contiguous tile), not ScaLAPACK's 2D block-cyclic.

- Block-cyclic exists to load-balance *triangular/trapezoidal* work (LU, Cholesky, QR). GEMM's work is uniform, so cyclic buys nothing and costs you index arithmetic and a worse local kernel (strided tiles).
- Say this in the report with the citation to ScaLAPACK — "we deliberately deviate from X because Y" is exactly the kind of justified choice that scores.

### 4.2 Handling non-divisible dimensions (G2)

Two options; pick **balanced uneven blocks**, not padding:

```
m_i = M/Pr + (i < M%Pr ? 1 : 0)          /* differs by at most 1 row */
row0_i = i*(M/Pr) + min(i, M%Pr)
```

- Padding with zeros is simpler but wastes up to `O((Pr+Pc)·n)` flops and, worse, silently hides bugs (zeros make wrong results look right).
- Uneven blocks force you to use **v-collectives** (`MPI_Allgatherv`, `MPI_Scatterv`) or per-step counts in the panel broadcast. The course lecture on v-collectives is precisely the justification to cite.

### 4.3 Local layout and leading dimension

Row-major with an explicit `ld` (round `ld` up to a multiple of the SIMD width, e.g. 8 doubles) and 64-byte-aligned allocation (`aligned_alloc`). This connects directly to the course's cache/false-sharing examples: mis-aligned `ld` and column-wise access are the two things `cache_01_slow.c` was built to demonstrate.

### 4.4 Initial distribution

Do **not** build the matrix on rank 0 and scatter it — that caps your problem size at one node's memory and makes rank 0 the bottleneck. Instead have every process generate its own block from a **deterministic, index-based PRNG** (e.g. a splitmix64 hashed with the global `(i,j)`), so the same global matrix is produced for any `P` and any grid. This makes results across configurations bit-comparable and is a reproducibility argument for the report.

---

## 5. Communicator topology

```
MPI_COMM_WORLD
   └─ grid (Pr × Pc × c)                        rank = (layer*Pr + row)*Pc + col
        ├─ row   comm  (Pc procs)  colour = layer*Pr + row   → A-panel broadcast
        ├─ col   comm  (Pr procs)  colour = layer*Pc + col   → B-panel broadcast
        └─ depth comm  (c procs)   colour = row*Pc + col     → final reduction of C
```

Build these with `MPI_Comm_split` (course lecture material). `MPI_Cart_create` + `MPI_Cart_sub` is the textbook alternative and gives you `MPI_Cart_shift` for free — worth using for the Cannon engine, which is *defined* by nearest-neighbour shifts. Mention `reorder = 1` in the report: it lets the MPI implementation map ranks onto the network topology, which is a free (if usually small) win and shows awareness of L14's "implementations exploit network topology" point.

**Rank-to-node mapping matters more than reorder.** With `mpiexec --map-by ppr:N:node`, consecutive ranks land on the same node; a row-major grid then puts each process *row* inside a node, so row broadcasts become intra-node (shared memory) and only column broadcasts cross the network. Make this a measured ablation — it is nearly free and often worth 10–20%.

---

## 6. Algorithm engines

Notation: `A` is `M×K`, `B` is `K×N`, `C` is `M×N`; `P` processes; grid `Pr × Pc × c`; word size `w` bytes; α = message latency, β = time per byte, γ = time per flop (all measured, see §9).

### 6.1 E0 — naive 1D row decomposition (baseline only)

Each process owns `M/P` rows of `A` and needs all of `B`. Per-process volume `≈ K·N·w` — independent of `P`, so it cannot scale. Keep it: it is a 30-line function and it makes the 2D result look like an argument rather than an assertion.

### 6.2 E1 — SUMMA (the workhorse)

```
for step s = 0 .. ceil(K/b)-1:
    kb   = min(b, K - s*b)
    root_c = owner column of global columns [s*b, s*b+kb)      /* A panel */
    root_r = owner row    of global rows    [s*b, s*b+kb)      /* B panel */
    if (my_col == root_c) pack A(:, local slice) -> Abuf
    MPI_Bcast(Abuf, m*kb, T, root_c, row_comm)
    if (my_row == root_r) pack B(local slice, :) -> Bbuf
    MPI_Bcast(Bbuf, kb*n, T, root_r, col_comm)
    kernel_gemm(m, n, kb, alpha, Abuf, kb, Bbuf, n, 1.0, C, ldc)
```
(Scale `C` by `beta` once, before the loop.)

**Cost model.**
```
T_comm = (K/b)·α·(log2 Pr + log2 Pc)  +  β·w·K·(M/Pr + N/Pc)
T_comp = γ · 2·M·N·K / P
```
Note the structure: the **bandwidth term does not depend on `b`**, only the **latency term** does (`∝ 1/b`). So `b` is purely a latency-vs-buffer-memory-vs-overlap-granularity knob — a clean, defensible sweep for the report.

For `M=N=K=n`, `Pr=Pc=√P`: volume `2n²/√P` words per process, `(n/b)·log P` messages.

Why SUMMA is the right core:
- It is the only classical algorithm that is *natively* generic in shape and grid (van de Geijn & Watts wrote it for exactly that reason).
- Broadcast-based ⇒ trivially replaced by `MPI_Ibcast` / persistent collectives (§7).
- Memory overhead is only the two panel buffers: `b·(M/Pr + N/Pc)·w`.

### 6.3 E2 — Cannon (comparison point)

`√P × √P` grid, initial skew of `A` by row index and `B` by column index, then `√P` steps of local multiply + nearest-neighbour shift.

```
T_comm = 2·√P·α + β·w·K·(M+N)/√P        (same volume as SUMMA, no log factor)
```
Same bandwidth cost, fewer messages (`2√P` vs `(K/b)·log P`), but: requires a square grid, requires divisibility (or an ugly generic re-derivation), the skew phase is pure overhead, and overlapping is awkward. Implement it, show it wins slightly on square/divisible cases at high `P` (latency), and lose on everything else. That contrast *is* your justification for choosing SUMMA as the base — which is a much stronger report than "we picked SUMMA".

### 6.4 E3 — 2.5D SUMMA (the first real extension)

Split the `k` dimension into `c` slabs, one per layer; each layer runs a full SUMMA on a `√(P/c) × √(P/c)` grid over its slab; sum the partial `C` across layers with one `MPI_Reduce` on `depth_comm`.

```
for each layer l:  C_partial = A(:, kslab_l) · B(kslab_l, :)      [SUMMA]
MPI_Reduce(C_partial -> C, m*n, T, MPI_SUM, root_layer, depth_comm)
```

**Cost model** (square case, per process):
```
T_comm ≈ β·w·2n²/√(cP)          +  β·w·(c·n²/P)     +  α·[ (K/c)/b·log(P/c) + log c ]
             SUMMA inside layer      depth reduce
```
The bandwidth term shrinks by **√c**; the reduce term grows linearly in `c`. They cross at **c ≈ P^(1/3)** — which recovers the classical 3D/DNS algorithm and matches the known communication lower bounds. That is a beautiful, cheap, *provable* result to put in the report: plot measured time vs `c ∈ {1,2,4,8,...}` next to the model curve and show the optimum lands where theory says.

**Memory cost:** `c` copies of `C` (`c·M·N/P` per process) — that, not `A`/`B`, is what limits `c`. Note in the report that this is the "k-slab" formulation; Solomonik & Demmel's original replicates `A` and `B` across layers instead. The k-slab variant composes with your existing SUMMA engine in ~40 lines and uses strictly less memory, at the cost of one extra initial redistribution. Say so and justify it.

**Floating-point caveat (good report material):** summation order changes with `c` and `b`, so results are not bitwise reproducible across configurations. Quantify it (report `‖C_c − C_1‖_F / ‖C_1‖_F`) instead of hiding it.

### 6.5 E4 — cost-model-driven planner (the originality claim)

This is the component that turns "another SUMMA implementation" into a project with a thesis. Instead of hardcoding `Pr = Pc = √P`, **enumerate all factorisations of `P`** and pick the one minimising the model of §6.4, subject to memory:

```c
best = INF;
for (c = 1; c <= P; c++) if (P % c == 0)
  for (Pr = 1; Pr <= P/c; Pr++) if ((P/c) % Pr == 0) {
      Pc = (P/c)/Pr;
      if (mem_needed(M,N,K,Pr,Pc,c,b) > mem_avail) continue;
      for (b = b_min; b <= b_max; b *= 2) {
          t = alpha*((K/c)/b*(log2(Pr)+log2(Pc)) + log2(c))
            + beta *w*((K/c)*(M/Pr + N/Pc) + (double)c*M*N/P)
            + gamma*2.0*M*N*K/P;
          if (t < best) { best = t; save(Pr,Pc,c,b); }
      }
  }
```

`P` is at most a few thousand, so the enumeration is microseconds. Sanity check against the closed form: ignoring latency and with `c = 1`, minimising `K(M/Pr + N/Pc)` under `Pr·Pc = P` gives **`Pr = √(P·M/N)`** — the grid aspect ratio should match the *output* aspect ratio. Your enumerator must reproduce that; if it does not, the model is wired wrong.

What this buys you experimentally:
- square `M=N=K` → returns ≈ `√P × √P`, `c = 1` (no regression vs the classic choice);
- **tall-skinny** (`M = N = 32768`, `K = 512`, typical of deep-learning and least-squares workloads) → returns a near-square grid with `c = 1` but a very different `b`;
- **short-fat** (`M = N = 1024`, `K = 4·10⁶`, typical of Gram-matrix / covariance computations) → returns large `c`, i.e. it *discovers* k-parallelism, where every fixed-2D implementation collapses.

The headline plot of your report writes itself: **speedup of planner-chosen configuration over fixed `√P × √P`, across five matrix shapes**. Expect anywhere from 1× (square) to several × (skewed). This is a scaled-down, honest reimplementation of the idea behind COSMA — say exactly that, cite it, and be explicit that you implement a cost-model grid selector, *not* COSMA's I/O-optimality proof.

---

## 7. Modern extensions worth adding, and why

Ordered by (value to the report) / (hours of work). Do 7.1 and 7.2 for sure.

### 7.1 Overlap: non-blocking collectives + lookahead

SUMMA's step `s+1` panels do not depend on step `s`'s computation, so:

```
post Ibcast for panel 0 into buf[0]
for s = 0..S-1:
    if (s+1 < S) post Ibcast for panel s+1 into buf[(s+1)%2]
    MPI_Waitall(2, req[s%2]);      /* A and B panels of step s */
    kernel_gemm(... buf[s%2] ...)
```

Two buffers = depth-1 lookahead; the memory cost is one extra panel pair. In the α-β model the ideal result is `T = max(T_comm, T_comp) + T_panel_0` instead of the sum.

**Be honest about the caveat** — this is the discussion paragraph that separates a good report from a great one: MPI makes no progress guarantee outside MPI calls. Without asynchronous progress (an extra progress thread, or hardware-offloaded collectives), `MPI_Ibcast` may do nothing until `MPI_Wait`, and you measure zero overlap. Define and report an **overlap efficiency**
```
η = (T_blocking − T_overlap) / min(T_comm, T_comp)
```
and test with and without `--mca opal_async_progress` style options (check what your Open MPI build offers). A negative or near-zero `η` honestly reported and explained beats a fabricated speedup.

### 7.2 MPI-4 persistent collectives (genuinely modern, rarely seen in coursework)

SUMMA repeats the *same* collective on the *same* buffers `K/b` times. That is the textbook case for **persistent collectives**, standardised in MPI-4.0 and available in Open MPI 5.x (which your slides list as the cluster's MPI):

```c
#if MPI_VERSION >= 4
/* setup once: one handle per (root, buffer) pair */
for (int r = 0; r < Pc; ++r)
  for (int d = 0; d < 2; ++d)
     MPI_Bcast_init(Abuf[d], max_m*b, T, r, row_comm, MPI_INFO_NULL, &reqA[r][d]);
/* per step: */
MPI_Start(&reqA[root_c][s%2]);  ...  MPI_Wait(&reqA[root_c][s%2], MPI_STATUS_IGNORE);
#endif
```

Why it should help: the algorithm selection, tree construction, buffer registration and (on RDMA networks) memory pinning happen once at init instead of `K/b` times. The subtlety worth writing about: in SUMMA the **root rotates**, so you need one persistent handle per root (`Pc` of them per buffer), not one per call site. Keep a compile-time fallback to `MPI_Ibcast` — a three-line `#if` that also demonstrates portability awareness.

Ablation for the report: blocking vs `Ibcast` vs persistent, at fixed everything else. Even a null result is publishable-quality content when the mechanism is explained.

### 7.3 Hybrid MPI + OpenMP

One rank per socket / NUMA domain, `OMP_NUM_THREADS` = cores per socket, instead of one rank per core.

Total network volume over all ranks is `P · 2n²/√P = 2n²√P`, so **reducing the rank count reduces total traffic as `√P`**. Going from 64 ranks to 4 ranks on the same 64 cores cuts aggregate volume by 4×, and panel buffers are shared by all threads in a rank instead of being replicated per core. Combine with `OMP_PROC_BIND=close`, `OMP_PLACES=cores` and `--map-by ppr:1:socket --bind-to socket`.

Experiment: fix total cores (e.g. 4 nodes × 32 cores), sweep ranks × threads ∈ {128×1, 64×2, 32×4, 8×16, 4×32}. This is one of the most informative plots you can produce and it costs no new code beyond `#pragma omp parallel for` in the kernel. It also ties the MPI half of the course back to the OpenMP half.

### 7.4 Type genericity (G3)

C++ is the honest answer (`template<typename T>` + a `mpi_type<T>()` trait, compiled with `mpicxx`, zero overhead). If the course requires C, use the include-template trick rather than `void*` + function pointers, which would destroy the inner loop:

```c
/* gemm_engine.inc — no include guard, included once per type */
void CAT(gemm_summa_, SCALAR_SUFFIX)(...) { ... }

/* engines.c */
#define SCALAR double
#define SCALAR_SUFFIX d
#define MPI_SCALAR MPI_DOUBLE
#include "gemm_engine.inc"
#undef ...
```
Payoff: a single `--dtype=float|double` flag gives you a mixed-precision experiment. `float` halves both the communication volume and the memory footprint and roughly doubles SIMD throughput — expect close to 2× on the bandwidth-bound regime, which is a very clean roofline story.

### 7.5 Optional stretch goals (pick at most one)

| Idea | Payoff | Risk |
|---|---|---|
| One-sided RMA (`MPI_Win`, `MPI_Get`) for the Cannon shifts | ablation of the communication substrate; the course explicitly references the RMA chapter | synchronisation epochs are easy to get subtly wrong |
| `MPI_Neighbor_alltoall` on a Cartesian communicator for Cannon | concise, lets MPI exploit topology | little measurable gain at small `P` |
| MPI-4 partitioned communication (`MPI_Psend_init` / `MPI_Pready`) | each OpenMP thread signals its slice as it finishes packing — fine-grained overlap | newest feature, thinnest implementation support; better as "future work" |
| GPU: cuBLAS local kernel + GPU-aware MPI | huge FLOP numbers | swallows the whole project; do it only if the deadline is far |

---

## 8. Local kernel (L1)

The distributed layer will not save you from a bad inner loop; a naive `ijk` kernel can be 20× off peak, and then your "scaling" plot is really measuring how bad your kernel is.

1. **Loop order.** Row-major ⇒ `i-k-j`: `C[i][j]` and `B[k][j]` are unit-stride in the inner loop, `A[i][k]` is a scalar broadcast. This is literally the `cache_01_slow.c` vs `cache_01_fast.c` lesson from the course, one level up.
2. **Cache blocking.** Tile with `(mc, kc, nc)` sized so that the `B` panel (`kc×nc`) fits in L2 and the `A` block (`mc×kc`) in L1: a good starting point on a 32 KB L1 / 512 KB L2 core is `kc = 256, nc = 512, mc = 64` for doubles — then tune.
3. **Packing.** Copy `A` tiles into a contiguous, aligned scratch buffer before the micro-kernel. You get this for free: SUMMA already packs panels for the broadcast, so the packed layout *is* the communication buffer. Point this out — it is a genuine cross-layer design win.
4. **Vectorisation.** `-O3 -march=native -funroll-loops`; check with `-fopt-info-vec`. Avoid `-ffast-math` or, if you use it, report it — it changes FP semantics and therefore your error numbers.
5. **Threading.** `#pragma omp parallel for collapse(2) schedule(static)` over `(i,j)` tiles; never over `k` (that would need a reduction and creates false sharing on `C`).
6. **Reference upper bound.** Link OpenBLAS/MKL `dgemm` behind a `--kernel=blas` flag and plot it as the ceiling. *Ask your instructor whether an external BLAS is permitted in the deliverable* — most courses want your own kernel but welcome the library as a reference line.

---

## 9. Performance model and calibration

The course distinguishes benchmarking from modelling; do both and put them on the same axes. You need three constants, each from its own micro-benchmark:

| Constant | Meaning | How to measure |
|---|---|---|
| `α` | message latency (s) | ping-pong, 0-byte payload, intercept of the linear fit — the ping-pong exercise is already in your L13 slides |
| `β` | inverse bandwidth (s/byte) | same ping-pong, slope for large messages; measure intra-node and inter-node separately, they differ by ~10× |
| `γ` | achieved s/flop | single-rank local kernel at a size that fits in cache-friendly blocks |

Then:
```
T_model(Pr,Pc,c,b) = α·[(K/c)/b·(log2 Pr + log2 Pc) + log2 c]
                   + β·w·[(K/c)·(M/Pr + N/Pc) + c·M·N/P]
                   + γ·2·M·N·K/P
```

**Roofline, distributed version.** The course's roofline is node-level (FLOP/byte vs DRAM). Extend it: define network arithmetic intensity `I_net = (2MNK/P) / (bytes moved over the network)`. For 2D SUMMA on square matrices this is `≈ n/(2√P)·(1/w)`, i.e. it *falls* as `√P` grows — this single formula explains your strong-scaling curve, and increasing `c` moves you back to the right on the roofline. Plotting your configurations as points on a network roofline is an unusual and very effective figure.

Also report:
- **GFLOP/s** = `2·M·N·K / (T · 10⁹)` and % of node peak,
- **speedup / efficiency** vs a well-defined baseline (best single-node run, *not* your own naive code — inflating speedup with a slow baseline is the classic sin the benchmarking lecture warns about),
- **Amdahl/Gustafson framing** when interpreting strong vs weak scaling.

---

## 10. Correctness and validation

**Layer 1 — small exact tests.** `P ∈ {1,2,3,5,7}` (prime `P` is the real generic-grid test), `M,N,K` deliberately non-divisible (e.g. 101×97×103), against a sequential reference gathered on rank 0. Also: identity matrices, `β ≠ 0`, `α ≠ 1`, zero-`K` edge case.

**Layer 2 — norm-wise check.** `‖C − C_ref‖_F / (‖A‖_F·‖B‖_F) ≤ c·K·ε`. Never test floating-point equality.

**Layer 3 — Freivalds at scale.** At `n = 32768` you cannot afford an `O(n³)` reference. Freivalds' randomised check costs `O(n²)`: draw a random vector `r`, verify `A(Br) ≈ Cr`.

Distributed recipe on your existing layout (vectors are `O(n)` words = a few hundred KB, so replicate them — no clever redistribution needed):
```
1. every process forms the same r (same seed) — no communication
2. z_loc = B_loc · r_loc                       (local, over its own columns)
3. MPI_Allreduce(z_loc, row_comm) then Allgather over col_comm -> full z
4. w_loc = A_loc · z_loc ; MPI_Allreduce over row_comm          -> w
5. u_loc = C_loc · r_loc ; MPI_Allreduce over row_comm          -> u
6. accept if ||w - u|| <= tol * ||w||;  repeat with k independent r
```
Failure probability `≤ 2^-k` over a finite field; with floating point it is a strong numerical smoke test rather than a proof — say so. Ship it behind `--verify=freivalds` and use it in every large run. This is a small piece of engineering that reviewers notice.

**Layer 4 — cross-configuration consistency.** Same problem, different `(Pr,Pc,c,b,engine)`; report the relative difference. Non-zero and non-alarming *is* the expected answer, and explaining why is a paragraph of real content.

---

## 11. Experimental plan

Baselines to hold constant: same node type, same compiler + flags, same MPI build. State them all.

| # | Experiment | Independent variable | What it demonstrates |
|---|---|---|---|
| E1 | Kernel micro-benchmark | naive / blocked / +OpenMP / BLAS | the local ceiling `γ` |
| E2 | Ping-pong | message size, intra vs inter node | `α`, `β` |
| E3 | Strong scaling | `P` = 1,2,4,…,N nodes at fixed `n` | speedup, efficiency, where it breaks |
| E4 | Weak scaling | `n` grown so that **memory per process** (`n²/P`) is constant — *state the function explicitly* | Gustafson-style scalability |
| E5 | Engine comparison | naive1D / Cannon / SUMMA / 2.5D | justifies the architectural choice |
| E6 | `c` sweep (2.5D) | `c` = 1,2,4,8 | validates the `c ≈ P^(1/3)` prediction |
| E7 | `b` sweep | `b` = 32…1024 | latency-vs-granularity trade-off |
| E8 | **Shape sweep + planner** | square / tall-skinny / short-fat × {fixed grid, planner} | **the originality claim** |
| E9 | Comm policy | blocking / Ibcast / persistent | overlap efficiency `η` |
| E10 | Hybrid mapping | ranks × threads at fixed core count | traffic reduction from fewer ranks |

**Statistics, per the benchmarking lecture (Hoefler & Belli):** ≥10 repetitions, drop the first (warm-up), `MPI_Barrier` before the timer, `MPI_Wtime` on each rank and take the **max** via `MPI_Reduce(MPI_MAX)` (the slowest rank defines the runtime), report **median with 95% confidence interval** (non-parametric — do not assume normality), use the **harmonic mean** for rates (GFLOP/s) and arithmetic only for times, never remove outliers. Interleave configurations across the queue rather than running each config back-to-back, so cluster drift does not correlate with your independent variable.

Emit one CSV row per run (`engine,dtype,M,N,K,P,Pr,Pc,c,b,threads,policy,rep,time_s,gflops,err`), and do all plotting from that CSV with a script in `scripts/`. This makes the whole result section regenerable with one command — which is what the reproducibility checklist is asking for.

---

## 12. Repository, build, and jobs

Follow the structure the course prescribes, extended slightly:

```
repo/
├── README.md            build, run, reproduce; compiler versions; how to change sizes
├── Makefile             (or CMakeLists.txt)
├── include/             pgrid.h dmat.h engines.h kernel.h planner.h
├── src/                 main.c pgrid.c dmat.c summa.c summa25d.c cannon.c
│                        naive1d.c kernel.c planner.c verify.c timing.c
├── tests/               test_small.sh (prime P, non-divisible sizes), test_freivalds.sh
├── jobs/                strong.pbs weak.pbs shapes.pbs hybrid.pbs
├── scripts/             run_sweep.sh  plot_*.py  collect.py
├── results/             raw CSVs (committed — they are your evidence)
└── plots/               figures used in the report
```

Example PBS job (adapt queue/module names — run `module avail` first, the MPI module name on your cluster is not the same as the gcc one):

```bash
#!/bin/bash
#PBS -N gemm2d_strong
#PBS -o results/strong.out
#PBS -e results/strong.err
#PBS -q short_cpuQ
#PBS -l walltime=1:00:00
#PBS -l select=4:ncpus=32:mpiprocs=32:mem=64gb

module load gcc91
module load <your-mpi-module>

cd $PBS_O_WORKDIR
make clean && make OPT="-O3 -march=native -fopenmp"

export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores

for NP in 1 2 4 8 16 32 64 128; do
  for REP in $(seq 1 10); do
    mpiexec -n $NP --map-by core --bind-to core \
      ./gemm2d --M 16384 --N 16384 --K 16384 \
               --engine summa --b 256 --verify freivalds \
               --csv results/strong.csv --rep $REP
  done
done
```

README must contain, per the course checklist: exact compiler and MPI versions, build command with flags, a runnable example for both laptop and cluster, the meaning of every CLI flag and its default, and which file/flag to change to alter problem size or process count. Then have a classmate clone it and reproduce one figure — that step catches 90% of reproducibility bugs.

---

## 13. Milestones

| Week | Goal | Exit criterion (go/no-go) |
|---|---|---|
| 1 | Skeleton, `pgrid`, `dmat`, naive 1D, sequential reference, CLI+CSV | correct on `P ∈ {1,2,3,5,7}` with non-divisible sizes |
| 2 | SUMMA + blocked local kernel | matches reference; ≥40% of BLAS single-node |
| 3 | Cannon; ping-pong calibration; cost model coded | model within ~30% of measured on ≥3 configs |
| 4 | 2.5D + depth reduce; Freivalds verification | `c` sweep reproduces the predicted optimum |
| 5 | Planner + shape sweep (**the result**) | planner ≥ fixed grid on every shape, ≫ on skewed |
| 6 | Ibcast/persistent policies; hybrid sweep | overlap efficiency measured and explained |
| 7 | Full benchmark campaign, plots | all CSVs in `results/`, all figures regenerable |
| 8 | Report + slides | 4 pages, ≤5 figures, references complete |

If you fall behind, cut in this order: 7.5 stretch goals → §7.2 persistent → E2 Cannon → G3 types. **Never** cut the planner (§6.5) or the verification (§10): they are, respectively, your contribution and your credibility.

---

## 14. Mapping onto the 4-page IEEE report

| Section | ~Words | Content | Course checkpoint |
|---|---|---|---|
| Abstract | <200 | problem, method (2D + 2.5D + planner), headline number | required |
| Introduction | 400 | why distributed GEMM; why generic shapes matter (DL, least squares, Gram matrices); state of the art: Cannon → SUMMA → 2.5D → COSMA; **the gap you address**: implementations that fix `√P×√P` regardless of shape | "state of the art + gap" is explicitly required |
| Methodology | 900 | data layout + descriptor, communicator hierarchy, **pseudo-code for SUMMA and 2.5D**, cost model, planner algorithm, verification strategy | pseudo-code is explicitly required |
| Experiments | 400 | cluster spec (CPU model, cores, memory, network), compiler + MPI versions and flags, `α/β/γ` calibration, statistical protocol | required |
| Results | 900 | Fig.1 strong+weak scaling; Fig.2 engine comparison; Fig.3 `c` sweep vs model; Fig.4 **planner vs fixed grid across shapes**; Fig.5 network roofline or comm-policy ablation | "compare to prior work", units, captions |
| Conclusions | 200 | takeaways, limitations (no GPU, no sparse, FP non-reproducibility), future work (COSMA-style optimality, partitioned comms, GPU) | required |

Four pages allows about **five** figures. Kill any experiment that does not earn one — but keep its data in `results/` and mention it in one sentence.

**Presentation (10–15 min):** 1 motivation, 1 the generic problem + why fixed grids fail, 2 architecture (the layer diagram + the SUMMA step picture), 1 cost model, 4–5 results, 1 limitations/future. Expect Q&A on: why not block-cyclic; why 2.5D costs memory; whether your overlap actually overlapped; how you know the result is correct at `n = 32768`.

---

## 15. Literature to cite

> **Verification note:** I could not run a live web search in this session, so these references come from my own knowledge (reliable to roughly May 2026). Check every DOI, year, and venue on ACM DL / IEEE Xplore / DBLP before submitting — and search for anything from the last year, especially on communication–computation overlap, where the field is moving fast.

**Foundational (cite in the introduction as the lineage):**
- L. E. Cannon, *A cellular computer to implement the Kalman filter algorithm*, PhD thesis, Montana State University, 1969.
- E. Dekel, D. Nassimi, S. Sahni, "Parallel matrix and graph algorithms", *SIAM J. Computing* 10(4), 1981 — the DNS 3D algorithm.
- R. van de Geijn, J. Watts, "SUMMA: Scalable Universal Matrix Multiplication Algorithm", *Concurrency: Practice and Experience* 9(4), 1997 — **your base algorithm; cite it for the genericity argument**.
- J. Choi, D. Walker, J. Dongarra, "PUMMA", *Concurrency* 6(7), 1994; and the ScaLAPACK users' guide, 1997 — for the block-cyclic layout you deliberately do not use.
- R. Agarwal, S. Balle, F. Gustavson, M. Joshi, P. Palkar, "A three-dimensional approach to parallel matrix multiplication", *IBM J. Res. Dev.* 39(5), 1995.

**Lower bounds (cite to justify that 2.5D is optimal, not just clever):**
- D. Irony, S. Toledo, A. Tiskin, "Communication lower bounds for distributed-memory matrix multiplication", *JPDC* 64(9), 2004.
- G. Ballard, J. Demmel, O. Holtz, O. Schwartz, "Minimizing communication in numerical linear algebra", *SIAM J. Matrix Anal. Appl.* 32(3), 2011.
- H. Al Daas, G. Ballard, L. Grigori, S. Kumar, K. Rouse, "Tight memory-independent parallel matrix multiplication communication lower bounds", *SPAA*, 2022 — the modern statement covering rectangular shapes; directly relevant to your generic-shape claim.

**The algorithms you actually extend:**
- E. Solomonik, J. Demmel, "Communication-optimal parallel 2.5D matrix multiplication and LU factorization algorithms", *Euro-Par*, 2011 — **the source for §6.4**.
- J. Demmel et al., "Communication-optimal parallel recursive rectangular matrix multiplication" (CARMA), *IPDPS*, 2013 — the shape-oblivious recursive alternative; cite as the design you did *not* take, and say why (recursion + irregular process subsets vs. a flat cost model you can calibrate).
- G. Kwasniewski, M. Kabić, M. Besta, J. VandeVondele, R. Solcà, T. Hoefler, "Red-blue pebbling revisited: near optimal parallel matrix-matrix multiplication" (COSMA), *SC*, 2019 — **the direct inspiration for the planner in §6.5**; be explicit that you implement a simplified cost-model selector, not their I/O-optimal decomposition.
- J. Poulson et al., "Elemental: A new framework for distributed memory dense matrix computations", *ACM TOMS* 39(2), 2013; M. Gates et al., "SLATE: design of a modern distributed and accelerated linear algebra library", *SC*, 2019 — modern library context, and a natural "compare against a real library" reference.

**Runtime / overlap (the modern layer, §7):**
- MPI Forum, *MPI: A Message-Passing Interface Standard, Version 4.0*, 2021 — persistent collectives (§7.2) and partitioned communication (§7.5).
- A. Jangda et al., "Breaking the computation and communication abstraction barrier in distributed machine learning workloads" (CoCoNet), *ASPLOS*, 2022.
- S. Wang et al., "Overlap communication with dependent computation via decomposition in large deep learning models", *ASPLOS*, 2023 — the modern motivation for fine-grained overlap of exactly this GEMM pattern.
- FLUX / kernel-fusion-based comm-overlap work, 2024 — verify the exact citation; useful as "future work" framing.

**Methodology (already on your slides — cite them, it shows you read them):**
- T. Hoefler, R. Belli, "Scientific benchmarking of parallel computing systems: twelve ways to tell the masses when reporting performance results", *SC*, 2015.
- S. Williams, A. Waterman, D. Patterson, "Roofline: an insightful visual performance model for multicore architectures", *CACM* 52(4), 2009.

---

## 16. Risks and how to defuse them

| Risk | Mitigation |
|---|---|
| Non-divisible sizes break everything late | build `dmat_t` with uneven blocks from day 1; test with primes immediately |
| Local kernel too slow ⇒ compute dominates ⇒ scaling looks perfect but means nothing | measure `γ` in week 2; report % of BLAS peak alongside every scaling plot |
| 2.5D runs out of memory | planner enforces the memory constraint; report `c_max` as a function of `n` and `P` |
| No measurable overlap from `Ibcast` | expected; report `η` and explain MPI progress semantics — this is content, not failure |
| Queue variance swamps the effect you measure | ≥10 reps, interleave configs, report CIs, note node IDs |
| Persistent collectives unsupported by the cluster MPI | `#if MPI_VERSION >= 4` fallback, decided at compile time |
| Scope creep (GPU, sparse, Strassen) | the cut list in §13 |

---

## 17. Two things to ask your instructor before week 1

1. **Is linking an external BLAS (OpenBLAS/MKL) allowed** as the local kernel, as a reference line, or not at all? This changes §8 substantially.
2. **Is "generic" meant as arbitrary shapes/sizes, arbitrary data types, or both?** The architecture above covers both, but knowing where the grading weight sits tells you whether to spend the week-6 slot on G3 or on the overlap experiments.
