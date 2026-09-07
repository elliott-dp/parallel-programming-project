# Design notes

Why each choice was made, and where it goes in the 4-page report.

---

## 1. The headline technique: intra-node panel deduplication

**The observation.** In SUMMA, every rank of a process row receives the *same* A-panel and every
rank of a process column receives the *same* B-panel. With `q` ranks per node inside that
communicator, the identical bytes are pulled across the network onto that node `q` times, and `q`
copies of the panel sit in that node's memory.

**The mechanism** (`src/chan.c`). The panel lives in an MPI-3 shared-memory window
(`MPI_Win_allocate_shared`) owned jointly by the ranks of the communicator that share a node. One
leader per node takes part in an inter-node `MPI_Bcast` over a leaders-only communicator; every
other rank reads the panel directly out of shared memory.

```
blocking:   root ──►  every rank of the row      (q copies per node cross the network)
shm:        root ──►  one leader per node ──► shared memory ──► the node's ranks
```

Off-node panel traffic falls by a factor of `q`; the per-node panel footprint falls from `q` copies
to one. Cost: two intra-node barriers per panel (three when there is more than one node).

**The bug that had to be fixed to make it correct.** With a private receive buffer per rank, a rank
that finishes its local update early can immediately start the next step — there is nothing shared
to corrupt. Once the buffer is shared, a fast rank racing ahead will overwrite the panel while a
slower rank on the same node is still multiplying with it. `chan_begin()` adds the "all readers are
done" barrier before the root writes. This is the interesting engineering content of the technique
and it belongs in the methodology section: **making the panel shared converts a private buffer into
a critical section.**

**Why it does not compose with lookahead.** The same barriers that make buffer reuse safe also
synchronise the ranks, which is exactly what overlap is trying to avoid. `--bcast shm` therefore
runs with depth-0 lookahead. Reporting this trade-off — bytes saved versus synchronisation added —
is worth more than reporting a speedup.

**Supporting flag: `--gridmap nodeaware`** (`src/pgrid.c`). The technique only helps when a process
row or column actually has several ranks on one node. The default linear rank order makes a row
intra-node and a column inter-node (or vice versa) depending on `Pr:Pc` and the ranks-per-node.
Node-aware mapping tiles each node with a `pr_n × pc_n` sub-block whose aspect ratio matches
`Pr:Pc`, so both rows *and* columns keep several ranks per node. It falls back to linear when the
grid is not divisible by the tile, and says so.

**Literature.** MPI+MPI shared-memory windows: Hoefler et al., *Computing* 95 (2013). NUMA-aware
shared-memory broadcast: Kurnosov, LNCS (2020) — 20–60% over Open MPI `coll/sm`. The cache-coherence
counter-result on multi-socket Xeon: Georgakoudis et al., ICPP-W 2023 (DOI 10.1145/3605573.3605616)
— cite it, and check for the degradation on your own nodes. Node-aware process grids: Irmler et al.,
Euro-Par 2023 (DOI 10.1007/978-3-031-39698-4_48). Two-level broadcast for SUMMA specifically:
Hasanov & Lastovetsky, *J. Supercomputing* 71 (2015).

---

## 2. Why SUMMA is the base engine

`src/engines.c: summa_schedule()` is the whole genericity argument in 20 lines. A's `k`-dimension is
partitioned over `Pc` and B's over `Pr`; those two partitions are *different*, so a panel of width
`b` may straddle an owner boundary. Each step is clipped to the intersection of both owners' ranges:

```c
end = min(k0 + b, end_of_A_owner, end_of_B_owner, Klen)
```

That single line is why arbitrary `M, N, K, Pr, Pc` work with no divisibility assumption, and it is
what Cannon cannot do without being rewritten. Keeping Cannon in the repo — and having it *refuse*
non-square grids rather than pad — turns "we chose SUMMA" into a measured argument.

Cost model (`src/planner.c`):

```
T_comm = alpha·[ceil(K/c/b)·(log2 Pr + log2 Pc) + log2 c]
       + beta·w·[(K/c)·(M/Pr + N/Pc) + c·M·N/P]
T_comp = gamma·2·M·N·K/P
```

The bandwidth term does not contain `b`; only the latency term does. So `b` is purely a
latency-versus-buffer-memory knob, which makes the `--b` sweep clean to interpret.

---

## 3. 2.5D in the k-slab formulation

Layer `L` owns the global `k`-range `[blk_off(K,c,L), +blk_size(K,c,L))`, runs a complete SUMMA on
it, and the partial `C` blocks are summed with one `MPI_Reduce` over the depth communicator.

The bandwidth term shrinks as `√c`, the reduce term grows as `c`; they cross at `c ≈ P^(1/3)`. The
planner reproduces this without being told: at `P = 64` with square matrices it returns
`4 × 4 × 4`, and `64^(1/3) = 4`.

This is the k-slab variant, not Solomonik & Demmel's original, which replicates `A` and `B` across
layers. The k-slab version composes with the existing SUMMA engine in ~40 lines and its only extra
memory is `c` copies of `C`. State the difference in the report — deviating from a paper *and
saying why* is worth more than following it.

Correctness detail worth a sentence in the methodology: `beta*C` is applied on **layer 0 only**, and
the other layers start at zero, so the depth reduce yields `alpha*A*B + beta*C` exactly once.

---

## 4. The planner

`plan_choose()` enumerates every factorisation `Pr·Pc·c = P` and every panel width in
`{32…1024}`, rejects anything over the memory budget, and returns the minimum of the cost model.
`P` is at most a few thousand, so the search takes microseconds.

Sanity check that must hold: ignoring latency with `c = 1`, minimising `K(M/Pr + N/Pc)` under
`Pr·Pc = P` gives `Pr = √(P·M/N)` — the grid aspect ratio should match the output aspect ratio. The
enumerator reproduces it (`--plan-report --plan-P 64 --M 65536 --N 1024 --K 1024` → `64 × 1`).

What it does across shapes at `P = 64` (from `--plan-report`, defaults):

| shape | `M, N, K` | chosen |
|---|---|---|
| square | 16384³ | `4 × 4 × 4` |
| tall-skinny | 32768, 32768, 512 | `8 × 8 × 1` |
| short-fat | 1024, 1024, 4·10⁶ | `1 × 1 × 64` |
| tall A | 65536, 1024, 1024 | `64 × 1 × 1` |

The short-fat row is the originality claim: the planner *discovers* k-parallelism where any fixed
2D implementation would keep a square grid and communicate far more than necessary.

`--calibrate` measures `alpha` and `beta` from a ping-pong between the first and last rank and
`gamma` from a cache-resident local kernel run, so the model uses this machine's constants rather
than guesses. Uncalibrated versus calibrated is itself an ablation.

**Literature.** COSMA: Kwasniewski et al., SC'19 (DOI 10.1145/3295500.3356181) — say explicitly that
this is a simplified cost-model selector, not their I/O-optimal decomposition. 2.5D: Solomonik &
Demmel, Euro-Par 2011. Lower bounds: Irony/Toledo/Tiskin, JPDC 2004; Al Daas et al., SPAA 2022.

---

## 5. Verification

Three layers, because a fast wrong answer is worthless:

1. `--verify ref` — exact, against a sequential reference from the same deterministic generator.
   `O(M·N·K)` on one rank, capped at `M·N ≤ 4·10⁶`. This is what the 36-case suite uses.
2. `--verify freivalds` — randomised, `O(n²)`, usable at full scale. Draw `r`, check
   `A(Br) ≈ Cr`. Vectors are `O(n)` words so they are simply replicated instead of redistributed.
   Handles `beta ≠ 0` by pre-computing `C₀·r` for each trial before the GEMM.
3. Cross-configuration consistency — same problem, different `(engine, c, b, bcast)`. The residual
   is non-zero and that is expected; explaining why is a paragraph of real content.

The deterministic index-based generator (`gen_elem`) is what makes all of this cheap: every rank
builds its own block with no communication, and the same global matrix is produced for any `P` and
any grid, so results are comparable across configurations.

---

## 6. Mapping to the report

| report section | code / artefact |
|---|---|
| Methodology — data layout | `dmat_t` descriptor, balanced uneven blocks, `blk_size/blk_off/blk_owner` |
| Methodology — pseudo-code | `summa_schedule()` + the SUMMA loop in `gemm_summa()` |
| Methodology — the technique | `chan.c`, the three-policy interface, `chan_begin()` and the race it prevents |
| Methodology — cost model | `plan_cost()`, `plan_choose()` |
| Experiments — setup | `--calibrate` output (`alpha, beta, gamma`), compiler/MPI versions from README |
| Results — Fig. 1 | `jobs/strong.pbs` → `--kind strong` |
| Results — Fig. 2 | engine comparison (`naive1d`/`cannon`/`summa`/`summa25d`) |
| Results — Fig. 3 | `jobs/csweep.pbs` → `--kind csweep`, against the `c = P^(1/3)` prediction |
| Results — Fig. 4 | `jobs/shapes.pbs` → `--kind shapes` — **the headline** |
| Results — Fig. 5 | `jobs/shm_ablation.pbs` → `--kind ablation` — time and off-node bytes |
| Conclusions — limitations | the honest notes at the end of the README |

Four pages allows about five figures. Anything that does not earn one stays in `results/` and gets a
single sentence.
