# Project status vs. the architecture guide

Assessment of `gemm2d.tar.gz` against `distributed-gemm-2d-architecture-guide.md`.
Date: 2026-09-07.

## How this was checked

The tarball was extracted and built from source, then exercised. Everything marked
"verified" below was actually run, not read:

* `make` — clean build with `mpicc -std=c11 -O3 -march=native -funroll-loops -fopenmp`,
  one warning (unused parameter `K` in `verify.c:99`).
* `tests/run_tests.sh` — **36/36 pass**.
* All four `--plan-report` cases quoted in `docs/DESIGN.md` §4 reproduce exactly.
* Extra probes not in the suite: `K=0`, Cannon with `beta != 0`, Cannon on non-divisible
  sizes (correctly refuses, `rc=-2`), `--plan` and `--plan --calibrate` end to end,
  cross-configuration consistency over `{b=16,128} x {c=1,2,4}` (residuals 7e-16 … 9e-16).

Environment caveat: a 4-core Xeon @2.8 GHz container with Open MPI 4.1.6, single node,
oversubscribed. Correctness results carry over; absolute timings and anything about
multi-node behaviour or OpenMP scaling do not.

## Verdict

**The code is at roughly week 5–6 of the 8-week plan. The evidence is at week 0.**

Every algorithmic component the guide calls mandatory exists and is correct: the layered
architecture, the `dmat_t` descriptor, uneven blocks, all four engines, the cost-model
planner, and all three verification layers. What is missing is not code — it is data. There
are no CSVs, no plots, no report, and six of the ten planned experiments have no path to a
figure. The guide's §13 exit criteria for weeks 2, 3, 4, 5 and 6 are all "code written,
criterion never checked".

## Requirement-by-requirement

| Guide | Requirement | Status |
|---|---|---|
| §1 G1 | arbitrary `M,N,K`, `alpha`/`beta` | **done** — verified at 61x53x47 for P=1..8 |
| §1 G1 | optional transposes | not implemented (guide marks optional) |
| §1 G2 | arbitrary `P` incl. primes, non-divisible dims | **done** — verified P=2,3,5,7; also P>rows |
| §1 G3 | `float` / `double` / `int32` one code path | **not implemented** — `typedef double scalar_t` fixed at compile time; no `--dtype`, no X-macro, no MPI type trait |
| §3 | L0–L5 layering | **done** — file layout matches the guide's sketch one-to-one |
| §3 | `dmat_t` descriptor (global dims + local + offsets) | **done** |
| §4.1 | plain 2D block, not block-cyclic | **done** |
| §4.2 | balanced uneven blocks, no padding | **done** — `blk_size`/`blk_off`/`blk_owner` |
| §4.3 | row-major + explicit `ld`, 64-byte aligned | **partial** — `aligned_alloc(64,…)` yes, but `ld == n` exactly; never rounded up to a SIMD multiple |
| §4.4 | index-based deterministic PRNG, no scatter from rank 0 | **done** — `gen_elem()`, splitmix-style; this is done well |
| §5 | row / col / depth communicators via `Comm_split` | **done** |
| §5 | rank-to-node mapping as a measured ablation | **done** — `--gridmap nodeaware`, plus `GEMM2D_FAKE_PPN` to test it on one box |
| §6.1 | E0 naive 1D baseline | **done** |
| §6.2 | E1 SUMMA | **done** — `summa_schedule()` clips each panel to the intersection of A's and B's owner ranges; this is the genuinely generic part |
| §6.3 | E2 Cannon | **done** — refuses non-square/non-divisible by design |
| §6.4 | E3 2.5D (k-slab) + depth reduce | **done** |
| §6.5 | E4 cost-model planner | **done** — and it reproduces the guide's analytic sanity check: `M=65536,N=1024,K=1024,P=64` → `64x1`, matching `Pr = sqrt(P*M/N)` |
| §7.1 | `MPI_Ibcast` + depth-1 lookahead | **done** |
| §7.1 | overlap efficiency `eta` reported | **not implemented** — `eta` appears nowhere in the C or the plot script |
| §7.2 | MPI-4 persistent collectives | **not implemented** — no `MPI_Bcast_init`, no `MPI_Start`, no `#if MPI_VERSION >= 4` anywhere. Substituted by `BC_SHM` (see below) |
| §7.3 | hybrid MPI+OpenMP | **partial** — kernel is threaded correctly; the ranks x threads sweep (E10) does not exist |
| §8 | i-k-j order, cache blocking, thread over i only | **done** — `MC/KC/NC` compile-time tunable, no reduction on C, no false sharing |
| §8.3 | packing into a contiguous scratch buffer | **not implemented** — no micro-kernel, no register blocking |
| §8.6 | BLAS reference line (`--kernel=blas`) | not implemented; §17 Q1 to the instructor is still unanswered |
| §9 | `alpha`/`beta`/`gamma` calibration | **done** — `--calibrate`; but the constants only go to stderr, never to a CSV |
| §9 | measured-vs-model on the same axes | **not implemented** — no plot kind, no data |
| §9 | % of node peak; network roofline / `I_net` | **not implemented** |
| §10 L1 | small exact tests, prime P, non-divisible | **done** — 36 cases |
| §10 L2 | norm-wise check, never FP equality | **done** |
| §10 L3 | distributed Freivalds | **done** — follows the guide's 6-step recipe, handles `beta != 0` via pre-computed `C0*r` |
| §10 L4 | cross-configuration consistency | **possible but not recorded** — see defect 1 |
| §11 | max-over-ranks timing, barrier before timer, raw rows not averages | **done** |
| §11 | median + non-parametric CI, harmonic mean for rates | **done** — in `scripts/plot_results.py` |
| §11 | interleave configurations across the queue | **not done in 3 of 4 job scripts** — see defect 3 |
| §12 | git repository with the prescribed tree | **not done** — see blocker 1 |
| §12 | `results/` CSVs committed as evidence | **not done, and actively prevented** — see blocker 2 |
| §14 | 4-page IEEE report | not started |
| §14 | 10–15 min presentation | not started |

## The one deliberate substitution

The guide's §7.2 (persistent collectives) was dropped and replaced by something not in the
guide at all: `BC_SHM`, a node-aware two-level broadcast built on an MPI-3 shared-memory
window, where one leader per node joins the inter-node broadcast and the node's other ranks
read the panel out of shared memory.

This is a defensible trade and arguably a stronger contribution — `docs/DESIGN.md` §1
documents it properly, including the race it introduces (`chan_begin()`: making the panel
shared converts a private buffer into a critical section) and why it cannot compose with
lookahead. Verified here: at 8 ranks arranged as 2 nodes x 4 with `--gridmap nodeaware`,
modelled off-node bytes halve (0.004 → 0.002 GB), exactly as the mechanism predicts.

Two consequences to handle rather than ignore:

* Nothing in the repo says persistent collectives were considered and dropped. The report's
  related-work / methodology needs one sentence, or a reader who knows MPI-4 will ask.
* §16's risk row "persistent collectives unsupported by the cluster MPI → `#if MPI_VERSION >= 4`
  fallback" is now moot, and the E9 ablation is `blocking / ibcast / shm` rather than the
  guide's `blocking / ibcast / persistent`.

## Blockers

1. **The repository is a tarball.** `git ls-files` returns exactly two paths: the guide and
   `gemm2d.tar.gz`. §12 asks for a git repository with the source tree; there is no history,
   no diffs, and nothing reviewable per file. The tarball also ships `src/*.o` and the
   `gemm2d` binary, which the project's own `.gitignore` excludes. Extracting and committing
   the tree is the single highest-value action available.
2. **`.gitignore` forbids the evidence.** It contains `results/*.csv` and `plots/*.png`,
   which directly contradicts §12 ("raw CSVs — committed, they are your evidence") and the
   project's own README ("`results/ plots/` evidence for the report"). Both directories
   currently hold only `.gitkeep`.
3. **No experimental data at all.** Zero CSVs. Every §13 exit criterion from week 2 onward is
   unverified, including the two the guide says must never be cut.

## Experiment coverage (§11)

| # | Experiment | Job script | Plot kind | Data |
|---|---|---|---|---|
| E1 | kernel micro-benchmark | — | — | — |
| E2 | ping-pong `alpha`,`beta` | — | — | — (measured inside `--calibrate`, stderr only) |
| E3 | strong scaling | `strong.pbs` | `strong` | — |
| E4 | **weak scaling** | **missing** | **missing** | — |
| E5 | engine comparison | missing | missing | — |
| E6 | `c` sweep | `csweep.pbs` | `csweep` | — |
| E7 | `b` sweep | missing | missing | — |
| E8 | **shape sweep + planner** (the headline) | `shapes.pbs` | `shapes` | — |
| E9 | comm policy | `shm_ablation.pbs` | `ablation` | — |
| E10 | hybrid ranks x threads | missing | missing | — |

Four of ten have a route to a figure. §14 budgets about five figures, so E4, E5, E7 and E10
do not all need one — but E4 (weak scaling) is named in §14's Fig. 1 alongside strong
scaling, and E10 is called "one of the most informative plots you can produce" for zero new
code. Those two are worth adding; E5 and E7 need only a job script since the flags exist.

## Defects found

1. **The CSV `err` column is always `-1`** for `summa`, `cannon` and `summa25d`.
   `main.c:80` initialises `err = -1.0`; verification runs at `main.c:318`, *after* the
   timed loop that writes the CSV rows at `main.c:306`. The value is never written back.
   Verified: a run printing `verify(freivalds): relative error = 7.405e-16` emitted two CSV
   rows both carrying `-1.000e+00`. `naive1d` fills it correctly, which makes the gap easy
   to miss. This matters because `docs/DESIGN.md` §5 maps the cross-configuration
   consistency check (§10 layer 4) onto the CSVs — that data is not in them.
2. **`naive1d` silently ignores `--beta`.** `main.c:149` forces `o.beta = 0.0`, and the
   reference check inside `gemm_naive1d` also omits the `beta*C` term, so a `--beta 2.0`
   run reports `relative error = 0.000e+00` while computing `alpha*A*B`. Verified. Low
   impact — it is the throwaway baseline — but it is a verifier that cannot fail.
3. **Three of four job scripts run configurations back-to-back**, which §11 explicitly warns
   against ("interleave configurations across the queue rather than running each config
   back-to-back, so cluster drift does not correlate with your independent variable").
   `strong.pbs` loops `for NP … for REP`, `csweep.pbs` `for C … for REP`, `shm_ablation.pbs`
   `for MAP/POLICY … for REP`. Only `shapes.pbs` interleaves. The fix is to hoist `REP` to
   the outer loop in all three.
4. **`strong.pbs` is likely to exceed its walltime.** It runs `n = 8192` (2·8192³ ≈ 1.1 TFLOP)
   at `NP=1` for 10 reps plus warm-ups, inside `#PBS -l walltime=1:00:00`. At the ~8.4 GFLOP/s
   single-thread rate measured here that is over 2 min per rep before `NP=2,4,8` even start.
   Either drop `NP=1,2` from the strong-scaling sweep (and state the baseline explicitly, per
   §9: the baseline should be the best single-node run) or raise the walltime.
5. **`wire_gb` is a model, not an instrument, and the two branches use different rules.**
   `chan.c:account()` charges the full panel to *every* off-node receiver under
   `blocking`/`ibcast`, but only to off-node *leaders* under `shm`. The README calls it
   "computed the same way for every policy", which is not quite true, and a real Open MPI
   broadcast is often already node-aware — so the blocking baseline is probably over-charged
   and Fig. 5's second panel is guaranteed its result by construction. Either add a caveat in
   the caption or validate against a real counter for one configuration.
6. **No test covers `--plan` or `--calibrate`.** Both work — verified end to end at
   `2.295e-16` (plan) and `8.308e-16` (plan+calibrate) — but the planner is the originality
   claim and it has zero regression coverage. Two lines in `run_tests.sh`.
7. Cosmetic: `plot_results.py:kind_strong` computes `sp` three times and discards two
   (lines 80–82) and assigns `base` unused (line 78); `verify.c:99` has an unused parameter.

## The kernel is the risk the guide flagged

§16 lists "local kernel too slow ⇒ compute dominates ⇒ scaling looks perfect but means
nothing" as a named risk, and it is the one that is currently live.

Measured here: **~8.4–8.7 GFLOP/s single-threaded** on an AVX-512-capable 2.8 GHz core, i.e.
roughly 10–20% of that core's FMA peak. The week-2 exit criterion is ≥40% of single-node
BLAS, and there is no BLAS reference in-tree to measure against. Contributing causes visible
in `kernel.c`: no packing and no register-blocked micro-kernel (§8.3 is described in
`DESIGN.md` but not implemented), and a data-dependent `if (a == 0.0) continue;` branch
inside the innermost `k` loop that will inhibit vectorisation and unrolling.

This is not just a performance number — it distorts the headline result. With `gamma` that
large, the planner's own model has `T_comp` dominating `T_comm` by 100–600x in all four of
the shape cases quoted in `DESIGN.md` (square 16384³ at P=64: 27.488 s compute vs 0.040 s
communication). The planner is then discriminating between configurations on under 0.2% of
modelled runtime, and the E8 "planner vs fixed grid" speedup — the originality claim — will
be compressed toward 1x, not because the planner is wrong but because a slow kernel hides
the communication it is optimising. **Fixing the kernel is a prerequisite for E8 to show
anything.**

Separately, no OpenMP speedup was observable here (8.37 / 9.64 / 8.53 GFLOP/s at 1/2/4
threads), but this container is too noisy and too small to conclude from — that measurement
has to be redone on the cluster with proper binding.

## Milestones (§13)

| Week | Exit criterion | Status |
|---|---|---|
| 1 | correct on `P in {1,2,3,5,7}`, non-divisible sizes | **met** — verified |
| 2 | matches reference; ≥40% of BLAS single-node | code done; **criterion unverified, probably not met** |
| 3 | model within ~30% of measured on ≥3 configs | code done; **never compared** |
| 4 | `c` sweep reproduces the predicted optimum | code done; **never run** |
| 5 | planner ≥ fixed grid on every shape | code done; **never run** |
| 6 | overlap efficiency measured and explained | Ibcast+lookahead done; persistent dropped; hybrid sweep missing; **`eta` never computed** |
| 7 | all CSVs in `results/`, all figures regenerable | **not started** |
| 8 | report + slides | **not started** |

## Suggested order of work

1. Extract the tarball into the repo, drop `*.o` and the binary, commit the tree. Remove
   `results/*.csv` and `plots/*.png` from `.gitignore`.
2. Fix the CSV `err` column (defect 1) — it is a few lines, and every subsequent run
   produces evidence that is currently being thrown away.
3. Hoist `REP` to the outer loop in the three job scripts (defect 3) and fix `strong.pbs`'s
   walltime/baseline (defect 4). Do this *before* the campaign, not after.
4. Get a BLAS reference line and improve the kernel until it is a credible fraction of it.
   Ask the instructor §17 Q1 (is external BLAS allowed?) — it gates this step.
5. Run the four existing jobs. That produces Figs. 1, 3, 4, 5 and closes the week 4 and 5
   exit criteria.
6. Add `weak.pbs` (E4) and `hybrid.pbs` (E10) plus their plot kinds — both are cheap and E10
   needs no new code.
7. Compute and report `eta` (§7.1), and add the measured-vs-model overlay (§9, week 3
   criterion). Both are plot-script work on data you will already have.
8. Ask §17 Q2 (does "generic" include data types?). If yes, G3 is a real gap and §7.4's
   include-template trick is about a day's work. If no, say so explicitly in the report —
   the README currently redefines the third genericity axis as "arbitrary decomposition",
   which is not the guide's G3, and a reader comparing the two will notice.
