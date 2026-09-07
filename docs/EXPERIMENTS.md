# Experiment plan

A deliberately small plan. Every entry says which machine runs it, roughly how long it takes,
and which figure or table it produces. Anything that does not earn a figure or a sentence is
not here.

The plan is split so that **the report is never blocked on the PBS queue**. The single-machine
half runs anywhere — a laptop, a container, an interactive cluster node —
and finishes in minutes. The cluster half is the part that genuinely needs multiple nodes.

---

## Half A — single machine (no queue, no waiting)

One command:

```bash
make
MPIRUN_FLAGS="--oversubscribe --allow-run-as-root -q" ./scripts/run_local.sh
```

Defaults to 4 ranks and small sizes; override with `NP=8 SIZE=1024 REPS=10 ./scripts/run_local.sh`.
Total runtime: a few minutes. It writes `results/kernel.csv`, `results/engines.csv`,
`results/bsweep.csv`, `results/consistency.csv`, `results/calibration_intranode.txt` and
`results/plan_report.txt`.

| # | Experiment | Produces | Why it works on one machine |
|---|---|---|---|
| E1 | kernel ablation: `naive` / `blocked` / `packed`, 1 vs all threads | **Figure — kernel** | one rank, zero MPI traffic; this *is* the measurement of `gamma` |
| E5 | engine comparison: `naive1d` / `summa` / `summa25d` / `cannon` | **Figure — engines** | `naive1d`'s per-rank volume is `K·N` regardless of `P`, so its disadvantage shows at `P=4` |
| E7 | panel-width sweep `b = 16…512` | **Figure — b sweep** | the model says only the latency term depends on `b`; a flat curve on one node *confirms* that |
| E2a | `alpha`, `beta`, `gamma` intra-node | table row | ping-pong within a node is a real measurement |
| §10.4 | cross-configuration consistency across `(engine, c, b)` | table | pure numerics, no scale needed |
| E8a | planner's chosen grid per shape at `P = 64,128,256` | table | `--plan-report` runs the cost model with **no MPI job at all** |

Plot them:

```bash
python3 scripts/plot_results.py results/kernel.csv  --kind kernel
python3 scripts/plot_results.py results/engines.csv --kind engines
python3 scripts/plot_results.py results/bsweep.csv  --kind bsweep
```

**Watch the confidence intervals.** On a contended or oversubscribed box, `E5` and `E7` produce
wide intervals — running `P` ranks on `P` cores while the OS and everything else compete gives
medians whose CI spans a factor of several. `E1` is immune (one rank, one process) and is therefore
the figure to trust from Half A. If `E5`'s intervals overlap, say so and lean on the cluster run
instead of presenting a ranking the data does not support. Reporting the interval and declining to
rank is a better answer than a bar chart that implies a difference which is not there.

**What Half A cannot tell you:** anything about the network. `shm` versus `blocking` is
meaningless here (the README says so), strong scaling past one node is not measurable, and
`beta` is intra-node only. Do not present single-machine numbers as scaling results.

---

## Half B — the cluster (submit, then stop waiting on it)

Submit all five at once. They are independent, so the queue can interleave them.

```bash
qsub jobs/strong.pbs        # E3  strong scaling            ~2 h wall
qsub jobs/weak.pbs          # E4  weak scaling              ~1 h
qsub jobs/shapes.pbs        # E8  planner vs fixed grid     ~2 h   <-- the headline
qsub jobs/csweep.pbs        # E6  2.5D replication sweep    ~1 h
qsub jobs/shm_ablation.pbs  # E9  broadcast policy          ~1 h
qsub jobs/hybrid.pbs        # E10 ranks x threads           ~1 h
```

| # | Experiment | Produces | Needs the cluster because |
|---|---|---|---|
| E3 | strong scaling, `P = 1…128`, `n = 8192` | **Figure — scaling** | speedup past one node is the whole point |
| E4 | weak scaling, `n = 2048·sqrt(P)` | **Figure — scaling** | same |
| E8b | planner vs fixed `8x8` across five shapes | **Figure — the headline** | the effect is a communication effect |
| E6 | `c` sweep against the `c ≈ P^(1/3)` prediction | **Figure — 2.5D** | the depth reduce has to cross the network |
| E9 | `blocking` / `ibcast` / `ibcast+lookahead` / `shm`, `linear` vs `nodeaware` | **Figure — ablation** | `shm` only pays with several ranks per node *and* more than one node |
| E10 | ranks x threads at a fixed 128 cores | figure or one sentence | aggregate traffic falls as `sqrt(P)` only across real nodes |
| E2b | inter-node `alpha`, `beta` | table row | comes free from `--calibrate` inside `shapes.pbs` |

Plot them:

```bash
python3 scripts/plot_results.py results/strong.csv       --kind strong
python3 scripts/plot_results.py results/weak.csv         --kind weak
python3 scripts/plot_results.py results/shapes.csv       --kind shapes
python3 scripts/plot_results.py results/csweep.csv       --kind csweep
python3 scripts/plot_results.py results/shm_ablation.csv --kind ablation
python3 scripts/plot_results.py results/hybrid.csv       --kind hybrid
```

**Before submitting anything**, check the module name — the guide is explicit that the MPI
module is not called the same thing as the gcc one:

```bash
module avail
```

then replace the `# module load <your-mpi-module>` placeholder in every `jobs/*.pbs`.

---

## The five figures

Four pages allows about five figures. This is the set, in priority order — the first two are
guaranteed because they come from Half A.

1. **Kernel ablation** (Half A) — `naive` / `blocked` / `packed`, 1 and N threads. Establishes
   that the local kernel is not the bottleneck, which every scaling claim depends on.
2. **Engine comparison** (Half A at small `P`, Half B at `P=64`) — the measured argument for
   choosing SUMMA rather than an assertion.
3. **Planner vs fixed grid across shapes** (Half B; Half A gives the model table as a fallback)
   — the originality claim.
4. **Strong + weak scaling** (Half B) — one figure, two panels.
5. **`c` sweep vs the `P^(1/3)` prediction**, or the broadcast-policy ablation (Half B) —
   whichever comes back cleaner. Keep the other's data in `results/` and give it one sentence.

## Statistics (already handled, do not re-derive)

`--reps 1 --warmup 1` inside a `REP` loop means one timed measurement per process launch, ten
launches per configuration, with the configurations **interleaved** across the queue so cluster
drift does not correlate with the independent variable. Time is the max over ranks between two
barriers. No averaging happens in C — the raw rows go to CSV and `plot_results.py` reports the
median with a percentile-bootstrap 95% CI, harmonic means for rates, and removes no outliers.

## If the queue does not come back in time

Present Half A, plus the planner's model table from `results/plan_report.txt`, and say plainly
that the cluster campaign is queued. A small, honest, fully reproducible result set beats a
large one you cannot defend. Half A alone covers figures 1 and 2, the `gamma` calibration, the
cross-configuration consistency table, and the planner's decisions per shape.
