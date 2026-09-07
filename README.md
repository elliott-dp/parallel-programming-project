# gemm2d — distributed-memory generic matrix multiplication with 2D decomposition

`C := alpha * A * B + beta * C`, with `A: M×K`, `B: K×N`, `C: M×N`, distributed over a
`Pr × Pc × c` process grid.

"Generic" means three things, all of them tested:

| | |
|---|---|
| **arbitrary shape** | any `M`, `N`, `K` — square, tall-skinny, short-fat |
| **arbitrary machine** | any `P` including primes, any grid, non-divisible dimensions |
| **arbitrary decomposition** | grid, replication depth and panel width chosen by a calibrated cost model, not hardcoded |

Four interchangeable engines (`naive1d`, `cannon`, `summa`, `summa25d`), three interchangeable
broadcast policies (`blocking`, `ibcast`, `shm`), two local kernels, two grid mappings — every one
of them selectable from the command line so each is a single-flag ablation.

---

## Build

```bash
make                      # mpicc -std=c11 -O3 -march=native -funroll-loops -fopenmp
make OPT="-O2 -g"         # override optimisation
make clean
```

Requirements: any MPI-3 implementation (MPI-4 is used when available), a C11 compiler, OpenMP.

Verified with: **gcc 13.3.0**, **Open MPI 4.1.6**, Ubuntu 24.04.
On the cluster: `module load gcc91` and the site MPI module — run `module avail` to get its exact
name, then put it in `jobs/*.pbs` where the placeholder is.

## Run

```bash
mpiexec -n 16 ./gemm2d --M 4096 --N 4096 --K 4096 --engine summa --b 256
mpiexec -n 64 ./gemm2d --M 8192 --N 8192 --K 8192 --engine summa25d --c 4 --bcast shm
mpiexec -n 64 ./gemm2d --M 1024 --N 1024 --K 262144 --engine summa25d --plan --calibrate
./gemm2d --plan-report --plan-P 128 --M 32768 --N 32768 --K 512    # no MPI job needed
./gemm2d --help
```

### Changing the problem size and the resources

Everything is a command-line flag; nothing is a compile-time constant.

* problem size → `--M --N --K` (default 512 each)
* number of processes → `mpiexec -n <P>`; the grid follows from `--Pr --Pc --c`, or is chosen
  automatically (aspect-matched) when they are omitted, or by the cost model with `--plan`
* threads per rank → `OMP_NUM_THREADS` (plus `OMP_PROC_BIND=close OMP_PLACES=cores`)
* panel width → `--b` (default 128)
* local cache tiles → `make OPT="-O3 -march=native -DMC=64 -DKC=256 -DNC=512"`

### All flags

| flag | values | default | meaning |
|---|---|---|---|
| `--M --N --K` | int | 512 | matrix dimensions |
| `--alpha --beta` | float | 1, 0 | `C := alpha*A*B + beta*C` |
| `--engine` | `naive1d cannon summa summa25d` | `summa` | algorithm |
| `--bcast` | `blocking ibcast shm` | `blocking` | panel broadcast policy |
| `--lookahead` | 0,1 | 0 | one step of lookahead (`ibcast` only) |
| `--kernel` | `naive blocked` | `blocked` | local kernel |
| `--gridmap` | `linear nodeaware` | `linear` | rank → grid-coordinate mapping |
| `--Pr --Pc --c` | int | auto | process grid, `Pr*Pc*c == P` |
| `--b` | int | 128 | SUMMA panel width |
| `--plan` | — | off | cost model picks `Pr, Pc, c, b` |
| `--plan-report` `--plan-P` | — / int | — | print the top configurations and exit |
| `--calibrate` | — | off | measure `alpha, beta, gamma` first |
| `--mem-gb` | float | 2.0 | memory budget per rank for the planner |
| `--verify` | `none ref freivalds` | `none` | correctness check |
| `--trials` | int | 2 | Freivalds trials |
| `--reps --warmup` | int | 3, 1 | timed / untimed repetitions |
| `--seed` | int | 1 | generator seed |
| `--csv --tag` | path / string | — | append one row per repetition |
| `--quiet` | — | off | suppress the human-readable line |

### Output

One CSV row per timed repetition (no averaging in C — the statistics live in the plotting script,
so the raw data is always available):

```
tag,engine,bcast,kernel,gridmap,plan,M,N,K,P,Pr,Pc,c,b,threads,lookahead,rep,
time_s,gflops,wire_gb,rank_gb,t_comm,t_comp,err
```

`time_s` is the **maximum over ranks** of the elapsed time between two barriers — the slowest rank
defines the runtime. `wire_gb` is modelled panel traffic crossing a node boundary, computed the same
way for every policy so the ablation is apples-to-apples.

## Test

```bash
make test
# on a laptop or container, where ranks are oversubscribed:
MPIRUN_FLAGS="--oversubscribe --allow-run-as-root -q" ./tests/run_tests.sh
```

36 cases: prime `P`, non-divisible dimensions, degenerate grids (`1×8`, `8×1`), more ranks than
matrix rows, `alpha`/`beta`, panel widths from 1 to 4096, every engine, every broadcast policy,
both verifiers. All must print `[OK]`.

`GEMM2D_FAKE_PPN=k` makes the code pretend that every `k` consecutive world ranks form a node. That
lets the shared-memory broadcast and the node-aware mapping be exercised and validated on a single
machine before you ever queue a multi-node job. The shared-memory window is real; only the grouping
is synthetic.

## Reproducing the figures

```bash
qsub jobs/strong.pbs          # E3  strong scaling
qsub jobs/csweep.pbs          # E6  2.5D replication sweep vs the c = P^(1/3) prediction
qsub jobs/shapes.pbs          # E8  planner vs fixed grid across five matrix shapes
qsub jobs/shm_ablation.pbs    # E9  blocking vs ibcast vs shared memory

python3 scripts/plot_results.py results/strong.csv        --kind strong
python3 scripts/plot_results.py results/csweep.csv        --kind csweep
python3 scripts/plot_results.py results/shapes.csv        --kind shapes
python3 scripts/plot_results.py results/shm_ablation.csv  --kind ablation
```

Statistics follow the course's benchmarking rules: median with a percentile-bootstrap 95% confidence
interval (no normality assumed), harmonic mean for rates, arithmetic mean never used for ratios, no
outlier removal. Figures land in `plots/`.

## Layout

```
include/gemm2d.h   all types and prototypes
src/util.c         counters, block partition, deterministic generator, dmat, node split
src/pgrid.c        process grid, communicator hierarchy, node-aware mapping
src/chan.c         panel broadcast channel: blocking / Ibcast / shared memory
src/kernel.c       local rank-k update, naive and cache-blocked + OpenMP
src/engines.c      SUMMA, 2.5D SUMMA, Cannon, naive 1D
src/verify.c       exact reference check and distributed Freivalds
src/planner.c      machine calibration and the cost-model grid selector
src/main.c         CLI, timing loop, CSV
tests/             correctness suite
jobs/              PBS scripts for the four experiments
scripts/           statistics and plotting
results/ plots/    evidence for the report
docs/DESIGN.md     why each choice was made, and how it maps to the report
```

## Honest notes

* **The shared-memory broadcast only pays off with several ranks per node and more than one node.**
  With one rank per node it degenerates to a plain broadcast plus two barriers, and it is slower. On
  a single oversubscribed machine it is *much* slower, because the barriers serialise ranks that are
  time-sharing one core. Do not benchmark it on a laptop and conclude anything.
* It also does not compose with lookahead: the shared panel buffer has to be re-synchronised before
  it can be overwritten, so the barriers that make it correct also remove the overlap. That is a
  result worth reporting, not a bug — see `docs/DESIGN.md`.
* Cannon deliberately refuses non-square grids and non-divisible dimensions instead of silently
  padding. That refusal is the argument for choosing SUMMA as the base engine.
* Floating-point summation order changes with `c` and `b`, so results are not bitwise identical
  across configurations. Quantify it rather than hiding it: run the same problem with different `c`
  and compare.
* `--verify ref` is `O(M*N*K)` on one rank and is capped at `M*N ≤ 4·10⁶`. At full scale use
  `--verify freivalds`, which is `O(n²)` and cheap enough to leave on in production runs.
