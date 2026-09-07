# Running on the cluster

Everything here runs in **your own terminal**, on your machine and then on the cluster.
It cannot be run from a sandboxed container: there is no interactive shell, outbound
SSH is blocked, and — more importantly — your cluster credentials should not go into it.

## 1. Get the code onto the cluster

From your laptop:

```bash
ssh <username>@<cluster-login-host>
```

Then on the cluster, either clone it:

```bash
git clone https://github.com/elliott-dp/parallel-programming-project.git
cd parallel-programming-project
```

or, if the cluster cannot reach GitHub (or the repo is private and you would rather not put
a token there), push it from your laptop instead:

```bash
# on your laptop, from the repo root
rsync -av --exclude '.git' --exclude '*.o' --exclude gemm2d \
    ./ <username>@<cluster-login-host>:~/gemm2d/
```

## 2. Build and smoke-test — NOT on the head node

**Cluster policy 6.2 forbids executing any job on the head nodes** (`hpc-head-n1`,
`hpc-head-n2`), and the test suite launches MPI ranks. So compile and test inside an
allocation, not on the login shell.

Either as a batch job:

```bash
qsub jobs/smoke.pbs           # builds, then runs the 43-case suite
qstat -u $USER                # wait for it
cat results/smoke.out         # every line must say [OK]
```

or interactively:

```bash
qsub -I -l select=1:ncpus=9:mpiprocs=9:mem=8gb -q short_cpuQ -l walltime=0:20:00
# once the shell lands on a compute node:
cd parallel-programming-project
. jobs/env.sh                 # loads gcc + MPI, or tells you the exact module to name
make
./tests/run_tests.sh
```

Editing files, `git` operations and `make` alone are ordinary interactive use. Running the
binary — even at one rank — is a job, so keep it inside an allocation.

## 2b. Survey the cluster before you size anything

`pbsnodes` and `qstat` are read-only scheduler queries. They run nothing on a compute node, so
they are not "jobs" and are fine on the head node — that is what it is for.

```bash
./scripts/cluster_probe.sh | tee results/cluster_survey.txt
```

It answers the three things you need before submitting:

| question | why it matters |
|---|---|
| which queues, and their walltime / size limits | `strong.pbs` asks for 2 h and 128 cores; if `short_cpuQ` caps below that, the job never starts |
| how many nodes, at what cores and memory | decides the `#PBS -l select=` you can actually get |
| which nodes are on Omni-Path vs 10 GbE | a sweep split across both fabrics is not a valid comparison |

The individual commands, if you would rather run them by hand:

```bash
qstat -Q                      # queues
qstat -Qf short_cpuQ          # that queue's limits
pbsnodes -aSj                 # one line per node: state, cores used/total, jobs
pbsnodes -a | less            # everything, including each node's resources
qstat -u $USER                # your own jobs
```

**What you cannot do:** `ssh` to a worker node (policy 6.2), or run `./gemm2d` / `mpirun` on
the head node (also 6.2). Looking at the nodes is fine; reaching them outside PBS is not.

If the probe shows a fabric tag, pin it in the select statement so every run of a sweep lands
on the same interconnect, and keep it identical across `strong.pbs` and `shm_ablation.pbs`.

## 3. Submit

**Submit `jobs/smoke.pbs` first and let it finish.** The other six refuse to start unless
`./gemm2d` already exists, and that is deliberate: they all share `$PBS_O_WORKDIR`, so if each
one rebuilt, a `make clean` in one job would delete the binary another job is executing. Build
once; then the six can run concurrently.

The module name is discovered automatically by `jobs/env.sh`, which every job script sources.
If the guess is wrong the job stops immediately with the exact command to fix it, rather than
failing halfway through a two-hour allocation.

```bash
qsub jobs/strong.pbs          # E3  strong scaling          ~2 h
qsub jobs/weak.pbs            # E4  weak scaling            ~1 h
qsub jobs/shapes.pbs          # E8  planner vs fixed grid   ~2 h   <-- the headline
qsub jobs/csweep.pbs          # E6  2.5D replication sweep  ~1 h
qsub jobs/shm_ablation.pbs    # E9  broadcast policy        ~1 h
qsub jobs/hybrid.pbs          # E10 ranks x threads         ~1 h
```

Override the module names if the autodetection picks wrong:

```bash
MPI_MODULE=mpich-3.2.1--gcc-9.1.0 qsub jobs/strong.pbs
```

Watch them with `qstat -u $USER`. Each job writes `results/<name>.out` and `.err`, plus a
`results/<name>.<jobid>.nodes` file listing the nodes it actually landed on.

**Check those `.nodes` files before comparing runs.** The cluster interconnects some nodes at
10 Gb/s Ethernet and others with Omni-Path. A strong-scaling curve or a broadcast-policy
ablation assembled from runs that landed on different fabrics is not a valid comparison — the
independent variable would be the network, not the thing you varied. Guide §11 asks for node
IDs to be noted for exactly this reason. If the sets differ, either resubmit to get a
consistent allocation or report the split honestly.

**Adjust `#PBS -l select=...` to match your queue.** The scripts ask for
`4:ncpus=32:mpiprocs=32:mem=64gb` (128 cores). If `short_cpuQ` gives you less, lower the rank
counts in the loops to match, or the jobs will sit in the queue or fail to start.

## 4. Plot

```bash
python3 scripts/plot_results.py results/strong.csv       --kind strong
python3 scripts/plot_results.py results/weak.csv         --kind weak
python3 scripts/plot_results.py results/shapes.csv       --kind shapes
python3 scripts/plot_results.py results/csweep.csv       --kind csweep
python3 scripts/plot_results.py results/shm_ablation.csv --kind ablation
python3 scripts/plot_results.py results/hybrid.csv       --kind hybrid
```

Without matplotlib it prints the tables and skips the figures, which is enough to read the
result. Copy the CSVs back to your laptop to plot there:

```bash
scp -r <username>@<cluster-login-host>:~/parallel-programming-project/results ./
```

## 5. Commit the evidence

`results/*.csv` and `plots/*.png` are deliberately **not** in `.gitignore` — guide §12 asks for
the raw CSVs to be committed as the report's evidence.

```bash
git add results plots && git commit -m "Cluster campaign results" && git push
```

## Cluster policy notes

Points from the usage regulations that touch this project:

* **6.2 — no jobs on head nodes.** Handled above: the smoke test is a job, not a login-shell
  command. Do not run `./gemm2d` or `mpirun` on `hpc-head-n1` / `hpc-head-n2`.
* **6.2 — no `ssh` to compute nodes.** Nothing here does that. Use `qsub -I` for an
  interactive session.
* **6.3 — job names must not identify a person.** The names are `gemm2d_strong`,
  `gemm2d_weak`, `gemm2d_shapes`, `gemm2d_csweep`, `gemm2d_shm`, `gemm2d_hybrid`,
  `gemm2d_smoke`. All fine — do not rename them to anything containing your name or ID.
* **2.2 — 250 GB home quota.** This project's footprint is negligible: source, a handful of
  CSVs and a few PNGs, well under 100 MB. The large numbers in the jobs (`n = 23170` at
  `P = 128` in the weak-scaling sweep) are RAM per node, not disk.
* **6.1 — teaching and research only.** This is coursework, which qualifies.

---

## If the queue does not come back in time

`docs/EXPERIMENTS.md` covers this: present the single-machine half, which is already committed
in `results/`, and say the cluster campaign is queued. Label those results as **single-node,
shared-memory** — they show the algorithm behaves as the model predicts, not that it scales
across a network.
