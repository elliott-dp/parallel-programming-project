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

## 2. Build and smoke-test on a login node

```bash
make
./tests/run_tests.sh          # 43 cases, all must print [OK]
```

If `make` cannot find `mpicc`, the module is not loaded in your interactive shell. The job
scripts handle this themselves (see below), but for the interactive test:

```bash
module avail                   # find the MPI module's exact name
module load gcc91 <that-name>
```

## 3. Submit

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

Watch them with `qstat -u $USER`. Each job writes `results/<name>.out` and `.err`.

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

---

## If the queue does not come back in time

`docs/EXPERIMENTS.md` covers this: present the single-machine half, which is already committed
in `results/`, and say the cluster campaign is queued. Label those results as **single-node,
shared-memory** — they show the algorithm behaves as the model predicts, not that it scales
across a network.
