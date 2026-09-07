#!/bin/bash
# run_local.sh -- every experiment that fits on ONE machine (laptop, container,
# or a single interactive cluster node).
#
# The point of this script is insurance: it produces real, plottable evidence
# without touching the PBS queue, so a report is never blocked on a job that
# has not come back. Everything here finishes in a few minutes.
#
#   ./scripts/run_local.sh              # defaults: 4 ranks, small sizes
#   NP=8 SIZE=1024 ./scripts/run_local.sh
#
# On an oversubscribed box:
#   MPIRUN_FLAGS="--oversubscribe --allow-run-as-root -q" ./scripts/run_local.sh
set -u
cd "$(dirname "$0")/.."
BIN=./gemm2d
MF=${MPIRUN_FLAGS:-}
NP=${NP:-4}
SIZE=${SIZE:-768}          # per-run M=N=K unless overridden
REPS=${REPS:-10}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

[ -x "$BIN" ] || { echo "build first: make"; exit 1; }
mkdir -p results plots

say(){ printf '\n=== %s ===\n' "$1"; }

# --------------------------------------------------------------------------
# E1  local kernel ablation -- one rank, no MPI traffic at all, so this is a
#     clean measurement of gamma. This is the figure that shows the local
#     kernel is not the bottleneck; guide S8 and S16 both hang on it.
# --------------------------------------------------------------------------
say "E1 kernel ablation (naive / blocked / packed, 1 and all threads)"
rm -f results/kernel.csv
for T in 1 ${OMP_MAX:-$(nproc)}; do
  for REP in $(seq 1 $REPS); do
    for K in naive blocked packed; do
      OMP_NUM_THREADS=$T mpirun $MF -n 1 $BIN \
        --M $SIZE --N $SIZE --K 256 --kernel $K --b 256 \
        --reps 1 --warmup 1 --quiet --tag "kernel_${K}_t${T}" \
        --csv results/kernel.csv
    done
  done
done

# --------------------------------------------------------------------------
# E5  engine comparison. naive1d's per-rank volume is K*N regardless of P, so
#     its disadvantage is visible even at small P -- that is the whole point
#     of keeping it. Cannon is included where the grid is square.
# --------------------------------------------------------------------------
say "E5 engine comparison at P=$NP"
rm -f results/engines.csv
SQ=1; q=$(awk "BEGIN{print int(sqrt($NP)+0.5)}"); [ $((q*q)) -eq $NP ] || SQ=0
for REP in $(seq 1 $REPS); do
  for E in naive1d summa summa25d; do
    C=1; [ "$E" = summa25d ] && C=2
    [ "$E" = summa25d ] && [ $((NP % 2)) -ne 0 ] && continue
    mpirun $MF -n $NP $BIN --M $SIZE --N $SIZE --K $SIZE --engine $E --c $C \
      --b 128 --reps 1 --warmup 1 --quiet --tag "engine_$E" \
      --csv results/engines.csv
  done
  if [ $SQ -eq 1 ]; then
    D=$(( SIZE - SIZE % q ))
    mpirun $MF -n $NP $BIN --M $D --N $D --K $D --engine cannon \
      --reps 1 --warmup 1 --quiet --tag "engine_cannon" \
      --csv results/engines.csv
  fi
done

# --------------------------------------------------------------------------
# E7  panel-width sweep. The cost model says the bandwidth term does not
#     depend on b and only the latency term does (~1/b), so on a single node
#     -- where latency is small -- b should barely matter. Measuring that
#     confirms the model rather than contradicting it; say so in the caption.
# --------------------------------------------------------------------------
say "E7 panel width sweep at P=$NP"
rm -f results/bsweep.csv
for REP in $(seq 1 $REPS); do
  for B in 16 32 64 128 256 512; do
    mpirun $MF -n $NP $BIN --M $SIZE --N $SIZE --K $SIZE --engine summa --b $B \
      --reps 1 --warmup 1 --quiet --tag "b$B" --csv results/bsweep.csv
  done
done

# --------------------------------------------------------------------------
# E2  intra-node alpha / beta / gamma. The cluster run gives the inter-node
#     numbers; these are the intra-node half of the same table.
# --------------------------------------------------------------------------
say "E2 machine calibration (intra-node)"
mpirun $MF -n 2 $BIN --M 64 --N 64 --K 64 --calibrate --reps 1 --warmup 0 \
  2>&1 | grep '^# calibrated' | tee results/calibration_intranode.txt

# --------------------------------------------------------------------------
# S10 layer 4 -- cross-configuration consistency. Same problem, different
#     decomposition; the residual is expected to be non-zero and small.
#     This is a table, not a figure, and it is cheap credibility.
# --------------------------------------------------------------------------
say "Cross-configuration consistency (guide S10 layer 4)"
rm -f results/consistency.csv
for CFG in "summa 1 16" "summa 1 64" "summa 1 256" "summa25d 2 32" "summa25d 4 64"; do
  set -- $CFG
  [ "$1" = summa25d ] && [ $((NP % $2)) -ne 0 ] && continue
  mpirun $MF -n $NP $BIN --M 401 --N 397 --K 389 --engine "$1" --c "$2" --b "$3" \
    --verify freivalds --trials 4 --reps 1 --warmup 0 --quiet \
    --tag "consist_$1_c$2_b$3" --csv results/consistency.csv
done

# --------------------------------------------------------------------------
# E8 (model half) -- the planner needs no MPI to report what it would choose.
#     This gives the shape table for the headline claim even if the queue
#     never returns.
# --------------------------------------------------------------------------
say "E8 planner decisions across shapes (model only, no MPI job needed)"
{
  for PP in 64 128 256; do
    for S in "8192 8192 8192 square" "16384 16384 512 tall_skinny" \
             "1024 1024 262144 short_fat" "32768 1024 2048 tall_A" \
             "1024 32768 2048 wide_B"; do
      set -- $S
      echo "## P=$PP shape=$4  M=$1 N=$2 K=$3"
      $BIN --plan-report --plan-P $PP --M $1 --N $2 --K $3
    done
  done
} | tee results/plan_report.txt | grep -E '^##|^ ' | head -40

say "done -- CSVs in results/"
ls -la results/
