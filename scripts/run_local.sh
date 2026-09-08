#!/bin/bash
# run_local.sh -- every experiment that fits on ONE machine (laptop, container,
# a cloud session, or a single interactive cluster node).
#
# Insurance: this produces real, plottable evidence without touching the PBS
# queue, so a report is never blocked on a job that has not come back.
#
#   ./scripts/run_local.sh
#   NP=8 SIZE=3072 REPS=10 ./scripts/run_local.sh
#
# METHODOLOGY -- two settings matter more than anything else here, and getting
# them wrong produces confidence intervals a factor of several wide:
#
#   1. BIND ranks to cores. Unbound ranks migrate between cores and the medians
#      become meaningless. Default is --bind-to core.
#   2. Make each run LONG ENOUGH. A run of a few milliseconds measures MPI
#      startup jitter, not the algorithm. SIZE is chosen so a run takes O(0.1s).
#
# Do not oversubscribe: NP must be <= the core count, or the ranks time-share
# and every timing comparison below becomes noise.
set -u
cd "$(dirname "$0")/.."
BIN=./gemm2d
NP=${NP:-$(nproc)}
SIZE=${SIZE:-2048}
REPS=${REPS:-10}
BIND=${BIND:---bind-to core}
MF="--allow-run-as-root -q $BIND"
CORES=$(nproc)
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export OMP_PROC_BIND=close OMP_PLACES=cores

[ -x "$BIN" ] || { echo "build first: make"; exit 1; }
[ "$NP" -le "$CORES" ] || echo "WARNING: NP=$NP > $CORES cores; timings will be noise"
mkdir -p results plots
say(){ printf '\n=== %s ===\n' "$1"; }
run(){ mpirun $MF "$@" < /dev/null; }

# --------------------------------------------------------------------------
# E1  local kernel ablation -- one rank, no MPI traffic, so this IS gamma.
# --------------------------------------------------------------------------
say "E1 kernel ablation (naive / blocked / packed)"
rm -f results/kernel.csv
for REP in $(seq 1 $REPS); do
  for T in 1 $CORES; do
    for K in naive blocked packed; do
      OMP_NUM_THREADS=$T run -n 1 $BIN --M 1024 --N 1024 --K 256 --kernel $K \
        --b 256 --reps 1 --warmup 1 --quiet --tag "kernel_${K}_t${T}" \
        --csv results/kernel.csv
    done
  done
done

# --------------------------------------------------------------------------
# E3  strong scaling WITHIN ONE NODE. Real, but it is a shared-memory result:
#     it says nothing about the network. Label it as such in the report.
# --------------------------------------------------------------------------
say "E3 strong scaling, single node, P=1..$NP"
rm -f results/strong_local.csv
PS=""; p=1; while [ $p -le $NP ]; do PS="$PS $p"; p=$((p*2)); done
for REP in $(seq 1 $REPS); do
  for P in $PS; do
    run -n $P $BIN --M $SIZE --N $SIZE --K $SIZE --engine summa --b 256 \
      --reps 1 --warmup 1 --quiet --tag strong_local --csv results/strong_local.csv
  done
done

# --------------------------------------------------------------------------
# E5  engine comparison. naive1d's per-rank volume is K*N regardless of P.
# --------------------------------------------------------------------------
say "E5 engine comparison at P=$NP"
rm -f results/engines.csv
q=$(awk "BEGIN{print int(sqrt($NP)+0.5)}"); SQ=0; [ $((q*q)) -eq $NP ] && SQ=1
for REP in $(seq 1 $REPS); do
  for E in naive1d summa summa25d; do
    C=1; [ "$E" = summa25d ] && C=2
    [ "$E" = summa25d ] && [ $((NP % 2)) -ne 0 ] && continue
    run -n $NP $BIN --M $SIZE --N $SIZE --K $SIZE --engine $E --c $C --b 256 \
      --reps 1 --warmup 1 --quiet --tag "engine_$E" --csv results/engines.csv
  done
  [ $SQ -eq 1 ] && run -n $NP $BIN --M $SIZE --N $SIZE --K $SIZE --engine cannon \
      --reps 1 --warmup 1 --quiet --tag engine_cannon --csv results/engines.csv
done

# --------------------------------------------------------------------------
# E6  2.5D replication sweep. The model puts the optimum near c = P^(1/3).
# --------------------------------------------------------------------------
say "E6 c sweep at P=$NP (model predicts c ~ P^(1/3))"
rm -f results/csweep_local.csv
CS=""; c=1; while [ $c -le $NP ]; do [ $((NP % c)) -eq 0 ] && CS="$CS $c"; c=$((c*2)); done
for REP in $(seq 1 $REPS); do
  for C in $CS; do
    run -n $NP $BIN --M $SIZE --N $SIZE --K $SIZE --engine summa25d --c $C \
      --b 256 --reps 1 --warmup 1 --quiet --tag csweep --csv results/csweep_local.csv
  done
done

# --------------------------------------------------------------------------
# E7  panel-width sweep. Only the latency term depends on b (as 1/b); the
#     bandwidth term does not. The measured split should show exactly that.
# --------------------------------------------------------------------------
say "E7 panel width sweep at P=$NP"
rm -f results/bsweep.csv
for REP in $(seq 1 $REPS); do
  for B in 16 32 64 128 256 512; do
    run -n $NP $BIN --M $SIZE --N $SIZE --K $SIZE --engine summa --b $B \
      --reps 1 --warmup 1 --quiet --tag "b$B" --csv results/bsweep.csv
  done
done

# --------------------------------------------------------------------------
# E8  planner vs fixed grid across shapes -- the originality claim, at small
#     scale. On one node the communication is intra-node, so treat this as
#     "does the planner pick a better decomposition", not as the final number.
# --------------------------------------------------------------------------
say "E8 planner vs fixed grid across shapes at P=$NP"
rm -f results/shapes_local.csv
FPR=$q; FPC=$((NP / q))          # the fixed near-square grid to beat
for REP in $(seq 1 $REPS); do
  while read -r M N K LAB <&3; do
    [ -z "$LAB" ] && continue
    run -n $NP $BIN --M $M --N $N --K $K --engine summa25d --c 1 \
      --Pr $FPR --Pc $FPC --b 256 --reps 1 --warmup 1 --quiet \
      --tag "fixed_$LAB" --csv results/shapes_local.csv
    run -n $NP $BIN --M $M --N $N --K $K --engine summa25d \
      --plan --calibrate --mem-gb 4 --reps 1 --warmup 1 --quiet \
      --tag "plan_$LAB" --csv results/shapes_local.csv
  done 3<<EOF
$SIZE $SIZE $SIZE square
4096 4096 256 tall_skinny
512 512 32768 short_fat
8192 512 1024 tall_A
512 8192 1024 wide_B
EOF
done

# --------------------------------------------------------------------------
# E10 hybrid ranks x threads at a FIXED core count.
# --------------------------------------------------------------------------
say "E10 hybrid mapping at $CORES cores"
rm -f results/hybrid_local.csv
for REP in $(seq 1 $REPS); do
  r=$CORES
  while [ $r -ge 1 ]; do
    t=$((CORES / r))
    OMP_NUM_THREADS=$t mpirun --allow-run-as-root -q --map-by slot:PE=$t --bind-to core \
      -n $r $BIN --M $SIZE --N $SIZE --K $SIZE --engine summa --b 256 \
      --reps 1 --warmup 1 --quiet --tag "hybrid_${r}x${t}" \
      --csv results/hybrid_local.csv 2>/dev/null
    r=$((r / 2))
  done
done

# --------------------------------------------------------------------------
# E9  broadcast policy -- BYTES ONLY. The off-node byte count is a determin-
#     istic property of the grid and the policy, so it is meaningful on one
#     machine. The TIMING half is not: the README explains why shm loses on a
#     single box. GEMM2D_FAKE_PPN makes the ranks pretend to be two nodes.
# --------------------------------------------------------------------------
say "E9 broadcast policy -- modelled off-node bytes (timing needs real nodes)"
rm -f results/policy_bytes.csv
# 8 ranks as 2 nodes x 4, on a 4x2 grid: each COLUMN then holds 2 ranks per node
# across 2 nodes, which is the configuration shm exists to exploit. This
# oversubscribes the cores, which is fine -- only the byte counters are read,
# and those are deterministic. The TIMES from this block are meaningless.
for POL in blocking ibcast shm; do
  for MAP in linear nodeaware; do
    GEMM2D_FAKE_PPN=4 mpirun --allow-run-as-root -q --oversubscribe -n 8 $BIN \
      --M 1024 --N 1024 --K 1024 --Pr 4 --Pc 2 \
      --engine summa --b 256 --bcast $POL --gridmap $MAP \
      --reps 1 --warmup 0 --quiet --tag "policy_${POL}_${MAP}" \
      --csv results/policy_bytes.csv < /dev/null
  done
done

# --------------------------------------------------------------------------
say "E2a machine calibration (intra-node)"
run -n 2 $BIN --M 64 --N 64 --K 64 --calibrate --reps 1 --warmup 0 2>&1 \
  | grep '^# calibrated' | tee results/calibration_intranode.txt

say "Cross-configuration consistency (guide S10 layer 4)"
rm -f results/consistency.csv
for CFG in "summa 1 16" "summa 1 64" "summa 1 256" "summa25d 2 32" "summa25d 4 64"; do
  set -- $CFG
  [ "$1" = summa25d ] && [ $((NP % $2)) -ne 0 ] && continue
  run -n $NP $BIN --M 401 --N 397 --K 389 --engine "$1" --c "$2" --b "$3" \
    --verify freivalds --trials 4 --reps 1 --warmup 0 --quiet \
    --tag "consist_$1_c$2_b$3" --csv results/consistency.csv
done

say "E8a planner decisions across shapes (model only, no MPI job)"
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
} > results/plan_report.txt

say "done"; ls results/
