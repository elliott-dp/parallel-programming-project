#!/bin/bash
# Correctness suite. Everything here must pass before any timing run.
#
# On a laptop/container you will need:
#     export MPIRUN_FLAGS="--oversubscribe --allow-run-as-root -q"
# On the cluster leave MPIRUN_FLAGS empty.
#
# GEMM2D_FAKE_PPN=k pretends that every k consecutive ranks form a node,
# which exercises the shared-memory broadcast without a second node.

set -u
cd "$(dirname "$0")/.."
BIN=./gemm2d
MF=${MPIRUN_FLAGS:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

pass=0; fail=0
run() {
    desc="$1"; shift
    out=$(mpirun $MF "$@" 2>&1)
    if echo "$out" | grep -q "\[OK\]"; then
        pass=$((pass+1)); printf "  ok    %s\n" "$desc"
    else
        fail=$((fail+1)); printf "  FAIL  %s\n" "$desc"; echo "$out" | sed 's/^/          /'
    fi
}

echo "== single rank =="
run "P=1 summa"            -n 1 $BIN --M 61 --N 53 --K 47 --engine summa --b 16 --verify ref --reps 1 --warmup 0 --quiet

echo "== generic P: primes and odd counts, non-divisible dimensions =="
for P in 2 3 4 5 6 7 8; do
run "P=$P summa 61x53x47"  -n $P $BIN --M 61 --N 53 --K 47 --engine summa --b 8 --verify ref --reps 1 --warmup 0 --quiet
done

echo "== rectangular grids =="
run "P=6 grid 2x3"         -n 6 $BIN --M 61 --N 53 --K 47 --Pr 2 --Pc 3 --engine summa --b 8 --verify ref --reps 1 --warmup 0 --quiet
run "P=6 grid 3x2"         -n 6 $BIN --M 61 --N 53 --K 47 --Pr 3 --Pc 2 --engine summa --b 8 --verify ref --reps 1 --warmup 0 --quiet
run "P=8 grid 1x8"         -n 8 $BIN --M 61 --N 53 --K 47 --Pr 1 --Pc 8 --engine summa --b 8 --verify ref --reps 1 --warmup 0 --quiet
run "P=8 grid 8x1"         -n 8 $BIN --M 61 --N 53 --K 47 --Pr 8 --Pc 1 --engine summa --b 8 --verify ref --reps 1 --warmup 0 --quiet

echo "== more ranks than rows/cols (empty blocks) =="
run "P=8 M=3 N=2 K=5"      -n 8 $BIN --M 3 --N 2 --K 5 --engine summa --b 2 --verify ref --reps 1 --warmup 0 --quiet

echo "== alpha / beta =="
run "alpha=2.5 beta=-1.5"  -n 4 $BIN --M 40 --N 36 --K 28 --alpha 2.5 --beta -1.5 --engine summa --b 8 --verify ref --reps 1 --warmup 0 --quiet

echo "== panel widths =="
for b in 1 3 8 64 4096; do
run "b=$b"                 -n 4 $BIN --M 61 --N 53 --K 47 --engine summa --b $b --verify ref --reps 1 --warmup 0 --quiet
done

echo "== broadcast policies =="
run "ibcast"               -n 4 $BIN --M 61 --N 53 --K 47 --bcast ibcast --b 8 --verify ref --reps 1 --warmup 0 --quiet
run "ibcast + lookahead"   -n 4 $BIN --M 61 --N 53 --K 47 --bcast ibcast --lookahead 1 --b 8 --verify ref --reps 1 --warmup 0 --quiet
run "shm (real nodes)"     -n 4 $BIN --M 61 --N 53 --K 47 --bcast shm --b 8 --verify ref --reps 1 --warmup 0 --quiet
GEMM2D_FAKE_PPN=2 run "shm 4 nodes x 2"  -n 8 $BIN --M 61 --N 53 --K 47 --bcast shm --b 8 --verify ref --reps 1 --warmup 0 --quiet
GEMM2D_FAKE_PPN=4 run "shm 2 nodes x 4"  -n 8 $BIN --M 61 --N 53 --K 47 --bcast shm --b 8 --verify ref --reps 1 --warmup 0 --quiet

echo "== node-aware mapping =="
GEMM2D_FAKE_PPN=4 run "nodeaware 4x2"    -n 8 $BIN --M 61 --N 53 --K 47 --Pr 4 --Pc 2 --gridmap nodeaware --bcast shm --b 8 --verify ref --reps 1 --warmup 0 --quiet

echo "== 2.5D =="
run "c=2 P=8"              -n 8 $BIN --M 61 --N 53 --K 47 --engine summa25d --c 2 --b 8 --verify ref --reps 1 --warmup 0 --quiet
run "c=4 P=8"              -n 8 $BIN --M 40 --N 36 --K 32 --engine summa25d --c 4 --b 4 --verify ref --reps 1 --warmup 0 --quiet
run "c=8 P=8 (pure 1D-k)"  -n 8 $BIN --M 40 --N 36 --K 32 --engine summa25d --c 8 --b 4 --verify ref --reps 1 --warmup 0 --quiet
GEMM2D_FAKE_PPN=2 run "c=2 + shm + beta" -n 8 $BIN --M 41 --N 37 --K 33 --engine summa25d --c 2 --bcast shm --b 4 --alpha 1.5 --beta 2.0 --verify ref --reps 1 --warmup 0 --quiet

echo "== baselines =="
run "cannon P=4"           -n 4 $BIN --M 40 --N 36 --K 28 --engine cannon --verify ref --reps 1 --warmup 0 --quiet
run "cannon P=9"           -n 9 $BIN --M 45 --N 36 --K 27 --engine cannon --verify ref --reps 1 --warmup 0 --quiet
run "naive1d P=4"          -n 4 $BIN --M 40 --N 36 --K 28 --engine naive1d --verify ref --reps 1 --warmup 0 --quiet

echo "== Freivalds at a size where the exact reference is too slow =="
run "freivalds summa"      -n 4 $BIN --M 400 --N 380 --K 360 --b 32 --verify freivalds --trials 3 --reps 1 --warmup 0 --quiet
run "freivalds 2.5D"       -n 8 $BIN --M 400 --N 380 --K 360 --engine summa25d --c 2 --b 32 --verify freivalds --trials 3 --reps 1 --warmup 0 --quiet
run "freivalds beta!=0"    -n 4 $BIN --M 400 --N 380 --K 360 --b 32 --alpha 1.5 --beta -0.5 --verify freivalds --trials 3 --reps 1 --warmup 0 --quiet

echo "== naive kernel path =="
run "kernel=naive"         -n 4 $BIN --M 61 --N 53 --K 47 --kernel naive --b 8 --verify ref --reps 1 --warmup 0 --quiet

echo "== planner (the originality claim -- was previously untested) =="
run "--plan"               -n 8 $BIN --M 61 --N 53 --K 47 --engine summa25d --plan --verify ref --reps 1 --warmup 0 --quiet
run "--plan --calibrate"   -n 8 $BIN --M 90 --N 80 --K 300 --engine summa25d --plan --calibrate --verify freivalds --trials 3 --reps 1 --warmup 0 --quiet
run "--plan on summa"      -n 4 $BIN --M 61 --N 53 --K 47 --engine summa --plan --verify ref --reps 1 --warmup 0 --quiet

echo "== naive1d honours alpha/beta =="
run "naive1d alpha/beta"   -n 4 $BIN --M 40 --N 36 --K 28 --engine naive1d --alpha 1.5 --beta -2.0 --verify ref --reps 1 --warmup 0 --quiet

echo
echo "passed: $pass   failed: $fail"
[ "$fail" -eq 0 ]
