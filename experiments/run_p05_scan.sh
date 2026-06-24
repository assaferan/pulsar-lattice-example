#!/usr/bin/env bash
# Parallel p=0.5 detection scan: one isolated process per n (so a C-level sieve
# abort kills only that point), CONC at a time. Each process uses p05_lava.py's
# THREADS sieve threads, so keep CONC*THREADS <= cores.
#
# Usage:  run_p05_scan.sh <model A|B> <n1> <n2> ...
# Example: nohup bash experiments/run_p05_scan.sh A 130 140 150 160 170 180 \
#              > experiments/p05_scan_A.log 2>&1 &
set -u
CONC="${CONC:-8}"
ROOT=/scratch/assaferan/GitHub
model="$1"; shift
source /scratch/assaferan/miniforge3/etc/profile.d/conda.sh
conda activate g6k
export PYTHONPATH="$ROOT/g6k:$ROOT/pulsar-lattice-example"
cd "$ROOT/pulsar-lattice-example"
printf '%s\n' "$@" | xargs -P "$CONC" -I{} \
  bash -c "python experiments/p05_lava.py $model 0.5 {} > experiments/p05_${model}_{}.log 2>&1; echo done {}"
echo "=== scan complete (model $model) ==="
cat experiments/p05_${model}_*.txt 2>/dev/null | sort -t= -k2 -n
