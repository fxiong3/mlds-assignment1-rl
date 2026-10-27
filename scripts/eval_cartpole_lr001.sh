#!/usr/bin/env bash
cd "$(dirname "$0")/.."
mkdir -p results/lr001
for s in 0 1 2; do
  .venv/bin/python scripts/parallel_evaluate.py --checkpoint checkpoints/cartpole_v1_no_baseline_seed_$s-episode-2000.pt --workers 4 --output-dir results/lr001 &
done
wait
for f in results/raw/cartpole_v1_no_baseline_seed_* results/lr001/cartpole_v1_no_baseline_seed_*; do
  base=$(basename "$f"); cp "$f" "results/lr001/${base/no_baseline/no_baseline_lr0.001}"
done
echo finished
