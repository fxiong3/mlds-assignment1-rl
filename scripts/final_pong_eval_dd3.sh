#!/usr/bin/env bash
# Run on deepdish3 (CPU-only torch in ~/torch_cpu_310): at START_UTC evaluate the
# newest checkpoint of each Pong run over 500 episodes.
set -uo pipefail
cd "$(dirname "$0")/.."
START_UTC="${1:-02:46}"
export PYTHONPATH="$HOME/torch_cpu_310"
until [ "$(date -u +%H:%M)" = "$START_UTC" ]; do sleep 15; done
sleep 20
latest() { ls checkpoints/"$1"-episode-*.pt | sort -t- -k3 -n | tail -1; }
NB=$(latest pong_v5_no_baseline_seed_1)
BL=$(latest pong_v5_baseline_seed_2)
echo "no-baseline: $NB  baseline: $BL  start: $(date -u)"
mkdir -p results/final
.venv/bin/python scripts/parallel_evaluate.py --checkpoint "$NB" --workers 18 --output-dir results/final > logs/final_eval_no_baseline.log 2>&1 &
.venv/bin/python scripts/parallel_evaluate.py --checkpoint "$BL" --workers 8 --output-dir results/final > logs/final_eval_baseline.log 2>&1 &
wait
echo "done: $(date -u)"
