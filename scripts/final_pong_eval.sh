#!/usr/bin/env bash
# Deadline-driven Pong evaluation: at a fixed UTC time, stop the baseline run,
# take the newest checkpoint of each Pong run and evaluate 500 episodes in parallel.
set -uo pipefail
cd "$(dirname "$0")/.."
START_UTC="${1:-03:15}"
until [ "$(date -u +%H:%M)" = "$START_UTC" ]; do sleep 20; done
if [ -f logs/pong_baseline_seed2.pid ]; then kill "$(cat logs/pong_baseline_seed2.pid)" 2>/dev/null; fi
latest() { ls checkpoints/"$1"-episode-*.pt | sort -t- -k3 -n | tail -1; }
NB=$(latest pong_v5_no_baseline_seed_1)
BL=$(latest pong_v5_baseline_seed_2)
echo "no-baseline: $NB  baseline: $BL  start: $(date -u)"
.venv/bin/python scripts/parallel_evaluate.py --checkpoint "$NB" --workers 14 > logs/final_eval_no_baseline.log 2>&1 &
.venv/bin/python scripts/parallel_evaluate.py --checkpoint "$BL" --workers 10 > logs/final_eval_baseline.log 2>&1 &
wait
echo "done: $(date -u)"
