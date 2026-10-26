#!/usr/bin/env bash
set -euo pipefail

.venv/bin/mlds-dp \
  --output-dir results/dp \
  --horizon 500 \
  --simulation-episodes 1000 \
  --simulation-steps 500 \
  --seed 0

