#!/usr/bin/env bash
set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "A CUDA GPU is required for the DeepDish Pong runs." >&2
  exit 1
fi

for config in configs/cartpole_no_baseline.yaml \
              configs/cartpole_baseline.yaml \
              configs/pong_no_baseline.yaml \
              configs/pong_baseline.yaml; do
  for seed in 0 1 2; do
    .venv/bin/mlds-train --config "$config" --seed "$seed"
  done
done

