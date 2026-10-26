# MLDS Assignment 1: Dynamic Programming and Policy Gradient

This repository contains reproducible implementations and experiments for:

- finite-horizon backward induction, value iteration, and policy iteration for the shuttle-dispatch MDP;
- REINFORCE on `CartPole-v1` and `Pong-v5`;
- REINFORCE with a learned state-value baseline;
- quantitative validation, plots, checkpoints, and 500-episode evaluation.

## Modeling assumptions

The shuttle state is the number of waiting customers, capped at 200. At each decision epoch, the action is either wait or dispatch. A dispatch removes up to 15 passengers. Holding cost is charged after dispatch and before new arrivals. Arrivals are uniformly distributed over `{1, 2, 3, 4, 5}`. Overflow probability is aggregated at state 200. The terminal value for the finite-horizon problem is zero, and all algorithms minimize discounted cumulative cost.

## Installation

Python 3.11 or 3.12 is required.

```bash
uv sync --extra dev
```

For Atari, install the legal Pong ROM through the mechanism provided by the execution environment. ROM files are not included in this repository.

## Tests

```bash
.venv/bin/pytest
.venv/bin/ruff check .
```

The tests cover boundary states, probability conservation, Bellman consistency, agreement between value and policy iteration, Pong preprocessing, action restrictions, discounted returns, baseline gradient separation, and short end-to-end CartPole runs.

## Dynamic programming experiments

```bash
./scripts/run_dp.sh
```

This writes the required value and policy plots, raw state-level results, Bellman residuals, value/policy agreement statistics, and Monte Carlo validation to `results/dp/`.

## Policy-gradient training

Run one experiment:

```bash
.venv/bin/mlds-train --config configs/cartpole_no_baseline.yaml --seed 0
.venv/bin/mlds-train --config configs/cartpole_baseline.yaml --seed 0
.venv/bin/mlds-train --config configs/pong_no_baseline.yaml --seed 0
.venv/bin/mlds-train --config configs/pong_baseline.yaml --seed 0
```

The Pong policy maps its two outputs only to environment actions 2 (`RIGHT`) and 3 (`LEFT`). The image preprocessing follows the assignment specification: crop rows 35–194, downsample by two, remove background values 144 and 109, and binarize remaining pixels.

On the DeepDish GPU environment, after installation and ROM setup:

```bash
./scripts/train_deepdish.sh
```

The script runs the four experiment configurations with seeds 0, 1, and 2. Checkpoints are written regularly so long Pong runs can be preserved.

## Evaluation

The required evaluation uses 500 episodes and samples from the learned stochastic policy by default:

```bash
.venv/bin/mlds-evaluate \
  --checkpoint checkpoints/CHECKPOINT.pt \
  --episodes 500 \
  --output-dir results/raw
```

Use `--deterministic` only for an additional argmax-policy comparison. The primary report results should use one protocol consistently.

## Plotting

```bash
.venv/bin/mlds-plot training \
  --input results/raw/TRAINING.csv \
  --output results/figures/training.png \
  --title "CartPole-v1 REINFORCE"

.venv/bin/mlds-plot evaluation \
  --input results/raw/EVALUATION.csv \
  --output results/figures/evaluation.png \
  --title "CartPole-v1: 500-Episode Evaluation"
```

Training plots contain both raw episode reward and the 100-episode moving average. Evaluation histograms display the sample count, mean, and population standard deviation.

## Reproducibility notes

- Each run records its full configuration and seed.
- Training metrics are saved as CSV rather than copied manually into figures.
- The first ten transitions of the first episode are saved as an audit trace.
- Baseline and no-baseline comparisons use the same seeds, training budgets, environment versions, and evaluation protocol.
- Large checkpoints, ROMs, and raw training logs are intentionally excluded from Git.

