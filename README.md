# MLDS Assignment 1: Dynamic Programming and Policy Gradient

This repository contains reproducible implementations and experiments for:

- finite-horizon backward induction, value iteration, and policy iteration for the shuttle-dispatch MDP;
- REINFORCE on `CartPole-v1` and `Pong-v5`;
- REINFORCE with a learned state-value baseline;
- quantitative validation, plots, checkpoints, and 500-episode evaluation.

## Modeling assumptions

The shuttle state is the number of waiting customers, capped at 200. At each decision epoch, the action is either wait or dispatch. A dispatch removes up to 15 passengers. Holding cost is charged after dispatch and before new arrivals. Arrivals are uniformly distributed over `{1, 2, 3, 4, 5}`. Overflow probability is aggregated at state 200. The terminal value for the finite-horizon problem is zero, and all algorithms minimize discounted cumulative cost.

## Installation

Python 3.10, 3.11, or 3.12 is required.

```bash
uv sync --extra dev
```

For Atari, install the legal Pong ROM through the mechanism provided by the execution environment. With the declared dependencies, run `.venv/bin/AutoROM --accept-license`. ROM files are not included in this repository.

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

The script runs the four experiment configurations with seeds 0, 1, and 2. Checkpoints include model, optimizer, RNG, episode, and metric state. Resume an interrupted run with `mlds-train --config CONFIG --episodes TARGET --resume CHECKPOINT`. Pong observations are stored on CPU and recomputed in bounded batches during optimization so an episode does not retain an unbounded GPU autograd graph.

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

## Validation experiments

Compare each learned policy against a uniform random policy using the same evaluation budget and episode cap:

```bash
.venv/bin/mlds-validate random --environment CartPole-v1 --episodes 500 --seed 0 --max-steps 500
.venv/bin/mlds-validate random --environment Pong-v5 --episodes 500 --seed 0 --max-steps 10000
```

Aggregate independent trained-policy evaluation files across seeds 0, 1, and 2:

```bash
.venv/bin/mlds-validate aggregate \
  --inputs results/raw/SEED_0.csv results/raw/SEED_1.csv results/raw/SEED_2.csv \
  --output results/validation/aggregate.json
```

The random-policy comparison tests whether learning improves over an untrained action rule. The multi-seed aggregate reports pooled performance, variation in seed means, and a bootstrap confidence interval over independent training seeds. The per-run trace records state, action probabilities, chosen action, reward, value estimate, and discounted return for the first ten steps. Training metrics also record raw return/advantage variance and gradient norms for the baseline comparison and numerical-stability checks.

## Reproducibility notes

- Each run records its full configuration and seed. This provides seed-controlled statistical reproducibility; exact CUDA bitwise reproducibility is not claimed.
- Training metrics are saved as CSV rather than copied manually into figures.
- The first ten transitions of the first episode are saved as an audit trace.
- Basic REINFORCE uses the uncentered discounted return directly. Baseline experiments use the learned value estimate and record variance before optional normalization.
- Baseline and no-baseline comparisons use the same seeds, training budgets, environment versions, and evaluation protocol.
- Large checkpoints, ROMs, and raw training logs are intentionally excluded from Git.

