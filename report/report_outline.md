# Assignment 1 Report Outline

## Repository and reproduction information

- GitHub username: `fxiong3`
- Repository: `mlds-assignment1-rl`
- Repository URL: to be inserted after repository creation
- Exact installation, training, evaluation, and plotting commands
- Hardware and software versions

## 1. Shuttle-dispatch MDP

### 1.1 Model and assumptions

Define state, actions, costs, transition probabilities, event order, queue cap, overflow handling, discount factor, and terminal value.

### 1.2 Finite-horizon backward induction

Describe the Bellman recursion for `T=500` and include the time-zero optimal value plot.

### 1.3 Infinite-horizon value iteration

State the stopping tolerance, number of iterations, final Bellman residual, and optimal-value plot.

### 1.4 Policy iteration

Describe exact policy evaluation and policy improvement. Include the optimal-policy plot and report the first dispatch state.

### 1.5 Edge cases and validation

Report probability conservation, capped-state aggregation, boundary states, small-problem checks, VI/PI agreement, finite/infinite comparison, Monte Carlo validation, and parameter sensitivity.

## 2. Basic policy gradient

### 2.1 Algorithm

Describe REINFORCE, discounted reward-to-go, neural networks, Adam settings, return normalization, seeds, checkpoint selection, and evaluation protocol.

### 2.2 CartPole-v1

Include the training curve with 100-episode moving average, the 500-episode histogram, mean, and standard deviation.

### 2.3 Pong-v5

Explain frame preprocessing, two-frame input, and the restriction to actions 2 and 3. Include the required plots and statistics.

### 2.4 Edge cases and validation

Include random-policy comparisons, first-ten-step traces, API termination handling, probability checks, preprocessing inspection, action checks, numerical stability, and multiple-seed results.

## 3. Policy gradient with baseline

Explain the learned state-value baseline, value loss, detached advantage, and fair comparison protocol. Repeat all required experiments and compare reward variance, advantage variance, learning speed, cross-seed stability, and final evaluation performance.

## 4. Additional three-month validation plan

Discuss larger seed counts, hyperparameter sweeps, bootstrap confidence intervals, significance tests, alternative baselines, actor-critic/GAE comparisons, architecture ablations, out-of-distribution initial states, independent-machine reproduction, and longer stability runs.

## 5. Conclusion

Summarize the DP solution, learned-policy performance, baseline effect, limitations, and strongest correctness evidence.

