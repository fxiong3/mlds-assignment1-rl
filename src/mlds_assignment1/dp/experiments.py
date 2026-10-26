"""Reproducible dynamic-programming experiments, plots, and validation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from mlds_assignment1.dp.algorithms import (
    finite_horizon_backward_induction,
    policy_iteration,
    value_iteration,
)
from mlds_assignment1.dp.ferry_mdp import FerryMDP

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def simulate_policy(
    mdp: FerryMDP,
    policy: Sequence[int],
    *,
    initial_state: int,
    episodes: int,
    steps: int,
    seed: int,
) -> NDArray[np.float64]:
    """Estimate discounted costs for a fixed stationary policy."""

    if len(policy) != mdp.max_queue + 1:
        raise ValueError("policy length must equal the number of states")
    if not 0 <= initial_state <= mdp.max_queue:
        raise ValueError("initial_state is outside the state space")
    if episodes <= 0 or steps <= 0:
        raise ValueError("episodes and steps must be positive")
    rng = np.random.default_rng(seed)
    arrival_probabilities = np.asarray(mdp.arrival_probabilities)
    totals = np.zeros(episodes, dtype=np.float64)

    for episode in range(episodes):
        state = initial_state
        discount = 1.0
        for _ in range(steps):
            action = int(policy[state])
            totals[episode] += discount * mdp.immediate_cost(state, action)
            remaining = mdp.remaining_passengers(state, action)
            arrival = int(rng.choice(mdp.arrivals, p=arrival_probabilities))
            state = min(mdp.max_queue, remaining + arrival)
            discount *= mdp.gamma
    return totals


def run_dp_analysis(
    *,
    output_dir: str | Path,
    horizon: int = 500,
    simulation_episodes: int = 1_000,
    simulation_steps: int = 500,
    seed: int = 0,
    max_queue: int = 200,
) -> dict[str, Any]:
    """Run all required DP algorithms and save figures plus validation evidence."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    mdp = FerryMDP(max_queue=max_queue)
    finite = finite_horizon_backward_induction(mdp, horizon=horizon)
    value_result = value_iteration(mdp)
    policy_result = policy_iteration(mdp)
    states = np.arange(max_queue + 1)

    frame = pd.DataFrame(
        {
            "state": states,
            "finite_horizon_value_t0": finite.initial_values,
            "finite_horizon_policy_t0": finite.initial_policy,
            "value_iteration_value": value_result.values,
            "value_iteration_policy": value_result.policy,
            "policy_iteration_value": policy_result.values,
            "policy_iteration_policy": policy_result.policy,
        }
    )
    frame.to_csv(output_path / "dp_values.csv", index=False)

    _plot_value(
        states,
        finite.initial_values,
        output_path / "finite_horizon_value.png",
        title=f"Finite-Horizon Optimal Value Function (T={horizon})",
    )
    _plot_value(
        states,
        value_result.values,
        output_path / "infinite_horizon_value.png",
        title="Infinite-Horizon Optimal Value Function",
    )
    _plot_policy(
        states,
        policy_result.policy,
        output_path / "optimal_policy.png",
    )

    initial_states = sorted({0, max_queue // 4, max_queue // 2, 3 * max_queue // 4, max_queue})
    simulation_validation: dict[str, dict[str, float]] = {}
    for state in initial_states:
        sampled_costs = simulate_policy(
            mdp,
            value_result.policy,
            initial_state=state,
            episodes=simulation_episodes,
            steps=simulation_steps,
            seed=seed + state,
        )
        empirical_mean = float(sampled_costs.mean())
        empirical_standard_error = float(
            sampled_costs.std(ddof=1) / np.sqrt(simulation_episodes)
        ) if simulation_episodes > 1 else 0.0
        predicted = float(value_result.values[state])
        simulation_validation[str(state)] = {
            "predicted_value": predicted,
            "empirical_mean": empirical_mean,
            "empirical_standard_error": empirical_standard_error,
            "absolute_error": abs(predicted - empirical_mean),
        }

    dispatch_states = np.flatnonzero(policy_result.policy == 1)
    metrics: dict[str, Any] = {
        "assumptions": {
            "event_order": "dispatch, post-dispatch holding cost, then arrivals",
            "queue_cap": max_queue,
            "overflow_handling": "aggregate at capped state",
            "terminal_value": 0.0,
            "objective": "minimize discounted cumulative cost",
        },
        "finite_horizon": horizon,
        "value_iteration_iterations": value_result.iterations,
        "policy_iteration_iterations": policy_result.iterations,
        "vi_bellman_residual": value_result.bellman_residual,
        "pi_bellman_residual": policy_result.bellman_residual,
        "vi_pi_max_value_difference": float(
            np.max(np.abs(value_result.values - policy_result.values))
        ),
        "vi_pi_policy_disagreements": int(
            np.count_nonzero(value_result.policy != policy_result.policy)
        ),
        "finite_infinite_max_value_difference": float(
            np.max(np.abs(finite.initial_values - value_result.values))
        ),
        "first_dispatch_state": int(dispatch_states[0]) if dispatch_states.size else None,
        "simulation_validation": simulation_validation,
    }
    (output_path / "dp_validation.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    return metrics


def _plot_value(
    states: NDArray[np.int64],
    values: NDArray[np.float64],
    path: Path,
    *,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(states, values, linewidth=2)
    axis.set_title(title)
    axis.set_xlabel("Number of customers waiting")
    axis.set_ylabel("Optimal expected discounted cost")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def _plot_policy(
    states: NDArray[np.int64],
    policy: NDArray[np.int64],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.step(states, policy, where="post", linewidth=2)
    axis.set_title("Optimal Shuttle Dispatch Policy")
    axis.set_xlabel("Number of customers waiting")
    axis.set_ylabel("Action")
    axis.set_yticks([0, 1], labels=["Wait", "Dispatch"])
    axis.set_ylim(-0.2, 1.2)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)

