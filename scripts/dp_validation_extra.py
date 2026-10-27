"""Extra behavioural checks for the shuttle MDP used in the report.

Writes results/dp/dp_validation_extra.json and the figures in report/figures/.
Run: PYTHONPATH=src python scripts/dp_validation_extra.py
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

from mlds_assignment1.dp.algorithms import (  # noqa: E402
    finite_horizon_backward_induction,
    policy_iteration,
    value_iteration,
)
from mlds_assignment1.dp.ferry_mdp import FerryMDP  # noqa: E402

FIGURES = Path("report/figures")
OUTPUT = Path("results/dp/dp_validation_extra.json")


def threshold(policy: np.ndarray) -> int | None:
    dispatch = np.flatnonzero(policy == 1)
    return int(dispatch[0]) if dispatch.size else None


def is_threshold_policy(policy: np.ndarray) -> bool:
    return bool(np.all(np.diff(policy) >= 0))


def sensitivity() -> list[dict[str, float | int | bool | None]]:
    base = {"capacity": 15, "dispatch_cost": 100.0, "holding_cost": 2.0, "gamma": 0.95}
    grid = {
        "dispatch_cost": [50.0, 100.0, 200.0, 400.0],
        "holding_cost": [1.0, 2.0, 4.0],
        "capacity": [10, 15, 20],
        "gamma": [0.5, 0.9, 0.95, 0.99],
    }
    rows = []
    for name, values in grid.items():
        for value in values:
            params = {**base, name: value}
            result = policy_iteration(FerryMDP(**params))
            rows.append(
                {
                    "parameter": name,
                    "value": value,
                    "threshold": threshold(result.policy),
                    "threshold_type": is_threshold_policy(result.policy),
                }
            )
    return rows


def deterministic_brute_force(horizon: int = 10) -> dict[str, float]:
    """Arrivals fixed at 3: compare backward induction with enumeration of all 2^T plans."""

    mdp = FerryMDP(arrivals=(3,), max_queue=60)
    induction = finite_horizon_backward_induction(mdp, horizon=horizon).initial_values
    worst_gap = 0.0
    for start in range(0, 41):
        best = np.inf
        for plan in itertools.product((0, 1), repeat=horizon):
            state, total = start, 0.0
            for step, action in enumerate(plan):
                total += mdp.gamma**step * mdp.immediate_cost(state, action)
                state = min(mdp.max_queue, mdp.remaining_passengers(state, action) + 3)
            best = min(best, total)
        worst_gap = max(worst_gap, abs(best - induction[start]))
    one_step = finite_horizon_backward_induction(FerryMDP(), horizon=1)
    hand = np.array([min(2.0 * s, 100.0 + 2.0 * max(0, s - 15)) for s in range(201)])
    return {
        "horizon": horizon,
        "max_abs_gap_enumeration_vs_induction": worst_gap,
        "one_step_max_abs_gap_vs_hand_formula": float(np.max(np.abs(one_step.initial_values - hand))),
        "one_step_dispatch_states": int(one_step.initial_policy.sum()),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    mdp = FerryMDP()
    finite = finite_horizon_backward_induction(mdp, horizon=500)
    vi = value_iteration(mdp)
    pi = policy_iteration(mdp)
    states = np.arange(mdp.max_queue + 1)

    horizons = [1, 5, 10, 25, 50, 100, 200, 300, 500]
    horizon_gap = [
        float(np.max(np.abs(finite_horizon_backward_induction(mdp, h).initial_values - vi.values)))
        for h in horizons
    ]

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(states, finite.initial_values, lw=2, label="$V_0$, backward induction (T=500)")
    axes[0].set_title("(a) Enumeration, T=500: value at t=0")
    axes[1].plot(states, vi.values, lw=2, label="$V^*$, value iteration")
    axes[1].plot(states, pi.values, "--", lw=1.5, label="$V^*$, policy iteration")
    axes[1].set_title("(b) Value iteration, T=$\\infty$")
    for axis in axes[:2]:
        axis.set_xlabel("Customers waiting $s$")
        axis.set_ylabel("Expected discounted cost")
    axes[2].step(states, pi.policy, where="post", lw=2, label="PI policy")
    axes[2].step(states, vi.policy, where="post", ls=":", lw=2, label="VI greedy policy")
    axes[2].axvline(threshold(pi.policy), color="gray", lw=0.8, ls="--",
                    label=f"threshold s={threshold(pi.policy)}")
    axes[2].set_yticks([0, 1], labels=["Wait", "Dispatch"])
    axes[2].set_ylim(-0.2, 1.3)
    axes[2].set_xlabel("Customers waiting $s$")
    axes[2].set_ylabel("Optimal action")
    axes[2].set_title("(c) Policy iteration: optimal policy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    figure.savefig(FIGURES / "dp_main.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    history = np.array(vi.residual_history)
    k = np.arange(1, len(history) + 1)
    axes[0].semilogy(k, history, lw=2, label="VI residual $\\|V_{k}-V_{k-1}\\|_\\infty$")
    axes[0].semilogy(k, history[0] * mdp.gamma ** (k - 1), "--", lw=1.2, label="$\\gamma^{k}$ contraction rate")
    axes[0].set_xlabel("Iteration $k$")
    axes[0].set_ylabel("Sup-norm change")
    axes[0].set_title("Value-iteration convergence")
    axes[1].semilogy(horizons, horizon_gap, "o-", lw=2, label="$\\|V_0^{(T)}-V^*\\|_\\infty$")
    axes[1].set_xlabel("Horizon $T$")
    axes[1].set_ylabel("Sup-norm gap")
    axes[1].set_title("Finite horizon approaches $V^*$")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "dp_convergence.png", dpi=180)
    plt.close(figure)

    report = {
        "threshold_base": threshold(pi.policy),
        "finite_horizon_t0_threshold": threshold(finite.initial_policy),
        "vi_iterations": vi.iterations,
        "pi_iterations": pi.iterations,
        "pi_residual_history": list(pi.residual_history),
        "horizon_gap": dict(zip(map(str, horizons), horizon_gap, strict=True)),
        "sensitivity": sensitivity(),
        "deterministic_check": deterministic_brute_force(),
        "value_at": {str(s): float(vi.values[s]) for s in (0, 13, 50, 100, 200)},
    }
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
