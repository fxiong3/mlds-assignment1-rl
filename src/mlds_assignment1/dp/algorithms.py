"""Finite- and infinite-horizon dynamic programming algorithms."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from mlds_assignment1.dp.ferry_mdp import FerryMDP

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FiniteHorizonResult:
    """Values and policies returned by finite-horizon backward induction."""

    values_by_time: FloatArray
    policies_by_time: IntArray

    @property
    def initial_values(self) -> FloatArray:
        return self.values_by_time[0]

    @property
    def initial_policy(self) -> IntArray:
        return self.policies_by_time[0]


@dataclass(frozen=True)
class InfiniteHorizonResult:
    """Result and convergence evidence for an infinite-horizon solver."""

    values: FloatArray
    policy: IntArray
    iterations: int
    bellman_residual: float
    residual_history: tuple[float, ...]


def _action_values(mdp: FerryMDP, values: FloatArray) -> FloatArray:
    transitions = mdp.transition_matrices()
    continuation = np.einsum("asj,j->sa", transitions, values)
    return mdp.cost_matrix() + mdp.gamma * continuation


def _bellman_residual(mdp: FerryMDP, values: FloatArray) -> float:
    optimal_update = np.min(_action_values(mdp, values), axis=1)
    return float(np.max(np.abs(optimal_update - values)))


def finite_horizon_backward_induction(
    mdp: FerryMDP,
    horizon: int = 500,
    terminal_values: FloatArray | None = None,
) -> FiniteHorizonResult:
    """Solve the finite-horizon MDP by exact backward induction."""

    if horizon <= 0:
        raise ValueError("horizon must be positive")
    state_count = mdp.max_queue + 1
    values = np.zeros((horizon + 1, state_count), dtype=np.float64)
    policies = np.zeros((horizon, state_count), dtype=np.int64)
    if terminal_values is not None:
        terminal_values = np.asarray(terminal_values, dtype=np.float64)
        if terminal_values.shape != (state_count,):
            raise ValueError(f"terminal_values must have shape ({state_count},)")
        values[horizon] = terminal_values

    for time_index in range(horizon - 1, -1, -1):
        q_values = _action_values(mdp, values[time_index + 1])
        policies[time_index] = np.argmin(q_values, axis=1)
        values[time_index] = np.min(q_values, axis=1)

    return FiniteHorizonResult(values, policies)


def value_iteration(
    mdp: FerryMDP,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
    initial_values: FloatArray | None = None,
) -> InfiniteHorizonResult:
    """Solve the discounted infinite-horizon MDP with value iteration."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    state_count = mdp.max_queue + 1
    if initial_values is None:
        values = np.zeros(state_count, dtype=np.float64)
    else:
        values = np.asarray(initial_values, dtype=np.float64).copy()
        if values.shape != (state_count,):
            raise ValueError(f"initial_values must have shape ({state_count},)")

    residual_history: list[float] = []
    completed_iterations = 0
    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        q_values = _action_values(mdp, values)
        updated_values = np.min(q_values, axis=1)
        residual = float(np.max(np.abs(updated_values - values)))
        residual_history.append(residual)
        values = updated_values
        if residual < tolerance:
            break
    else:
        raise RuntimeError("value iteration did not converge")

    q_values = _action_values(mdp, values)
    policy = np.argmin(q_values, axis=1).astype(np.int64)
    return InfiniteHorizonResult(
        values=values,
        policy=policy,
        iterations=completed_iterations,
        bellman_residual=_bellman_residual(mdp, values),
        residual_history=tuple(residual_history),
    )


def policy_iteration(
    mdp: FerryMDP,
    tolerance: float = 1e-10,
    max_iterations: int = 1_000,
    initial_policy: IntArray | None = None,
) -> InfiniteHorizonResult:
    """Solve the discounted MDP with exact policy evaluation."""

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    state_count = mdp.max_queue + 1
    if initial_policy is None:
        policy = np.zeros(state_count, dtype=np.int64)
    else:
        policy = np.asarray(initial_policy, dtype=np.int64).copy()
        if policy.shape != (state_count,):
            raise ValueError(f"initial_policy must have shape ({state_count},)")
        if not np.isin(policy, mdp.actions).all():
            raise ValueError("initial_policy contains an invalid action")

    transitions = mdp.transition_matrices()
    costs = mdp.cost_matrix()
    identity = np.eye(state_count)
    residual_history: list[float] = []
    completed_iterations = 0

    for iteration in range(1, max_iterations + 1):
        completed_iterations = iteration
        state_indices = np.arange(state_count)
        policy_transition = transitions[policy, state_indices]
        policy_cost = costs[state_indices, policy]
        values = np.linalg.solve(identity - mdp.gamma * policy_transition, policy_cost)

        q_values = _action_values(mdp, values)
        improved_policy = np.argmin(q_values, axis=1).astype(np.int64)
        residual_history.append(_bellman_residual(mdp, values))
        if np.array_equal(improved_policy, policy):
            policy = improved_policy
            break
        policy = improved_policy
    else:
        raise RuntimeError("policy iteration did not converge")

    return InfiniteHorizonResult(
        values=values,
        policy=policy,
        iterations=completed_iterations,
        bellman_residual=_bellman_residual(mdp, values),
        residual_history=tuple(residual_history),
    )

