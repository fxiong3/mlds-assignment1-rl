from __future__ import annotations

import numpy as np

from mlds_assignment1.dp.algorithms import (
    finite_horizon_backward_induction,
    policy_iteration,
    value_iteration,
)
from mlds_assignment1.dp.ferry_mdp import FerryMDP


def test_one_step_finite_horizon_equals_minimum_immediate_cost() -> None:
    mdp = FerryMDP(max_queue=20)
    result = finite_horizon_backward_induction(mdp, horizon=1)

    expected = np.minimum(
        [mdp.immediate_cost(state, 0) for state in mdp.states],
        [mdp.immediate_cost(state, 1) for state in mdp.states],
    )
    np.testing.assert_allclose(result.initial_values, expected)


def test_empty_queue_does_not_dispatch() -> None:
    mdp = FerryMDP(max_queue=40)
    finite = finite_horizon_backward_induction(mdp, horizon=20)
    infinite = value_iteration(mdp, tolerance=1e-11)

    assert finite.initial_policy[0] == 0
    assert infinite.policy[0] == 0


def test_value_and_policy_iteration_agree() -> None:
    mdp = FerryMDP(max_queue=60)
    vi = value_iteration(mdp, tolerance=1e-11)
    pi = policy_iteration(mdp, tolerance=1e-11)

    np.testing.assert_allclose(vi.values, pi.values, atol=1e-7)
    np.testing.assert_array_equal(vi.policy, pi.policy)
    assert vi.bellman_residual < 1e-9
    assert pi.bellman_residual < 1e-9


def test_value_iteration_is_independent_of_initial_values() -> None:
    mdp = FerryMDP(max_queue=35)
    zeros = value_iteration(mdp, initial_values=np.zeros(36), tolerance=1e-10)
    high = value_iteration(mdp, initial_values=np.full(36, 1_000.0), tolerance=1e-10)

    np.testing.assert_allclose(zeros.values, high.values, atol=1e-7)
    np.testing.assert_array_equal(zeros.policy, high.policy)


def test_optimal_cost_is_non_decreasing_with_queue_length() -> None:
    result = value_iteration(FerryMDP(max_queue=80), tolerance=1e-10)
    assert np.all(np.diff(result.values) >= -1e-9)

