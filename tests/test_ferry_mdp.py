from __future__ import annotations

import numpy as np
import pytest

from mlds_assignment1.dp.ferry_mdp import FerryMDP


@pytest.fixture
def mdp() -> FerryMDP:
    return FerryMDP()


def test_transition_probabilities_sum_to_one(mdp: FerryMDP) -> None:
    for state in range(mdp.max_queue + 1):
        for action in mdp.actions:
            probabilities = mdp.transition_probabilities(state, action)
            assert sum(probabilities.values()) == pytest.approx(1.0)


def test_queue_cap_aggregates_probability_mass(mdp: FerryMDP) -> None:
    assert mdp.transition_probabilities(200, 0) == {200: pytest.approx(1.0)}
    assert mdp.transition_probabilities(199, 0) == {
        200: pytest.approx(1.0),
    }


@pytest.mark.parametrize(
    ("state", "action", "expected"),
    [
        (0, 0, 0.0),
        (0, 1, 100.0),
        (14, 1, 100.0),
        (15, 1, 100.0),
        (16, 1, 102.0),
        (16, 0, 32.0),
    ],
)
def test_cost_is_dispatch_plus_post_dispatch_holding_cost(
    mdp: FerryMDP,
    state: int,
    action: int,
    expected: float,
) -> None:
    assert mdp.immediate_cost(state, action) == expected


def test_transition_matrix_rows_are_stochastic(mdp: FerryMDP) -> None:
    matrices = mdp.transition_matrices()
    assert matrices.shape == (2, 201, 201)
    np.testing.assert_allclose(matrices.sum(axis=2), 1.0)


@pytest.mark.parametrize("state", [-1, 201])
def test_invalid_state_rejected(mdp: FerryMDP, state: int) -> None:
    with pytest.raises(ValueError):
        mdp.transition_probabilities(state, 0)


@pytest.mark.parametrize("action", [-1, 2])
def test_invalid_action_rejected(mdp: FerryMDP, action: int) -> None:
    with pytest.raises(ValueError):
        mdp.transition_probabilities(0, action)

