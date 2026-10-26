"""Shuttle-dispatch Markov decision process used in Assignment 1."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class FerryMDP:
    """Finite-state shuttle-dispatch MDP.

    At each decision epoch the agent first decides whether to dispatch. A
    dispatched shuttle removes up to ``capacity`` passengers. Holding cost is
    charged on the passengers left behind, and arrivals then join the queue.
    Queue lengths above ``max_queue`` are aggregated into the capped state.
    """

    capacity: int = 15
    dispatch_cost: float = 100.0
    holding_cost: float = 2.0
    arrivals: tuple[int, ...] = (1, 2, 3, 4, 5)
    arrival_probabilities: tuple[float, ...] | None = None
    max_queue: int = 200
    gamma: float = 0.95

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        if self.max_queue < 0:
            raise ValueError("max_queue must be non-negative")
        if not self.arrivals:
            raise ValueError("arrivals must not be empty")
        if any(arrival < 0 for arrival in self.arrivals):
            raise ValueError("arrivals must be non-negative")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must be in [0, 1)")

        probabilities = self.arrival_probabilities
        if probabilities is None:
            probabilities = tuple(1.0 / len(self.arrivals) for _ in self.arrivals)
            object.__setattr__(self, "arrival_probabilities", probabilities)
        if len(probabilities) != len(self.arrivals):
            raise ValueError("arrival probabilities must align with arrivals")
        if any(probability < 0.0 for probability in probabilities):
            raise ValueError("arrival probabilities must be non-negative")
        if not np.isclose(sum(probabilities), 1.0):
            raise ValueError("arrival probabilities must sum to one")

    @property
    def states(self) -> range:
        """Return all queue-length states."""

        return range(self.max_queue + 1)

    @property
    def actions(self) -> tuple[int, int]:
        """Return 0 (wait) and 1 (dispatch)."""

        return (0, 1)

    def remaining_passengers(self, state: int, action: int) -> int:
        """Return the queue after dispatch and before new arrivals."""

        self._validate_state_action(state, action)
        return max(0, state - self.capacity * action)

    def immediate_cost(self, state: int, action: int) -> float:
        """Return dispatch cost plus post-dispatch holding cost."""

        remaining = self.remaining_passengers(state, action)
        return self.dispatch_cost * action + self.holding_cost * remaining

    def transition_probabilities(self, state: int, action: int) -> dict[int, float]:
        """Return the sparse next-state distribution for one state-action pair."""

        remaining = self.remaining_passengers(state, action)
        distribution: dict[int, float] = {}
        assert self.arrival_probabilities is not None
        for arrival, probability in zip(
            self.arrivals,
            self.arrival_probabilities,
            strict=True,
        ):
            next_state = min(self.max_queue, remaining + arrival)
            distribution[next_state] = distribution.get(next_state, 0.0) + probability
        return distribution

    @cached_property
    def _transition_matrices(self) -> NDArray[np.float64]:
        matrices = np.zeros((len(self.actions), self.max_queue + 1, self.max_queue + 1))
        for state in self.states:
            for action in self.actions:
                for next_state, probability in self.transition_probabilities(
                    state,
                    action,
                ).items():
                    matrices[action, state, next_state] = probability
        return matrices

    def transition_matrices(self) -> NDArray[np.float64]:
        """Return dense matrices with shape (action, state, next_state)."""

        return self._transition_matrices.copy()

    @cached_property
    def _cost_matrix(self) -> NDArray[np.float64]:
        return np.array(
            [
                [self.immediate_cost(state, action) for action in self.actions]
                for state in self.states
            ],
            dtype=np.float64,
        )

    def cost_matrix(self) -> NDArray[np.float64]:
        """Return immediate costs with shape (state, action)."""

        return self._cost_matrix.copy()

    def _validate_state_action(self, state: int, action: int) -> None:
        if state < 0 or state > self.max_queue:
            raise ValueError(f"state must be in [0, {self.max_queue}]")
        if action not in self.actions:
            raise ValueError(f"action must be one of {self.actions}")

