from __future__ import annotations

import numpy as np
import pytest
import torch

from mlds_assignment1.rl.core import discounted_returns, normalize_advantages
from mlds_assignment1.rl.pong import PONG_ACTIONS, preprocess_pong_frame


def test_discounted_returns() -> None:
    result = discounted_returns([1.0, 2.0, 3.0], gamma=0.5)
    torch.testing.assert_close(result, torch.tensor([2.75, 3.5, 3.0]))


def test_discounted_returns_rejects_invalid_gamma() -> None:
    with pytest.raises(ValueError):
        discounted_returns([1.0], gamma=1.1)


def test_normalized_advantages_are_centered_and_scaled() -> None:
    values = torch.tensor([1.0, 2.0, 5.0, 8.0])
    normalized = normalize_advantages(values)
    assert normalized.mean().item() == pytest.approx(0.0, abs=1e-7)
    assert normalized.std(unbiased=False).item() == pytest.approx(1.0, rel=1e-6)


def test_constant_advantages_normalize_to_zero() -> None:
    normalized = normalize_advantages(torch.ones(5))
    torch.testing.assert_close(normalized, torch.zeros(5))


def test_pong_preprocessing_matches_assignment_definition() -> None:
    frame = np.full((210, 160, 3), 144, dtype=np.uint8)
    frame[35, 0, 0] = 255
    frame[37, 2, 0] = 109
    processed = preprocess_pong_frame(frame)

    assert processed.shape == (80, 80)
    assert processed.dtype == np.float32
    assert processed[0, 0] == 1.0
    assert processed[1, 1] == 0.0
    assert set(np.unique(processed)).issubset({0.0, 1.0})


def test_pong_preprocessing_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError):
        preprocess_pong_frame(np.zeros((80, 80), dtype=np.uint8))


def test_pong_action_mapping_is_exactly_left_and_right() -> None:
    assert PONG_ACTIONS == (2, 3)

