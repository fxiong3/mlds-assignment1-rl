"""Pong-specific preprocessing and action mapping."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

PONG_ACTIONS = (2, 3)


def preprocess_pong_frame(image: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Convert a 210x160x3 Pong frame to the assignment's 80x80 binary image."""

    if image.shape != (210, 160, 3):
        raise ValueError("Pong frame must have shape (210, 160, 3)")
    processed = image[35:195:2, ::2, 0].copy()
    processed[(processed == 144) | (processed == 109)] = 0
    processed[processed != 0] = 1
    return processed.astype(np.float32)


def map_policy_action(action_index: int) -> int:
    """Map binary policy output 0/1 to Pong RIGHT/LEFT actions 2/3."""

    if action_index not in (0, 1):
        raise ValueError("policy action index must be 0 or 1")
    return PONG_ACTIONS[action_index]

