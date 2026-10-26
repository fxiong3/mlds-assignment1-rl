"""Pure policy-gradient utilities shared by both environments."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def discounted_returns(rewards: Sequence[float], gamma: float) -> Tensor:
    """Compute reward-to-go returns for one episode."""

    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    returns = torch.empty(len(rewards), dtype=torch.float32)
    running_return = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running_return = float(rewards[index]) + gamma * running_return
        returns[index] = running_return
    return returns


def normalize_advantages(advantages: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Standardize advantages, returning zeros for constant inputs."""

    if advantages.numel() == 0:
        return advantages
    centered = advantages - advantages.mean()
    standard_deviation = centered.std(unbiased=False)
    if standard_deviation <= epsilon:
        return torch.zeros_like(advantages)
    return centered / (standard_deviation + epsilon)

