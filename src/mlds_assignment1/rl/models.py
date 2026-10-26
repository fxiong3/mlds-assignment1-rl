"""Small actor-critic networks used for REINFORCE experiments."""

from __future__ import annotations

from torch import Tensor, nn


class CartPoleActorCritic(nn.Module):
    """Shared MLP with policy and optional value-baseline heads."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_size, 2)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(observations)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values


class PongActorCritic(nn.Module):
    """Compact convolutional policy/value network for two-frame Pong input."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, 2)
        self.value_head = nn.Linear(256, 1)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(observations)
        logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return logits, values

