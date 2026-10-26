from __future__ import annotations

import torch

from mlds_assignment1.rl.models import CartPoleActorCritic, PongActorCritic


def test_cartpole_actor_critic_output_shapes() -> None:
    model = CartPoleActorCritic(hidden_size=32)
    logits, values = model(torch.zeros(7, 4))
    assert logits.shape == (7, 2)
    assert values.shape == (7,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()


def test_pong_actor_critic_output_shapes() -> None:
    model = PongActorCritic()
    logits, values = model(torch.zeros(3, 2, 80, 80))
    assert logits.shape == (3, 2)
    assert values.shape == (3,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(values).all()

