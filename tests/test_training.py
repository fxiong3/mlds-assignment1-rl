from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from mlds_assignment1.rl.training import (
    ExperimentConfig,
    build_model,
    compute_reinforce_loss,
    experiment_name,
    load_checkpoint_data,
    make_environment,
    observation_to_tensor,
    resolve_device,
    seed_everything,
    train,
)


def test_seed_everything_repeats_numpy_and_torch_draws() -> None:
    seed_everything(42)
    first_numpy = np.random.random(3)
    first_torch = torch.rand(3)
    seed_everything(42)
    np.testing.assert_allclose(np.random.random(3), first_numpy)
    torch.testing.assert_close(torch.rand(3), first_torch)


def test_cartpole_environment_can_reset_and_step() -> None:
    environment = make_environment("CartPole-v1")
    try:
        observation, _ = environment.reset(seed=7)
        assert observation.shape == (4,)
        next_observation, reward, terminated, truncated, _ = environment.step(0)
        assert next_observation.shape == (4,)
        assert reward == 1.0
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        environment.close()


def test_cartpole_observation_conversion() -> None:
    tensor, previous = observation_to_tensor(
        "CartPole-v1",
        np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
        previous_frame=None,
        device=torch.device("cpu"),
    )
    assert tensor.shape == (1, 4)
    assert previous is None


def test_reinforce_loss_without_baseline() -> None:
    log_probabilities = torch.tensor([-0.2, -0.4], requires_grad=True)
    returns = torch.tensor([2.0, 1.0])
    values = torch.tensor([9.0, 9.0], requires_grad=True)
    policy_loss, value_loss, raw_advantages = compute_reinforce_loss(
        log_probabilities,
        returns,
        values,
        use_baseline=False,
        normalize=False,
    )

    assert policy_loss.item() == pytest.approx(0.8)
    assert value_loss.item() == 0.0
    torch.testing.assert_close(raw_advantages, returns)


def test_reinforce_loss_with_baseline_detaches_advantage_from_policy() -> None:
    log_probabilities = torch.tensor([-0.2, -0.4], requires_grad=True)
    returns = torch.tensor([2.0, 1.0])
    values = torch.tensor([0.5, 0.25], requires_grad=True)
    policy_loss, value_loss, raw_advantages = compute_reinforce_loss(
        log_probabilities,
        returns,
        values,
        use_baseline=True,
        normalize=False,
    )

    torch.testing.assert_close(raw_advantages, torch.tensor([1.5, 0.75]))
    assert policy_loss.item() == pytest.approx(0.6)
    assert value_loss.item() == pytest.approx((1.5**2 + 0.75**2) / 2)
    policy_loss.backward(retain_graph=True)
    assert values.grad is None


def test_experiment_config_validates_environment() -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(environment="MountainCar-v0")


@pytest.mark.parametrize(
    "overrides",
    [
        {"checkpoint_every": 0},
        {"hidden_size": 0},
        {"max_steps_per_episode": 0},
        {"value_loss_coefficient": -1.0},
        {"optimization_batch_size": 0},
    ],
)
def test_experiment_config_rejects_invalid_training_parameters(overrides) -> None:
    with pytest.raises(ValueError):
        ExperimentConfig(environment="CartPole-v1", **overrides)


def test_cartpole_training_smoke_test_writes_auditable_artifacts(tmp_path) -> None:
    config = ExperimentConfig(
        environment="CartPole-v1",
        episodes=2,
        seed=3,
        max_steps_per_episode=5,
        checkpoint_every=1,
        output_dir=str(tmp_path / "raw"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )

    model, metrics = train(config)

    assert isinstance(model, torch.nn.Module)
    assert metrics["episode"].tolist() == [1, 2]
    assert metrics["episode_length"].max() <= 5
    run_name = experiment_name(config)
    assert (tmp_path / "raw" / f"{run_name}-training.csv").exists()
    assert (tmp_path / "raw" / f"{run_name}-trace.json").exists()
    assert (tmp_path / "raw" / f"{run_name}-config.json").exists()
    assert (tmp_path / "checkpoints" / f"{run_name}-episode-2.pt").exists()


def test_explicit_cpu_device_and_model_factory() -> None:
    config = ExperimentConfig(environment="CartPole-v1", device="cpu")
    assert resolve_device(config.device) == torch.device("cpu")
    assert isinstance(build_model(config), torch.nn.Module)


def test_training_can_resume_from_checkpoint(tmp_path) -> None:
    first_config = ExperimentConfig(
        environment="CartPole-v1",
        episodes=1,
        seed=21,
        max_steps_per_episode=3,
        checkpoint_every=1,
        output_dir=str(tmp_path / "raw"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )
    train(first_config)
    first_checkpoint = (
        tmp_path
        / "checkpoints"
        / f"{experiment_name(first_config)}-episode-1.pt"
    )
    resumed_config = ExperimentConfig(
        **{**first_config.__dict__, "episodes": 2}
    )

    _, metrics = train(resumed_config, resume_checkpoint=first_checkpoint)

    assert metrics["episode"].tolist() == [1, 2]
    assert (
        tmp_path
        / "checkpoints"
        / f"{experiment_name(resumed_config)}-episode-2.pt"
    ).exists()


def test_resume_rejects_training_semantic_changes(tmp_path) -> None:
    config = ExperimentConfig(
        environment="CartPole-v1",
        episodes=1,
        seed=5,
        max_steps_per_episode=3,
        checkpoint_every=1,
        output_dir=str(tmp_path / "raw"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )
    train(config)
    checkpoint = (
        tmp_path / "checkpoints" / f"{experiment_name(config)}-episode-1.pt"
    )
    incompatible = ExperimentConfig(**{**config.__dict__, "episodes": 2, "gamma": 0.1})

    with pytest.raises(ValueError, match="gamma"):
        train(incompatible, resume_checkpoint=checkpoint)


def test_resume_matches_continuous_training_exactly(tmp_path) -> None:
    common = {
        "environment": "CartPole-v1",
        "seed": 31,
        "max_steps_per_episode": 3,
        "checkpoint_every": 1,
        "device": "cpu",
    }
    continuous_config = ExperimentConfig(
        **common,
        episodes=2,
        output_dir=str(tmp_path / "continuous_raw"),
        checkpoint_dir=str(tmp_path / "continuous_checkpoints"),
    )
    continuous_model, continuous_metrics = train(continuous_config)
    interrupted_config = ExperimentConfig(
        **common,
        episodes=1,
        output_dir=str(tmp_path / "resumed_raw"),
        checkpoint_dir=str(tmp_path / "resumed_checkpoints"),
    )
    train(interrupted_config)
    first_checkpoint = (
        tmp_path
        / "resumed_checkpoints"
        / f"{experiment_name(interrupted_config)}-episode-1.pt"
    )
    resumed_config = ExperimentConfig(**{**interrupted_config.__dict__, "episodes": 2})
    resumed_model, resumed_metrics = train(
        resumed_config,
        resume_checkpoint=first_checkpoint,
    )

    pd.testing.assert_frame_equal(continuous_metrics, resumed_metrics)
    for name, parameter in continuous_model.state_dict().items():
        torch.testing.assert_close(parameter, resumed_model.state_dict()[name])
    assert load_checkpoint_data(first_checkpoint)["episode"] == 1

