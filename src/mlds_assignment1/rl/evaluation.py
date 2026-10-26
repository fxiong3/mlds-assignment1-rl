"""Checkpoint evaluation with fixed 500-episode assignment protocol."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical

from mlds_assignment1.rl.pong import map_policy_action
from mlds_assignment1.rl.training import (
    ExperimentConfig,
    build_model,
    load_checkpoint_data,
    make_environment,
    observation_to_tensor,
    resolve_device,
    seed_everything,
)


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device_override: str = "auto",
) -> tuple[nn.Module, ExperimentConfig, int]:
    """Load a checkpoint created by this project."""

    checkpoint = load_checkpoint_data(checkpoint_path)
    saved_config = ExperimentConfig(**checkpoint["config"])
    model = build_model(saved_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    config = replace(saved_config, device=device_override)
    return model, config, int(checkpoint["episode"])


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    episodes: int = 500,
    output_dir: str | Path = "results/raw",
    deterministic: bool = False,
    device: str = "auto",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate a frozen policy and write per-episode rewards plus summary statistics."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    model, config, training_episode = load_checkpoint(
        checkpoint_path,
        device_override=device,
    )
    device = resolve_device(config.device)
    model = model.to(device)
    model.eval()
    seed_everything(config.seed + 100_000)
    environment = make_environment(config.environment)
    records: list[dict[str, float | int]] = []

    try:
        for episode in range(1, episodes + 1):
            observation, _ = environment.reset(seed=config.seed + 100_000 + episode)
            previous_frame: np.ndarray | None = None
            episode_reward = 0.0
            step_index = 0
            while True:
                model_input, previous_frame = observation_to_tensor(
                    config.environment,
                    observation,
                    previous_frame,
                    device,
                )
                with torch.no_grad():
                    logits, _ = model(model_input)
                    if deterministic:
                        action_index = int(torch.argmax(logits, dim=-1).item())
                    else:
                        action_index = int(Categorical(logits=logits).sample().item())
                environment_action = (
                    map_policy_action(action_index)
                    if config.environment == "Pong-v5"
                    else action_index
                )
                observation, reward, terminated, truncated, _ = environment.step(
                    environment_action
                )
                episode_reward += float(reward)
                step_index += 1
                reached_limit = (
                    config.max_steps_per_episode is not None
                    and step_index >= config.max_steps_per_episode
                )
                if terminated or truncated or reached_limit:
                    break
            records.append(
                {
                    "episode": episode,
                    "episode_reward": episode_reward,
                    "episode_length": step_index,
                }
            )
    finally:
        environment.close()

    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "environment": config.environment,
        "training_episode": training_episode,
        "episodes": episodes,
        "deterministic": deterministic,
        "mean_reward": float(frame["episode_reward"].mean()),
        "standard_deviation": float(frame["episode_reward"].std(ddof=0)),
        "minimum_reward": float(frame["episode_reward"].min()),
        "maximum_reward": float(frame["episode_reward"].max()),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = Path(checkpoint_path).stem
    frame.to_csv(output_path / f"{checkpoint_stem}-evaluation.csv", index=False)
    (output_path / f"{checkpoint_stem}-summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return frame, summary

