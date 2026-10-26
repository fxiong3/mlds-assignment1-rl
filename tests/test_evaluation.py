from __future__ import annotations

import json

import pandas as pd

from mlds_assignment1.rl.evaluation import evaluate_checkpoint
from mlds_assignment1.rl.training import ExperimentConfig, build_model, save_checkpoint


def test_evaluation_writes_episodes_and_summary(tmp_path) -> None:
    config = ExperimentConfig(
        environment="CartPole-v1",
        episodes=1,
        seed=11,
        max_steps_per_episode=5,
        output_dir=str(tmp_path / "raw"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        device="cpu",
    )
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, build_model(config), config, episode=1)

    frame, summary = evaluate_checkpoint(
        checkpoint,
        episodes=3,
        output_dir=tmp_path / "evaluation",
        deterministic=True,
    )

    assert frame.shape[0] == 3
    assert frame["episode_length"].max() <= 5
    assert summary["episodes"] == 3
    assert summary["mean_reward"] == frame["episode_reward"].mean()
    written_frame = pd.read_csv(tmp_path / "evaluation" / "model-evaluation.csv")
    assert written_frame.shape[0] == 3
    written_summary = json.loads(
        (tmp_path / "evaluation" / "model-summary.json").read_text()
    )
    assert written_summary["episodes"] == 3

