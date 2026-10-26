from __future__ import annotations

import json

import pandas as pd

from mlds_assignment1.rl.validation import (
    aggregate_evaluations,
    evaluate_random_policy,
)


def test_random_policy_benchmark_is_reproducible(tmp_path) -> None:
    first_frame, first_summary = evaluate_random_policy(
        "CartPole-v1",
        episodes=3,
        seed=7,
        output_dir=tmp_path / "first",
        max_steps_per_episode=5,
    )
    second_frame, second_summary = evaluate_random_policy(
        "CartPole-v1",
        episodes=3,
        seed=7,
        output_dir=tmp_path / "second",
        max_steps_per_episode=5,
    )

    pd.testing.assert_frame_equal(first_frame, second_frame)
    assert first_summary["mean_reward"] == second_summary["mean_reward"]
    assert (tmp_path / "first" / "cartpole_v1_random_seed_7-summary.json").exists()


def test_aggregate_evaluations_reports_across_seed_statistics(tmp_path) -> None:
    inputs = []
    for seed, rewards in ((0, [1.0, 2.0]), (1, [3.0, 4.0])):
        path = tmp_path / f"seed_{seed}.csv"
        pd.DataFrame(
            {
                "episode": [1, 2],
                "episode_reward": rewards,
                "episode_length": [10, 10],
            }
        ).to_csv(path, index=False)
        inputs.append(path)

    summary = aggregate_evaluations(inputs, tmp_path / "aggregate.json")

    assert summary["seeds"] == 2
    assert summary["episodes"] == 4
    assert summary["pooled_mean_reward"] == 2.5
    assert len(summary["per_seed"]) == 2
    assert json.loads((tmp_path / "aggregate.json").read_text())["seeds"] == 2

