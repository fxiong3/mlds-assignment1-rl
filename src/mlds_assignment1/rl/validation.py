"""Quantitative validation beyond unit tests for policy-gradient experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlds_assignment1.rl.pong import PONG_ACTIONS
from mlds_assignment1.rl.training import make_environment


def evaluate_random_policy(
    environment_name: str,
    *,
    episodes: int,
    seed: int,
    output_dir: str | Path,
    max_steps_per_episode: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run the assignment-compatible random-policy performance benchmark."""

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    rng = np.random.default_rng(seed)
    environment = make_environment(environment_name)
    records: list[dict[str, float | int]] = []
    try:
        for episode in range(1, episodes + 1):
            _, _ = environment.reset(seed=seed + episode)
            reward_total = 0.0
            step = 0
            while True:
                if environment_name == "Pong-v5":
                    action = int(rng.choice(PONG_ACTIONS))
                else:
                    action = int(rng.integers(environment.action_space.n))
                _, reward, terminated, truncated, _ = environment.step(action)
                reward_total += float(reward)
                step += 1
                reached_limit = (
                    max_steps_per_episode is not None
                    and step >= max_steps_per_episode
                )
                if terminated or truncated or reached_limit:
                    break
            records.append(
                {
                    "episode": episode,
                    "episode_reward": reward_total,
                    "episode_length": step,
                }
            )
    finally:
        environment.close()

    frame = pd.DataFrame(records)
    summary: dict[str, Any] = {
        "environment": environment_name,
        "policy": "uniform_random",
        "seed": seed,
        "episodes": episodes,
        "mean_reward": float(frame["episode_reward"].mean()),
        "standard_deviation": float(frame["episode_reward"].std(ddof=0)),
        "minimum_reward": float(frame["episode_reward"].min()),
        "maximum_reward": float(frame["episode_reward"].max()),
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    name = f"{environment_name.lower().replace('-', '_')}_random_seed_{seed}"
    frame.to_csv(output_path / f"{name}-evaluation.csv", index=False)
    (output_path / f"{name}-summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return frame, summary


def aggregate_evaluations(
    evaluation_csvs: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Aggregate independent-seed evaluation files with a simple 95% CI."""

    if not evaluation_csvs:
        raise ValueError("at least one evaluation CSV is required")
    per_seed: list[dict[str, Any]] = []
    all_rewards: list[pd.Series] = []
    for path in evaluation_csvs:
        frame = pd.read_csv(path)
        rewards = frame["episode_reward"]
        all_rewards.append(rewards)
        per_seed.append(
            {
                "file": str(path),
                "episodes": int(len(rewards)),
                "mean_reward": float(rewards.mean()),
                "standard_deviation": float(rewards.std(ddof=0)),
            }
        )
    pooled = pd.concat(all_rewards, ignore_index=True)
    seed_means = np.asarray([item["mean_reward"] for item in per_seed])
    seed_mean_standard_deviation = (
        float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0
    )
    if len(seed_means) > 1:
        rng = np.random.default_rng(0)
        bootstrap_means = rng.choice(
            seed_means,
            size=(10_000, len(seed_means)),
            replace=True,
        ).mean(axis=1)
        bootstrap_interval: list[float] | None = [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ]
    else:
        bootstrap_interval = None
    summary: dict[str, Any] = {
        "seeds": len(per_seed),
        "episodes": int(len(pooled)),
        "pooled_mean_reward": float(pooled.mean()),
        "pooled_standard_deviation": float(pooled.std(ddof=0)),
        "mean_across_seeds": float(seed_means.mean()),
        "seed_mean_standard_deviation": seed_mean_standard_deviation,
        "seed_mean_bootstrap_95_percent_ci": bootstrap_interval,
        "per_seed": per_seed,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    random_parser = subparsers.add_parser("random")
    random_parser.add_argument("--environment", required=True)
    random_parser.add_argument("--episodes", type=int, default=500)
    random_parser.add_argument("--seed", type=int, default=0)
    random_parser.add_argument("--output-dir", default="results/validation")
    random_parser.add_argument("--max-steps", type=int)
    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--inputs", nargs="+", required=True)
    aggregate_parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "random":
        _, summary = evaluate_random_policy(
            arguments.environment,
            episodes=arguments.episodes,
            seed=arguments.seed,
            output_dir=arguments.output_dir,
            max_steps_per_episode=arguments.max_steps,
        )
    else:
        summary = aggregate_evaluations(arguments.inputs, arguments.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
