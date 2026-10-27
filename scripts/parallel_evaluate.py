"""Split the 500-episode evaluation of one checkpoint across CPU worker processes.

Each worker evaluates a disjoint block of episode indices with the same per-episode
seeding as ``evaluate_checkpoint`` (``seed + 100_000 + episode``), so the merged
result matches the sequential protocol up to sampling order.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.distributions import Categorical

from mlds_assignment1.rl.evaluation import load_checkpoint
from mlds_assignment1.rl.pong import map_policy_action
from mlds_assignment1.rl.training import make_environment, observation_to_tensor


def run_block(checkpoint: str, episodes: list[int]) -> list[dict[str, float]]:
    torch.set_num_threads(1)
    model, config, _ = load_checkpoint(checkpoint, device_override="cpu")
    model.eval()
    device = torch.device("cpu")
    environment = make_environment(config.environment)
    records = []
    try:
        for episode in episodes:
            torch.manual_seed(config.seed + 100_000 + episode)
            observation, _ = environment.reset(seed=config.seed + 100_000 + episode)
            previous_frame: np.ndarray | None = None
            total, steps = 0.0, 0
            while True:
                model_input, previous_frame = observation_to_tensor(
                    config.environment, observation, previous_frame, device
                )
                with torch.no_grad():
                    logits, _ = model(model_input)
                    action = int(Categorical(logits=logits).sample().item())
                if config.environment == "Pong-v5":
                    action = map_policy_action(action)
                observation, reward, terminated, truncated, _ = environment.step(action)
                total += float(reward)
                steps += 1
                limit = config.max_steps_per_episode
                if terminated or truncated or (limit is not None and steps >= limit):
                    break
            records.append({"episode": episode, "episode_reward": total, "episode_length": steps})
    finally:
        environment.close()
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output-dir", default="results/raw")
    args = parser.parse_args()
    if args.episodes <= 0 or args.workers <= 0:
        raise ValueError("episodes and workers must be positive")

    indices = list(range(1, args.episodes + 1))
    blocks = [indices[i :: args.workers] for i in range(args.workers)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = pool.map(run_block, [args.checkpoint] * len(blocks), blocks)
        records = [row for block in results for row in block]

    frame = pd.DataFrame(records).sort_values("episode").reset_index(drop=True)
    _, config, training_episode = load_checkpoint(args.checkpoint, device_override="cpu")
    rewards = frame["episode_reward"]
    summary = {
        "checkpoint": args.checkpoint,
        "environment": config.environment,
        "training_episode": training_episode,
        "episodes": args.episodes,
        "deterministic": False,
        "mean_reward": float(rewards.mean()),
        "standard_deviation": float(rewards.std(ddof=0)),
        "minimum_reward": float(rewards.min()),
        "maximum_reward": float(rewards.max()),
        "mean_episode_length": float(frame["episode_length"].mean()),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stem = Path(args.checkpoint).stem
    frame.to_csv(output / f"{stem}-evaluation.csv", index=False)
    (output / f"{stem}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
