"""Evaluate a trained policy checkpoint over 500 episodes by default."""

from __future__ import annotations

import argparse
import json

from mlds_assignment1.rl.evaluation import evaluate_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--output-dir", default="results/raw")
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of sampling the learned policy",
    )
    arguments = parser.parse_args()
    _, summary = evaluate_checkpoint(
        arguments.checkpoint,
        episodes=arguments.episodes,
        output_dir=arguments.output_dir,
        deterministic=arguments.deterministic,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

