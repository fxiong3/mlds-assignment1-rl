"""Train one configured REINFORCE experiment."""

from __future__ import annotations

import argparse

from mlds_assignment1.rl.config import load_experiment_config
from mlds_assignment1.rl.training import experiment_name, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an experiment YAML file")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    parser.add_argument("--episodes", type=int, help="Override the training episode count")
    parser.add_argument("--device", help="Override device: auto, cpu, cuda, or mps")
    arguments = parser.parse_args()
    config = load_experiment_config(
        arguments.config,
        overrides={
            "seed": arguments.seed,
            "episodes": arguments.episodes,
            "device": arguments.device,
        },
    )
    _, metrics = train(config)
    print(
        f"Completed {experiment_name(config)}: "
        f"episodes={len(metrics)}, "
        f"final_moving_average_100={metrics['moving_average_100'].iloc[-1]:.2f}"
    )


if __name__ == "__main__":
    main()

