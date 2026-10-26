"""Command-line entry point for the shuttle-dispatch experiments."""

from __future__ import annotations

import argparse

from mlds_assignment1.dp.experiments import run_dp_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/dp")
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--simulation-episodes", type=int, default=1_000)
    parser.add_argument("--simulation-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    metrics = run_dp_analysis(
        output_dir=arguments.output_dir,
        horizon=arguments.horizon,
        simulation_episodes=arguments.simulation_episodes,
        simulation_steps=arguments.simulation_steps,
        seed=arguments.seed,
    )
    print(
        "DP analysis complete: "
        f"VI/PI policy disagreements={metrics['vi_pi_policy_disagreements']}, "
        f"Bellman residual={metrics['vi_bellman_residual']:.3e}"
    )


if __name__ == "__main__":
    main()

