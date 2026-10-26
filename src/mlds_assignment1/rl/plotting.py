"""Create assignment-ready training curves and evaluation histograms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def plot_training_curve(
    metrics_csv: str | Path,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot raw episode rewards with the required 100-episode moving average."""

    frame = pd.read_csv(metrics_csv)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(
        frame["episode"],
        frame["episode_reward"],
        alpha=0.35,
        linewidth=0.8,
        label="Episode reward",
    )
    axis.plot(
        frame["episode"],
        frame["moving_average_100"],
        linewidth=2,
        label="100-episode moving average",
    )
    axis.set_title(title)
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Episode reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def plot_evaluation_histogram(
    evaluation_csv: str | Path,
    output_path: str | Path,
    *,
    title: str,
) -> None:
    """Plot evaluation rewards and annotate sample mean and standard deviation."""

    frame = pd.read_csv(evaluation_csv)
    rewards = frame["episode_reward"]
    mean = rewards.mean()
    standard_deviation = rewards.std(ddof=0)
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(rewards, bins="auto", edgecolor="black", alpha=0.8)
    axis.axvline(mean, color="tab:red", linestyle="--", linewidth=2, label="Mean")
    axis.set_title(title)
    axis.set_xlabel("Episode reward")
    axis.set_ylabel("Count")
    axis.text(
        0.98,
        0.95,
        f"n={len(rewards)}\nmean={mean:.2f}\nstd={standard_deviation:.2f}",
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="plot_type", required=True)
    training_parser = subparsers.add_parser("training")
    training_parser.add_argument("--input", required=True)
    training_parser.add_argument("--output", required=True)
    training_parser.add_argument("--title", required=True)
    evaluation_parser = subparsers.add_parser("evaluation")
    evaluation_parser.add_argument("--input", required=True)
    evaluation_parser.add_argument("--output", required=True)
    evaluation_parser.add_argument("--title", required=True)
    arguments = parser.parse_args()
    if arguments.plot_type == "training":
        plot_training_curve(arguments.input, arguments.output, title=arguments.title)
    else:
        plot_evaluation_histogram(arguments.input, arguments.output, title=arguments.title)


if __name__ == "__main__":
    main()

