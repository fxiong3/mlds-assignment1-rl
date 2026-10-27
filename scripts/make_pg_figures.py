"""Build the policy-gradient figures and summary table used in the report.

Run from the repository root after syncing results:
    python scripts/make_pg_figures.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

FIGURES = Path("report/figures")
RAW = Path("results/raw")
LR001 = Path("results/lr001")
RANDOM_MEAN = {"CartPole-v1": 22.98, "Pong-v5": -20.43}


def newest_evaluation(pattern: str, *folders: Path) -> Path | None:
    """Return the evaluation CSV with the largest training-episode number."""

    candidates = [path for folder in folders for path in folder.glob(pattern)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: int(re.search(r"episode-(\d+)", p.name).group(1)))


def experiment_figure(
    *,
    primary: Path,
    others: list[tuple[Path, str]],
    evaluation: Path | None,
    environment: str,
    title: str,
    output: Path,
) -> dict[str, float]:
    if not primary.exists():
        return {}
    frame = pd.read_csv(primary)
    figure, (left, right) = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw={"width_ratios": [1.6, 1]})
    left.plot(frame["episode"], frame["episode_reward"], color="tab:blue", alpha=0.25, lw=0.6,
              label="Episode reward")
    left.plot(frame["episode"], frame["moving_average_100"], color="tab:blue", lw=2,
              label="100-episode moving average")
    for path, label in others:
        if path.exists():
            other = pd.read_csv(path)
            left.plot(other["episode"], other["moving_average_100"], lw=1, alpha=0.8, label=label)
    left.axhline(RANDOM_MEAN[environment], color="gray", ls=":", lw=1, label="Uniform random policy")
    left.set_xlabel("Training episode")
    left.set_ylabel("Episode reward")
    left.set_title(f"{title}: training")
    left.grid(alpha=0.25)
    left.legend(fontsize=7, loc="best")

    stats: dict[str, float] = {}
    if evaluation is not None:
        rewards = pd.read_csv(evaluation)["episode_reward"]
        episode = int(re.search(r"episode-(\d+)", evaluation.name).group(1))
        stats = {"n": len(rewards), "mean": rewards.mean(), "std": rewards.std(ddof=0), "checkpoint": episode}
        left.axvline(episode, color="black", ls="--", lw=0.8, label="Evaluated checkpoint")
        left.legend(fontsize=7, loc="best")
        bins = range(int(rewards.min()), int(rewards.max()) + 2) if environment == "Pong-v5" else 30
        right.hist(rewards, bins=bins, color="tab:blue", edgecolor="black", alpha=0.8, label="Rollout rewards")
        right.axvline(stats["mean"], color="tab:red", ls="--", lw=2,
                      label=f"Mean {stats['mean']:.2f} (std {stats['std']:.2f})")
        right.set_title(f"{len(rewards)} rollouts, checkpoint ep. {episode}")
    else:
        right.text(0.5, 0.5, "evaluation pending", ha="center", transform=right.transAxes)
        right.set_title("Rollout rewards")
    right.set_xlabel("Episode reward")
    right.set_ylabel("Number of episodes")
    right.grid(axis="y", alpha=0.25)
    right.legend(fontsize=7, loc="upper left")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)
    return stats


def validation_figure(output: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axis = axes[0]
    for seed in range(3):
        for folder, tag, colour in ((RAW, "lr0.01", "tab:red"), (LR001, "lr0.001", "tab:green")):
            path = folder / f"cartpole_v1_no_baseline_{tag}_seed_{seed}-training.csv"
            if path.exists():
                frame = pd.read_csv(path)
                axis.plot(frame["episode"], frame["moving_average_100"], color=colour, lw=1,
                          label=f"lr={tag[2:]}" if seed == 0 else None)
    gamma_path = LR001 / "cartpole_v1_no_baseline_lr0.001_gamma0.99_seed_10-training.csv"
    if gamma_path.exists():
        frame = pd.read_csv(gamma_path)
        axis.plot(frame["episode"], frame["moving_average_100"], color="tab:purple", lw=1.5,
                  label="lr=0.001, $\\gamma$=0.99")
    axis.axhline(RANDOM_MEAN["CartPole-v1"], color="gray", ls=":", label="Uniform random")
    axis.set_title("CartPole, no baseline: lr and $\\gamma$")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("100-episode moving average")

    axis = axes[1]
    for seed, colour in zip(range(3), ("tab:blue", "tab:orange", "tab:green"), strict=True):
        frame = pd.read_csv(RAW / f"cartpole_v1_baseline_seed_{seed}-training.csv")
        ratio = (frame["raw_advantage_variance"] / frame["return_variance"].clip(lower=1e-8)).rolling(50).median()
        axis.plot(frame["episode"], ratio, color=colour, lw=1.2, label=f"seed {seed}")
    axis.axhline(1.0, color="gray", ls=":", label="no baseline (ratio 1)")
    axis.set_yscale("log")
    axis.set_title("CartPole: Var(G - b(s)) / Var(G)")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Within-episode variance ratio")

    axis = axes[2]
    for path, label in (
        (RAW / "pong_v5_no_baseline_seed_1-training.csv", "No baseline"),
        (RAW / "pong_v5_baseline_seed_2-training.csv", "Learned baseline"),
    ):
        if path.exists():
            frame = pd.read_csv(path)
            axis.plot(frame["episode"], frame["gradient_norm"].rolling(50).median(), lw=1.5, label=label)
    axis.set_title("Pong: gradient norm before clipping")
    axis.set_xlabel("Training episode")
    axis.set_ylabel("Rolling median (50 ep.)")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    final_folders = (Path("results/final"), RAW, Path("results/fallback"))
    summary = {
        "cartpole_no_baseline": experiment_figure(
            primary=LR001 / "cartpole_v1_no_baseline_lr0.001_seed_0-training.csv",
            others=[(LR001 / f"cartpole_v1_no_baseline_lr0.001_seed_{s}-training.csv", f"Moving avg, seed {s}")
                    for s in (1, 2)],
            evaluation=newest_evaluation("cartpole_v1_no_baseline_lr0.001_seed_0-episode-*-evaluation.csv", LR001),
            environment="CartPole-v1",
            title="CartPole, no baseline (seed 0)",
            output=FIGURES / "cartpole_no_baseline.png",
        ),
        "cartpole_baseline": experiment_figure(
            primary=RAW / "cartpole_v1_baseline_seed_0-training.csv",
            others=[(RAW / f"cartpole_v1_baseline_seed_{s}-training.csv", f"Moving avg, seed {s}") for s in (1, 2)],
            evaluation=newest_evaluation("cartpole_v1_baseline_seed_0-episode-*-evaluation.csv", RAW),
            environment="CartPole-v1",
            title="CartPole, learned baseline (seed 0)",
            output=FIGURES / "cartpole_baseline.png",
        ),
        "pong_no_baseline": experiment_figure(
            primary=RAW / "pong_v5_no_baseline_seed_1-training.csv",
            others=[(RAW / "pong_v5_no_baseline_seed_0-training.csv", "Moving avg, lr=0.01 run")],
            evaluation=newest_evaluation("pong_v5_no_baseline_seed_1-episode-*-evaluation.csv", *final_folders),
            environment="Pong-v5",
            title="Pong, no baseline (lr=0.001)",
            output=FIGURES / "pong_no_baseline.png",
        ),
        "pong_baseline": experiment_figure(
            primary=RAW / "pong_v5_baseline_seed_2-training.csv",
            others=[(RAW / "pong_v5_baseline_seed_0-training.csv", "Moving avg, lr=0.01 run")],
            evaluation=newest_evaluation("pong_v5_baseline_seed_2-episode-*-evaluation.csv", *final_folders),
            environment="Pong-v5",
            title="Pong, learned baseline (lr=0.001)",
            output=FIGURES / "pong_baseline.png",
        ),
    }
    validation_figure(FIGURES / "pg_validation.png")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
