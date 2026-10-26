from __future__ import annotations

import pandas as pd

from mlds_assignment1.rl.plotting import (
    plot_evaluation_histogram,
    plot_training_curve,
)


def test_plotting_creates_required_training_and_evaluation_figures(tmp_path) -> None:
    training_csv = tmp_path / "training.csv"
    evaluation_csv = tmp_path / "evaluation.csv"
    training_figure = tmp_path / "figures" / "training.png"
    evaluation_figure = tmp_path / "figures" / "evaluation.png"
    pd.DataFrame(
        {
            "episode": [1, 2, 3],
            "episode_reward": [10.0, 20.0, 30.0],
            "moving_average_100": [10.0, 15.0, 20.0],
        }
    ).to_csv(training_csv, index=False)
    pd.DataFrame(
        {
            "episode": [1, 2, 3],
            "episode_reward": [100.0, 200.0, 200.0],
            "episode_length": [100, 200, 200],
        }
    ).to_csv(evaluation_csv, index=False)

    plot_training_curve(training_csv, training_figure, title="Training")
    plot_evaluation_histogram(evaluation_csv, evaluation_figure, title="Evaluation")

    assert training_figure.stat().st_size > 0
    assert evaluation_figure.stat().st_size > 0

