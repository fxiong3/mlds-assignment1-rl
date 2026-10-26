from __future__ import annotations

from mlds_assignment1.rl.config import load_experiment_config


def test_load_config_allows_cli_overrides(tmp_path) -> None:
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        "environment: CartPole-v1\nepisodes: 100\ngamma: 0.95\nseed: 0\n",
        encoding="utf-8",
    )

    config = load_experiment_config(
        config_path,
        overrides={"episodes": 2, "seed": 17, "device": "cpu"},
    )

    assert config.environment == "CartPole-v1"
    assert config.episodes == 2
    assert config.seed == 17
    assert config.device == "cpu"

