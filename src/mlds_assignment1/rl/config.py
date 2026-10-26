"""YAML configuration loading with explicit command-line overrides."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from mlds_assignment1.rl.training import ExperimentConfig


def load_experiment_config(
    path: str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load and validate one experiment configuration."""

    contents = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(contents, dict):
        raise ValueError("configuration must contain a YAML mapping")
    if overrides:
        contents.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig(**contents)

