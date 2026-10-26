from __future__ import annotations

import json

import pytest

from mlds_assignment1.dp.experiments import run_dp_analysis, simulate_policy
from mlds_assignment1.dp.ferry_mdp import FerryMDP


def test_simulation_matches_hand_calculation_for_deterministic_arrivals() -> None:
    mdp = FerryMDP(
        max_queue=10,
        arrivals=(1,),
        arrival_probabilities=(1.0,),
        gamma=0.5,
    )
    policy = [0] * 11
    costs = simulate_policy(
        mdp,
        policy,
        initial_state=0,
        episodes=3,
        steps=3,
        seed=5,
    )
    assert costs.tolist() == pytest.approx([2.0, 2.0, 2.0])


def test_dp_analysis_writes_required_figures_and_validation(tmp_path) -> None:
    metrics = run_dp_analysis(
        output_dir=tmp_path,
        horizon=10,
        simulation_episodes=5,
        simulation_steps=10,
        max_queue=30,
    )

    assert metrics["vi_pi_policy_disagreements"] == 0
    assert metrics["vi_bellman_residual"] < 1e-8
    assert (tmp_path / "dp_values.csv").exists()
    assert (tmp_path / "dp_validation.json").exists()
    assert (tmp_path / "finite_horizon_value.png").exists()
    assert (tmp_path / "infinite_horizon_value.png").exists()
    assert (tmp_path / "optimal_policy.png").exists()
    written = json.loads((tmp_path / "dp_validation.json").read_text())
    assert written["vi_pi_policy_disagreements"] == 0

