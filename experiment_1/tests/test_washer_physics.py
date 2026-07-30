from __future__ import annotations

import mujoco
import numpy as np
import pytest

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.scene_builder import SceneBuilder


def test_compliant_insertion_deflects_fingers(
    config: AppConfig, tmp_path
) -> None:
    result = run_simulation(
        config,
        "washer_insertion_test",
        output_dir=tmp_path / "insertion",
    )
    metrics = result["metrics"]
    assert result["success"], result["failure_reason"]
    assert metrics["minimum_peg_insertion_depth_m"] >= 0.003
    assert metrics["peak_washer_strain_proxy"] > 0
    assert metrics["peak_insertion_force_n"] > 0
    assert metrics["peak_insertion_force_n"] <= config.controller.max_downward_force_n


def test_pullout_has_measurable_retention(config: AppConfig, tmp_path) -> None:
    result = run_simulation(
        config,
        "washer_pullout_test",
        output_dir=tmp_path / "pullout",
    )
    assert result["success"], result["failure_reason"]
    assert (
        result["metrics"]["peak_retention_force_above_weight_n"]
        >= config.metrics.retention_force_epsilon_n
    )


def test_stiffer_finger_has_monotonic_restoring_torque(config: AppConfig) -> None:
    angle = 0.15
    low = config.with_overrides(["washer.effective_stiffness_nm_rad=0.003"])
    high = config.with_overrides(["washer.effective_stiffness_nm_rad=0.009"])
    assert high.washer.effective_stiffness_nm_rad * angle > (
        low.washer.effective_stiffness_nm_rad * angle
    )


@pytest.mark.slow
def test_tetrahedral_mode_compiles_and_stays_finite(config: AppConfig) -> None:
    deformable = config.with_overrides(["washer.mode=deformable_flex"])
    scene = SceneBuilder(deformable).build("washer_insertion_test")
    assert scene.model.nflex == 4
    assert scene.model.nflexvert > 1000
    for _ in range(1000):
        mujoco.mj_step(scene.model, scene.data)
    assert np.all(np.isfinite(scene.data.qpos))
