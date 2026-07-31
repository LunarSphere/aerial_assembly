from __future__ import annotations

import mujoco
import numpy as np

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.scene_builder import SceneBuilder
from aerial_gripper_sim.string_model import _cable_vertices, _is_captured_by_j_hook


def test_inextensible_cable_has_slack_and_sags(fast_config: AppConfig) -> None:
    scene = SceneBuilder(fast_config).build("single_string_ramp_test")
    model, data = scene.model, scene.data
    sample = scene.string_model.sample(model, data, scene.payload_geom_ids)[0]
    assert sample.slack_m > 0.0
    assert abs(sample.axial_strain) < 1.0e-6
    assert sample.endpoint_error_m < fast_config.strings.endpoint_error_limit_m
    cable = scene.string_model.cables[0]
    initial = _cable_vertices(model, data, cable)
    for _ in range(600):
        mujoco.mj_step(model, data)
    final = _cable_vertices(model, data, cable)
    chord_mid_z = (final[0, 2] + final[-1, 2]) / 2.0
    assert final[len(final) // 2, 2] < chord_mid_z
    assert np.max(np.abs(final[:, 2] - initial[:, 2])) < 0.02
    settled = scene.string_model.sample(model, data, scene.payload_geom_ids)[0]
    assert abs(settled.axial_strain) < 1.0e-6
    assert settled.endpoint_error_m < fast_config.strings.endpoint_error_limit_m


def test_slack_string_has_no_compressive_tension(fast_config: AppConfig) -> None:
    scene = SceneBuilder(fast_config).build("single_string_ramp_test")
    sample = scene.string_model.sample(
        scene.model, scene.data, scene.payload_geom_ids
    )[0]
    assert sample.slack_m > 0.0
    assert sample.peak_tension_n >= 0.0
    assert sample.peak_tension_n < 0.01


def test_single_string_ramp_contact_is_finite(
    fast_config: AppConfig, tmp_path
) -> None:
    result = run_simulation(
        fast_config,
        "single_string_ramp_test",
        output_dir=tmp_path / "single_string",
    )
    assert result["success"], result["failure_reason"]
    assert result["metrics"]["minimum_hook_distance_m"] < 0.005
    assert result["metrics"]["peak_gripper_force_n"] < 2.0


def test_j_hook_capture_classifier_uses_measured_throat() -> None:
    inside_first_pocket = np.array([[0.0, -0.0060, 0.0420]])
    above_return = np.array([[0.0, -0.0060, 0.0460]])
    outside_slope = np.array([[0.0, -0.0075, 0.0420]])
    assert _is_captured_by_j_hook(inside_first_pocket, 0.0003)
    assert not _is_captured_by_j_hook(above_return, 0.0003)
    assert not _is_captured_by_j_hook(outside_slope, 0.0003)
