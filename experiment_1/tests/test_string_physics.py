from __future__ import annotations

import mujoco
import numpy as np

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.scene_builder import SceneBuilder


def test_pretension_and_sag(fast_config: AppConfig) -> None:
    scene = SceneBuilder(fast_config).build("single_string_ramp_test")
    model, data = scene.model, scene.data
    sample = scene.string_model.sample(model, data, scene.payload_geom_ids)[0]
    assert abs(sample.mean_tension_n - fast_config.strings.pretension_n) <= 0.04
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, "string_0")
    start = int(model.flex_vertadr[flex_id])
    number = int(model.flex_vertnum[flex_id])
    initial = np.asarray(data.flexvert_xpos[start : start + number]).copy()
    for _ in range(600):
        mujoco.mj_step(model, data)
    final = np.asarray(data.flexvert_xpos[start : start + number])
    chord_mid_z = (final[0, 2] + final[-1, 2]) / 2.0
    assert final[number // 2, 2] < chord_mid_z
    assert np.max(np.abs(final[:, 2] - initial[:, 2])) < 0.02


def test_slack_string_has_no_compressive_tension(fast_config: AppConfig) -> None:
    scene = SceneBuilder(fast_config).build("single_string_ramp_test")
    flex_id = mujoco.mj_name2id(
        scene.model, mujoco.mjtObj.mjOBJ_FLEX, "string_0"
    )
    start = int(scene.model.flex_edgeadr[flex_id])
    number = int(scene.model.flex_edgenum[flex_id])
    scene.model.flexedge_length0[start : start + number] += 0.001
    mujoco.mj_forward(scene.model, scene.data)
    sample = scene.string_model.sample(
        scene.model, scene.data, scene.payload_geom_ids
    )[0]
    assert sample.peak_tension_n == 0.0


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
