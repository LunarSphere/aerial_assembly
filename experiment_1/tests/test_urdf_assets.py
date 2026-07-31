from __future__ import annotations

import numpy as np

from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.geometry import (
    load_mesh,
    normalization_transform,
    recover_urdf_target_assembly,
)
from aerial_gripper_sim.urdf_assets import load_fixed_mesh_assembly


def test_new_urdf_resolves_block_and_four_washers(config: AppConfig) -> None:
    assert config.paths.assembly_urdf is not None
    assembly = load_fixed_mesh_assembly(config.paths.assembly_urdf)
    assert assembly.robot_name == "ghast_0"
    np.testing.assert_allclose(
        assembly.block.extents_m, [0.035, 0.040, 0.048], atol=2.0e-8
    )
    assert len(assembly.washers) == 4
    for washer in assembly.washers:
        np.testing.assert_allclose(
            washer.extents_m, [0.009, 0.0089868, 0.0012], atol=2.0e-8
        )


def test_new_urdf_fixed_joints_recover_target_grid(config: AppConfig) -> None:
    assert config.paths.assembly_urdf is not None
    assembly = load_fixed_mesh_assembly(config.paths.assembly_urdf)
    block = load_mesh(assembly.block.mesh_path)
    block_transform = normalization_transform("BB_0.stl", block, 1.0)
    target = recover_urdf_target_assembly(
        assembly,
        block_transform,
        load_mesh(config.paths.raw_assets / "SLW_0.stl"),
        config.units.stl_to_m,
    )
    centers = np.asarray(target["washer_centers_target_m"])
    np.testing.assert_allclose(
        np.unique(np.round(centers[:, 0], 7)),
        [-0.0095, 0.0095],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(
        np.unique(np.round(centers[:, 1], 7)),
        [-0.0141926, 0.0141926],
        atol=1.0e-7,
    )
    np.testing.assert_allclose(centers[:, 2], 0.0404, atol=2.0e-7)
