from __future__ import annotations

import numpy as np

from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.geometry import (
    load_mesh,
    normalization_transform,
    recover_target_assembly,
)


def test_recovered_washers_form_symmetric_nonoverlapping_grid(
    config: AppConfig,
) -> None:
    transforms = {}
    for name in ("BB_0.stl", "SLW_0.stl"):
        mesh = load_mesh(config.paths.raw_assets / name)
        transforms[name] = normalization_transform(
            name, mesh, config.units.stl_to_m
        )
    result = recover_target_assembly(config.paths.raw_assets, transforms)
    centers = np.asarray(result["washer_centers_target_m"])
    assert centers.shape == (4, 3)
    assert np.allclose(sorted(np.unique(np.round(centers[:, 0], 6))), [-0.0095, 0.0095])
    assert np.allclose(
        sorted(np.unique(np.round(centers[:, 1], 6))),
        [-0.014168, 0.014168],
    )
    distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
    assert np.min(distances[np.nonzero(distances)]) > 0.008
