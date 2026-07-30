from __future__ import annotations

import numpy as np

from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.geometry import (
    detect_anchor_pairs,
    load_mesh,
    normalization_transform,
)


def test_seven_ordered_symmetric_anchor_pairs(config: AppConfig) -> None:
    mesh = load_mesh(config.paths.raw_assets / "GR_0.stl")
    transform = normalization_transform("GR_0.stl", mesh, config.units.stl_to_m)
    pairs = detect_anchor_pairs(mesh, transform)
    assert len(pairs) == 7
    assert all(pair.source == "section_circle_fit" for pair in pairs)
    y = np.asarray([pair.left_m[1] for pair in pairs])
    assert np.all(np.diff(y) > 0)
    assert np.ptp(np.diff(y)) < 2.0e-4
    for pair in pairs:
        assert np.allclose(pair.left_m[1:], pair.right_m[1:], atol=2.5e-4)
        assert np.isclose(pair.left_m[0], -pair.right_m[0], atol=5.0e-4)
        assert 0.0013 <= pair.diameter_m <= 0.0017
