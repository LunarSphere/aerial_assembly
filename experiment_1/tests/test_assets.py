from __future__ import annotations

import numpy as np
from pathlib import Path

from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.geometry import GeometryPipeline


def test_raw_asset_extents_and_topology(config: AppConfig) -> None:
    report = GeometryPipeline(config).inspect(write=False)["assets"]
    expected_mm = {
        "GR_0.stl": (106.472, 20.0, 44.960),
        "BB_0.stl": (35.0, 40.0, 48.0),
        "SLW_0.stl": (9.0, 8.987, 1.2),
    }
    for name, expected in expected_mm.items():
        actual = np.asarray(report[name]["extents_m"]) * 1000.0
        assert np.allclose(actual, expected, atol=0.08), (name, actual)
        assert report[name]["watertight"]
    assert report["Ghast_0.stl"]["connected_component_count"] == 5
    assert report["Ghast_0.stl"]["watertight"]


def test_scale_is_detected_as_millimetres(config: AppConfig) -> None:
    report = GeometryPipeline(config).inspect(write=False)
    assert {
        asset["inferred_stl_to_m"] for asset in report["assets"].values()
    } == {0.001}


def test_collision_proxy_preserves_nonconvex_volume(config: AppConfig) -> None:
    proxy = GeometryPipeline(config).preprocess()["collision_proxy"]
    validation = proxy["validation"]
    assert validation["valid"], validation["errors"]
    assert validation["part_count"] > 1
    assert validation["proxy_fill_ratio"] < 0.5
    assert proxy["excluded_hook_intrusion_parts"]
    assert Path(proxy["diagnostic"]).is_file()
