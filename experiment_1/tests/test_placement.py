from __future__ import annotations

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig


def test_placement_fixture_aligns_and_inserts(
    config: AppConfig, tmp_path
) -> None:
    result = run_simulation(
        config, "placement_only", output_dir=tmp_path / "placement"
    )
    assert result["success"], result["failure_reason"]
    metrics = result["metrics"]
    assert metrics["placed_translation_error_m"] <= 0.003
    assert metrics["placed_rotation_error_deg"] <= 5.0
    assert metrics["minimum_peg_insertion_depth_m"] >= 0.003
