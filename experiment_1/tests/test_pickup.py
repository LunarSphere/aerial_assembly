from __future__ import annotations

import pytest

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="known hook-contact regression: nominal payload lift is below 20 mm",
)
def test_nominal_seven_string_pickup(config: AppConfig, tmp_path) -> None:
    reduced = config.with_overrides(["strings.segments_per_string=24"])
    result = run_simulation(
        reduced, "seven_string_pickup", output_dir=tmp_path / "pickup"
    )
    assert result["success"], result["failure_reason"]
