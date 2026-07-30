from __future__ import annotations

import pytest

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="full cycle is blocked by the documented nominal pickup regression",
)
def test_full_cycle_without_attachment(config: AppConfig, tmp_path) -> None:
    reduced = config.with_overrides(["strings.segments_per_string=24"])
    result = run_simulation(
        reduced, "full_cycle", output_dir=tmp_path / "full_cycle"
    )
    assert result["success"], result["failure_reason"]
