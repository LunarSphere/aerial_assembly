from __future__ import annotations

import pytest

from aerial_gripper_sim.cli import run_simulation
from aerial_gripper_sim.config import AppConfig


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="known release regression: cable middle segments remain frictionally caught",
)
def test_release_clears_strings(config: AppConfig, tmp_path) -> None:
    reduced = config.with_overrides(["strings.segments_per_string=24"])
    result = run_simulation(
        reduced, "release_only", output_dir=tmp_path / "release"
    )
    assert result["success"], result["failure_reason"]
    assert result["metrics"]["remaining_string_block_contacts"] == 0
