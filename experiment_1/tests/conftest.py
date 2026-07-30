from __future__ import annotations

from pathlib import Path

import pytest

from aerial_gripper_sim.config import AppConfig


@pytest.fixture(scope="session")
def config() -> AppConfig:
    return AppConfig.load(Path("configs/default.yaml"))


@pytest.fixture(scope="session")
def fast_config(config: AppConfig) -> AppConfig:
    return config.with_overrides(
        [
            "strings.segments_per_string=16",
            "output.sample_hz=100",
        ]
    )
