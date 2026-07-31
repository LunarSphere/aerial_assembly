from __future__ import annotations

import numpy as np

from aerial_gripper_sim.config import AppConfig
from aerial_gripper_sim.controller import ScenarioController


def test_takeup_uses_measured_support_after_capture(config: AppConfig) -> None:
    controller = ScenarioController(
        config,
        "pick_and_place",
        np.array([0.0, 0.0, 0.13]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.12, 0.0]),
    )
    measurements = {
        "string_payload_contacts": 25,
        "total_string_tension_n": 0.708,
        "captured_strings": 1,
        "taut_strings": 3,
        "payload_lift_m": 0.023,
    }

    assert controller._takeup_support_detected(measurements)


def test_takeup_rejects_contact_without_payload_support(config: AppConfig) -> None:
    controller = ScenarioController(
        config,
        "pick_and_place",
        np.array([0.0, 0.0, 0.13]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.12, 0.0]),
    )
    measurements = {
        "string_payload_contacts": 25,
        "total_string_tension_n": 0.708,
        "captured_strings": 4,
        "taut_strings": 3,
        "payload_lift_m": 0.0,
    }

    assert not controller._takeup_support_detected(measurements)
