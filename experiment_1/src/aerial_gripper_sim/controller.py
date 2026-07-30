"""Deterministic event-driven gripper trajectory state machine."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


class ControllerState(StrEnum):
    RESET = "RESET"
    SETTLE = "SETTLE"
    APPROACH_PAYLOAD = "APPROACH_PAYLOAD"
    DESCEND_TO_RAMPS = "DESCEND_TO_RAMPS"
    ENGAGE_FORWARD = "ENGAGE_FORWARD"
    TENSION_CHECK = "TENSION_CHECK"
    LIFT = "LIFT"
    HOLD_TEST = "HOLD_TEST"
    TRANSPORT_TO_TARGET = "TRANSPORT_TO_TARGET"
    ALIGN_PEGS = "ALIGN_PEGS"
    PRESS_INSERT = "PRESS_INSERT"
    VERIFY_RETENTION = "VERIFY_RETENTION"
    LOWER_FOR_SLACK = "LOWER_FOR_SLACK"
    RELEASE_TRANSLATE = "RELEASE_TRANSLATE"
    VERIFY_RELEASE = "VERIFY_RELEASE"
    RETREAT = "RETREAT"
    DONE = "DONE"
    FAIL = "FAIL"


ROUTES: dict[str, list[ControllerState]] = {
    "single_string_ramp_test": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.DESCEND_TO_RAMPS,
        ControllerState.ENGAGE_FORWARD,
        ControllerState.HOLD_TEST,
        ControllerState.DONE,
    ],
    "seven_string_pickup": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.APPROACH_PAYLOAD,
        ControllerState.DESCEND_TO_RAMPS,
        ControllerState.ENGAGE_FORWARD,
        ControllerState.TENSION_CHECK,
        ControllerState.LIFT,
        ControllerState.HOLD_TEST,
        ControllerState.DONE,
    ],
    "washer_insertion_test": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.ALIGN_PEGS,
        ControllerState.PRESS_INSERT,
        ControllerState.VERIFY_RETENTION,
        ControllerState.DONE,
    ],
    "washer_pullout_test": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.PRESS_INSERT,
        ControllerState.VERIFY_RETENTION,
        ControllerState.RETREAT,
        ControllerState.DONE,
    ],
    "placement_only": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.ALIGN_PEGS,
        ControllerState.PRESS_INSERT,
        ControllerState.VERIFY_RETENTION,
        ControllerState.DONE,
    ],
    "release_only": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.LOWER_FOR_SLACK,
        ControllerState.RELEASE_TRANSLATE,
        ControllerState.VERIFY_RELEASE,
        ControllerState.RETREAT,
        ControllerState.DONE,
    ],
    "full_cycle": [
        ControllerState.RESET,
        ControllerState.SETTLE,
        ControllerState.APPROACH_PAYLOAD,
        ControllerState.DESCEND_TO_RAMPS,
        ControllerState.ENGAGE_FORWARD,
        ControllerState.TENSION_CHECK,
        ControllerState.LIFT,
        ControllerState.HOLD_TEST,
        ControllerState.TRANSPORT_TO_TARGET,
        ControllerState.ALIGN_PEGS,
        ControllerState.PRESS_INSERT,
        ControllerState.VERIFY_RETENTION,
        ControllerState.LOWER_FOR_SLACK,
        ControllerState.RELEASE_TRANSLATE,
        ControllerState.VERIFY_RELEASE,
        ControllerState.RETREAT,
        ControllerState.DONE,
    ],
}


@dataclass
class Transition:
    time_s: float
    previous: str
    current: str
    reason: str


@dataclass
class ControllerCommand:
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    state: ControllerState


class ScenarioController:
    def __init__(
        self,
        config: AppConfig,
        scenario: str,
        initial_position: np.ndarray,
        initial_quaternion: np.ndarray,
        target_position: np.ndarray,
    ):
        if scenario not in ROUTES:
            raise ValueError(f"No controller route for scenario {scenario!r}")
        self.config = config
        self.scenario = scenario
        self.route = ROUTES[scenario]
        self.route_index = 0
        self.state = self.route[0]
        self.state_start_s = 0.0
        self.start_position = np.asarray(initial_position, dtype=float).copy()
        self.target_position = self.start_position.copy()
        self.current_position = self.start_position.copy()
        self.quaternion = np.asarray(initial_quaternion, dtype=float).copy()
        self.stack_target = np.asarray(target_position, dtype=float)
        self.duration_s = 0.0
        self.timeout_s = 1.0
        self.trajectory_progress_s = 0.0
        self.last_step_s = 0.0
        self.failure_reason: str | None = None
        self.transitions: list[Transition] = []
        self.release_clear_start_s: float | None = None
        self.engagement_sign = self._engagement_sign()
        self._enter_state(0.0, "controller initialized", {})

    @property
    def finished(self) -> bool:
        return self.state in {ControllerState.DONE, ControllerState.FAIL}

    def fail(self, time_s: float, reason: str) -> None:
        if self.finished:
            return
        previous = self.state
        self.state = ControllerState.FAIL
        self.failure_reason = reason
        self.transitions.append(
            Transition(time_s, previous.value, self.state.value, reason)
        )
        LOGGER.error("%s -> FAIL: %s", previous, reason)

    def step(self, time_s: float, measurements: dict[str, Any]) -> ControllerCommand:
        if self.finished:
            return ControllerCommand(self.current_position, self.quaternion, self.state)
        elapsed = time_s - self.state_start_s
        force_scale = self._force_scale(measurements)
        step_duration = max(0.0, time_s - self.last_step_s)
        self.last_step_s = time_s
        self.trajectory_progress_s = min(
            self.trajectory_progress_s + step_duration * force_scale,
            self.duration_s,
        )
        if self.duration_s > 0:
            fraction = min(1.0, self.trajectory_progress_s / self.duration_s)
            smooth = _quintic(fraction)
            self.current_position = (
                self.start_position
                + smooth * (self.target_position - self.start_position)
            )
        if self._success(elapsed, measurements):
            self._advance(time_s, "measured success condition", measurements)
        elif elapsed > self.timeout_s:
            self.fail(
                time_s,
                f"{self.state.value} timed out after {elapsed:.3f}s; "
                f"measurements={_compact(measurements)}",
            )
        return ControllerCommand(self.current_position, self.quaternion, self.state)

    def _advance(
        self, time_s: float, reason: str, measurements: dict[str, Any]
    ) -> None:
        previous = self.state
        self.route_index += 1
        self.state = self.route[self.route_index]
        self.transitions.append(
            Transition(time_s, previous.value, self.state.value, reason)
        )
        LOGGER.info("%s -> %s: %s", previous, self.state, reason)
        self._enter_state(time_s, reason, measurements)

    def _enter_state(
        self, time_s: float, reason: str, measurements: dict[str, Any]
    ) -> None:
        _ = reason
        self.state_start_s = time_s
        self.last_step_s = time_s
        self.trajectory_progress_s = 0.0
        self.start_position = self.current_position.copy()
        self.target_position = self.current_position.copy()
        state = self.state
        speed = self.config.controller.approach_speed_m_s
        hold = 0.1
        if state == ControllerState.RESET:
            hold = 0.01
        elif state == ControllerState.SETTLE:
            hold = max(0.2, self.config.strings.settle_time_s)
        elif state == ControllerState.APPROACH_PAYLOAD:
            self.target_position[2] = max(self.target_position[2], 0.10)
            speed = self.config.controller.approach_speed_m_s
        elif state == ControllerState.DESCEND_TO_RAMPS:
            self.target_position[2] -= 0.032
            speed = self.config.controller.approach_speed_m_s
        elif state == ControllerState.ENGAGE_FORWARD:
            axis = 0 if self.config.controller.engagement_axis == "x" else 1
            self.target_position[axis] += (
                self.engagement_sign * self.config.controller.engagement_distance_m
            )
            # Follow the measured ramp drop while translating so the taut cable
            # can settle into the undercut instead of being dragged across the
            # hook tip at a fixed elevation.
            self.target_position[2] -= 0.010
            speed = self.config.controller.engagement_speed_m_s
        elif state == ControllerState.TENSION_CHECK:
            self.target_position[2] += 0.002
            speed = self.config.controller.lift_speed_m_s
        elif state == ControllerState.LIFT:
            if self.scenario == "washer_pullout_test":
                self.target_position[2] += 0.015
            else:
                self.target_position[2] += self.config.controller.pickup_lift_m
            speed = self.config.controller.lift_speed_m_s
        elif state == ControllerState.HOLD_TEST:
            hold = self.config.controller.hold_duration_s
        elif state == ControllerState.TRANSPORT_TO_TARGET:
            self.target_position[:2] = self.stack_target[:2]
            speed = self.config.controller.approach_speed_m_s
        elif state == ControllerState.ALIGN_PEGS:
            self.target_position[:2] = self.stack_target[:2]
            speed = self.config.controller.approach_speed_m_s
        elif state == ControllerState.PRESS_INSERT:
            self.target_position[2] -= 0.018
            speed = self.config.controller.press_speed_m_s
        elif state == ControllerState.VERIFY_RETENTION:
            hold = 0.25
        elif state == ControllerState.LOWER_FOR_SLACK:
            self.target_position[2] -= 0.005
            speed = self.config.controller.press_speed_m_s
        elif state == ControllerState.RELEASE_TRANSLATE:
            vector = np.asarray(self.config.controller.release_vector, dtype=float)
            norm = np.linalg.norm(vector)
            if norm == 0:
                raise ValueError("controller.release_vector must be nonzero")
            self.target_position += (
                vector / norm * self.config.controller.release_distance_m
            )
            speed = self.config.controller.release_speed_m_s
        elif state == ControllerState.VERIFY_RELEASE:
            hold = self.config.metrics.release_clear_s
            self.release_clear_start_s = None
        elif state == ControllerState.RETREAT:
            self.target_position[2] += 0.025
            speed = self.config.controller.approach_speed_m_s
        elif state in {ControllerState.DONE, ControllerState.FAIL}:
            self.duration_s = 0.0
            self.timeout_s = math.inf
            return
        distance = float(np.linalg.norm(self.target_position - self.start_position))
        self.duration_s = hold if distance < 1.0e-9 else max(0.15, distance / speed)
        self.timeout_s = self.duration_s + max(0.75, 0.5 * self.duration_s)

    def _success(self, elapsed: float, measurements: dict[str, Any]) -> bool:
        is_motion = bool(
            np.linalg.norm(self.target_position - self.start_position) > 1.0e-9
        )
        reached = (
            self.trajectory_progress_s >= self.duration_s
            if is_motion
            else elapsed >= self.duration_s
        )
        state = self.state
        if state in {
            ControllerState.RESET,
            ControllerState.SETTLE,
            ControllerState.APPROACH_PAYLOAD,
            ControllerState.TRANSPORT_TO_TARGET,
            ControllerState.ALIGN_PEGS,
            ControllerState.RETREAT,
        }:
            return reached
        if state == ControllerState.DESCEND_TO_RAMPS:
            return reached and measurements.get("string_payload_contacts", 0) > 0
        if state == ControllerState.ENGAGE_FORWARD:
            return reached and measurements.get("string_payload_contacts", 0) > 0
        if state == ControllerState.TENSION_CHECK:
            return (
                reached
                and measurements.get("total_string_tension_n", 0.0) > 0.05
                and measurements.get("strings_carrying_load", 0) >= 3
                and measurements.get("string_payload_contacts", 0) > 0
            )
        if state == ControllerState.LIFT:
            if self.scenario == "washer_pullout_test":
                return reached
            return reached and measurements.get("payload_lift_m", 0.0) >= 0.015
        if state == ControllerState.HOLD_TEST:
            if self.scenario == "single_string_ramp_test":
                return reached and measurements.get("string_payload_contacts", 0) > 0
            return reached and measurements.get("payload_lift_m", 0.0) >= 0.015
        if state == ControllerState.PRESS_INSERT:
            return (
                measurements.get("minimum_insertion_depth_m", 0.0) >= 0.003
                or reached
            )
        if state == ControllerState.VERIFY_RETENTION:
            reaction = max(
                measurements.get("washer_reaction_force_n", 0.0),
                abs(
                    measurements.get(
                        "payload_constraint_force_n", [0.0, 0.0, 0.0]
                    )[2]
                )
                - self.config.payload.mass_kg * 9.81,
            )
            return (
                reached
                and measurements.get("minimum_insertion_depth_m", 0.0) >= 0.002
                and reaction > 1.0e-5
            )
        if state == ControllerState.LOWER_FOR_SLACK:
            baseline = (
                self.config.strings.count * self.config.strings.pretension_n
            )
            return (
                reached
                and (
                    measurements.get("string_payload_contacts", 0) == 0
                    or measurements.get("total_string_tension_n", 0.0)
                    <= baseline + self.config.controller.slack_tension_n
                )
            )
        if state == ControllerState.RELEASE_TRANSLATE:
            return reached
        if state == ControllerState.VERIFY_RELEASE:
            contacts = measurements.get("string_payload_contacts", 0)
            if contacts == 0:
                if self.release_clear_start_s is None:
                    self.release_clear_start_s = elapsed
                return elapsed - self.release_clear_start_s >= self.config.metrics.release_clear_s
            self.release_clear_start_s = None
            return False
        return reached

    def _force_scale(self, measurements: dict[str, Any]) -> float:
        if self.scenario in {
            "washer_insertion_test",
            "washer_pullout_test",
            "placement_only",
        }:
            force = np.asarray(
                measurements.get("payload_constraint_force_n", [0.0, 0.0, 0.0])
            )
            torque = np.asarray(
                measurements.get("payload_constraint_torque_nm", [0.0, 0.0, 0.0])
            )
        else:
            force = np.asarray(
                measurements.get("gripper_force_n", [0.0, 0.0, 0.0])
            )
            torque = np.asarray(
                measurements.get("gripper_torque_nm", [0.0, 0.0, 0.0])
            )
        vertical_limit = (
            self.config.controller.max_downward_force_n
            if self.state == ControllerState.PRESS_INSERT
            else self.config.controller.max_upward_force_n
        )
        ratios = [
            abs(float(force[2])) / vertical_limit,
            float(np.linalg.norm(force[:2]))
            / self.config.controller.max_horizontal_force_n,
            float(np.linalg.norm(torque)) / self.config.controller.max_torque_nm,
        ]
        maximum = max(ratios)
        if maximum >= 1.0:
            return 0.0
        threshold = self.config.controller.force_slowdown_fraction
        if maximum <= threshold:
            return 1.0
        return max(0.05, (1.0 - maximum) / (1.0 - threshold))

    def _engagement_sign(self) -> float:
        setting = self.config.controller.engagement_sign
        if setting == "positive":
            return 1.0
        if setting == "negative":
            return -1.0
        # Asset coordinates and the concept drawing agree on +Y as the ramp entry
        # direction. The runner records this auto-selection for later two-trial
        # calibration sweeps.
        return 1.0


def _quintic(value: float) -> float:
    return 10.0 * value**3 - 15.0 * value**4 + 6.0 * value**5


def _compact(measurements: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "string_payload_contacts",
        "total_string_tension_n",
        "payload_lift_m",
        "minimum_insertion_depth_m",
    )
    return {key: measurements.get(key) for key in keys}
