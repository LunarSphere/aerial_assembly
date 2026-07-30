"""Scenario metrics and mechanics-derived acceptance checks."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .config import AppConfig


def compute_metrics(
    records: list[dict[str, Any]],
    scenario: str,
    config: AppConfig,
    controller_success: bool,
) -> dict[str, Any]:
    if not records:
        return {"success": False, "failure_reason": "no telemetry records"}
    positions = np.asarray([item["payload_position_m"] for item in records])
    quaternions = np.asarray([item["payload_quaternion_wxyz"] for item in records])
    tensions = np.asarray([item["mean_string_tensions_n"] for item in records])
    if tensions.ndim == 1:
        tensions = tensions[:, None]
    forces = np.asarray([item["gripper_force_n"] for item in records])
    torques = np.asarray([item["gripper_torque_nm"] for item in records])
    contacts = np.asarray([item["string_payload_contacts"] for item in records])
    gripper_positions = np.asarray(
        [item["gripper_position_m"] for item in records]
    )
    times = np.asarray([item["time_s"] for item in records])
    initial_z = positions[0, 2]
    lift = positions[:, 2] - initial_z
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    euler_deg = rotations.as_euler("xyz", degrees=True)
    insertion = [
        min(item["insertion_depths_m"])
        for item in records
        if item["insertion_depths_m"]
    ]
    zero_contact_duration = _trailing_clear_duration(times, contacts)
    supported_duration = _maximum_duration(
        times,
        (lift >= config.metrics.pickup_min_lift_m) & (contacts > 0),
    )
    pickup_success = bool(
        np.max(lift) >= config.metrics.pickup_min_lift_m
        and supported_duration
        >= config.metrics.pickup_hold_s - 1.0 / config.output.sample_hz
        and np.max(np.abs(euler_deg[:, :2])) <= config.metrics.max_payload_tilt_deg
    )
    target_xy = (
        np.array([0.0, 0.12])
        if scenario in {"full_cycle", "release_only"}
        else np.zeros(2)
    )
    placed_translation_error = float(np.linalg.norm(positions[-1, :2] - target_xy))
    placed_rotation_error = float(np.max(np.abs(euler_deg[-1, :2])))
    payload_constraint_forces = np.asarray(
        [item["payload_constraint_force_n"] for item in records]
    )
    peak_vertical_constraint_force = float(
        np.max(np.abs(payload_constraint_forces[:, 2]))
    )
    placement_success = bool(
        insertion
        and max(insertion) >= 0.003
        and placed_translation_error
        <= config.metrics.placed_translation_tolerance_m
        and placed_rotation_error
        <= config.metrics.placed_rotation_tolerance_deg
        and peak_vertical_constraint_force
        <= config.controller.max_downward_force_n * 1.02
    )
    separation = np.linalg.norm(gripper_positions - positions, axis=1)
    separation_increase = float(separation[-1] - separation[0])
    never_contacted = bool(np.max(contacts) == 0)
    release_success = bool(
        zero_contact_duration >= config.metrics.release_clear_s
        or (
            never_contacted
            and separation_increase >= config.controller.release_distance_m * 0.5
        )
    )
    excess_vertical_force = np.maximum(
        np.abs(payload_constraint_forces[:, 2]) - config.payload.mass_kg * 9.81,
        0.0,
    )
    peak_retention_force = float(np.max(excess_vertical_force))
    pullout_success = bool(
        placement_success
        and peak_retention_force >= config.metrics.retention_force_epsilon_n
    )
    relevant = {
        "single_string_ramp_test": bool(np.max(contacts) > 0),
        "seven_string_pickup": pickup_success,
        "washer_insertion_test": placement_success,
        "washer_pullout_test": pullout_success,
        "placement_only": placement_success,
        "release_only": release_success,
        "full_cycle": pickup_success and placement_success and release_success,
    }.get(scenario, controller_success)
    if tensions.shape[1]:
        denominator = np.maximum(np.mean(tensions, axis=1), 1.0e-9)
        load_imbalance = np.std(tensions, axis=1) / denominator
        peak_tensions = np.max(tensions, axis=0).tolist()
        mean_tensions = np.mean(tensions, axis=0).tolist()
        peak_imbalance = float(np.max(load_imbalance))
    else:
        peak_tensions = []
        mean_tensions = []
        peak_imbalance = 0.0
    insertion_forces = [
        abs(item["payload_constraint_force_n"][2])
        for item in records
        if item.get("state") == "PRESS_INSERT"
    ]
    hook_distances = [
        min(item["minimum_hook_distances_m"])
        for item in records
        if item["minimum_hook_distances_m"]
    ]
    return {
        "success": bool(controller_success and relevant),
        "pickup_success": pickup_success,
        "placement_success": placement_success,
        "release_success": release_success,
        "number_of_strings_carrying_load_peak": int(
            max(item["strings_carrying_load"] for item in records)
        ),
        "peak_tension_per_string_n": peak_tensions,
        "mean_tension_per_string_n": mean_tensions,
        "peak_load_sharing_imbalance": peak_imbalance,
        "peak_gripper_force_n": float(np.max(np.linalg.norm(forces, axis=1))),
        "peak_gripper_torque_nm": float(np.max(np.linalg.norm(torques, axis=1))),
        "payload_initial_position_m": positions[0].tolist(),
        "payload_final_position_m": positions[-1].tolist(),
        "maximum_lift_height_m": float(np.max(lift)),
        "supported_hold_duration_s": supported_duration,
        "payload_peak_roll_pitch_yaw_deg": np.max(np.abs(euler_deg), axis=0).tolist(),
        "placed_translation_error_m": placed_translation_error,
        "placed_rotation_error_deg": placed_rotation_error,
        "minimum_peg_insertion_depth_m": float(max(insertion)) if insertion else 0.0,
        "peak_washer_reaction_force_n": float(
            max(item["washer_reaction_force_n"] for item in records)
        ),
        "peak_payload_constraint_force_n": float(
            np.max(np.linalg.norm(payload_constraint_forces, axis=1))
        ),
        "peak_insertion_force_n": max(insertion_forces, default=0.0),
        "peak_retention_force_above_weight_n": peak_retention_force,
        "peak_washer_strain_proxy": float(
            max(
                (
                    max(item["washer_strain_proxy"])
                    for item in records
                    if item["washer_strain_proxy"]
                ),
                default=0.0,
            )
        ),
        "minimum_hook_distance_m": min(hook_distances, default=1.0),
        "release_clear_duration_s": zero_contact_duration,
        "gripper_payload_separation_increase_m": separation_increase,
        "release_time_s": _first_clear_time(
            times, contacts, config.metrics.release_clear_s
        ),
        "remaining_string_block_contacts": int(contacts[-1]),
        "target_disturbance_m": 0.0,
        "simulated_duration_s": float(times[-1] - times[0]),
        "thresholds": {
            "pickup_min_lift_m": config.metrics.pickup_min_lift_m,
            "pickup_hold_s": config.metrics.pickup_hold_s,
            "max_payload_tilt_deg": config.metrics.max_payload_tilt_deg,
            "minimum_insertion_depth_m": 0.003,
            "release_clear_s": config.metrics.release_clear_s,
            "placed_translation_tolerance_m": config.metrics.placed_translation_tolerance_m,
            "placed_rotation_tolerance_deg": config.metrics.placed_rotation_tolerance_deg,
            "retention_force_epsilon_n": config.metrics.retention_force_epsilon_n,
            "max_downward_force_n": config.controller.max_downward_force_n,
        },
    }


def _maximum_duration(times: np.ndarray, condition: np.ndarray) -> float:
    maximum = 0.0
    start: float | None = None
    for time, active in zip(times, condition, strict=True):
        if active and start is None:
            start = float(time)
        elif not active and start is not None:
            maximum = max(maximum, float(time) - start)
            start = None
    if start is not None:
        maximum = max(maximum, float(times[-1]) - start)
    return maximum


def _first_clear_time(
    times: np.ndarray, contacts: np.ndarray, required_duration_s: float
) -> float | None:
    start: float | None = None
    seen_contact = False
    for time, count in zip(times, contacts, strict=True):
        if count != 0:
            seen_contact = True
            start = None
        elif seen_contact and start is None:
            start = float(time)
        if start is not None and float(time) - start >= required_duration_s:
            return start
    return None


def _trailing_clear_duration(times: np.ndarray, contacts: np.ndarray) -> float:
    nonzero = np.flatnonzero(contacts > 0)
    if not len(nonzero):
        return 0.0
    last_contact = int(nonzero[-1])
    return float(times[-1] - times[last_contact])
