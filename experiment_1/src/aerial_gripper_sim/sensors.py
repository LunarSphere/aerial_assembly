"""Simulation measurements and numerical safety checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .config import AppConfig
from .scene_builder import BuiltScene


class SimulationSafetyError(RuntimeError):
    """Raised when continuing would make output physically meaningless."""


@dataclass
class SensorSuite:
    scene: BuiltScene
    config: AppConfig
    initial_payload_position: np.ndarray

    @classmethod
    def create(cls, scene: BuiltScene, config: AppConfig) -> "SensorSuite":
        if scene.payload_body_id >= 0:
            initial = np.asarray(scene.data.xpos[scene.payload_body_id]).copy()
        else:
            initial = np.zeros(3)
        return cls(scene, config, initial)

    def sample(self) -> dict[str, Any]:
        model = self.scene.model
        data = self.scene.data
        string_samples = self.scene.string_model.sample(
            model, data, self.scene.payload_geom_ids
        )
        payload_position = (
            np.asarray(data.xpos[self.scene.payload_body_id]).copy()
            if self.scene.payload_body_id >= 0
            else np.zeros(3)
        )
        payload_quat = (
            np.asarray(data.xquat[self.scene.payload_body_id]).copy()
            if self.scene.payload_body_id >= 0
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        gripper_force, gripper_torque = _free_body_constraint_wrench(
            model, data, self.scene.gripper_body_id
        )
        payload_force, payload_torque = _free_body_constraint_wrench(
            model, data, self.scene.payload_body_id
        )
        washer_samples = (
            self.scene.washer_model.sample(
                model, data, peg_bottom_z=float(payload_position[2])
            )
            if self.scene.washer_model is not None and self.scene.payload_body_id >= 0
            else []
        )
        tensions = [sample.mean_tension_n for sample in string_samples]
        contacts = sum(sample.contact_count for sample in string_samples)
        insertion_depths = [sample.insertion_depth_m for sample in washer_samples]
        measurement = {
            "time_s": float(data.time),
            "payload_position_m": payload_position.tolist(),
            "payload_quaternion_wxyz": payload_quat.tolist(),
            "payload_lift_m": float(payload_position[2] - self.initial_payload_position[2]),
            "gripper_position_m": np.asarray(
                data.xpos[self.scene.gripper_body_id]
            ).tolist(),
            "gripper_force_n": gripper_force.tolist(),
            "gripper_torque_nm": gripper_torque.tolist(),
            "payload_constraint_force_n": payload_force.tolist(),
            "payload_constraint_torque_nm": payload_torque.tolist(),
            "total_string_tension_n": float(sum(tensions)),
            "mean_string_tensions_n": tensions,
            "peak_string_tensions_n": [
                sample.peak_tension_n for sample in string_samples
            ],
            "strings_carrying_load": int(sum(value > 0.01 for value in tensions)),
            "string_payload_contacts": contacts,
            "string_contact_impulse_ns": float(
                sum(sample.contact_impulse_ns for sample in string_samples)
            ),
            "string_slack_m": [sample.slack_m for sample in string_samples],
            "minimum_hook_distances_m": [
                sample.minimum_payload_distance_m for sample in string_samples
            ],
            "minimum_insertion_depth_m": (
                float(min(insertion_depths)) if insertion_depths else 0.0
            ),
            "insertion_depths_m": insertion_depths,
            "washer_strain_proxy": [
                sample.strain_proxy for sample in washer_samples
            ],
            "washer_reaction_force_n": float(
                sum(sample.reaction_force_n for sample in washer_samples)
            ),
            "contact_count": int(data.ncon),
        }
        return measurement

    def check_safety(self) -> None:
        model = self.scene.model
        data = self.scene.data
        arrays = (data.qpos, data.qvel, data.qacc)
        if not all(np.all(np.isfinite(array)) for array in arrays):
            raise SimulationSafetyError("NaN or Inf detected in simulation state")
        mujoco.mj_energyPos(model, data)
        mujoco.mj_energyVel(model, data)
        kinetic = float(data.energy[1])
        if not np.isfinite(kinetic) or kinetic > self.config.simulation.max_kinetic_energy_j:
            raise SimulationSafetyError(
                f"Exploding kinetic energy: {kinetic:.6g} J exceeds "
                f"{self.config.simulation.max_kinetic_energy_j:.6g} J"
            )
        gripper_error = float(
            np.linalg.norm(
                data.xpos[self.scene.gripper_body_id]
                - data.mocap_pos[self.scene.gripper_mocap_id]
            )
        )
        if gripper_error > self.config.simulation.max_constraint_error_m:
            raise SimulationSafetyError(
                f"Gripper command constraint error {gripper_error:.6g} m exceeds "
                f"{self.config.simulation.max_constraint_error_m:.6g}"
            )
        if data.ncon:
            penetration = float(min(data.contact[index].dist for index in range(data.ncon)))
            if penetration < -0.0015:
                raise SimulationSafetyError(
                    f"Contact penetration {penetration:.6g} m indicates tunneling"
                )
        active_warnings = [
            (index, item.number, item.lastinfo)
            for index, item in enumerate(data.warning)
            if item.number
        ]
        if active_warnings:
            raise SimulationSafetyError(f"MuJoCo warnings: {active_warnings}")


def _free_body_constraint_wrench(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return the generalized constraint wrench for a body's free joint."""
    if body_id < 0 or int(model.body_dofnum[body_id]) < 6:
        return np.zeros(3), np.zeros(3)
    address = int(model.body_dofadr[body_id])
    wrench = np.asarray(data.qfrc_constraint[address : address + 6]).copy()
    return wrench[:3], wrench[3:]
