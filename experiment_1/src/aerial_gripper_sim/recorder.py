"""Telemetry persistence, plots, summary figures, and failure bundles."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import AppConfig
from .controller import ScenarioController
from .metrics import compute_metrics
from .scene_builder import BuiltScene


class RunRecorder:
    def __init__(self, output_dir: Path, config: AppConfig, scene: BuiltScene):
        self.output_dir = output_dir
        self.config = config
        self.scene = scene
        self.records: list[dict[str, Any]] = []
        self.wall_start = time.perf_counter()
        output_dir.mkdir(parents=True, exist_ok=True)

    def append(self, sample: dict[str, Any], state: str) -> None:
        row = dict(sample)
        row["state"] = state
        self.records.append(row)

    def finalize(
        self,
        controller: ScenarioController,
        failure_reason: str | None = None,
    ) -> dict[str, Any]:
        wall_time = time.perf_counter() - self.wall_start
        controller_success = controller.state.value == "DONE" and failure_reason is None
        metrics = compute_metrics(
            self.records,
            self.scene.scenario,
            self.config,
            controller_success,
        )
        if failure_reason:
            metrics["success"] = False
            metrics["failure_reason"] = failure_reason
        metrics["wall_time_s"] = wall_time
        simulated = metrics.get("simulated_duration_s", 0.0)
        metrics["real_time_factor"] = simulated / wall_time if wall_time else 0.0
        table = _records_to_frame(self.records)
        table.to_csv(self.output_dir / "telemetry.csv", index=False)
        try:
            table.to_parquet(self.output_dir / "telemetry.parquet", index=False)
        except Exception as exc:
            metrics["parquet_warning"] = str(exc)
        result = {
            "scenario": self.scene.scenario,
            "success": metrics["success"],
            "failure_reason": failure_reason or controller.failure_reason,
            "metrics": metrics,
            "resolved_config": self.config.to_dict(),
            "asset_cache_key": self.scene.manifest["cache_key"],
            "fallbacks": {
                "string_backend": self.config.strings.backend,
                "collision_backend": self.scene.manifest["collision_proxy"]["backend"],
                "collision_fallback_reason": self.scene.manifest["collision_proxy"].get(
                    "fallback_reason"
                ),
                "washer_mode": self.config.washer.mode,
                "engagement_auto_selection": (
                    "+Y" if self.config.controller.engagement_sign == "auto" else None
                ),
            },
            "transitions": [asdict(item) for item in controller.transitions],
            "reproduction_command": (
                f"uv run aerial-gripper-sim run --scenario {self.scene.scenario} "
                "--config configs/default.yaml"
            ),
        }
        (self.output_dir / "results.json").write_text(
            json.dumps(_finite_json(result), indent=2)
        )
        (self.output_dir / "model.xml").write_text(self.scene.xml)
        self._plots(table, result)
        return result

    def failure_bundle(self, reason: str) -> None:
        bundle = self.output_dir / "failure_bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        (bundle / "reason.txt").write_text(reason)
        (bundle / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2)
        )
        (bundle / "model.xml").write_text(self.scene.xml)
        np.savez_compressed(
            bundle / "last_valid_state.npz",
            qpos=np.asarray(self.scene.data.qpos),
            qvel=np.asarray(self.scene.data.qvel),
            time_s=np.asarray([self.scene.data.time]),
        )
        tail = self.records[-200:]
        (bundle / "last_valid_telemetry.json").write_text(
            json.dumps(_finite_json(tail), indent=2)
        )

    def _plots(self, frame: pd.DataFrame, result: dict[str, Any]) -> None:
        if frame.empty:
            return
        figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(frame["time_s"], frame["payload_z_m"], label="payload Z")
        axes[0].set_ylabel("Height (m)")
        axes[0].grid(alpha=0.3)
        axes[1].plot(frame["time_s"], frame["total_string_tension_n"])
        axes[1].set_ylabel("Total tension (N)")
        axes[1].grid(alpha=0.3)
        axes[2].plot(frame["time_s"], frame["string_payload_contacts"])
        axes[2].set_ylabel("String contacts")
        axes[2].set_xlabel("Time (s)")
        axes[2].grid(alpha=0.3)
        figure.suptitle(
            f"{self.scene.scenario}: {'PASS' if result['success'] else 'FAIL'}"
        )
        figure.tight_layout()
        figure.savefig(self.output_dir / "summary.png", dpi=160)
        plt.close(figure)


def _records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in records:
        position = item["payload_position_m"]
        rows.append(
            {
                "time_s": item["time_s"],
                "state": item["state"],
                "payload_x_m": position[0],
                "payload_y_m": position[1],
                "payload_z_m": position[2],
                "payload_lift_m": item["payload_lift_m"],
                "total_string_tension_n": item["total_string_tension_n"],
                "strings_carrying_load": item["strings_carrying_load"],
                "taut_strings": item["taut_strings"],
                "captured_strings": item["captured_strings"],
                "string_payload_contacts": item["string_payload_contacts"],
                "max_string_endpoint_error_m": item[
                    "max_string_endpoint_error_m"
                ],
                "max_string_axial_strain_abs": item[
                    "max_string_axial_strain_abs"
                ],
                "minimum_insertion_depth_m": item["minimum_insertion_depth_m"],
                "washer_reaction_force_n": item["washer_reaction_force_n"],
                "payload_constraint_force_norm_n": float(
                    np.linalg.norm(item["payload_constraint_force_n"])
                ),
                "payload_constraint_force_z_n": item[
                    "payload_constraint_force_n"
                ][2],
                "contact_count": item["contact_count"],
                "gripper_force_norm_n": float(
                    np.linalg.norm(item["gripper_force_n"])
                ),
                "gripper_torque_norm_nm": float(
                    np.linalg.norm(item["gripper_torque_nm"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _finite_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _finite_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_finite_json(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
