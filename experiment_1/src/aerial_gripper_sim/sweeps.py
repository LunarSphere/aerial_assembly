"""Deterministic robustness sweeps and sensitivity plots."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cli import run_simulation
from .config import AppConfig


def run_robustness_sweep(
    base_config: AppConfig, output_dir: Path, trials: int
) -> dict[str, Any]:
    if trials <= 0:
        raise ValueError("trials must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base_config.simulation.seed)
    rows: list[dict[str, Any]] = []
    for trial in range(trials):
        values = {
            "payload_x_offset_m": float(rng.uniform(-0.003, 0.003)),
            "payload_y_offset_m": float(rng.uniform(-0.003, 0.003)),
            "payload_yaw_deg": float(rng.uniform(-8.0, 8.0)),
            "gripper_height_error_m": float(rng.uniform(-0.002, 0.002)),
            "engagement_distance_m": float(rng.uniform(0.005, 0.010)),
            "engagement_speed_m_s": float(rng.uniform(0.007, 0.014)),
            "release_angle_deg": float(rng.uniform(-30.0, 30.0)),
            "pretension_n": float(rng.uniform(0.06, 0.14)),
            "string_radius_m": float(rng.uniform(0.00025, 0.00038)),
            "axial_stiffness_n_per_m": float(rng.uniform(800.0, 1600.0)),
            "string_friction": float(rng.uniform(0.35, 0.75)),
            "payload_mass_kg": float(rng.uniform(0.018, 0.035)),
            "washer_stiffness_nm_rad": float(rng.uniform(0.0035, 0.009)),
            "washer_friction": float(rng.uniform(0.55, 1.0)),
            "peg_misalignment_m": float(rng.uniform(0.0, 0.002)),
            "press_force_limit_n": float(rng.uniform(0.65, 1.3)),
        }
        angle = np.deg2rad(values["release_angle_deg"])
        peg_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        overrides = [
            (
                "perturbations.payload_xy_offset_m="
                f"[{values['payload_x_offset_m']},{values['payload_y_offset_m']}]"
            ),
            f"perturbations.payload_yaw_deg={values['payload_yaw_deg']}",
            (
                "perturbations.gripper_height_error_m="
                f"{values['gripper_height_error_m']}"
            ),
            (
                "perturbations.peg_xy_misalignment_m="
                f"[{values['peg_misalignment_m'] * np.cos(peg_angle)},"
                f"{values['peg_misalignment_m'] * np.sin(peg_angle)}]"
            ),
            f"strings.pretension_n={values['pretension_n']}",
            f"strings.radius_m={values['string_radius_m']}",
            f"strings.axial_stiffness_n_per_m={values['axial_stiffness_n_per_m']}",
            f"strings.friction=[{values['string_friction']},0.01,0.001]",
            f"payload.mass_kg={values['payload_mass_kg']}",
            f"washer.effective_stiffness_nm_rad={values['washer_stiffness_nm_rad']}",
            f"washer.peg_friction=[{values['washer_friction']},0.02,0.002]",
            f"controller.engagement_distance_m={values['engagement_distance_m']}",
            f"controller.engagement_speed_m_s={values['engagement_speed_m_s']}",
            f"controller.release_vector=[{np.sin(angle)},{-np.cos(angle)},0]",
            f"controller.max_downward_force_n={values['press_force_limit_n']}",
            f"simulation.seed={base_config.simulation.seed + trial}",
        ]
        config = base_config.with_overrides(overrides)
        result = run_simulation(
            config,
            "full_cycle",
            output_dir=output_dir / f"trial_{trial:04d}",
        )
        rows.append(
            {
                "trial": trial,
                **values,
                "success": result["success"],
                "failure_reason": result.get("failure_reason"),
                **{
                    f"metric_{key}": value
                    for key, value in result["metrics"].items()
                    if isinstance(value, (int, float, bool))
                },
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "sweep.csv", index=False)
    frame.to_parquet(output_dir / "sweep.parquet", index=False)
    _plot_sensitivity(frame, output_dir / "sensitivity.png")
    summary = {
        "trials": trials,
        "successes": int(frame["success"].sum()),
        "success_rate": float(frame["success"].mean()),
        "seed": base_config.simulation.seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _plot_sensitivity(frame: pd.DataFrame, path: Path) -> None:
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=["trial"], errors="ignore"
    )
    correlations = numeric.corr(numeric_only=True)["success"].drop(
        "success", errors="ignore"
    )
    correlations = correlations.reindex(
        correlations.abs().sort_values(ascending=False).index
    ).head(14)
    figure, axis = plt.subplots(figsize=(10, 6))
    correlations.sort_values().plot.barh(ax=axis)
    axis.set_xlabel("Pearson correlation with success")
    axis.set_title("Robustness sweep sensitivity (screening metric)")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
