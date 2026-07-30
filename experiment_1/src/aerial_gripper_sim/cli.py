"""Command-line interface for preprocessing, scenarios, and robustness sweeps."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Sequence

import mujoco
import numpy as np

from .config import AppConfig, ConfigError
from .controller import ControllerState, ScenarioController
from .geometry import GeometryError, GeometryPipeline
from .recorder import RunRecorder
from .scene_builder import SceneBuilder
from .sensors import SensorSuite, SimulationSafetyError
from .visualization import VideoRecorder, launch_passive

LOGGER = logging.getLogger(__name__)

SCENARIOS = (
    "asset_inspection",
    "single_string_ramp_test",
    "seven_string_pickup",
    "washer_insertion_test",
    "washer_pullout_test",
    "placement_only",
    "release_only",
    "full_cycle",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerial-gripper-sim",
        description="MuJoCo mechanics simulation for an actuatorless string gripper",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-assets", help="inspect immutable raw STL assets and save JSON"
    )
    _add_config_arguments(inspect_parser)

    preprocess_parser = subparsers.add_parser(
        "preprocess", help="normalize assets, detect anchors, and build collision proxies"
    )
    _add_config_arguments(preprocess_parser)
    preprocess_parser.add_argument("--force", action="store_true")

    run_parser = subparsers.add_parser("run", help="run one deterministic scenario")
    _add_config_arguments(run_parser)
    run_parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    run_parser.add_argument("--viewer", action="store_true")
    run_parser.add_argument("--record", type=Path)
    run_parser.add_argument("--output", type=Path)

    sweep_parser = subparsers.add_parser(
        "sweep", help="run deterministic parameter/Monte Carlo robustness trials"
    )
    _add_config_arguments(sweep_parser)
    sweep_parser.add_argument("--output", type=Path, required=True)
    sweep_parser.add_argument("--trials", type=int, default=12)
    return parser


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="override a typed YAML value; may be repeated",
    )


def load_config(arguments: argparse.Namespace) -> AppConfig:
    return AppConfig.load(arguments.config).with_overrides(arguments.overrides)


def run_simulation(
    config: AppConfig,
    scenario: str,
    *,
    output_dir: Path | None = None,
    viewer: bool = False,
    record_path: Path | None = None,
) -> dict:
    if scenario == "asset_inspection":
        report = GeometryPipeline(config).inspect()
        return {"scenario": scenario, "success": True, "inspection": report}
    scene = SceneBuilder(config).build(scenario)
    output = output_dir or config.paths.outputs / scenario
    recorder = RunRecorder(output, config, scene)
    sensors = SensorSuite.create(scene, config)
    control_mocap_id = scene.gripper_mocap_id
    if scenario in {
        "washer_insertion_test",
        "washer_pullout_test",
        "placement_only",
    }:
        body_id = mujoco.mj_name2id(
            scene.model, mujoco.mjtObj.mjOBJ_BODY, "payload_target"
        )
        control_mocap_id = int(scene.model.body_mocapid[body_id])
    initial_position = np.asarray(scene.data.mocap_pos[control_mocap_id]).copy()
    initial_quat = np.asarray(scene.data.mocap_quat[control_mocap_id]).copy()
    controller_target = scene.target_position_m.copy()
    controller_target[:2] += np.asarray(
        config.perturbations.peg_xy_misalignment_m
    )
    controller = ScenarioController(
        config,
        scenario,
        initial_position,
        initial_quat,
        controller_target,
    )
    viewer_handle = launch_passive(scene) if viewer else None
    video = (
        VideoRecorder(scene, record_path, config.output.render_fps)
        if record_path is not None
        else None
    )
    sample_interval = max(
        1, round(1.0 / (config.output.sample_hz * config.simulation.timestep_s))
    )
    video_interval = max(
        1, round(1.0 / (config.output.render_fps * config.simulation.timestep_s))
    )
    latest = sensors.sample()
    failure_reason: str | None = None
    step = 0
    try:
        _check_initial_penetration(scene)
        while (
            not controller.finished
            and scene.data.time < config.simulation.duration_limit_s
        ):
            command = controller.step(float(scene.data.time), latest)
            scene.data.mocap_pos[control_mocap_id] = command.position_m
            scene.data.mocap_quat[control_mocap_id] = command.quaternion_wxyz
            mujoco.mj_step(scene.model, scene.data)
            step += 1
            if step % sample_interval == 0:
                latest = sensors.sample()
                recorder.append(latest, controller.state.value)
            if step % 20 == 0:
                sensors.check_safety()
            if video is not None and step % video_interval == 0:
                video.capture()
            if viewer_handle is not None:
                if not viewer_handle.is_running():
                    controller.fail(float(scene.data.time), "interactive viewer closed")
                    break
                viewer_handle.sync()
        if not controller.finished:
            controller.fail(
                float(scene.data.time),
                f"scenario exceeded duration limit {config.simulation.duration_limit_s}s",
            )
        failure_reason = controller.failure_reason
    except (SimulationSafetyError, FloatingPointError, RuntimeError) as exc:
        failure_reason = str(exc)
        controller.fail(float(scene.data.time), failure_reason)
        recorder.failure_bundle(failure_reason)
    finally:
        if video is not None:
            video.close()
        if viewer_handle is not None:
            viewer_handle.close()
    if not recorder.records:
        latest = sensors.sample()
        recorder.append(latest, controller.state.value)
    if failure_reason:
        recorder.failure_bundle(failure_reason)
    result = recorder.finalize(controller, failure_reason)
    LOGGER.info(
        "%s: %s (%.3fs simulated, RTF %.3f)",
        scenario,
        "PASS" if result["success"] else "FAIL",
        result["metrics"].get("simulated_duration_s", 0.0),
        result["metrics"].get("real_time_factor", 0.0),
    )
    return result


def _check_initial_penetration(scene: object) -> None:
    model = scene.model
    data = scene.data
    severe = [
        float(data.contact[index].dist)
        for index in range(data.ncon)
        if data.contact[index].dist < -0.001
    ]
    if severe:
        raise SimulationSafetyError(
            f"Initial interpenetration reaches {min(severe):.6g} m"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, arguments.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        config = load_config(arguments)
        if arguments.command == "inspect-assets":
            report = GeometryPipeline(config).inspect()
            print(json.dumps(report, indent=2))
            return 0
        if arguments.command == "preprocess":
            manifest = GeometryPipeline(config).preprocess(force=arguments.force)
            print(json.dumps(manifest, indent=2))
            return 0
        if arguments.command == "run":
            result = run_simulation(
                config,
                arguments.scenario,
                output_dir=arguments.output,
                viewer=arguments.viewer,
                record_path=arguments.record,
            )
            print(json.dumps(result, indent=2))
            return 0 if result["success"] else 1
        if arguments.command == "sweep":
            from .sweeps import run_robustness_sweep

            summary = run_robustness_sweep(
                config, arguments.output, arguments.trials
            )
            print(json.dumps(summary, indent=2))
            return 0
    except (ConfigError, GeometryError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    parser.error(f"Unhandled command {arguments.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
