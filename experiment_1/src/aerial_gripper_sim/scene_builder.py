"""MuJoCo scene construction, model metadata, and initial state."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import trimesh

from .config import AppConfig
from .geometry import GeometryPipeline
from .string_model import StringModel, make_string_model
from .washer_model import WasherModel

LOGGER = logging.getLogger(__name__)


@dataclass
class BuiltScene:
    model: mujoco.MjModel
    data: mujoco.MjData
    xml: str
    manifest: dict[str, Any]
    string_model: StringModel
    washer_model: WasherModel | None
    payload_geom_ids: set[int]
    payload_body_id: int
    gripper_body_id: int
    gripper_mocap_id: int
    target_position_m: np.ndarray
    scenario: str


class SceneBuilder:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pipeline = GeometryPipeline(config)

    def build(self, scenario: str) -> BuiltScene:
        manifest = self.pipeline.preprocess()
        single_string = scenario == "single_string_ramp_test"
        include_payload = scenario not in {"asset_inspection"}
        include_target = scenario in {
            "washer_insertion_test",
            "washer_pullout_test",
            "placement_only",
            "release_only",
            "full_cycle",
        }
        anchors = manifest["anchors"]
        string_model = make_string_model(self.config.strings, anchors)
        target_position = np.array([0.0, 0.12, 0.0])
        if scenario in {"washer_insertion_test", "washer_pullout_test", "placement_only"}:
            target_position = np.zeros(3)
        washer_model = (
            WasherModel(
                self.config,
                manifest["target_assembly"]["washer_transforms_target_m"],
            )
            if include_target
            else None
        )
        xml = self._build_xml(
            scenario,
            manifest,
            string_model,
            washer_model,
            target_position,
            include_payload=include_payload,
            include_target=include_target,
            single_string=single_string,
        )
        try:
            model = mujoco.MjModel.from_xml_string(xml)
        except ValueError as exc:
            failure_path = self.config.paths.outputs / "model_compile_failure.xml"
            failure_path.parent.mkdir(parents=True, exist_ok=True)
            failure_path.write_text(xml)
            raise RuntimeError(
                f"MuJoCo model compilation failed; XML saved to {failure_path}: {exc}"
            ) from exc
        data = mujoco.MjData(model)
        string_model.initialize_pretension(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        payload_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "payload"
        )
        gripper_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper"
        )
        mocap_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "gripper_target"
        )
        mocap_id = int(model.body_mocapid[mocap_body_id])
        payload_geom_ids = {
            index
            for index in range(model.ngeom)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index) or "").startswith(
                "payload_collision_"
            )
        }
        return BuiltScene(
            model=model,
            data=data,
            xml=xml,
            manifest=manifest,
            string_model=string_model,
            washer_model=washer_model,
            payload_geom_ids=payload_geom_ids,
            payload_body_id=payload_body_id,
            gripper_body_id=gripper_body_id,
            gripper_mocap_id=mocap_id,
            target_position_m=target_position,
            scenario=scenario,
        )

    def _build_xml(
        self,
        scenario: str,
        manifest: dict[str, Any],
        string_model: StringModel,
        washer_model: WasherModel | None,
        target_position: np.ndarray,
        *,
        include_payload: bool,
        include_target: bool,
        single_string: bool,
    ) -> str:
        collision_assets, collision_geoms = _collision_xml(
            manifest["collision_proxy"]
        )
        processed = manifest["processed_meshes"]
        asset_xml = f"""
          <mesh name="gripper_visual" file="{Path(processed['GR_0.stl']).resolve()}"/>
          <mesh name="block_visual" file="{Path(processed['BB_0.stl']).resolve()}"/>
          <mesh name="washer_visual" file="{Path(processed['SLW_0.stl']).resolve()}"/>
          {collision_assets}
        """
        gripper_position = _gripper_initial_position(scenario)
        gripper_position[2] += self.config.perturbations.gripper_height_error_m
        gripper_geoms = (
            '<geom name="gripper_collision" type="mesh" mesh="gripper_visual" '
            'contype="1" conaffinity="2" rgba="0.25 0.35 0.8 0.25"/>'
            if self.config.collision.gripper_enabled
            else '<geom name="gripper_visual_geom" type="mesh" mesh="gripper_visual" '
            'contype="0" conaffinity="0" rgba="0.25 0.35 0.8 1"/>'
        )
        string_count: int | None = 1 if single_string else None
        if scenario in {
            "washer_insertion_test",
            "washer_pullout_test",
            "placement_only",
        }:
            string_count = 0
        string_xml = "".join(string_model.xml_fragments(count=string_count))
        payload_xml = ""
        payload_driver_xml = ""
        payload_equality_xml = ""
        if include_payload:
            payload_position = _payload_initial_position(scenario, target_position)
            payload_position[:2] += np.asarray(
                self.config.perturbations.payload_xy_offset_m
            )
            yaw = np.deg2rad(self.config.perturbations.payload_yaw_deg)
            payload_quat = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
            mass, center, inertia = _payload_inertia(
                Path(processed["BB_0.stl"]), self.config.payload.mass_kg
            )
            payload_xml = f"""
            <body name="payload" pos="{_vec(payload_position)}"
              quat="{_vec(payload_quat)}">
              <freejoint name="payload_free"/>
              <inertial pos="{_vec(center)}" mass="{mass:.12g}"
                fullinertia="{_inertia(inertia)}"/>
              <geom name="payload_visual" type="mesh" mesh="block_visual"
                contype="0" conaffinity="0" rgba="0.8 0.55 0.12 1"/>
              {_prefix_geoms(collision_geoms, "payload", "2", "29")}
              {_hook_shelf_geoms("payload", "2", "29")}
            </body>
            """
            if scenario in {
                "washer_insertion_test",
                "washer_pullout_test",
                "placement_only",
            }:
                payload_driver_xml = (
                    f'<body name="payload_target" mocap="true" '
                    f'pos="{_vec(payload_position)}"/>'
                )
                payload_equality_xml = (
                    '<weld name="payload_test_rig_weld" body1="payload" '
                    'body2="payload_target" solref="0.002 1" '
                    'solimp="0.95 0.99 0.001 0.5 2"/>'
                )
        target_xml = ""
        washer_xml = ""
        if include_target:
            target_xml = f"""
            <body name="target_block" pos="{_vec(target_position)}">
              <geom name="target_visual" type="mesh" mesh="block_visual"
                contype="0" conaffinity="0" rgba="0.55 0.55 0.6 1"/>
              {_prefix_geoms(collision_geoms, "target", "16", "5")}
              {_hook_shelf_geoms("target", "16", "5")}
            </body>
            """
            if washer_model is not None:
                washer_xml = "".join(washer_model.xml_fragments(target_position))
        sites = "".join(
            f'<site name="anchor_left_{item["index"]}" pos="{_vec(item["left_m"])}" size="0.0006"/>'
            f'<site name="anchor_right_{item["index"]}" pos="{_vec(item["right_m"])}" size="0.0006"/>'
            for item in manifest["anchors"]
        )
        option = self.config.simulation
        washer_default_xml = ""
        if (
            washer_model is not None
            and self.config.washer.mode == "deformable_flex"
        ):
            washer_default_xml = (
                '<default class="washer_flex">'
                f'<joint stiffness="{washer_model.reduced_joint_stiffness_n_per_m:.12g}" '
                f'damping="{self.config.washer.damping:.12g}"/>'
                "</default>"
            )
        return f"""
        <mujoco model="actuatorless_aerial_gripper_{scenario}">
          <compiler angle="radian" autolimits="true" balanceinertia="true"/>
          <option timestep="{option.timestep_s:.12g}" gravity="0 0 -9.81"
            integrator="{option.integrator}" solver="{option.solver}"
            iterations="{option.iterations}" ls_iterations="{option.ls_iterations}"/>
          <size nconmax="5000" njmax="10000"/>
          <visual>
            <global offwidth="{self.config.output.render_width}"
              offheight="{self.config.output.render_height}"/>
            <quality shadowsize="2048"/>
          </visual>
          <default>
            <geom margin="{option.contact_margin_m:.12g}" gap="{option.contact_gap_m:.12g}"
              solref="{' '.join(map(str, option.solref))}"
              solimp="{' '.join(map(str, option.solimp))}"/>
            {washer_default_xml}
          </default>
          <asset>
            <texture name="ground_tex" type="2d" builtin="checker"
              rgb1=".18 .18 .2" rgb2=".28 .28 .3" width="256" height="256"/>
            <material name="ground_mat" texture="ground_tex" texrepeat="3 3"/>
            {asset_xml}
          </asset>
          <worldbody>
            <light pos="0 -0.3 0.4" dir="0 0.5 -1" diffuse="0.8 0.8 0.8"/>
            <camera name="overview" pos="0.22 -0.30 0.20" xyaxes="0.80 0.60 0 -0.25 0.34 0.91"/>
            <geom name="ground" type="plane" size="0.5 0.5 0.01"
              material="ground_mat" contype="1" conaffinity="2"
              friction="{' '.join(map(str, self.config.payload.ground_friction))}"/>
            <body name="gripper_target" mocap="true" pos="{_vec(gripper_position)}"/>
            {payload_driver_xml}
            <body name="gripper" pos="{_vec(gripper_position)}">
              <freejoint name="gripper_free"/>
              <inertial pos="0 0 -0.02" mass="0.05" diaginertia="0.00008 0.00008 0.00008"/>
              {gripper_geoms}
              <site name="gripper_reference" pos="0 0 0" size="0.001"/>
              {sites}
              {string_xml}
            </body>
            {payload_xml}
            {target_xml}
            {washer_xml}
          </worldbody>
          <equality>
            <weld name="gripper_command_weld" body1="gripper" body2="gripper_target"
              solref="0.002 1" solimp="0.95 0.99 0.001 0.5 2"/>
            {payload_equality_xml}
          </equality>
        </mujoco>
        """


def _collision_xml(proxy: dict[str, Any]) -> tuple[str, str]:
    assets: list[str] = []
    geoms: list[str] = []
    if proxy["backend"] == "coacd":
        for index, path in enumerate(proxy["parts"]):
            assets.append(
                f'<mesh name="block_hull_{index}" file="{Path(path).resolve()}"/>'
            )
            geoms.append(f'<geom type="mesh" mesh="block_hull_{index}"/>')
    else:
        boxes = json.loads(Path(proxy["procedural_boxes"]).read_text())
        for item in boxes:
            geoms.append(
                f'<geom type="box" pos="{_vec(item["center_m"])}" '
                f'size="{_vec(item["halfsize_m"])}"/>'
            )
    return "".join(assets), "".join(geoms)


def _prefix_geoms(
    raw_geoms: str, prefix: str, contype: str, conaffinity: str
) -> str:
    chunks = raw_geoms.split("<geom ")[1:]
    output: list[str] = []
    for index, chunk in enumerate(chunks):
        output.append(
            f'<geom name="{prefix}_collision_{index}" contype="{contype}" '
            f'conaffinity="{conaffinity}" friction="0.6 0.01 0.001" {chunk}'
        )
    return "".join(output)


def _hook_shelf_geoms(prefix: str, contype: str, conaffinity: str) -> str:
    """Critical-feature analytic overlay derived from block side sections.

    CoACD approximates the bulk mesh well but divides the 1–2 mm hook lips at
    hull seams. These thin boxes reproduce only the measured horizontal
    undercut shelves; all ramp entry and release motion still uses contact.
    """
    centers_y = (-0.0055, -0.0010, 0.0035, 0.0080)
    geoms: list[str] = []
    index = 0
    for y in centers_y:
        geoms.append(
            f'<geom name="{prefix}_collision_hook_{index}" type="box" '
            f'pos="0 {y:.8g} 0.0448" '
            'size="0.0174 0.0015 0.00045" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001"/>'
        )
        index += 1
        geoms.append(
            f'<geom name="{prefix}_collision_hook_wall_{index}" type="box" '
            f'pos="0 {y + 0.00185:.8g} 0.042" '
            'size="0.0174 0.00035 0.0028" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001"/>'
        )
        index += 1
    return "".join(geoms)


def _payload_inertia(
    path: Path, configured_mass: float
) -> tuple[float, np.ndarray, np.ndarray]:
    mesh = trimesh.load_mesh(path, process=True)
    mass = configured_mass
    scale = mass / float(mesh.mass)
    inertia = np.asarray(mesh.moment_inertia) * scale
    eigenvalues = np.linalg.eigvalsh(inertia)
    if np.any(eigenvalues <= 0):
        raise ValueError(f"Computed payload inertia is not positive definite: {eigenvalues}")
    return mass, np.asarray(mesh.center_mass), inertia


def _gripper_initial_position(scenario: str) -> np.ndarray:
    if scenario in {"washer_insertion_test", "washer_pullout_test", "placement_only"}:
        return np.array([0.0, 0.0, 0.105])
    if scenario == "release_only":
        return np.array([0.0, 0.12, 0.095])
    return np.array([0.0, 0.0, 0.130])


def _payload_initial_position(scenario: str, target: np.ndarray) -> np.ndarray:
    if scenario in {"washer_insertion_test", "washer_pullout_test", "placement_only"}:
        return target + np.array([0.0, 0.0, 0.052])
    if scenario == "release_only":
        # Closest non-interpenetrating pose with pegs already at the washer
        # entry plane; gravity settles the unrigged payload into the fingers.
        return target + np.array([0.0, 0.0, 0.041])
    return np.zeros(3)


def _vec(values: Any) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _inertia(matrix: np.ndarray) -> str:
    values = (
        matrix[0, 0],
        matrix[1, 1],
        matrix[2, 2],
        matrix[0, 1],
        matrix[0, 2],
        matrix[1, 2],
    )
    return _vec(values)
