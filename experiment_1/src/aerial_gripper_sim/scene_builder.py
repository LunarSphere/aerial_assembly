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
        use_j_hook_overlay = (
            manifest.get("block_source", {}).get("kind") == "urdf"
        )
        payload_hook_geoms = (
            _j_hook_geoms("payload", "2", "29")
            if use_j_hook_overlay
            else _hook_shelf_geoms("payload", "2", "29")
        )
        target_hook_geoms = (
            _j_hook_geoms("target", "16", "5")
            if use_j_hook_overlay
            else _hook_shelf_geoms("target", "16", "5")
        )
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
        string_extension_xml = string_model.extension_xml()
        string_equality_xml = "".join(string_model.equality_fragments())
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
              {payload_hook_geoms}
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
              {target_hook_geoms}
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
          {string_extension_xml}
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
            {string_equality_xml}
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
    """STL-section-derived retaining lips without filling ramp entrances.

    At x=0 the four repeated vertical walls are at y=-5.3635+4.4*i mm.
    Their rounded lips extend 0.7762 mm toward -Y with an underside at
    z=45.4019 mm.  The boxes below remain inside that measured material.
    """
    wall_y = (-0.0053635, -0.0009635, 0.0034365, 0.0078365)
    geoms: list[str] = []
    for index, y in enumerate(wall_y):
        slope_top = np.array([y - 0.0034365, 0.0480])
        slope_bottom = np.array([y - 0.00084, 0.0406])
        tangent = slope_bottom - slope_top
        slope_length = float(np.linalg.norm(tangent))
        angle = float(np.arctan2(tangent[1], tangent[0]))
        normal_into_cavity = np.array([-np.sin(angle), np.cos(angle)])
        # Only the exposed +normal face matters geometrically. Extend the box
        # 0.4 mm into STL-solid material so compliant contact cannot tunnel
        # through a numerically paper-thin proxy during load transfer.
        half_thickness = 0.00020
        slope_center = (
            (slope_top + slope_bottom) / 2.0
            - normal_into_cavity * half_thickness
        )
        geoms.append(
            f'<geom name="{prefix}_collision_hook_slope_{index}" type="box" '
            f'pos="0 {slope_center[0]:.12g} {slope_center[1]:.12g}" '
            f'size="0.0174 {slope_length / 2.0:.12g} {half_thickness:.12g}" '
            f'quat="{np.cos(angle / 2.0):.12g} '
            f'{np.sin(angle / 2.0):.12g} 0 0" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )
        geoms.append(
            f'<geom name="{prefix}_collision_hook_lip_{index}" type="box" '
            f'pos="0 {y - 0.0003881:.8g} 0.0456019" '
            'size="0.0174 0.0003881 0.0002" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )
        geoms.append(
            f'<geom name="{prefix}_collision_hook_wall_{index}" type="box" '
            f'pos="0 {y + 0.0002:.8g} 0.04320095" '
            'size="0.0174 0.0002 0.00220095" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )
    return "".join(geoms)


def _j_hook_geoms(prefix: str, contype: str, conaffinity: str) -> str:
    """Section-derived inner faces for the URDF block's deeper J hooks.

    CoACD preserves the large tooth cavities but rounds away parts of the
    sub-millimetre return arm. These boxes extend into STL-solid material while
    placing their exposed faces on the measured inner stem and return surfaces.
    """
    pitch = 0.0044
    slope_top = np.array([-0.0088, 0.0480])
    slope_bottom = np.array([-0.00616697, 0.04033811])
    arm_top_start = np.array([-0.00481473, 0.0480])
    arm_top_end = np.array([-0.00626723, 0.04576631])
    nose_points = np.array(
        [
            [-0.00626723, 0.04576631],
            [-0.00628082, 0.04574064],
            [-0.00629253, 0.04570378],
            [-0.00629611, 0.04566528],
            [-0.00628918, 0.04561746],
            [-0.00626932, 0.04557341],
            [-0.00623887, 0.04553588],
            [-0.00619926, 0.04550822],
            [-0.00615389, 0.04549158],
            [-0.00610569, 0.04548811],
            [-0.00605821, 0.04549706],
            [-0.00601504, 0.04551876],
            [-0.00599961, 0.04553046],
        ]
    )
    return_start = np.array([-0.0059996, 0.0455305])
    return_end = np.array([-0.0047932, 0.0468934])
    stem_start = np.array([-0.00519603, 0.04041361])
    stem_end = np.array([-0.0047455, 0.0467588])
    valley_points = np.array(
        [
            [-0.00616697, 0.04033811],
            [-0.00612224, 0.04024220],
            [-0.00602023, 0.04012106],
            [-0.00586066, 0.04002871],
            [-0.00567875, 0.03999874],
            [-0.00549856, 0.04003773],
            [-0.00534379, 0.04013791],
            [-0.00524793, 0.04026398],
            [-0.00519603, 0.04041361],
        ]
    )
    geoms: list[str] = []

    def add_surface(
        name: str,
        index: int,
        start: np.ndarray,
        end: np.ndarray,
        *,
        offset: np.ndarray,
        left_normal_sign: float,
        half_thickness: float,
    ) -> None:
        tangent = end - start
        length = float(np.linalg.norm(tangent))
        angle = float(np.arctan2(tangent[1], tangent[0]))
        left_normal = np.array([-np.sin(angle), np.cos(angle)])
        center = (
            (start + end) / 2.0
            + offset
            + left_normal_sign * left_normal * half_thickness
        )
        geoms.append(
            f'<geom name="{prefix}_collision_j_{name}_{index}" type="box" '
            f'pos="0 {center[0]:.12g} {center[1]:.12g}" '
            f'size="0.0174 {length / 2.0:.12g} {half_thickness:.12g}" '
            f'quat="{np.cos(angle / 2.0):.12g} '
            f'{np.sin(angle / 2.0):.12g} 0 0" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )

    for index in range(4):
        offset = np.array([index * pitch, 0.0])
        slope_tangent = slope_bottom - slope_top
        slope_length = float(np.linalg.norm(slope_tangent))
        slope_angle = float(np.arctan2(slope_tangent[1], slope_tangent[0]))
        slope_normal_into_cavity = np.array(
            [-np.sin(slope_angle), np.cos(slope_angle)]
        )
        slope_half_thickness = 0.00020
        slope_center = (
            (slope_top + slope_bottom) / 2.0
            + offset
            - slope_normal_into_cavity * slope_half_thickness
        )
        geoms.append(
            f'<geom name="{prefix}_collision_j_slope_{index}" type="box" '
            f'pos="0 {slope_center[0]:.12g} {slope_center[1]:.12g}" '
            f'size="0.0174 {slope_length / 2.0:.12g} '
            f'{slope_half_thickness:.12g}" '
            f'quat="{np.cos(slope_angle / 2.0):.12g} '
            f'{np.sin(slope_angle / 2.0):.12g} 0 0" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )
        add_surface(
            "arm_top",
            index,
            arm_top_start,
            arm_top_end,
            offset=offset,
            left_normal_sign=1.0,
            half_thickness=0.00014,
        )
        for segment, (start, end) in enumerate(
            zip(nose_points[:-1], nose_points[1:], strict=True)
        ):
            add_surface(
                f"nose_{segment}",
                index,
                start,
                end,
                offset=offset,
                left_normal_sign=1.0,
                half_thickness=0.00010,
            )
        for segment, (start, end) in enumerate(
            zip(valley_points[:-1], valley_points[1:], strict=True)
        ):
            add_surface(
                f"valley_{segment}",
                index,
                start,
                end,
                offset=offset,
                left_normal_sign=-1.0,
                half_thickness=0.00016,
            )

        return_tangent = return_end - return_start
        return_length = float(np.linalg.norm(return_tangent))
        return_angle = float(
            np.arctan2(return_tangent[1], return_tangent[0])
        )
        return_normal_into_solid = np.array(
            [-np.sin(return_angle), np.cos(return_angle)]
        )
        return_half_thickness = 0.00014
        return_center = (
            (return_start + return_end) / 2.0
            + offset
            + return_normal_into_solid * return_half_thickness
        )
        geoms.append(
            f'<geom name="{prefix}_collision_j_return_{index}" type="box" '
            f'pos="0 {return_center[0]:.12g} {return_center[1]:.12g}" '
            f'size="0.0174 {return_length / 2.0:.12g} '
            f'{return_half_thickness:.12g}" '
            f'quat="{np.cos(return_angle / 2.0):.12g} '
            f'{np.sin(return_angle / 2.0):.12g} 0 0" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )

        stem_tangent = stem_end - stem_start
        stem_length = float(np.linalg.norm(stem_tangent))
        stem_angle = float(np.arctan2(stem_tangent[1], stem_tangent[0]))
        stem_normal_into_cavity = np.array(
            [-np.sin(stem_angle), np.cos(stem_angle)]
        )
        stem_half_thickness = 0.00020
        stem_center = (
            (stem_start + stem_end) / 2.0
            + offset
            - stem_normal_into_cavity * stem_half_thickness
        )
        geoms.append(
            f'<geom name="{prefix}_collision_j_stem_{index}" type="box" '
            f'pos="0 {stem_center[0]:.12g} {stem_center[1]:.12g}" '
            f'size="0.0174 {stem_length / 2.0:.12g} '
            f'{stem_half_thickness:.12g}" '
            f'quat="{np.cos(stem_angle / 2.0):.12g} '
            f'{np.sin(stem_angle / 2.0):.12g} 0 0" '
            f'contype="{contype}" conaffinity="{conaffinity}" '
            'friction="0.7 0.01 0.001" '
            'solref="0.001 1" solimp="0.99 0.999 0.0001 0.5 2"/>'
        )
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
