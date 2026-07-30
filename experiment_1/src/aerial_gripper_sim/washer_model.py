"""Compliant-finger and tetrahedral-flex self-locking washer models."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import meshio
import mujoco
import numpy as np

from .config import AppConfig

LOGGER = logging.getLogger(__name__)


@dataclass
class WasherTelemetry:
    name: str
    max_finger_angle_rad: float
    mean_finger_angle_rad: float
    strain_proxy: float
    reaction_force_n: float
    insertion_depth_m: float


class WasherModel:
    def __init__(self, config: AppConfig, transforms: list[list[list[float]]]):
        self.config = config
        self.transforms = [np.asarray(item, dtype=float) for item in transforms]
        self.joint_names: list[list[str]] = []
        self.flex_names: list[str] = []
        self._initial_flex_vertices: dict[str, np.ndarray] = {}

    @property
    def reduced_joint_stiffness_n_per_m(self) -> float:
        """Energy-equivalent stiffness for each trilinear control coordinate."""
        volume = 17.845177e-9
        diameter = 0.009
        return self.config.washer.young_modulus_pa * volume / (8.0 * diameter**2)

    def xml_fragments(self, target_position: np.ndarray) -> list[str]:
        if self.config.washer.mode == "deformable_flex":
            return self._deformable_fragments(target_position)
        return self._compliant_fragments(target_position)

    def _compliant_fragments(self, target_position: np.ndarray) -> list[str]:
        fragments: list[str] = []
        self.joint_names = []
        outer_radius = 0.00415
        inner_radius = 0.00120
        finger_length = outer_radius - inner_radius
        finger_width = 0.00072
        finger_thickness = 0.00030
        ring_segments = 16
        for washer_index, transform in enumerate(self.transforms):
            center = target_position + transform[:3, 3]
            rotation = transform[:3, :3]
            # Convert the recovered rotation to a normalized MuJoCo quaternion.
            quat = _matrix_to_quat(rotation)
            ring_geoms: list[str] = []
            for segment in range(ring_segments):
                a0 = 2.0 * math.pi * segment / ring_segments
                a1 = 2.0 * math.pi * (segment + 1) / ring_segments
                p0 = (outer_radius * math.cos(a0), outer_radius * math.sin(a0), 0.0)
                p1 = (outer_radius * math.cos(a1), outer_radius * math.sin(a1), 0.0)
                ring_geoms.append(
                    f'<geom name="washer_{washer_index}_ring_{segment}" type="capsule" '
                    f'fromto="{_vec(p0)} {_vec(p1)}" size="0.00042" '
                    'contype="8" conaffinity="2" rgba="0.2 0.75 0.3 1"/>'
                )
            finger_bodies: list[str] = []
            washer_joints: list[str] = []
            for finger in range(self.config.washer.finger_count):
                angle = 2.0 * math.pi * finger / self.config.washer.finger_count
                hinge = (
                    outer_radius * math.cos(angle),
                    outer_radius * math.sin(angle),
                    0.0,
                )
                joint_name = f"washer_{washer_index}_finger_{finger}_hinge"
                washer_joints.append(joint_name)
                finger_bodies.append(
                    f"""
                    <body name="washer_{washer_index}_finger_{finger}"
                      pos="{_vec(hinge)}" euler="0 0 {angle:.12g}">
                      <joint name="{joint_name}" type="hinge" axis="0 1 0"
                        limited="true" range="-0.9 0.9"
                        stiffness="{self.config.washer.effective_stiffness_nm_rad:.12g}"
                        damping="{self.config.washer.finger_joint_damping:.12g}"/>
                      <geom name="washer_{washer_index}_finger_{finger}_geom"
                        type="box" pos="{-finger_length / 2:.12g} 0 0"
                        size="{finger_length / 2:.12g} {finger_width / 2:.12g} {finger_thickness / 2:.12g}"
                        density="{self.config.washer.density_kg_per_m3:.12g}"
                        contype="8" conaffinity="2"
                        friction="{' '.join(map(str, self.config.washer.peg_friction))}"
                        rgba="0.15 0.9 0.3 1"/>
                    </body>
                    """
                )
            self.joint_names.append(washer_joints)
            fragments.append(
                f"""
                <body name="washer_{washer_index}" pos="{_vec(center)}"
                  quat="{_vec(quat)}">
                  {''.join(ring_geoms)}
                  {''.join(finger_bodies)}
                </body>
                """
            )
        return fragments

    def _deformable_fragments(self, target_position: np.ndarray) -> list[str]:
        mesh_path, _pin_ids = tetrahedralize_washer(self.config)
        fragments: list[str] = []
        self.flex_names = []
        volume = 17.845177e-9
        physical_mass = volume * self.config.washer.density_kg_per_m3
        mass = physical_mass * self.config.washer.numerical_mass_scale
        control_stiffness = self.reduced_joint_stiffness_n_per_m
        for index, transform in enumerate(self.transforms):
            name = f"washer_flex_{index}"
            self.flex_names.append(name)
            center = target_position + transform[:3, 3]
            quat = _matrix_to_quat(transform[:3, :3])
            fragments.append(
                f"""
                <body name="washer_flex_parent_{index}" childclass="washer_flex">
                  <flexcomp name="{name}" type="gmsh" dof="trilinear"
                    file="{mesh_path.resolve()}"
                    pos="{_vec(center)}" quat="{_vec(quat)}"
                    radius="0.00005" mass="{mass:.12g}" rgba="0.1 0.85 0.3 1">
                    <elasticity young="{self.config.washer.young_modulus_pa:.12g}"
                      poisson="{self.config.washer.poisson_ratio:.12g}"
                      damping="{self.config.washer.damping:.12g}"/>
                    <contact selfcollide="none" contype="8" conaffinity="2" condim="3"
                      friction="{' '.join(map(str, self.config.washer.peg_friction))}"/>
                  </flexcomp>
                </body>
                """
            )
        LOGGER.info(
            "Deformable washers use trilinear DOFs with %.6g N/m control-point "
            "springs; the full tetrahedral mesh remains active for collision",
            control_stiffness,
        )
        return fragments

    def sample(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        peg_bottom_z: float | None = None,
    ) -> list[WasherTelemetry]:
        telemetry: list[WasherTelemetry] = []
        if self.config.washer.mode == "deformable_flex":
            for name, transform in zip(self.flex_names, self.transforms, strict=True):
                flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
                start = int(model.flex_vertadr[flex_id])
                number = int(model.flex_vertnum[flex_id])
                vertices = np.asarray(data.flexvert_xpos[start : start + number])
                if name not in self._initial_flex_vertices:
                    self._initial_flex_vertices[name] = vertices.copy()
                rest = self._initial_flex_vertices[name]
                displacement = np.linalg.norm(vertices - rest, axis=1)
                telemetry.append(
                    WasherTelemetry(
                        name=name,
                        max_finger_angle_rad=float("nan"),
                        mean_finger_angle_rad=float("nan"),
                        strain_proxy=float(np.max(displacement) / 0.0045),
                        reaction_force_n=0.0,
                        insertion_depth_m=_depth(peg_bottom_z, transform[2, 3]),
                    )
                )
            return telemetry
        for index, (names, transform) in enumerate(
            zip(self.joint_names, self.transforms, strict=True)
        ):
            angles: list[float] = []
            reactions: list[float] = []
            for name in names:
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                qpos_address = int(model.jnt_qposadr[joint_id])
                dof_address = int(model.jnt_dofadr[joint_id])
                angles.append(float(data.qpos[qpos_address]))
                reactions.append(float(abs(data.qfrc_constraint[dof_address])))
            absolute = np.abs(angles)
            telemetry.append(
                WasherTelemetry(
                    name=f"washer_{index}",
                    max_finger_angle_rad=float(np.max(absolute)),
                    mean_finger_angle_rad=float(np.mean(absolute)),
                    strain_proxy=float(np.max(absolute) / 0.9),
                    reaction_force_n=float(sum(reactions)),
                    insertion_depth_m=_depth(peg_bottom_z, transform[2, 3]),
                )
            )
        return telemetry


def tetrahedralize_washer(config: AppConfig) -> tuple[Path, list[int]]:
    output = config.paths.processed_assets / "washer_tetra.msh"
    gmsh_output = config.paths.processed_assets / "washer_tetra_gmsh.msh"
    metadata = config.paths.processed_assets / "washer_tetra_pins.json"
    if output.is_file() and metadata.is_file():
        import json

        cached = json.loads(metadata.read_text())
        if cached.get("schema_version") == 3:
            return output, cached["pin_ids"]
    import gmsh
    import json

    source = config.paths.processed_assets / "SLW_0.stl"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("washer")
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.merge(str(source.resolve()))
        angle = 40.0 * math.pi / 180.0
        # The normalized STL is already watertight. Asking Gmsh to force every
        # patch to be parametrizable is both unnecessary and pathological for
        # the washer's thin radial fingers, so retain it as a discrete surface.
        gmsh.model.mesh.classifySurfaces(angle, True, False, math.pi)
        surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
        if not surfaces:
            raise RuntimeError("Gmsh found no washer surfaces")
        # A discrete volume can be bounded directly by the classified STL
        # patches and does not require fragile CAD-style surface
        # parametrization.
        gmsh.model.addDiscreteEntity(3, -1, surfaces)
        gmsh.option.setNumber("Mesh.MeshSizeMin", config.washer.mesh_size_m)
        gmsh.option.setNumber("Mesh.MeshSizeMax", config.washer.mesh_size_m)
        # MuJoCo's flexcomp Gmsh reader expects the 4.1 node-block layout.
        gmsh.option.setNumber("Mesh.MshFileVersion", 4.1)
        gmsh.model.mesh.generate(3)
        output.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(gmsh_output.resolve()))
    finally:
        gmsh.finalize()
    volume_mesh = meshio.read(gmsh_output)
    tetrahedra = volume_mesh.cells_dict.get("tetra")
    if tetrahedra is None or not len(tetrahedra):
        LOGGER.warning(
            "Gmsh could not fill the discrete STL volume; using the "
            "deterministic TetGen tetrahedral fallback"
        )
        points, tetrahedra = _fallback_tetrahedra(source, config.washer.mesh_size_m)
    else:
        points = volume_mesh.points[:, :3]
    _write_mujoco_gmsh(output, points, tetrahedra)
    radii = np.linalg.norm(points[:, :2], axis=1)
    pin_ids = np.flatnonzero(radii >= 0.00355).astype(int).tolist()
    if not pin_ids:
        raise RuntimeError("No outer-annulus tetrahedral vertices were selected for pinning")
    metadata.write_text(
        json.dumps({"schema_version": 3, "pin_ids": pin_ids}, indent=2)
    )
    return output, pin_ids


def _write_mujoco_gmsh(
    path: Path, points: np.ndarray, tetrahedra: np.ndarray
) -> None:
    """Write the strict Gmsh 4.1 subset consumed by MuJoCo flexcomp.

    Gmsh groups nodes by their source surface/volume entities. MuJoCo requires
    one node block, so the generated volume is normalized to a single
    sequential block without changing points or tetrahedral connectivity.
    """
    node_count = len(points)
    element_count = len(tetrahedra)
    lines = [
        "$MeshFormat",
        "4.1 0 8",
        "$EndMeshFormat",
        "$Nodes",
        f"1 {node_count} 1 {node_count}",
        f"3 1 0 {node_count}",
        *(str(index) for index in range(1, node_count + 1)),
        *(" ".join(f"{float(value):.17g}" for value in point) for point in points),
        "$EndNodes",
        "$Elements",
        f"1 {element_count} 1 {element_count}",
        f"3 1 4 {element_count}",
        *(
            f"{index} " + " ".join(str(int(node) + 1) for node in element)
            for index, element in enumerate(tetrahedra, start=1)
        ),
        "$EndElements",
        "",
    ]
    path.write_text("\n".join(lines))


def _fallback_tetrahedra(
    source: Path, mesh_size_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fill a watertight surface with quality-controlled tetrahedra.

    This is the documented fallback for STL topologies that Gmsh can retain as
    discrete surfaces but cannot reparametrize into a CAD volume.
    """
    import tetgen
    import trimesh

    surface = trimesh.load_mesh(source, process=True)
    if not surface.is_watertight:
        raise RuntimeError("Washer surface must be watertight for tetrahedralization")
    _ = mesh_size_m
    points, tetrahedra, _, _ = tetgen.TetGen(
        surface.vertices, surface.faces
    ).tetrahedralize(
        order=1,
        mindihedral=2.0,
        minratio=2.0,
        steinerleft=500,
        quiet=True,
    )
    candidate_points = np.asarray(points)
    tetrahedra = np.asarray(tetrahedra, dtype=int)
    p0 = candidate_points[tetrahedra[:, 0]]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        np.cross(
            candidate_points[tetrahedra[:, 1]] - p0,
            candidate_points[tetrahedra[:, 2]] - p0,
        ),
        candidate_points[tetrahedra[:, 3]] - p0,
    )
    # Near-coplanar STL facets can induce numerically singular slivers. They
    # account for less than 1e-5 of volume but destabilize elasticity.
    tetrahedra = tetrahedra[np.abs(signed_six_volume) / 6.0 > 1.0e-16]
    signed_six_volume = np.einsum(
        "ij,ij->i",
        np.cross(
            candidate_points[tetrahedra[:, 1]]
            - candidate_points[tetrahedra[:, 0]],
            candidate_points[tetrahedra[:, 2]]
            - candidate_points[tetrahedra[:, 0]],
        ),
        candidate_points[tetrahedra[:, 3]]
        - candidate_points[tetrahedra[:, 0]],
    )
    negative = signed_six_volume < 0
    tetrahedra[negative, :2] = tetrahedra[negative, 1::-1]
    used = np.unique(tetrahedra)
    remap = np.full(len(candidate_points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    points = candidate_points[used]
    tetrahedra = remap[tetrahedra]
    if not len(tetrahedra):
        raise RuntimeError("TetGen washer fallback produced no tetrahedra")
    return points, tetrahedra


def _matrix_to_quat(matrix: np.ndarray) -> tuple[float, float, float, float]:
    from scipy.spatial.transform import Rotation

    # ICP on an almost axisymmetric star can return a reflected frame. Project
    # to the closest proper rotation; a reflection is not a legal rigid pose.
    u, _, vt = np.linalg.svd(matrix)
    proper = u @ vt
    if np.linalg.det(proper) < 0:
        u[:, -1] *= -1
        proper = u @ vt
    xyzw = Rotation.from_matrix(proper).as_quat()
    return (float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2]))


def _vec(values: Any) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _depth(peg_bottom_z: float | None, washer_z: float) -> float:
    if peg_bottom_z is None:
        return 0.0
    return max(0.0, washer_z - peg_bottom_z)
