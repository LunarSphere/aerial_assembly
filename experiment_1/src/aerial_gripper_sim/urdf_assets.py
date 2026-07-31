"""Minimal URDF assembly reader for fixed, mesh-only CAD exports."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


class UrdfAssetError(RuntimeError):
    """Raised when a URDF cannot provide an unambiguous fixed assembly."""


@dataclass(frozen=True)
class UrdfMeshInstance:
    link_name: str
    mesh_path: Path
    mesh_scale: tuple[float, float, float]
    root_from_mesh: np.ndarray
    volume_m3: float
    extents_m: tuple[float, float, float]


@dataclass(frozen=True)
class UrdfAssembly:
    path: Path
    robot_name: str
    block: UrdfMeshInstance
    washers: tuple[UrdfMeshInstance, ...]
    block_mesh_from_washer_mesh: tuple[np.ndarray, ...]
    referenced_meshes: tuple[Path, ...]

    def metadata(self) -> dict[str, Any]:
        return {
            "urdf_path": str(self.path),
            "robot_name": self.robot_name,
            "block_link": self.block.link_name,
            "block_mesh": str(self.block.mesh_path),
            "block_mesh_scale": list(self.block.mesh_scale),
            "washer_links": [item.link_name for item in self.washers],
            "washer_meshes": [str(item.mesh_path) for item in self.washers],
            "washer_mesh_scales": [list(item.mesh_scale) for item in self.washers],
        }


def load_fixed_mesh_assembly(path: str | Path) -> UrdfAssembly:
    """Load a fixed URDF and identify its one block plus four washer meshes."""
    urdf_path = Path(path)
    if not urdf_path.is_file():
        raise UrdfAssetError(f"URDF does not exist: {urdf_path}")
    try:
        root = ET.parse(urdf_path).getroot()
    except ET.ParseError as exc:
        raise UrdfAssetError(f"Cannot parse {urdf_path}: {exc}") from exc
    if root.tag != "robot":
        raise UrdfAssetError(f"{urdf_path} root element must be <robot>")

    link_visuals: dict[str, tuple[Path, tuple[float, float, float], np.ndarray]] = {}
    link_names: set[str] = set()
    for link in root.findall("link"):
        link_name = link.get("name")
        if not link_name:
            raise UrdfAssetError("Every URDF link must have a name")
        link_names.add(link_name)
        visuals = link.findall("visual")
        if not visuals:
            continue
        if len(visuals) != 1:
            raise UrdfAssetError(
                f"Link {link_name!r} must contain exactly one visual mesh"
            )
        visual = visuals[0]
        mesh = visual.find("geometry/mesh")
        if mesh is None or not mesh.get("filename"):
            raise UrdfAssetError(f"Link {link_name!r} visual must reference a mesh")
        mesh_path = _resolve_mesh_uri(urdf_path, mesh.get("filename", ""))
        if not mesh_path.is_file():
            raise UrdfAssetError(
                f"Link {link_name!r} references missing mesh {mesh_path}"
            )
        scale = _vector(mesh.get("scale"), (1.0, 1.0, 1.0), "mesh scale")
        if any(value <= 0 for value in scale):
            raise UrdfAssetError(f"Link {link_name!r} mesh scale must be positive")
        link_visuals[link_name] = (mesh_path, scale, _origin(visual.find("origin")))

    children: set[str] = set()
    joints: list[tuple[str, str, np.ndarray]] = []
    for joint in root.findall("joint"):
        joint_name = joint.get("name", "<unnamed>")
        if joint.get("type") != "fixed":
            raise UrdfAssetError(
                f"Joint {joint_name!r} is not fixed; this importer handles fixed CAD assemblies"
            )
        parent = joint.find("parent")
        child = joint.find("child")
        parent_name = parent.get("link") if parent is not None else None
        child_name = child.get("link") if child is not None else None
        if parent_name not in link_names or child_name not in link_names:
            raise UrdfAssetError(f"Joint {joint_name!r} references an unknown link")
        if child_name in children:
            raise UrdfAssetError(f"Link {child_name!r} has more than one parent")
        children.add(child_name)
        joints.append((parent_name, child_name, _origin(joint.find("origin"))))

    roots = link_names - children
    if len(roots) != 1:
        raise UrdfAssetError(f"Expected one URDF root link, found {sorted(roots)}")
    root_link = next(iter(roots))
    root_from_link: dict[str, np.ndarray] = {root_link: np.eye(4)}
    pending = list(joints)
    while pending:
        next_pending: list[tuple[str, str, np.ndarray]] = []
        progressed = False
        for parent, child, parent_from_child in pending:
            if parent not in root_from_link:
                next_pending.append((parent, child, parent_from_child))
                continue
            root_from_link[child] = root_from_link[parent] @ parent_from_child
            progressed = True
        if not progressed:
            raise UrdfAssetError("URDF joint graph is disconnected or cyclic")
        pending = next_pending

    instances: list[UrdfMeshInstance] = []
    for link_name, (mesh_path, scale, link_from_mesh) in link_visuals.items():
        loaded = trimesh.load(mesh_path, force="mesh", process=True)
        if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
            raise UrdfAssetError(f"{mesh_path} is not a usable triangle mesh")
        scaled = loaded.copy()
        scaled.apply_scale(np.asarray(scale))
        instances.append(
            UrdfMeshInstance(
                link_name=link_name,
                mesh_path=mesh_path,
                mesh_scale=scale,
                root_from_mesh=root_from_link[link_name] @ link_from_mesh,
                volume_m3=float(abs(scaled.volume)),
                extents_m=tuple(float(value) for value in scaled.extents),
            )
        )
    if len(instances) != 5:
        raise UrdfAssetError(
            f"Expected five visual mesh instances (one block, four washers), found {len(instances)}"
        )
    ordered = sorted(instances, key=lambda item: item.volume_m3, reverse=True)
    block = ordered[0]
    washers = tuple(ordered[1:])
    if block.volume_m3 <= 100.0 * max(item.volume_m3 for item in washers):
        raise UrdfAssetError("Largest URDF mesh is not unambiguously the block")
    reference_volume = float(np.mean([item.volume_m3 for item in washers]))
    if any(
        abs(item.volume_m3 - reference_volume) / reference_volume > 0.01
        for item in washers
    ):
        raise UrdfAssetError("The four URDF washer instances are not equal-sized")

    mesh_from_root = np.linalg.inv(block.root_from_mesh)
    block_from_washers = tuple(
        mesh_from_root @ washer.root_from_mesh for washer in washers
    )
    return UrdfAssembly(
        path=urdf_path,
        robot_name=root.get("name", ""),
        block=block,
        washers=washers,
        block_mesh_from_washer_mesh=block_from_washers,
        referenced_meshes=tuple(sorted({item.mesh_path for item in instances})),
    )


def load_scaled_mesh(instance: UrdfMeshInstance) -> trimesh.Trimesh:
    loaded = trimesh.load(instance.mesh_path, force="mesh", process=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise UrdfAssetError(f"{instance.mesh_path} is not one triangle mesh")
    result = loaded.copy()
    result.apply_scale(np.asarray(instance.mesh_scale))
    return result


def _resolve_mesh_uri(urdf_path: Path, uri: str) -> Path:
    if uri.startswith("package://"):
        package_and_path = uri.removeprefix("package://")
        parts = package_and_path.split("/", 1)
        if len(parts) != 2:
            raise UrdfAssetError(f"Invalid package mesh URI: {uri}")
        package_root = urdf_path.parent.parent
        return (package_root / parts[1]).resolve()
    candidate = Path(uri)
    if candidate.is_absolute():
        return candidate
    return (urdf_path.parent / candidate).resolve()


def _vector(
    text: str | None,
    default: tuple[float, float, float],
    label: str,
) -> tuple[float, float, float]:
    if text is None:
        return default
    try:
        values = tuple(float(value) for value in text.split())
    except ValueError as exc:
        raise UrdfAssetError(f"Invalid {label}: {text!r}") from exc
    if len(values) != 3:
        raise UrdfAssetError(f"{label} must contain three values")
    return values


def _origin(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4)
    xyz = _vector(element.get("xyz"), (0.0, 0.0, 0.0), "origin xyz")
    roll, pitch, yaw = _vector(
        element.get("rpy"), (0.0, 0.0, 0.0), "origin rpy"
    )
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = xyz
    return result
