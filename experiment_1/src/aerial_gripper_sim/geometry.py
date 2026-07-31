"""STL inspection, coordinate normalization, anchor detection, and target recovery."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from .config import AppConfig
from .urdf_assets import (
    UrdfAssembly,
    UrdfAssetError,
    load_fixed_mesh_assembly,
    load_scaled_mesh,
)

LOGGER = logging.getLogger(__name__)

ASSET_NAMES = ("GR_0.stl", "BB_0.stl", "SLW_0.stl", "Ghast_0.stl")


class GeometryError(RuntimeError):
    """Raised when supplied geometry cannot support a trustworthy simulation."""


@dataclass(frozen=True)
class AnchorPair:
    index: int
    left_m: tuple[float, float, float]
    right_m: tuple[float, float, float]
    diameter_m: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "left_m": list(self.left_m),
            "right_m": list(self.right_m),
            "diameter_m": self.diameter_m,
            "source": self.source,
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_mesh(path: Path, *, process: bool = True) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=process)
    if not isinstance(loaded, trimesh.Trimesh):
        raise GeometryError(f"{path} did not load as one triangle mesh")
    if len(loaded.faces) == 0:
        raise GeometryError(f"{path} contains no triangles")
    return loaded


def inspect_asset(path: Path, scale: float) -> dict[str, Any]:
    raw = load_mesh(path, process=False)
    merged = load_mesh(path, process=True)
    components = list(merged.split(only_watertight=False))
    face_areas = raw.area_faces
    sorted_faces = np.sort(raw.faces, axis=1)
    duplicate_faces = len(sorted_faces) - len(np.unique(sorted_faces, axis=0))
    unique_edges, edge_counts = np.unique(
        np.sort(merged.edges, axis=1), axis=0, return_counts=True
    )
    _ = unique_edges
    nonmanifold_edges = int(np.count_nonzero(edge_counts != 2))
    inertia = np.asarray(merged.moment_inertia, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(inertia)
    order = np.argsort(eigenvalues)
    principal_axes = eigenvectors[:, order]
    extents = merged.extents * scale
    inferred_scale = _infer_scale(merged.extents)
    warnings: list[str] = []
    if np.count_nonzero(face_areas <= 1.0e-12):
        warnings.append("degenerate triangles detected")
    if duplicate_faces:
        warnings.append(f"{duplicate_faces} duplicate indexed faces detected")
    if nonmanifold_edges:
        warnings.append(f"{nonmanifold_edges} boundary or nonmanifold edges after merge")
    if merged.volume < 0:
        warnings.append("negative signed volume suggests inverted normals")
    if not merged.is_watertight:
        warnings.append("mesh is not watertight after merging coincident vertices")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bounds_raw": merged.bounds.tolist(),
        "bounds_m": (merged.bounds * scale).tolist(),
        "extents_m": extents.tolist(),
        "centroid_m": (merged.centroid * scale).tolist(),
        "volume_m3": float(abs(merged.volume) * scale**3),
        "surface_area_m2": float(merged.area * scale**2),
        "raw_face_count": int(len(raw.faces)),
        "raw_vertex_count": int(len(raw.vertices)),
        "merged_face_count": int(len(merged.faces)),
        "merged_vertex_count": int(len(merged.vertices)),
        "watertight": bool(merged.is_watertight),
        "connected_component_count": len(components),
        "component_volumes_m3": [
            float(abs(item.volume) * scale**3)
            for item in sorted(components, key=lambda part: abs(part.volume), reverse=True)
        ],
        "principal_axes_raw": principal_axes.tolist(),
        "candidate_up_raw": [0.0, 0.0, 1.0],
        "inferred_stl_to_m": inferred_scale,
        "degenerate_triangle_count": int(np.count_nonzero(face_areas <= 1.0e-12)),
        "duplicate_face_count": int(duplicate_faces),
        "nonmanifold_edge_count": nonmanifold_edges,
        "warnings": warnings,
    }


def _infer_scale(extents: np.ndarray) -> float:
    largest = float(np.max(extents))
    if 5.0 <= largest <= 1000.0:
        return 0.001
    if 0.005 <= largest <= 1.0:
        return 1.0
    raise GeometryError(
        f"Cannot infer STL unit scale from largest extent {largest:g}; "
        "set units.stl_to_m explicitly and validate the source"
    )


def normalization_transform(name: str, mesh: trimesh.Trimesh, scale: float) -> np.ndarray:
    bounds = np.asarray(mesh.bounds, dtype=float)
    center_xy = (bounds[0, :2] + bounds[1, :2]) / 2.0
    if name == "GR_0.stl":
        origin_raw = np.array([center_xy[0], center_xy[1], bounds[1, 2]])
    elif name == "BB_0.stl":
        origin_raw = np.array([center_xy[0], center_xy[1], bounds[0, 2]])
    else:
        origin_raw = np.array(
            [center_xy[0], center_xy[1], (bounds[0, 2] + bounds[1, 2]) / 2.0]
        )
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -origin_raw * scale
    return transform


def apply_transform_copy(mesh: trimesh.Trimesh, transform: np.ndarray) -> trimesh.Trimesh:
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def _fit_circle_yz(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    yz = points[:, 1:3]
    design = np.column_stack((2.0 * yz[:, 0], 2.0 * yz[:, 1], np.ones(len(yz))))
    target = np.sum(yz**2, axis=1)
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    center = solution[:2]
    radius = math.sqrt(max(0.0, float(solution[2] + np.dot(center, center))))
    distances = np.linalg.norm(yz - center, axis=1)
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)))
    return center, radius, residual


def _section_holes(mesh: trimesh.Trimesh, x_raw: float) -> list[tuple[np.ndarray, float]]:
    section = mesh.section(plane_origin=[x_raw, 0.0, 0.0], plane_normal=[1.0, 0.0, 0.0])
    if section is None:
        return []
    candidates: list[tuple[np.ndarray, float]] = []
    for loop in section.discrete:
        points = np.asarray(loop, dtype=float)
        if len(points) < 8 or np.linalg.norm(points[0] - points[-1]) > 0.1:
            continue
        center_yz, radius, residual = _fit_circle_yz(points)
        diameter = 2.0 * radius
        if 1.30 <= diameter <= 1.70 and residual <= 0.06:
            center = np.array([x_raw, center_yz[0], center_yz[1]], dtype=float)
            candidates.append((center, diameter))
    return candidates


def detect_anchor_pairs(
    mesh: trimesh.Trimesh,
    transform: np.ndarray,
    expected_count: int = 7,
) -> list[AnchorPair]:
    max_abs_x = float(np.max(np.abs(mesh.bounds[:, 0])))
    section_offset = max_abs_x - 4.0
    left = _section_holes(mesh, -section_offset)
    right = _section_holes(mesh, section_offset)
    source = "section_circle_fit"
    if len(left) != expected_count or len(right) != expected_count:
        LOGGER.warning(
            "Automatic anchor detection found %d left and %d right holes; using checked fallback",
            len(left),
            len(right),
        )
        left, right = _fallback_anchors()
        source = "checked_fallback"
    left.sort(key=lambda item: item[0][1])
    right.sort(key=lambda item: item[0][1])
    if len(left) != expected_count or len(right) != expected_count:
        raise GeometryError(
            f"Expected seven string holes per arm, got left={len(left)}, right={len(right)}"
        )
    pairs: list[AnchorPair] = []
    for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
        left_center, left_diameter = left_item
        right_center, right_diameter = right_item
        mismatch = np.linalg.norm(left_center[1:] - right_center[1:])
        if mismatch > 0.25:
            raise GeometryError(
                f"String hole pair {index} differs by {mismatch:.3f} mm in y/z; "
                "left/right pairing is ambiguous"
            )
        left_local = trimesh.transform_points([left_center], transform)[0]
        right_local = trimesh.transform_points([right_center], transform)[0]
        pairs.append(
            AnchorPair(
                index=index,
                left_m=tuple(float(value) for value in left_local),
                right_m=tuple(float(value) for value in right_local),
                diameter_m=float((left_diameter + right_diameter) * 0.0005),
                source=source,
            )
        )
    spacing = np.diff([pair.left_m[1] for pair in pairs])
    if len(spacing) and np.ptp(spacing) > 0.00020:
        raise GeometryError(
            f"Detected hole spacing is not uniform enough: {spacing.tolist()} m"
        )
    return pairs


def _fallback_anchors() -> tuple[list[tuple[np.ndarray, float]], list[tuple[np.ndarray, float]]]:
    y_values = [-17.70, -15.20, -12.70, -10.20, -7.70, -5.20, -2.70]
    left = [(np.array([-49.0, y, -32.65]), 1.5) for y in y_values]
    right = [(np.array([49.0, y, -32.65]), 1.5) for y in y_values]
    return left, right


def recover_target_assembly(
    raw_dir: Path,
    transforms: dict[str, np.ndarray],
) -> dict[str, Any]:
    reference = load_mesh(raw_dir / "Ghast_0.stl")
    components = sorted(
        reference.split(only_watertight=True),
        key=lambda item: abs(item.volume),
        reverse=True,
    )
    if len(components) != 5:
        raise GeometryError(
            f"Ghast_0.stl must contain five watertight components, found {len(components)}"
        )
    block_source = load_mesh(raw_dir / "BB_0.stl")
    washer_source = load_mesh(raw_dir / "SLW_0.stl")
    block_target = components[0]
    washer_targets = components[1:]
    source_volume = abs(block_source.volume)
    if abs(abs(block_target.volume) - source_volume) / source_volume > 1.0e-4:
        raise GeometryError("Largest Ghast component does not match BB_0 volume")
    block_to_ghast, block_cost = _register_rigid(block_source, block_target)
    washer_to_ghast: list[np.ndarray] = []
    washer_costs: list[float] = []
    for target in washer_targets:
        matrix, cost = _register_rigid(washer_source, target)
        washer_to_ghast.append(matrix)
        washer_costs.append(cost)
    # Use registration for orientation, but derive placement from component
    # bounding-box centers. The reference assembly contains re-tessellated washer
    # instances whose star vertices differ slightly even though their volume and
    # extents match, making the center a more reliable placement observable.
    block_bounds = block_target.bounds
    block_origin = np.array(
        [
            np.mean(block_bounds[:, 0]),
            np.mean(block_bounds[:, 1]),
            block_bounds[0, 2],
        ]
    )
    block_rotation = block_to_ghast[:3, :3]
    washer_local: list[np.ndarray] = []
    for target_component, placement in zip(
        washer_targets, washer_to_ghast, strict=True
    ):
        target_center = np.mean(target_component.bounds, axis=0)
        matrix = np.eye(4)
        matrix[:3, :3] = block_rotation.T @ placement[:3, :3]
        matrix[:3, 3] = (
            block_rotation.T @ (target_center - block_origin)
        ) * transforms["BB_0.stl"][0, 0]
        washer_local.append(matrix)
    washer_local.sort(key=lambda matrix: (matrix[0, 3], matrix[1, 3]))
    centers = np.array([matrix[:3, 3] for matrix in washer_local])
    if len(cKDTree(centers).query_pairs(r=0.008)):
        raise GeometryError("Recovered washer transforms overlap")
    x_unique = np.unique(np.round(centers[:, 0], 6))
    y_unique = np.unique(np.round(centers[:, 1], 6))
    if len(x_unique) != 2 or len(y_unique) != 2:
        raise GeometryError(
            "Recovered washers do not form a symmetric two-by-two target pattern"
        )
    return {
        "block_to_ghast_raw": block_to_ghast.tolist(),
        "block_registration_rms_raw": block_cost,
        "washer_transforms_target_m": [matrix.tolist() for matrix in washer_local],
        "washer_registration_rms_raw": washer_costs,
        "washer_centers_target_m": centers.tolist(),
        "spacing_x_m": float(abs(x_unique[1] - x_unique[0])),
        "spacing_y_m": float(abs(y_unique[1] - y_unique[0])),
    }


def recover_urdf_target_assembly(
    assembly: UrdfAssembly,
    block_transform: np.ndarray,
    legacy_washer_source: trimesh.Trimesh,
    legacy_washer_scale: float,
) -> dict[str, Any]:
    """Convert explicit URDF fixed-joint washer poses into block simulation space."""
    urdf_washer = load_scaled_mesh(assembly.washers[0])
    legacy_washer = legacy_washer_source.copy()
    legacy_washer.apply_scale(legacy_washer_scale)
    extent_error = float(
        np.max(np.abs(np.asarray(urdf_washer.extents) - legacy_washer.extents))
    )
    volume_error = float(
        abs(abs(urdf_washer.volume) - abs(legacy_washer.volume))
        / abs(legacy_washer.volume)
    )
    if extent_error > 0.0001 or volume_error > 0.02:
        raise GeometryError(
            "URDF washer mesh does not match SLW_0.stl closely enough: "
            f"extent_error={extent_error:.6g} m, volume_error={volume_error:.2%}"
        )

    urdf_washer_normalization = normalization_transform(
        "SLW_0.stl", urdf_washer, 1.0
    )
    washer_local = [
        block_transform
        @ block_from_washer
        @ np.linalg.inv(urdf_washer_normalization)
        for block_from_washer in assembly.block_mesh_from_washer_mesh
    ]
    washer_local.sort(key=lambda matrix: (matrix[0, 3], matrix[1, 3]))
    centers = np.array([matrix[:3, 3] for matrix in washer_local])
    if len(cKDTree(centers).query_pairs(r=0.008)):
        raise GeometryError("URDF washer transforms overlap")
    x_unique = np.unique(np.round(centers[:, 0], 6))
    y_unique = np.unique(np.round(centers[:, 1], 6))
    if len(x_unique) != 2 or len(y_unique) != 2:
        raise GeometryError(
            "URDF washers do not form a symmetric two-by-two target pattern"
        )
    return {
        "source": "urdf_fixed_joints",
        "urdf": assembly.metadata(),
        "washer_transforms_target_m": [matrix.tolist() for matrix in washer_local],
        "washer_centers_target_m": centers.tolist(),
        "spacing_x_m": float(abs(x_unique[1] - x_unique[0])),
        "spacing_y_m": float(abs(y_unique[1] - y_unique[0])),
        "legacy_washer_extent_error_m": extent_error,
        "legacy_washer_volume_error": volume_error,
    }


def _register_rigid(
    source: trimesh.Trimesh, target: trimesh.Trimesh
) -> tuple[np.ndarray, float]:
    initial = np.eye(4)
    initial[:3, 3] = target.centroid - source.centroid
    try:
        matrix, transformed, cost = trimesh.registration.icp(
            source.vertices,
            target.vertices,
            initial=initial,
            threshold=1.0e-7,
            max_iterations=100,
            reflection=False,
            scale=False,
        )
        rms = float(np.sqrt(np.mean(cKDTree(target.vertices).query(transformed)[0] ** 2)))
        if not np.isfinite(rms) or rms > 0.02:
            raise GeometryError(f"ICP residual {rms:g} raw units is too high")
        return np.asarray(matrix), rms
    except Exception as first_exc:
        try:
            # trimesh's surface sampler uses NumPy's legacy global generator.
            # Preserve caller state while making this fallback reproducible.
            random_state = np.random.get_state()
            np.random.seed(1)
            try:
                matrix, _ = trimesh.registration.mesh_other(
                    source,
                    target,
                    samples=min(1200, max(500, len(source.vertices))),
                    scale=False,
                    icp_first=30,
                    icp_final=100,
                )
            finally:
                np.random.set_state(random_state)
            transformed = trimesh.transform_points(source.vertices, matrix)
            rms = float(
                np.sqrt(np.mean(cKDTree(target.vertices).query(transformed)[0] ** 2))
            )
            # The reference assembly re-tessellates/rotates the washer instances.
            # A 0.2 mm vertex RMS is strict relative to the 9 mm outer diameter
            # while allowing the principal-axis ambiguity of the eightfold star.
            if rms > 0.2:
                raise GeometryError(f"principal-axis ICP residual is {rms:g}")
            return np.asarray(matrix), rms
        except Exception as second_exc:
            raise GeometryError(
                f"Rigid component registration failed: {first_exc}; {second_exc}"
            ) from second_exc


def save_anchor_diagnostic(
    mesh: trimesh.Trimesh,
    pairs: Iterable[AnchorPair],
    path: Path,
) -> None:
    pair_list = list(pairs)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, side in zip(axes, ("left", "right"), strict=True):
        points = np.array(
            [getattr(pair, f"{side}_m") for pair in pair_list], dtype=float
        )
        axis.scatter(points[:, 1] * 1000.0, points[:, 2] * 1000.0, s=50)
        for pair, point in zip(pair_list, points, strict=True):
            axis.annotate(str(pair.index), (point[1] * 1000.0, point[2] * 1000.0))
        axis.set(
            title=f"{side.title()} inner-mouth anchors",
            xlabel="Y (mm)",
            ylabel="Z (mm)",
            aspect="equal",
        )
        axis.grid(True, alpha=0.3)
    figure.suptitle(
        f"GR_0 string-hole detection — {pair_list[0].source if pair_list else 'none'}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


class GeometryPipeline:
    """Hash-keyed preprocessing pipeline. Raw assets are never modified."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.raw_dir = config.paths.raw_assets
        self.processed_dir = config.paths.processed_assets

    def inspect(self, *, write: bool = True) -> dict[str, Any]:
        missing = [name for name in ASSET_NAMES if not (self.raw_dir / name).is_file()]
        if missing:
            raise GeometryError(f"Missing raw assets: {', '.join(missing)}")
        report = {
            "stl_to_m": self.config.units.stl_to_m,
            "assets": {
                name: inspect_asset(self.raw_dir / name, self.config.units.stl_to_m)
                for name in ASSET_NAMES
            },
        }
        if self.config.paths.assembly_urdf is not None:
            try:
                assembly = load_fixed_mesh_assembly(
                    self.config.paths.assembly_urdf
                )
            except UrdfAssetError as exc:
                raise GeometryError(str(exc)) from exc
            report["urdf_assembly"] = {
                **assembly.metadata(),
                "block_extents_m": list(assembly.block.extents_m),
                "block_volume_m3": assembly.block.volume_m3,
                "washer_extents_m": list(assembly.washers[0].extents_m),
                "washer_volume_m3": assembly.washers[0].volume_m3,
            }
        if write:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            (self.processed_dir / "asset_inspection.json").write_text(
                json.dumps(report, indent=2)
            )
        return report

    def preprocess(self, *, force: bool = False) -> dict[str, Any]:
        inspection = self.inspect(write=True)
        assembly: UrdfAssembly | None = None
        urdf_hashes: dict[str, str] = {}
        if self.config.paths.assembly_urdf is not None:
            try:
                assembly = load_fixed_mesh_assembly(
                    self.config.paths.assembly_urdf
                )
            except UrdfAssetError as exc:
                raise GeometryError(str(exc)) from exc
            urdf_hashes[str(assembly.path)] = sha256_file(assembly.path)
            urdf_hashes.update(
                {
                    str(path): sha256_file(path)
                    for path in assembly.referenced_meshes
                }
            )
        key_payload = {
            "pipeline_version": 5,
            "assets": {
                name: item["sha256"] for name, item in inspection["assets"].items()
            },
            "urdf_assets": urdf_hashes,
            "scale": self.config.units.stl_to_m,
            "collision": {
                "backend": self.config.collision.backend,
                "threshold": self.config.collision.coacd_threshold,
                "max_hulls": self.config.collision.coacd_max_hulls,
            },
        }
        cache_key = hashlib.sha256(
            json.dumps(key_payload, sort_keys=True).encode()
        ).hexdigest()
        manifest_path = self.processed_dir / "manifest.json"
        if not force and manifest_path.is_file():
            existing = json.loads(manifest_path.read_text())
            if existing.get("cache_key") == cache_key:
                LOGGER.info("Processed assets are current (%s)", cache_key[:12])
                return existing
        transforms: dict[str, np.ndarray] = {}
        processed_meshes: dict[str, str] = {}
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for name in ("GR_0.stl", "BB_0.stl", "SLW_0.stl"):
            if name == "BB_0.stl" and assembly is not None:
                mesh = load_scaled_mesh(assembly.block)
                scale = 1.0
            else:
                mesh = load_mesh(self.raw_dir / name)
                scale = self.config.units.stl_to_m
            transform = normalization_transform(name, mesh, scale)
            transforms[name] = transform
            normalized = apply_transform_copy(mesh, transform)
            destination = self.processed_dir / name
            normalized.export(destination)
            processed_meshes[name] = str(destination)
        gripper_raw = load_mesh(self.raw_dir / "GR_0.stl")
        anchors = detect_anchor_pairs(
            gripper_raw, transforms["GR_0.stl"], self.config.strings.count
        )
        save_anchor_diagnostic(
            gripper_raw, anchors, self.processed_dir / "anchor_detection.png"
        )
        if assembly is None:
            target = recover_target_assembly(self.raw_dir, transforms)
            block_source = {
                "kind": "legacy_stl",
                "path": str(self.raw_dir / "BB_0.stl"),
            }
        else:
            target = recover_urdf_target_assembly(
                assembly,
                transforms["BB_0.stl"],
                load_mesh(self.raw_dir / "SLW_0.stl"),
                self.config.units.stl_to_m,
            )
            block_source = {
                "kind": "urdf",
                **assembly.metadata(),
            }
        from .collision_proxies import build_block_collision_proxy

        collision = build_block_collision_proxy(
            self.processed_dir / "BB_0.stl",
            self.processed_dir,
            self.config,
            exclude_hook_intrusions=assembly is not None,
        )
        manifest = {
            "cache_key": cache_key,
            "key_inputs": key_payload,
            "inspection_path": str(self.processed_dir / "asset_inspection.json"),
            "processed_meshes": processed_meshes,
            "block_source": block_source,
            "raw_to_sim_transforms": {
                name: matrix.tolist() for name, matrix in transforms.items()
            },
            "anchors": [pair.to_dict() for pair in anchors],
            "target_assembly": target,
            "collision_proxy": collision,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        return manifest
