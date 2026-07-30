"""Deterministic collision decomposition with a cavity-preserving fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from .config import AppConfig
from .geometry import GeometryError, load_mesh

LOGGER = logging.getLogger(__name__)


def build_block_collision_proxy(
    normalized_block_path: Path,
    output_dir: Path,
    config: AppConfig,
) -> dict[str, Any]:
    mesh = load_mesh(normalized_block_path)
    backend = config.collision.backend
    fallback_reason: str | None = None
    parts: list[trimesh.Trimesh] = []
    if backend == "coacd":
        try:
            parts = _coacd_parts(
                mesh,
                threshold=config.collision.coacd_threshold,
                max_hulls=config.collision.coacd_max_hulls,
            )
        except Exception as exc:
            fallback_reason = f"CoACD unavailable or failed: {type(exc).__name__}: {exc}"
            LOGGER.warning("%s; using voxel-box procedural proxy", fallback_reason)
            backend = "procedural"
    proxy_dir = output_dir / "bb_collision"
    proxy_dir.mkdir(parents=True, exist_ok=True)
    if backend == "coacd":
        part_paths: list[str] = []
        for index, part in enumerate(parts):
            path = proxy_dir / f"hull_{index:03d}.stl"
            part.export(path)
            part_paths.append(str(path))
        metrics = validate_proxy(mesh, parts)
        if not metrics["valid"]:
            raise GeometryError(
                "Convex decomposition failed hook-cavity validation: "
                + "; ".join(metrics["errors"])
            )
        diagnostic = output_dir / "collision_proxy_overlay.png"
        _save_proxy_diagnostic(mesh, parts, diagnostic)
        return {
            "backend": "coacd",
            "fallback_reason": fallback_reason,
            "parts": part_paths,
            "procedural_boxes": None,
            "validation": metrics,
            "diagnostic": str(diagnostic),
        }
    boxes = _voxel_box_proxy(mesh, pitch=0.0010)
    boxes_path = proxy_dir / "procedural_boxes.json"
    boxes_path.write_text(json.dumps(boxes, indent=2))
    return {
        "backend": "procedural_voxel_boxes",
        "fallback_reason": fallback_reason,
        "parts": [],
        "procedural_boxes": str(boxes_path),
        "validation": {
            "valid": True,
            "errors": [],
            "pitch_m": 0.001,
            "note": "Surface voxel boxes preserve openings but are slower than CoACD.",
        },
        "diagnostic": None,
    }


def _coacd_parts(
    mesh: trimesh.Trimesh, *, threshold: float, max_hulls: int
) -> list[trimesh.Trimesh]:
    import coacd

    coacd_mesh = coacd.Mesh(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    try:
        raw_parts = coacd.run_coacd(
            coacd_mesh,
            threshold=threshold,
            max_convex_hull=max_hulls,
            preprocess_mode="auto",
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=150,
            seed=1,
        )
    except TypeError:
        raw_parts = coacd.run_coacd(coacd_mesh, threshold=threshold)
    parts = [
        trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        for vertices, faces in raw_parts
    ]
    parts = [part for part in parts if len(part.faces) >= 4 and part.volume > 1.0e-12]
    if not parts:
        raise GeometryError("CoACD returned no usable convex pieces")
    if len(parts) > max_hulls:
        raise GeometryError(f"CoACD returned {len(parts)} hulls, limit is {max_hulls}")
    return parts


def validate_proxy(
    visual_mesh: trimesh.Trimesh, parts: list[trimesh.Trimesh]
) -> dict[str, Any]:
    hull_volume = float(visual_mesh.convex_hull.volume)
    source_volume = float(abs(visual_mesh.volume))
    proxy_volume = float(sum(abs(part.volume) for part in parts))
    fill_ratio = proxy_volume / hull_volume
    volume_error = abs(proxy_volume - source_volume) / source_volume
    errors: list[str] = []
    # A global hull fills the pickup undercuts. The supplied part has substantially
    # less volume than that hull; a valid decomposition should retain this signal.
    source_fill_ratio = source_volume / hull_volume
    if fill_ratio > min(0.98, source_fill_ratio + 0.18):
        errors.append(
            f"proxy fills too much of the global convex hull ({fill_ratio:.3f})"
        )
    if volume_error > 0.35:
        errors.append(f"proxy/source volume differs by {volume_error:.1%}")
    if len(parts) < 2:
        errors.append("proxy is a single convex hull and cannot preserve undercuts")
    return {
        "valid": not errors,
        "errors": errors,
        "part_count": len(parts),
        "source_volume_m3": source_volume,
        "proxy_volume_m3": proxy_volume,
        "global_hull_volume_m3": hull_volume,
        "source_fill_ratio": source_fill_ratio,
        "proxy_fill_ratio": fill_ratio,
        "relative_volume_error": volume_error,
    }


def _voxel_box_proxy(mesh: trimesh.Trimesh, pitch: float) -> list[dict[str, list[float]]]:
    voxel = mesh.voxelized(pitch).fill()
    points = np.asarray(voxel.points)
    if len(points) > 20_000:
        raise GeometryError(
            f"Procedural proxy would require {len(points)} boxes; install/fix CoACD"
        )
    half = pitch / 2.0
    return [
        {
            "center_m": [float(value) for value in point],
            "halfsize_m": [half, half, half],
        }
        for point in points
    ]


def _save_proxy_diagnostic(
    source: trimesh.Trimesh,
    parts: list[trimesh.Trimesh],
    path: Path,
) -> None:
    """Save orthographic source/proxy overlays for cavity inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    proxy_vertices = np.vstack([part.vertices for part in parts])
    source_vertices = np.asarray(source.vertices)
    views = ((0, 1, "XY"), (1, 2, "YZ"), (0, 2, "XZ"))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for axis, (horizontal, vertical, label) in zip(axes, views, strict=True):
        axis.scatter(
            proxy_vertices[:, horizontal] * 1000,
            proxy_vertices[:, vertical] * 1000,
            s=1,
            c="#f08a24",
            alpha=0.35,
            label="convex proxy",
        )
        axis.scatter(
            source_vertices[:, horizontal] * 1000,
            source_vertices[:, vertical] * 1000,
            s=2,
            c="#2457a6",
            alpha=0.65,
            label="source",
        )
        axis.set(title=label, xlabel="mm", ylabel="mm", aspect="equal")
        axis.grid(alpha=0.2)
    axes[0].legend(markerscale=4)
    figure.suptitle("BB_0 visual mesh vs. collision decomposition")
    figure.savefig(path, dpi=180)
    plt.close(figure)
