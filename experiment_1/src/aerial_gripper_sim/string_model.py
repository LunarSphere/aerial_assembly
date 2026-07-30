"""Physical colliding string backends and string telemetry."""

from __future__ import annotations

import abc
import math
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from .config import StringConfig


@dataclass
class StringTelemetry:
    name: str
    segment_tension_n: list[float]
    mean_tension_n: float
    peak_tension_n: float
    endpoint_tension_n: tuple[float, float]
    current_length_m: float
    rest_length_m: float
    slack_m: float
    contact_count: int
    contact_impulse_ns: float
    minimum_payload_distance_m: float


class StringModel(abc.ABC):
    """Backend-neutral colliding cable interface."""

    def __init__(self, config: StringConfig, anchors: list[dict[str, Any]]):
        self.config = config
        self.anchors = anchors

    @abc.abstractmethod
    def xml_fragments(self, count: int | None = None) -> list[str]:
        """Return MJCF fragments to place inside the gripper body."""

    @abc.abstractmethod
    def initialize_pretension(self, model: mujoco.MjModel) -> dict[str, float]:
        """Shorten physical rest lengths and report requested rest geometry."""

    @abc.abstractmethod
    def sample(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        payload_geom_ids: set[int],
    ) -> list[StringTelemetry]:
        """Sample physical cable state."""


class FlexStringModel(StringModel):
    """MuJoCo 1-D flex cables with colliding capsule elements."""

    def __init__(self, config: StringConfig, anchors: list[dict[str, Any]]):
        super().__init__(config, anchors)
        self.names: list[str] = []

    def xml_fragments(self, count: int | None = None) -> list[str]:
        selected = self.anchors[: count if count is not None else len(self.anchors)]
        fragments: list[str] = []
        self.names = []
        segments = self.config.segments_per_string
        area = math.pi * self.config.radius_m**2
        for anchor in selected:
            left = np.asarray(anchor["left_m"], dtype=float)
            right = np.asarray(anchor["right_m"], dtype=float)
            span = float(np.linalg.norm(right - left))
            center = (left + right) / 2.0
            name = f"string_{int(anchor['index'])}"
            self.names.append(name)
            spacing = span / segments
            # Flex contact at sub-millimetre radius is ill-conditioned when each
            # numerical node has sub-microgram mass. The scale is explicit and
            # recorded; reported material density remains the physical value.
            mass = (
                self.config.density_kg_per_m3
                * area
                * span
                * self.config.numerical_mass_scale
            )
            # The supplied pairs differ negligibly in y/z, so a grid along local
            # X exactly connects their fitted inner-mouth centers.
            fragments.append(
                f"""
                <flexcomp name="{name}" type="grid" dim="1"
                  count="{segments + 1} 1 1"
                  spacing="{spacing:.12g} {spacing:.12g} {spacing:.12g}"
                  pos="{center[0]:.12g} {center[1]:.12g} {center[2]:.12g}"
                  radius="{self.config.radius_m:.12g}"
                  mass="{mass:.12g}" inertiabox="{self.config.radius_m * 2:.12g}"
                  rgba="0.95 0.25 0.08 1">
                  <edge equality="true" solref="0.003 1"
                    solimp="0.9 0.95 0.001 0.5 2"
                    stiffness="0" damping="{self.config.damping_n_s_per_m:.12g}"/>
                  <contact contype="4" conaffinity="26" condim="3"
                    friction="{' '.join(map(str, self.config.friction))}"
                    solref="0.012 1" solimp="0.7 0.9 0.002 0.5 2"
                    margin="0.00002" gap="0" selfcollide="none"/>
                  <pin id="0 {segments}"/>
                </flexcomp>
                """
            )
        return fragments

    def initialize_pretension(self, model: mujoco.MjModel) -> dict[str, float]:
        achieved: dict[str, float] = {}
        extension_per_edge = (
            self.config.pretension_n / self.config.axial_stiffness_n_per_m
        )
        for name in self.names:
            flex_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_FLEX, name
            )
            if flex_id < 0:
                continue
            start = int(model.flex_edgeadr[flex_id])
            number = int(model.flex_edgenum[flex_id])
            current_rest = model.flexedge_length0[start : start + number]
            if np.any(current_rest <= extension_per_edge):
                raise ValueError(
                    f"Pretension {self.config.pretension_n} N would require "
                    f"non-positive rest edges in {name}"
                )
            current_rest[:] -= extension_per_edge
            achieved[name] = float(np.sum(current_rest))
        return achieved

    def sample(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        payload_geom_ids: set[int],
    ) -> list[StringTelemetry]:
        result: list[StringTelemetry] = []
        for name in self.names:
            flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, name)
            start = int(model.flex_edgeadr[flex_id])
            number = int(model.flex_edgenum[flex_id])
            edge_slice = slice(start, start + number)
            lengths = np.asarray(data.flexedge_length[edge_slice])
            rest = np.asarray(model.flexedge_length0[edge_slice])
            velocities = np.asarray(data.flexedge_velocity[edge_slice])
            tension = np.maximum(
                0.0,
                self.config.axial_stiffness_n_per_m * (lengths - rest)
                + self.config.damping_n_s_per_m * velocities,
            )
            contact_count, impulse = _flex_payload_contacts(
                model, data, flex_id, payload_geom_ids
            )
            vertex_start = int(model.flex_vertadr[flex_id])
            vertex_count = int(model.flex_vertnum[flex_id])
            vertices = np.asarray(
                data.flexvert_xpos[vertex_start : vertex_start + vertex_count]
            )
            result.append(
                StringTelemetry(
                    name=name,
                    segment_tension_n=tension.tolist(),
                    mean_tension_n=float(np.mean(tension)),
                    peak_tension_n=float(np.max(tension)),
                    endpoint_tension_n=(float(tension[0]), float(tension[-1])),
                    current_length_m=float(np.sum(lengths)),
                    rest_length_m=float(np.sum(rest)),
                    slack_m=float(max(0.0, np.sum(rest) - np.linalg.norm(vertices[-1] - vertices[0]))),
                    contact_count=contact_count,
                    contact_impulse_ns=impulse,
                    minimum_payload_distance_m=_minimum_hook_distance(
                        model, data, vertices, payload_geom_ids
                    ),
                )
            )
        return result


class SegmentedStringModel(FlexStringModel):
    """Documented deterministic fallback.

    MuJoCo's native 1-D flex is itself a segmented colliding capsule cable, so
    this fallback intentionally uses the same generated topology with stiffer
    edge constraints and fewer segments. Keeping it behind the same interface
    makes fallback selection explicit in run metadata.
    """

    def xml_fragments(self, count: int | None = None) -> list[str]:
        original_segments = self.config.segments_per_string
        self.config.segments_per_string = max(12, original_segments // 2)
        try:
            return super().xml_fragments(count)
        finally:
            self.config.segments_per_string = original_segments


def make_string_model(
    config: StringConfig, anchors: list[dict[str, Any]]
) -> StringModel:
    if config.backend == "flex":
        return FlexStringModel(config, anchors)
    return SegmentedStringModel(config, anchors)


def _flex_payload_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    flex_id: int,
    payload_geom_ids: set[int],
) -> tuple[int, float]:
    count = 0
    impulse = 0.0
    force = np.zeros(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        flex_values = {
            int(value)
            for value in np.asarray(contact.flex).reshape(-1)
            if int(value) >= 0
        }
        geom_values = {
            int(value)
            for value in np.asarray(contact.geom).reshape(-1)
            if int(value) >= 0
        }
        if flex_id in flex_values and geom_values.intersection(payload_geom_ids):
            count += 1
            mujoco.mj_contactForce(model, data, index, force)
            impulse += float(np.linalg.norm(force[:3]) * model.opt.timestep)
    return count, impulse


def _minimum_hook_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    vertices_world: np.ndarray,
    payload_geom_ids: set[int],
) -> float:
    """Distance to the four measured hook-valley centerlines."""
    if not payload_geom_ids:
        return 1.0
    geom_id = next(iter(payload_geom_ids))
    body_id = int(model.geom_bodyid[geom_id])
    rotation = np.asarray(data.xmat[body_id]).reshape(3, 3)
    local = (vertices_world - data.xpos[body_id]) @ rotation
    within_span = np.abs(local[:, 0]) <= 0.0175
    candidates = local[within_span]
    if not len(candidates):
        candidates = local
    centers_y = np.asarray((-0.0055, -0.0010, 0.0035, 0.0080))
    dy = candidates[:, None, 1] - centers_y[None, :]
    dz = candidates[:, None, 2] - 0.041
    return float(np.min(np.sqrt(dy**2 + dz**2)))
