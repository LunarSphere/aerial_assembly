"""Typed YAML configuration and command-line override handling."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    """Raised when configuration is missing, unknown, or physically invalid."""


@dataclass
class UnitsConfig:
    stl_to_m: float = 0.001


@dataclass
class PathsConfig:
    raw_assets: Path = Path("assets/raw")
    processed_assets: Path = Path("assets/processed")
    outputs: Path = Path("outputs")
    assembly_urdf: Path | None = None


@dataclass
class SimulationConfig:
    timestep_s: float = 0.00025
    duration_limit_s: float = 30.0
    seed: int = 1
    headless: bool = True
    integrator: Literal["implicitfast", "implicit", "euler", "rk4"] = "implicitfast"
    solver: Literal["Newton", "CG", "PGS"] = "Newton"
    iterations: int = 100
    ls_iterations: int = 20
    contact_margin_m: float = 0.0001
    contact_gap_m: float = 0.0
    solref: tuple[float, float] = (0.003, 1.0)
    solimp: tuple[float, float, float, float, float] = (0.9, 0.95, 0.001, 0.5, 2.0)
    max_kinetic_energy_j: float = 5.0
    max_constraint_error_m: float = 0.005


@dataclass
class PayloadConfig:
    mass_kg: float = 0.025
    density_mode: bool = False
    density_kg_m3: float = 1200.0
    ground_friction: tuple[float, float, float] = (0.7, 0.01, 0.001)
    block_friction: tuple[float, float, float] = (0.6, 0.01, 0.001)


@dataclass
class StringConfig:
    count: int = 7
    backend: Literal["cable", "flex", "segmented"] = "cable"
    radius_m: float = 0.00030
    segments_per_string: int = 48
    slack_length_m: float = 0.00100
    endpoint_error_limit_m: float = 0.00005
    pretension_n: float = 0.10
    axial_stiffness_n_per_m: float = 1200.0
    damping_n_s_per_m: float = 0.5
    density_kg_per_m3: float = 1100.0
    friction: tuple[float, float, float] = (0.5, 0.01, 0.001)
    bending_stiffness: float = 1.0e-8
    pretension_tolerance_n: float = 0.04
    settle_time_s: float = 0.15
    numerical_mass_scale: float = 1.0


@dataclass
class WasherConfig:
    mode: Literal["compliant_fingers", "deformable_flex"] = "compliant_fingers"
    finger_count: int = 8
    young_modulus_pa: float = 8.0e6
    poisson_ratio: float = 0.45
    damping: float = 0.02
    density_kg_per_m3: float = 1200.0
    peg_friction: tuple[float, float, float] = (0.8, 0.02, 0.002)
    effective_stiffness_nm_rad: float = 0.006
    finger_joint_damping: float = 0.0004
    mesh_size_m: float = 0.00055
    insertion_fraction: float = 0.45
    numerical_mass_scale: float = 1000.0


@dataclass
class ControllerConfig:
    engagement_axis: Literal["x", "y"] = "y"
    engagement_sign: Literal["auto", "positive", "negative"] = "auto"
    release_vector: tuple[float, float, float] = (0.0, -1.0, 0.0)
    approach_speed_m_s: float = 0.02
    engagement_speed_m_s: float = 0.01
    takeup_speed_m_s: float = 0.005
    lift_speed_m_s: float = 0.015
    press_speed_m_s: float = 0.003
    release_speed_m_s: float = 0.01
    max_downward_force_n: float = 30.0
    max_upward_force_n: float = 5.0
    max_horizontal_force_n: float = 30.0
    max_torque_nm: float = 0.08
    force_slowdown_fraction: float = 0.8
    pickup_lift_m: float = 0.025
    transport_lift_m: float = 0.055
    placement_press_distance_m: float = 0.030
    max_takeup_m: float = 0.030
    engagement_distance_m: float = 0.0055
    ramp_follow_drop_m: float = 0.0100
    seating_distance_m: float = 0.0
    seating_rise_m: float = 0.0030
    release_distance_m: float = 0.025
    hold_duration_s: float = 0.5
    slack_tension_n: float = 0.04
    taut_reaction_n: float = 0.020
    capture_hold_s: float = 0.050
    minimum_captured_strings: int = 4


@dataclass
class CollisionConfig:
    backend: Literal["coacd", "procedural"] = "coacd"
    coacd_threshold: float = 0.03
    coacd_max_hulls: int = 64
    gripper_enabled: bool = False


@dataclass
class OutputConfig:
    sample_hz: float = 200.0
    render_width: int = 960
    render_height: int = 720
    render_fps: int = 30
    debug_geometry: bool = True


@dataclass
class MetricsConfig:
    pickup_min_lift_m: float = 0.020
    pickup_hold_s: float = 0.5
    max_payload_tilt_deg: float = 20.0
    release_clear_s: float = 0.25
    placed_translation_tolerance_m: float = 0.003
    placed_rotation_tolerance_deg: float = 5.0
    contact_force_epsilon_n: float = 0.01
    retention_force_epsilon_n: float = 0.02


@dataclass
class PerturbationConfig:
    payload_xy_offset_m: tuple[float, float] = (0.0, 0.0)
    payload_yaw_deg: float = 0.0
    gripper_height_error_m: float = 0.0
    peg_xy_misalignment_m: tuple[float, float] = (0.0, 0.0)


@dataclass
class AppConfig:
    units: UnitsConfig = field(default_factory=UnitsConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    payload: PayloadConfig = field(default_factory=PayloadConfig)
    strings: StringConfig = field(default_factory=StringConfig)
    washer: WasherConfig = field(default_factory=WasherConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    collision: CollisionConfig = field(default_factory=CollisionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    perturbations: PerturbationConfig = field(default_factory=PerturbationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path)
        try:
            raw = yaml.safe_load(config_path.read_text()) or {}
        except OSError as exc:
            raise ConfigError(f"Cannot read configuration {config_path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError(f"Top-level YAML value in {config_path} must be a mapping")
        config = _construct_dataclass(cls, raw, "")
        config.validate()
        return config

    def with_overrides(self, overrides: list[str]) -> "AppConfig":
        raw = dataclasses.asdict(self)
        for override in overrides:
            if "=" not in override:
                raise ConfigError(f"Override must be section.key=value, got {override!r}")
            dotted_key, encoded = override.split("=", 1)
            parts = dotted_key.split(".")
            cursor: Any = raw
            for part in parts[:-1]:
                if not isinstance(cursor, dict) or part not in cursor:
                    raise ConfigError(f"Unknown override path {dotted_key!r}")
                cursor = cursor[part]
            if not isinstance(cursor, dict) or parts[-1] not in cursor:
                raise ConfigError(f"Unknown override path {dotted_key!r}")
            cursor[parts[-1]] = yaml.safe_load(encoded)
        result = _construct_dataclass(type(self), raw, "")
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "units.stl_to_m": self.units.stl_to_m,
            "simulation.timestep_s": self.simulation.timestep_s,
            "simulation.duration_limit_s": self.simulation.duration_limit_s,
            "payload.mass_kg": self.payload.mass_kg,
            "strings.radius_m": self.strings.radius_m,
            "strings.axial_stiffness_n_per_m": self.strings.axial_stiffness_n_per_m,
            "strings.numerical_mass_scale": self.strings.numerical_mass_scale,
            "strings.endpoint_error_limit_m": self.strings.endpoint_error_limit_m,
            "washer.young_modulus_pa": self.washer.young_modulus_pa,
            "washer.mesh_size_m": self.washer.mesh_size_m,
            "washer.numerical_mass_scale": self.washer.numerical_mass_scale,
            "controller.max_downward_force_n": self.controller.max_downward_force_n,
            "controller.max_upward_force_n": self.controller.max_upward_force_n,
            "controller.takeup_speed_m_s": self.controller.takeup_speed_m_s,
            "controller.transport_lift_m": self.controller.transport_lift_m,
            "controller.placement_press_distance_m": (
                self.controller.placement_press_distance_m
            ),
            "controller.ramp_follow_drop_m": self.controller.ramp_follow_drop_m,
            "controller.seating_rise_m": self.controller.seating_rise_m,
            "metrics.retention_force_epsilon_n": self.metrics.retention_force_epsilon_n,
        }
        bad = [name for name, value in positive.items() if value <= 0]
        if bad:
            raise ConfigError(f"Values must be positive: {', '.join(bad)}")
        if self.strings.count != 7:
            raise ConfigError("This geometry requires exactly seven strings")
        if self.strings.segments_per_string < 8:
            raise ConfigError("strings.segments_per_string must be at least 8")
        if self.strings.slack_length_m < 0:
            raise ConfigError("strings.slack_length_m must be non-negative")
        if abs(self.controller.seating_distance_m) > 0.02:
            raise ConfigError(
                "controller.seating_distance_m magnitude must not exceed 0.02 m"
            )
        if not 1 <= self.controller.minimum_captured_strings <= self.strings.count:
            raise ConfigError(
                "controller.minimum_captured_strings must be between 1 and strings.count"
            )
        if not 0.0 <= self.washer.poisson_ratio < 0.5:
            raise ConfigError("washer.poisson_ratio must be in [0, 0.5)")
        if self.simulation.timestep_s > self.strings.radius_m:
            raise ConfigError(
                "The timestep is too large for millimeter string collision; "
                "simulation.timestep_s must not exceed strings.radius_m"
            )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(dataclasses.asdict(self))


def _construct_dataclass(cls: type[Any], raw: dict[str, Any], path: str) -> Any:
    fields = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        prefix = f"{path}." if path else ""
        raise ConfigError(f"Unknown configuration keys: {', '.join(prefix + key for key in unknown)}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in raw.items():
        annotation = hints[name]
        nested = _dataclass_type(annotation)
        if nested is not None:
            if not isinstance(value, dict):
                raise ConfigError(f"{path + '.' if path else ''}{name} must be a mapping")
            kwargs[name] = _construct_dataclass(nested, value, f"{path}.{name}".strip("."))
        else:
            kwargs[name] = _coerce_value(annotation, value, f"{path}.{name}".strip("."))
    return cls(**kwargs)


def _dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        return annotation
    return None


def _coerce_value(annotation: Any, value: Any, path: str) -> Any:
    origin = get_origin(annotation)
    if annotation is Path:
        return Path(value)
    if origin is Literal:
        if value not in get_args(annotation):
            raise ConfigError(f"{path} must be one of {get_args(annotation)}, got {value!r}")
        return value
    if origin is tuple:
        args = get_args(annotation)
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{path} must contain {len(args)} values")
        return tuple(_coerce_value(kind, item, path) for kind, item in zip(args, value, strict=True))
    if origin in (Union, UnionType):
        for kind in get_args(annotation):
            try:
                return _coerce_value(kind, value, path)
            except (TypeError, ValueError, ConfigError):
                continue
        raise ConfigError(f"{path} has invalid value {value!r}")
    if annotation in (float, int, str, bool):
        if annotation is bool and not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean")
        try:
            return annotation(value)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{path} has invalid value {value!r}") from exc
    return copy.deepcopy(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
