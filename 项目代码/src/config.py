from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Mapping
import math
import warnings

import numpy as np
import yaml


# Coordinate convention used by the 2D solver:
# - component 0: x axis, positive to the right
# - component 1: z (vertical) axis, positive upward
# The task-book outlet ray points backward/downward from the AUV:
# d_out = [-cos(theta), -sin(theta)], with theta measured below the -x axis.
THETA_OUT_DEG = 37.5
THETA_OUT_RAD = float(np.deg2rad(THETA_OUT_DEG))
D_OUT = np.array(
    [-np.cos(THETA_OUT_RAD), -np.sin(THETA_OUT_RAD)],
    dtype=float,
)


@dataclass(frozen=True)
class SimulationConfig:
    """All model, numerical and output parameters.

    Values marked as demonstration defaults in ``configs/cable_1p0mm.yaml`` must be
    replaced or calibrated before engineering conclusions are drawn.
    """

    # Case definition
    case_id: str = "default_case"
    v_A: float = 0.20
    h_A: float = 0.60
    v_out: float = 0.20
    t_end: float = 12.0
    dt: float = 0.00025
    ramp_time: float = 2.0
    # Extra initially deployed material is required for a non-vertical outlet
    # direction when the anchor starts directly below the AUV.
    initial_extra_length: float = 0.02
    # Measured outlet direction is not supplied by the task book. 37.5 deg is
    # retained as the legacy default; production YAML files must set it.
    outlet_angle_deg: float = THETA_OUT_DEG
    enforce_outlet_direction: bool = True

    # Cable
    # 按你提供的表格：外径 0.001 m，线密度 0.00110 kg/m，断裂力 500 N
    diameter: float = 0.001
    line_density: float = 0.00110
    # EA 需要查资料并选取合理的默认值；此处给出基于轻质聚合物包层类缆线的保守估计（N）
    EA: float = 2000.0
    EI: float = 2e-5
    ds: float = 0.02
    axial_damping: float = 0.002
    # Kelvin-Voigt bending time constant; bending damping coefficient is EI*tau.
    bending_damping_time: float = 0.01
    # Compressive edge regularization preserves material arc length while
    # still allowing macroscopic cable buckling through the bending model.
    # A 10% ratio is the lowest value that remains mesh-stable at ds=1 mm.
    compression_stiffness_ratio: float = 0.10
    payout_buffer_segments: int = 8
    gravity: float = 9.80665
    # 附加质量系数：针对细长截面，取 1.0 作为常见保守值（可按需调整）
    added_mass_coefficient: float = 1.0

    # Water
    water_density: float = 1000.0
    drag_coefficient: float = 1.2
    # 切向阻力系数设置为小但非零值，避免完全忽略切向阻尼
    tangential_drag_coefficient: float = 0.01
    current_velocity_x: float = 0.0
    current_velocity_z: float = 0.0

    # Seabed/contact
    seabed_type: str = "flat"
    seabed_level: float = 0.0
    seabed_amplitude: float = 0.0
    seabed_wavelength: float = 1.0
    seabed_center: float = 0.0
    seabed_width: float = 1.0
    seabed_friction: float = 0.45
    seabed_stiffness: float = 8.0e5
    seabed_damping: float = 0.02
    friction_smoothing_velocity: float = 0.005
    # 接触容差/力容忍度不宜过于接近机器精度，设置为 1e-6 较为稳健
    contact_force_tolerance: float = 1.0e-6
    # 接触间隙容差与离散尺度相关，设为 1e-4 m（0.1 mm）
    contact_gap_tolerance: float = 1.0e-4
    # 渗透限经验法则：penetration_limit ≈ (1%~5%) × ds，且不超过缆径的 10%~20%
    penetration_limit: float = 0.001

    # Safety and classification
    break_force: float = 500.0
    safety_factor: float = 3.0
    # Explicit working limit. When omitted, break_force / safety_factor is used.
    allowable_tension: float | None = None
    # 表格给出最小弯曲直径 0.06 m → 半径 0.03 m
    min_bend_radius: float = 0.03
    slack_tension: float = 1.0e-4
    active_velocity_tolerance: float = 1.0e-3
    steady_fraction: float = 0.30

    # Numerical controls
    auto_substep: bool = True
    # 更保守的稳定因子（用于显式时间步估计等），保持在 0.1 左右
    stability_factor: float = 0.1
    # 限制内部子步数量为合理值以避免无限循环或过高开销
    max_internal_substeps: int = 2000
    # 最小段长不要过小以致数值问题，设为 1e-6（比原来的 1e-10 更稳健）
    minimum_segment_length: float = 1.0e-6
    # 曲率容差放宽到 1e-6，避免数值上不切实际的极高精度要求
    curvature_tolerance: float = 1.0e-6
    minimum_resolved_segment_ratio: float = 0.90
    remesh_enabled: bool = True
    remesh_trigger_ratio: float = 0.90
    diagnostic_zone_length: float = 0.05
    relaxation_time: float = 0.30
    relaxation_damping: float = 6.0
    fail_on_break: bool = False
    fail_on_penetration: bool = False

    # Output
    output_root: str = "outputs"
    snapshot_interval: float = 0.10
    save_history: bool = True
    save_plots: bool = True
    save_snapshots: bool = True
    # Engineering trend shown in the standard tension plots. Raw samples are
    # retained in CSV and in separate ``*_raw.png`` plots for peak/safety use.
    tension_filter_window: float = 0.50
    maximum_insertion_tension_jump: float = 1.0e-3
    maximum_reported_tension_step: float = 1.0e-4

    @property
    def area(self) -> float:
        return math.pi * self.diameter**2 / 4.0

    @property
    def displaced_mass_per_length(self) -> float:
        return self.water_density * self.area

    @property
    def effective_mass_per_length(self) -> float:
        return self.line_density - self.displaced_mass_per_length

    @property
    def effective_weight_per_length(self) -> float:
        return self.effective_mass_per_length * self.gravity

    @property
    def T_max(self) -> float:
        """活动缆最大允许张力（安全工作载荷）"""
        if self.allowable_tension is not None:
            return float(self.allowable_tension)
        return self.break_force / self.safety_factor

    @property
    def low_load_bend_radius(self) -> float:
        return 10.0 * self.diameter

    @property
    def current_velocity(self) -> tuple[float, float]:
        return (self.current_velocity_x, self.current_velocity_z)

    @property
    def initial_rest_length(self) -> float:
        return self.h_A + self.initial_extra_length

    @property
    def outlet_direction(self) -> np.ndarray:
        angle = float(np.deg2rad(self.outlet_angle_deg))
        return np.array([-np.cos(angle), -np.sin(angle)], dtype=float)

    @property
    def l0_min(self) -> float:
        # Keep the active payout segment away from a near-zero spring length.
        # Insertion at ds + l0_min therefore leaves at least half a standard
        # segment, which avoids the EA/l0 stiffness singularity.
        return max(1.0e-6, 0.5 * self.ds)

    @property
    def num_steps(self) -> int:
        return int(math.ceil(self.t_end / self.dt)) + 1

    @property
    def output_dir(self) -> Path:
        return Path(self.output_root) / self.case_id

    def with_overrides(self, **kwargs: Any) -> "SimulationConfig":
        return replace(self, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        positive = {
            "h_A": self.h_A,
            "t_end": self.t_end,
            "dt": self.dt,
            "diameter": self.diameter,
            "line_density": self.line_density,
            "EA": self.EA,
            "ds": self.ds,
            "water_density": self.water_density,
            "seabed_stiffness": self.seabed_stiffness,
            "friction_smoothing_velocity": self.friction_smoothing_velocity,
            "break_force": self.break_force,
            "min_bend_radius": self.min_bend_radius,
            "snapshot_interval": self.snapshot_interval,
            "tension_filter_window": self.tension_filter_window,
            "maximum_insertion_tension_jump": self.maximum_insertion_tension_jump,
            "maximum_reported_tension_step": self.maximum_reported_tension_step,
        }
        bad = [name for name, value in positive.items() if value <= 0.0]
        if bad:
            raise ValueError(f"Parameters must be positive: {', '.join(bad)}")
        if self.v_A < 0.0 or self.v_out < 0.0:
            raise ValueError("v_A and v_out must be non-negative")
        if self.initial_extra_length < 0.0:
            raise ValueError("initial_extra_length must be non-negative")
        if self.seabed_friction < 0.0:
            raise ValueError("seabed_friction must be non-negative")
        if not (0.0 < self.compression_stiffness_ratio <= 1.0):
            raise ValueError("compression_stiffness_ratio must be in (0, 1]")
        if self.payout_buffer_segments < 1:
            raise ValueError("payout_buffer_segments must be at least 1")
        if self.bending_damping_time < 0.0:
            raise ValueError("bending_damping_time must be non-negative")
        if not (0.0 < self.steady_fraction <= 1.0):
            raise ValueError("steady_fraction must be in (0, 1]")
        if self.max_internal_substeps < 1:
            raise ValueError("max_internal_substeps must be at least 1")
        if not (0.001 <= self.ds <= 0.05):
            raise ValueError("ds must be within 0.001--0.05 m")
        if not (0.0 <= self.outlet_angle_deg <= 180.0):
            raise ValueError("outlet_angle_deg must be within 0--180 degrees")
        if self.effective_mass_per_length < 0.0:
            warnings.warn(
                "Cable is positively buoyant with the current diameter and line density.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.allowable_tension is not None:
            if self.allowable_tension <= 0.0:
                raise ValueError("allowable_tension must be positive when provided")
            if self.allowable_tension > self.break_force:
                raise ValueError("allowable_tension must not exceed break_force")
        if not (0.0 < self.minimum_resolved_segment_ratio <= 1.0):
            raise ValueError("minimum_resolved_segment_ratio must be in (0, 1]")
        if not (0.0 < self.remesh_trigger_ratio <= 1.0):
            raise ValueError("remesh_trigger_ratio must be in (0, 1]")
        if self.diagnostic_zone_length <= 0.0:
            raise ValueError("diagnostic_zone_length must be positive")


_MODEL_SECTIONS = {
    "case",
    "cable",
    "water",
    "seabed",
    "safety",
    "numerical",
    "output",
}
_AUXILIARY_SECTIONS = {"metadata", "convergence", "experiments"}


def _flatten_sections(data: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten YAML model sections and reject silent typos/default fallbacks."""
    flattened: dict[str, Any] = {}
    valid = {f.name for f in fields(SimulationConfig)}
    for key, value in data.items():
        if key in valid:
            flattened[key] = value
        elif key in _MODEL_SECTIONS:
            if not isinstance(value, Mapping):
                raise ValueError(f"Configuration section '{key}' must be a mapping")
            unknown = sorted(set(value) - valid)
            if unknown:
                raise ValueError(
                    f"Unknown keys in configuration section '{key}': {', '.join(unknown)}"
                )
            for sub_key, sub_value in value.items():
                if sub_key in flattened:
                    raise ValueError(f"Duplicate configuration parameter: {sub_key}")
                flattened[sub_key] = sub_value
        elif key not in _AUXILIARY_SECTIONS:
            raise ValueError(f"Unknown top-level configuration key: {key}")
    return flattened


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> SimulationConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    kwargs = _flatten_sections(raw)
    if overrides:
        kwargs.update(overrides)
    cfg = SimulationConfig(**kwargs)
    cfg.validate()
    return cfg


def save_config(cfg: SimulationConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, allow_unicode=True, sort_keys=False)
