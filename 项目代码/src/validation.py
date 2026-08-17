from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .geometry import SegmentGeometry
from .metrics import StepMetrics
from .state import CableState


def validate_state_finite(state: CableState) -> None:
    arrays = {
        "position": state.position,
        "velocity": state.velocity,
        "rest_length": state.rest_length,
    }
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise FloatingPointError(f"Non-finite values detected in {name}")
    if np.any(state.rest_length <= 0.0):
        raise FloatingPointError("Non-positive rest length detected")


def validate_geometry(geom: SegmentGeometry, cfg: SimulationConfig) -> None:
    if not np.all(np.isfinite(geom.length)):
        raise FloatingPointError("Non-finite segment length")
    if np.any(geom.length < cfg.minimum_segment_length):
        # A zero tail can occur only at insertion; proportional insertion should
        # normally avoid it, so persistent zero length is treated as an error.
        raise FloatingPointError("Zero or inverted geometric segment detected")


def check_safety(metrics: StepMetrics, cfg: SimulationConfig) -> None:
    # 超过设计允许张力（T_max），可选失败
    if cfg.fail_on_break and metrics.max_active_tension >= cfg.T_max:
        raise RuntimeError(
            f"Maximum allowable tension exceeded: "
            f"{metrics.max_active_tension:.6g} N >= T_max={cfg.T_max:.6g} N"
        )

    # 超过破断力（极限）
    if metrics.max_active_tension >= cfg.break_force:
        raise RuntimeError(
            f"Breaking-force limit exceeded: "
            f"{metrics.max_active_tension:.6g} N >= MBL={cfg.break_force:.6g} N"
        )

    # 海底穿透
    if cfg.fail_on_penetration and metrics.max_penetration > cfg.penetration_limit:
        raise RuntimeError(
            f"Penetration limit exceeded: {metrics.max_penetration:.6g} m"
        )


def assert_length_conservation(state: CableState, cfg: SimulationConfig, atol: float = 1e-10) -> None:
    expected = cfg.initial_rest_length + state.payout_integral
    if not np.isclose(state.total_rest_length, expected, atol=atol, rtol=1e-10):
        raise AssertionError(
            f"Rest-length conservation failed:\n"
            f"  actual   = {state.total_rest_length:.12f}\n"
            f"  expected = {expected:.12f}\n"
            f"  diff     = {state.total_rest_length - expected:.6e}\n"
            f"  atol     = {atol}"
        )
