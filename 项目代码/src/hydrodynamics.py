from __future__ import annotations

import numpy as np

from .config import SimulationConfig
from .geometry import NodalProperties
from .state import CableState


def gravity_buoyancy(nodal: NodalProperties, cfg: SimulationConfig) -> np.ndarray:
    force = np.zeros((nodal.representative_length.size, 2), dtype=float)
    force[:, 1] = -cfg.effective_weight_per_length * nodal.representative_length
    return force


def _sanitize_and_clip(x: np.ndarray, clip: float, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        bad = np.where(~np.isfinite(x))[0][:5]
        raise ValueError(f"{name} contains non-finite values, sample idx={bad.tolist()}")
    return np.clip(x, -clip, clip)


def hydrodynamic_drag(
    state: CableState, nodal: NodalProperties, cfg: SimulationConfig
) -> np.ndarray:
    water_velocity = np.asarray(cfg.current_velocity, dtype=np.float64)
    state_velocity = np.asarray(state.velocity, dtype=np.float64)

    if not np.isfinite(water_velocity).all():
        raise ValueError("cfg.current_velocity contains non-finite values")
    if not np.isfinite(state_velocity).all():
        bad = np.argwhere(~np.isfinite(state_velocity))[:5]
        raise ValueError(f"state.velocity contains non-finite values, sample idx={bad.tolist()}")

    relative = water_velocity[None, :] - state_velocity

    # 经验阈值，先保守防炸；后续可放到 cfg
    speed_clip = 1e3  # m/s，按你的物理场景再调小
    normal_speed = np.einsum("ij,ij->i", relative, nodal.normal)
    normal_speed = _sanitize_and_clip(normal_speed, speed_clip, "normal_speed")

    normal_scalar = (
        0.5
        * cfg.water_density
        * cfg.drag_coefficient
        * cfg.diameter
        * nodal.representative_length
        * np.abs(normal_speed)
        * normal_speed
    )
    force = normal_scalar[:, None] * nodal.normal

    if cfg.tangential_drag_coefficient > 0.0:
        tangential_speed = np.einsum("ij,ij->i", relative, nodal.tangent)
        tangential_speed = _sanitize_and_clip(
            tangential_speed, speed_clip, "tangential_speed"
        )

        tangential_scalar = (
            0.5
            * cfg.water_density
            * cfg.tangential_drag_coefficient
            * np.pi
            * cfg.diameter
            * nodal.representative_length
            * np.abs(tangential_speed)
            * tangential_speed
        )
        force += tangential_scalar[:, None] * nodal.tangent

    if not np.isfinite(force).all():
        bad = np.argwhere(~np.isfinite(force))[:5]
        raise ValueError(f"hydrodynamic drag force contains non-finite values, sample idx={bad.tolist()}")

    return force