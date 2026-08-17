from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from .geometry import SegmentGeometry
from .state import CableState


@dataclass(frozen=True)
class AxialResult:
    strain: np.ndarray
    strain_rate: np.ndarray
    axial_force: np.ndarray
    tension: np.ndarray


def compute_tension(
    state: CableState, geom: SegmentGeometry, cfg: SimulationConfig
) -> AxialResult:
    l0 = np.maximum(state.rest_length, cfg.l0_min * 1.0e-6)
    strain = (geom.length - state.rest_length) / l0
    relative_velocity = np.diff(state.velocity, axis=0)
    strain_rate = np.einsum("ij,ij->i", relative_velocity, geom.tangent) / l0
    trial = cfg.EA * strain + cfg.axial_damping * strain_rate
    axial_force = np.where(trial >= 0.0, trial, cfg.compression_stiffness_ratio * trial)
    tension = np.maximum(axial_force, 0.0)
    return AxialResult(
        strain=strain,
        strain_rate=strain_rate,
        axial_force=axial_force,
        tension=tension,
    )


def assemble_tension_forces(axial: AxialResult, geom: SegmentGeometry) -> np.ndarray:
    n_nodes = geom.tangent.shape[0] + 1
    force = np.zeros((n_nodes, 2), dtype=float)
    segment_force = axial.axial_force[:, None] * geom.tangent
    force[:-1] += segment_force
    force[1:] -= segment_force
    return force
