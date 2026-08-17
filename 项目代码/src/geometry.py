from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from .state import CableState


@dataclass(frozen=True)
class SegmentGeometry:
    vector: np.ndarray
    length: np.ndarray
    tangent: np.ndarray


@dataclass(frozen=True)
class NodalProperties:
    representative_length: np.ndarray
    mass: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray


def compute_segment_geometry(state: CableState, cfg: SimulationConfig) -> SegmentGeometry:
    vector = np.diff(state.position, axis=0)
    length = np.linalg.norm(vector, axis=1)
    safe = np.maximum(length, cfg.minimum_segment_length)
    tangent = vector / safe[:, None]
    zero = length <= cfg.minimum_segment_length
    if np.any(zero):
        # A newly inserted material node can initially coincide with the AUV
        # guide point. Reuse the preceding direction instead of producing NaN.
        for idx in np.flatnonzero(zero):
            if idx > 0:
                tangent[idx] = tangent[idx - 1]
            else:
                tangent[idx] = np.array([0.0, 1.0])
    return SegmentGeometry(vector=vector, length=length, tangent=tangent)


def distribute_mass_and_length(
    state: CableState, geom: SegmentGeometry, cfg: SimulationConfig
) -> NodalProperties:
    n = state.num_nodes
    representative = np.zeros(n, dtype=float)
    representative[0] = 0.5 * state.rest_length[0]
    representative[-1] = 0.5 * state.rest_length[-1]
    if n > 2:
        representative[1:-1] = 0.5 * (state.rest_length[:-1] + state.rest_length[1:])

    structural_mass = cfg.line_density * representative
    added_mass = cfg.added_mass_coefficient * cfg.water_density * cfg.area * representative
    mass = structural_mass + added_mass

    nodal_tangent = np.zeros((n, 2), dtype=float)
    nodal_tangent[0] = geom.tangent[0]
    nodal_tangent[-1] = geom.tangent[-1]
    if n > 2:
        raw = geom.tangent[:-1] + geom.tangent[1:]
        norm = np.linalg.norm(raw, axis=1)
        fallback = geom.tangent[:-1]
        nodal_tangent[1:-1] = np.where(
            (norm > cfg.minimum_segment_length)[:, None],
            raw / np.maximum(norm, cfg.minimum_segment_length)[:, None],
            fallback,
        )
    nodal_normal = np.column_stack((-nodal_tangent[:, 1], nodal_tangent[:, 0]))
    return NodalProperties(representative, mass, nodal_tangent, nodal_normal)


def discrete_curvature_radius(position: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    n = position.shape[0]
    radius = np.full(n, np.inf, dtype=float)
    if n < 3:
        return radius

    if not np.all(np.isfinite(position)):
        return radius#防止本身有 inf/nan
    a = position[1:-1] - position[:-2]
    b = position[2:] - position[1:-1]
    c = position[2:] - position[:-2]
    #防止叉乘溢出
    max_coord = 1.0e6
    a = np.clip(a, -max_coord, max_coord)
    b = np.clip(b, -max_coord, max_coord)
    c = np.clip(c, -max_coord, max_coord)
    cross = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
    valid = denom > 1.0e-12
    curvature = np.zeros_like(cross)
    curvature[valid] = 2.0 * cross[valid] / denom[valid]
    curved = curvature > cfg.curvature_tolerance
    local_radius = np.full_like(curvature, np.inf)
    local_radius[curved] = 1.0 / curvature[curved]
    radius[1:-1] = local_radius
    return radius


def bending_forces(
    position: np.ndarray,
    rest_length: np.ndarray,
    EI: float,
    min_length: float = 1.0e-8,
) -> np.ndarray:
    """Return the negative gradient of a nonlinear turning-angle energy.

    ``E_i = 0.5 * EI * theta_i**2 / h_i`` remains meaningful when a slack
    segment becomes much shorter than its rest length. Its angle gradient
    grows as ``1 / geometric_length``, preventing mesh nodes from collapsing
    without introducing an axial compression law.
    """
    n = position.shape[0]
    f = np.zeros_like(position)

    if n < 3 or EI <= 0.0:
        return f

    edge_left = position[1:-1] - position[:-2]
    edge_right = position[2:] - position[1:-1]
    length_left_sq = np.maximum(
        np.einsum("ij,ij->i", edge_left, edge_left), min_length**2
    )
    length_right_sq = np.maximum(
        np.einsum("ij,ij->i", edge_right, edge_right), min_length**2
    )
    cross = edge_left[:, 0] * edge_right[:, 1] - edge_left[:, 1] * edge_right[:, 0]
    dot = np.einsum("ij,ij->i", edge_left, edge_right)
    angle = np.arctan2(cross, dot)

    rotate_left = np.column_stack((-edge_left[:, 1], edge_left[:, 0]))
    rotate_right = np.column_stack((-edge_right[:, 1], edge_right[:, 0]))
    grad_previous = rotate_left / length_left_sq[:, None]
    grad_next = rotate_right / length_right_sq[:, None]
    grad_center = -grad_previous - grad_next
    local_scale = np.maximum(
        0.5 * (rest_length[:-1] + rest_length[1:]), min_length
    )
    coefficient = EI * angle / local_scale

    f[:-2] -= coefficient[:, None] * grad_previous
    f[1:-1] -= coefficient[:, None] * grad_center
    f[2:] -= coefficient[:, None] * grad_next

    return f


def bending_damping_forces(
    position: np.ndarray,
    velocity: np.ndarray,
    rest_length: np.ndarray,
    coefficient: float,
    min_length: float = 1.0e-8,
) -> np.ndarray:
    """Return dissipative forces from discrete angular velocity.

    The Rayleigh dissipation is
    ``D = sum(0.5 * coefficient * theta_dot**2 / h)``. Therefore the
    mechanical power of the returned force is exactly ``-2D <= 0``.
    """
    n = position.shape[0]
    force = np.zeros_like(position)
    if n < 3 or coefficient <= 0.0:
        return force
    edge_left = position[1:-1] - position[:-2]
    edge_right = position[2:] - position[1:-1]
    length_left_sq = np.maximum(
        np.einsum("ij,ij->i", edge_left, edge_left), min_length**2
    )
    length_right_sq = np.maximum(
        np.einsum("ij,ij->i", edge_right, edge_right), min_length**2
    )
    rotate_left = np.column_stack((-edge_left[:, 1], edge_left[:, 0]))
    rotate_right = np.column_stack((-edge_right[:, 1], edge_right[:, 0]))
    grad_previous = rotate_left / length_left_sq[:, None]
    grad_next = rotate_right / length_right_sq[:, None]
    grad_center = -grad_previous - grad_next
    angle_rate = (
        np.einsum("ij,ij->i", grad_previous, velocity[:-2])
        + np.einsum("ij,ij->i", grad_center, velocity[1:-1])
        + np.einsum("ij,ij->i", grad_next, velocity[2:])
    )
    local_scale = np.maximum(
        0.5 * (rest_length[:-1] + rest_length[1:]), min_length
    )
    multiplier = coefficient * angle_rate / local_scale
    force[:-2] -= multiplier[:, None] * grad_previous
    force[1:-1] -= multiplier[:, None] * grad_center
    force[2:] -= multiplier[:, None] * grad_next
    return force


def bending_energy(
    position: np.ndarray,
    rest_length: np.ndarray,
    EI: float,
    min_length: float = 1.0e-8,
) -> float:
    """Return the nonlinear turning-angle energy used by ``bending_forces``."""
    if position.shape[0] < 3 or EI <= 0.0:
        return 0.0
    edge_left = position[1:-1] - position[:-2]
    edge_right = position[2:] - position[1:-1]
    cross = edge_left[:, 0] * edge_right[:, 1] - edge_left[:, 1] * edge_right[:, 0]
    dot = np.einsum("ij,ij->i", edge_left, edge_right)
    angle = np.arctan2(cross, dot)
    local_scale = np.maximum(
        0.5 * (rest_length[:-1] + rest_length[1:]), min_length
    )
    return float(np.sum(0.5 * EI * angle**2 / local_scale))
