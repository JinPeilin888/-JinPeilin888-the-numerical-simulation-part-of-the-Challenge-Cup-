from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from .seabed import Seabed
from .state import CableState


@dataclass(frozen=True)
class RemeshResult:
    applied: bool
    minimum_ratio_before: float
    minimum_ratio_after: float
    geometric_length_before: float
    geometric_length_after: float


def smoothstep(xi: float) -> float:
    if xi <= 0.0:
        return 0.0
    if xi >= 1.0:
        return 1.0
    return 3.0 * xi**2 - 2.0 * xi**3


def smoothstep_integral(xi: float) -> float:
    """Integral of smoothstep from 0 to xi for 0<=xi<=1."""
    xi = min(max(xi, 0.0), 1.0)
    return xi**3 - 0.5 * xi**4


def ramped_speed(target: float, t: float, ramp_time: float) -> float:
    if ramp_time <= 0.0:
        return target
    return target * smoothstep(t / ramp_time)


def ramped_displacement(target_speed: float, t: float, ramp_time: float) -> float:
    if t <= 0.0:
        return 0.0
    if ramp_time <= 0.0:
        return target_speed * t
    if t <= ramp_time:
        return target_speed * ramp_time * smoothstep_integral(t / ramp_time)
    # Integral over the ramp equals half the ramp duration.
    return target_speed * (t - 0.5 * ramp_time)


def auv_trajectory(t: float, cfg: SimulationConfig, seabed: Seabed) -> tuple[np.ndarray, np.ndarray]:
    x = ramped_displacement(cfg.v_A, t, cfg.ramp_time)
    vx = ramped_speed(cfg.v_A, t, cfg.ramp_time)
    z_bottom = float(seabed.height(x))
    slope = float(seabed.slope(x))
    position = np.array([x, z_bottom + cfg.h_A], dtype=float)
    velocity = np.array([vx, slope * vx], dtype=float)
    return position, velocity


def payout_speed(t: float, cfg: SimulationConfig) -> float:
    return ramped_speed(cfg.v_out, t, cfg.ramp_time)


def _payout_buffer_bounds(
    state: CableState, cfg: SimulationConfig
) -> tuple[int, int]:
    """Return ``[start, stop)`` for physical edges in the outlet ALE buffer."""
    physical_stop = max(0, state.rest_length.size - 1)
    count = min(cfg.payout_buffer_segments, physical_stop)
    return physical_stop - count, physical_stop


def payout_buffer_rest_lengths(
    state: CableState, cfg: SimulationConfig
) -> np.ndarray:
    start, stop = _payout_buffer_bounds(state, cfg)
    return state.rest_length[start:stop]


def grow_tail_rest_length(state: CableState, t: float, dt: float, cfg: SimulationConfig) -> None:
    """Inject material smoothly across a short ALE buffer before the guide.

    The final edge is a fixed Eulerian guide segment. New rest length is shared
    by the last ``payout_buffer_segments`` physical edges, preventing one edge
    from cycling between 0.5 and 1.5 times ``ds``.
    """
    increment = payout_speed(t + 0.5 * dt, cfg) * dt
    start, stop = _payout_buffer_bounds(state, cfg)
    if stop <= start:
        state.rest_length[-1] += increment
    else:
        state.rest_length[start:stop] += increment / (stop - start)
    state.payout_integral += increment


def insert_nodes_if_needed(
    state: CableState,
    auv_position: np.ndarray,
    auv_velocity: np.ndarray,
    cfg: SimulationConfig,
) -> int:
    """Insert nodes by splitting the longest edge in the outlet ALE buffer.

    Existing nodes are never moved. The new node is interpolated at the same
    fraction in position, velocity and rest length, so geometry, strain,
    velocity and bending angles are continuous across insertion. The fixed
    guide node and guide segment remain unchanged.
    """
    inserted = 0

    if state.rest_length.size < 2 or state.num_nodes < 3:
        return inserted

    while True:
        start, stop = _payout_buffer_bounds(state, cfg)
        count = stop - start
        if count <= 0:
            break
        buffer_rest = state.rest_length[start:stop].copy()
        total = float(np.sum(buffer_rest))
        if total < (count + 1) * cfg.ds - 1.0e-12:
            break

        candidates = np.flatnonzero(buffer_rest >= cfg.ds - 1.0e-12)
        if not candidates.size:
            candidates = np.array([int(np.argmax(buffer_rest))])
        turning_cost: list[tuple[float, float, int]] = []
        for local in candidates:
            edge_index = start + int(local)
            cost = 0.0
            for node in (edge_index, edge_index + 1):
                if node <= 0 or node >= state.num_nodes - 1:
                    continue
                left = state.position[node] - state.position[node - 1]
                right = state.position[node + 1] - state.position[node]
                cross = float(left[0] * right[1] - left[1] * right[0])
                dot = float(np.dot(left, right))
                cost += abs(float(np.arctan2(cross, dot)))
            turning_cost.append(
                (cost, -float(state.rest_length[edge_index]), edge_index)
            )
        edge = min(turning_cost)[2]
        fraction = 0.5
        new_position = (
            state.position[edge]
            + fraction * (state.position[edge + 1] - state.position[edge])
        )
        new_velocity = (
            state.velocity[edge]
            + fraction * (state.velocity[edge + 1] - state.velocity[edge])
        )
        new_contact = bool(
            state.contact_prev[edge] and state.contact_prev[edge + 1]
        )
        old_rest = float(state.rest_length[edge])
        state.position = np.insert(state.position, edge + 1, new_position, axis=0)
        state.velocity = np.insert(state.velocity, edge + 1, new_velocity, axis=0)
        state.contact_prev = np.insert(
            state.contact_prev, edge + 1, new_contact
        )
        state.rest_length[edge] = fraction * old_rest
        state.rest_length = np.insert(
            state.rest_length, edge + 1, (1.0 - fraction) * old_rest
        )
        inserted += 1

    return inserted


def redistribute_nodes_if_needed(
    state: CableState,
    cfg: SimulationConfig,
) -> RemeshResult:
    """Redistribute material nodes along the existing polyline when one collapses.

    The operation keeps node count, segment rest lengths, total material length,
    endpoints and the geometric polyline path. Positions and velocities are
    interpolated by cumulative arc length. This is a mesh-quality operation, not
    an added physical force; the outlet boundary is re-applied by the caller.
    """
    vector = np.diff(state.position, axis=0)
    geometric_length = np.linalg.norm(vector, axis=1)
    ratios = geometric_length / np.maximum(
        state.rest_length, cfg.minimum_segment_length
    )
    standard = ratios[:-1] if ratios.size > 1 else ratios
    minimum_before = float(np.min(standard)) if standard.size else 1.0
    total_before = float(np.sum(geometric_length))

    if (
        not cfg.remesh_enabled
        or state.num_nodes < 4
        or minimum_before >= cfg.remesh_trigger_ratio
        or total_before <= cfg.minimum_segment_length
    ):
        return RemeshResult(
            False,
            minimum_before,
            minimum_before,
            total_before,
            total_before,
        )

    old_arc = np.concatenate(([0.0], np.cumsum(geometric_length)))
    material = np.concatenate(([0.0], np.cumsum(state.rest_length)))
    target_arc = total_before * material / material[-1]

    position = np.column_stack(
        [np.interp(target_arc, old_arc, state.position[:, axis]) for axis in range(2)]
    )
    velocity = np.column_stack(
        [np.interp(target_arc, old_arc, state.velocity[:, axis]) for axis in range(2)]
    )
    nearest = np.searchsorted(old_arc, target_arc, side="left")
    nearest = np.clip(nearest, 0, state.num_nodes - 1)
    left = np.maximum(nearest - 1, 0)
    choose_left = np.abs(target_arc - old_arc[left]) <= np.abs(
        target_arc - old_arc[nearest]
    )
    nearest = np.where(choose_left, left, nearest)

    position[0] = state.position[0]
    position[-1] = state.position[-1]
    velocity[0] = state.velocity[0]
    velocity[-1] = state.velocity[-1]
    state.position = position
    state.velocity = velocity
    state.contact_prev = state.contact_prev[nearest]

    after_length = np.linalg.norm(np.diff(state.position, axis=0), axis=1)
    after_ratios = after_length / np.maximum(
        state.rest_length, cfg.minimum_segment_length
    )
    standard_after = after_ratios[:-1] if after_ratios.size > 1 else after_ratios
    minimum_after = float(np.min(standard_after)) if standard_after.size else 1.0
    return RemeshResult(
        True,
        minimum_before,
        minimum_after,
        total_before,
        float(np.sum(after_length)),
    )
