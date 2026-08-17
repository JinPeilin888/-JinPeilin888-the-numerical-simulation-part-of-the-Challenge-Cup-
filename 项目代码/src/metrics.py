from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from .axial import AxialResult
from .config import SimulationConfig
from .contact import ContactResult
from .geometry import NodalProperties, SegmentGeometry, discrete_curvature_radius
from .state import CableState


@dataclass(frozen=True)
class StepMetrics:
    time: float
    auv_x: float
    auv_z: float
    payout_length: float
    expected_payout_length: float
    length_conservation_error: float
    min_standard_segment_ratio: float
    max_axial_strain: float
    min_axial_strain: float
    num_nodes: int
    tail_tension: float
    outlet_zone_max_tension: float
    max_tension_segment_index: int
    payout_reservoir_tension: float
    payout_reservoir_rest_length: float
    payout_reservoir_geometric_length: float
    payout_reservoir_strain: float
    max_tension: float
    max_active_tension: float
    mean_suspended_tension: float
    tdp_index: int
    tdp_segment_fraction: float
    tdp_x: float
    tdp_z: float
    layback: float
    suspended_length: float
    min_bend_radius: float
    min_bend_radius_raw: float
    min_bend_radius_node: int
    min_bend_radius_raw_node: int
    min_bend_radius_outlet: float
    min_bend_radius_tdp: float
    min_bend_radius_free: float
    max_penetration: float
    slack_ratio: float
    kinetic_energy: float
    outlet_direction_dot: float
    outlet_direction_error_deg: float
    tension_ratio: float  # max_active_tension / T_max
    tension_safe: bool
    bend_ratio: float  # min_bend_radius / minimum allowed bend radius
    bend_safe: bool
    mesh_safe: bool
    has_new_landing: bool
    new_landing_x: float | None

    def to_row(self) -> dict[str, object]:
        return asdict(self)


def _tdp_index(contact: np.ndarray) -> int:
    """Return the last material contact node, excluding the AUV guide point."""
    indices = np.flatnonzero(contact[:-1])
    return int(indices[-1]) if indices.size else 0


def _interpolate_tdp(
    state: CableState,
    geom: SegmentGeometry,
    contact: ContactResult,
    tdp_idx: int,
    cfg: SimulationConfig,
) -> tuple[np.ndarray, float]:
    """Interpolate the TDP between the last contact node and next free node.

    ``alpha`` is the fraction measured from node ``tdp_idx`` toward node
    ``tdp_idx + 1``. The returned z coordinate is projected onto a linearly
    interpolated seabed so that small penalty penetrations do not place the TDP
    visibly below the seabed.
    """
    n = state.num_nodes
    p0 = state.position[tdp_idx]
    bed0 = float(contact.seabed_height[tdp_idx])

    if tdp_idx >= n - 1 or tdp_idx >= geom.length.size:
        return np.array([float(p0[0]), bed0], dtype=float), 0.0

    p1 = state.position[tdp_idx + 1]
    bed1 = float(contact.seabed_height[tdp_idx + 1])
    gap0 = float(p0[1] - bed0)
    gap1 = float(p1[1] - bed1)

    alpha = 0.0
    denominator = gap1 - gap0

    # Interpolate only across a genuine contact-to-free transition. A contact
    # node may lie slightly above the mathematical seabed because contact uses
    # a finite gap tolerance; in that case alpha naturally remains zero.
    if (
        gap0 <= cfg.contact_gap_tolerance
        and gap1 > cfg.contact_gap_tolerance
        and abs(denominator) > cfg.minimum_segment_length
    ):
        alpha = float(np.clip(-gap0 / denominator, 0.0, 1.0))

    x_tdp = float(p0[0] + alpha * (p1[0] - p0[0]))
    z_tdp = float(bed0 + alpha * (bed1 - bed0))
    return np.array([x_tdp, z_tdp], dtype=float), alpha


def _active_segment_metrics(
    geom: SegmentGeometry,
    axial: AxialResult,
    tdp_idx: int,
    tdp_fraction: float,
    cfg: SimulationConfig,
) -> tuple[float, float, float, float]:
    """Return suspended length and length-weighted active tension metrics."""
    if tdp_idx >= geom.length.size:
        return 0.0, 0.0, 0.0, 0.0

    active_length = np.asarray(geom.length[tdp_idx:], dtype=float).copy()
    active_tension = np.asarray(axial.tension[tdp_idx:], dtype=float).copy()

    # The final edge is a kinematic payout reservoir between the controlled
    # guide node and the AUV outlet.  Both endpoints are prescribed, so its
    # penalty-spring force is not a physical free-cable tension.  Use the
    # adjacent physical edge as the outlet reaction for suspended-tension and
    # slack metrics while retaining the raw reservoir value separately.
    if cfg.enforce_outlet_direction and active_tension.size >= 2:
        active_tension[-1] = active_tension[-2]

    # Only the portion after the interpolated TDP is suspended.
    active_length[0] *= max(0.0, 1.0 - tdp_fraction)
    suspended_length = float(np.sum(active_length))

    resolved = active_length > cfg.minimum_segment_length
    if not np.any(resolved):
        return suspended_length, 0.0, 0.0, 0.0

    lengths = active_length[resolved]
    tensions = active_tension[resolved]
    total_length = float(np.sum(lengths))

    max_active = float(np.max(tensions))
    mean_suspended = float(np.sum(tensions * lengths) / total_length)
    slack_ratio = float(np.sum(lengths[tensions < cfg.slack_tension]) / total_length)
    return suspended_length, max_active, mean_suspended, slack_ratio


def _minimum_bend_radii(
    state: CableState,
    geom: SegmentGeometry,
    tdp_idx: int,
    cfg: SimulationConfig,
) -> tuple[float, float, int, int, float, float, float]:
    """Return resolved and raw minimum discrete bend radii.

    ``raw`` uses every finite three-node curvature result. ``resolved`` no longer
    hides every sub-20-mm segment. It rejects only samples adjacent to a
    geometrically collapsed element according to the configured mesh-quality
    ratio. Outlet, TDP and free-span values are reported separately.
    """
    radii = discrete_curvature_radius(state.position, cfg)
    finite = np.isfinite(radii)

    if np.any(finite):
        raw_candidates = np.flatnonzero(finite)
        raw_local = int(np.argmin(radii[raw_candidates]))
        raw_node = int(raw_candidates[raw_local])
        raw_min = float(radii[raw_node])
    else:
        raw_node = -1
        raw_min = math.inf

    resolved = finite.copy()
    if state.num_nodes >= 3:
        segment_ratio = geom.length / np.maximum(
            state.rest_length, cfg.minimum_segment_length
        )
        local_geometry_ok = (
            segment_ratio[:-1] >= cfg.minimum_resolved_segment_ratio
        ) & (
            segment_ratio[1:] >= cfg.minimum_resolved_segment_ratio
        )
        resolved[1:-1] &= local_geometry_ok

    if np.any(resolved):
        candidates = np.flatnonzero(resolved)
        local = int(np.argmin(radii[candidates]))
        node = int(candidates[local])
        minimum = float(radii[node])
    else:
        node = -1
        minimum = math.inf

    material_coordinate = np.concatenate(([0.0], np.cumsum(state.rest_length)))
    outlet_mask = resolved & (
        material_coordinate >= material_coordinate[-1] - cfg.diagnostic_zone_length
    )
    tdp_coordinate = material_coordinate[min(max(tdp_idx, 0), state.num_nodes - 1)]
    tdp_mask = resolved & (
        np.abs(material_coordinate - tdp_coordinate) <= cfg.diagnostic_zone_length
    )
    free_mask = resolved & (material_coordinate >= tdp_coordinate) & ~outlet_mask

    def masked_min(mask: np.ndarray) -> float:
        return float(np.min(radii[mask])) if np.any(mask) else math.inf

    return (
        minimum,
        raw_min,
        node,
        raw_node,
        masked_min(outlet_mask),
        masked_min(tdp_mask),
        masked_min(free_mask),
    )


def compute_metrics(
    state: CableState,
    geom: SegmentGeometry,
    axial: AxialResult,
    nodal: NodalProperties,
    contact: ContactResult,
    auv_position: np.ndarray,
    t: float,
    cfg: SimulationConfig,
) -> StepMetrics:
    tdp_idx = _tdp_index(contact.contact)
    tdp_position, tdp_fraction = _interpolate_tdp(
        state,
        geom,
        contact,
        tdp_idx,
        cfg,
    )
    layback = max(0.0, float(auv_position[0] - tdp_position[0]))

    (
        suspended_length,
        max_active,
        mean_suspended,
        slack_ratio,
    ) = _active_segment_metrics(
        geom,
        axial,
        tdp_idx,
        tdp_fraction,
        cfg,
    )

    (
        min_radius,
        min_radius_raw,
        min_radius_node,
        min_radius_raw_node,
        min_radius_outlet,
        min_radius_tdp,
        min_radius_free,
    ) = _minimum_bend_radii(state, geom, tdp_idx, cfg)
    T_limit = cfg.T_max
    tension_ratio = float(max_active / T_limit) if T_limit > 0 else math.inf
    tension_safe = max_active < cfg.T_max

    bend_ratio = (
        float(min_radius / cfg.min_bend_radius)
        if cfg.min_bend_radius > 0
        else math.inf
    )
    bend_safe = min_radius >= cfg.min_bend_radius

    kinetic = float(0.5 * np.sum(nodal.mass * np.sum(state.velocity ** 2, axis=1)))

    outlet_vector = state.position[-2] - state.position[-1]
    outlet_length = float(np.linalg.norm(outlet_vector))
    if not cfg.enforce_outlet_direction:
        outlet_dot = 1.0
    elif outlet_length > cfg.minimum_segment_length:
        outlet_unit = outlet_vector / outlet_length
        outlet_dot = float(
            np.clip(np.dot(outlet_unit, cfg.outlet_direction), -1.0, 1.0)
        )
    else:
        outlet_dot = 1.0
    outlet_error_deg = (
        0.0
        if 1.0 - outlet_dot <= 1.0e-12
        else float(np.degrees(np.arccos(outlet_dot)))
    )


    new_contact = contact.contact & ~state.contact_prev
    candidate = np.flatnonzero(new_contact)
    candidate = candidate[(candidate > 0) & (candidate < state.num_nodes - 1)]
    has_new_landing = candidate.size > 0
    landing_x = float(state.position[candidate[0], 0]) if has_new_landing else None

    payout_length = state.total_rest_length
    expected = cfg.initial_rest_length + state.payout_integral
    conservation_error = payout_length - expected
    segment_ratio = geom.length / np.maximum(
        state.rest_length, cfg.minimum_segment_length
    )
    standard_ratio = segment_ratio[:-1] if segment_ratio.size > 1 else segment_ratio
    min_standard_segment_ratio = (
        float(np.min(standard_ratio)) if standard_ratio.size else 1.0
    )
    mesh_safe = min_standard_segment_ratio >= cfg.minimum_resolved_segment_ratio
    physical_strain = (
        axial.strain[:-1]
        if cfg.enforce_outlet_direction and axial.strain.size >= 2
        else axial.strain
    )
    physical_tension = (
        axial.tension[:-1]
        if cfg.enforce_outlet_direction and axial.tension.size >= 2
        else axial.tension
    )
    max_axial_strain = (
        max(0.0, float(np.max(physical_strain))) if physical_strain.size else 0.0
    )
    min_axial_strain = (
        min(0.0, float(np.min(physical_strain))) if physical_strain.size else 0.0
    )
    max_tension = float(np.max(physical_tension)) if physical_tension.size else 0.0
    max_tension_segment_index = (
        int(np.argmax(physical_tension)) if physical_tension.size else -1
    )
    tail_tension = float(physical_tension[-1]) if physical_tension.size else 0.0
    outlet_zone_max_tension = (
        float(np.max(physical_tension[-min(5, physical_tension.size) :]))
        if physical_tension.size
        else 0.0
    )
    reservoir_tension = float(axial.tension[-1]) if axial.tension.size else 0.0
    reservoir_rest_length = (
        float(state.rest_length[-1]) if state.rest_length.size else 0.0
    )
    reservoir_geometric_length = (
        float(geom.length[-1]) if geom.length.size else 0.0
    )
    reservoir_strain = float(axial.strain[-1]) if axial.strain.size else 0.0
    max_penetration = (
        float(np.max(contact.penetration)) if contact.penetration.size else 0.0
    )

    return StepMetrics(
        time=float(t),
        auv_x=float(auv_position[0]),
        auv_z=float(auv_position[1]),
        payout_length=payout_length,
        expected_payout_length=expected,
        length_conservation_error=float(conservation_error),
        min_standard_segment_ratio=min_standard_segment_ratio,
        max_axial_strain=max_axial_strain,
        min_axial_strain=min_axial_strain,
        num_nodes=state.num_nodes,
        tail_tension=tail_tension,
        outlet_zone_max_tension=outlet_zone_max_tension,
        max_tension_segment_index=max_tension_segment_index,
        payout_reservoir_tension=reservoir_tension,
        payout_reservoir_rest_length=reservoir_rest_length,
        payout_reservoir_geometric_length=reservoir_geometric_length,
        payout_reservoir_strain=reservoir_strain,
        max_tension=max_tension,
        max_active_tension=max_active,
        mean_suspended_tension=mean_suspended,
        tdp_index=tdp_idx,
        tdp_segment_fraction=tdp_fraction,
        tdp_x=float(tdp_position[0]),
        tdp_z=float(tdp_position[1]),
        layback=layback,
        suspended_length=suspended_length,
        min_bend_radius=min_radius,
        min_bend_radius_raw=min_radius_raw,
        min_bend_radius_node=min_radius_node,
        min_bend_radius_raw_node=min_radius_raw_node,
        min_bend_radius_outlet=min_radius_outlet,
        min_bend_radius_tdp=min_radius_tdp,
        min_bend_radius_free=min_radius_free,
        max_penetration=max_penetration,
        slack_ratio=slack_ratio,
        kinetic_energy=kinetic,
        outlet_direction_dot=outlet_dot,
        outlet_direction_error_deg=outlet_error_deg,
        tension_ratio=tension_ratio,
        tension_safe=tension_safe,
        bend_ratio=bend_ratio,
        bend_safe=bend_safe,
        mesh_safe=mesh_safe,
        has_new_landing=has_new_landing,
        new_landing_x=landing_x,
    )
