from __future__ import annotations

from dataclasses import dataclass
import traceback
import numpy as np

from .axial import AxialResult, assemble_tension_forces, compute_tension
from .config import D_OUT, SimulationConfig
from .contact import ContactResult, seabed_contact_and_friction
from .deployment import (
    auv_trajectory,
    grow_tail_rest_length,
    insert_nodes_if_needed,
    payout_buffer_rest_lengths,
    payout_speed,
    redistribute_nodes_if_needed,
)
from .geometry import NodalProperties, SegmentGeometry, compute_segment_geometry, distribute_mass_and_length
from .hydrodynamics import gravity_buoyancy, hydrodynamic_drag
from .integrators import linearly_implicit_euler
from .metrics import StepMetrics, compute_metrics
from .recorder import Recorder
from .seabed import Seabed, anchor_position
from .state import CableState, initialize_vertical_cable
from .validation import assert_length_conservation, check_safety, validate_geometry, validate_state_finite
from .geometry import bending_damping_forces, bending_forces

@dataclass(frozen=True)
class Evaluation:
    geometry: SegmentGeometry
    axial: AxialResult
    nodal: NodalProperties
    contact: ContactResult
    total_force: np.ndarray


def _physical_peak_tension(axial: AxialResult, cfg: SimulationConfig) -> float:
    """Peak tension excluding the controlled payout-reservoir edge."""
    values = (
        axial.tension[:-1]
        if cfg.enforce_outlet_direction and axial.tension.size >= 2
        else axial.tension
    )
    return float(np.max(values)) if values.size else 0.0


def evaluate_forces(state: CableState, seabed: Seabed, cfg: SimulationConfig) -> Evaluation:
    geom = compute_segment_geometry(state, cfg)
    validate_geometry(geom, cfg)
    axial = compute_tension(state, geom, cfg)
    nodal = distribute_mass_and_length(state, geom, cfg)
    f_total = assemble_tension_forces(axial, geom)
    f_total += gravity_buoyancy(nodal, cfg)
    f_total += hydrodynamic_drag(state, nodal, cfg)
    contact = seabed_contact_and_friction(state, seabed, cfg)
    f_total += contact.force
    f_total += bending_forces(
        state.position,
        state.rest_length,
        cfg.EI,
    )
    f_total += bending_damping_forces(
        state.position,
        state.velocity,
        state.rest_length,
        cfg.EI * cfg.bending_damping_time,
    )
    return Evaluation(geom, axial, nodal, contact, f_total)


def apply_boundaries(
    state: CableState,
    anchor: np.ndarray,
    auv_position: np.ndarray,
    auv_velocity: np.ndarray,
    outlet_direction: np.ndarray | None = D_OUT,
) -> None:
    """
    Apply kinematic boundary conditions at both ends of the cable.

    - Anchor (node 0): fixed in position and velocity.
    - AUV outlet (last node): position and velocity prescribed by trajectory.
    - Outlet direction is fixed at 37.5° via D_OUT.
    - The final segment is a fixed-length Eulerian guide segment on the outlet
      ray. Material payout occurs in the adjacent free-cable segment, so the
      guide node never changes identity during insertion.
    """
    # ---- Anchor (seabed fixed end) ----
    state.position[0] = anchor
    state.velocity[0] = 0.0

    # ---- AUV outlet (cable termination) ----
    state.position[-1] = auv_position
    state.velocity[-1] = auv_velocity

    # ---- Need at least two nodes to define a segment ----
    if state.num_nodes < 2:
        return

    if outlet_direction is None:
        return

    # ---- Fixed spatial guide: exact direction and neutral guide strain ----
    L = max(float(state.rest_length[-1]), 1.0e-12)
    direction = np.asarray(outlet_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    state.position[-2] = state.position[-1] + L * direction

    state.velocity[-2] = state.velocity[-1]


def relax_initial_state(state: CableState, seabed: Seabed, cfg: SimulationConfig) -> CableState:
    if cfg.relaxation_time <= 0.0:
        return state
    anchor = anchor_position(seabed)
    fixed_auv = np.array([0.0, float(seabed.height(0.0)) + cfg.h_A])
    zero = np.zeros(2, dtype=float)
    outlet_direction = (
        cfg.outlet_direction if cfg.enforce_outlet_direction else None
    )
    apply_boundaries(state, anchor, fixed_auv, zero, outlet_direction)
    dt = min(float(cfg.dt), 0.002, float(cfg.relaxation_time))
    steps = int(np.ceil(cfg.relaxation_time / dt))
    dt = cfg.relaxation_time / steps
    for _ in range(steps):
        evaluation = evaluate_forces(state, seabed, cfg)
        linearly_implicit_euler(
            state,
            evaluation.total_force,
            evaluation.nodal.mass,
            dt,
            cfg,
            evaluation.geometry,
            evaluation.axial,
            evaluation.nodal,
            evaluation.contact,
            zero,
            velocity_damping=cfg.relaxation_damping,
        )
        apply_boundaries(state, anchor, fixed_auv, zero, outlet_direction)
        validate_state_finite(state)
    # Start the deployment from the relaxed geometry without residual velocity.
    state.velocity[:] = 0.0
    contact = seabed_contact_and_friction(state, seabed, cfg)
    state.contact_prev = contact.contact.copy()
    state.payout_integral = 0.0
    return state


def _record_current(
    state: CableState,
    t: float,
    cfg: SimulationConfig,
    seabed: Seabed,
    recorder: Recorder,
) -> StepMetrics:
    auv_pos, auv_vel = auv_trajectory(t, cfg, seabed)
    outlet_direction = (
        cfg.outlet_direction if cfg.enforce_outlet_direction else None
    )
    apply_boundaries(
        state, anchor_position(seabed), auv_pos, auv_vel, outlet_direction
    )
    evaluation = evaluate_forces(state, seabed, cfg)
    metrics = compute_metrics(
        state,
        evaluation.geometry,
        evaluation.axial,
        evaluation.nodal,
        evaluation.contact,
        auv_pos,
        t,
        cfg,
    )
    recorder.append(metrics, state)
    state.contact_prev = evaluation.contact.contact.copy()
    check_safety(metrics, cfg)
    return metrics


def run_simulation(cfg: SimulationConfig, raise_on_error: bool = True) -> dict[str, object]:
    cfg.validate()
    seabed = Seabed(cfg)
    anchor = anchor_position(seabed)
    state = initialize_vertical_cable(cfg, seabed_height=float(anchor[1]))
    recorder = Recorder(cfg)

    try:
        state = relax_initial_state(state, seabed, cfg)
        current_time = 0.0
        _record_current(state, current_time, cfg, seabed, recorder)

        outer_steps = int(np.ceil(cfg.t_end / cfg.dt))
        for outer_index in range(outer_steps):
            current_time = outer_index * cfg.dt
            outer_end = min((outer_index + 1) * cfg.dt, cfg.t_end)
            outer_dt = outer_end - current_time
            # The stiff elastic/contact response is handled implicitly. Keep
            # only an advection-based accuracy limit for boundary motion and
            # material payout; it is normally looser than the configured dt.
            boundary_speed = max(cfg.v_A, cfg.v_out, 1.0e-12)
            accuracy_dt = 0.1 * cfg.ds / boundary_speed
            substeps = (
                max(1, int(np.ceil(outer_dt / accuracy_dt)))
                if cfg.auto_substep
                else 1
            )
            inner_dt = outer_dt / substeps
            for sub in range(substeps):
                t0 = current_time + sub * inner_dt
                auv0, auv_vel0 = auv_trajectory(t0, cfg, seabed)
                outlet_direction = (
                    cfg.outlet_direction if cfg.enforce_outlet_direction else None
                )
                apply_boundaries(
                    state, anchor, auv0, auv_vel0, outlet_direction
                )

                grow_tail_rest_length(state, t0, inner_dt, cfg)
                buffer_before = payout_buffer_rest_lengths(state, cfg)
                insertion_due = (
                    buffer_before.size > 0
                    and float(np.sum(buffer_before))
                    >= (buffer_before.size + 1) * cfg.ds - 1.0e-12
                )
                if insertion_due:
                    before_geometry = compute_segment_geometry(state, cfg)
                    before_axial = compute_tension(state, before_geometry, cfg)
                    before_peak = _physical_peak_tension(before_axial, cfg)
                    before_nodes = state.num_nodes
                    before_rest = state.total_rest_length
                    before_tail = float(buffer_before[-1])
                    before_buffer_total = float(np.sum(buffer_before))
                inserted = insert_nodes_if_needed(state, auv0, auv_vel0, cfg)
                # Insertion changes array sizes; boundaries remain the first/last nodes.
                apply_boundaries(
                    state, anchor, auv0, auv_vel0, outlet_direction
                )
                if inserted:
                    after_geometry = compute_segment_geometry(state, cfg)
                    after_axial = compute_tension(state, after_geometry, cfg)
                    after_peak = _physical_peak_tension(after_axial, cfg)
                    buffer_after = payout_buffer_rest_lengths(state, cfg)
                    after_ratio = after_geometry.length / np.maximum(
                        state.rest_length, cfg.minimum_segment_length
                    )
                    recorder.record_insertion_event(
                        time=float(t0),
                        inserted_nodes=int(inserted),
                        node_count_before=int(before_nodes),
                        node_count_after=int(state.num_nodes),
                        rest_length_before=float(before_rest),
                        rest_length_after=float(state.total_rest_length),
                        rest_length_error=float(state.total_rest_length - before_rest),
                        tail_rest_length_before=float(before_tail),
                        tail_rest_length_after=float(
                            payout_buffer_rest_lengths(state, cfg)[-1]
                        ),
                        payout_buffer_rest_length_before=before_buffer_total,
                        payout_buffer_rest_length_after=float(
                            np.sum(buffer_after)
                        ),
                        payout_buffer_min_rest_length_after=float(
                            np.min(buffer_after)
                        ),
                        payout_buffer_max_rest_length_after=float(
                            np.max(buffer_after)
                        ),
                        minimum_physical_rest_length_after=float(
                            np.min(state.rest_length[:-1])
                        ),
                        peak_tension_before=float(before_peak),
                        peak_tension_after=float(after_peak),
                        peak_tension_jump=float(after_peak - before_peak),
                        minimum_geometric_ratio_after=float(np.min(after_ratio)),
                    )

                evaluation = evaluate_forces(state, seabed, cfg)
                midpoint = t0 + 0.5 * inner_dt
                _, auv_vel_mid = auv_trajectory(midpoint, cfg, seabed)
                linearly_implicit_euler(
                    state,
                    evaluation.total_force,
                    evaluation.nodal.mass,
                    inner_dt,
                    cfg,
                    evaluation.geometry,
                    evaluation.axial,
                    evaluation.nodal,
                    evaluation.contact,
                    auv_vel_mid,
                )
                end_time = t0 + inner_dt
                auv1, auv_vel1 = auv_trajectory(end_time, cfg, seabed)
                apply_boundaries(
                    state, anchor, auv1, auv_vel1, outlet_direction
                )
                remesh = redistribute_nodes_if_needed(state, cfg)
                if remesh.applied:
                    apply_boundaries(
                        state, anchor, auv1, auv_vel1, outlet_direction
                    )
                    recorder.record_remesh_event(
                        time=float(t0 + inner_dt),
                        node_count=int(state.num_nodes),
                        minimum_ratio_before=remesh.minimum_ratio_before,
                        minimum_ratio_after=remesh.minimum_ratio_after,
                        geometric_length_before=remesh.geometric_length_before,
                        geometric_length_after=remesh.geometric_length_after,
                        rest_length=float(state.total_rest_length),
                    )
                if inserted:
                    final_geometry = compute_segment_geometry(state, cfg)
                    final_axial = compute_tension(state, final_geometry, cfg)
                    final_ratio = final_geometry.length / np.maximum(
                        state.rest_length, cfg.minimum_segment_length
                    )
                    final_standard = (
                        final_ratio[:-1] if final_ratio.size > 1 else final_ratio
                    )
                    final_peak = _physical_peak_tension(final_axial, cfg)
                    recorder.complete_last_insertion_event(
                        peak_tension_post_step=final_peak,
                        peak_tension_post_step_jump=float(final_peak - before_peak),
                        minimum_standard_ratio_post_step=(
                            float(np.min(final_standard))
                            if final_standard.size else 1.0
                        ),
                        remeshed_in_same_step=bool(remesh.applied),
                    )
                validate_state_finite(state)
                assert_length_conservation(state, cfg, atol=1.0e-9)

            current_time = outer_end
            _record_current(state, current_time, cfg, seabed, recorder)

        return recorder.finalize(state, seabed, converged=True)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        output_dir = cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "error_traceback.txt").open("w", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        summary = recorder.finalize(state, seabed, converged=False, error=error)
        if raise_on_error:
            raise
        return summary
