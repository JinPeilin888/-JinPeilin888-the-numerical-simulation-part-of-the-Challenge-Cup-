from __future__ import annotations

import math
import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve

from .config import SimulationConfig
from .state import CableState


def estimate_stable_timestep(cfg: SimulationConfig) -> float:
    """Conservative explicit time scale from axial and contact stiffness."""
    axial_scale = cfg.ds * math.sqrt(cfg.line_density / cfg.EA)
    representative_mass = cfg.line_density * cfg.ds
    contact_scale = math.sqrt(representative_mass / cfg.seabed_stiffness)
    return cfg.stability_factor * min(axial_scale, contact_scale)


def choose_substeps(outer_dt: float, cfg: SimulationConfig) -> tuple[int, float]:
    if not cfg.auto_substep:
        return 1, outer_dt
    stable = estimate_stable_timestep(cfg)
    # A newly split tail segment can be much shorter than ds. Its adjacent
    # material node still carries roughly half a standard-segment mass, so use
    # that local spring-mass scale to protect the explicit update.
    tail_node_mass = 0.5 * cfg.line_density * cfg.ds
    tail_scale = math.sqrt(tail_node_mass * cfg.l0_min / cfg.EA)
    stable = min(stable, cfg.stability_factor * tail_scale)
    count = max(1, int(math.ceil(outer_dt / stable)))
    if count > cfg.max_internal_substeps:
        raise RuntimeError(
            f"Required internal substeps ({count}) exceed max_internal_substeps "
            f"({cfg.max_internal_substeps}). Reduce dt, EA or contact stiffness, "
            "or raise the explicit limit knowingly."
        )
    return count, outer_dt / count


def semi_implicit_euler(
    state: CableState,
    total_force: np.ndarray,
    mass: np.ndarray,
    dt: float,
    velocity_damping: float = 0.0,
) -> None:
    if state.num_nodes <= 2:
        return
    free = slice(1, -1)
    safe_mass = np.maximum(mass[free], 1.0e-16)
    acceleration = total_force[free] / safe_mass[:, None]
    state.velocity[free] += dt * acceleration
    if velocity_damping > 0.0:
        state.velocity[free] *= math.exp(-velocity_damping * dt)
    state.position[free] += dt * state.velocity[free]


def _append_block(
    rows: list[int],
    cols: list[int],
    values: list[float],
    row_node: int,
    col_node: int,
    block: np.ndarray,
) -> None:
    for local_row in range(2):
        for local_col in range(2):
            value = float(block[local_row, local_col])
            if value != 0.0:
                rows.append(2 * row_node + local_row)
                cols.append(2 * col_node + local_col)
                values.append(value)


def _linearized_force_matrices(
    state: CableState,
    geom,
    axial,
    nodal,
    contact,
    cfg: SimulationConfig,
):
    """Build dF/dx and dF/dv for the stiff and dissipative forces."""
    n = state.num_nodes
    size = 2 * n
    k_rows: list[int] = []
    k_cols: list[int] = []
    k_values: list[float] = []
    d_rows: list[int] = []
    d_cols: list[int] = []
    d_values: list[float] = []
    identity = np.eye(2)

    for index in range(state.num_segments):
        l0 = max(float(state.rest_length[index]), cfg.minimum_segment_length)
        length = max(float(geom.length[index]), cfg.minimum_segment_length)
        tangent = np.asarray(geom.tangent[index], dtype=float)
        tt = np.outer(tangent, tangent)
        active = bool(abs(axial.axial_force[index]) > 0.0 or abs(axial.strain[index]) > 0.0)
        # Always include the tensile tangent stiffness in the implicit matrix.
        # For a currently slack segment this acts as an active-set predictor:
        # the physical force remains zero, but one step cannot jump far across
        # the slack/taut boundary and create an artificial insertion impulse.
        stiffness_ratio = (
            cfg.compression_stiffness_ratio if axial.strain[index] < 0.0 else 1.0
        )
        elastic = (stiffness_ratio * cfg.EA / l0) * tt
        tension = float(axial.axial_force[index])
        if tension > 0.0:
            elastic += (tension / length) * (identity - tt)
        damping = (
            (stiffness_ratio * cfg.axial_damping / l0) * tt
            if active
            else np.zeros((2, 2), dtype=float)
        )

        left = index
        right = index + 1
        for matrix, rows, cols, values in (
            (elastic, k_rows, k_cols, k_values),
            (damping, d_rows, d_cols, d_values),
        ):
            _append_block(rows, cols, values, left, left, -matrix)
            _append_block(rows, cols, values, left, right, matrix)
            _append_block(rows, cols, values, right, left, matrix)
            _append_block(rows, cols, values, right, right, -matrix)

    if cfg.EI > 0.0 and n >= 3:
        edge_left = state.position[1:-1] - state.position[:-2]
        edge_right = state.position[2:] - state.position[1:-1]
        length_left_sq = np.maximum(
            np.einsum("ij,ij->i", edge_left, edge_left),
            cfg.minimum_segment_length**2,
        )
        length_right_sq = np.maximum(
            np.einsum("ij,ij->i", edge_right, edge_right),
            cfg.minimum_segment_length**2,
        )
        rotate_left = np.column_stack((-edge_left[:, 1], edge_left[:, 0]))
        rotate_right = np.column_stack((-edge_right[:, 1], edge_right[:, 0]))
        grad_previous = rotate_left / length_left_sq[:, None]
        grad_next = rotate_right / length_right_sq[:, None]
        grad_center = -grad_previous - grad_next
        local_scale = np.maximum(
            0.5 * (state.rest_length[:-1] + state.rest_length[1:]),
            cfg.minimum_segment_length,
        )
        coefficients = cfg.EI / local_scale
        damping_coefficients = (
            cfg.EI * cfg.bending_damping_time / local_scale
        )
        for center, coefficient in enumerate(coefficients, start=1):
            nodes = (center - 1, center, center + 1)
            gradients = (
                grad_previous[center - 1],
                grad_center[center - 1],
                grad_next[center - 1],
            )
            for row_local, row_node in enumerate(nodes):
                for col_local, col_node in enumerate(nodes):
                    block = -float(coefficient) * np.outer(
                        gradients[row_local], gradients[col_local]
                    )
                    _append_block(
                        k_rows, k_cols, k_values, row_node, col_node, block
                    )
                    damping_block = -float(
                        damping_coefficients[center - 1]
                    ) * np.outer(
                        gradients[row_local], gradients[col_local]
                    )
                    _append_block(
                        d_rows,
                        d_cols,
                        d_values,
                        row_node,
                        col_node,
                        damping_block,
                    )

    for index in range(n):
        normal = np.asarray(contact.normal[index], dtype=float)
        nn = np.outer(normal, normal)
        represented = float(nodal.representative_length[index])
        if bool(contact.contact[index]):
            contact_stiffness = cfg.seabed_stiffness * represented
            _append_block(
                k_rows,
                k_cols,
                k_values,
                index,
                index,
                -contact_stiffness * nn,
            )
            normal_velocity = float(np.dot(state.velocity[index], normal))
            if normal_velocity < 0.0:
                contact_damping = cfg.seabed_damping * represented
                _append_block(
                    d_rows,
                    d_cols,
                    d_values,
                    index,
                    index,
                    -contact_damping * nn,
                )

        relative = np.asarray(cfg.current_velocity) - state.velocity[index]
        normal_speed = float(np.dot(relative, nodal.normal[index]))
        drag_normal = (
            cfg.water_density
            * cfg.drag_coefficient
            * cfg.diameter
            * represented
            * abs(normal_speed)
        )
        drag_matrix = drag_normal * np.outer(
            nodal.normal[index], nodal.normal[index]
        )
        if cfg.tangential_drag_coefficient > 0.0:
            tangential_speed = float(np.dot(relative, nodal.tangent[index]))
            drag_tangent = (
                cfg.water_density
                * cfg.tangential_drag_coefficient
                * math.pi
                * cfg.diameter
                * represented
                * abs(tangential_speed)
            )
            drag_matrix += drag_tangent * np.outer(
                nodal.tangent[index], nodal.tangent[index]
            )
        _append_block(
            d_rows, d_cols, d_values, index, index, -drag_matrix
        )

    stiffness = coo_matrix(
        (k_values, (k_rows, k_cols)), shape=(size, size)
    ).tocsr()
    damping = coo_matrix(
        (d_values, (d_rows, d_cols)), shape=(size, size)
    ).tocsr()
    return stiffness, damping


def linearly_implicit_euler(
    state: CableState,
    total_force: np.ndarray,
    mass: np.ndarray,
    dt: float,
    cfg: SimulationConfig,
    geom,
    axial,
    nodal,
    contact,
    auv_velocity: np.ndarray,
    velocity_damping: float = 0.0,
) -> None:
    """Advance one stiffly stable step with the outlet-ray constraint.

    Axial, bending, contact and velocity-dependent forces are linearized at the
    current state. The penultimate node is a fixed spatial guide moving with
    the AUV; payout material is injected into the adjacent free-cable edge.
    """
    n = state.num_nodes
    if n <= 2:
        state.velocity[0] = 0.0
        state.velocity[-1] = auv_velocity
        return

    safe_mass = np.maximum(np.asarray(mass, dtype=float), 1.0e-16)
    mass_diagonal = np.repeat(safe_mass, 2)
    mass_matrix = diags(mass_diagonal, format="csr")
    stiffness, damping = _linearized_force_matrices(
        state, geom, axial, nodal, contact, cfg
    )

    force0 = np.asarray(total_force, dtype=float).copy()
    if velocity_damping > 0.0:
        force0 -= (
            velocity_damping * safe_mass[:, None] * state.velocity
        )
        damping = damping - velocity_damping * mass_matrix

    system = mass_matrix - dt * damping - dt * dt * stiffness
    old_velocity = state.velocity.reshape(-1)
    rhs = (
        mass_diagonal * old_velocity
        + dt * force0.reshape(-1)
        - dt * damping.dot(old_velocity)
    )

    known = np.zeros(2 * n, dtype=float)
    known[-2:] = np.asarray(auv_velocity, dtype=float)
    guide = n - 2
    if cfg.enforce_outlet_direction:
        known[2 * guide : 2 * guide + 2] = np.asarray(auv_velocity, dtype=float)

    p_rows: list[int] = []
    p_cols: list[int] = []
    p_values: list[float] = []
    column = 0
    free_node_stop = n - 2 if cfg.enforce_outlet_direction else n - 1
    for node in range(1, free_node_stop):
        for component in range(2):
            p_rows.append(2 * node + component)
            p_cols.append(column)
            p_values.append(1.0)
            column += 1
    transform = coo_matrix(
        (p_values, (p_rows, p_cols)), shape=(2 * n, column)
    ).tocsr()
    reduced_system = transform.T @ system @ transform
    reduced_rhs = transform.T @ (rhs - system.dot(known))
    reduced_velocity = spsolve(reduced_system.tocsc(), reduced_rhs)
    velocity = known + transform.dot(reduced_velocity)
    state.velocity[:] = velocity.reshape(n, 2)
    state.position[1:-1] += dt * state.velocity[1:-1]
