import numpy as np
import pytest

from src.axial import compute_tension
from src.config import D_OUT, SimulationConfig
from src.deployment import (
    insert_nodes_if_needed,
    redistribute_nodes_if_needed,
)
from src.geometry import compute_segment_geometry
from src.simulation import apply_boundaries
from src.state import CableState, initialize_vertical_cable


@pytest.mark.parametrize("ds", [0.02, 0.03, 0.05])
def test_node_insertion_conserves_rest_length_for_allowed_ds(ds):
    cfg = SimulationConfig(h_A=0.30, ds=ds, payout_buffer_segments=1)
    residual = max(2.0 * cfg.l0_min, 0.001)
    tail_length = ds + residual
    state = CableState(
        position=np.array(
            [[0.0, 0.0], [0.0, ds], [0.05, 0.08], [0.1, 0.1]]
        ),
        velocity=np.zeros((4, 2)),
        rest_length=np.array([ds, tail_length, ds]),
        contact_prev=np.array([True, False, False, False]),
    )

    before = state.total_rest_length
    inserted = insert_nodes_if_needed(state, state.position[-1], np.zeros(2), cfg)

    assert inserted == 1
    assert state.num_nodes == 5
    assert state.total_rest_length == pytest.approx(before, rel=1e-12, abs=1e-12)
    assert np.min(state.rest_length[:-1]) >= 0.5 * ds - 1.0e-12
    assert state.rest_length[-3] == pytest.approx(ds)
    assert state.rest_length[-2] == pytest.approx(residual)
    assert state.rest_length[-1] == pytest.approx(ds)


def test_node_is_not_inserted_below_plan_threshold():
    cfg = SimulationConfig(h_A=0.30, ds=0.05, payout_buffer_segments=1)
    tail_length = cfg.ds + 0.5 * cfg.l0_min
    state = CableState(
        position=np.array(
            [[0.0, 0.0], [0.0, 0.05], [0.05, 0.08], [0.1, 0.1]]
        ),
        velocity=np.zeros((4, 2)),
        rest_length=np.array([0.05, tail_length, 0.05]),
        contact_prev=np.array([True, False, False, False]),
    )

    inserted = insert_nodes_if_needed(state, state.position[-1], np.zeros(2), cfg)
    assert inserted == 0
    assert state.num_nodes == 4


@pytest.mark.parametrize("ds", [0.02, 0.03, 0.05])
@pytest.mark.parametrize("height", [0.30, 0.50, 1.00])
def test_prescribed_initial_segments_do_not_exceed_ds(ds, height):
    cfg = SimulationConfig(h_A=height, ds=ds)
    state = initialize_vertical_cable(cfg)
    assert np.all(state.rest_length <= ds + 1e-12)
    assert state.total_rest_length == pytest.approx(cfg.initial_rest_length)


def test_initial_geometry_is_outlet_compatible_without_local_strain_spike():
    cfg = SimulationConfig(h_A=0.50, ds=0.02, EA=2000.0)
    state = initialize_vertical_cable(cfg)
    geom = compute_segment_geometry(state, cfg)
    axial = compute_tension(state, geom, cfg)

    outlet = state.position[-2] - state.position[-1]
    np.testing.assert_allclose(outlet / np.linalg.norm(outlet), D_OUT, atol=1.0e-12)
    assert np.max(axial.strain) < 0.05
    assert np.max(axial.tension) < cfg.T_max


def test_apply_boundaries_enforces_outlet_direction_and_velocity_projection():
    state = CableState(
        position=np.array([[0.3, -0.1], [0.2, 0.8], [0.9, 0.6]], dtype=float),
        velocity=np.array([[1.0, -2.0], [0.4, 0.2], [0.6, -0.3]], dtype=float),
        rest_length=np.array([0.25, 0.35], dtype=float),
        contact_prev=np.array([True, False, False], dtype=bool),
    )
    anchor = np.array([0.0, 0.0], dtype=float)
    auv_position = state.position[-1].copy()
    auv_velocity = np.array([0.1, -0.2], dtype=float)
    guide_length = state.rest_length[-1]

    apply_boundaries(state, anchor, auv_position, auv_velocity)

    np.testing.assert_allclose(state.position[0], anchor)
    np.testing.assert_allclose(state.velocity[0], np.zeros(2))
    np.testing.assert_allclose(state.position[-1], auv_position)
    np.testing.assert_allclose(state.velocity[-1], auv_velocity)

    outlet_vector = state.position[-2] - state.position[-1]
    outlet_length = np.linalg.norm(outlet_vector)
    np.testing.assert_allclose(outlet_length, guide_length)
    np.testing.assert_allclose(outlet_vector / outlet_length, D_OUT, atol=1.0e-12)

    np.testing.assert_allclose(state.velocity[-2], state.velocity[-1])


def test_redistribution_repairs_collapsed_standard_segment_without_length_loss():
    cfg = SimulationConfig(ds=0.01, remesh_trigger_ratio=0.5)
    state = CableState(
        position=np.array(
            [[0.0, 0.0], [0.0001, 0.0], [0.02, 0.01], [0.03, 0.02]],
            dtype=float,
        ),
        velocity=np.zeros((4, 2)),
        rest_length=np.array([0.01, 0.01, 0.01]),
        contact_prev=np.array([True, False, False, False]),
    )
    before = state.total_rest_length
    result = redistribute_nodes_if_needed(state, cfg)
    assert result.applied
    assert state.total_rest_length == pytest.approx(before)
    assert result.minimum_ratio_after >= result.minimum_ratio_before
