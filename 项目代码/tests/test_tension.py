import numpy as np
import pytest

from src.axial import compute_tension
from src.config import SimulationConfig
from src.geometry import compute_segment_geometry
from src.state import CableState


def state_for(length, rest=1.0, velocities=None):
    position = np.array([[0.0, 0.0], [length, 0.0]])
    velocity = np.zeros_like(position) if velocities is None else np.asarray(velocities, dtype=float)
    return CableState(position, velocity, np.array([rest]), np.array([True, False]))


def test_no_extension_gives_zero_tension():
    cfg = SimulationConfig(EA=10.0, axial_damping=0.1)
    state = state_for(1.0)
    result = compute_tension(state, compute_segment_geometry(state, cfg), cfg)
    assert result.tension[0] == 0.0


def test_compression_is_regularized_but_not_reported_as_tension():
    cfg = SimulationConfig(
        EA=10.0,
        axial_damping=0.0,
        compression_stiffness_ratio=0.01,
    )
    state = state_for(0.9)
    result = compute_tension(state, compute_segment_geometry(state, cfg), cfg)
    assert result.tension[0] == 0.0
    assert result.axial_force[0] == pytest.approx(-0.01)


def test_equal_endpoint_motion_has_zero_strain_rate():
    cfg = SimulationConfig(EA=10.0, axial_damping=1.0)
    state = state_for(1.0, velocities=[[2.0, 3.0], [2.0, 3.0]])
    result = compute_tension(state, compute_segment_geometry(state, cfg), cfg)
    assert np.isclose(result.strain_rate[0], 0.0)
