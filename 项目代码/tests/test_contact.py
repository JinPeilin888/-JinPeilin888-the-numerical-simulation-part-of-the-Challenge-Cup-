import numpy as np
import pytest

from src.config import SimulationConfig
from src.contact import seabed_contact_and_friction
from src.seabed import Seabed
from src.state import CableState


def test_no_contact_above_seabed():
    cfg = SimulationConfig(h_A=1.0)
    state = CableState(
        np.array([[0.0, 0.0], [0.5, 0.2], [1.0, 1.0]]),
        np.zeros((3, 2)),
        np.array([0.5, 0.5]),
        np.array([True, False, False]),
    )
    result = seabed_contact_and_friction(state, Seabed(cfg), cfg)
    assert result.normal_force[1] == 0.0


def test_support_is_upward_and_friction_opposes_motion():
    cfg = SimulationConfig(h_A=1.0, seabed_friction=0.5)
    state = CableState(
        np.array([[0.0, 0.0], [0.5, -0.01], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [0.2, -0.1], [0.0, 0.0]]),
        np.array([0.5, 0.5]),
        np.array([True, False, False]),
    )
    result = seabed_contact_and_friction(state, Seabed(cfg), cfg)
    assert result.force[1, 1] > 0.0
    assert result.force[1, 0] < 0.0


def test_downward_node_above_seabed_is_not_contact():
    cfg = SimulationConfig(h_A=1.0)
    state = CableState(
        np.array([[0.0, 0.0], [0.5, 0.20], [1.0, 1.0]]),
        np.array([[0.0, 0.0], [0.0, -0.10], [0.0, 0.0]]),
        np.array([0.5, 0.5]),
        np.array([True, False, False]),
    )
    result = seabed_contact_and_friction(state, Seabed(cfg), cfg)
    assert result.normal_force[1] == 0.0
    assert not result.contact[1]


def test_distributed_contact_force_is_mesh_independent():
    cfg = SimulationConfig(h_A=1.0, seabed_friction=0.0)

    coarse_position = np.column_stack((np.linspace(0.0, 1.0, 3), np.full(3, -0.01)))
    coarse = CableState(
        coarse_position,
        np.zeros_like(coarse_position),
        np.full(2, 0.5),
        np.zeros(3, dtype=bool),
    )
    fine_position = np.column_stack((np.linspace(0.0, 1.0, 5), np.full(5, -0.01)))
    fine = CableState(
        fine_position,
        np.zeros_like(fine_position),
        np.full(4, 0.25),
        np.zeros(5, dtype=bool),
    )

    coarse_force = seabed_contact_and_friction(coarse, Seabed(cfg), cfg)
    fine_force = seabed_contact_and_friction(fine, Seabed(cfg), cfg)
    assert np.sum(coarse_force.normal_force) == pytest.approx(
        np.sum(fine_force.normal_force), rel=1.0e-12
    )
