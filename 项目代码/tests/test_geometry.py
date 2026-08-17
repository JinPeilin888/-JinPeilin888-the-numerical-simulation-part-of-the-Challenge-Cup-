import numpy as np

from src.config import SimulationConfig
from src.geometry import (
    bending_damping_forces,
    bending_energy,
    bending_forces,
    compute_segment_geometry,
    discrete_curvature_radius,
)
from src.state import CableState


def make_state(position):
    position = np.asarray(position, dtype=float)
    return CableState(position, np.zeros_like(position), np.ones(position.shape[0] - 1), np.zeros(position.shape[0], dtype=bool))


def test_segment_lengths_and_tangents():
    cfg = SimulationConfig(h_A=2.0, ds=1.0)
    state = make_state([[0, 0], [1, 0], [1, 1]])
    geom = compute_segment_geometry(state, cfg)
    np.testing.assert_allclose(geom.length, [1.0, 1.0])
    np.testing.assert_allclose(geom.tangent, [[1.0, 0.0], [0.0, 1.0]])


def test_three_point_curvature_circle():
    cfg = SimulationConfig()
    state = make_state([[1, 0], [0, 1], [-1, 0]])
    radius = discrete_curvature_radius(state.position, cfg)
    assert np.isclose(radius[1], 1.0)


def test_bending_force_has_zero_net_force_and_torque():
    position = np.array(
        [[0.0, 0.0], [0.8, 0.2], [1.5, 0.9], [2.2, 0.4]],
        dtype=float,
    )
    rest = np.array([0.8, 0.9, 0.8], dtype=float)
    force = bending_forces(position, rest, EI=0.03)
    np.testing.assert_allclose(np.sum(force, axis=0), np.zeros(2), atol=1.0e-12)
    torque = np.sum(position[:, 0] * force[:, 1] - position[:, 1] * force[:, 0])
    assert abs(float(torque)) <= 1.0e-12


def test_straight_cable_has_zero_bending_force():
    position = np.column_stack((np.linspace(0.0, 1.0, 6), np.zeros(6)))
    force = bending_forces(position, np.full(5, 0.2), EI=0.03)
    np.testing.assert_allclose(force, np.zeros_like(position), atol=1.0e-12)


def test_bending_force_matches_negative_energy_gradient():
    position = np.array(
        [[0.0, 0.0], [0.35, 0.08], [0.71, 0.24], [1.02, 0.05]],
        dtype=float,
    )
    rest = np.array([0.36, 0.40, 0.41], dtype=float)
    EI = 0.03
    analytical = bending_forces(position, rest, EI)
    numerical = np.zeros_like(position)
    step = 1.0e-7
    for node in range(position.shape[0]):
        for component in range(2):
            plus = position.copy()
            minus = position.copy()
            plus[node, component] += step
            minus[node, component] -= step
            numerical[node, component] = -(
                bending_energy(plus, rest, EI) - bending_energy(minus, rest, EI)
            ) / (2.0 * step)
    error = np.linalg.norm(analytical - numerical) / np.linalg.norm(numerical)
    assert error <= 1.0e-7


def test_bending_damping_never_adds_mechanical_power():
    position = np.array(
        [[0.0, 0.0], [0.35, 0.08], [0.71, 0.24], [1.02, 0.05]],
        dtype=float,
    )
    velocity = np.array(
        [[0.2, -0.1], [-0.3, 0.4], [0.1, 0.25], [-0.2, -0.15]],
        dtype=float,
    )
    force = bending_damping_forces(
        position, velocity, np.array([0.36, 0.40, 0.41]), coefficient=0.02
    )
    assert float(np.sum(force * velocity)) <= 1.0e-12
    np.testing.assert_allclose(np.sum(force, axis=0), np.zeros(2), atol=1.0e-12)


def test_bending_damping_is_zero_for_rigid_rotation():
    position = np.array(
        [[0.0, 0.0], [0.35, 0.08], [0.71, 0.24], [1.02, 0.05]],
        dtype=float,
    )
    velocity = np.column_stack((-position[:, 1], position[:, 0]))
    force = bending_damping_forces(
        position, velocity, np.array([0.36, 0.40, 0.41]), coefficient=0.02
    )
    np.testing.assert_allclose(force, np.zeros_like(position), atol=1.0e-12)
