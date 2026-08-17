import numpy as np

from src.catenary import solve_from_horizontal_tension, solve_from_suspended_length


def test_catenary_end_height_and_tail_tension():
    sol = solve_from_horizontal_tension(0.02, 0.003, 0.5)
    sample = sol.sample(points=50)
    assert np.isclose(sample["z"][-1], 0.5)
    assert np.isclose(sol.tail_tension, 0.02 + 0.003 * 0.5)
    assert np.isclose(sol.minimum_radius, sol.a)


def test_inverse_from_suspended_length():
    sol = solve_from_suspended_length(0.8, 0.003, 0.5)
    assert np.isclose(sol.suspended_length, 0.8)
