import numpy as np

from src.config import SimulationConfig
from src.geometry import NodalProperties
from src.hydrodynamics import gravity_buoyancy


def test_effective_weight_direction_and_value():
    cfg = SimulationConfig(diameter=0.001, line_density=0.00110, water_density=1000.0)
    nodal = NodalProperties(
        representative_length=np.array([1.0]),
        mass=np.array([cfg.line_density]),
        tangent=np.array([[1.0, 0.0]]),
        normal=np.array([[0.0, 1.0]]),
    )
    force = gravity_buoyancy(nodal, cfg)
    assert force[0, 1] < 0.0
    assert np.isclose(force[0, 1], -cfg.effective_weight_per_length)
