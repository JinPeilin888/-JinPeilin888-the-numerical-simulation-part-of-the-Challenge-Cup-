import pytest

from src.config import SimulationConfig


@pytest.mark.parametrize("ds", [0.001, 0.002, 0.005, 0.01, 0.05])
def test_ds_inside_revised_range_is_valid(ds):
    SimulationConfig(ds=ds).validate()


@pytest.mark.parametrize("ds", [0.0009, 0.051])
def test_ds_outside_revised_range_is_rejected(ds):
    with pytest.raises(ValueError, match="ds must be within"):
        SimulationConfig(ds=ds).validate()


def test_allowable_tension_overrides_break_force_ratio():
    cfg = SimulationConfig(break_force=2.0, safety_factor=3.0, allowable_tension=0.5)
    cfg.validate()
    assert cfg.T_max == pytest.approx(0.5)
