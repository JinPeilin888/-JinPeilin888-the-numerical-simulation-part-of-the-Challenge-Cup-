import numpy as np

from src.config import SimulationConfig
from src.deployment import grow_tail_rest_length
from src.state import CableState
from src.validation import assert_length_conservation


def test_payout_length_conservation_without_insertion():
    cfg = SimulationConfig(
        h_A=0.1,
        ds=0.05,
        v_out=0.02,
        ramp_time=0.0,
        initial_extra_length=0.0,
    )
    state = CableState(
        position=np.array([[0.0, 0.0], [0.0, 0.05], [0.0, 0.1]]),
        velocity=np.zeros((3, 2)),
        rest_length=np.array([0.05, 0.05]),
        contact_prev=np.array([True, False, False]),
    )
    grow_tail_rest_length(state, 0.0, 0.1, cfg)
    assert_length_conservation(state, cfg)
