from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig


@dataclass
class CableState:
    position: np.ndarray
    velocity: np.ndarray
    rest_length: np.ndarray
    contact_prev: np.ndarray
    payout_integral: float = 0.0

    def copy(self) -> "CableState":
        return CableState(
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            rest_length=self.rest_length.copy(),
            contact_prev=self.contact_prev.copy(),
            payout_integral=float(self.payout_integral),
        )

    @property
    def num_nodes(self) -> int:
        return int(self.position.shape[0])

    @property
    def num_segments(self) -> int:
        return int(self.rest_length.shape[0])

    @property
    def total_rest_length(self) -> float:
        return float(np.sum(self.rest_length))


def initialize_vertical_cable(cfg: SimulationConfig, seabed_height: float = 0.0) -> CableState:
    """Create a boundary-compatible cable whose rest length equals ``h_A``.

    The final segment is the active payout segment. Initial segment lengths
    are distributed uniformly, are no larger than ``ds``, and sum exactly to
    the requested initial length. ``initial_extra_length`` supplies the small
    amount of material geometrically required by a non-vertical outlet tangent.
    """
    initial_length = float(cfg.initial_rest_length)
    n_segments = max(1, int(np.ceil(initial_length / cfg.ds)))
    rest = np.full(n_segments, initial_length / n_segments, dtype=float)

    anchor = np.array([0.0, seabed_height], dtype=float)
    auv = np.array([0.0, seabed_height + cfg.h_A], dtype=float)
    position = np.zeros((n_segments + 1, 2), dtype=float)

    if not cfg.enforce_outlet_direction:
        cumulative = np.concatenate(([0.0], np.cumsum(rest)))
        position[:, 0] = 0.0
        position[:, 1] = seabed_height + cumulative
        position[-1] = auv
    elif n_segments == 1:
        position[0] = anchor
        position[-1] = auv
    else:
        # Make the guide segment satisfy the prescribed outlet direction from
        # the first force evaluation. The cubic's bow is solved so its arc
        # length equals the physical material length before the guide. This is
        # essential on fine grids: parameter-space sampling of a shorter curve
        # creates a hidden initial compression that excites stiff bending modes.
        guide_length = float(rest[-1])
        outlet_direction = cfg.outlet_direction
        guide_node = auv + guide_length * outlet_direction
        transition_length = min(0.04, 0.25 * cfg.h_A)
        control_2 = guide_node + transition_length * outlet_direction
        chord = guide_node - anchor
        chord_length = float(np.linalg.norm(chord))
        normal = np.array([-chord[1], chord[0]], dtype=float) / max(
            chord_length, 1.0e-12
        )
        base_control_1 = anchor + chord / 3.0
        parameter = np.linspace(0.0, 1.0, 4001)[:, None]

        def bezier(bow: float) -> np.ndarray:
            control_1 = base_control_1 + bow * normal
            return (
                (1.0 - parameter) ** 3 * anchor
                + 3.0 * (1.0 - parameter) ** 2 * parameter * control_1
                + 3.0 * (1.0 - parameter) * parameter**2 * control_2
                + parameter**3 * guide_node
            )

        def arc_length(points: np.ndarray) -> float:
            return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))

        target_length = float(np.sum(rest[:-1]))
        low = 0.0
        high = max(0.02, cfg.initial_extra_length, 0.05 * cfg.h_A)
        while arc_length(bezier(high)) < target_length:
            high *= 2.0
            if high > 10.0 * max(cfg.h_A, target_length):
                raise RuntimeError("Unable to construct the initial cable arc")
        for _ in range(60):
            middle = 0.5 * (low + high)
            if arc_length(bezier(middle)) < target_length:
                low = middle
            else:
                high = middle
        curve = bezier(0.5 * (low + high))
        increments = np.linalg.norm(np.diff(curve, axis=0), axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(increments)))
        targets = np.concatenate(([0.0], np.cumsum(rest[:-1])))
        position[:-1, 0] = np.interp(targets, cumulative, curve[:, 0])
        position[:-1, 1] = np.interp(targets, cumulative, curve[:, 1])
        position[-1] = auv
    velocity = np.zeros_like(position)
    contact_prev = np.zeros(position.shape[0], dtype=bool)
    contact_prev[0] = True
    return CableState(position, velocity, rest, contact_prev, payout_integral=0.0)
