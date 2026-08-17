from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig
from .seabed import Seabed
from .state import CableState


@dataclass(frozen=True)
class ContactResult:
    force: np.ndarray
    normal_force: np.ndarray
    penetration: np.ndarray
    contact: np.ndarray
    tangent_velocity: np.ndarray
    seabed_height: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray


def seabed_contact_and_friction(
    state: CableState, seabed: Seabed, cfg: SimulationConfig
) -> ContactResult:
    x = state.position[:, 0]
    z = state.position[:, 1]
    height = seabed.height(x)
    tangent, normal = seabed.frame(x)

    # Vertical gap and penetration are the MVP measures specified by the plan.
    # Contact damping must not act on a node that is still clearly in the water.
    gap = z - height
    penetration = np.maximum(-gap, 0.0)
    normal_velocity = np.einsum("ij,ij->i", state.velocity, normal)
    near_seabed = gap <= cfg.contact_gap_tolerance
    closing_velocity = np.minimum(normal_velocity, 0.0)
    representative = np.zeros(state.num_nodes, dtype=float)
    representative[0] = 0.5 * state.rest_length[0]
    representative[-1] = 0.5 * state.rest_length[-1]
    if state.num_nodes > 2:
        representative[1:-1] = 0.5 * (
            state.rest_length[:-1] + state.rest_length[1:]
        )

    # The configured coefficients are distributed foundation properties.
    # Scaling by represented cable length makes total support independent of
    # the number of contact nodes and removes a mesh-dependent stiffness jump.
    nodal_stiffness = cfg.seabed_stiffness * representative
    nodal_damping = cfg.seabed_damping * representative
    trial_normal_force = (
        nodal_stiffness * penetration
        - nodal_damping * closing_velocity
    )
    normal_force = np.where(
        near_seabed,
        np.maximum(trial_normal_force, 0.0),
        0.0,
    )
    support = normal_force[:, None] * normal

    tangent_velocity = np.einsum("ij,ij->i", state.velocity, tangent)
    friction_scalar = -cfg.seabed_friction * normal_force * np.tanh(
        tangent_velocity / cfg.friction_smoothing_velocity
    )
    friction = friction_scalar[:, None] * tangent
    total = support + friction

    # Geometry decides whether a material node is in the contact band.
    # This avoids false contact caused only by downward velocity damping.
    contact = near_seabed.copy()
    # The anchor is known to be on the seabed even when penalty force is zero.
    contact[0] = True
    # The moving guide point is not treated as a material-seabed contact node.
    contact[-1] = False
    return ContactResult(
        force=total,
        normal_force=normal_force,
        penetration=penetration,
        contact=contact,
        tangent_velocity=tangent_velocity,
        seabed_height=height,
        tangent=tangent,
        normal=normal,
    )
