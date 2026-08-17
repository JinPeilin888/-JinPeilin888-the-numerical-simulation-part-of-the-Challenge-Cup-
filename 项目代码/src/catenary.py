from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class CatenarySolution:
    horizontal_tension: float
    effective_weight: float
    height: float
    a: float
    layback: float
    suspended_length: float
    tail_tension: float
    minimum_radius: float

    def sample(self, points: int = 200) -> dict[str, np.ndarray]:
        x = np.linspace(0.0, self.layback, points)
        u = x / self.a
        z = self.a * (np.cosh(u) - 1.0)
        arc_length = self.a * np.sinh(u)
        vertical_tension = self.effective_weight * arc_length
        total_tension = np.sqrt(self.horizontal_tension**2 + vertical_tension**2)
        angle = np.arctan2(vertical_tension, self.horizontal_tension)
        radius = self.a * np.cosh(u) ** 2
        return {
            "x": x,
            "z": z,
            "arc_length": arc_length,
            "vertical_tension": vertical_tension,
            "total_tension": total_tension,
            "angle": angle,
            "radius": radius,
        }


def solve_from_horizontal_tension(
    horizontal_tension: float, effective_weight: float, height: float
) -> CatenarySolution:
    if horizontal_tension <= 0.0:
        raise ValueError("horizontal_tension must be positive")
    if effective_weight <= 0.0:
        raise ValueError("effective_weight must be positive for the hanging catenary formula")
    if height < 0.0:
        raise ValueError("height must be non-negative")
    a = horizontal_tension / effective_weight
    layback = a * math.acosh(1.0 + height / a)
    suspended = a * math.sinh(layback / a)
    tail = horizontal_tension + effective_weight * height
    return CatenarySolution(
        horizontal_tension=horizontal_tension,
        effective_weight=effective_weight,
        height=height,
        a=a,
        layback=layback,
        suspended_length=suspended,
        tail_tension=tail,
        minimum_radius=a,
    )


def solve_from_suspended_length(
    suspended_length: float, effective_weight: float, height: float
) -> CatenarySolution:
    """Solve the TDP catenary when height and suspended arc length are known."""
    if height <= 0.0:
        raise ValueError("height must be positive")
    if suspended_length <= height:
        raise ValueError("suspended_length must be greater than height")
    a = (suspended_length**2 - height**2) / (2.0 * height)
    horizontal_tension = a * effective_weight
    return solve_from_horizontal_tension(horizontal_tension, effective_weight, height)


def catenary_shape(x: np.ndarray, horizontal_tension: float, effective_weight: float) -> np.ndarray:
    a = horizontal_tension / effective_weight
    return a * (np.cosh(np.asarray(x, dtype=float) / a) - 1.0)
