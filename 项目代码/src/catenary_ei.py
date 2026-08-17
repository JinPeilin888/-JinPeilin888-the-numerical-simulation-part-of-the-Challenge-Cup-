from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class ElasticCatenarySolution:
    horizontal_tension: float
    effective_weight: float
    height: float
    EI: float
    a: float
    layback: float
    suspended_length: float
    tail_tension: float
    minimum_radius: float

    def sample(self, points: int = 200) -> dict[str, np.ndarray]:
        xs = np.linspace(0.0, self.layback, points)

        if self.EI <= 0.0:
            # 退化成普通 catenary
            z = self.a * (np.cosh(xs / self.a) - 1.0)
            curvature = np.abs(self.effective_weight / self.horizontal_tension) / np.cosh(xs / self.a)**2
        else:
            A = -(self.effective_weight / self.horizontal_tension) * self.a**2
            B = self.effective_weight / (2.0 * self.horizontal_tension)
            z = A * np.cosh(xs / self.a) + B * xs**2 + self.height - A

            dz = A / self.a * np.sinh(xs / self.a) + 2.0 * B * xs
            d2z = A / self.a**2 * np.cosh(xs / self.a) + 2.0 * B
            curvature = np.abs(d2z) / (1.0 + dz**2)**1.5

        radius = np.where(curvature > 1e-18, 1.0 / curvature, np.inf)
        arc = np.cumsum(np.sqrt(1.0 + np.gradient(z, xs)**2)) * (xs[1] - xs[0])

        return {
            "x": xs,
            "z": z,
            "radius": radius,
            "curvature": curvature,
            "arc_length": arc,
        }


def solve_elastic_catenary(
    horizontal_tension: float,
    effective_weight: float,
    height: float,
    EI: float,
    tol: float = 1e-12,
) -> ElasticCatenarySolution:
    if horizontal_tension <= 0.0:
        raise ValueError("horizontal_tension must be positive")
    if effective_weight <= 0.0:
        raise ValueError("effective_weight must be positive")
    if height <= 0.0:
        raise ValueError("height must be positive")

    a = np.sqrt(EI / horizontal_tension) if EI > 0.0 else 1e12

    # 用弧长约束求水平跨度 Lx
    def residual(Lx: float) -> float:
        if Lx <= 0.0 or Lx >= height * 2:
            return 1e6
        if EI <= 0.0:
            # 退化情况（你已有解析解）
            s = horizontal_tension / effective_weight * np.sinh(Lx / (horizontal_tension / effective_weight))
            return s - height
        else:
            A = -(effective_weight / horizontal_tension) * a**2
            B = effective_weight / (2.0 * horizontal_tension)
            xs = np.linspace(0, Lx, 400)
            dz = A / a * np.sinh(xs / a) + 2.0 * B * xs
            integrand = np.sqrt(1.0 + dz**2)
            s = np.trapz(integrand, xs)
            return s - height

    Lx = brentq(residual, 1e-6, height * 1.999, xtol=tol)

    # 最小弯曲半径出现在最低点
    if EI <= 0.0:
        min_radius = horizontal_tension / effective_weight
    else:
        min_radius = EI / (0.5 * effective_weight * a**2)

    tail_tension = horizontal_tension + effective_weight * height

    return ElasticCatenarySolution(
        horizontal_tension=horizontal_tension,
        effective_weight=effective_weight,
        height=height,
        EI=EI,
        a=a,
        layback=Lx,
        suspended_length=height,
        tail_tension=tail_tension,
        minimum_radius=min_radius,
    )