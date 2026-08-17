from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimulationConfig


@dataclass(frozen=True)
class Seabed:
    cfg: SimulationConfig

    def height(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        kind = self.cfg.seabed_type.lower()
        if kind == "flat":
            return np.full_like(x_arr, self.cfg.seabed_level, dtype=float)
        if kind == "sinusoidal":
            wave = max(self.cfg.seabed_wavelength, 1.0e-12)
            return self.cfg.seabed_level + self.cfg.seabed_amplitude * np.sin(2.0 * np.pi * x_arr / wave)
        if kind == "gaussian_bump":
            width = max(self.cfg.seabed_width, 1.0e-12)
            u = (x_arr - self.cfg.seabed_center) / width
            return self.cfg.seabed_level + self.cfg.seabed_amplitude * np.exp(-0.5 * u * u)
        raise ValueError(f"Unsupported seabed_type: {self.cfg.seabed_type}")

    def slope(self, x: np.ndarray | float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=float)
        kind = self.cfg.seabed_type.lower()
        if kind == "flat":
            return np.zeros_like(x_arr, dtype=float)
        if kind == "sinusoidal":
            wave = max(self.cfg.seabed_wavelength, 1.0e-12)
            return (
                self.cfg.seabed_amplitude
                * (2.0 * np.pi / wave)
                * np.cos(2.0 * np.pi * x_arr / wave)
            )
        if kind == "gaussian_bump":
            width = max(self.cfg.seabed_width, 1.0e-12)
            u = (x_arr - self.cfg.seabed_center) / width
            return -self.cfg.seabed_amplitude * u * np.exp(-0.5 * u * u) / width
        raise ValueError(f"Unsupported seabed_type: {self.cfg.seabed_type}")

    def frame(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        slope = self.slope(x)
        denom = np.sqrt(1.0 + slope * slope)
        tangent = np.column_stack((1.0 / denom, slope / denom))
        normal = np.column_stack((-slope / denom, 1.0 / denom))
        return tangent, normal


def anchor_position(seabed: Seabed) -> np.ndarray:
    return np.array([0.0, float(seabed.height(0.0))], dtype=float)
