from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.simulation import run_simulation


def _read_velocities(path: Path) -> list[float]:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    values = raw.get("experiments", {}).get(
        "current_velocity_x_values",
        [-0.20, -0.10, 0.0, 0.10, 0.20],
    )
    velocities = [float(value) for value in values]
    expected = [-0.20, -0.10, 0.0, 0.10, 0.20]
    if velocities != expected:
        raise ValueError(f"Current sweep must use exactly {expected}, got {velocities}")
    return velocities


def _tag(value: float) -> str:
    sign = "p" if value >= 0.0 else "m"
    return f"{sign}{abs(value):.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the five prescribed uniform-current cases.")
    parser.add_argument("--config", default="configs/cable_0p52mm.yaml")
    parser.add_argument("--output-root", default="outputs/current_sweep")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--ramp-time", type=float, default=None)
    parser.add_argument("--relaxation-time", type=float, default=None)
    parser.add_argument("--ds", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    base = load_config(config_path)
    rows: list[dict[str, object]] = []
    for velocity in _read_velocities(config_path):
        overrides: dict[str, object] = {
            "case_id": f"current_{_tag(velocity)}",
            "output_root": args.output_root,
            "current_velocity_x": velocity,
            "save_plots": False,
            "save_snapshots": False,
        }
        for name in ("t_end", "ramp_time", "relaxation_time", "ds", "dt"):
            value = getattr(args, name)
            if value is not None:
                overrides[name] = value
        cfg = base.with_overrides(**overrides)
        cfg.validate()
        summary = run_simulation(cfg, raise_on_error=False)
        summary.update(
            {
                "current_velocity_x": velocity,
                "current_direction": (
                    "同航向" if velocity > 0.0 else "逆航向" if velocity < 0.0 else "静水"
                ),
                "ds": cfg.ds,
                "dt": cfg.dt,
            }
        )
        rows.append(summary)
        print(velocity, summary.get("classification"))

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with (output / "current_sweep.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
