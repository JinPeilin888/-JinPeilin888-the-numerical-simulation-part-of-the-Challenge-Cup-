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


def _read_convergence_values(config_path: str | Path, base_dt: float) -> tuple[list[float], list[float]]:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    section = raw.get("convergence", {})
    ds_values = [float(value) for value in section.get("ds_values", [0.05, 0.03, 0.02])]
    dt_values = [
        float(value)
        for value in section.get("dt_values", [base_dt, base_dt / 2.0, base_dt / 4.0])
    ]

    if not ds_values:
        raise ValueError("convergence.ds_values must not be empty")
    if not dt_values or any(value <= 0.0 for value in dt_values):
        raise ValueError("convergence.dt_values must contain positive values")

    # Validate every ds through SimulationConfig.load_config/with_overrides later,
    # and give an early, readable error here as well.
    if any(not (0.001 <= value <= 0.05) for value in ds_values):
        raise ValueError("Every convergence.ds_values entry must be within 0.001--0.05 m")

    return ds_values, dt_values


def _tag(value: float, digits: int = 6) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ds and dt convergence cases.")
    parser.add_argument("--config", default="configs/cable_1p0mm.yaml")
    parser.add_argument("--output-root", default="outputs/convergence")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--ramp-time", type=float, default=None)
    parser.add_argument("--relaxation-time", type=float, default=None)
    args = parser.parse_args()

    base = load_config(args.config)
    ds_values, dt_values = _read_convergence_values(args.config, base.dt)

    cases: list[tuple[str, dict[str, float]]] = []
    for index, ds in enumerate(ds_values):
        # Pair each refined grid with its synchronized time step when provided.
        paired_dt = (
            dt_values[index]
            if len(dt_values) == len(ds_values)
            else base.dt * ds / base.ds
        )
        cases.append((f"grid_ds_{_tag(ds)}", {"ds": ds, "dt": paired_dt}))
    for dt in dt_values:
        # Time-step convergence: vary dt and keep the baseline ds.
        cases.append((f"time_dt_{_tag(dt)}", {"ds": base.ds, "dt": dt}))

    rows = []
    for case_id, overrides in cases:
        extra = {
            "save_plots": False,
            "save_snapshots": False,
        }
        if args.t_end is not None:
            extra["t_end"] = args.t_end
        if args.ramp_time is not None:
            extra["ramp_time"] = args.ramp_time
        if args.relaxation_time is not None:
            extra["relaxation_time"] = args.relaxation_time
        cfg = base.with_overrides(
            case_id=case_id,
            output_root=args.output_root,
            **overrides,
            **extra,
        )
        cfg.validate()
        summary = run_simulation(cfg, raise_on_error=False)
        summary.update({"study": "grid" if case_id.startswith("grid_") else "time", "ds": cfg.ds, "dt": cfg.dt})
        rows.append(summary)

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    with (output / "convergence_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
