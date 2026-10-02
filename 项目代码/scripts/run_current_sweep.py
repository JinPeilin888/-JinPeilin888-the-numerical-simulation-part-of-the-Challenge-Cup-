from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import json

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

def _load_completed(
    output_root: str,
    case_id: str,
    expected_config: dict[str, object],
) -> dict[str, object] | None:
    case_dir = Path(output_root) / case_id
    summary_path = case_dir / "summary.json"
    resolved_path = case_dir / "resolved_config.json"

    if not summary_path.exists() or not resolved_path.exists():
        return None

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(summary, dict):
        return None
    if summary.get("case_id") != case_id:
        return None
    if resolved != expected_config:
        return None

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every prescribed case at each uniform-current velocity."
    )
    parser.add_argument(
        "--velocity",
        type=float,
        nargs="+",
        default=None,
        help="Current velocities in m/s, e.g. --velocity -0.1 0.1",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过配置完全一致且已有完整 summary.json 的工况",
    )
    parser.add_argument("--config", default="configs/cable_0p52mm.yaml")
    parser.add_argument("--cases", default="configs/cases.csv")
    parser.add_argument("--output-root", default="outputs/current_sweep")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--ramp-time", type=float, default=None)
    parser.add_argument("--relaxation-time", type=float, default=None)
    parser.add_argument("--ds", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    args = parser.parse_args()
    config_path = Path(args.config)
    base = load_config(config_path)

    with Path(args.cases).open(
            "r",
            encoding="utf-8",
            newline="",
    ) as handle:
        case_rows = list(csv.DictReader(handle))

    if len(case_rows) != 60:
        raise ValueError(
            f"Expected 60 base cases, got {len(case_rows)}"
        )

    rows: list[dict[str, object]] = []

    velocities = (
        args.velocity
        if args.velocity is not None
        else _read_velocities(config_path)
    )

    total_cases = len(case_rows) * len(velocities)
    print(
        f"{len(case_rows)} base cases × "
        f"{len(velocities)} velocities = "
        f"{total_cases} simulations"
    )
    completed = 0

    for case_row in case_rows:
        v_a = float(case_row["v_A"])
        h_a = float(case_row["h_A"])
        ratio = float(case_row["v_out_ratio"])

        for velocity in velocities:
            completed += 1

            case_id = (
                f"{case_row['case_id']}"
                f"_U_{_tag(velocity)}"
            )

            overrides: dict[str, object] = {
                "case_id": case_id,
                "output_root": args.output_root,
                "v_A": v_a,
                "h_A": h_a,
                "v_out": v_a * ratio,
                "current_velocity_x": velocity,
                "current_velocity_z": 0.0,
                "save_plots": False,
                "save_snapshots": False,
            }

            for name in (
                    "t_end",
                    "ramp_time",
                    "relaxation_time",
                    "ds",
                    "dt",
            ):
                value = getattr(args, name)
                if value is not None:
                    overrides[name] = value

            cfg = base.with_overrides(**overrides)
            cfg.validate()

            summary = None

            if args.resume:
                summary = _load_completed(
                    args.output_root,
                    case_id,
                    cfg.to_dict(),
                )

            if summary is not None:
                print(
                    f"[{completed}/{total_cases}] resumed "
                    f"{case_id}: {summary.get('classification')}"
                )
            else:
                print(
                    f"[{completed}/{total_cases}] "
                    f"{case_id}: U={velocity:+.2f} m/s"
                )

                summary = run_simulation(
                    cfg,
                    raise_on_error=False,
                )

            summary.update(
                {
                    "source_case_id": case_row["case_id"],
                    "v_A": v_a,
                    "h_A": h_a,
                    "v_out_ratio": ratio,
                    "v_out": v_a * ratio,
                    "current_velocity_x": velocity,
                    "current_velocity_z": 0.0,
                    "current_direction": (
                        "同航向"
                        if velocity > 0.0
                        else "逆航向"
                        if velocity < 0.0
                        else "静水"
                    ),
                    "ds": cfg.ds,
                    "dt": cfg.dt,
                }
            )

            rows.append(summary)

            print(
                f"[{completed}/{total_cases}] completed "
                f"{case_id}: "
                f"{summary.get('classification')}"
            )

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
