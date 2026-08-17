from __future__ import annotations

import argparse
import csv
from itertools import product
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.simulation import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run low/mid/high sensitivity cases for EA, C_D and mu.")
    parser.add_argument("--config", default="configs/cable_1p0mm.yaml")
    parser.add_argument("--output-root", default="outputs/sensitivity")
    parser.add_argument("--low-factor", type=float, default=0.5)
    parser.add_argument("--high-factor", type=float, default=2.0)
    args = parser.parse_args()

    base = load_config(args.config)
    levels = {
        "low": args.low_factor,
        "mid": 1.0,
        "high": args.high_factor,
    }
    rows = []
    for (ea_name, ea_factor), (cd_name, cd_factor), (mu_name, mu_factor) in product(
        levels.items(), levels.items(), levels.items()
    ):
        case_id = f"EA_{ea_name}_CD_{cd_name}_MU_{mu_name}"
        cfg = base.with_overrides(
            case_id=case_id,
            output_root=args.output_root,
            EA=base.EA * ea_factor,
            drag_coefficient=base.drag_coefficient * cd_factor,
            seabed_friction=base.seabed_friction * mu_factor,
            save_plots=False,
        )
        summary = run_simulation(cfg, raise_on_error=False)
        summary.update({
            "EA_level": ea_name,
            "CD_level": cd_name,
            "mu_level": mu_name,
            "EA_value": cfg.EA,
            "CD_value": cfg.drag_coefficient,
            "mu_value": cfg.seabed_friction,
        })
        rows.append(summary)
        print(case_id, summary.get("classification"))

    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with (output / "sensitivity_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
