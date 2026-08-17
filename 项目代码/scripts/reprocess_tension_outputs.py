from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SimulationConfig
from src.recorder import Recorder


def reprocess(case_dir: Path) -> dict[str, object]:
    cfg = SimulationConfig(
        **json.loads((case_dir / "resolved_config.json").read_text(encoding="utf-8"))
    )
    with (case_dir / "time_history.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.pop("filtered_tail_tension", None)
        row.pop("filtered_max_active_tension", None)

    recorder = Recorder(cfg)
    recorder.rows = rows
    recorder._write_history(case_dir)
    if cfg.save_plots:
        recorder._plot_time_series(case_dir)

    filtered_tail, filtered_active = recorder._filtered_tensions()
    filtered_step = max(
        float(np.max(np.abs(np.diff(filtered_tail))))
        if filtered_tail.size > 1
        else 0.0,
        float(np.max(np.abs(np.diff(filtered_active))))
        if filtered_active.size > 1
        else 0.0,
    )
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.update(
        {
            "filtered_peak_tail_tension": float(np.max(filtered_tail)),
            "filtered_peak_active_tension": float(np.max(filtered_active)),
            "tension_filter_window_s": cfg.tension_filter_window,
            "max_filtered_tension_step_N": filtered_step,
            "reported_curve_smooth": (
                filtered_step <= cfg.maximum_reported_tension_step
            ),
        }
    )
    summary["curve_smooth"] = bool(
        summary.get("insertion_smooth") and summary["reported_curve_smooth"]
    )
    if summary["curve_smooth"] and summary.get("classification") == "rejected_curve_smoothness":
        summary["classification"] = (
            "warning_high_slack"
            if float(summary.get("max_slack_ratio", 0.0)) > 0.80
            else "acceptable_mvp"
        )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild tension trend columns and plots from raw history."
    )
    parser.add_argument("case_dir", type=Path)
    parser.add_argument(
        "--filter-window",
        type=float,
        help="Override the reporting-only tension trend window in seconds.",
    )
    parser.add_argument(
        "--maximum-step",
        type=float,
        help="Override the reporting-only adjacent-step acceptance in N.",
    )
    args = parser.parse_args()
    if args.filter_window is not None or args.maximum_step is not None:
        config_path = args.case_dir / "resolved_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if args.filter_window is not None:
            config["tension_filter_window"] = args.filter_window
        if args.maximum_step is not None:
            config["maximum_reported_tension_step"] = args.maximum_step
        SimulationConfig(**config).validate()
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = reprocess(args.case_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
