from __future__ import annotations
import numpy as np
np.seterr(over="raise", invalid="raise", divide="raise")
import argparse
import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import load_config
from src.simulation import run_simulation


EXPECTED_V_A = (0.10, 0.20, 0.25, 0.30)
EXPECTED_H_A = (0.30, 0.50, 1.00)
EXPECTED_RATIOS = (0.90, 1.00, 1.05, 1.10, 1.20)


def _validate_cases(rows: list[dict[str, str]]) -> None:
    expected = {
        (v_a, h_a, ratio)
        for v_a in EXPECTED_V_A
        for h_a in EXPECTED_H_A
        for ratio in EXPECTED_RATIOS
    }
    actual = {
        (float(row["v_A"]), float(row["h_A"]), float(row["v_out_ratio"]))
        for row in rows
    }
    case_ids = [row["case_id"] for row in rows]
    if len(rows) != 60 or len(actual) != 60 or len(set(case_ids)) != 60:
        raise ValueError(
            "cases.csv must contain exactly 60 unique case IDs and parameter triples"
        )
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"60-case matrix mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _case_config(
    config_path: str,
    row: dict[str, str],
    output_root: str,
    t_end: float | None,
) -> Any:
    v_a = float(row["v_A"])
    h_a = float(row["h_A"])
    ratio = float(row["v_out_ratio"])
    overrides = {
        "case_id": row["case_id"],
        "v_A": v_a,
        "h_A": h_a,
        "v_out": v_a * ratio,
        "output_root": output_root,
        "save_history": False,
        "save_plots": False,
        "save_snapshots": False,
    }
    if t_end is not None:
        overrides["t_end"] = t_end
    return load_config(config_path, overrides)


def _run_one(
    config_path: str,
    row: dict[str, str],
    output_root: str,
    t_end: float | None,
) -> dict[str, Any]:
    cfg = _case_config(config_path, row, output_root, t_end)
    return run_simulation(cfg, raise_on_error=False)


def _load_completed(
    config_path: str,
    row: dict[str, str],
    output_root: str,
    t_end: float | None,
) -> dict[str, Any] | None:
    cfg = _case_config(config_path, row, output_root, t_end)
    case_dir = Path(output_root) / row["case_id"]
    resolved_path = case_dir / "resolved_config.json"
    summary_path = case_dir / "summary.json"
    if not resolved_path.exists() or not summary_path.exists():
        return None
    try:
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if resolved != cfg.to_dict():
        return None
    if summary.get("case_id") != row["case_id"]:
        return None
    return summary



def _write_comparison_plots(summaries: list[dict[str, Any]], output: Path) -> None:
    valid = [s for s in summaries if s.get("numerically_converged") and s.get("v_out_ratio") is not None]
    if not valid:
        return
    metrics = {
        "peak_active_tension": "Peak active tension (N)",
        "min_bend_radius": "Minimum bend radius (m)",
        "mean_layback": "Mean steady layback (m)",
        "max_slack_ratio": "Maximum slack ratio (-)",
    }
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in valid:
        groups.setdefault((float(row["v_A"]), float(row["h_A"])), []).append(row)
    for key, ylabel in metrics.items():
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        plotted = False
        for (v_a, h_a), rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda r: float(r["v_out_ratio"]))
            x = np.asarray([float(r["v_out_ratio"]) for r in rows])
            values = [r.get(key) for r in rows]
            mask = np.asarray([value is not None for value in values], dtype=bool)
            if not np.any(mask):
                continue
            y = np.asarray([float(value) if value is not None else np.nan for value in values])
            ax.plot(x[mask], y[mask], marker="o", label=f"vA={v_a:g}, hA={h_a:g}")
            plotted = True
        if plotted:
            ax.set_xlabel("Payout ratio v_out / v_A")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(ncol=2, fontsize=8)
            fig.tight_layout()
            fig.savefig(output / f"sweep_{key}.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the prescribed 60-case matrix.")
    parser.add_argument("--config", default="configs/cable_0p52mm.yaml")
    parser.add_argument("--cases", default="configs/cases.csv")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", default="outputs/sweep_60_freshwater")
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse a case only when summary and resolved_config exactly match.",
    )
    args = parser.parse_args()

    with Path(args.cases).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    _validate_cases(rows)
    summaries: list[dict[str, Any]] = []
    pending = rows
    if args.resume:
        pending = []
        for row in rows:
            completed = _load_completed(
                args.config, row, args.output_root, args.t_end
            )
            if completed is None:
                pending.append(row)
            else:
                summaries.append(completed)
        print(f"[resume] reused {len(summaries)}, pending {len(pending)}")

    if args.workers <= 1:
        for index, row in enumerate(pending, start=1):
            print(f"[{index}/{len(pending)}] {row['case_id']}")
            summaries.append(
                _run_one(args.config, row, args.output_root, args.t_end)
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            future_map = {
                pool.submit(
                    _run_one,
                    args.config,
                    row,
                    args.output_root,
                    args.t_end,
                ): row["case_id"]
                for row in pending
            }
            for index, future in enumerate(as_completed(future_map), start=1):
                case_id = future_map[future]
                try:
                    summaries.append(future.result())
                except Exception as exc:
                    summaries.append({
                        "case_id": case_id,
                        "numerically_converged": False,
                        "classification": "worker_failure",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                print(f"[{index}/{len(pending)}] completed {case_id}")

    summaries.sort(key=lambda item: str(item.get("case_id", "")))
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for item in summaries:
        for key in item:
            if key not in columns:
                columns.append(key)
    with (output / "sweep_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "sweep_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summaries, handle, ensure_ascii=False, indent=2)
    _write_comparison_plots(summaries, output)
    required = (
        "numerically_converged",
        "tension_safe",
        "break_safe",
        "bend_safe",
        "penetration_safe",
        "mesh_safe",
        "insertion_rest_length_safe",
        "curve_smooth",
    )
    failed = [
        item["case_id"]
        for item in summaries
        if not all(item.get(key) is True for key in required)
    ]
    validation = {
        "expected_case_count": 60,
        "completed_case_count": len(summaries),
        "unique_case_count": len({item.get("case_id") for item in summaries}),
        "all_constraints_passed": not failed and len(summaries) == 60,
        "failed_case_ids": failed,
    }
    with (output / "sweep_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
    if not validation["all_constraints_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
