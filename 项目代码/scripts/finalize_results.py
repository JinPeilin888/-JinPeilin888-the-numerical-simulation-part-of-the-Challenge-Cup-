from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


RESULTS = PROJECT_ROOT / "results"
FINAL_CASE = (
    PROJECT_ROOT
    / "outputs"
    / "final_validation"
    / "final_0p52mm_freshwater_upper_mass"
)
FINAL_GIF = PROJECT_ROOT / "outputs" / "final_validation" / "demo_animation.gif"
SWEEP_ROOT = PROJECT_ROOT / "outputs" / "sweep_60_freshwater"
TASK_BOOK = Path(
    "/Users/gerrilynn/Desktop/CUMCM/金沛霖—仿真任务/"
    "MGC-01-1.0-ETFE水下柔性光缆铺放仿真任务书.docx"
)
REQUIRED_GATES = (
    "numerically_converged",
    "tension_safe",
    "break_safe",
    "bend_safe",
    "penetration_safe",
    "mesh_safe",
    "insertion_rest_length_safe",
    "curve_smooth",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe(summary: dict[str, Any]) -> bool:
    return all(summary.get(key) is True for key in REQUIRED_GATES)


def _extreme(
    summaries: list[dict[str, Any]], key: str, operation: str
) -> dict[str, Any]:
    rows = [row for row in summaries if row.get(key) is not None]
    chooser = max if operation == "max" else min
    selected = chooser(rows, key=lambda row: float(row[key]))
    return {
        "value": float(selected[key]),
        "case_id": selected["case_id"],
    }


def _write_numerical_results(
    final_summary: dict[str, Any], sweep: dict[str, Any]
) -> None:
    rows: list[dict[str, Any]] = []
    for key, value in final_summary.items():
        if isinstance(value, (bool, int, float)) and value is not None:
            rows.append({"category": "final_case", "metric": key, "value": value})
    for key, value in sweep.items():
        if isinstance(value, (bool, int, float)) and value is not None:
            rows.append({"category": "sweep_60", "metric": key, "value": value})
    with (RESULTS / "numerical_results.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=("category", "metric", "value"))
        writer.writeheader()
        writer.writerows(rows)


def _refresh_summary_table(case_root: Path, table_path: Path) -> None:
    """Refresh a result table from reprocessed case summaries.

    Study labels such as current direction and grid size are preserved from the
    previous table because they are experiment metadata, not solver outputs.
    """
    prior: dict[str, dict[str, Any]] = {}
    if table_path.exists():
        with table_path.open("r", encoding="utf-8", newline="") as handle:
            prior = {row["case_id"]: row for row in csv.DictReader(handle)}
    rows: list[dict[str, Any]] = []
    for path in sorted(case_root.glob("*/summary.json")):
        summary = _read_json(path)
        metadata = prior.get(summary["case_id"], {})
        row = dict(summary)
        for key, value in metadata.items():
            if key not in row:
                row[key] = value
        rows.append(row)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _format_report(
    cfg,
    final: dict[str, Any],
    sweep: dict[str, Any],
    animation: dict[str, Any],
    pytest_passed: int,
) -> str:
    return f"""# MGC-01 水下柔性光缆仿真最终验收报告

结论：正式 60 个不同工况和触底代表工况均通过全部门禁；标准张力曲线平滑，未滤波原始张力独立保留并用于安全判定；GIF 的力学状态与逐帧几何审计均通过。

## 验收结果

| 项目 | 最不利结果 | 限值 | 结论 |
|---|---:|---:|---|
| 60 工况数量/唯一性 | {sweep['completed_case_count']}/{sweep['unique_case_count']} | 60/60 | 通过 |
| 原始活动缆峰值张力 | {sweep['maximum_peak_active_tension']['value']:.9g} N | < {cfg.T_max:.9g} N（工作）且 < {cfg.break_force:g} N（破断） | 通过 |
| 最小动态弯曲半径 | {sweep['minimum_bend_radius']['value']:.9g} m | ≥ {cfg.min_bend_radius:.9g} m（15D） | 通过 |
| 最大海床穿透 | {sweep['maximum_penetration']['value']:.9g} m | ≤ {cfg.penetration_limit:.9g} m | 通过 |
| 最大出口方向误差 | {sweep['maximum_outlet_error_deg']['value']:.9g}° | ≤ 1e-6° | 通过 |
| 最大无应变长度误差 | {sweep['maximum_length_error']['value']:.9g} m | ≤ 1e-9 m | 通过 |
| 最小标准单元长度比 | {sweep['minimum_mesh_ratio']['value']:.9g} | ≥ 0.90 | 通过 |
| 最大插入瞬时张力跳变 | {sweep['maximum_insertion_jump']['value']:.9g} N | ≤ {cfg.maximum_insertion_tension_jump:g} N | 通过 |
| 0.5 s 趋势曲线最大相邻步差 | {sweep['maximum_filtered_step']['value']:.9g} N | ≤ {cfg.maximum_reported_tension_step:g} N | 通过 |
| 单元测试 | {pytest_passed} 项通过 | 0 项失败 | 通过 |

60 工况严格对应 `4 × 3 × 5` 笛卡尔积：AUV 速度 4 档、离底高度 3 档、放缆比 5 档；每个工况均计算到 12 s。全部工况同时通过数值收敛、工作张力、破断力、弯曲半径、穿透、网格、插入守恒和曲线平滑八项门禁。

## 曲线与 GIF

- 触底代表工况原始峰值张力为 {final['peak_active_tension']:.9g} N；0.5 s 趋势最大相邻步差为 {final['max_filtered_tension_step_N']:.9g} N。
- GIF 为 {animation['gif_size_px'][0]}×{animation['gif_size_px'][1]}、{animation['gif_frames']} 帧、{animation['time_start_s']:g}–{animation['time_end_s']:g} s；锚点固定、AUV 轨迹、37.5° 出口方向、长度守恒、海床穿透、自交、弯曲半径和帧连续性全部通过。
- 任务书标称线密度在淡水中略微正浮，正式 60 工况因此用于名义参数扫描；GIF 采用任务书允许公差上限 0.24 g/m，展示可审计的真实触底过程，首次触底时间为 {final['first_land_time']:.6g} s。

## 使用边界

EA、EI、阻力系数和摩擦系数仍是工程暂定值，需用厂家数据或试验标定后才能用于产品认证。模型是二维离散杆，不表示扭转、三维自接触和打结。
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh final validation files from completed artifacts."
    )
    parser.add_argument("--pytest-passed", type=int, required=True)
    args = parser.parse_args()

    cfg_path = PROJECT_ROOT / "configs" / "cable_0p52mm.yaml"
    cfg = load_config(cfg_path)
    _refresh_summary_table(
        PROJECT_ROOT / "outputs" / "final_validation" / "convergence",
        RESULTS / "convergence_result_0p52mm.csv",
    )
    _refresh_summary_table(
        PROJECT_ROOT / "outputs" / "final_validation" / "current_sweep",
        RESULTS / "current_sweep.csv",
    )
    convergence_summaries = [
        _read_json(path)
        for path in sorted(
            (PROJECT_ROOT / "outputs" / "final_validation" / "convergence").glob(
                "*/summary.json"
            )
        )
    ]
    current_summaries = [
        _read_json(path)
        for path in sorted(
            (PROJECT_ROOT / "outputs" / "final_validation" / "current_sweep").glob(
                "*/summary.json"
            )
        )
    ]
    if len(convergence_summaries) != 7 or not all(
        _safe(row) for row in convergence_summaries
    ):
        raise RuntimeError("Convergence cases do not all pass the current gates")
    if len(current_summaries) != 5 or not all(
        _safe(row) for row in current_summaries
    ):
        raise RuntimeError("Five-current cases do not all pass the current gates")
    final_summary = _read_json(FINAL_CASE / "summary.json")
    animation = _read_json(RESULTS / "animation_audit.json")
    sweep_validation = _read_json(SWEEP_ROOT / "sweep_validation.json")
    summaries: list[dict[str, Any]] = _read_json(SWEEP_ROOT / "sweep_summary.json")

    resolved = [
        _read_json(SWEEP_ROOT / row["case_id"] / "resolved_config.json")
        for row in summaries
    ]
    t_ends = {float(item["t_end"]) for item in resolved}
    unique_triples = {
        (float(row["v_A"]), float(row["h_A"]), float(row["v_out_ratio"]))
        for row in summaries
    }
    failed = [row["case_id"] for row in summaries if not _safe(row)]
    sweep_metrics: dict[str, Any] = {
        "completed_case_count": len(summaries),
        "unique_case_count": len({row["case_id"] for row in summaries}),
        "unique_parameter_triple_count": len(unique_triples),
        "t_end_values_s": sorted(t_ends),
        "failed_case_ids": failed,
        "all_constraints_passed": bool(
            sweep_validation.get("all_constraints_passed")
            and len(summaries) == 60
            and len(unique_triples) == 60
            and t_ends == {12.0}
            and not failed
        ),
        "maximum_peak_active_tension": _extreme(
            summaries, "peak_active_tension", "max"
        ),
        "minimum_bend_radius": _extreme(summaries, "min_bend_radius", "min"),
        "maximum_penetration": _extreme(summaries, "max_penetration", "max"),
        "maximum_outlet_error_deg": _extreme(
            summaries, "max_outlet_direction_error_deg", "max"
        ),
        "maximum_length_error": _extreme(
            summaries, "max_length_conservation_error", "max"
        ),
        "minimum_mesh_ratio": _extreme(
            summaries, "min_standard_segment_ratio", "min"
        ),
        "maximum_insertion_jump": _extreme(
            summaries, "max_insertion_tension_jump_N", "max"
        ),
        "maximum_filtered_step": _extreme(
            summaries, "max_filtered_tension_step_N", "max"
        ),
        "new_landing_case_count": sum(
            row.get("first_land_time") is not None for row in summaries
        ),
    }
    if not sweep_metrics["all_constraints_passed"]:
        raise RuntimeError(f"60-case validation failed: {sweep_metrics}")
    if not _safe(final_summary) or not animation.get("passed"):
        raise RuntimeError("Final case or animation audit is not safe")

    parameter_audit = {
        "config": str(cfg_path),
        "config_sha256": _sha256(cfg_path),
        "task_book": str(TASK_BOOK),
        "task_book_sha256": _sha256(TASK_BOOK),
        "parameter_status": "provisional",
        "water_density_kg_m3": cfg.water_density,
        "break_force_N": cfg.break_force,
        "outlet_angle_deg": cfg.outlet_angle_deg,
        "outlet_enforced": cfg.enforce_outlet_direction,
        "allowable_tension_N": cfg.T_max,
        "dynamic_bend_radius_m": cfg.min_bend_radius,
        "low_load_bend_radius_m": cfg.low_load_bend_radius,
    }
    _write_json(RESULTS / "parameter_audit.json", parameter_audit)
    _write_json(RESULTS / "sweep_60_metrics.json", sweep_metrics)

    metrics = _read_json(RESULTS / "metrics.json")
    metrics.update(
        {
            "parameter_audit": parameter_audit,
            "final_summary": final_summary,
            "animation": animation,
            "sweep_60": sweep_metrics,
            "pytest_passed": args.pytest_passed,
            "current_all_safe": True,
        }
    )
    metrics["all_constraints_passed"] = bool(
        metrics.get("convergence", {}).get("passed")
        and metrics.get("insertion", {}).get("passed")
        and metrics.get("current_all_safe")
        and sweep_metrics["all_constraints_passed"]
        and _safe(final_summary)
        and animation["passed"]
    )
    if not metrics["all_constraints_passed"]:
        raise RuntimeError("Combined validation did not pass")
    _write_json(RESULTS / "metrics.json", metrics)
    _write_numerical_results(final_summary, sweep_metrics)

    analysis = _format_report(
        cfg, final_summary, sweep_metrics, animation, args.pytest_passed
    )
    (RESULTS / "结果分析.md").write_text(analysis, encoding="utf-8")
    (RESULTS / "final_validation_report.md").write_text(
        analysis, encoding="utf-8"
    )

    manifest_paths = (
        RESULTS / "metrics.json",
        RESULTS / "numerical_results.csv",
        RESULTS / "sweep_60_metrics.json",
        RESULTS / "final_validation_report.md",
        RESULTS / "animation_audit.json",
        SWEEP_ROOT / "sweep_summary.csv",
        SWEEP_ROOT / "sweep_summary.json",
        SWEEP_ROOT / "sweep_validation.json",
        FINAL_CASE / "summary.json",
        FINAL_CASE / "max_tension.png",
        FINAL_CASE / "max_tension_raw.png",
        FINAL_CASE / "cable_shape.png",
        FINAL_GIF,
    )
    manifest = {
        "task_book": str(TASK_BOOK),
        "task_book_sha256": _sha256(TASK_BOOK),
        "commands": [
            ".venv/bin/python run.py",
            ".venv/bin/python scripts/run_sweep.py --workers 8 --resume --output-root outputs/sweep_60_freshwater",
            ".venv/bin/python -m pytest -q",
            f".venv/bin/python scripts/finalize_results.py --pytest-passed {args.pytest_passed}",
        ],
        "outputs": [
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(path),
            }
            for path in manifest_paths
        ],
    }
    _write_json(RESULTS / "复现清单.json", manifest)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
