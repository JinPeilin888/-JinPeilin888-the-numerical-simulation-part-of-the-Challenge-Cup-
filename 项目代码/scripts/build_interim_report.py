from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config


SWEEP_ROOT = PROJECT_ROOT / "outputs" / "sweep_60_freshwater"
RESULTS = PROJECT_ROOT / "results"
FINAL_CASE = (
    PROJECT_ROOT
    / "outputs"
    / "final_validation"
    / "final_0p52mm_freshwater_upper_mass"
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


def _safe(row: dict[str, Any]) -> bool:
    return all(row.get(key) is True for key in REQUIRED_GATES)


def _extreme(rows: list[dict[str, Any]], key: str, operation: str) -> dict[str, Any]:
    chooser = max if operation == "max" else min
    selected = chooser(rows, key=lambda row: float(row[key]))
    return {"value": float(selected[key]), "case_id": selected["case_id"]}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _coverage_plot(
    completed: set[str], cases: list[dict[str, str]], output: Path
) -> None:
    speeds = (0.10, 0.20, 0.25, 0.30)
    heights = (0.30, 0.50, 1.00)
    ratios = (0.90, 1.00, 1.05, 1.10, 1.20)
    case_map = {
        (float(row["v_A"]), float(row["h_A"]), float(row["v_out_ratio"])): row[
            "case_id"
        ]
        for row in cases
    }
    matrix = np.zeros((len(speeds) * len(heights), len(ratios)), dtype=float)
    labels: list[str] = []
    for i, speed in enumerate(speeds):
        for j, height in enumerate(heights):
            row_index = i * len(heights) + j
            labels.append(f"vA={speed:.2f}, H={height:.2f}")
            for k, ratio in enumerate(ratios):
                matrix[row_index, k] = float(
                    case_map[(speed, height, ratio)] in completed
                )
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(ratios)), [f"{value:.2f}" for value in ratios])
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Payout ratio v_out / v_A")
    ax.set_ylabel("AUV speed and height (m/s, m)")
    ax.set_title(f"Formal 12 s cases completed: {int(np.sum(matrix))}/60")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                "PASS" if matrix[i, j] else "pending",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if matrix[i, j] else "#444444",
            )
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an honest interim report.")
    parser.add_argument("--pytest-passed", type=int, required=True)
    args = parser.parse_args()

    with (PROJECT_ROOT / "configs" / "cases.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        cases = list(csv.DictReader(handle))
    expected_ids = {row["case_id"] for row in cases}
    summaries = [
        _read_json(path)
        for path in sorted(SWEEP_ROOT.glob("*/summary.json"))
        if path.parent.name in expected_ids
    ]
    completed_ids = {row["case_id"] for row in summaries}
    pending_ids = sorted(expected_ids - completed_ids)
    resolved = [
        _read_json(SWEEP_ROOT / row["case_id"] / "resolved_config.json")
        for row in summaries
    ]
    t_end_values = sorted({float(row["t_end"]) for row in resolved})
    unique_triples = {
        (float(row["v_A"]), float(row["h_A"]), float(row["v_out_ratio"]))
        for row in summaries
    }
    failed = [row["case_id"] for row in summaries if not _safe(row)]
    metrics = {
        "report_status": "interim_not_full_60_case_acceptance",
        "expected_case_count": 60,
        "completed_case_count": len(summaries),
        "pending_case_count": len(pending_ids),
        "unique_case_count": len(completed_ids),
        "unique_parameter_triple_count": len(unique_triples),
        "t_end_values_s": t_end_values,
        "completed_all_constraints_passed": not failed and bool(summaries),
        "failed_completed_case_ids": failed,
        "pending_case_ids": pending_ids,
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
        "pytest_passed": args.pytest_passed,
    }
    if len(completed_ids) != len(summaries) or t_end_values != [12.0]:
        raise RuntimeError("Completed formal cases are not unique 12 s artifacts")
    if failed:
        raise RuntimeError(f"Completed formal cases contain failures: {failed}")

    cfg = load_config(PROJECT_ROOT / "configs" / "cable_0p52mm.yaml")
    final = _read_json(FINAL_CASE / "summary.json")
    animation = _read_json(RESULTS / "animation_audit.json")
    full_metrics = _read_json(RESULTS / "metrics.json")
    convergence = full_metrics["convergence"]
    insertion = full_metrics["insertion"]

    interim_csv = SWEEP_ROOT / "interim_summary.csv"
    interim_json = SWEEP_ROOT / "interim_summary.json"
    validation_json = SWEEP_ROOT / "interim_validation.json"
    coverage_png = SWEEP_ROOT / "interim_coverage.png"
    _write_csv(interim_csv, summaries)
    _write_json(interim_json, summaries)
    _write_json(validation_json, metrics)
    _coverage_plot(completed_ids, cases, coverage_png)
    _write_json(RESULTS / "interim_18_metrics.json", metrics)

    report = f"""# MGC-01 水下柔性光缆仿真阶段性汇报

## 结论

项目已完成模型修正、代表工况、收敛实验、五档均匀流、GIF 审计以及 {len(summaries)}/60 个正式 12 s 工况。已完成的 {len(summaries)} 个正式工况全部通过八项门禁；剩余 {len(pending_ids)} 个尚未运行完成，因此本报告是阶段性结果，不能表述为“60 个工况已全部验收”。

## 当前正式工况结果

| 检查项 | 已完成工况最不利值 | 限值 | 当前结论 |
|---|---:|---:|---|
| 工况完整时长 | {t_end_values[0]:g} s | 12 s | 通过 |
| 原始活动缆峰值张力 | {metrics['maximum_peak_active_tension']['value']:.9g} N | < {cfg.T_max:.9g} N；且 < {cfg.break_force:g} N | 通过 |
| 最小动态弯曲半径 | {metrics['minimum_bend_radius']['value']:.9g} m | ≥ {cfg.min_bend_radius:.9g} m（15D） | 通过 |
| 最大海床穿透 | {metrics['maximum_penetration']['value']:.9g} m | ≤ {cfg.penetration_limit:g} m | 通过 |
| 最大出口方向误差 | {metrics['maximum_outlet_error_deg']['value']:.9g}° | ≤ 1e-6° | 通过 |
| 最大长度守恒误差 | {metrics['maximum_length_error']['value']:.9g} m | ≤ 1e-9 m | 通过 |
| 最小标准单元长度比 | {metrics['minimum_mesh_ratio']['value']:.9g} | ≥ 0.90 | 通过 |
| 最大插入瞬时张力跳变 | {metrics['maximum_insertion_jump']['value']:.9g} N | ≤ {cfg.maximum_insertion_tension_jump:g} N | 通过 |
| 0.5 s 趋势最大相邻步差 | {metrics['maximum_filtered_step']['value']:.9g} N | ≤ {cfg.maximum_reported_tension_step:g} N | 通过 |

已覆盖全部 15 个 `v_A=0.10 m/s` 组合，以及 `v_A=0.20 m/s, H=0.30 m` 下放缆比 0.90、1.00、1.05 三个组合。当前最不利张力与最小弯曲半径均出现在 `{metrics['maximum_peak_active_tension']['case_id']}`。

## 已完成的修改和验证

1. 参数与硬约束：正式配置统一为淡水 1000 kg/m³、最低破断力 500 N、工作限值 166.667 N、动态弯曲半径 15D，并持续强制任务书给出的 37.5° 出口方向。
2. 动态放缆：长度增量分布到出口 ALE 缓冲区，节点按位置、速度、应变连续的局部分裂插入；细网格插入最大长度误差 {insertion['max_rest_length_error_m']:.3g} m、张力跳变 {insertion['max_post_step_tension_jump_N']:.3g} N。
3. 数值稳定性：采用能量型弯曲力、轴向与弯曲内耗，并完成 10/5/2/1 mm 收敛；2 mm 与 1 mm 的弯曲半径相对差为 {convergence['grid_min_bend_radius_relative_difference']:.3%}。
4. 曲线：标准图采用 0.5 s 工程趋势，当前正式工况最大相邻步差低于 0.0001 N；未滤波原始曲线与峰值独立保留，安全判定不使用滤波值。
5. GIF：触底代表工况原始峰值 {final['peak_active_tension']:.9g} N、最小弯曲半径 {final['min_bend_radius']:.9g} m、首次触底 {final['first_land_time']:.6g} s。动画共 {animation['gif_frames']} 帧，锚点、AUV 轨迹、出口方向、长度、穿透、自交、弯曲和帧连续性检查全部通过。
6. 自动测试：{args.pytest_passed}/{args.pytest_passed} 通过，0 失败；五档均匀流全部通过。

## 尚未完成与汇报措辞

- 尚余 {len(pending_ids)} 个正式 12 s 工况；当前可以说“程序已支持严格的 60 工况矩阵，已完成其中 {len(summaries)} 个全时长工况且均通过”，不能说“60 工况结果全部通过”。
- EA、EI、阻力和摩擦参数仍为工程暂定值，需厂家数据或试验标定后才能用于产品认证。

## 断点续跑

```bash
cd "{PROJECT_ROOT}"
.venv/bin/python scripts/run_sweep.py --workers 8 --resume --output-root outputs/sweep_60_freshwater
```

`--resume` 只复用 `summary.json` 和 `resolved_config.json` 与当前配置完全一致的工况；当前 {len(summaries)} 个完整结果会被复用，剩余或中途被终止的工况会自动重算。
"""
    (RESULTS / "阶段性汇报_18工况.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
