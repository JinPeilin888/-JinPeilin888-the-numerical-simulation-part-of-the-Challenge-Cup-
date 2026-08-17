from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _fmt(value, digits=6):
    if value is None:
        return "未记录"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}g}"
    return str(value)


def single_case_report(summary: dict, output: Path) -> None:
    text = f"""# 二维动态铺缆单工况结果报告

## 工况

- 工况编号：`{summary.get('case_id')}`
- AUV 速度：{_fmt(summary.get('v_A'))} m/s
- 离底高度：{_fmt(summary.get('h_A'))} m
- 放缆速度：{_fmt(summary.get('v_out'))} m/s
- 放缆比：{_fmt(summary.get('v_out_ratio'))}

## 主要结果

- 尾部峰值张力：{_fmt(summary.get('peak_tail_tension'))} N
- 活动光缆峰值张力：{_fmt(summary.get('peak_active_tension'))} N
- 稳态尾部平均张力：{_fmt(summary.get('steady_tail_tension_mean'))} N
- 展示趋势峰值：{_fmt(summary.get('filtered_peak_active_tension'))} N
- 展示趋势滤波窗口：{_fmt(summary.get('tension_filter_window_s'))} s
- 展示趋势最大相邻步差：{_fmt(summary.get('max_filtered_tension_step_N'))} N
- 最小弯曲半径：{_fmt(summary.get('min_bend_radius'))} m
- 平均 Layback：{_fmt(summary.get('mean_layback'))} m
- 最终 Layback：{_fmt(summary.get('final_layback'))} m
- 平均悬空长度：{_fmt(summary.get('mean_suspended_length'))} m
- 首次着地时间：{_fmt(summary.get('first_land_time'))} s
- 首次着地位置：{_fmt(summary.get('first_land_x'))} m
- 最大海底穿透：{_fmt(summary.get('max_penetration'))} m
- 最大松缆比例：{_fmt(summary.get('max_slack_ratio'))}
- 最大长度守恒误差：{_fmt(summary.get('max_length_conservation_error'))} m
- 最大出口方向误差：{_fmt(summary.get('max_outlet_direction_error_deg'))}°
- 插入瞬时最大张力跳变：{_fmt(summary.get('max_insertion_tension_jump_N'))} N

## 判定

- 张力硬限制通过：{_fmt(summary.get('tension_safe'))}
- 弯曲半径硬限制通过：{_fmt(summary.get('bend_safe'))}
- 穿透量限制通过：{_fmt(summary.get('penetration_safe'))}
- 网格质量限制通过：{_fmt(summary.get('mesh_safe'))}
- 节点插入长度守恒通过：{_fmt(summary.get('insertion_rest_length_safe'))}
- 报告曲线平滑门禁通过：{_fmt(summary.get('curve_smooth'))}
- 数值收敛完成：{_fmt(summary.get('numerically_converged'))}
- MVP 分类：`{summary.get('classification')}`

## 使用边界

本报告的安全判定使用未滤波原始张力；滤波曲线只用于观察工程趋势。模型为含转角弯曲能量和弯曲内耗的二维集中质量离散杆，不表示三维自接触、扭转、线盘机构及 AUV 受缆反力后的姿态变化。EA、EI、水动力和海床参数在实测标定前只适合工程趋势比较，不能作为产品认证值。
"""
    output.write_text(text, encoding="utf-8")


def sweep_report(csv_path: Path, output: Path) -> None:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    valid = [r for r in rows if r.get("numerically_converged", "").lower() == "true"]
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
    accepted = [
        row
        for row in rows
        if all(row.get(key, "").lower() == "true" for key in required)
    ]
    text = f"""# 二维动态铺缆工况扫描报告

- 总工况数：{len(rows)}
- 数值完成工况：{len(valid)}
- 全部八项门禁通过工况：{len(accepted)}
- 未通过工况：{len(rows) - len(accepted)}

完整数值见 `{csv_path.name}`。逐工况同时检查数值收敛、工作张力、500 N 破断力、15D 动态弯曲半径、海床穿透、网格质量、插入长度守恒及曲线平滑；安全张力采用未滤波原始值。
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a concise Markdown result report.")
    parser.add_argument("input", help="summary.json or sweep_summary.csv")
    parser.add_argument("--output")
    args = parser.parse_args()
    source = Path(args.input)
    output = Path(args.output) if args.output else source.with_name("report.md")
    if source.suffix.lower() == ".json":
        single_case_report(json.loads(source.read_text(encoding="utf-8")), output)
    elif source.suffix.lower() == ".csv":
        sweep_report(source, output)
    else:
        raise ValueError("Input must be summary.json or sweep_summary.csv")
    print(output)


if __name__ == "__main__":
    main()
