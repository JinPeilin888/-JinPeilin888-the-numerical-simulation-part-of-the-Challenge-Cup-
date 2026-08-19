### 跑1mm缆要求
#### 跑1mm缆标准工况下8m的路程
##### 1、运行1mm标准工况
```bash
python scripts/run_case.py --config configs/cable_1p0mm.yaml --case-id cable_1p0mm_standard_8m --v-A 0.20 --h-A 0.50 --v-out 0.21 --t-end 40 --output-root outputs_1p0mm/1mm_standard_case
```
这一步可以得到：
* time_history.csv
* node_snapshots.npz
* tail_tension.png
* tail_tension_raw.png
* max_tension.png
* max_tension_raw.png
* suspended_length.png
* layback.png
* min_bend_radius.png
* cable_shape.png
* summary.json
* resolved_config.json
读者可以自行修改outputs后的输出目录

##### 2、得到1mm标准工况的动图
```bash
python scripts/make_animation.py "outputs_1p0mm/1mm_standard_case/cable_1p0mm_standard_8m/node_snapshots.npz" --output "outputs_1p0mm/1mm_standard_case/cable_1p0mm_standard_8m/animation_8m.gif" --fps 10
```

##### 3、跑1mm缆5种水平流速
```bash
python scripts/run_current_sweep_single.py --config configs/cable_1p0mm.yaml --output-root outputs_1p0mm/current_sweep_1mm --t-end 40
```
注意：这里最好把cable_1p0mm.yaml配置文件里的v_out放缆比改成0.21

##### 4、跑60工况（结果需要有time_history.csv，resolved_config.json,summary.json,insertion_events.csv，请读者自查)
其中time_history.csv可以生成需要的相关时间曲线等 

建议租服务器跑
```bash
python3 scripts/run_sweep.py --config configs/cable_1p0mm.yaml --workers 15 --resume --output-root outputs/sweep_60_freshwater_1p0mm
```

##### 5、跑60工况（敏感性分析）
输出应为类似网格时间步收敛.csv
```text
Set-Location -LiteralPath 'C:\Users\金沛霖\Downloads\MGC01_interim_code_results_18cases\项目代码' #就是进入项目代码目录，读者自行修改

$env:MPLCONFIGDIR = Join-Path $env:TEMP 'mgc01-mpl-cache'
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null #清理缓存，加快速度

@'
import csv
from pathlib import Path

from src.config import load_config
from src.simulation import run_simulation

# 加载1 mm缆配置
base = load_config(Path("configs/cable_1p0mm.yaml"))

# 与原收敛表保持相同工况：AUV速度和放缆速度均为0.21 m/s
base = base.with_overrides(v_out=0.21)

output_root = Path("outputs/final_validation_1p0mm/convergence")
result_file = Path("results/convergence_results_1p0mm.csv")
#output_root → 7个算例各自的详细结果文件夹
 result_file → 7个算例合并后的总表

# 空间收敛：网格减小时同步减小时间步
grid_pairs = (
    (0.010, 0.000250),
    (0.005, 0.000125),
    (0.002, 0.000050),
    (0.001, 0.000025),
)

# 时间收敛：固定ds=0.01 m，检查dt、dt/2、dt/4
cases = [
    ("grid", ds, dt) for ds, dt in grid_pairs
] + [
    ("time", 0.010, dt)
    for dt in (0.000500, 0.000250, 0.000125)
]

rows = []

for study, ds, dt in cases:
    case_id = f"{study}_ds_{ds:g}_dt_{dt:g}".replace(".", "p")

    cfg = base.with_overrides(
        case_id=case_id,
        output_root=str(output_root),
        ds=ds,
        dt=dt,
        t_end=0.10,
        ramp_time=0.10,
        relaxation_time=0.05,
        save_plots=False,
        save_snapshots=False,
    )
    cfg.validate()

    print(f"Running {case_id}: ds={ds:g} m, dt={dt:g} s")
    summary = run_simulation(cfg, raise_on_error=False)
    summary.update({
        "study": study,
        "ds": ds,
        "dt": dt,
    })
    rows.append(summary)

columns = []
for row in rows:
    for key in row:
        if key not in columns:
            columns.append(key)

result_file.parent.mkdir(parents=True, exist_ok=True)
with result_file.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print()
print(f"完成：{result_file.resolve()}")
for row in rows:
    print(
        row["case_id"],
        "converged =", row.get("numerically_converged"),
        "classification =", row.get("classification"),
        "error =", row.get("error"),
    )
'@ | python -
```
##### 6、运行额外增加的正负0.1流速
```bash
python scripts/run_current_sweep.py --config configs/cable_1p0mm.yaml --cases configs/cases.csv --output-root outputs/current_sweep_pm0p1_12s --velocity -0.1 0.1
```

额外要求如下：
* 1,60个工况可以只跑12秒，但是必须在结果中包含time_history.csv
```text
这些工况的话可以挑一些代表性（各10个吧）的记录成一个excel表做成附件。这个excel表里面的张力-时间曲线也可以用表格数值来表示。用time_history.csv来制作excel表
```
* 2,60个工况必须跑水平流速(但是可以只包括正负0.1）
* 3,跑1mm标准工况(包括生成动图所需的.npz文件、time_history.csv用于生成图表的)