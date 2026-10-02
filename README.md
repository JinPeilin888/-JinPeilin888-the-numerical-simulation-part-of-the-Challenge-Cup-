# MGC-01 ETFE 水下柔性光缆铺放仿真

本项目用 Python 实现二维水下柔性光缆动力学仿真，用于分析 AUV 航行放缆过程中的缆形、张力、弯曲半径、悬空长度、落后距与海床接触状态。代码支持 0.52 mm 和 1.0 mm 两种缆径、单工况计算、60 工况批算、均匀流扫描、网格/时间步收敛分析、敏感性分析与 GIF 动画生成。

> 请先阅读“[版本状态](#版本状态)”和“[工程使用边界](#工程使用边界)”。本包含有历史阶段记录和后续批算结果，不能只根据文件夹名判断完成度。

## 功能概览

- 二维 `x-z` 平面动力学求解，其中 `x` 向右、`z` 向上。
- 考虑重力、浮力、水动力阻力、轴向弹性与阻尼、弯曲刚度、海床接触和摩擦。
- 支持 AUV 轨迹、放缆速度比、出口方向约束、动态节点插入和网格重分布。
- 输出原始/平滑张力、最小弯曲半径、落后距、悬空长度、缆形和节点快照。
- 对数值收敛、张力、破断、弯曲半径、海床穿透、网格质量、插入守恒和曲线平滑性进行自动检查。

## 版本状态

2026-08-24 目录审计结果如下：

| 结果集 | 已有结果 | 八项门禁全通过 | 未通过工况 |
| --- | ---: | ---: | --- |
| 0.52 mm，60 工况 | 60/60 | 59/60 | `VA_0p30_H_0p30_R_0p90` |
| 1.0 mm，60 工况 | 60/60 | 58/60 | `VA_0p25_H_0p30_R_0p90`、`VA_0p30_H_0p30_R_0p90` |

上述 3 个工况的分类为 `rejected_curve_smoothness`；其余大部分工况为 `warning_high_slack`，它是警告分类，不等于八项硬门禁失败。权威汇总文件是：

- `提交材料/0.52mm缆最终成果提交/sweep_60_freshwater_0p52mm/sweep_validation.json`
- `提交材料/1mm缆最终成果提交/sweep_60_freshwater_1p0mm/sweep_validation.json`

## 目录结构

```text
MGC01_interim_code_results_18cases/
├── README.md                         # 本文档
├── 00_打开说明.md                  # 历史阶段包说明
├── 任务书/                           # 原始任务书
├── 提交材料/                       # 整理后的图表、汇总和两种缆径结果
│   ├── 0.52mm缆最终成果提交/
│   ├── 1mm缆最终成果提交/
│   └── docs/项目设计思路.md
└── 项目代码/
    ├── configs/                    # 缆型配置和 60 工况矩阵
    ├── src/                        # 求解器、力学模块、记录与验证
    ├── scripts/                    # 单工况、批算、海流、收敛、敏感性、动画脚本
    ├── tests/                      # pytest 测试
    ├── figures/                    # 项目图件
    ├── outputs/                    # 代码目录中的计算输出
    ├── model_contract.yaml         # 模型口径、假设与验收契约
    ├── pyproject.toml
    ├── requirements.txt
    └── run.py                      # 综合验证入口
```

## 环境准备

`pyproject.toml` 要求 Python 3.10 或更高版本，项目历史记录建议使用 Python 3.12。主要依赖包括 NumPy、SciPy、PyYAML、Matplotlib、Pillow 和 pytest。

在项目根目录进入代码目录：

```powershell
Set-Location -LiteralPath '.\项目代码'
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS：

```bash
cd 项目代码
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

验证安装：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py --smoke-test
```

本整理版在 2026-08-24 使用 Python 3.14.6 复核为 `46 passed`；为保持与原始计算环境一致，正式重算仍建议使用 Python 3.12。

## 快速开始

以下命令默认都在 `项目代码/` 目录执行。

### 1. 运行单个工况

0.52 mm 标准工况：

```powershell
.\.venv\Scripts\python.exe scripts/run_case.py `
  --config configs/cable_0p52mm.yaml `
  --case-id cable_0p52mm_standard_8m `
  --v-A 0.20 --h-A 0.50 --v-out 0.21 `
  --t-end 40 `
  --output-root outputs_0p52mm/standard_case
```

1.0 mm 标准工况：

```powershell
.\.venv\Scripts\python.exe scripts/run_case.py `
  --config configs/cable_1p0mm.yaml `
  --case-id cable_1p0mm_standard_8m `
  --v-A 0.20 --h-A 0.50 --v-out 0.21 `
  --t-end 40 `
  --output-root outputs_1p0mm/standard_case
```

命令行参数优先级高于 YAML，YAML 参数优先级高于代码默认值。如果希望某参数保持 YAML 中的设定，不要在命令行重复传入。

### 2. 生成缆形动画

单工况必须保存 `node_snapshots.npz` 才能制作动画。默认配置已启用快照保存。

```powershell
.\.venv\Scripts\python.exe scripts/make_animation.py `
  "outputs_0p52mm/standard_case/cable_0p52mm_standard_8m/node_snapshots.npz" `
  --output "outputs_0p52mm/standard_case/animation_8m.gif" `
  --fps 10
```

实际算例子目录由 `output_root/case_id` 组成。如找不到快照文件，先在输出目录中确认真实的 `case_id` 路径。

### 3. 运行 60 工况矩阵

`configs/cases.csv` 是固定的 `4 × 3 × 5 = 60` 工况矩阵：

- AUV 速度 `v_A`：0.10、0.20、0.25、0.30 m/s；
- AUV 高度 `h_A`：0.30、0.50、1.00 m；
- 放缆比 `v_out / v_A`：0.90、1.00、1.05、1.10、1.20。

0.52 mm：

```powershell
.\.venv\Scripts\python.exe scripts/run_sweep.py `
  --config configs/cable_0p52mm.yaml `
  --cases configs/cases.csv `
  --workers 8 --resume --t-end 12 `
  --output-root outputs/sweep_60_freshwater_0p52mm
```

1.0 mm：

```powershell
.\.venv\Scripts\python.exe scripts/run_sweep.py `
  --config configs/cable_1p0mm.yaml `
  --cases configs/cases.csv `
  --workers 8 --resume --t-end 12 `
  --output-root outputs/sweep_60_freshwater_1p0mm
```

`--resume` 只会复用同时存在 `summary.json` 和 `resolved_config.json`，且已解析配置与当前配置完全一致的工况。文件不完整或配置有变化时，该工况会自动重算。正式批算期间不要修改求解代码或 YAML，否则无法保证整批结果同口径。

### 4. 运行均匀流工况

对一个代表工况运行 YAML 中定义的 5 档流速 `-0.2/-0.1/0/+0.1/+0.2 m/s`：

```powershell
.\.venv\Scripts\python.exe scripts/run_current_sweep_single.py `
  --config configs/cable_0p52mm.yaml `
  --t-end 12 `
  --output-root outputs/current_sweep_single_0p52mm
```

对 60 个基础工况分别运行 `-0.1/+0.1 m/s`，总计 120 次仿真：

```powershell
.\.venv\Scripts\python.exe scripts/run_current_sweep.py `
  --config configs/cable_0p52mm.yaml `
  --cases configs/cases.csv `
  --velocity -0.1 0.1 `
  --resume --t-end 12 `
  --output-root outputs/current_sweep_pm0p1_12s_0p52mm
```

1.0 mm 缆运行时请同时更换 `--config` 和 `--output-root`，防止覆盖或混合两种缆径的结果。

### 5. 收敛与敏感性分析

```powershell
# 网格/时间步收敛
.\.venv\Scripts\python.exe scripts/run_convergence.py `
  --config configs/cable_0p52mm.yaml `
  --output-root outputs/convergence_0p52mm

# 参数敏感性
.\.venv\Scripts\python.exe scripts/run_sensitivity.py `
  --config configs/cable_0p52mm.yaml `
  --output-root outputs/sensitivity_0p52mm `
  --low-factor 0.5 --high-factor 2.0
```

收敛分析使用 YAML `convergence` 段定义的 `ds` 和 `dt`；高精度网格计算量较大，建议先用短 `t_end` 验证环境后再正式运行。

## 输入与配置

两份主配置文件是：

- `项目代码/configs/cable_0p52mm.yaml`
- `项目代码/configs/cable_1p0mm.yaml`

配置分为 `case`、`cable`、`water`、`seabed`、`safety`、`numerical`、`output`、`convergence` 和 `experiments` 等段。常用参数如下：

| 参数 | 含义 | 单位 |
| --- | --- | --- |
| `v_A` | AUV 航行速度 | m/s |
| `h_A` | AUV 相对海床高度 | m |
| `v_out` | 放缆速度 | m/s |
| `t_end` | 仿真时长 | s |
| `dt` | 外层时间步 | s |
| `ds` | 标准缆段长度 | m |
| `EA` | 轴向刚度 | N |
| `EI` | 弯曲刚度 | N·m² |
| `current_velocity_x` | 水平均匀流速 | m/s |
| `min_bend_radius` | 最小动态弯曲半径 | m |
| `allowable_tension` | 许用工作张力 | N |

每次正式计算都应保留 `resolved_config.json`，它是复现该工况时实际生效参数的主要依据。

## 输出说明

单工况的输出位于 `<output_root>/<case_id>/`。根据 `save_history`、`save_plots` 和 `save_snapshots` 开关，通常包含：

| 文件 | 用途 |
| --- | --- |
| `summary.json` | 关键指标、分类和八项安全/数值门禁 |
| `resolved_config.json` | 该工况实际使用的完整配置 |
| `time_history.csv` | 逐时刻张力、缆长、落后距、弯曲半径等时序 |
| `insertion_events.csv` | 动态节点插入事件与误差 |
| `node_snapshots.npz` | 节点位置快照，用于生成动画 |
| `tail_tension*.png` | 尾端张力原始/平滑曲线 |
| `max_tension*.png` | 最大张力原始/平滑曲线 |
| `min_bend_radius.png` | 最小弯曲半径时程 |
| `suspended_length.png` | 悬空长度时程 |
| `layback.png` | 落后距时程 |
| `cable_shape.png` | 缆形结果 |

批算目录还会生成 `sweep_summary.csv/json` 和指标对比图。安全判定使用原始未滤波峰值；平滑曲线只用于趋势展示，不能替代原始数据做安全结论。

## 结果判读

`summary.json` 中以下 8 个字段应全部为 `true`：

1. `numerically_converged`：数值计算完成且有限；
2. `tension_safe`：工作张力未超限；
3. `break_safe`：未达破断力限值；
4. `bend_safe`：最小弯曲半径合格；
5. `penetration_safe`：海床穿透量合格；
6. `mesh_safe`：网格质量合格；
7. `insertion_rest_length_safe`：节点插入无应力长度误差合格；
8. `curve_smooth`：报告曲线相邻步长的平滑性合格。

建议对单个工况的审查顺序为：`resolved_config.json` → `summary.json` → `time_history.csv` → 原始图与缆形/动画。批算验收以 `sweep_validation.json` 和 `failed_case_ids` 为准。

工况名 `VA_0p20_H_0p50_R_1p05` 表示 `v_A=0.20 m/s`、`h_A=0.50 m`、`v_out/v_A=1.05`。文件名中用 `p` 代替小数点；海流后缀中 `m0p10` 和 `p0p10` 分别表示 `-0.10 m/s` 和 `+0.10 m/s`。

## 工程使用边界

- 0.52 mm 缆的直径和线密度来自任务书；1.0 mm 缆的部分参数为典型值。
- `EA`、`EI`、阻力系数、附加质量系数和海床摩擦等仍含工程暂定值，需用厂家资料或试验数据标定。
- 配置中的 500 N 最低破断力与安全系数 3 对应约 166.67 N 许用工作张力；使用前应核对实际产品规格。
- 当前结果属于数值验证和方案分析，不构成产品认证或现场作业授权。
- 当前仓库未提供许可证文件；对外分发或开源前需由项目负责人补充授权条款。

