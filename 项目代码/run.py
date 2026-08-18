from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.make_animation import create_animation
from src.config import SimulationConfig, load_config
from src.deployment import auv_trajectory, ramped_displacement
from src.geometry import bending_energy, bending_forces, discrete_curvature_radius
from src.seabed import Seabed
from src.simulation import run_simulation
from utils.plot_style import (
    PALETTE,
    add_panel_labels,
    apply_publication_style,
    export_figure,
    publication_subplots,
)


RESULTS = PROJECT_ROOT / "results"
FIGURES = PROJECT_ROOT / "figures"
FINAL_OUTPUT = PROJECT_ROOT / "outputs" / "final_validation"
TASK_BOOK = Path(
    "/Users/gerrilynn/Desktop/CUMCM/金沛霖—仿真任务/"
    "MGC-01-1.0-ETFE水下柔性光缆铺放仿真任务书.docx"
)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty result table: {path}")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_summary(summary: dict[str, Any]) -> bool:
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
    return all(summary.get(key) is True for key in required)


def _relative_difference(a: float, b: float, floor: float = 1.0e-14) -> float:
    return abs(a - b) / max(abs(a), abs(b), floor)


def build_buoyancy_matrix(cfg: SimulationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for water_density in (1000.0, 1025.0):
        displaced = water_density * math.pi * cfg.diameter**2 / 4.0
        for line_density in (0.00018, 0.00021, 0.00024):
            effective = line_density - displaced
            rows.append(
                {
                    "diameter_mm": 1000.0 * cfg.diameter,
                    "water": "fresh" if water_density == 1000.0 else "seawater",
                    "water_density_kg_m3": water_density,
                    "line_density_g_m": 1000.0 * line_density,
                    "displaced_mass_g_m": 1000.0 * displaced,
                    "effective_mass_g_m": 1000.0 * effective,
                    "effective_weight_mN_m": 1000.0 * effective * cfg.gravity,
                    "buoyancy_state": "positive" if effective < 0.0 else "negative",
                }
            )
    return rows


def bending_validation(cfg: SimulationConfig) -> list[dict[str, Any]]:
    position = np.array(
        [[0.0, 0.0], [0.35, 0.08], [0.71, 0.24], [1.02, 0.05]],
        dtype=float,
    )
    rest = np.array([0.36, 0.40, 0.41], dtype=float)
    force = bending_forces(position, rest, EI=0.03)
    numerical = np.zeros_like(position)
    step = 1.0e-7
    for node in range(position.shape[0]):
        for component in range(2):
            plus = position.copy()
            minus = position.copy()
            plus[node, component] += step
            minus[node, component] -= step
            numerical[node, component] = -(
                bending_energy(plus, rest, 0.03)
                - bending_energy(minus, rest, 0.03)
            ) / (2.0 * step)
    gradient_error = float(
        np.linalg.norm(force - numerical) / np.linalg.norm(numerical)
    )
    net_force = float(np.linalg.norm(np.sum(force, axis=0)))
    torque = float(
        abs(np.sum(position[:, 0] * force[:, 1] - position[:, 1] * force[:, 0]))
    )
    rows = [
        {"test": "energy_gradient", "value": gradient_error, "acceptance": 1.0e-5, "passed": gradient_error <= 1.0e-5},
        {"test": "net_force", "value": net_force, "acceptance": 1.0e-10, "passed": net_force <= 1.0e-10},
        {"test": "net_torque", "value": torque, "acceptance": 1.0e-10, "passed": torque <= 1.0e-10},
    ]
    exact_radius = 0.02
    for ds in (0.01, 0.005, 0.002, 0.001):
        delta = ds / exact_radius
        theta = np.arange(9, dtype=float) * delta
        circle = np.column_stack(
            (exact_radius * np.sin(theta), exact_radius * (1.0 - np.cos(theta)))
        )
        local_cfg = cfg.with_overrides(ds=ds)
        radius = discrete_curvature_radius(circle, local_cfg)[1:-1]
        error = float(np.max(np.abs(radius - exact_radius)) / exact_radius)
        rows.append(
            {
                "test": "constant_curvature_circle",
                "ds_m": ds,
                "value": error,
                "acceptance": 1.0e-10,
                "passed": error <= 1.0e-10,
            }
        )
    return rows


def run_convergence(base: SimulationConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_root = FINAL_OUTPUT / "convergence"
    grid_pairs = ((0.01, 0.00025), (0.005, 0.000125), (0.002, 0.00005), (0.001, 0.000025))
    cases: list[tuple[str, float, float]] = [
        ("grid", ds, dt) for ds, dt in grid_pairs
    ] + [
        ("time", 0.01, dt) for dt in (0.0005, 0.00025, 0.000125)
    ]
    rows: list[dict[str, Any]] = []
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
        summary = run_simulation(cfg, raise_on_error=False)
        summary.update({"study": study, "ds": ds, "dt": dt})
        rows.append(summary)

    grid = [row for row in rows if row["study"] == "grid"]
    all_cases_safe = all(_safe_summary(row) for row in rows)
    if not all_cases_safe:
        failed = [
            {
                "case_id": row.get("case_id"),
                "error": row.get("error"),
                "classification": row.get("classification"),
            }
            for row in rows
            if not _safe_summary(row)
        ]
        return rows, {
            "all_cases_safe": False,
            "failed_cases": failed,
            "passed": False,
        }
    fine, finest = grid[-2], grid[-1]
    comparisons: dict[str, Any] = {}
    for key, limit in (
        ("mean_layback", 0.05),
        ("mean_suspended_length", 0.05),
        ("min_bend_radius", 0.10),
    ):
        value = _relative_difference(float(fine[key]), float(finest[key]))
        comparisons[f"grid_{key}_relative_difference"] = value
        comparisons[f"grid_{key}_passed"] = value <= limit
    tension_abs = abs(float(fine["peak_active_tension"]) - float(finest["peak_active_tension"]))
    tension_rel = _relative_difference(
        float(fine["peak_active_tension"]), float(finest["peak_active_tension"])
    )
    tension_passed = tension_rel <= 0.05 or tension_abs <= 0.01 * base.T_max
    comparisons.update(
        {
            "grid_peak_tension_relative_difference": tension_rel,
            "grid_peak_tension_absolute_difference_N": tension_abs,
            "grid_peak_tension_passed": tension_passed,
            "all_cases_safe": all_cases_safe,
        }
    )
    comparisons["passed"] = all(
        value is True for key, value in comparisons.items() if key.endswith("_passed") or key == "all_cases_safe"
    )
    return rows, comparisons


def collect_insertion_audit(
    convergence_rows: list[dict[str, Any]], base: SimulationConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    finest = min(
        (row for row in convergence_rows if row["study"] == "grid"),
        key=lambda row: float(row["ds"]),
    )
    source = FINAL_OUTPUT / "convergence" / str(finest["case_id"]) / "insertion_events.csv"
    rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(source)]
    for index, row in enumerate(rows, start=1):
        row["event_index"] = index
        row["case_id"] = finest["case_id"]
        row["ds"] = finest["ds"]
    rest_error = max(abs(float(row["rest_length_error"])) for row in rows)
    tension_jump = max(abs(float(row["peak_tension_jump"])) for row in rows)
    minimum_ratio = min(float(row["minimum_standard_ratio_post_step"]) for row in rows)
    minimum_physical = min(
        float(row["minimum_physical_rest_length_after"]) for row in rows
    )
    ds = float(finest["ds"])
    metrics = {
        "event_count": len(rows),
        "max_rest_length_error_m": rest_error,
        "max_post_step_tension_jump_N": tension_jump,
        "minimum_post_step_mesh_ratio": minimum_ratio,
        "minimum_physical_rest_length_m": minimum_physical,
        "rest_length_passed": rest_error <= 1.0e-12,
        "tension_jump_passed": tension_jump <= base.maximum_insertion_tension_jump,
        "mesh_ratio_passed": minimum_ratio >= 0.90,
        "minimum_rest_length_passed": minimum_physical >= 0.5 * ds - 1.0e-12,
    }
    metrics["passed"] = all(value is True for key, value in metrics.items() if key.endswith("_passed"))
    return rows, metrics


def run_current_sweep(base: SimulationConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for velocity in (-0.20, -0.10, 0.0, 0.10, 0.20):
        case_id = f"current_{velocity:+.2f}".replace("+", "p").replace("-", "m").replace(".", "p")
        cfg = base.with_overrides(
            case_id=case_id,
            output_root=str(FINAL_OUTPUT / "current_sweep"),
            current_velocity_x=velocity,
            t_end=0.50,
            ramp_time=0.10,
            relaxation_time=0.30,
            ds=0.01,
            dt=0.00025,
            save_plots=True,
            save_snapshots=True,
        )
        cfg.validate()
        summary = run_simulation(cfg, raise_on_error=False)
        summary.update(
            {
                "current_velocity_x": velocity,
                "current_direction": "with AUV" if velocity > 0 else "against AUV" if velocity < 0 else "still water",
                "ds": cfg.ds,
                "dt": cfg.dt,
            }
        )
        rows.append(summary)
    return rows


def _proper_segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def cross(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
        first = q - p
        second = r - p
        return float(first[0] * second[1] - first[1] * second[0])

    ab_c = cross(a, b, c)
    ab_d = cross(a, b, d)
    cd_a = cross(c, d, a)
    cd_b = cross(c, d, b)
    tolerance = 1.0e-12
    return ab_c * ab_d < -tolerance and cd_a * cd_b < -tolerance


def audit_animation(case_dir: Path, gif_path: Path) -> dict[str, Any]:
    config_data = json.loads((case_dir / "resolved_config.json").read_text(encoding="utf-8"))
    cfg = SimulationConfig(**config_data)
    seabed = Seabed(cfg)
    data = np.load(case_dir / "node_snapshots.npz")
    times = np.asarray(data["time"], dtype=float)
    counts = np.asarray(data["node_count"], dtype=int)
    offsets = np.asarray(data["offsets"], dtype=int)
    position_data = np.asarray(data["position_data"], dtype=float)
    frames = [position_data[offsets[i]:offsets[i + 1]] for i in range(times.size)]
    rest_offsets = np.asarray(data["rest_offsets"], dtype=int)
    rest_data = np.asarray(data["rest_length_data"], dtype=float)
    rest_frames = [
        rest_data[rest_offsets[i]:rest_offsets[i + 1]]
        for i in range(times.size)
    ]

    anchor = frames[0][0].copy()
    max_anchor_drift = 0.0
    max_auv_error = 0.0
    max_outlet_error = 0.0
    max_penetration = 0.0
    self_intersections = 0
    max_frame_nearest_displacement = 0.0
    max_snapshot_length_error = 0.0
    minimum_frame_bend_radius = math.inf
    for index, (time, frame) in enumerate(zip(times, frames)):
        max_anchor_drift = max(max_anchor_drift, float(np.linalg.norm(frame[0] - anchor)))
        expected_auv, _ = auv_trajectory(float(time), cfg, seabed)
        max_auv_error = max(max_auv_error, float(np.linalg.norm(frame[-1] - expected_auv)))
        outlet = frame[-2] - frame[-1]
        outlet /= np.linalg.norm(outlet)
        dot = float(np.clip(np.dot(outlet, cfg.outlet_direction), -1.0, 1.0))
        error = 0.0 if 1.0 - dot <= 1.0e-12 else float(np.degrees(np.arccos(dot)))
        max_outlet_error = max(max_outlet_error, error)
        bed = seabed.height(frame[:, 0])
        max_penetration = max(max_penetration, float(np.max(np.maximum(bed - frame[:, 1], 0.0))))
        expected_length = cfg.initial_rest_length + ramped_displacement(
            cfg.v_out, float(time), cfg.ramp_time
        )
        max_snapshot_length_error = max(
            max_snapshot_length_error,
            abs(float(np.sum(rest_frames[index])) - expected_length),
        )
        radii = discrete_curvature_radius(frame, cfg)
        finite = radii[np.isfinite(radii)]
        if finite.size:
            minimum_frame_bend_radius = min(
                minimum_frame_bend_radius, float(np.min(finite))
            )
        for left in range(frame.shape[0] - 1):
            for right in range(left + 2, frame.shape[0] - 1):
                if _proper_segments_intersect(
                    frame[left], frame[left + 1], frame[right], frame[right + 1]
                ):
                    self_intersections += 1
        if index:
            previous = frames[index - 1]
            distances = np.linalg.norm(frame[:, None, :] - previous[None, :, :], axis=2)
            nearest = max(float(np.max(np.min(distances, axis=1))), float(np.max(np.min(distances, axis=0))))
            max_frame_nearest_displacement = max(max_frame_nearest_displacement, nearest)

    image = Image.open(gif_path)
    gif_frames = int(getattr(image, "n_frames", 1))
    summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    snapshot_cadence = float(np.max(np.diff(times))) if times.size > 1 else 0.0
    continuity_limit = max(
        0.04,
        4.0 * max(cfg.v_A, cfg.v_out) * max(snapshot_cadence, cfg.dt),
    )
    checks = {
        "time_monotonic": bool(np.all(np.diff(times) > 0.0)),
        "node_count_non_decreasing": bool(np.all(np.diff(counts) >= 0)),
        "anchor_fixed": max_anchor_drift <= 1.0e-10,
        "auv_trajectory_correct": max_auv_error <= 1.0e-10,
        "outlet_direction_correct": max_outlet_error <= 1.0e-6,
        "penetration_safe": max_penetration <= cfg.penetration_limit,
        "snapshot_length_conserved": max_snapshot_length_error <= 1.0e-9,
        "frame_bend_radius_safe": minimum_frame_bend_radius >= cfg.min_bend_radius,
        "no_proper_self_intersection": self_intersections == 0,
        "frame_continuity": max_frame_nearest_displacement <= continuity_limit,
        "snapshot_cadence_sufficient": snapshot_cadence <= 0.0500001,
        "gif_frame_count_matches": gif_frames == len(frames),
        "simulation_constraints_safe": _safe_summary(summary),
    }
    return {
        "case_id": cfg.case_id,
        "gif_path": str(gif_path),
        "gif_sha256": _sha256(gif_path),
        "gif_size_px": list(image.size),
        "gif_frames": gif_frames,
        "snapshot_frames": len(frames),
        "time_start_s": float(times[0]),
        "time_end_s": float(times[-1]),
        "max_anchor_drift_m": max_anchor_drift,
        "max_auv_position_error_m": max_auv_error,
        "max_outlet_direction_error_deg": max_outlet_error,
        "max_penetration_m": max_penetration,
        "proper_self_intersection_count": self_intersections,
        "max_frame_nearest_displacement_m": max_frame_nearest_displacement,
        "frame_continuity_limit_m": continuity_limit,
        "snapshot_cadence_s": snapshot_cadence,
        "max_snapshot_length_error_m": max_snapshot_length_error,
        "minimum_frame_bend_radius_m": minimum_frame_bend_radius,
        "checks": checks,
        "passed": all(checks.values()),
    }


def plot_results(
    buoyancy: list[dict[str, Any]],
    convergence: list[dict[str, Any]],
    insertion: list[dict[str, Any]],
    currents: list[dict[str, Any]],
    final_case_dir: Path,
) -> None:
    apply_publication_style(language="en", width="report")

    fig, ax = publication_subplots(1, 1, width="report", aspect=0.58)
    for water, color, marker in (("fresh", PALETTE["primary"], "o"), ("seawater", PALETTE["cyan"], "s")):
        rows = [row for row in buoyancy if row["water"] == water]
        ax.plot(
            [row["line_density_g_m"] for row in rows],
            [row["effective_weight_mN_m"] for row in rows],
            marker=marker,
            color=color,
            label=water,
        )
    ax.axhline(0.0, color=PALETTE["contrast"], linestyle="--", linewidth=1.0)
    ax.set(title="Effective cable weight", xlabel="Line density (g/m)", ylabel="Effective weight (mN/m)")
    ax.legend()
    export_figure(fig, FIGURES / "result_q1_buoyancy")
    plt.close(fig)

    grid = [row for row in convergence if row["study"] == "grid"]
    ds_mm = np.asarray([1000.0 * float(row["ds"]) for row in grid])
    fig, axes = publication_subplots(2, 2, width="report", aspect=0.78)
    series = (
        ("peak_active_tension", "Peak tension", "Tension (N)"),
        ("min_bend_radius", "Minimum bend radius", "Radius (m)"),
        ("mean_layback", "Mean Layback", "Layback (m)"),
        ("mean_suspended_length", "Mean suspended length", "Length (m)"),
    )
    for axis, (key, title, ylabel) in zip(axes.flat, series):
        axis.plot(ds_mm, [float(row[key]) for row in grid], marker="o", color=PALETTE["primary"])
        axis.set_xscale("log")
        axis.invert_xaxis()
        axis.set(title=title, xlabel="Grid size (mm)", ylabel=ylabel)
    add_panel_labels(axes.flat)
    export_figure(fig, FIGURES / "result_q2_convergence")
    plt.close(fig)

    event_index = np.arange(1, len(insertion) + 1)
    fig, axes = publication_subplots(1, 2, width="report", aspect=0.50)
    axes[0].plot(event_index, [1.0e3 * abs(float(row["peak_tension_jump"])) for row in insertion], marker="o", color=PALETTE["primary"])
    axes[0].set(title="Tension continuity", xlabel="Insertion event", ylabel="|T after - T before| (mN)")
    axes[1].plot(event_index, [float(row["minimum_standard_ratio_post_step"]) for row in insertion], marker="s", color=PALETTE["cyan"])
    axes[1].axhline(0.90, color=PALETTE["contrast"], linestyle="--", label="Acceptance")
    axes[1].set(title="Mesh quality", xlabel="Insertion event", ylabel="Minimum length ratio")
    axes[1].legend()
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "result_q3_insertion")
    plt.close(fig)

    velocities = np.asarray([float(row["current_velocity_x"]) for row in currents])
    fig, axes = publication_subplots(1, 3, width="report", aspect=0.42)
    for axis, key, title, ylabel in (
        (axes[0], "peak_active_tension", "Tail/active tension", "Peak tension (N)"),
        (axes[1], "mean_layback", "Layback response", "Layback (m)"),
        (axes[2], "mean_suspended_length", "Suspended length", "Length (m)"),
    ):
        axis.plot(velocities, [float(row[key]) for row in currents], marker="o", color=PALETTE["primary"])
        axis.axvline(0.0, color=PALETTE["neutral"], linewidth=0.8)
        axis.set(title=title, xlabel="Current U (m/s)", ylabel=ylabel)
    add_panel_labels(axes)
    export_figure(fig, FIGURES / "result_q4_current_response")
    plt.close(fig)

    data = np.load(final_case_dir / "node_snapshots.npz")
    offsets = data["offsets"]
    positions = data["position_data"]
    final = positions[offsets[-2]:offsets[-1]]
    config = SimulationConfig(**json.loads((final_case_dir / "resolved_config.json").read_text(encoding="utf-8")))
    fig, ax = publication_subplots(1, 1, width="report", aspect=0.48)
    ax.plot(final[:, 0], final[:, 1], "-", linewidth=1.8, color=PALETTE["primary"], label="Cable")
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=1.2, label="Seabed")
    guide = np.vstack((final[-1], final[-1] + 0.05 * config.outlet_direction))
    ax.plot(guide[:, 0], guide[:, 1], "--", color=PALETTE["contrast"], label="37.5 deg outlet")
    ax.scatter(final[-1, 0], final[-1, 1], marker="s", color=PALETTE["contrast"], zorder=3)
    ax.set(title="Final cable shape", xlabel="x (m)", ylabel="z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    export_figure(fig, FIGURES / "result_q5_final_shape")
    plt.close(fig)


def build_evidence(
    buoyancy: list[dict[str, Any]],
    bending: list[dict[str, Any]],
    convergence_metrics: dict[str, Any],
    insertion_metrics: dict[str, Any],
    currents: list[dict[str, Any]],
    animation: dict[str, Any],
) -> dict[str, Any]:
    nominal = [
        row
        for row in buoyancy
        if math.isclose(float(row["line_density_g_m"]), 0.21, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    nominal.sort(key=lambda row: str(row["water"]))
    if len(nominal) != 2:
        raise RuntimeError(f"Expected fresh/seawater nominal buoyancy rows, found {len(nominal)}")
    all_bending = all(bool(row["passed"]) for row in bending)
    all_currents = all(_safe_summary(row) for row in currents)
    return {
        "questions": {
            "q1": {
                "headline": "0.52 mm nominal cable is slightly positively buoyant in both fresh and seawater; the upper mass tolerance is negatively buoyant.",
                "key_results": [
                    {"name": "Fresh-water nominal effective mass", "value": nominal[0]["effective_mass_g_m"], "unit": "g/m", "source": "results/buoyancy_matrix.csv"},
                    {"name": "Seawater nominal effective mass", "value": nominal[1]["effective_mass_g_m"], "unit": "g/m", "source": "results/buoyancy_matrix.csv"},
                ],
                "validation": [{"check": "Six density/water combinations generated", "value": len(buoyancy), "acceptance": "6", "passed": len(buoyancy) == 6}],
                "figures": [{"path": "figures/result_q1_buoyancy.png", "finding": "The zero crossing lies inside the stated line-density tolerance."}],
                "interpretation": "Fresh/seawater density and manufacturing tolerance can reverse the direction of effective weight, so a single nominal-weight conclusion is invalid.",
                "robustness": "All two water densities and all three stated mass levels were evaluated without interpolation.",
                "limitations": ["EA, EI and the 0.52 mm tensile rating remain provisional until a product-specific datasheet or test is supplied."],
            },
            "q2": {
                "headline": "The energy-derived bend force passes gradient, force and torque checks; 2 mm and 1 mm grids agree on bend radius within the 10% criterion.",
                "key_results": [
                    {"name": "Fine-grid bend-radius relative difference", "value": convergence_metrics["grid_min_bend_radius_relative_difference"], "unit": "1", "source": "results/metrics.json"},
                    {"name": "Maximum bend-gradient validation error", "value": max(float(row["value"]) for row in bending if row["test"] == "energy_gradient"), "unit": "1", "source": "results/bending_validation.csv"},
                ],
                "validation": [
                    {"check": "Bending benchmark suite", "value": all_bending, "acceptance": "all true", "passed": all_bending},
                    {"check": "Grid and time convergence", "value": convergence_metrics["passed"], "acceptance": "true", "passed": convergence_metrics["passed"]},
                ],
                "figures": [{"path": "figures/result_q2_convergence.png", "finding": "Global geometry stabilizes rapidly and the 1 mm bend-radius result remains above 15D."}],
                "interpretation": "The previous empirical bend-force objection is resolved at implementation level, while the millimetre-grid experiment quantifies the remaining discretization sensitivity.",
                "robustness": "The grid study pairs 10/5/2/1 mm with synchronized time steps and separately varies time step at 10 mm.",
                "limitations": ["The model remains two-dimensional and does not represent torsion, self-contact friction or optical attenuation."],
            },
            "q3": {
                "headline": "Node insertion conserves material length and same-step redistribution removes short-element collapse without a tension impulse.",
                "key_results": [
                    {"name": "Maximum insertion length error", "value": insertion_metrics["max_rest_length_error_m"], "unit": "m", "source": "results/insertion_audit.csv"},
                    {"name": "Minimum post-step mesh ratio", "value": insertion_metrics["minimum_post_step_mesh_ratio"], "unit": "1", "source": "results/insertion_audit.csv"},
                ],
                "validation": [{"check": "Insertion audit", "value": insertion_metrics["passed"], "acceptance": "true", "passed": insertion_metrics["passed"]}],
                "figures": [{"path": "figures/result_q3_insertion.png", "finding": "All insertion events return above the 0.90 mesh-quality threshold in the same time step."}],
                "interpretation": "Insertion and redistribution are now explicitly auditable rather than inferred only from the final node count.",
                "robustness": "The audit uses the most demanding 1 mm case, which has the largest number of insertion events.",
                "limitations": ["Redistribution preserves the current polyline and interpolates velocity; it is a numerical mesh operation rather than a material-contact model."],
            },
            "q4": {
                "headline": "All five prescribed uniform-current cases satisfy the hard constraints; opposing current produces the largest tensile response.",
                "key_results": [
                    {"name": "Maximum current-sweep tension", "value": max(float(row["peak_active_tension"]) for row in currents), "unit": "N", "source": "results/current_sweep.csv"},
                    {"name": "Minimum current-sweep bend radius", "value": min(float(row["min_bend_radius"]) for row in currents), "unit": "m", "source": "results/current_sweep.csv"},
                ],
                "validation": [{"check": "Five-current hard constraints", "value": all_currents, "acceptance": "all true", "passed": all_currents}],
                "figures": [{"path": "figures/result_q4_current_response.png", "finding": "Tension and suspended length respond to current direction; Layback is unchanged while the TDP remains at the anchor."}],
                "interpretation": "Hydrodynamic force uses cable velocity relative to the water. At this early representative interval the positive-buoyancy cable has not formed a new seabed contact, so Layback does not yet separate by current.",
                "robustness": "The experiment uses exactly the five requested velocities and does not expand the 60-case payout matrix.",
                "limitations": ["Long-duration current-dependent TDP migration requires product-calibrated buoyancy and seabed parameters."],
            },
            "q5": {
                "headline": "The regenerated GIF follows the prescribed AUV path, fixes the anchor, maintains 37.5 degrees, and contains no proper cable self-intersection.",
                "key_results": [
                    {"name": "Maximum outlet error in animation", "value": animation["max_outlet_direction_error_deg"], "unit": "deg", "source": "results/animation_audit.json"},
                    {"name": "Maximum frame-to-frame shape displacement", "value": animation["max_frame_nearest_displacement_m"], "unit": "m", "source": "results/animation_audit.json"},
                ],
                "validation": [{"check": "Mechanical and visual animation audit", "value": animation["passed"], "acceptance": "true", "passed": animation["passed"]}],
                "figures": [{"path": "figures/result_q5_final_shape.png", "finding": "The final cable curve is continuous, remains above the seabed, and is tangent to the displayed 37.5-degree guide."}],
                "interpretation": "The old free-outlet animation has been replaced by one generated from the constrained, audited simulation snapshots.",
                "robustness": "Every stored frame is checked for boundary drift, outlet direction, penetration, proper self-intersection and temporal continuity.",
                "limitations": ["Visual plausibility is conditional on the provisional 0.52 mm stiffness and strength parameters."],
            },
        }
    }


def run_smoke() -> int:
    base = load_config(PROJECT_ROOT / "configs" / "cable_0p52mm.yaml")
    cfg = base.with_overrides(
        case_id="p1_smoke",
        output_root=str(FINAL_OUTPUT),
        t_end=0.05,
        dt=0.00025,
        save_plots=False,
        save_snapshots=False,
    )
    summary = run_simulation(cfg, raise_on_error=False)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if _safe_summary(summary) else 1


def run_full() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)
    base = load_config(PROJECT_ROOT / "configs" / "cable_0p52mm.yaml")

    buoyancy = build_buoyancy_matrix(base)
    _write_csv(RESULTS / "buoyancy_matrix.csv", buoyancy)
    parameter_audit = {
        "config": str(PROJECT_ROOT / "configs" / "cable_0p52mm.yaml"),
        "config_sha256": _sha256(PROJECT_ROOT / "configs" / "cable_0p52mm.yaml"),
        "parameter_status": "provisional",
        "outlet_angle_deg": base.outlet_angle_deg,
        "outlet_enforced": base.enforce_outlet_direction,
        "allowable_tension_N": base.T_max,
        "dynamic_bend_radius_m": base.min_bend_radius,
        "low_load_bend_radius_m": base.low_load_bend_radius,
    }
    (RESULTS / "parameter_audit.json").write_text(
        json.dumps(parameter_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    bending = bending_validation(base)
    _write_csv(RESULTS / "bending_validation.csv", bending)
    if not all(bool(row["passed"]) for row in bending):
        raise RuntimeError("Bending validation failed")

    convergence, convergence_metrics = run_convergence(base)
    _write_csv(RESULTS / "convergence_results.csv", convergence)
    if not convergence_metrics["passed"]:
        raise RuntimeError(f"Convergence validation failed: {convergence_metrics}")

    insertion, insertion_metrics = collect_insertion_audit(convergence, base)
    _write_csv(RESULTS / "insertion_audit.csv", insertion)
    if not insertion_metrics["passed"]:
        raise RuntimeError(f"Insertion validation failed: {insertion_metrics}")

    currents = run_current_sweep(base)
    _write_csv(RESULTS / "current_sweep.csv", currents)
    if not all(_safe_summary(row) for row in currents):
        raise RuntimeError("At least one uniform-current case violates a hard constraint")

    final_cfg = base.with_overrides(
        case_id="final_0p52mm_freshwater_upper_mass",
        output_root=str(FINAL_OUTPUT),
        line_density=0.00024,
        t_end=8.0,
        dt=0.00025,
        ds=0.01,
        snapshot_interval=0.05,
        save_plots=True,
        save_snapshots=True,
    )
    final_summary = run_simulation(final_cfg, raise_on_error=False)
    if not _safe_summary(final_summary):
        raise RuntimeError(f"Final animation case violates a hard constraint: {final_summary}")
    final_case_dir = FINAL_OUTPUT / final_cfg.case_id
    local_gif = FINAL_OUTPUT / "demo_animation.gif"
    create_animation(final_case_dir / "node_snapshots.npz", local_gif, fps=10)
    animation = audit_animation(final_case_dir, local_gif)
    (RESULTS / "animation_audit.json").write_text(
        json.dumps(animation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not animation["passed"]:
        raise RuntimeError(f"Animation audit failed: {animation}")

    plot_results(buoyancy, convergence, insertion, currents, final_case_dir)

    numerical_rows = [
        {"category": "final", "metric": key, "value": value}
        for key, value in final_summary.items()
        if isinstance(value, (int, float, bool)) and value is not None
    ]
    _write_csv(RESULTS / "numerical_results.csv", numerical_rows)
    metrics = {
        "all_constraints_passed": True,
        "parameter_audit": parameter_audit,
        "convergence": convergence_metrics,
        "insertion": insertion_metrics,
        "current_all_safe": all(_safe_summary(row) for row in currents),
        "final_summary": final_summary,
        "animation": animation,
    }
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    evidence = build_evidence(
        buoyancy,
        bending,
        convergence_metrics,
        insertion_metrics,
        currents,
        animation,
    )
    (RESULTS / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    analysis_lines = [
        "# 结果分析",
        "",
        "- 正式模型采用任务书指定的 1000 kg/m³ 淡水与 500 N 最低破断力。",
        f"- 最终算例原始峰值张力：{final_summary['peak_active_tension']:.6g} N；安全工作限值：{base.T_max:.6g} N。",
        f"- 最小动态弯曲半径：{final_summary['min_bend_radius']:.6g} m；限值：{base.min_bend_radius:.6g} m。",
        f"- 插入后最大张力跳变：{final_summary['max_insertion_tension_jump_N']:.6g} N。",
        f"- 出口方向最大误差：{final_summary['max_outlet_direction_error_deg']:.6g}°。",
        f"- 标准张力图显示 {base.tension_filter_window:g} s 工程趋势；原始曲线和原始峰值独立保留，安全判定不使用滤波值。",
        "",
        "结论：本流水线只有在力学、网格、曲线平滑和 GIF 逐帧检查全部通过时才会完成。",
    ]
    (RESULTS / "结果分析.md").write_text("\n".join(analysis_lines) + "\n", encoding="utf-8")
    manifest_outputs = [
        RESULTS / "numerical_results.csv",
        RESULTS / "metrics.json",
        RESULTS / "结果分析.md",
        RESULTS / "current_sweep.csv",
        RESULTS / "convergence_results.csv",
        RESULTS / "animation_audit.json",
        FIGURES / "result_q1_buoyancy.svg",
        FIGURES / "result_q2_convergence.svg",
        FIGURES / "result_q3_insertion.svg",
        FIGURES / "result_q4_current_response.svg",
        FIGURES / "result_q5_final_shape.svg",
        local_gif,
    ]
    manifest = {
        "task_book": str(TASK_BOOK),
        "task_book_sha256": _sha256(TASK_BOOK),
        "command": ".venv/bin/python run.py",
        "parameters": {
            "grid_mm": [10, 5, 2, 1],
            "current_m_s": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "animation_t_end_s": 8.0,
        },
        "outputs": [
            {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": _sha256(path)}
            for path in manifest_outputs
            if path.exists()
        ],
    }
    (RESULTS / "复现清单.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete revised cable validation workflow.")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    return run_smoke() if args.smoke_test else run_full()


if __name__ == "__main__":
    raise SystemExit(main())
