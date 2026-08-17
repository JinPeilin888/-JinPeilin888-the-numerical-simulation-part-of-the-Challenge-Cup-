from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from .config import SimulationConfig
from .metrics import StepMetrics
from .seabed import Seabed
from .state import CableState


class Recorder:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.rows: list[dict[str, Any]] = []
        self.snapshot_times: list[float] = []
        self.snapshot_positions: list[np.ndarray] = []
        self.snapshot_rest_lengths: list[np.ndarray] = []
        self.snapshot_tdp: list[int] = []
        self._next_snapshot_time = 0.0
        self.first_land_time: float | None = None
        self.first_land_x: float | None = None
        self.insertion_events: list[dict[str, Any]] = []
        self.remesh_events: list[dict[str, Any]] = []

    def record_insertion_event(self, **event: Any) -> None:
        self.insertion_events.append(dict(event))

    def complete_last_insertion_event(self, **event: Any) -> None:
        if self.insertion_events:
            self.insertion_events[-1].update(event)

    def record_remesh_event(self, **event: Any) -> None:
        self.remesh_events.append(dict(event))

    def append(self, metrics: StepMetrics, state: CableState) -> None:
        self.rows.append(metrics.to_row())
        if self.first_land_time is None and metrics.has_new_landing:
            self.first_land_time = metrics.time
            self.first_land_x = metrics.new_landing_x
        if self.cfg.save_snapshots and metrics.time + 1.0e-12 >= self._next_snapshot_time:
            self.snapshot_times.append(metrics.time)
            self.snapshot_positions.append(state.position.copy())
            self.snapshot_rest_lengths.append(state.rest_length.copy())
            self.snapshot_tdp.append(metrics.tdp_index)
            self._next_snapshot_time += self.cfg.snapshot_interval

    def _write_history(self, output_dir: Path) -> None:
        if not self.rows or not self.cfg.save_history:
            (output_dir / "time_history.csv").unlink(missing_ok=True)
            return
        filtered_tail, filtered_active = self._filtered_tensions()
        rows = []
        for index, source in enumerate(self.rows):
            row = dict(source)
            row["filtered_tail_tension"] = float(filtered_tail[index])
            row["filtered_max_active_tension"] = float(filtered_active[index])
            rows.append(row)
        columns = list(rows[0].keys())
        with (output_dir / "time_history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def _write_snapshots(self, output_dir: Path) -> None:
        if not self.snapshot_positions:
            (output_dir / "node_snapshots.npz").unlink(missing_ok=True)
            return
        counts = np.asarray([p.shape[0] for p in self.snapshot_positions], dtype=np.int64)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        position_data = np.concatenate(self.snapshot_positions, axis=0)
        rest_counts = np.asarray(
            [value.shape[0] for value in self.snapshot_rest_lengths], dtype=np.int64
        )
        rest_offsets = np.concatenate(([0], np.cumsum(rest_counts)))
        rest_length_data = np.concatenate(self.snapshot_rest_lengths)
        np.savez_compressed(
            output_dir / "node_snapshots.npz",
            time=np.asarray(self.snapshot_times, dtype=float),
            node_count=counts,
            offsets=offsets,
            position_data=position_data,
            rest_count=rest_counts,
            rest_offsets=rest_offsets,
            rest_length_data=rest_length_data,
            tdp_index=np.asarray(self.snapshot_tdp, dtype=np.int64),
        )

    @staticmethod
    def _write_event_table(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            path.unlink(missing_ok=True)
            return
        columns: list[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _finite_or_none(value: float | None) -> float | None:
        if value is None:
            return None
        return float(value) if math.isfinite(float(value)) else None

    def _filtered_tensions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return zero-phase engineering trends without changing raw peaks.

        The configured 0.5 s window is shorter than the seconds-scale
        deployment response, but longer than the node-insertion cadence. A
        median stage removes narrow mesh/contact pulses, then Savitzky-Golay
        smoothing preserves the slow trend. Raw samples remain available for
        peak safety checks and diagnostics.
        """
        if not self.rows:
            empty = np.asarray([], dtype=float)
            return empty, empty
        time = np.asarray([float(row["time"]) for row in self.rows])
        tail = np.asarray([float(row["tail_tension"]) for row in self.rows])
        active = np.asarray(
            [float(row["max_active_tension"]) for row in self.rows]
        )
        if time.size < 5:
            return tail.copy(), active.copy()
        dt = float(np.median(np.diff(time)))
        if not math.isfinite(dt) or dt <= 0.0:
            return tail.copy(), active.copy()
        window = max(5, int(round(self.cfg.tension_filter_window / dt)))
        if window % 2 == 0:
            window += 1
        if window > time.size:
            window = time.size if time.size % 2 == 1 else time.size - 1
        if window < 5:
            return tail.copy(), active.copy()
        filtered_tail = savgol_filter(
            median_filter(tail, size=window, mode="nearest"),
            window,
            3,
            mode="interp",
        )
        filtered_active = savgol_filter(
            median_filter(active, size=window, mode="nearest"),
            window,
            3,
            mode="interp",
        )
        return np.maximum(filtered_tail, 0.0), np.maximum(filtered_active, 0.0)

    def build_summary(self, converged: bool, error: str | None = None) -> dict[str, Any]:
        if not self.rows:
            return {
                "case_id": self.cfg.case_id,
                "numerically_converged": False,
                "error": error or "No output rows were recorded",
            }
        tail = np.asarray([float(r["tail_tension"]) for r in self.rows])
        active = np.asarray([float(r["max_active_tension"]) for r in self.rows])
        filtered_tail, filtered_active = self._filtered_tensions()
        peak_index = int(np.argmax(active))
        radius = np.asarray([float(r["min_bend_radius"]) for r in self.rows])
        layback = np.asarray([float(r["layback"]) for r in self.rows])
        suspended = np.asarray([float(r["suspended_length"]) for r in self.rows])
        penetration = np.asarray([float(r["max_penetration"]) for r in self.rows])
        slack = np.asarray([float(r["slack_ratio"]) for r in self.rows])
        conservation = np.asarray([abs(float(r["length_conservation_error"])) for r in self.rows])
        segment_ratio = np.asarray(
            [float(r["min_standard_segment_ratio"]) for r in self.rows]
        )
        axial_strain = np.asarray([float(r["max_axial_strain"]) for r in self.rows])
        compression_strain = np.asarray([float(r["min_axial_strain"]) for r in self.rows])
        outlet_error_deg = np.asarray([float(r["outlet_direction_error_deg"]) for r in self.rows])
        outlet_radius = np.asarray([float(r["min_bend_radius_outlet"]) for r in self.rows])
        tdp_radius = np.asarray([float(r["min_bend_radius_tdp"]) for r in self.rows])
        free_radius = np.asarray([float(r["min_bend_radius_free"]) for r in self.rows])
        steady_start = max(0, int((1.0 - self.cfg.steady_fraction) * tail.size))
        finite_radius = radius[np.isfinite(radius)]
        min_radius = float(np.min(finite_radius)) if finite_radius.size else math.inf
        peak_active = float(np.max(active))
        tension_safe = peak_active < self.cfg.T_max
        break_safe = peak_active < self.cfg.break_force
        bend_safe = min_radius >= self.cfg.min_bend_radius
        penetration_safe = float(np.max(penetration)) <= self.cfg.penetration_limit
        mesh_safe = float(np.min(segment_ratio)) >= self.cfg.minimum_resolved_segment_ratio
        filtered_step = max(
            float(np.max(np.abs(np.diff(filtered_tail))))
            if filtered_tail.size > 1 else 0.0,
            float(np.max(np.abs(np.diff(filtered_active))))
            if filtered_active.size > 1 else 0.0,
        )
        insertion_jump = max(
            (
                abs(float(event.get("peak_tension_jump", 0.0)))
                for event in self.insertion_events
            ),
            default=0.0,
        )
        minimum_inserted_rest_ratio = min(
            (
                float(event.get("minimum_physical_rest_length_after", self.cfg.ds))
                / self.cfg.ds
                for event in self.insertion_events
            ),
            default=1.0,
        )
        insertion_rest_length_safe = minimum_inserted_rest_ratio >= 0.5 - 1.0e-12
        insertion_smooth = (
            insertion_jump <= self.cfg.maximum_insertion_tension_jump
        )
        reported_curve_smooth = (
            filtered_step <= self.cfg.maximum_reported_tension_step
        )
        curve_smooth = insertion_smooth and reported_curve_smooth

        if not converged:
            classification = "numerical_failure"
        elif (
            not tension_safe
            or not bend_safe
            or not penetration_safe
            or not mesh_safe
            or not insertion_rest_length_safe
        ):
            classification = "rejected_hard_limit"
        elif not curve_smooth:
            classification = "rejected_curve_smoothness"
        elif float(np.max(slack)) > 0.80:
            classification = "warning_high_slack"
        else:
            classification = "acceptable_mvp"

        ratio = self.cfg.v_out / self.cfg.v_A if self.cfg.v_A > 0.0 else None
        return {
            "case_id": self.cfg.case_id,
            "v_A": self.cfg.v_A,
            "h_A": self.cfg.h_A,
            "v_out_ratio": ratio,
            "v_out": self.cfg.v_out,
            "peak_tail_tension": float(np.max(tail)),
            "tail_tension_p95": float(np.percentile(tail, 95)),
            "peak_active_tension": peak_active,
            "peak_tension_time": float(self.rows[peak_index]["time"]),
            "peak_tension_segment_index": int(
                self.rows[peak_index]["max_tension_segment_index"]
            ),
            "steady_tail_tension_mean": float(np.mean(tail[steady_start:])),
            "filtered_peak_tail_tension": float(np.max(filtered_tail)),
            "filtered_peak_active_tension": float(np.max(filtered_active)),
            "tension_filter_window_s": self.cfg.tension_filter_window,
            "max_filtered_tension_step_N": filtered_step,
            "max_insertion_tension_jump_N": insertion_jump,
            "minimum_inserted_rest_length_ratio": minimum_inserted_rest_ratio,
            "min_bend_radius": self._finite_or_none(min_radius),
            "min_bend_radius_outlet": self._finite_or_none(
                float(np.min(outlet_radius[np.isfinite(outlet_radius)]))
                if np.any(np.isfinite(outlet_radius)) else math.inf
            ),
            "min_bend_radius_tdp": self._finite_or_none(
                float(np.min(tdp_radius[np.isfinite(tdp_radius)]))
                if np.any(np.isfinite(tdp_radius)) else math.inf
            ),
            "min_bend_radius_free": self._finite_or_none(
                float(np.min(free_radius[np.isfinite(free_radius)]))
                if np.any(np.isfinite(free_radius)) else math.inf
            ),
            "low_load_bend_radius_reference": self.cfg.low_load_bend_radius,
            "mean_layback": float(np.mean(layback[steady_start:])),
            "final_layback": float(layback[-1]),
            "mean_suspended_length": float(np.mean(suspended[steady_start:])),
            "first_land_time": self.first_land_time,
            "first_land_x": self.first_land_x,
            "max_penetration": float(np.max(penetration)),
            "max_slack_ratio": float(np.max(slack)),
            "max_length_conservation_error": float(np.max(conservation)),
            "min_standard_segment_ratio": float(np.min(segment_ratio)),
            "max_axial_strain": float(np.max(axial_strain)),
            "min_axial_strain": float(np.min(compression_strain)),
            "max_outlet_direction_error_deg": float(np.max(outlet_error_deg)),
            "insertion_event_count": len(self.insertion_events),
            "remesh_event_count": len(self.remesh_events),
            "tension_safe": bool(tension_safe),
            "break_safe": bool(break_safe),
            "bend_safe": bool(bend_safe),
            "penetration_safe": bool(penetration_safe),
            "mesh_safe": bool(mesh_safe),
            "insertion_rest_length_safe": bool(insertion_rest_length_safe),
            "insertion_smooth": bool(insertion_smooth),
            "reported_curve_smooth": bool(reported_curve_smooth),
            "curve_smooth": bool(curve_smooth),
            "numerically_converged": bool(converged),
            "classification": classification,
            "error": error,
        }

    def _plot_time_series(self, output_dir: Path) -> None:
        if not self.rows:
            return
        t = np.asarray([r["time"] for r in self.rows], dtype=float)
        filtered_tail, filtered_active = self._filtered_tensions()
        series = {
            "tail_tension.png": ("tail_tension", "Tail tension (N)"),
            "max_tension.png": ("max_active_tension", "Maximum active tension (N)"),
            "suspended_length.png": ("suspended_length", "Suspended length (m)"),
            "layback.png": ("layback", "Layback (m)"),
            "min_bend_radius.png": ("min_bend_radius", "Minimum bend radius (m)"),
        }
        for filename, (key, ylabel) in series.items():
            y = np.asarray([r[key] for r in self.rows], dtype=float)
            y[~np.isfinite(y)] = np.nan
            fig, ax = plt.subplots(figsize=(7.2, 4.2))
            if key == "min_bend_radius":
                finite = y[np.isfinite(y) & (y > 0.0)]
                display = y.copy()
                if finite.size:
                    display_cap = float(np.percentile(finite, 99.5))
                    display = np.minimum(display, display_cap)
                ax.plot(t, display)
                ax.set_yscale("log")
                ax.axhline(
                    self.cfg.min_bend_radius,
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.2,
                    label="Allowed minimum",
                )
                ax.legend()
                ylabel = "Minimum bend radius (m, log scale)"
            else:
                if key in {"tail_tension", "max_active_tension"}:
                    filtered = (
                        filtered_tail
                        if key == "tail_tension"
                        else filtered_active
                    )
                    ax.plot(
                        t,
                        filtered,
                        linewidth=1.6,
                        label=f"{self.cfg.tension_filter_window:g} s trend",
                    )
                    ax.legend()
                    raw_name = (
                        "tail_tension_raw.png"
                        if key == "tail_tension"
                        else "max_tension_raw.png"
                    )
                    raw_fig, raw_ax = plt.subplots(figsize=(7.2, 4.2))
                    raw_ax.plot(t, y, linewidth=0.8)
                    raw_ax.set_xlabel("Time (s)")
                    raw_ax.set_ylabel(ylabel.replace("Maximum active", "Raw maximum active"))
                    raw_ax.grid(True, alpha=0.3)
                    raw_fig.tight_layout()
                    raw_fig.savefig(output_dir / raw_name, dpi=180)
                    plt.close(raw_fig)
                else:
                    ax.plot(t, y)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / filename, dpi=180)
            plt.close(fig)

    def _plot_final_shape(self, output_dir: Path, state: CableState, seabed: Seabed) -> None:
        x = state.position[:, 0]
        margin = max(self.cfg.ds, 0.05)
        xx = np.linspace(min(0.0, float(np.min(x))) - margin, float(np.max(x)) + margin, 400)
        zz = seabed.height(xx)
        fig, ax = plt.subplots(figsize=(8.0, 4.5))
        ax.plot(xx, zz, linewidth=2.0, label="Seabed")
        ax.plot(x, state.position[:, 1], "o-", markersize=2.5, linewidth=1.2, label="Cable")
        if self.rows:
            idx = int(self.rows[-1]["tdp_index"])
            ax.scatter([state.position[idx, 0]], [state.position[idx, 1]], marker="x", s=60, label="TDP")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "cable_shape.png", dpi=180)
        plt.close(fig)

    def finalize(
        self,
        state: CableState,
        seabed: Seabed,
        converged: bool = True,
        error: str | None = None,
    ) -> dict[str, Any]:
        output_dir = self.cfg.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        if converged:
            (output_dir / "error_traceback.txt").unlink(missing_ok=True)
        self._write_history(output_dir)
        self._write_snapshots(output_dir)
        self._write_event_table(output_dir / "insertion_events.csv", self.insertion_events)
        self._write_event_table(output_dir / "remesh_events.csv", self.remesh_events)
        summary = self.build_summary(converged=converged, error=error)
        with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
            json.dump(self.cfg.to_dict(), handle, ensure_ascii=False, indent=2)
        if self.cfg.save_plots and self.rows:
            self._plot_time_series(output_dir)
            self._plot_final_shape(output_dir, state, seabed)
        elif not self.cfg.save_plots:
            for filename in (
                "tail_tension.png",
                "tail_tension_raw.png",
                "max_tension.png",
                "max_tension_raw.png",
                "suspended_length.png",
                "layback.png",
                "min_bend_radius.png",
                "cable_shape.png",
            ):
                (output_dir / filename).unlink(missing_ok=True)
        return summary
