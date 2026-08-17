from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from src.config import SimulationConfig
from src.seabed import Seabed


def create_animation(
    snapshot_file: str | Path,
    output_file: str | Path | None = None,
    fps: int = 10,
) -> Path:
    source = Path(snapshot_file)
    data = np.load(source)
    times = data["time"]
    offsets = data["offsets"]
    positions = data["position_data"]
    tdp = data["tdp_index"]
    frames = [positions[offsets[i]:offsets[i + 1]] for i in range(len(times))]

    output = Path(output_file) if output_file else source.with_name("animation.gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = source.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    cfg = SimulationConfig(**config) if config else SimulationConfig()
    seabed = Seabed(cfg)
    angle = float(config.get("outlet_angle_deg", 37.5))
    direction = np.array(
        [-np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        dtype=float,
    )
    diameter_mm = 1000.0 * float(config.get("diameter", 0.0))

    all_x = positions[:, 0]
    all_z = positions[:, 1]
    margin_x = max(0.05, 0.05 * max(np.ptp(all_x), 1.0))
    margin_z = max(0.05, 0.05 * max(np.ptp(all_z), 1.0))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    line, = ax.plot([], [], "o-", color="#557FA8", markersize=2.2, linewidth=1.4, label="Cable")
    tdp_point = ax.scatter([], [], marker="x", s=55, color="#F08A3C", label="TDP")
    auv_point = ax.scatter([], [], marker="s", s=42, color="#E06060", label="AUV outlet")
    anchor_point = ax.scatter([], [], marker="o", s=35, color="#333333", label="Anchor")
    guide_line, = ax.plot([], [], "--", color="#E06060", linewidth=1.1, label=f"Outlet {angle:g}°")
    title = ax.set_title("")
    x_min = float(np.min(all_x) - margin_x)
    x_max = float(np.max(all_x) + margin_x)
    bed_x = np.linspace(x_min, x_max, 600)
    bed_z = seabed.height(bed_x)
    ax.plot(bed_x, bed_z, color="#333333", linewidth=1.5, label="Seabed")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(
        float(min(np.min(bed_z) - margin_z, np.min(all_z) - margin_z)),
        float(max(np.max(bed_z) + margin_z, np.max(all_z) + margin_z)),
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper left", fontsize=8, ncol=2)

    guide_length = max(0.04, 4.0 * float(config.get("ds", 0.01)))

    def update(index: int):
        p = frames[index]
        line.set_data(p[:, 0], p[:, 1])
        idx = min(int(tdp[index]), p.shape[0] - 1)
        tdp_point.set_offsets(p[idx:idx + 1])
        auv_point.set_offsets(p[-1:])
        anchor_point.set_offsets(p[:1])
        guide = np.vstack((p[-1], p[-1] + guide_length * direction))
        guide_line.set_data(guide[:, 0], guide[:, 1])
        cable_label = f"{diameter_mm:g} mm cable | " if diameter_mm > 0.0 else ""
        title.set_text(f"{cable_label}t = {times[index]:.2f} s")
        return line, tdp_point, auv_point, anchor_point, guide_line, title

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
    )
    if output.suffix.lower() == ".gif":
        animation.save(output, writer=PillowWriter(fps=fps), dpi=100)
    else:
        animation.save(output, fps=fps, dpi=100)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a GIF/MP4 from node_snapshots.npz.")
    parser.add_argument("snapshot_file")
    parser.add_argument("--output")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    output = create_animation(args.snapshot_file, args.output, args.fps)
    print(output)


if __name__ == "__main__":
    main()
