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

from src.catenary import solve_from_horizontal_tension, solve_from_suspended_length
from src.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the analytical catenary benchmark.")
    parser.add_argument("--config", default="configs/cable_1p0mm.yaml")
    parser.add_argument("--horizontal-tension", type=float, default=0.01)
    parser.add_argument("--suspended-length", type=float)
    parser.add_argument("--output", default="outputs/catenary_validation")
    args = parser.parse_args()

    cfg = load_config(args.config)
    q = cfg.effective_weight_per_length
    if args.suspended_length is None:
        solution = solve_from_horizontal_tension(args.horizontal_tension, q, cfg.h_A)
    else:
        solution = solve_from_suspended_length(args.suspended_length, q, cfg.h_A)
    sample = solution.sample()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sample["x"], sample["z"])
    ax.set_xlabel("Layback coordinate x (m)")
    ax.set_ylabel("Height z (m)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "analytical_catenary.png", dpi=180)
    plt.close(fig)

    payload = {
        "horizontal_tension": solution.horizontal_tension,
        "effective_weight": solution.effective_weight,
        "height": solution.height,
        "a": solution.a,
        "layback": solution.layback,
        "suspended_length": solution.suspended_length,
        "tail_tension": solution.tail_tension,
        "minimum_radius": solution.minimum_radius,
    }
    with (output / "analytical_catenary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
