from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.simulation import run_simulation


def collect_overrides(args: argparse.Namespace) -> dict[str, object]:
    """
    Only override parameters that were explicitly provided via CLI.
    """
    overrides = {}

    cli_values = {
        "case_id": args.case_id,
        "v_A": args.v_A,
        "h_A": args.h_A,
        "v_out": args.v_out,
        "t_end": args.t_end,
        "dt": args.dt,
        "output_root": args.output_root,
    }

    for key, value in cli_values.items():
        if value is not None:
            overrides[key] = value

    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one dynamic cable-deployment case."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/cable_0p52mm.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument("--case-id", type=str, default=None)
    parser.add_argument("--v-A", type=float, default=None)
    parser.add_argument("--h-A", type=float, default=None)
    parser.add_argument("--v-out", type=float, default=None)
    parser.add_argument("--t-end", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument(
        "--no-raise",
        action="store_true",
        help="Save failure summary instead of re-raising exceptions",
    )

    args = parser.parse_args()

    # 只把用户真正显式指定的参数传进去
    overrides = collect_overrides(args)

    # 优先级：CLI > YAML > SimulationConfig defaults
    cfg = load_config(args.config, overrides)

    summary = run_simulation(cfg, raise_on_error=not args.no_raise)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
