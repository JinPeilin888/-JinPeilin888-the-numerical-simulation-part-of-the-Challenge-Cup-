import csv
from pathlib import Path

import numpy as np

from src.config import SimulationConfig
from src.simulation import run_simulation


def test_short_simulation_writes_outputs(tmp_path: Path):
    cfg = SimulationConfig(
        case_id="smoke",
        v_A=0.05,
        h_A=0.15,
        v_out=0.0525,
        t_end=0.05,
        dt=0.01,
        auto_substep=False,
        ramp_time=0.02,
        ds=0.05,
        EA=0.1,
        relaxation_time=0.01,
        output_root=str(tmp_path),
        save_plots=False,
        snapshot_interval=0.01,
        maximum_reported_tension_step=1.0,
    )
    summary = run_simulation(cfg)
    assert summary["numerically_converged"] is True
    assert summary["max_outlet_direction_error_deg"] < 1.0e-6
    assert summary["curve_smooth"] is True
    assert summary["insertion_rest_length_safe"] is True
    assert (tmp_path / "smoke" / "time_history.csv").exists()
    assert (tmp_path / "smoke" / "summary.json").exists()
    with (tmp_path / "smoke" / "time_history.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    outlet_dot = np.asarray([float(row["outlet_direction_dot"]) for row in rows], dtype=float)
    outlet_err = np.asarray([float(row["outlet_direction_error_deg"]) for row in rows], dtype=float)
    assert np.all(outlet_dot > 1.0 - 1.0e-8)
    assert np.max(outlet_err) < 1.0e-6
    assert "payout_reservoir_tension" in rows[0]
    assert "filtered_tail_tension" in rows[0]
    assert "filtered_max_active_tension" in rows[0]
