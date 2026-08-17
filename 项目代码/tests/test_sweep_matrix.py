import csv
import json
from pathlib import Path

import pytest

from scripts.run_sweep import _case_config, _load_completed, _validate_cases


def _load_cases() -> list[dict[str, str]]:
    path = Path(__file__).resolve().parents[1] / "configs" / "cases.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_prescribed_matrix_contains_exactly_60_unique_cases():
    rows = _load_cases()
    _validate_cases(rows)
    assert len(rows) == 60


def test_prescribed_matrix_rejects_duplicate_or_missing_case():
    rows = _load_cases()
    rows[-1] = dict(rows[0])
    with pytest.raises(ValueError, match="exactly 60 unique"):
        _validate_cases(rows)


def test_resume_requires_exact_resolved_config(tmp_path: Path):
    row = _load_cases()[0]
    config_path = str(
        Path(__file__).resolve().parents[1] / "configs" / "cable_0p52mm.yaml"
    )
    cfg = _case_config(config_path, row, str(tmp_path), 0.05)
    case_dir = tmp_path / row["case_id"]
    case_dir.mkdir()
    (case_dir / "resolved_config.json").write_text(
        json.dumps(cfg.to_dict()), encoding="utf-8"
    )
    (case_dir / "summary.json").write_text(
        json.dumps({"case_id": row["case_id"], "numerically_converged": True}),
        encoding="utf-8",
    )
    assert _load_completed(config_path, row, str(tmp_path), 0.05) is not None
    resolved = cfg.to_dict()
    resolved["dt"] *= 2.0
    (case_dir / "resolved_config.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )
    assert _load_completed(config_path, row, str(tmp_path), 0.05) is None
