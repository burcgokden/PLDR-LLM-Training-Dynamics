import json
from pathlib import Path
import sys

import pytest


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "analysis"))
sys.path.insert(0, str(EXP / "confirm"))

import e0_decision as D  # noqa: E402
import launch_downstream as L  # noqa: E402
from require_e0_pass import require_e0_pass  # noqa: E402


def pass_record():
    return {
        "schema_version": D.SCHEMA_VERSION,
        "protocol_hash": "a" * 64,
        "schema_hash": "b" * 64,
        "source_hash": "c" * 64,
        "calibration_hash": "d" * 64,
        "curvature_kind": "gauss_newton",
        "metric_kind": "preconditioned",
        "unit_kind": "physical",
        "residual_certificate_pass": True,
        "n_upper_crossings": 40,
        "pilot_region_status": "nonempty",
        "model_consistent": True,
        "uncertainty_resolved": True,
        "prerequisites": {
            name: True for name in D.REQUIRED_PREREQUISITES
        },
        "family_p_values": {
            name: 1e-6 for name in D.INFERENTIAL_FAMILIES
        },
        "family_directional_pass": {
            name: True for name in D.INFERENTIAL_FAMILIES
        },
    }


def test_hash_bound_pass_and_nonpass_denial(tmp_path):
    path = tmp_path / "E0_RECORD.json"
    record = pass_record()
    path.write_text(json.dumps(record), encoding="utf-8")
    require_e0_pass(path, "c" * 64, "a" * 64)
    with pytest.raises(RuntimeError, match="source hash is stale"):
        require_e0_pass(path, "e" * 64, "a" * 64)

    record["n_upper_crossings"] = 39
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="NOT_CONFIRMED"):
        require_e0_pass(path)


def test_launcher_authorizes_before_dry_run(monkeypatch):
    seen = []
    monkeypatch.setattr(L, "authorize", lambda record: seen.append(record))
    assert L.launch("E7", ["python3", "analysis.py"], dry_run=True) == 0
    assert len(seen) == 1
    with pytest.raises(ValueError, match="E1"):
        L.launch("E0", ["python3", "analysis.py"], dry_run=True)
