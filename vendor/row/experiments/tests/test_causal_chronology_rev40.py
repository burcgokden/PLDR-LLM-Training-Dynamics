import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.finalize_causal_lock import build as build_lock
from confirm.attach_causal_successor import attach
from confirm.confirmation_artifacts import digest_object


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_npz(path: Path, **arrays) -> None:
    with path.open("wb") as stream:
        np.savez_compressed(stream, **arrays)


def test_successor_requires_digest_bound_source_prediction(tmp_path):
    checkpoint_digest = "1" * 64
    source_rows = np.arange(12, dtype=np.float64).reshape(3, 4)
    metadata = {
        "checkpoint_sha256": checkpoint_digest,
        "role": "construction",
        "seed": 8444,
        "layer": 1,
        "global_step_before": 2000,
    }
    source = tmp_path / "source.npz"
    _write_npz(
        source,
        schema_version=np.asarray("pldr-causal-row-ledger-v1"),
        certificate_schema=np.asarray(
            "pldr-causal-remainder-certificate-v1"),
        certificate_sha256=np.asarray("2" * 64),
        certificate_frozen_before_successor=np.asarray(True),
        source_rows=source_rows,
        row_jvp=np.zeros_like(source_rows),
        row_remainder_radius=np.zeros(3),
        gate_predicted=np.ones(4),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    source_digest = _sha256(source)
    report = tmp_path / "prediction.json"
    report.write_text(json.dumps({
        "schema_version": "pldr-causal-row-analysis-v1",
        "ledger_sha256": source_digest,
        "metadata": metadata,
        "prediction": {
            "decision": "UNRESOLVED",
            "decision_uses_successor": False,
            "coverage": None,
        },
        "valid": True,
    }), encoding="utf-8")
    normalized = np.stack((source_rows, source_rows - 1.0))
    gates = np.ones((2, 4), dtype=np.float64)
    successor = tmp_path / "successor.npz"
    dependency = {
        "source_ledger_sha256": source_digest,
        "sha256": _sha256(report),
    }
    _write_npz(
        successor,
        schema_version=np.asarray("pldr-chronological-pair-ledger-v1"),
        successor_evaluated=np.asarray(True),
        prediction_dependency_json=np.asarray(json.dumps(dependency)),
        normalized_rows=normalized,
        gate=gates,
        gate_update_max_abs_residual=np.zeros(1),
        checkpoint_sha256=np.asarray(checkpoint_digest),
        role=np.asarray("construction"),
        seed=np.asarray(8444),
        layer=np.asarray(1),
        step_start=np.asarray(2000),
    )
    result = attach(source, report, successor)
    assert np.array_equal(result["successor_rows"], source_rows - 1.0)
    assert np.array_equal(result["gate_successor"], np.ones(4))
    assert result["gate_successor_max_abs_residual"].item() == 0.0
    assert result["prediction_report_sha256"].item() == _sha256(report)


def test_construction_lock_accepts_only_source_only_reports(tmp_path):
    report = tmp_path / "construction.json"
    report.write_text(json.dumps({
        "schema_version": "pldr-causal-row-analysis-v1",
        "ledger_sha256": "3" * 64,
        "metadata": {
            "role": "construction", "seed": 8444,
            "layer": 0, "global_step_before": 1000,
        },
        "prediction": {
            "decision": "CONTRACTION_CERTIFIED",
            "decision_uses_successor": False,
            "coverage": None,
            "contraction_margin_squared": 1.0,
            "reexpansion_margin_squared": -1.0,
        },
        "valid": True,
    }), encoding="utf-8")
    lock = build_lock([report])
    unsigned = dict(lock)
    recorded = unsigned.pop("lock_sha256")
    assert recorded == digest_object(unsigned)
    assert lock["constant_enlargement_after_lock"] is False
    assert lock["source_only_construction"] is True
