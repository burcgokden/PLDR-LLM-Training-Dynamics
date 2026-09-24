#!/usr/bin/env python3
"""Attach a prediction-gated successor to a source-only causal ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


CAUSAL_SCHEMA = "pldr-causal-row-ledger-v1"
CHRONOLOGY_SCHEMA = "pldr-chronological-pair-ledger-v1"
REPORT_SCHEMA = "pldr-causal-row-analysis-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    if name not in record:
        raise ValueError(f"artifact omits {name}")
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def attach(
    source_path: str | Path,
    prediction_path: str | Path,
    successor_path: str | Path,
) -> dict[str, Any]:
    source_file = Path(source_path).resolve()
    report_file = Path(prediction_path).resolve()
    successor_file = Path(successor_path).resolve()
    source_digest = _sha256(source_file)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    prediction = report.get("prediction", {}) if isinstance(report, dict) else {}
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != REPORT_SCHEMA
        or report.get("ledger_sha256") != source_digest
        or prediction.get("decision_uses_successor") is not False
        or prediction.get("coverage") is not None
        or report.get("valid") is not True
    ):
        raise ValueError("the prediction is not bound to the source-only ledger")

    with np.load(source_file, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != CAUSAL_SCHEMA:
            raise ValueError("unknown causal source schema")
        if "successor_rows" in record:
            raise ValueError("causal source already contains a successor")
        output = {name: np.asarray(record[name]) for name in record.files}
        source_rows = _array(record, "source_rows", 2)
        predicted_gate = _array(record, "gate_predicted", 1)
        metadata = json.loads(_scalar(record, "metadata_json", str))

    with np.load(successor_file, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != CHRONOLOGY_SCHEMA:
            raise ValueError("unknown chronological successor schema")
        if _scalar(record, "successor_evaluated", bool) is not True:
            raise ValueError("chronological artifact has no successor")
        dependency = json.loads(
            _scalar(record, "prediction_dependency_json", str))
        if (
            dependency.get("source_ledger_sha256") != source_digest
            or dependency.get("sha256") != _sha256(report_file)
        ):
            raise ValueError("successor was not gated by this prediction")
        normalized_rows = _array(record, "normalized_rows", 3)
        gate = _array(record, "gate", 2)
        native_gate_residual = _array(
            record, "gate_update_max_abs_residual", 1)
        if normalized_rows.shape[0] != 2 or gate.shape[0] != 2:
            raise ValueError("successor artifact must contain one transition")
        if (
            gate.shape[1:] != predicted_gate.shape
            or native_gate_residual.shape != (1,)
        ):
            raise ValueError("successor gate diagnostics have invalid shapes")
        expected_metadata = {
            "checkpoint_sha256": _scalar(record, "checkpoint_sha256", str),
            "role": _scalar(record, "role", str),
            "seed": _scalar(record, "seed", int),
            "layer": _scalar(record, "layer", int),
            "global_step_before": _scalar(record, "step_start", int),
        }
    if not isinstance(metadata, dict) or any(
        metadata.get(name) != value for name, value in expected_metadata.items()
    ):
        raise ValueError("source and successor metadata disagree")
    reconstructed_source = normalized_rows[0] * gate[0]
    if not np.array_equal(source_rows, reconstructed_source):
        raise ValueError("successor rerun changed the source physical rows")
    output["successor_rows"] = normalized_rows[1] * gate[1]
    gate_scale = max(
        1.0,
        float(np.max(np.abs(predicted_gate))),
        float(np.max(np.abs(gate[1]))),
    )
    gate_tolerance = (
        64.0 * float(np.finfo(np.float32).eps) * gate_scale)
    output["gate_successor"] = gate[1]
    output["gate_successor_max_abs_residual"] = np.asarray(
        float(np.max(np.abs(gate[1] - predicted_gate))))
    output["gate_native_update_max_abs_residual"] = np.asarray(
        float(native_gate_residual[0]))
    output["gate_successor_float32_tolerance"] = np.asarray(gate_tolerance)
    output["prediction_report_sha256"] = np.asarray(_sha256(report_file))
    output["successor_chronology_sha256"] = np.asarray(
        _sha256(successor_file))
    return output


def write_npz(path: str | Path, arrays: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_ledger")
    parser.add_argument("prediction_report")
    parser.add_argument("successor_chronology")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    arrays = attach(
        arguments.source_ledger,
        arguments.prediction_report,
        arguments.successor_chronology,
    )
    write_npz(arguments.output, arrays)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "source_ledger_sha256": _sha256(Path(arguments.source_ledger)),
        "successor_attached": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
