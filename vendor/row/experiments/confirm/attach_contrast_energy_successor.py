#!/usr/bin/env python3
"""Attach a prediction-gated native successor as coverage-only pair energy."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SCHEMA = "pldr-chronological-pair-ledger-v1"
REPORT_SCHEMA = "pldr-contrast-energy-analysis-v1"
JET_SCHEMA = "pldr-validated-contrast-energy-jets-v1"
OUTPUT_SCHEMA = "pldr-contrast-energy-successor-v1"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
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
    value = np.asarray(record[name])
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid shape or values")
    return np.ascontiguousarray(value)


def _single(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != (1,):
        raise ValueError(f"{name} must contain one update value")
    return kind(value[0].item())


def attach(
    source_path: str | Path,
    prediction_path: str | Path,
    successor_path: str | Path,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    prediction_file = Path(prediction_path).resolve()
    successor = Path(successor_path).resolve()
    source_digest = sha256_path(source)
    report = json.loads(prediction_file.read_text(encoding="utf-8"))
    prediction = report.get("prediction", {}) if isinstance(report, dict) else {}
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("source_sha256") != source_digest
        or prediction.get("decision_uses_successor") is not False
        or prediction.get("coverage") is not None
        or report.get("valid") is not True
    ):
        raise ValueError(
            "coverage attachment needs an immutable source prediction")
    jets = Path(report["jets_path"]).resolve()
    if sha256_path(jets) != report["jets_sha256"]:
        raise ValueError("prediction-bound jet artifact changed")
    with np.load(jets, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != JET_SCHEMA:
            raise ValueError("unknown prediction-bound jet schema")
        pairs = _array(record, "pairs", 2).astype(np.int64, copy=False)

    provenance_fields = (
        "update_batch_sha256",
        "source_loss",
        "raw_gradient_sha256",
        "clipped_gradient_sha256",
        "optimizer_state_sha256",
        "source_parameter_sha256",
        "parameter_displacement_sha256",
        "predicted_successor_parameter_sha256",
    )

    with np.load(source, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown source chronology schema")
        if _scalar(record, "successor_evaluated", bool):
            raise ValueError("source chronology already opened a successor")
        source_rows = _array(record, "normalized_rows", 3)
        source_gate = _array(record, "gate", 2)
        source_physical = _array(record, "physical_rows", 3)
        source_identity = {
            "checkpoint_sha256": _scalar(record, "checkpoint_sha256", str),
            "registry_sha256": _scalar(record, "registry_sha256", str),
            "lock_sha256": _scalar(record, "lock_sha256", str),
            "role": _scalar(record, "role", str),
            "seed": _scalar(record, "seed", int),
            "layer": _scalar(record, "layer", int),
            "step_start": _scalar(record, "step_start", int),
            "data_cursor_before": _scalar(record, "data_cursor_before", int),
            "data_cursor_after": _scalar(record, "data_cursor_after", int),
            "data_order_sha256": _scalar(record, "data_order_sha256", str),
        }
        source_provenance = {
            name: _single(
                record, name, float if name == "source_loss" else str)
            for name in provenance_fields
        }
        source_successor_digest = _single(
            record, "successor_parameter_sha256", str)
    if (
        source_rows.shape[0] != 1
        or source_gate.shape[0] != 1
        or source_physical.shape[0] != 1
        or source_successor_digest
    ):
        raise ValueError(
            "source chronology must contain exactly one source state")

    with np.load(successor, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown successor chronology schema")
        if not _scalar(record, "successor_evaluated", bool):
            raise ValueError("successor chronology did not execute the update")
        dependency = json.loads(_scalar(
            record, "prediction_dependency_json", str))
        if (
            dependency.get("source_ledger_sha256") != source_digest
            or dependency.get("sha256") != sha256_path(prediction_file)
            or dependency.get("report_schema_version") != REPORT_SCHEMA
        ):
            raise ValueError(
                "native successor is not gated by this prediction")
        normalized = _array(record, "normalized_rows", 3)
        gate = _array(record, "gate", 2)
        physical = _array(record, "physical_rows", 3)
        successor_identity = {
            "checkpoint_sha256": _scalar(record, "checkpoint_sha256", str),
            "registry_sha256": _scalar(record, "registry_sha256", str),
            "lock_sha256": _scalar(record, "lock_sha256", str),
            "role": _scalar(record, "role", str),
            "seed": _scalar(record, "seed", int),
            "layer": _scalar(record, "layer", int),
            "step_start": _scalar(record, "step_start", int),
            "data_cursor_before": _scalar(record, "data_cursor_before", int),
            "data_cursor_after": _scalar(record, "data_cursor_after", int),
            "data_order_sha256": _scalar(record, "data_order_sha256", str),
        }
        successor_provenance = {
            name: _single(
                record, name, float if name == "source_loss" else str)
            for name in provenance_fields
        }
        successor_parameter_digest = _single(
            record, "successor_parameter_sha256", str)
        native_gate_residual = _array(
            record, "gate_update_max_abs_residual", 1)
        full_parameter_match = _single(
            record, "full_parameter_update_bitwise_match", bool)
    if (
        source_identity != successor_identity
        or source_provenance != successor_provenance
    ):
        raise ValueError("source and successor provenance disagree")
    if (
        normalized.shape[0] != 2
        or gate.shape[0] != 2
        or physical.shape[0] != 2
        or len(successor_parameter_digest) != 64
        or full_parameter_match is not True
        or successor_parameter_digest
        != source_provenance["predicted_successor_parameter_sha256"]
    ):
        raise ValueError("successor chronology must contain one transition")
    if (
        not np.array_equal(source_rows[0], normalized[0])
        or not np.array_equal(source_gate[0], gate[0])
        or not np.array_equal(source_physical[0], physical[0])
    ):
        raise ValueError("successor replay changed the source state")
    successor_rows = physical[1]
    if np.any(pairs[:, 1] >= len(successor_rows)):
        raise ValueError("pair registry leaves the successor row registry")
    contrasts = (
        successor_rows[pairs[:, 0]] - successor_rows[pairs[:, 1]])
    successor_energy = np.einsum(
        "pd,pd->p", contrasts, contrasts, optimize=True)
    metadata = {
        **source_identity,
        **source_provenance,
        "successor_parameter_sha256": successor_parameter_digest,
        "full_parameter_update_bitwise_match": True,
    }
    return {
        "schema_version": np.asarray(OUTPUT_SCHEMA),
        "source_sha256": np.asarray(source_digest),
        "prediction_sha256": np.asarray(sha256_path(prediction_file)),
        "successor_chronology_sha256": np.asarray(sha256_path(successor)),
        "successor_parameter_sha256": np.asarray(
            successor_parameter_digest),
        "successor_energy": successor_energy,
        "native_gate_update_max_abs_residual": native_gate_residual,
        "metadata_json": np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":"))),
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("prediction")
    parser.add_argument("successor")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    arrays = attach(arguments.source, arguments.prediction, arguments.successor)
    target = Path(arguments.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(target)
    print(json.dumps({
        "output": str(target), "pair_count": len(arrays["successor_energy"])
    }, sort_keys=True))


if __name__ == "__main__":
    main()
