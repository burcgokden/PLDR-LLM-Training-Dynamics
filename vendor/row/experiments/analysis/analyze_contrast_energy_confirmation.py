#!/usr/bin/env python3
"""Analyze one immutable source-only contrast-energy certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "confirm"))

from contrast_energy import finite_registry_prediction  # noqa: E402


JET_SCHEMA = "pldr-validated-contrast-energy-jets-v1"
REPORT_SCHEMA = "pldr-contrast-energy-analysis-v1"


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


def analyze(
    jet_path: str | Path,
    successor_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(jet_path).resolve()
    with np.load(path, allow_pickle=False) as record:
        if _scalar(record, "schema_version", str) != JET_SCHEMA:
            raise ValueError("unknown validated contrast-energy schema")
        if _scalar(record, "successor_values_present", bool):
            raise ValueError("deciding jet artifact contains a successor")
        metadata = json.loads(_scalar(record, "metadata_json", str))
        pairs = _array(record, "pairs", 2).astype(np.int64, copy=False)
        enclosure = {
            name: _array(record, name, 1)
            for name in (
                "source_lower", "source_upper",
                "successor_lower", "successor_upper",
            )
        }
        source_sha256 = _scalar(record, "source_sha256", str)
        taylor_model_sha256 = _scalar(record, "taylor_model_sha256", str)
    successor_energy = None
    successor_digest = None
    if successor_path is not None:
        successor = Path(successor_path).resolve()
        successor_digest = sha256_path(successor)
        with np.load(successor, allow_pickle=False) as record:
            if _scalar(record, "source_sha256", str) != source_sha256:
                raise ValueError("successor is not bound to the source artifact")
            successor_energy = _array(record, "successor_energy", 1)
    prediction = finite_registry_prediction(
        pairs, enclosure, successor_energy=successor_energy)
    valid = (
        prediction["coverage"] is None
        or prediction["coverage"]["all_pairs_enclosed"]
    )
    return {
        "schema_version": REPORT_SCHEMA,
        "valid": bool(valid),
        "status": (
            "resolved" if prediction["decision"] != "unresolved"
            else "unresolved"
        ),
        "reason": (
            None if prediction["decision"] != "unresolved"
            else "strict_bounds_cross_source_diameter"
        ),
        "jets_path": str(path),
        "jets_sha256": sha256_path(path),
        "source_sha256": source_sha256,
        "taylor_model_sha256": taylor_model_sha256,
        "successor_sha256": successor_digest,
        "metadata": metadata,
        "prediction": prediction,
    }


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jets")
    parser.add_argument("--successor")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-resolved", action="store_true")
    parser.add_argument("--require-coverage", action="store_true")
    arguments = parser.parse_args()
    report = analyze(arguments.jets, arguments.successor)
    if (
        arguments.require_resolved
        and report["prediction"]["decision"] == "unresolved"
    ):
        raise SystemExit("contrast-energy prediction is unresolved")
    coverage = report["prediction"]["coverage"]
    if arguments.require_coverage and (
        coverage is None or not coverage["all_pairs_enclosed"]
        or not coverage["strict_sign_correct"]
    ):
        raise SystemExit("contrast-energy successor coverage failed")
    write_json(arguments.output, report)
    print(json.dumps({
        "decision": report["prediction"]["decision"],
        "output": str(Path(arguments.output).resolve()),
        "valid": report["valid"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
