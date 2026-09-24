#!/usr/bin/env python3
"""Freeze causal constants before any held-out trajectory is opened."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.causal_confirmation_specs import (  # noqa: E402
    ANCHORS,
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    REGISTRY,
)


REPORT_SCHEMA = "pldr-causal-row-analysis-v1"
PLGA_REPORT_SCHEMA = "pldr-causal-plga-analysis-v1"


def _digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inflate(value: Any, factor: float = 2.0) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError("construction PLGA maximum is invalid")
    return float(np.nextafter(
        max(factor * number, np.finfo(np.float64).tiny), np.inf))


def _load_plga(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source = Path(path).resolve()
    report = json.loads(source.read_text(encoding="utf-8"))
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != PLGA_REPORT_SCHEMA
        or report.get("role") != "construction"
        or report.get("record_count") != 51
        or report.get("decision") != "CONSTRUCTION_SUMMARY"
        or not all(report.get("checks", {}).values())
        or not isinstance(report.get("maxima"), dict)
    ):
        raise ValueError("PLGA construction report is incomplete or invalid")
    maxima = report["maxima"]
    constants = {
        "plga_power_secant_bound": _inflate(maxima["power_secant"]),
        "plga_transfer_coefficient_bound": _inflate(
            maxima["transfer_coefficient"]),
        "centered_logit_response_coefficient": _inflate(
            maxima["centered_response_ratio"]),
        "centered_logit_remainder_bound": 0.0,
    }
    provenance = {
        "path": str(source),
        "sha256": _sha256(source),
        "report_schema": PLGA_REPORT_SCHEMA,
        "record_count": 51,
        "maxima": maxima,
    }
    return constants, provenance


def build(
    paths: list[str | Path],
    plga_path: str | Path | None = None,
    *,
    require_complete_grid: bool = False,
) -> dict[str, Any]:
    if not paths:
        raise ValueError("construction lock needs source-only reports")
    records = []
    seen = set()
    for path in paths:
        source = Path(path).resolve()
        report = json.loads(source.read_text(encoding="utf-8"))
        prediction = (
            report.get("prediction", {}) if isinstance(report, dict) else {})
        metadata = report.get("metadata", {}) if isinstance(report, dict) else {}
        key = (
            metadata.get("role"), metadata.get("seed"),
            metadata.get("layer"), metadata.get("global_step_before"),
        )
        if (
            report.get("schema_version") != REPORT_SCHEMA
            or metadata.get("role") not in {"development", "construction"}
            or prediction.get("decision_uses_successor") is not False
            or prediction.get("coverage") is not None
            or report.get("valid") is not True
            or key in seen
        ):
            raise ValueError("construction report violates the causal lock contract")
        seen.add(key)
        records.append({
            "path": str(source),
            "report_sha256": _sha256(source),
            "ledger_sha256": report["ledger_sha256"],
            "role": str(metadata["role"]),
            "seed": int(metadata["seed"]),
            "layer": int(metadata["layer"]),
            "global_step_before": int(metadata["global_step_before"]),
            "decision": prediction["decision"],
            "contraction_margin_squared": float(
                prediction["contraction_margin_squared"]),
            "reexpansion_margin_squared": float(
                prediction["reexpansion_margin_squared"]),
        })
    records.sort(key=lambda row: (
        row["role"], row["seed"], row["layer"],
        row["global_step_before"]))
    if require_complete_grid:
        expected = {
            (role, seed, layer, anchor)
            for role, seed in (
                ("development", 7444), ("construction", 8444))
            for layer in range(3)
            for anchor in ANCHORS
        }
        observed = {
            (row["role"], row["seed"], row["layer"],
             row["global_step_before"])
            for row in records
        }
        if observed != expected or len(records) != 102:
            raise ValueError("causal construction grid is incomplete")

    if plga_path is None:
        plga_constants = {
            "plga_power_secant_bound": 1.0,
            "plga_transfer_coefficient_bound": 1.0,
            "centered_logit_response_coefficient": 1.0,
            "centered_logit_remainder_bound": 0.0,
        }
        plga_provenance = None
    else:
        plga_constants, plga_provenance = _load_plga(plga_path)

    relative = 8.0 * 64.0 * np.finfo(np.float32).eps
    lock: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "constant_enlargement_after_lock": False,
        "source_only_construction": True,
        "primary_block_length": 1,
        "history_length": int(REGISTRY["history_length"]),
        "segment_grid": [0.0, 0.5, 1.0],
        "shape_safety_factor": 2.0,
        "numerical_relative_coefficient": float(relative),
        "numerical_absolute_charge": 0.0,
        "pair_chunk_size": int(REGISTRY["pair_chunk_size"]),
        "arithmetic_safety_factor": 8.0,
        "intervention_minimum_effect": 1.0e-12,
        "plga_construction_multiplier": 2.0,
        **plga_constants,
        "decision_rule": (
            "strict comparison of the successor upper maximum with the "
            "source-diameter lower bound, or a successor lower energy "
            "with the source-diameter upper bound"),
        "construction_records": records,
        "plga_construction_report": plga_provenance,
    }
    lock["lock_sha256"] = _digest(lock)
    return lock


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--plga-report", required=True)
    parser.add_argument("--require-complete-grid", action="store_true")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    lock = build(
        arguments.reports,
        arguments.plga_report,
        require_complete_grid=arguments.require_complete_grid,
    )
    write_json(arguments.output, lock)
    print(json.dumps({
        "lock_sha256": lock["lock_sha256"],
        "output": str(Path(arguments.output).resolve()),
        "records": len(lock["construction_records"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
