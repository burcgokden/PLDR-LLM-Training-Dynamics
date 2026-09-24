#!/usr/bin/env python3
"""Independently analyze source-frozen block-normal confirmation records."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.block_normal import (  # noqa: E402
    block_prediction,
    classify_coverage,
)
from confirm.block_normal_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    BLOCK_RECORD_SCHEMA,
    CAMPAIGN_ID,
    SCIENTIFIC_OUTCOMES,
)


def _vector(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector")
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def _validate_record(record: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version",
        "campaign_id",
        "stage",
        "trajectory",
        "seed",
        "block_start",
        "block_end",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "prediction_created_at_ns",
        "native_endpoint_opened_at_ns",
        "technical_valid",
        "source_values_only",
        "gains",
        "quadratic_coefficients",
        "forces",
        "radii",
        "initial_normal_upper",
        "prefix_products",
        "force_contributions",
        "normal_envelope",
        "comparisons",
    }
    if not isinstance(record, dict) or not required.issubset(record):
        raise ValueError("block-normal record is missing required fields")
    if (
        record["schema_version"] != BLOCK_RECORD_SCHEMA
        or record["campaign_id"] != CAMPAIGN_ID
        or record["technical_valid"] is not True
        or record["source_values_only"] is not True
        or not isinstance(record["seed"], int)
        or isinstance(record["seed"], bool)
        or not isinstance(record["block_start"], int)
        or not isinstance(record["block_end"], int)
        or record["block_end"] <= record["block_start"]
        or record["prediction_created_at_ns"]
        >= record["native_endpoint_opened_at_ns"]
        or not Path(record["source_checkpoint_path"]).is_absolute()
        or len(record["source_checkpoint_sha256"]) != 64
    ):
        raise ValueError("block-normal provenance or chronology is invalid")
    prediction = block_prediction(
        record["gains"],
        record["quadratic_coefficients"],
        record["forces"],
        record["radii"],
        float(record["initial_normal_upper"]),
    )
    recorded_prefix = _vector(record["prefix_products"], "prefix_products")
    recorded_envelope = _vector(record["normal_envelope"], "normal_envelope")
    recorded_forces = np.asarray(
        record["force_contributions"], dtype=np.float64
    )
    if (
        recorded_prefix.shape != prediction["prefix_products"].shape
        or recorded_envelope.shape != prediction["envelope"].shape
        or recorded_forces.shape != prediction["force_contributions"].shape
        or not np.array_equal(recorded_prefix, prediction["prefix_products"])
        or not np.array_equal(recorded_envelope, prediction["envelope"])
        or not np.array_equal(
            recorded_forces, prediction["force_contributions"]
        )
    ):
        raise ValueError("recorded chronological envelope does not replay")
    comparisons = record["comparisons"]
    if not isinstance(comparisons, list) or not comparisons:
        raise ValueError("record needs at least one scientific comparison")
    normalized = []
    for comparison in comparisons:
        fields = {
            "prediction",
            "map_id",
            "context",
            "layer",
            "head",
            "observed_lower",
            "observed_upper",
            "bound_lower",
            "bound_upper",
        }
        if not isinstance(comparison, dict) or set(comparison) != fields:
            raise ValueError("scientific comparison has an invalid schema")
        outcome = classify_coverage(
            [comparison["observed_lower"]],
            [comparison["observed_upper"]],
            [comparison["bound_lower"]],
            [comparison["bound_upper"]],
            tolerance=0.0,
        )[0]
        normalized.append({**comparison, "outcome": str(outcome)})
    return {
        "record": record,
        "prediction": prediction,
        "comparisons": normalized,
    }


def analyze_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Retain every per-map outcome and aggregate only afterward."""

    if not records:
        raise ValueError("at least one block-normal record is required")
    validated = [_validate_record(record) for record in records]
    outcomes = Counter()
    by_stage: dict[str, Counter] = defaultdict(Counter)
    by_layer: dict[str, Counter] = defaultdict(Counter)
    by_head: dict[str, Counter] = defaultdict(Counter)
    by_context: dict[str, Counter] = defaultdict(Counter)
    comparisons = []
    prefix_products = []
    force_contributions = []
    invariant_residuals = []
    for entry in validated:
        record = entry["record"]
        prediction = entry["prediction"]
        prefix_products.append(prediction["prefix_products"])
        force_contributions.append(prediction["force_contributions"])
        invariant_residuals.append(
            prediction["invariant_radius_residuals"]
        )
        for comparison in entry["comparisons"]:
            outcome = comparison["outcome"]
            outcomes[outcome] += 1
            by_stage[str(record["stage"])][outcome] += 1
            if comparison["layer"] is not None:
                by_layer[str(comparison["layer"])][outcome] += 1
            if comparison["head"] is not None:
                by_head[str(comparison["head"])][outcome] += 1
            if comparison["context"] is not None:
                by_context[str(comparison["context"])][outcome] += 1
            comparisons.append({
                "stage": record["stage"],
                "trajectory": record["trajectory"],
                "block_start": record["block_start"],
                "block_end": record["block_end"],
                **comparison,
            })
    physical = [
        row for row in comparisons
        if row["prediction"] == "physical_collapse"
    ]
    physical_outcomes = {row["outcome"] for row in physical}
    if physical and physical_outcomes == {"confirmed"}:
        all_map = "confirmed"
    elif "not_confirmed" in physical_outcomes:
        all_map = "not_confirmed"
    else:
        all_map = "unresolved"
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "technical_valid": True,
        "record_count": len(records),
        "comparison_count": len(comparisons),
        "outcome_counts": {
            name: outcomes[name] for name in SCIENTIFIC_OUTCOMES
        },
        "outcomes_by_stage": {
            key: {name: count[name] for name in SCIENTIFIC_OUTCOMES}
            for key, count in sorted(by_stage.items())
        },
        "outcomes_by_layer": {
            key: {name: count[name] for name in SCIENTIFIC_OUTCOMES}
            for key, count in sorted(by_layer.items())
        },
        "outcomes_by_head": {
            key: {name: count[name] for name in SCIENTIFIC_OUTCOMES}
            for key, count in sorted(by_head.items())
        },
        "outcomes_by_context": {
            key: {name: count[name] for name in SCIENTIFIC_OUTCOMES}
            for key, count in sorted(by_context.items())
        },
        "all_map_physical_outcome": all_map,
        "comparisons": comparisons,
        "prefix_products": prefix_products,
        "force_contributions": force_contributions,
        "invariant_radius_residuals": invariant_residuals,
        "scientific_outcomes_halted_descendants": False,
    }
    if sum(result["outcome_counts"].values()) != len(comparisons):
        raise AssertionError("scientific outcome accounting is incomplete")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", nargs="*", type=Path, default=[])
    parser.add_argument("--record-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-valid", action="store_true")
    arguments = parser.parse_args()
    paths = list(arguments.records)
    for directory in arguments.record_dirs:
        paths.extend(sorted(directory.glob("*.record.json")))
    if not paths:
        raise ValueError("analysis needs records or record directories")
    if len(paths) != len({path.resolve() for path in paths}):
        raise ValueError("analysis record membership contains duplicates")
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in paths
    ]
    report = analyze_records(records)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "technical_valid": report["technical_valid"],
        "record_count": report["record_count"],
        "outcome_counts": report["outcome_counts"],
        "output": str(arguments.output.resolve()),
    }, sort_keys=True))
    if arguments.require_valid and not report["technical_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
