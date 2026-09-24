#!/usr/bin/env python3
"""Analyze complete-stack confirmation records without refitting them."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = "pldr-full-stack-confirmation-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"confirmation record is not an object: {path}")
    return value


def validate_record(record):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown full-stack confirmation schema")
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        raise ValueError("full-stack confirmation digest does not replay")
    if record.get("scalar_projection_role") != "diagnostic_only":
        raise ValueError("a scalar projection entered the primary decision")
    if record.get("trace_fitting_used") is not False:
        raise ValueError("a fitted trace entered the primary decision")
    if record.get("caller_selected_radius_multipliers_used") is not False:
        raise ValueError("a caller-selected radius entered the primary decision")
    conditions = record.get("conditions")
    if not isinstance(conditions, dict) or not conditions:
        raise ValueError("record has no closed condition ledger")
    if any(not isinstance(value, bool) for value in conditions.values()):
        raise ValueError("condition ledger must be Boolean")
    expected = "CONFIRMED" if all(conditions.values()) else "NOT_CONFIRMED"
    if record.get("decision") != expected:
        raise ValueError("record decision disagrees with its condition ledger")
    return True


def analyze(records):
    records = list(records)
    if not records:
        raise ValueError("at least one full-stack record is required")
    for record in records:
        validate_record(record)
    condition_names = set(records[0]["conditions"])
    if any(set(record["conditions"]) != condition_names for record in records):
        raise ValueError("record condition ledgers have different schemas")
    condition_counts = {
        name: sum(record["conditions"][name] for record in records)
        for name in sorted(condition_names)
    }
    normal_envelopes = [
        float(record["normal_closure"][
            "maximum_validation_residual_envelope"])
        for record in records
    ]
    right_inverse_residuals = [
        float(record["normal_closure"]["right_inverse_residual"])
        for record in records
    ]
    direct_margins = [
        float(record["criterion"]) - max(
            float(value) for value in record["direct_row_map_uppers"])
        for record in records
    ]
    if not all(math.isfinite(value) for value in (
        normal_envelopes + right_inverse_residuals + direct_margins
    )):
        raise ValueError("analysis inputs contain nonfinite summaries")
    confirmed = sum(record["decision"] == "CONFIRMED" for record in records)
    result = {
        "schema_version": "pldr-full-stack-confirmation-analysis-v1",
        "record_count": len(records),
        "confirmed_count": confirmed,
        "not_confirmed_count": len(records) - confirmed,
        "all_records_confirmed": confirmed == len(records),
        "time_semantics_counts": dict(sorted(Counter(
            record["time_semantics"] for record in records
        ).items())),
        "condition_pass_counts": condition_counts,
        "maximum_normal_validation_residual_envelope": max(normal_envelopes),
        "maximum_right_inverse_residual": max(right_inverse_residuals),
        "minimum_direct_criterion_margin": min(direct_margins),
        "trace_refit_performed": False,
        "decision_rule": "all_registered_primitive_conditions_per_record",
    }
    result["analysis_sha256"] = _digest(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = analyze(_load(path) for path in arguments.records)
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
