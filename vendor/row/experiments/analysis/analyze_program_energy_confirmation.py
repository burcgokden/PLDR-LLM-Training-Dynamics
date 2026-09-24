#!/usr/bin/env python3
"""Aggregate sealed Q/T/N/I/A/R records in dependency order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from confirmation_artifacts import (  # noqa: E402
    STAGE_RECORD_SCHEMA_VERSION,
    seal_record,
    verify_sealed_record,
    write_json_atomic,
)
from program_energy_protocol_specs import RESOURCE_CAPS, STAGES  # noqa: E402


STAGE_ORDER = tuple(stage["id"] for stage in STAGES)
STAGE_BY_ID = {stage["id"]: stage for stage in STAGES}


def _load_record(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    verify_sealed_record(record)
    required = {
        "schema_version", "campaign_id", "stage", "time_semantics",
        "request_sha256", "resolved_artifacts", "derived_measurements",
        "resource_metadata", "decision", "scientific_record_sha256",
        "record_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError(f"record {path} has missing or unknown fields")
    if record["schema_version"] != STAGE_RECORD_SCHEMA_VERSION:
        raise ValueError(f"record {path} has an unknown schema")
    stage = record["stage"]
    if stage not in STAGE_BY_ID:
        raise ValueError(f"record {path} has an unknown stage")
    expected = set(STAGE_BY_ID[stage]["required_checks"])
    measurements = record["derived_measurements"]
    if not isinstance(measurements, dict) or set(measurements) != expected:
        raise ValueError(f"record {path} has the wrong derived measurements")
    measured_pass = all(
        isinstance(value, dict)
        and set(value) == {"value", "passed", "derivation"}
        and value["passed"] is True
        for value in measurements.values()
    )
    resource_pass = record["resource_metadata"].get("passed") is True
    decision = (
        "CONFIRMED" if measured_pass and resource_pass else "NOT_CONFIRMED")
    if record["decision"] != decision:
        raise ValueError(f"record {path} decision is inconsistent")
    return record


def analyze(records):
    loaded = [_load_record(path) for path in records]
    campaign_ids = {record["campaign_id"] for record in loaded}
    if len(campaign_ids) > 1:
        raise ValueError("stage records belong to different campaigns")
    by_stage = {stage: [] for stage in STAGE_ORDER}
    rows = []
    total_gpu_hours = 0.0
    for path, record in zip(records, loaded):
        by_stage[record["stage"]].append(record)
        total_gpu_hours += float(
            record["resource_metadata"]["recomputed_gpu_hours"])
        rows.append({
            "path": str(path),
            "stage": record["stage"],
            "decision": record["decision"],
            "scientific_record_sha256":
                record["scientific_record_sha256"],
            "record_sha256": record["record_sha256"],
        })

    summary = {}
    confirmed = set()
    for stage_id in STAGE_ORDER:
        stage = STAGE_BY_ID[stage_id]
        dependencies_satisfied = all(
            dependency in confirmed for dependency in stage["dependencies"])
        values = by_stage[stage_id]
        if not values:
            verdict = "PENDING"
        elif (
            dependencies_satisfied
            and all(value["decision"] == "CONFIRMED" for value in values)
        ):
            verdict = "CONFIRMED"
            confirmed.add(stage_id)
        else:
            verdict = "NOT_CONFIRMED"
        summary[stage_id] = {
            "record_count": len(values),
            "dependencies": list(stage["dependencies"]),
            "dependencies_satisfied": dependencies_satisfied,
            "verdict": verdict,
            "required_checks": list(stage["required_checks"]),
        }

    result = {
        "schema_version": "pldr-collapse-campaign-analysis-v2",
        "campaign_id": next(iter(campaign_ids), None),
        "records": rows,
        "stages": summary,
        "resource_summary": {
            "recomputed_gpu_hours": total_gpu_hours,
            "hard_gpu_hours": RESOURCE_CAPS["hard_gpu_hours"],
            "within_hard_cap":
                total_gpu_hours <= RESOURCE_CAPS["hard_gpu_hours"],
        },
        "theory_status": {
            "instrument_qualified":
                summary["Q"]["verdict"] == "CONFIRMED",
            "normal_attraction_confirmed":
                summary["N"]["verdict"] == "CONFIRMED",
            "optimizer_corridor_confirmed":
                summary["I"]["verdict"] == "CONFIRMED",
            "same_source_bridge_confirmed":
                summary["A"]["verdict"] == "CONFIRMED",
            "complete_campaign_replayed":
                summary["R"]["verdict"] == "CONFIRMED",
        },
        "decision": (
            "CONFIRMED"
            if summary["R"]["verdict"] == "CONFIRMED"
            and total_gpu_hours <= RESOURCE_CAPS["hard_gpu_hours"]
            else "PENDING"
        ),
    }
    return seal_record(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = analyze(arguments.records)
    write_json_atomic(arguments.output, result)
    print(arguments.output)


if __name__ == "__main__":
    main()
