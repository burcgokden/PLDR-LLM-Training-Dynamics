#!/usr/bin/env python3
"""Generate the frozen comprehensive-collapse confirmation protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from comprehensive_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    PREDICTIONS,
    RAW_REQUIRED,
    RESOURCE_CAPS,
    SCHEMA_VERSION,
    STAGES,
)


OUTPUT = ROOT / "experiments" / "protocols" / "comprehensive_collapse_confirmation"


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def expected_files():
    files = {}
    campaign = {
        "schema_version": SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "resource_caps": RESOURCE_CAPS,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "predictions": list(PREDICTIONS),
        "construction_validation_rule": (
            "Numeric intervals, schedules, selected edges, metrics, radii, "
            "subdivision depths, partitions, and challenge seeds are sealed "
            "before validation artifacts are opened."
        ),
    }
    files[Path("campaign_design.json")] = _json(campaign)
    files[Path("construction_protocol.json")] = _json({
        "schema_version": SCHEMA_VERSION,
        "stage": "CONSTRUCTION",
        "raw_required": ["construction_summary"],
        "purpose": (
            "Content-address the construction-only numeric summary before "
            "any validation artifact is opened."),
    })
    for stage in STAGES:
        files[Path(f"{stage['id'].lower()}_protocol.json")] = _json({
            "schema_version": SCHEMA_VERSION,
            "stage": stage,
            "raw_required": list(RAW_REQUIRED[stage["id"]]),
            "analyzer": "experiments/analysis/comprehensive_confirmation_analysis.py",
            "producer": "experiments/confirm/comprehensive_live_producers.py",
        })
    files[Path("prediction_lock_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "PLDR comprehensive-collapse numeric prediction lock",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "code_sha256", "protocol_sha256",
            "partition_sha256", "construction_request_sha256", "schedule",
            "selected_edges", "intervals",
            "path_metrics", "radii", "direct_conversion", "application",
            "lock_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-prediction-lock-v4"},
            "campaign_id": {"type": "string", "minLength": 1},
            "code_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "partition_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "construction_request_sha256": {
                "type": "array",
                "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "minItems": 1,
            },
            "schedule": {"type": "array", "minItems": 1},
            "selected_edges": {"type": "array", "minItems": 1},
            "intervals": {"type": "array", "minItems": 1},
            "path_metrics": {"type": "array", "minItems": 1},
            "radii": {"type": "array", "minItems": 1},
            "direct_conversion": {"type": "array", "minItems": 1},
            "application": {"type": "array", "minItems": 1},
            "lock_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("evidence_request_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Checkpoint-bound stage evidence request",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "stage", "artifact_root",
            "protocol", "artifacts", "request_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-evidence-request-v4"},
            "campaign_id": {"type": "string", "minLength": 1},
            "stage": {
                "enum": [
                    *[stage["id"] for stage in STAGES], "CONSTRUCTION"]},
            "artifact_root": {"type": "string", "minLength": 1},
            "protocol": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "sha256"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "sha256": {
                        "type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
            },
            "artifacts": {"type": "object", "minProperties": 1},
            "request_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("README.md")] = (
        "# Comprehensive collapse confirmation protocols\n\n"
        "These generated protocols measure the finite-stencil parameter normal, "
        "selected pairwise centered-logit frame, signed true-loss response, full "
        "AdamW successor, scheduled path metric, variable-radius trajectory, and "
        "oriented same-source application bound. Construction and validation "
        "partitions are disjoint. Scientific stages remain blocked until both "
        "target devices complete Q and one context-256 N engineering run finishes "
        "below the memory cap.\n"
    ).encode("utf-8")
    checksums = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix())
    )
    files[Path("CHECKSUMS.sha256")] = checksums.encode("ascii")
    return files


def generate(check=False):
    expected = expected_files()
    if check:
        actual = {
            path.relative_to(OUTPUT)
            for path in OUTPUT.rglob("*") if path.is_file()
        } if OUTPUT.is_dir() else set()
        stale = [
            path.as_posix() for path, payload in expected.items()
            if not (OUTPUT / path).is_file() or (OUTPUT / path).read_bytes() != payload
        ]
        unknown = sorted(actual - set(expected), key=lambda item: item.as_posix())
        if stale or unknown:
            raise SystemExit(
                "comprehensive protocols are stale: "
                + ", ".join(stale + [f"unknown:{path}" for path in unknown]))
        print(f"comprehensive protocols: verified ({len(expected)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = OUTPUT / relative
        destination.write_bytes(payload)
    print(f"comprehensive protocols: wrote {len(expected)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    generate(parser.parse_args().check)


if __name__ == "__main__":
    main()
