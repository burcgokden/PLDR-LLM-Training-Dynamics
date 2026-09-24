#!/usr/bin/env python3
"""Generate the prospective gate-shape confirmation protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from gate_shape_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    PREDICTIONS,
    RAW_REQUIRED,
    RESOURCE_CAPS,
    SCHEMA_VERSION,
    STAGES,
    STATUS,
)


OUTPUT = ROOT / "experiments" / "protocols" / "gate_shape_confirmation"


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def expected_files():
    files = {}
    files[Path("campaign_design.json")] = _json({
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "architecture": ARCHITECTURE,
        "resource_caps": RESOURCE_CAPS,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "predictions": list(PREDICTIONS),
        "evidence_order": [
            "realize one CPU fixture",
            "complete Q on both registered devices",
            "record measured unit costs",
            "freeze code, registry, protocols, and construction plan",
            "run construction trajectory and seal its lock",
            "open validation trajectory and intervention evidence",
            "rebuild all decisions in a fresh process",
        ],
        "construction_validation_rule": (
            "No validation checkpoint, energy, spectrum, continuation, or "
            "application value may set a source charge, gain, mechanism label, "
            "spectral threshold, downstream bound, or remainder radius."
        ),
    })
    for stage in STAGES:
        files[Path(f"{stage['id'].lower()}_protocol.json")] = _json({
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "stage": stage,
            "raw_required": list(RAW_REQUIRED[stage["id"]]),
            "producer": (
                "experiments/confirm/gate_shape_application.py"
                if stage["id"] == "A" else "experiments/confirm/gate_shape_live.py"),
            "analyzer": "experiments/analysis/analyze_gate_shape_confirmation.py",
            "scientific_arrays": "npz_without_pickle",
        })
    files[Path("registry_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Prospective PLDR gate-shape measurement registry",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "context_length", "anchors",
            "construction", "validation", "application_pairs",
            "pair_rule", "dataset_sha256", "tokenizer_sha256",
            "registry_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-gate-shape-registry-v1"},
            "context_length": {"const": 256},
            "anchors": {"const": ARCHITECTURE["anchor_source_steps"]},
            "construction": {"type": "array", "minItems": 4},
            "validation": {"type": "array", "minItems": 4},
            "application_pairs": {"type": "array", "minItems": 32},
            "pair_rule": {"const": "adjacent-flattened-row-pairs-v1"},
            "dataset_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "tokenizer_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("construction_lock_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Source-owned gate-shape prediction lock",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "code_sha256",
            "protocol_sha256", "registry_sha256", "source_requests",
            "capture_blocks", "spectral_threshold_rule", "energy_charges",
            "forcing_charges", "mechanism_labels", "intervention_arms",
            "application_bounds", "source_archive_sha256", "source_owned",
            "validation_artifacts", "target_energy_read",
            "empty_margin_decision", "safety_factor", "lock_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-gate-shape-lock-v2"},
            "campaign_id": {"type": "string", "minLength": 1},
            "code_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "protocol_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "source_requests": {"type": "array", "minItems": 1},
            "capture_blocks": {"type": "array", "minItems": 1},
            "spectral_threshold_rule": {
                "const": "max(reconstruction_residual,100eps*max(opnorm,1))"},
            "energy_charges": {"type": "array", "minItems": 1},
            "forcing_charges": {"type": "array", "minItems": 1},
            "mechanism_labels": {"type": "array", "minItems": 1},
            "source_archive_sha256": {"type": "array", "minItems": 1},
            "source_owned": {"const": True},
            "validation_artifacts": {"const": []},
            "target_energy_read": {"const": False},
            "empty_margin_decision": {"type": "null"},
            "safety_factor": {"type": "number", "minimum": 1.0},
            "intervention_arms": {"type": "array", "minItems": 1},
            "application_bounds": {"type": "array", "minItems": 1},
            "lock_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("evidence_request_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Checkpoint-bound gate-shape evidence request",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "stage", "artifact_root",
            "protocol", "artifacts", "request_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-gate-shape-request-v1"},
            "campaign_id": {"type": "string", "minLength": 1},
            "stage": {"enum": [stage["id"] for stage in STAGES]},
            "artifact_root": {"type": "string", "minLength": 1},
            "protocol": {"type": "object"},
            "artifacts": {"type": "object", "minProperties": 1},
            "request_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("README.md")] = (
        "# Gate-shape collapse confirmation protocols\n\n"
        "These files describe the prospective Q/T/TR/G/C/I/A/R program for "
        "the exact final-LayerNorm gate and normalized-shape theory. The "
        "deciding sectors are 16 by 16 loss matrices and 48 by 48 AdamW "
        "blocks. No large parameter Jacobian, dense parameter-normal SVD, or "
        "rank-length Lanczos basis belongs to this program.\n\n"
        "Status: `DRAFT_PENDING_TWO_DEVICE_QUALIFICATION`. The protocols are "
        "deterministic working specifications, not a frozen scientific "
        "registry. Run the same CPU-realized Q fixture on both registered "
        "devices, record measured unit costs, and only then freeze the "
        "protocol and construction registry.\n"
    ).encode("utf-8")
    checksums = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix())
    )
    files[Path("CHECKSUMS.sha256")] = checksums.encode("ascii")
    return files


def generate(check=False):
    expected = expected_files()
    actual = {
        path.relative_to(OUTPUT)
        for path in OUTPUT.rglob("*") if path.is_file()
    } if OUTPUT.is_dir() else set()
    stale = [
        path.as_posix() for path, payload in expected.items()
        if not (OUTPUT / path).is_file()
        or (OUTPUT / path).read_bytes() != payload
    ]
    unknown = sorted(actual - set(expected), key=lambda item: item.as_posix())
    if check:
        if stale or unknown:
            raise SystemExit(
                "gate-shape protocols are stale: "
                + ", ".join(stale + [f"unknown:{path}" for path in unknown]))
        print(f"gate-shape protocols: verified ({len(expected)} files)")
        return
    if unknown:
        raise SystemExit(
            "refusing to overwrite gate-shape protocol directory with "
            "unknown files: " + ", ".join(path.as_posix() for path in unknown))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = OUTPUT / relative
        destination.write_bytes(payload)
    print(f"gate-shape protocols: wrote {len(expected)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    generate(parser.parse_args().check)


if __name__ == "__main__":
    main()
