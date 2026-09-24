#!/usr/bin/env python3
"""Generate the tracked source-restoring confirmation protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "source_restoring_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_restoring_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    CERTIFICATE_POLICY,
    CONSTRUCTION_EDGES,
    DESIGN_SCHEMA,
    HELDOUT_ANCHORS,
    MASK_CONTRACT,
    OPTIMIZER,
    PILOT_EDGES,
    REGISTRY,
    RESOURCE_BUDGET,
    STAGES,
    TIMECOURSE_ANCHORS,
    TRAINING_SCHEMA,
    TRAJECTORIES,
)


def canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def design() -> dict:
    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "training_campaign_schema": TRAINING_SCHEMA,
        "architecture": ARCHITECTURE,
        "registry": REGISTRY,
        "optimizer": OPTIMIZER,
        "trajectories": TRAJECTORIES,
        "pilot_edges": PILOT_EDGES,
        "construction_edges": CONSTRUCTION_EDGES,
        "timecourse_anchors": TIMECOURSE_ANCHORS,
        "heldout_anchors": HELDOUT_ANCHORS,
        "mask_contract": MASK_CONTRACT,
        "certificate_policy": CERTIFICATE_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "stages": STAGES,
        "policy_lock": {
            "construction_precedes_heldout": True,
            "source_certificate_precedes_successor_open": True,
            "scientific_outcome_does_not_prune": True,
            "technical_invalidity_halts_descendants": True,
        },
    }


def schemas() -> dict[str, dict]:
    return {
        "source_artifact_schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Source row-map JVP artifact",
            "type": "object",
            "required": [
                "schema_version",
                "source_rows",
                "gate_velocity_rows",
                "shape_velocity_rows",
                "input_velocity_rows",
                "metadata_json",
            ],
            "additionalProperties": False,
        },
        "validated_curvature_schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Validated source-segment energy-curvature bounds",
            "type": "object",
            "required": [
                "schema_version",
                "campaign_id",
                "parent_source_sha256",
                "source_parameter_sha256",
                "successor_values_used",
                "energy_curvature_abs_upper",
                "endpoint_contrast_norm_upper",
                "native_contrast_radius",
                "metadata_json",
            ],
            "additionalProperties": False,
        },
        "remainder_enclosure_schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Validated source-segment remainder enclosure",
            "type": "object",
            "required": [
                "schema_version",
                "campaign_id",
                "parent_source_sha256",
                "source_parameter_sha256",
                "successor_values_used",
                "validation_method",
                "remainder_lower",
                "remainder_upper",
                "endpoint_contrast_norm_upper",
                "native_contrast_radius",
                "metadata_json",
            ],
            "additionalProperties": False,
        },
        "training_campaign_schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Content-addressed native training campaign",
            "type": "object",
            "required": [
                "schema_version",
                "campaign_id",
                "registry_sha256",
                "trajectories",
                "spec_sha256",
            ],
            "additionalProperties": True,
        },
        "edge_record_schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Trainer-native chronological source-restoring edge",
            "type": "object",
            "required": [
                "campaign_id",
                "step",
                "source_artifact_path",
                "source_artifact_sha256",
                "registry_sha256",
                "data_order_sha256",
                "certificate_created_at_ns",
                "prediction_created_at_ns",
                "successor_opened_at_ns",
                "source_checkpoint_path",
                "source_checkpoint_sha256",
                "native_successor_checkpoint_path",
                "native_successor_checkpoint_sha256",
                "prediction_sha256",
                "certificate_sha256",
                "technical_valid",
                "scientific_outcome",
                "source_diameter_squared",
                "source_diameter_squared_lower",
                "source_diameter_squared_upper",
                "source_upper_multiplier",
                "source_successor_diameter_squared_upper",
                "native_successor_diameter_squared",
                "native_successor_diameter_squared_lower",
                "native_successor_diameter_squared_upper",
                "certificate_coverage_fraction",
                "native_peak_memory_reserved_bytes",
            ],
            "additionalProperties": True,
        },
    }


def readme() -> str:
    return f"""# Source-restoring row-map confirmation

Campaign `{CAMPAIGN_ID}` evaluates the source-state theorem on every one of
288 context-layer-head maps and every one of 580,608 within-map row pairs.
The source artifact contains the complete clipped AdamW direction and three
disjoint JVP blocks. A content-bound reducer converts independently validated
energy-curvature bounds to directed Taylor remainder endpoints. A certificate
is emitted only after that remainder enclosure has been linked by SHA-256.
The trainer-native checkpoint at step n+1 is opened only after the prediction
is immutable. Consecutive edge records must link the exact path and digest of
that checkpoint to the next source.

P0 runs deterministic CPU and both-GPU qualification. P1 records updates
1000 through 1003. P2 records the consecutive block 1004 through 1023. P3
keeps the complete 90-point construction time course. P4 and P5 apply the
locked policy to seeds 8444 and 9444. Scientific outcomes are retained and
do not prune later nodes; only technical invalidity halts descendants.
"""


def expected() -> dict[str, bytes]:
    files = {
        "README.md": readme().encode(),
        "campaign_design.json": canonical(design()),
    }
    files.update({name: canonical(value) for name, value in schemas().items()})
    manifest = "".join(
        f"{hashlib.sha256(files[name]).hexdigest()}  {name}\n"
        for name in sorted(files)
    ).encode("ascii")
    files["CHECKSUMS.sha256"] = manifest
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    files = expected()
    if arguments.check:
        actual = (
            {path.name for path in OUTPUT.iterdir()}
            if OUTPUT.is_dir()
            else set()
        )
        if actual != set(files):
            raise SystemExit("source-restoring protocol membership is stale")
        for name, payload in files.items():
            if (OUTPUT / name).read_bytes() != payload:
                raise SystemExit(f"source-restoring protocol is stale: {name}")
        print(f"source-restoring protocols: verified ({len(files)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file() and path.name not in files:
            path.unlink()
    for name, payload in files.items():
        (OUTPUT / name).write_bytes(payload)
    print(f"source-restoring protocols: wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
