#!/usr/bin/env python3
"""Generate the frozen observable-balance protocol and record inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.observable_balance_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    ARCHIVE_PRIOR_ATTEMPTS,
    CAMPAIGN_ID,
    CONSTRUCTION_SCHEMA,
    DESIGN_SCHEMA,
    DIRECTION_CLASSES,
    LOCK_SCHEMA,
    PREDICTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    SOURCE_STEPS,
    STRATUM_SCHEMA,
    TRAJECTORIES,
    VALIDATION_SCHEMA,
    campaign_design,
)


OUTPUT = EXPERIMENTS / "protocols" / "observable_balance_confirmation"


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()


def record_inventory() -> dict[str, Any]:
    construction = [
        f"records/construction-A-L{layer}-S{source}.npz"
        for layer in range(3)
        for source in SOURCE_STEPS
    ]
    prediction = [
        f"records/prediction-{trajectory['label']}-L{layer}-S{source}.npz"
        for trajectory in TRAJECTORIES
        if trajectory["role"] == "heldout"
        for layer in range(3)
        for source in SOURCE_STEPS
    ]
    validation = [path.replace("prediction-", "validation-") for path in prediction]
    stratum = [
        f"records/stratum-control-{trajectory['label']}-L{layer}-S4096.npz"
        for trajectory in TRAJECTORIES
        for layer in range(3)
    ]
    return {
        "schema_version": "pldr-observable-balance-inventory-v1",
        "campaign_id": CAMPAIGN_ID,
        "qualification": "records/qualification-A-L0-S4096.npz",
        "construction": construction,
        "lock": "reports/construction-lock.json",
        "prediction": prediction,
        "validation": validation,
        "stratum_control": stratum,
        "analysis_outputs": [
            "reports/final-analysis.json",
            "reports/observable_balance_macros.tex",
            "reports/observable_balance_results.tex",
            "reports/EVIDENCE.sha256",
        ],
        "expected_counts": {
            "qualification": 1,
            "construction": 9,
            "prediction": 18,
            "validation": 18,
            "stratum_control": 9,
            "analysis": 1,
            "total_plan_nodes": 57,
            "heldout_cells": 54,
            "construction_cells": 216,
        },
    }


def _npz_schema(title: str, schema_version: str, required: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "additionalProperties": True,
        "required_arrays": ["schema_version", "campaign_id", "terminal_status", *required],
        "constants": {
            "schema_version": schema_version,
            "campaign_id": CAMPAIGN_ID,
        },
        "terminal_policy": (
            "A schema-valid adverse scientific terminal may replace measurement "
            "arrays only when terminal_status is adverse-scientific-terminal."
        ),
    }


def generated_files() -> dict[Path, bytes]:
    design = campaign_design()
    design_payload = dict(design)
    design_payload["design_sha256"] = digest_object(design)
    inventory = record_inventory()
    schemas = {
        "qualification.schema.json": _npz_schema(
            "Observable-balance resource qualification",
            QUALIFICATION_SCHEMA,
            ["qualified", "balance_terms", "observed_response"],
        ),
        "construction.schema.json": _npz_schema(
            "Observable-balance construction record",
            CONSTRUCTION_SCHEMA,
            [
                "horizons",
                "direction_names",
                "amplitudes",
                "balance_terms",
                "observed_responses",
                "cross_grams",
                "evaluable",
            ],
        ),
        "prediction.schema.json": _npz_schema(
            "Held-out homogeneous prediction",
            PREDICTION_SCHEMA,
            ["stage", "lock_sha256", "homogeneous_terms"],
        ),
        "validation.schema.json": _npz_schema(
            "Held-out finite-increment balance validation",
            VALIDATION_SCHEMA,
            [
                "stage",
                "prediction_sha256",
                "balance_terms",
                "observed_responses",
                "cross_grams",
                "evaluable",
            ],
        ),
        "stratum.schema.json": _npz_schema(
            "Gate-zero controlled-gradient record",
            STRATUM_SCHEMA,
            [
                "natural_clipped_gate_gradient",
                "natural_predicted_gate",
                "natural_actual_gate",
                "controlled_actual_gate",
                "theory_code_pass",
            ],
        ),
        "lock.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Observable-balance construction lock",
            "type": "object",
            "required": [
                "schema_version",
                "campaign_id",
                "layers",
                "lock_sha256",
            ],
            "properties": {
                "schema_version": {"const": LOCK_SCHEMA},
                "campaign_id": {"const": CAMPAIGN_ID},
            },
        },
        "analysis.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Observable-balance final analysis",
            "type": "object",
            "required": [
                "schema_version",
                "campaign_id",
                "construction",
                "heldout",
                "stratum_control",
                "resource",
            ],
            "properties": {
                "schema_version": {"const": ANALYSIS_SCHEMA},
                "campaign_id": {"const": CAMPAIGN_ID},
            },
        },
    }
    files = {
        Path("campaign-design.json"): canonical(design_payload),
        Path("archive-attempt-ledger.json"): canonical(
            {
                "schema_version": "pldr-archive-attempt-ledger-v1",
                **ARCHIVE_PRIOR_ATTEMPTS,
                "prior_total_hours": sum(
                    float(row["hours"])
                    for row in ARCHIVE_PRIOR_ATTEMPTS["components"]
                ),
            }
        ),
        Path("record-inventory.json"): canonical(inventory),
        Path("README.md"): (
            "# Finite-increment observable-balance protocol\n\n"
            "This generated protocol fixes the exact three-term balance, numerical "
            "gates, held-out campaign decision, controlled gate-gradient test, "
            "resource limits, and outcome-independent record inventory.\n"
        ).encode("utf-8"),
        **{Path(name): canonical(value) for name, value in schemas.items()},
    }
    checksum = "".join(
        f"{hashlib.sha256(payload).hexdigest()}  {path.as_posix()}\n"
        for path, payload in sorted(files.items(), key=lambda row: row[0].as_posix())
    ).encode("ascii")
    files[Path("CHECKSUMS.sha256")] = checksum
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    files = generated_files()
    if arguments.check:
        committed_extras = {Path("launch-authorization.json")}
        actual = {
            path.relative_to(OUTPUT)
            for path in OUTPUT.rglob("*")
            if path.is_file()
        } if OUTPUT.is_dir() else set()
        if actual - committed_extras != set(files) or not (
            actual & committed_extras
        ) <= committed_extras:
            raise SystemExit("observable-balance protocol membership is stale")
        for relative, payload in files.items():
            if (OUTPUT / relative).read_bytes() != payload:
                raise SystemExit(f"observable-balance protocol is stale: {relative}")
        print(f"observable-balance protocols: verified ({len(files)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for relative, payload in files.items():
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print(f"observable-balance protocols: generated ({len(files)} files)")


if __name__ == "__main__":
    main()
