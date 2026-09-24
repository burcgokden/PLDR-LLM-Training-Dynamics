#!/usr/bin/env python3
"""Generate the deterministic observable full-state cocycle protocol bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "observable_cocycle_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.observable_cocycle_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CALIBRATION_SCHEMA,
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    PREDICTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    SOURCE_STEPS,
    STRATUM_SCHEMA,
    TRAJECTORIES,
    VALIDATION_SCHEMA,
    campaign_design,
    validate_design,
)


def canonical(value: Any) -> bytes:
    """Encode JSON deterministically."""

    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def compact_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def record_inventory() -> dict[str, Any]:
    """Return every record opened by the fixed execution graph."""

    calibration = []
    prediction = []
    validation = []
    stratum = []
    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        for step in SOURCE_STEPS:
            for layer in range(3):
                stem = f"{label}-layer{layer}-step{step}"
                if trajectory["role"] == "construction":
                    calibration.append(
                        {
                            "trajectory": label,
                            "layer": layer,
                            "step": step,
                            "path": f"records/calibration/{stem}.npz",
                        }
                    )
                else:
                    prediction.append(
                        {
                            "trajectory": label,
                            "layer": layer,
                            "step": step,
                            "path": f"records/prediction/{stem}.npz",
                        }
                    )
                    validation.append(
                        {
                            "trajectory": label,
                            "layer": layer,
                            "step": step,
                            "prediction_path": f"records/prediction/{stem}.npz",
                            "path": f"records/validation/{stem}.npz",
                        }
                    )
        for layer in range(3):
            stem = f"{label}-layer{layer}-step4096"
            stratum.append(
                {
                    "trajectory": label,
                    "layer": layer,
                    "step": 4_096,
                    "path": f"records/stratum/{stem}.npz",
                }
            )
    return {
        "schema_version": "pldr-observable-cocycle-record-inventory-v1",
        "campaign_id": CAMPAIGN_ID,
        "qualification": {
            "trajectory": "A",
            "layer": 0,
            "step": 4_096,
            "path": "records/qualification/A-layer0-step4096.npz",
        },
        "calibration": calibration,
        "lock_output": "reports/construction-lock.json",
        "prediction": prediction,
        "validation": validation,
        "stratum": stratum,
        "analysis_outputs": [
            "reports/final-analysis.json",
            "reports/confirmation_results.tex",
            "reports/confirmation_macros.tex",
            "reports/EVIDENCE.sha256",
        ],
    }


def _schema(title: str, schema_version: str, required_arrays: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "properties": {
            "schema_version": {"const": schema_version},
            "campaign_id": {"const": CAMPAIGN_ID},
            "trajectory": {"type": "string", "minLength": 1},
            "source_step": {"type": "integer", "minimum": 0},
            "arrays": {
                "type": "array",
                "items": {"type": "string"},
                "allOf": [
                    {"contains": {"const": name}} for name in required_arrays
                ],
            },
        },
        "required": [
            "schema_version",
            "campaign_id",
            "trajectory",
            "source_step",
            "arrays",
        ],
        "additionalProperties": False,
    }


def schemas() -> dict[str, dict[str, Any]]:
    """Describe the mandatory arrays of each atomic NPZ record."""

    return {
        "qualification.schema.json": _schema(
            "Development resource and derivative qualification",
            QUALIFICATION_SCHEMA,
            [
                "qualified",
                "relative_response_residual",
                "maximum_adamw_replay_relative_residual",
            ],
        ),
        "calibration.schema.json": _schema(
            "Construction full-state cocycle calibration",
            CALIBRATION_SCHEMA,
            [
                "horizons",
                "direction_names",
                "predicted_responses",
                "observed_centered_responses",
            ],
        ),
        "prediction.schema.json": _schema(
            "Held-out source-only cocycle prediction",
            PREDICTION_SCHEMA,
            ["lock_sha256", "endpoint_actions", "predicted_centered_responses"],
        ),
        "validation.schema.json": _schema(
            "Held-out signed nonlinear validation",
            VALIDATION_SCHEMA,
            ["prediction_sha256", "observed_centered_responses"],
        ),
        "stratum.schema.json": _schema(
            "Gate-zero stratum invariance test",
            STRATUM_SCHEMA,
            ["source_row_quotient_norm", "observed_force_norm", "gate_after"],
        ),
        "lock.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Construction horizon lock",
            "type": "object",
            "properties": {"schema_version": {"const": LOCK_SCHEMA}},
            "required": ["schema_version", "lock_sha256", "layers"],
        },
        "analysis.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Observable full-state cocycle analysis",
            "type": "object",
            "properties": {
                "schema_version": {"const": ANALYSIS_SCHEMA},
                "campaign_id": {"const": CAMPAIGN_ID},
            },
            "required": ["schema_version", "campaign_id"],
        },
    }


def generated_files() -> dict[Path, bytes]:
    design = campaign_design()
    validate_design(design)
    signed = dict(design)
    signed["design_sha256"] = compact_digest(design)
    files: dict[Path, bytes] = {
        Path("campaign-design.json"): canonical(signed),
        Path("record-inventory.json"): canonical(record_inventory()),
        Path("README.md"): (
            "# Observable full-state cocycle confirmation protocol\n\n"
            "This deterministic bundle fixes all trajectories, checkpoints, layers, "
            "source updates, horizons, state directions, signed amplitudes, gate-zero "
            "stratum tests, resource limits, and adverse-outcome rules.  Construction "
            "calibration is sealed before held-out prediction and validation.\n"
        ).encode("utf-8"),
    }
    files.update(
        {Path(name): canonical(value) for name, value in schemas().items()}
    )
    lines = [
        f"{hashlib.sha256(payload).hexdigest()}  {path.as_posix()}"
        for path, payload in sorted(files.items(), key=lambda item: item[0].as_posix())
    ]
    files[Path("CHECKSUMS.sha256")] = ("\n".join(lines) + "\n").encode("ascii")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    stale = []
    for relative, payload in generated_files().items():
        destination = arguments.output / relative
        if arguments.check:
            if not destination.is_file() or destination.read_bytes() != payload:
                stale.append(relative.as_posix())
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
    if stale:
        raise SystemExit("observable-cocycle protocols are stale: " + ", ".join(stale))
    verb = "verified" if arguments.check else "generated"
    print(f"observable-cocycle protocols: {verb} ({len(generated_files())} files)")


if __name__ == "__main__":
    main()
