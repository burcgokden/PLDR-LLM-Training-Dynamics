#!/usr/bin/env python3
"""Generate the deterministic layer-resolved cocycle protocol bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "layer_cocycle_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.layer_cocycle_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    GATE_OPERATOR_SCHEMA,
    GATE_OPERATOR_STEP,
    INTERVENTION_SCHEMA,
    INTERVENTION_STEP,
    PERTURBATION_SCHEMA,
    PLGA_SCHEMA,
    PRECISION_SCHEMA,
    PRECISION_STEPS,
    TRAJECTORIES,
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

    precision = []
    gate = []
    intervention = []
    perturbation = []
    plga = []
    for trajectory in TRAJECTORIES:
        label = str(trajectory["label"])
        for step in PRECISION_STEPS:
            precision.append(
                {
                    "trajectory": label,
                    "step": step,
                    "path": f"records/precision/{label}-step{step}.npz",
                }
            )
        for layer in range(3):
            stem = f"{label}-layer{layer}-step{GATE_OPERATOR_STEP}"
            gate.append(
                {
                    "trajectory": label,
                    "layer": layer,
                    "step": GATE_OPERATOR_STEP,
                    "path": f"records/gate-operator/{stem}.npz",
                }
            )
            intervention.append(
                {
                    "trajectory": label,
                    "layer": layer,
                    "step": INTERVENTION_STEP,
                    "path": f"records/intervention/{stem}.npz",
                }
            )
            perturbation.append(
                {
                    "trajectory": label,
                    "layer": layer,
                    "step": INTERVENTION_STEP,
                    "prediction_path": (
                        f"records/perturbation/{stem}-prediction.npz"
                    ),
                    "validation_path": f"records/perturbation/{stem}.npz",
                }
            )
        plga.append(
            {
                "trajectory": label,
                "step": 65_536,
                "path": f"records/plga/{label}-step65536.npz",
            }
        )
    return {
        "schema_version": "pldr-layer-cocycle-record-inventory-v1",
        "campaign_id": CAMPAIGN_ID,
        "precision": precision,
        "gate_operator": gate,
        "intervention": intervention,
        "perturbation": perturbation,
        "plga": plga,
        "analysis_outputs": [
            "reports/final-analysis.json",
            "reports/confirmation_results.tex",
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
            "step": {"type": "integer", "minimum": 0},
            "arrays": {
                "type": "array",
                "items": {"type": "string"},
                "allOf": [
                    {"contains": {"const": name}} for name in required_arrays
                ],
            },
        },
        "required": ["schema_version", "campaign_id", "trajectory", "step", "arrays"],
        "additionalProperties": False,
    }


def schemas() -> dict[str, dict[str, Any]]:
    """Describe the mandatory arrays of each atomic NPZ record."""

    return {
        "precision.schema.json": _schema(
            "Float64 and float32 row-map replay",
            PRECISION_SCHEMA,
            ["float64_energy", "float64_shape_energy", "resolution_relative_resolved"],
        ),
        "gate_operator.schema.json": _schema(
            "Lifted final-gate AdamW operator",
            GATE_OPERATOR_SCHEMA,
            ["analytic_operator", "autodiff_operator", "invariance_defect"],
        ),
        "intervention.schema.json": _schema(
            "Selected-layer one-step intervention",
            INTERVENTION_SCHEMA,
            ["arm_names", "arm_energy", "arm_physical_response_from_native"],
        ),
        "perturbation.schema.json": _schema(
            "Source-written sign-paired perturbation validation",
            PERTURBATION_SCHEMA,
            ["prediction_sha256", "relative_residual", "endpoint_distance_by_map"],
        ),
        "plga.schema.json": _schema(
            "PLGA quotient defect replay",
            PLGA_SCHEMA,
            ["row_preservation_residuals", "row_constant_defect_norm", "triangle_signed_slack"],
        ),
        "analysis.schema.json": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Layer-resolved cocycle analysis",
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
            "# Layer-resolved cocycle confirmation protocol\n\n"
            "This deterministic bundle fixes all trajectories, checkpoints, layers, "
            "arithmetic paths, intervention arms, perturbation amplitudes, PLGA "
            "defect measurements, resource limits, and adverse-outcome rules.\n"
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
        raise SystemExit("layer-cocycle protocols are stale: " + ", ".join(stale))
    verb = "verified" if arguments.check else "generated"
    print(f"layer-cocycle protocols: {verb} ({len(generated_files())} files)")


if __name__ == "__main__":
    main()
