#!/usr/bin/env python3
"""Generate deterministic protocols for block-normal confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "block_normal_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.block_normal_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    BLOCK_RECORD_SCHEMA,
    CAMPAIGN_ID,
    DESIGN_SCHEMA,
    NATIVE_EVIDENCE_SCHEMA,
    QUALIFICATION_SCHEMA,
    SOURCE_EVIDENCE_SCHEMA,
    campaign_design,
    validate_design,
)


def resource_schema(title, required):
    schema = _schema(title, required)
    schema["properties"] = {"executor_contract": {"const": "pldr-resource-execution-v2"}}
    return schema


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def _schema(title: str, required: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": title,
        "type": "object",
        "required": required,
        "additionalProperties": True,
    }


def schemas() -> dict[str, dict[str, Any]]:
    return {
        "finite_transport_schema.json": _schema(
            "Exact finite row-program transport",
            [
                "schema_version",
                "campaign_id",
                "source_names",
                "topological_order",
                "centered_transport_residual_norm",
                "energy_balance_residual",
                "plga_multiplied_fields",
                "outward_radius",
            ],
        ),
        "normal_projection_schema.json": _schema(
            "Constrained normal projection",
            [
                "schema_version",
                "campaign_id",
                "projected_reference",
                "primal_residual_norm",
                "kkt_stationarity_residual_norm",
                "rank",
                "condition_estimate",
                "metric_eigenvalue_min",
                "metric_eigenvalue_max",
                "frozen_coordinate_residual",
                "tube_membership",
                "iteration_trace",
            ],
        ),
        "source_evidence_schema.json": _schema(
            "Sealed source-only block prediction",
            [
                "schema_version",
                "campaign_id",
                "stage",
                "trajectory",
                "seed",
                "block_start",
                "block_end",
                "source_checkpoint_path",
                "source_checkpoint_sha256",
                "created_at_ns",
                "operators",
                "quadratic_coefficients",
                "forces",
                "radii",
                "initial_normal_upper",
                "comparison_bounds",
            ],
        ),
        "native_evidence_schema.json": _schema(
            "Later native endpoint observations bound to a sealed source",
            [
                "schema_version",
                "campaign_id",
                "source_prediction_sha256",
                "source_checkpoint_sha256",
                "native_checkpoint_path",
                "native_checkpoint_sha256",
                "opened_at_ns",
                "observations",
            ],
        ),
        "block_record_schema.json": _schema(
            "Source-frozen block-normal prediction and native comparison",
            [
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
            ],
        ),
        "native_determinism_schema.json": _schema(
            "Native deterministic clone and conversion charges",
            [
                "schema_version",
                "campaign_id",
                "tensor_comparisons",
                "row_comparisons",
                "native_centered_row_radius",
                "conversion_centered_row_radius",
                "charges_combined",
                "source_checkpoint_path",
                "source_checkpoint_sha256",
                "native_successor_path",
                "native_successor_sha256",
            ],
        ),
        "resource_record_schema.json": resource_schema(
            "Resource observation record with v2 execution admission",
            [
                "schema_version",
                "command",
                "command_sha256",
                "device",
                "wall_seconds",
                "peak_host_rss_bytes",
                "peak_gpu_allocated_bytes",
                "peak_gpu_reserved_bytes",
                "peak_output_bytes",
                "caps",
                "cap_status",
                "exit_code",
                "technical_valid",
                "executor_contract",
                "completion_observed_seconds",
                "total_supervisor_seconds",
                "final_output_bytes",
                "deadline_outcome",
                "cleanup_complete",
                "measurement_error",
                "surviving_descendants",
                "detached_descendants",
            ],
        ),
        "retention_closure_schema.json": _schema(
            "Complete checkpoint retention closure",
            [
                "schema_version",
                "complete_checkpoint_path",
                "complete_checkpoint_sha256",
                "model_only_checkpoint_path",
                "model_only_checkpoint_sha256",
                "open_descendants",
                "recovery",
                "measured_storage_bytes",
                "cleanup_authorized",
                "deletion_performed",
            ],
        ),
        "analysis_schema.json": _schema(
            "Independent block-normal analysis",
            [
                "schema_version",
                "campaign_id",
                "technical_valid",
                "record_count",
                "comparison_count",
                "outcome_counts",
                "comparisons",
                "prefix_products",
                "force_contributions",
                "invariant_radius_residuals",
            ],
        ),
        "qualification_schema.json": _schema(
            "Block-normal Q0 deterministic fixture qualification",
            [
                "schema_version",
                "campaign_id",
                "qualification_mode",
                "checkpoint_backed",
                "scientific_confirmation",
                "status",
                "technical_valid",
                "device",
                "conditions",
                "fixture_outcome_counts",
                "maximum_residuals",
                "block",
                "retention",
                "resource_executor",
                "resources",
            ],
        ),
        "launch_plan_schema.json": _schema(
            "Concrete Q0-Q5 execution graph",
            [
                "schema_version",
                "campaign_id",
                "bundle_root",
                "dataset",
                "tokenizer",
                "registry",
                "orders",
                "resource_budget",
                "nodes",
                "analysis_command",
            ],
        ),
    }


def readme() -> str:
    return f"""# Block-normal row-map confirmation

Campaign {CAMPAIGN_ID} tests the source-side block-normal hypotheses for all
288 registered maps and all 580,608 within-map row pairs. Protocol generation
is deterministic. Construction choices use seed 7444 only. Seeds 8444 and
9444 remain held out until block length, reduced rank, projection rule,
metric rule, perturbation amplitudes, and tolerances are sealed.

Q0 is a deterministic kernel, schema, executor, and recovery-fixture
qualification. It uses synthetic tensors and isolated fixture artifacts and
does not count as scientific confirmation. Checkpoint-backed implemented-model
measurements begin at Q1.

The campaign retains confirmed, not_confirmed, and unresolved scientific
outcomes. A scientific outcome never prunes later nodes. Technical invalidity
blocks only descendants. Complete checkpoints remain distinct from model-only
copies until every dependent transport, chart, block, comparison, and recovery
record closes.

The append-only evidence boundary is:

    python3 scripts/seal_block_normal_evidence.py source --help
    python3 scripts/seal_block_normal_evidence.py native --help

The source command accepts only source-computed operators, tube data, and
comparison bounds. The native command hashes that sealed source before it
opens a distinct successor checkpoint and requires exact comparison
membership.

Campaign staging also verifies the digest-bound shared registration
foundation before copying its registry and three frozen data orders.
"""


def expected() -> dict[str, bytes]:
    design = campaign_design()
    validate_design(design)
    files = {
        "README.md": readme().encode(),
        "campaign_design.json": canonical(design),
    }
    files.update({
        name: canonical(value) for name, value in schemas().items()
    })
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
            raise SystemExit("block-normal protocol membership is stale")
        for name, payload in files.items():
            if (OUTPUT / name).read_bytes() != payload:
                raise SystemExit(f"block-normal protocol is stale: {name}")
        print(f"block-normal protocols: verified ({len(files)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file() and path.name not in files:
            path.unlink()
    for name, payload in files.items():
        (OUTPUT / name).write_bytes(payload)
    print(
        "block-normal protocols: wrote "
        f"{OUTPUT.relative_to(ROOT)} ({len(files)} files)"
    )


if __name__ == "__main__":
    main()
