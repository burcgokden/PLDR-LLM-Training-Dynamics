#!/usr/bin/env python3
"""Generate strict protocols for the orbitwise row-map campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "orbitwise_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.orbitwise_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    DENSE_TIMEPOINT_SCHEMA,
    FULL_TIMEPOINT_SCHEMA,
    GATE_PREDICTION_SCHEMA,
    GATE_RESULT_SCHEMA,
    QUALIFICATION_SCHEMA,
    REGISTRY,
    SOURCE_LINK_SCHEMA,
    SOURCE_PREDICTION_SCHEMA,
    campaign_design,
    validate_design,
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    compact = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _object(
    title: str,
    properties: dict[str, Any],
    *,
    required: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "title": title,
        "properties": properties,
        "required": list(required if required is not None else properties),
        "additionalProperties": False,
    }


def _array(items: dict[str, Any], count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "array", "items": items}
    if count is not None:
        result.update({"minItems": count, "maxItems": count})
    return result


STRING = {"type": "string", "minLength": 1}
DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
INTEGER = {"type": "integer"}
NONNEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
NUMBER = {"type": "number"}
NONNEGATIVE = {"type": "number", "minimum": 0.0}
BOOLEAN = {"type": "boolean"}


def _checks(names: tuple[str, ...]) -> dict[str, Any]:
    return _object("Derived checks", {name: BOOLEAN for name in names})


def _timepoint(schema: str, maps: int) -> dict[str, Any]:
    coordinate = _array(NONNEGATIVE, REGISTRY["feature_count"])
    pair = _array(NONNEGATIVE_INTEGER, 2)
    return _object(
        "Native orbitwise row-map timepoint",
        {
            "schema_version": {"const": schema},
            "campaign_id": {"const": CAMPAIGN_ID},
            "step": NONNEGATIVE_INTEGER,
            "registry_sha256": DIGEST,
            "map_order": {"const": REGISTRY["map_order"]},
            "energy": _array(NONNEGATIVE, maps),
            "diameter_squared": _array(NONNEGATIVE, maps),
            "maximizing_pairs": _array(pair, maps),
            "gate_shape_coordinate_energy": _array(coordinate, maps),
            "normalized_shape_coordinate_energy": _array(coordinate, maps),
            "layer_final_gate": _array(
                _array(NUMBER, REGISTRY["feature_count"]),
                len(REGISTRY["layers"]),
            ),
            "coordinate_factorization_relative_residual": NONNEGATIVE,
            "plga_quotient_ratio": _array(NONNEGATIVE, maps),
            "plga_absolute_response": _array(NONNEGATIVE, maps),
            "plga_force": _array(NONNEGATIVE, maps),
            "cross_map_pairs_formed": {"const": False},
        },
    )


def schemas() -> dict[str, dict[str, Any]]:
    source_map = _object(
        "Prospective source prediction for one map",
        {
            "map_id": STRING,
            "source_energy": NONNEGATIVE,
            "predicted_energy": NONNEGATIVE,
            "predicted_reopening": NONNEGATIVE,
            "gain_defined": BOOLEAN,
            "source_gain": NONNEGATIVE,
            "radial_dissipation": _array(NUMBER, 3),
            "charge_gram": _array(_array(NUMBER, 3), 3),
            "predicted_native_parameter_sha256": DIGEST,
        },
    )
    source_prediction = _object(
        "Source-only reopening prediction",
        {
            "schema_version": {"const": SOURCE_PREDICTION_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "trajectory": STRING,
            "seed": INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "source_binding_sha256": DIGEST,
            "producer_command_sha256": DIGEST,
            "created_at_ns": NONNEGATIVE_INTEGER,
            "map_records": _array(source_map, REGISTRY["map_count"]),
            "checks": _checks((
                "source_checkpoint_opened",
                "map_population_complete",
                "functional_step_finite",
                "source_ledger_closed",
                "prediction_has_no_native_successor_input",
            )),
            "technical_valid": BOOLEAN,
        },
    )
    link_map = _object(
        "Native linkage of one prospective reopening prediction",
        {
            "map_id": STRING,
            "source_energy": NONNEGATIVE,
            "endpoint_energy": NONNEGATIVE,
            "realized_reopening": NONNEGATIVE,
            "source_reopening_upper": NONNEGATIVE,
            "native_radius": NONNEGATIVE,
            "native_charge": NONNEGATIVE,
            "bound_slack": NUMBER,
            "zero_source_energy": BOOLEAN,
        },
    )
    source_link = _object(
        "Prospective source-to-native reopening link",
        {
            "schema_version": {"const": SOURCE_LINK_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "trajectory": STRING,
            "seed": INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "prediction_sha256": DIGEST,
            "endpoint_binding_sha256": DIGEST,
            "created_at_ns": NONNEGATIVE_INTEGER,
            "map_records": _array(link_map, REGISTRY["map_count"]),
            "checks": _checks((
                "prediction_precedes_successor_binding",
                "checkpoint_chronology",
                "native_parameter_link_closed",
                "map_population_complete",
                "all_values_finite",
            )),
            "scientific_checks": _checks((
                "source_reopening_bound_closed",
            )),
            "technical_valid": BOOLEAN,
        },
    )
    gate_prediction = _object(
        "Sealed final-gate intervention direction",
        {
            "schema_version": {"const": GATE_PREDICTION_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "trajectory": STRING,
            "seed": INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "horizon": {"const": 64},
            "source_checkpoint_sha256": DIGEST,
            "prediction": {"const": "freeze-retains-more-physical-energy"},
            "created_at_ns": NONNEGATIVE_INTEGER,
            "technical_valid": {"const": True},
        },
    )
    gate_map = _object(
        "Paired final-gate intervention result for one map",
        {
            "map_id": STRING,
            "control_energy": NONNEGATIVE,
            "frozen_gate_energy": NONNEGATIVE,
            "frozen_minus_control_energy": NUMBER,
            "supports_registered_direction": BOOLEAN,
        },
    )
    gate_result = _object(
        "Paired final-gate intervention result",
        {
            "schema_version": {"const": GATE_RESULT_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "horizon": {"const": 64},
            "prediction_sha256": DIGEST,
            "control_log_sha256": DIGEST,
            "frozen_log_sha256": DIGEST,
            "map_records": _array(gate_map, REGISTRY["map_count"]),
            "checks": _checks((
                "prediction_precedes_branches",
                "endpoint_present",
                "map_population_complete",
                "all_values_finite",
            )),
            "technical_valid": BOOLEAN,
        },
    )
    qualification = _object(
        "Orbitwise kernel and device qualification",
        {
            "schema_version": {"const": QUALIFICATION_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "device": STRING,
            "peak_gpu_allocated_bytes": NONNEGATIVE_INTEGER,
            "peak_gpu_reserved_bytes": NONNEGATIVE_INTEGER,
            "checks": _checks((
                "design_canonical",
                "effective_gate_bounds",
                "three_channel_identity",
                "reopening_envelope",
                "zero_energy_reopening",
                "schema_rejects_unknown_field",
                "requested_device_exercised",
            )),
            "technical_valid": BOOLEAN,
        },
    )
    summary = _object(
        "Finite channel summary",
        {
            "minimum": NONNEGATIVE,
            "median": NONNEGATIVE,
            "p90": NONNEGATIVE,
            "maximum": NONNEGATIVE,
        },
    )
    trajectory = _object(
        "One trajectory analysis",
        {
            "trajectory": STRING,
            "seed": INTEGER,
            "role": {"enum": ["construction", "transfer"]},
            "full_snapshot_count": NONNEGATIVE_INTEGER,
            "dense_snapshot_count": NONNEGATIVE_INTEGER,
            "map_count": {"const": REGISTRY["map_count"]},
            "terminal_update": NONNEGATIVE_INTEGER,
            "terminal_physical_gain": summary,
            "terminal_shape_gain": summary,
            "terminal_gate_only_gain": summary,
            "terminal_alignment_gain": summary,
            "gain_defined_map_count": NONNEGATIVE_INTEGER,
            "log_gain_defined_map_count": NONNEGATIVE_INTEGER,
            "zero_initial_physical_energy_count": NONNEGATIVE_INTEGER,
            "initial_plga_absolute_response": summary,
            "terminal_plga_absolute_response": summary,
            "initial_plga_force": summary,
            "terminal_plga_force": summary,
            "maximum_factorization_relative_residual": NONNEGATIVE,
            "maximum_three_channel_gain_residual": NONNEGATIVE,
            "maximum_three_channel_log_residual": NONNEGATIVE,
            "maximum_reopening_envelope_violation": NONNEGATIVE,
            "dense_blocks": _array(_object(
                "Dense reopening block",
                {
                    "center": NONNEGATIVE_INTEGER,
                    "edge_count": {"const": 64},
                    "map_count": {"const": REGISTRY["sentinel_map_count"]},
                    "anchor_energy": summary,
                    "reopening_budget": summary,
                    "block_maximum": summary,
                    "envelope_slack": summary,
                },
            ), 4),
            "technical_valid": BOOLEAN,
        },
    )
    analysis = _object(
        "Complete finite orbitwise analysis",
        {
            "schema_version": {"const": ANALYSIS_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "finite_horizon_only": {"const": True},
            "trajectory_count": {"const": 3},
            "trajectories": _array(trajectory, 3),
            "source_links": _array(STRING, 8),
            "source_link_map_record_count": {
                "const": 8 * REGISTRY["map_count"],
            },
            "source_reopening_bound_holds": BOOLEAN,
            "source_reopening_minimum_slack": NUMBER,
            "gate_interventions": _array(STRING, 3),
            "gate_intervention_map_record_count": {
                "const": 3 * REGISTRY["map_count"],
            },
            "gate_registered_direction_support_count": NONNEGATIVE_INTEGER,
            "gate_registered_direction_support_fraction": NONNEGATIVE,
            "all_input_artifacts_technical": BOOLEAN,
            "technical_valid": BOOLEAN,
        },
    )
    return {
        "full_timepoint.schema.json": _timepoint(
            FULL_TIMEPOINT_SCHEMA, REGISTRY["map_count"]
        ),
        "dense_timepoint.schema.json": _timepoint(
            DENSE_TIMEPOINT_SCHEMA, REGISTRY["sentinel_map_count"]
        ),
        "source_prediction.schema.json": source_prediction,
        "source_link.schema.json": source_link,
        "gate_prediction.schema.json": gate_prediction,
        "gate_result.schema.json": gate_result,
        "qualification.schema.json": qualification,
        "analysis.schema.json": analysis,
    }


README = """# Orbitwise row-map confirmation protocol

This generated directory freezes the 18,000-update schedule, three seeds,
full and dense observer populations, source edges, gate interventions,
tolerances, and resource caps.

Regenerate or verify from the repository root:

    python3 scripts/gen_orbitwise_protocols.py
    python3 scripts/gen_orbitwise_protocols.py --check
    python3 scripts/stage_orbitwise_confirmation.py --check
"""


def generated_files() -> dict[Path, bytes]:
    design = campaign_design()
    validate_design(design)
    signed_design = dict(design)
    signed_design["spec_sha256"] = digest_object(design)
    files = {
        Path("campaign_design.json"): canonical(design),
        Path("training_campaign_spec.json"): canonical(signed_design),
        Path("README.md"): README.encode("utf-8"),
    }
    files.update({
        Path(name): canonical(schema)
        for name, schema in schemas().items()
    })
    checksums = "".join(
        f"{hashlib.sha256(files[path]).hexdigest()}  {path.as_posix()}\n"
        for path in sorted(files)
    ).encode("ascii")
    files[Path("CHECKSUMS.sha256")] = checksums
    return files


def _existing_files() -> set[Path]:
    if not OUTPUT.is_dir():
        return set()
    return {
        path.relative_to(OUTPUT)
        for path in OUTPUT.rglob("*")
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    expected = generated_files()
    if arguments.check:
        if _existing_files() != set(expected):
            raise SystemExit("orbitwise protocol membership is stale")
        for relative, payload in expected.items():
            if (OUTPUT / relative).read_bytes() != payload:
                raise SystemExit(
                    f"orbitwise protocol is stale: {relative.as_posix()}"
                )
        print(f"orbitwise protocols: verified ({len(expected)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for existing in sorted(_existing_files() - set(expected)):
        (OUTPUT / existing).unlink()
    for relative, payload in expected.items():
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print(f"orbitwise protocols: wrote {len(expected)} files")


if __name__ == "__main__":
    main()
