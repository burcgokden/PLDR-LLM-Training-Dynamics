#!/usr/bin/env python3
"""Generate strict protocols for source-resolved row-map confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "source_resolved_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.source_resolved_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    BLOCK_RESPONSE_SCHEMA,
    CAMPAIGN_ID,
    CHECKPOINT_BINDING_SCHEMA,
    CONSTRUCTION_POLICY_SCHEMA,
    EDGE_SCHEMA,
    INTERVENTION_SCHEMA,
    INTERVENTION_PREDICTION_SCHEMA,
    MASK_RESPONSE_SCHEMA,
    QUALIFICATION_SCHEMA,
    RANK_TUBE_SCHEMA,
    TIMECOURSE_SCHEMA,
    TIMEPOINT_SCHEMA,
    campaign_design,
    validate_design,
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


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


def _array(items: dict[str, Any], *, minimum: int = 0) -> dict[str, Any]:
    return {"type": "array", "items": items, "minItems": minimum}


STRING = {"type": "string", "minLength": 1}
DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
INTEGER = {"type": "integer"}
NONNEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
NUMBER = {"type": "number"}
NONNEGATIVE = {"type": "number", "minimum": 0.0}
BOOLEAN = {"type": "boolean"}


def _header(schema_version: str) -> dict[str, Any]:
    return {
        "schema_version": {"const": schema_version},
        "campaign_id": {"const": CAMPAIGN_ID},
    }


def _checks(names: tuple[str, ...]) -> dict[str, Any]:
    return _object(
        "Derived validity checks",
        {name: BOOLEAN for name in names},
    )


def schemas() -> dict[str, dict[str, Any]]:
    tensor = _object(
        "Checkpoint tensor binding",
        {
            "name": STRING,
            "shape": _array(NONNEGATIVE_INTEGER),
            "dtype": STRING,
            "sha256": DIGEST,
        },
    )
    checkpoint = _object(
        "Complete checkpoint content binding",
        {
            **_header(CHECKPOINT_BINDING_SCHEMA),
            "trajectory": STRING,
            "seed": INTEGER,
            "step": NONNEGATIVE_INTEGER,
            "checkpoint_path": STRING,
            "checkpoint_sha256": DIGEST,
            "content_sha256": DIGEST,
            "tensor_inventory": _array(tensor, minimum=1),
            "optimizer_state_sha256": DIGEST,
            "batch_sha256": DIGEST,
            "cursor": NONNEGATIVE_INTEGER,
            "code_sha256": DIGEST,
            "tokenizer_sha256": DIGEST,
            "dataset_sha256": DIGEST,
            "order_sha256": DIGEST,
            "operation_order_sha256": DIGEST,
            "created_at_ns": NONNEGATIVE_INTEGER,
            "checks": _checks((
                "checkpoint_is_mapping",
                "required_keys_present",
                "model_tensors_nonempty",
                "optimizer_bound",
                "batch_bound",
                "cursor_bound",
                "lineage_bound",
                "content_digest_replays",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    map_identity = {
        "map_id": STRING,
        "context": STRING,
        "layer": NONNEGATIVE_INTEGER,
        "head": NONNEGATIVE_INTEGER,
    }
    edge_map = _object(
        "One map on one native edge",
        {
            **map_identity,
            "source_energy": NONNEGATIVE,
            "endpoint_energy": NONNEGATIVE,
            "source_diameter": NONNEGATIVE,
            "endpoint_diameter": NONNEGATIVE,
            "source_gain": NONNEGATIVE,
            "gain_defined": BOOLEAN,
            "source_names": _array(STRING, minimum=1),
            "radial_dissipation": _array(NUMBER, minimum=1),
            "charge_gram": _array(_array(NUMBER, minimum=1), minimum=1),
            "gate_work": NUMBER,
            "shape_work": NUMBER,
            "gate_charge": NONNEGATIVE,
            "shape_charge": NONNEGATIVE,
            "gate_shape_interaction": NUMBER,
            "native_radius": NONNEGATIVE,
            "native_energy_upper": NONNEGATIVE,
            "transport_relative_residual": NONNEGATIVE,
            "energy_ledger_relative_residual": NONNEGATIVE,
            "shapley_efficiency_relative_residual": NONNEGATIVE,
        },
    )
    edge = _object(
        "Complete source transport edge",
        {
            **_header(EDGE_SCHEMA),
            "trajectory": STRING,
            "seed": INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "source_checkpoint_binding_sha256": DIGEST,
            "endpoint_checkpoint_binding_sha256": DIGEST,
            "producer_command_sha256": DIGEST,
            "map_records": _array(edge_map, minimum=1),
            "checks": _checks((
                "checkpoint_chronology",
                "map_population_complete",
                "finite_transport_closed",
                "energy_ledger_closed",
                "shapley_efficiency_closed",
                "native_link_closed",
                "all_values_finite",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    coordinate_tensor = _object(
        "AdamW coordinate partition for one tensor",
        {
            "name": STRING,
            "coordinate_count": NONNEGATIVE_INTEGER,
            "active_count": NONNEGATIVE_INTEGER,
            "structurally_inactive_count": NONNEGATIVE_INTEGER,
            "unresolved_count": NONNEGATIVE_INTEGER,
            "zero_second_moment_indices": _array(NONNEGATIVE_INTEGER),
            "structurally_inactive_indices": _array(NONNEGATIVE_INTEGER),
            "zero_second_moment_indices_sha256": DIGEST,
            "structural_proof_sha256": DIGEST,
        },
    )
    response = _object(
        "One row-observable response",
        {
            **map_identity,
            "direction_id": STRING,
            "direction_class": {
                "enum": ["energy-gradient", "random-audit", "krylov"]
            },
            "parameter_response": NONNEGATIVE,
            "row_response": NONNEGATIVE,
            "finite": BOOLEAN,
        },
    )
    mask_response = _object(
        "AdamW coordinate mask and one-update response",
        {
            **_header(MASK_RESPONSE_SCHEMA),
            "checkpoint_binding_sha256": DIGEST,
            "coordinate_tensors": _array(coordinate_tensor, minimum=1),
            "responses": _array(response, minimum=1),
            "unresolved_coordinate_count": NONNEGATIVE_INTEGER,
            "smooth_full_state_eligible": BOOLEAN,
            "checks": _checks((
                "coordinate_partition_exhaustive",
                "structural_proofs_bound",
                "no_division_before_mask",
                "response_population_complete",
                "all_responses_finite",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    krylov_action = _object(
        "One lifted block Krylov action",
        {
            "direction_class": {"const": "krylov"},
            "direction_index": NONNEGATIVE_INTEGER,
            "input_state_norm": NONNEGATIVE,
            "output_state_norm": NONNEGATIVE,
            "source_row_response_norm": NONNEGATIVE,
            "endpoint_row_response_norm": NONNEGATIVE,
            "row_response_gain": NONNEGATIVE,
            "arnoldi_residual_norm": NONNEGATIVE,
        },
    )
    audit_action = _object(
        "One independent lifted block audit",
        {
            "direction_class": {"const": "orthogonal-audit"},
            "direction_index": NONNEGATIVE_INTEGER,
            "input_state_norm": NONNEGATIVE,
            "output_state_norm": NONNEGATIVE,
            "source_row_response_norm": NONNEGATIVE,
            "endpoint_row_response_norm": NONNEGATIVE,
            "row_response_gain": NONNEGATIVE,
            "omitted_state_response_norm": NONNEGATIVE,
        },
    )
    block_resources = _object(
        "Reduced block resources",
        {
            "action_wall_seconds": NONNEGATIVE,
            "peak_gpu_reserved_bytes": NONNEGATIVE_INTEGER,
        },
    )
    block_response = _object(
        "Matrix-free reduced lifted AdamW block response",
        {
            **_header(BLOCK_RESPONSE_SCHEMA),
            "checkpoint_binding_sha256": _array(DIGEST, minimum=1),
            "edge_sha256": _array(DIGEST, minimum=1),
            "mask_response_sha256": DIGEST,
            "block_start": NONNEGATIVE_INTEGER,
            "block_length": NONNEGATIVE_INTEGER,
            "state_scope": {
                "const": "parameters-first-moments-second-moments"
            },
            "reduced_operator_kind": {
                "const": "euclidean-arnoldi-diagnostic"
            },
            "reduced_rank_requested": NONNEGATIVE_INTEGER,
            "reduced_rank_achieved": NONNEGATIVE_INTEGER,
            "krylov_breakdown": BOOLEAN,
            "hessenberg": _array(_array(NUMBER)),
            "ritz_modulus": _array(NONNEGATIVE),
            "reduced_singular_values": _array(NONNEGATIVE),
            "krylov_actions": _array(krylov_action, minimum=1),
            "orthogonal_audits": _array(audit_action, minimum=1),
            "maximum_audit_omitted_response": NONNEGATIVE,
            "maximum_audit_row_gain": NONNEGATIVE,
            "smooth_full_state_eligible": BOOLEAN,
            "unresolved_coordinate_count": NONNEGATIVE_INTEGER,
            "basis_maximum_abs_inner_product": NONNEGATIVE,
            "resources": block_resources,
            "full_operator_theorem_claimed": {"const": False},
            "checks": _checks((
                "binding_sequence_complete",
                "native_edge_clone_closure",
                "coordinate_mask_opened",
                "krylov_actions_finite",
                "audit_population_complete",
                "basis_orthogonality_within_tolerance",
                "reduced_operator_labeled",
                "full_operator_not_claimed",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    rank_record = _object(
        "Constraint rank record",
        {
            **map_identity,
            "operator_scope": STRING,
            "constraint_dimension_exact": BOOLEAN,
            "target_row_count": NONNEGATIVE_INTEGER,
            "relative_threshold": NONNEGATIVE,
            "estimated_numerical_rank": NONNEGATIVE_INTEGER,
            "exact_algebraic_rank": NONNEGATIVE_INTEGER,
            "smallest_relative_singular_estimate": NONNEGATIVE,
            "condition_estimate": NONNEGATIVE,
            "rank_threshold_pass": BOOLEAN,
        },
    )
    analytic_bounds = _object(
        "Analytic local bounds and diagnostic radius",
        {
            "layernorm_hessian_upper": NONNEGATIVE,
            "plga_z_derivative_upper": NONNEGATIVE,
            "plga_z_second_derivative_upper": NONNEGATIVE,
            "block_quadratic_coefficient_upper": NONNEGATIVE,
            "observed_orbit_radius_gain": NONNEGATIVE,
            "sampled_block_row_gain": NONNEGATIVE,
            "diagnostic_radius_gain_used": NONNEGATIVE,
            "diagnostic_radius_lower": NONNEGATIVE,
            "diagnostic_radius_upper": {
                "type": ["number", "null"], "minimum": 0.0
            },
            "diagnostic_radius_eligible": BOOLEAN,
            "realized_displacement": NONNEGATIVE,
            "realized_displacement_inside_diagnostic_radius": BOOLEAN,
            "diagnostic_only": {"const": True},
        },
    )
    block_test = _object(
        "Reduced empirical block screen",
        {
            "block_start": NONNEGATIVE_INTEGER,
            "block_length": NONNEGATIVE_INTEGER,
            "reduced_krylov_rank": NONNEGATIVE_INTEGER,
            "audit_direction_count": NONNEGATIVE_INTEGER,
            "audit_response_squared_maximum": NONNEGATIVE,
            "metric_cap": NONNEGATIVE,
            "metric_cap_screen_pass": BOOLEAN,
            "metric_upper_bound_certified": {"const": False},
            "observed_block_diameter_envelope": NONNEGATIVE,
            "source_diameter": NONNEGATIVE,
            "layernorm_baseline": NONNEGATIVE,
            "physical_scale_screen_pass": BOOLEAN,
            "audit_response_maximum": NONNEGATIVE,
            "full_operator_theorem_claimed": {"const": False},
        },
    )
    rank_tube = _object(
        "Interface rank and diagnostic local-tube evidence",
        {
            **_header(RANK_TUBE_SCHEMA),
            "source_checkpoint_binding_sha256": DIGEST,
            "endpoint_checkpoint_binding_sha256": DIGEST,
            "mask_response_sha256": _array(DIGEST, minimum=1),
            "edge_sha256": _array(DIGEST, minimum=1),
            "block_response_sha256": DIGEST,
            "per_map_rank": _array(rank_record, minimum=1),
            "stacked_rank": rank_record,
            "analytic_bounds": analytic_bounds,
            "block_test": block_test,
            "local_robustness_outcome": {
                "enum": list(("confirmed", "not_confirmed", "unresolved"))
            },
            "checks": _checks((
                "rank_threshold_frozen",
                "rank_diagnostics_finite",
                "analytic_bounds_not_fitted",
                "diagnostic_radius_labeled",
                "full_state_rank_not_inferred",
                "sampled_response_not_promoted_to_bound",
                "lifted_block_response_bound",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    fixed_288 = {"type": "array", "items": NONNEGATIVE, "minItems": 288, "maxItems": 288}
    fixed_64 = {"type": "array", "items": NONNEGATIVE, "minItems": 64, "maxItems": 64}
    fixed_signed_64 = {"type": "array", "items": NUMBER, "minItems": 64, "maxItems": 64}
    fixed_3x64 = {"type": "array", "items": fixed_signed_64, "minItems": 3, "maxItems": 3}
    pair = {"type": "array", "items": NONNEGATIVE_INTEGER, "minItems": 2, "maxItems": 2}
    timepoint = _object(
        "One native online row-map timepoint",
        {
            **_header(TIMEPOINT_SCHEMA),
            "step": NONNEGATIVE_INTEGER,
            "registry_sha256": DIGEST,
            "map_order": {"const": "context-major-layer-major-head-major-v1"},
            "energy": fixed_288,
            "diameter_squared": fixed_288,
            "maximizing_pairs": {"type": "array", "items": pair, "minItems": 288, "maxItems": 288},
            "gate_shape_coordinate_energy": {"type": "array", "items": fixed_64, "minItems": 288, "maxItems": 288},
            "normalized_shape_coordinate_energy": {"type": "array", "items": fixed_64, "minItems": 288, "maxItems": 288},
            "layer_final_gate": fixed_3x64,
            "coordinate_factorization_relative_residual": NONNEGATIVE,
            "plga_quotient_ratio": fixed_288,
            "plga_force": fixed_288,
            "cross_map_pairs_formed": {"const": False},
        },
    )

    time_map = _object(
        "One map at one scheduled update",
        {
            **map_identity,
            "update": NONNEGATIVE_INTEGER,
            "energy": NONNEGATIVE,
            "diameter": NONNEGATIVE,
            "gate_shape_coordinate_energy": _array(NONNEGATIVE, minimum=1),
            "normalized_shape_coordinate_energy": _array(NONNEGATIVE, minimum=1),
            "source_gain_from_previous": NONNEGATIVE,
            "gain_defined": BOOLEAN,
            "cumulative_gain": {"type": ["number", "null"], "minimum": 0.0},
            "cumulative_log_gain": {"type": ["number", "null"]},
            "plga_quotient_ratio": NONNEGATIVE,
            "plga_force": NONNEGATIVE,
        },
    )
    gate_record = _object(
        "Final gates at one scheduled update",
        {
            "update": NONNEGATIVE_INTEGER,
            "layer_final_gate": fixed_3x64,
            "coordinate_factorization_relative_residual": NONNEGATIVE,
        },
    )
    timecourse = _object(
        "Complete finite row-energy time course",
        {
            **_header(TIMECOURSE_SCHEMA),
            "trajectory": STRING,
            "seed": INTEGER,
            "role": {"enum": ["construction", "heldout"]},
            "policy_sha256": DIGEST,
            "updates": _array(NONNEGATIVE_INTEGER, minimum=1),
            "checkpoint_binding_sha256": _array(DIGEST, minimum=1),
            "map_records": _array(time_map, minimum=1),
            "gate_records": _array(gate_record, minimum=1),
            "plga_selected_cap": {"type": ["number", "null"]},
            "checks": _checks((
                "cadence_exact",
                "terminal_present",
                "map_population_complete",
                "no_omitted_gain_factor",
                "cumulative_gain_replays",
                "coordinate_factorization_closed",
                "policy_precedes_heldout",
                "all_values_finite",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    prediction_map = _object(
        "One source-only intervention prediction",
        {
            **map_identity,
            "source_predicted_energy_change": NUMBER,
        },
    )
    intervention_prediction = _object(
        "Prediction frozen before native intervention endpoints",
        {
            **_header(INTERVENTION_PREDICTION_SCHEMA),
            "source_checkpoint_binding_sha256": DIGEST,
            "source_checkpoint_sha256": DIGEST,
            "source_step": NONNEGATIVE_INTEGER,
            "window": {"enum": ["pre-onset", "onset", "late"]},
            "intervention": STRING,
            "created_at_ns": NONNEGATIVE_INTEGER,
            "records": _array(prediction_map, minimum=1),
            "checks": _checks((
                "source_checkpoint_opened",
                "source_step_matches_frozen_window",
                "control_and_arm_functionally_derived",
                "map_population_complete",
                "all_values_finite",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    intervention_record = _object(
        "One paired intervention response",
        {
            **map_identity,
            "window": {"enum": ["pre-onset", "onset", "late"]},
            "intervention": STRING,
            "source_predicted_energy_change": NUMBER,
            "native_energy_change": NUMBER,
            "prediction_residual": NUMBER,
            "relative_prediction_residual": NONNEGATIVE,
            "control_checkpoint_sha256": DIGEST,
            "intervention_checkpoint_sha256": DIGEST,
        },
    )
    intervention = _object(
        "Controlled intervention evidence",
        {
            **_header(INTERVENTION_SCHEMA),
            "construction_policy_sha256": DIGEST,
            "prediction_sha256": DIGEST,
            "prediction_created_at_ns": NONNEGATIVE_INTEGER,
            "native_opened_at_ns": NONNEGATIVE_INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "prediction_relative_tolerance": NONNEGATIVE,
            "window_updates": _object(
                "Frozen intervention windows",
                {
                    "pre-onset": NONNEGATIVE_INTEGER,
                    "onset": NONNEGATIVE_INTEGER,
                    "late": NONNEGATIVE_INTEGER,
                },
            ),
            "records": _array(intervention_record, minimum=1),
            "checks": _checks((
                "windows_frozen_before_interventions",
                "source_step_matches_policy_window",
                "paired_batches_equal",
                "paired_rng_equal",
                "source_prediction_precedes_native",
                "checkpoint_links_exact",
                "map_population_complete",
                "all_values_finite",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    coarse_certificate = _object(
        "Construction-selected coarse source certificate",
        {
            "source_names": _array(STRING, minimum=1),
            "radial_dissipation_lower": _array(NUMBER, minimum=1),
            "signed_gram_sum_upper": NUMBER,
            "native_charge_upper": NONNEGATIVE,
            "selection_quantile": NONNEGATIVE,
        },
    )
    construction_policy = _object(
        "Policy sealed after construction and before heldout launch",
        {
            **_header(CONSTRUCTION_POLICY_SCHEMA),
            "design_sha256": DIGEST,
            "construction_timecourse_sha256": DIGEST,
            "construction_edge_sha256": _array(DIGEST, minimum=1),
            "created_at_ns": NONNEGATIVE_INTEGER,
            "plga_selected_cap": {"type": ["number", "null"]},
            "window_updates": _object(
                "Frozen intervention windows",
                {
                    "pre-onset": NONNEGATIVE_INTEGER,
                    "onset": NONNEGATIVE_INTEGER,
                    "late": NONNEGATIVE_INTEGER,
                },
            ),
            "coarse_certificate": coarse_certificate,
            "checks": _checks((
                "construction_only",
                "windows_derived_deterministically",
                "plga_cap_from_frozen_menu",
                "edge_population_nonempty",
                "policy_digest_inputs_bound",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    outcome_count = _object(
        "Scientific outcome counts",
        {
            "confirmed": NONNEGATIVE_INTEGER,
            "not_confirmed": NONNEGATIVE_INTEGER,
            "unresolved": NONNEGATIVE_INTEGER,
            "technically_invalid": NONNEGATIVE_INTEGER,
        },
    )
    grouped_outcome = _object(
        "Closed grouped outcome count",
        {"key": STRING, "counts": outcome_count},
    )
    outcome_record = _object(
        "Retained mapwise scientific outcome",
        {
            **map_identity,
            "stage": STRING,
            "trajectory": STRING,
            "criterion": STRING,
            "observed": NUMBER,
            "outcome": {"enum": ["confirmed", "not_confirmed", "unresolved", "technically_invalid"]},
        },
    )
    analysis = _object(
        "Source-resolved campaign analysis",
        {
            **_header(ANALYSIS_SCHEMA),
            "input_sha256": _array(DIGEST, minimum=1),
            "map_record_count": NONNEGATIVE_INTEGER,
            "outcome_counts": outcome_count,
            "outcomes_by_layer": _object(
                "Layer outcome map",
                {"0": outcome_count, "1": outcome_count, "2": outcome_count},
            ),
            "outcomes_by_head": _object(
                "Head outcome map",
                {"0": outcome_count, "1": outcome_count, "2": outcome_count, "3": outcome_count},
            ),
            "outcomes_by_context": _array(grouped_outcome),
            "outcomes_by_stage": _array(grouped_outcome),
            "outcomes_by_trajectory": _array(grouped_outcome),
            "outcome_records": _array(outcome_record, minimum=1),
            "finite_horizon_only": {"const": True},
            "negative_outcomes_retained": {"const": True},
            "technical_valid": BOOLEAN,
        },
    )

    qualification = _object(
        "Kernel and device qualification",
        {
            **_header(QUALIFICATION_SCHEMA),
            "mode": {"enum": ["fixture", "real-checkpoint"]},
            "device": STRING,
            "checkpoint_binding_sha256": {"type": ["string", "null"]},
            "maximum_identity_residuals": _object(
                "Numerical identity residuals",
                {
                    "gate_shape": NONNEGATIVE,
                    "symmetric_transport": NONNEGATIVE,
                    "energy_ledger": NONNEGATIVE,
                    "shapley_efficiency": NONNEGATIVE,
                },
            ),
            "peak_gpu_allocated_bytes": NONNEGATIVE_INTEGER,
            "peak_gpu_reserved_bytes": NONNEGATIVE_INTEGER,
            "peak_host_rss_bytes": NONNEGATIVE_INTEGER,
            "checks": _checks((
                "kernels_pass",
                "schema_rejects_unknown_fields",
                "device_exercised",
                "checkpoint_content_opened",
                "recovery_pass",
                "forced_cap_record_pass",
            )),
            "technical_valid": BOOLEAN,
        },
    )

    result = {
        "checkpoint_binding.schema.json": checkpoint,
        "edge.schema.json": edge,
        "mask_response.schema.json": mask_response,
        "block_response.schema.json": block_response,
        "rank_tube.schema.json": rank_tube,
        "timepoint.schema.json": timepoint,
        "timecourse.schema.json": timecourse,
        "intervention_prediction.schema.json": intervention_prediction,
        "intervention.schema.json": intervention,
        "construction_policy.schema.json": construction_policy,
        "analysis.schema.json": analysis,
        "qualification.schema.json": qualification,
    }
    for schema in result.values():
        schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    return result


def readme() -> str:
    return f"""# Source-resolved row-map confirmation

Campaign `{CAMPAIGN_ID}` measures the physical row quotient directly.  It
records exact gate-shape and finite-source ledgers, AdamW coordinate strata,
rank and analytic tube eligibility, full finite time courses, and controlled
interventions.  Construction choices are locked before either heldout order
is opened.

Every scientific node is a producer that opens a complete checkpoint or
creates one with the registered trainer.  Assemblers only consume strict,
content-bound producer artifacts.  A noncheckpoint file, caller-supplied
operator, unknown field, nonfinite number, stale digest, or incomplete map
population fails before analysis.

Generate and stage with:

    python3 scripts/gen_source_resolved_protocols.py
    python3 scripts/stage_source_resolved_confirmation.py

Check without mutation with the same commands followed by `--check`.
"""


def expected() -> dict[str, bytes]:
    design = campaign_design()
    validate_design(design)
    files = {
        "README.md": readme().encode("utf-8"),
        "campaign_design.json": canonical(design),
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
            {path.name for path in OUTPUT.iterdir() if path.is_file()}
            if OUTPUT.is_dir() else set()
        )
        if actual != set(files):
            raise SystemExit("source-resolved protocol membership is stale")
        for name, payload in files.items():
            if (OUTPUT / name).read_bytes() != payload:
                raise SystemExit(f"source-resolved protocol is stale: {name}")
        print(f"source-resolved protocols: verified ({len(files)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file() and path.name not in files:
            path.unlink()
    for name, payload in files.items():
        (OUTPUT / name).write_bytes(payload)
    print(
        "source-resolved protocols: wrote "
        f"{OUTPUT.relative_to(ROOT)} ({len(files)} files)"
    )


if __name__ == "__main__":
    main()
