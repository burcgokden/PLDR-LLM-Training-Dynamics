#!/usr/bin/env python3
"""Generate the frozen observable-margin confirmation protocols."""

from __future__ import annotations
from companion_paths import acquisition_identity

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.pycache_prefix = "/dev/null"


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from observable_margin_specs import (  # noqa: E402
    ANCHORS,
    ARCHITECTURE,
    ARITHMETIC,
    BRIDGE_SECTORS,
    CAMPAIGN_ID,
    INTERVENTION_COMPARISON_NAMES,
    INTERVENTION_OPERATOR_BY_NAME,
    INTERVENTION_OPERATOR_CONTRACTS,
    INTERVENTION_OPERATOR_FIELDS,
    OBSERVABLE_CONTRACT,
    REGISTRY,
    RESOURCE_CAPS,
    RUNS,
    SCHEMA_VERSION,
    STAGES,
    STATISTICS,
    STATUS,
    STOP_GATES,
    TRANSFER,
    VALIDATION_FIXTURES,
)


OUTPUT = ROOT / "experiments" / "protocols" / "observable_margin_confirmation"
FINAL_RELEASE_COMMAND = ("make", "release-check-current")
FINAL_RELEASE_PROVENANCE = {
    "canonical_command": list(FINAL_RELEASE_COMMAND),
    "realized_environment": {
        "required_report_fields": [
            "environment_attestation_path",
            "environment_attestation_sha256",
            "environment_import_root",
            "environment_package_inventory_count",
            "environment_package_inventory_sha256",
            "environment_replay_pass",
            "environment_replay_error",
        ],
        "required_attestation_fields": [
            "python.import_root",
            "installed_distributions.count",
            "installed_distributions.sha256",
        ],
        "rule": (
            "The digest-bound realized attestation supplies the one resolved "
            "Python import root and the normalized installed-package inventory "
            "count and digest used by every child process."
        ),
    },
    "repository": {
        "live_manifest": "MANIFEST.sha256",
        "staged_manifest": "preflight/repository-MANIFEST.sha256",
        "required_container_fields": [
            "live_manifest_path",
            "staged_manifest_path",
            "before",
            "after",
            "stable_across_release_command",
        ],
        "required_snapshot_fields": [
            "pass",
            "target_count",
            "manifest_sha256",
            "staged_manifest_sha256",
            "staged_manifest_equal",
            "target_aggregate_sha256",
            "error",
        ],
        "rule": (
            "Freshly enumerate the exact current regular-file release closure, "
            "verify every MANIFEST target without expanding historical "
            "CHECKSUMS targets, and require the staged manifest seal to equal "
            "the live manifest byte for byte."
        ),
    },
    "manuscript_export": {
        "revision": acquisition_identity('observable-margin-manuscript-identity'),
        "strict_file_count": 18,
        "nonself_manifest_entry_count": 17,
        "required_container_fields": [
            "root_path",
            "live_manifest_path",
            "staged_manifest_path",
            "before",
            "after",
            "stable_across_release_command",
        ],
        "required_snapshot_fields": [
            "pass",
            "file_count",
            "manifest_entry_count",
            "manifest_sha256",
            "staged_manifest_sha256",
            "staged_manifest_equal",
            "target_aggregate_sha256",
            "error",
        ],
        "rule": (
            "Require exactly 18 regular files and the 17-entry nonself "
            "MANIFEST, reject missing, extra, symbolic-link, and nonregular "
            "nodes, and require the staged export seal to equal the live "
            "manifest byte for byte."
        ),
    },
    "independent_verification": {
        "top_level_report_fields": [
            "release_plan_paths_pass",
            "repository_release_command_exact",
            "repository_release_verification",
            "manuscript_export_verification",
        ],
        "phases": ["before", "after"],
        "rule": (
            "Independently recompute pass booleans, counts, manifest digests, "
            "staged-seal equality, and all-target aggregate digests before and "
            "after the canonical release command, then require exact stability."
        ),
    },
}


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


def expected_files():
    files = {}
    files[Path("campaign_design.json")] = _json({
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "status": STATUS,
        "architecture": ARCHITECTURE,
        "runs": list(RUNS),
        "anchors": ANCHORS,
        "registry": REGISTRY,
        "arithmetic": ARITHMETIC,
        "bridge_sector_order": list(BRIDGE_SECTORS),
        "observable_contract": OBSERVABLE_CONTRACT,
        "statistics": STATISTICS,
        "intervention_operator_contracts": list(
            INTERVENTION_OPERATOR_CONTRACTS),
        "transfer": TRANSFER,
        "resource_caps": RESOURCE_CAPS,
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "stop_gates": list(STOP_GATES),
        "ownership_rule": (
            "Seed 4444 owns cell and envelope selection. Seeds 5555 and "
            "6666 remain unopened until the construction lock is sealed."
        ),
        "final_release_gate": list(FINAL_RELEASE_COMMAND),
        "final_release_provenance": FINAL_RELEASE_PROVENANCE,
        "final_results_manifest": (
            "omission-sensitive pre-release results plus immutable "
            "campaign-ledger prefix; executor output binding seals the final "
            "release attestation"),
    })
    for stage in STAGES:
        stage_protocol = {
            "schema_version": SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "status": STATUS,
            "stage": stage,
            "native_producer": (
                "experiments/confirm/observable_margin_live.py"
                if stage["id"] in {"B", "H", "M"}
                else "experiments/confirm/observable_margin_controls.py"
                if stage["id"] in {"ROB", "I", "F"} else None),
            "planner": (
                "experiments/confirm/run_observable_margin_confirmation.py"),
            "analyzer": "experiments/analysis/analyze_observable_margin.py",
            "executor": "urn:pldr:unavailable:4760e88189d8163c866d60ff6ec5ba5e8f56f8a29b8806edb4c2341410733491",
            "evidence": "compact-npz-no-raw-parameter-nodes",
            "robustness_scientific_gate": False if stage["id"] == "ROB" else None,
        }
        if stage["id"] == "F":
            stage_protocol["final_release_provenance"] = FINAL_RELEASE_PROVENANCE
        files[Path(f"{stage['id'].lower()}_protocol.json")] = _json(stage_protocol)
    files[Path("validation_fixtures.json")] = _json({
        "schema_version": "pldr-observable-margin-fixtures-v1",
        "fixtures": VALIDATION_FIXTURES,
    })
    files[Path("record_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Observable-margin stage/job attempt record",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "stage", "job_id",
            "record_key", "attempt", "status", "role", "command",
            "native_arithmetic", "deciding_arithmetic", "inputs", "outputs",
            "checks", "resources", "failure", "record_sha256",
        ],
        "properties": {
            "schema_version": {
                "const": "pldr-observable-margin-stage-record-v1"},
            "campaign_id": {"const": CAMPAIGN_ID},
            "stage": {"enum": [stage["id"] for stage in STAGES]},
            "job_id": {"type": "string", "minLength": 1},
            "record_key": {"type": "string", "pattern": "^[A-Z]+:.+$"},
            "attempt": {"type": "integer", "minimum": 1},
            "status": {"enum": ["PASS", "FAIL", "BLOCKED"]},
            "role": {"enum": ["construction", "heldout", "shared", "control"]},
            "command": {"type": "array", "minItems": 1},
            "native_arithmetic": {"type": "string"},
            "deciding_arithmetic": {"type": "string"},
            "inputs": {"type": "array", "minItems": 1},
            "outputs": {"type": "array"},
            "checks": {"type": "object", "minProperties": 1},
            "resources": {"type": "object"},
            "failure": {"type": ["object", "null"]},
            "record_sha256": {
                "type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
    })
    files[Path("ledger_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Compact observable-margin NumPy ledger contract",
        "schema_version": "pldr-observable-margin-ledger-v1",
        "container": "npz-without-pickle",
        "required_scalars": [
            "campaign_id", "stage", "job_id", "seed", "role", "layer",
            "anchor", "block_length", "native_dtype", "shadow_dtype",
            "checkpoint_sha256", "checkpoint_complete_state_sha256",
            "checkpoint_branch_state_sha256",
            "checkpoint_measurement_registry_sha256", "gate_parameter_name",
            "checkpoint_gate_decay_mask_sha256", "probe_registry_sha256",
            "probe_chunk_ids_sha256", "union_graph_sha256", "graph_rule",
            "graph_transfer_mapping", "graph_vertex_count",
            "union_source_physical_sha256", "producer_sha256",
            "resource_resolved_device",
        ],
        "required_arrays": [
            "local_step", "node_probe_registry_sha256",
            "node_probe_chunk_ids_sha256", "update_batch_sha256", "gamma",
            "normalized_edge_contrast", "graph_edges", "graph_weights",
            "block_vertex_offsets", "block_head_index",
            "union_source_physical", "q", "linear_q", "quadratic_q",
            "adaptive_direction", "bridge_sector_direction",
            "source_microbatch_weight", "source_microbatch_nonpadding_count",
            "bridge_sector_microbatch_direction",
            "native_gate_first_moment_before",
            "native_gate_second_moment_before", "realized_clipped_gate_gradient",
            "shadow_gate_first_moment_after", "shadow_gate_second_moment_after",
            "shadow_gate_preconditioner", "optimizer_step_after", "adam_beta1",
            "adam_beta2", "adam_epsilon", "learning_rate", "weight_decay",
            "checkpoint_gate_decay_mask", "gate_decay_mask",
            "optimizer_gate_weight_decay", "roundoff_charge", "affine_factor",
            "affine_forcing", "vertex_diameter", "effective_resistance_max",
            "producer_energy", "plga_pair_id", "plga_context_index",
            "plga_head", "plga_anchor_row", "plga_generator",
            "plga_collapsed_anchor_generator", "plga_left", "plga_right",
            "plga_weight", "plga_bias", "plga_exponent", "plga_coupling",
            "plga_coupling_bias", "plga_coefficient", "plga_curvature_left",
            "plga_curvature_right", "plga_secant_prediction",
            "plga_secant_identity_residual", "plga_secant_operator_bound",
            "plga_generator_difference_norm", "plga_stacked_generator_bound",
            "plga_reference_centered_logits", "plga_candidate_centered_logits",
            "plga_reference_margin", "plga_reference_winner",
            "plga_candidate_winner", "plga_centered_logit_difference_norm",
            "plga_downstream_score_ratio", "plga_reference_query",
            "plga_candidate_query", "plga_reference_key", "plga_candidate_key",
            "plga_reference_pre_mask_score", "plga_candidate_pre_mask_score",
            "plga_score_query_term", "plga_score_generator_term",
            "plga_score_key_term", "plga_score_difference_prediction",
            "plga_score_identity_residual",
            "plga_query_operator_norm", "plga_key_operator_norm",
            "plga_score_difference_norm", "plga_candidate_cache_replay_residual",
        ],
        "forbidden": [
            "raw_model_node", "raw_optimizer_node", "raw_parameter_node",
            "normalized_shape", "normalized_shape_node", "pickle",
        ],
        "probe_invariant": REGISTRY["non_telescoping_guard"],
        "arithmetic": ARITHMETIC,
        "bridge_sector_order": list(BRIDGE_SECTORS),
        "shadow_moment_rule": (
            "Promote stored native m_t, v_t, and realized clipped gradient; "
            "recompute m/v, bias correction, preconditioner, and all seven "
            "sectors in float64 before measuring the native-successor charge."),
    })
    files[Path("registry_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Trainer-compatible observable probe and intervention registry",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "context_length", "construction", "validation",
            "anchors", "context_head_block_rule", "block_row_rule",
            "graph_transfer", "plga_binding_rule", "intervention_units",
            "intervention_unit_rule", "dataset_sha256", "tokenizer_sha256",
            "registry_sha256",
        ],
        "properties": {
            "schema_version": {"const": "pldr-observable-probe-registry-v1"},
            "context_length": {"const": 256},
            "construction": {
                "type": "array", "minItems": REGISTRY["construction_contexts"],
                "maxItems": REGISTRY["construction_contexts"]},
            "validation": {
                "type": "array", "minItems": REGISTRY["heldout_contexts"],
                "maxItems": REGISTRY["heldout_contexts"]},
            "anchors": {"const": ANCHORS["complete_optimizer_steps"]},
            "context_head_block_rule": {
                "const": REGISTRY["context_head_block_rule"]},
            "block_row_rule": {"const": REGISTRY["block_row_rule"]},
            "graph_transfer": {"const": REGISTRY["graph_transfer"]},
            "plga_binding_rule": {
                "const": "same-node-context-block-anchor-row-zero-b-locks-layer-node-v1"},
            "intervention_units": {
                "type": "array", "minItems": 24, "maxItems": 24},
            "intervention_unit_rule": {
                "const": REGISTRY["intervention_unit_rule"]},
            "dataset_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "tokenizer_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "registry_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "observable_graph_contract": REGISTRY,
    })
    files[Path("construction_lock_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Construction-owned observable envelope lock",
        "schema_version": "pldr-observable-margin-construction-lock-v1",
        "required": [
            "selected", "lower_bound_rule", "safety_multiplier",
            "numerical_nonvacuity", "diameter_nonvacuity", "graph",
            "intervention", "plga", "frozen_envelopes", "bindings",
            "source_report", "lock_sha256",
        ],
        "no_heldout_artifacts": True,
        "enlargement_allowed": False,
        "frozen_arrays": TRANSFER["frozen_arrays"],
        "frozen_scalars": TRANSFER["frozen_scalars"],
        "exact_union_graph_fields": [
            "rule", "neighbors", "mapping", "union_graph_sha256",
            "vertex_count", "edges", "weights", "effective_resistance",
            "source_physical_sha256",
        ],
        "numerical_nonvacuity_fields": ["relative_margin_floor", "rule"],
        "diameter_nonvacuity_fields": [
            "construction_source_vertex_diameter",
            "endpoint_usefulness_target",
            "endpoint_fraction_of_construction_source",
            "bound_to_observed_ratio_max",
        ],
        "intervention_comparison_order": list(
            INTERVENTION_COMPARISON_NAMES),
        "intervention_operator_contracts": list(
            INTERVENTION_OPERATOR_CONTRACTS),
        "intervention_criterion_fields": [
            "sites_per_comparison", "heldout_seed_site_count",
            "lower_effect_order_statistic_per_seed",
            "positive_sites_required_per_seed", "criterion",
        ],
        "plga_endpoint_ownership": (
            "registry-binds-context-head-row-order-anchor-rule;"
            "construction-lock-binds-selected-layer-and-node"),
        "plga_identity_fields": [
            "selected_layer", "selected_node", "anchor_row",
            "ordinal_context_head_binding",
            "construction_comparison_id_by_ordinal",
        ],
    })
    files[Path("intervention_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": "pldr-observable-intervention-decision-v1",
        "comparison_order": list(INTERVENTION_COMPARISON_NAMES),
        "operator_contracts": list(INTERVENTION_OPERATOR_CONTRACTS),
        "required_locked_comparison_fields": [
            *INTERVENTION_OPERATOR_FIELDS,
            "direction", "effect_lower_threshold",
        ],
        "comparisons": STATISTICS["intervention_comparisons"],
        "sites_per_comparison": STATISTICS["intervention_sites"],
        "sites_per_heldout_seed": STATISTICS["sites_per_heldout_seed"],
        "positive_sites_required_per_seed": STATISTICS[
            "positive_sites_required_per_seed"],
        "lower_effect_order_statistic_per_seed": STATISTICS[
            "lower_effect_order_statistic_per_seed"],
        "deterministic_seed_stratified_criterion": STATISTICS[
            "deterministic_criterion"],
        "randomized_or_independent_replicates": False,
        "p_value_or_significance_claim": False,
        "required_summary_fields": [
            "sites", "sites_by_seed", "positive_direction_sites_by_seed",
            "positive_sites_required_per_seed",
            "lower_effect_order_statistic_per_seed",
            "lower_effect_by_seed", "locked_effect_lower_threshold",
            "seed_stratified_sign_coverage_pass",
            "seed_stratified_effect_floor_pass",
            "directional_prediction_pass",
            "registered_intervention_execution_pass",
            "operator_contract_pass",
            "intended_intervention_target_change_pass",
            "executed_successor_change_pass",
            "direct_operator_non_target_invariance_pass",
            "selected_gate_optimizer_state_retention_pass",
            "native_selected_gate_moment_transition_pass",
            "registered_scheduler_sequence_pass",
            "per_step_execution_evidence_pass",
            "normalized_shape_activation_binding_pass",
            "clamp_raw_gate_gradient_change_pass",
            "no_op_rejection_pass", "locked_direction_pass",
            "locked_lower_effect_interval_pass",
        ],
        "required_unit_fields": [
            "unit_id", "seed", "rng_substream", "future_update_interval",
            "operator_contract", "operator_contract_unchanged",
            "common_initial_state", "common_minibatch_sequence",
            "intended_intervention_target_changed",
            "executed_successor_changed",
            "direct_operator_non_target_invariant",
            "selected_gate_optimizer_state_retained",
            "native_selected_gate_moment_transition_valid",
            "registered_scheduler_sequence_unchanged",
            "per_step_execution_evidence_complete",
            "normalized_shape_activation_binding_valid",
            "clamp_raw_gate_gradient_changed", "no_op_rejected",
            "locked_direction", "locked_effect_lower_threshold",
            "locked_direction_unchanged",
            "locked_lower_effect_interval_unchanged", "difference",
            "intended_intervention_target_difference_norm_by_step",
            "intended_intervention_target_changed_by_step",
            "raw_gradient_changed_by_step",
            "direct_non_target_parameter_invariant_by_step",
            "direct_non_target_optimizer_invariant_by_step",
            "selected_gate_optimizer_state_retained_by_step",
            "native_selected_gate_moment_transition_valid_by_step",
            "registered_scheduler_sequence_unchanged_by_step",
            "normalized_shape_activation_binding",
            "baseline", "modified",
        ],
        "required_arm_evidence_fields": [
            "intervention_target_residual_by_comparison",
            "raw_gradient_signature_by_step",
            "clipped_gradient_signature_by_step",
            "native_selected_gate_moment_transition_by_step",
            "floating_point_charge", "sector_norm_by_step",
            "successor_gate_sha256_by_step",
            "successor_model_sha256_by_step",
            "successor_optimizer_sha256_by_step",
            "successor_scheduler_sha256_by_step",
            "pre_clamp_actual_normalized_activation_signature_by_step",
            "forced_fixed_source_normalized_activation_signature_by_step",
            "effective_executed_normalized_activation_signature_by_step",
            "effective_minus_actual_normalized_activation_norm_by_step",
            "fixed_source_reference_model_sha256_before",
            "fixed_source_reference_model_sha256_after",
            "fixed_source_model_binding_by_step",
            "clamp_hook_removed_before_probe_by_step",
            "selected_gate_optimizer_binding_by_comparison",
        ],
        "direct_operator_non_target_semantics": (
            "Within each arm and step, the registered direct intervention "
            "operator leaves all non-target parameters and optimizer state "
            "byte-unchanged; this does not assert cross-arm equality after "
            "the trajectories causally diverge."),
        "aggregate_suppression_moment_semantics": (
            "Native optimizer.step and scheduler.step execute first. The "
            "complete selected-gate adaptive displacement is then suppressed "
            "while the selected-gate Adam moments and clock from that native "
            "step are retained and evolve on the modified trajectory."),
        "fixed_source_activation_clamp_semantics": (
            "The unadvanced digest-bound checkpoint source model runs in "
            "training mode with captured/restored arm RNG on the same "
            "minibatch. Its detached normalized activation replaces the "
            "selected final metric-LayerNorm activation only during loss "
            "forward/backward; gamma and beta remain trainable, gradient "
            "through the LayerNorm input is blocked, and the post-update "
            "fixed-registry probe is unclamped."),
        "floating_point_charge_rule": (
            "Each modified arm uses its own float64 counterfactual shadow; "
            "the deliberate intervention effect is never classified as "
            "roundoff."),
    })
    files[Path("plga_margin_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": "pldr-observable-plga-margin-v1",
        "qualified_set_must_be_nonempty_for_each_heldout_seed": True,
        "comparison_rule": "same-context-collapsed-anchor-union-subset-v1",
        "score_transfer_rule": (
            "deltaQ-G0-K0-plus-Q1-deltaG-K0-plus-Q1-G1-deltaK-v1"),
        "curvature_cache_reduction": (
            "reference-and-candidate-QK-byte-equal-so-only-Q-deltaG-KT-remains"),
        "required_endpoint_arrays": [
            "plga_reference_query", "plga_candidate_query",
            "plga_reference_key", "plga_candidate_key",
            "plga_curvature_left", "plga_curvature_right",
            "plga_reference_pre_mask_score",
            "plga_candidate_pre_mask_score",
        ],
        "criterion": "sqrt(2)*remainder_bound < centered_logit_gap",
        "required_comparison_fields": [
            "positive_bases", "real_exponent", "query_operator_norm",
            "key_operator_norm", "score_difference_norm",
            "query_endpoint_fixed", "key_endpoint_fixed",
            "score_query_term_norm", "score_generator_term_norm",
            "score_key_term_norm", "score_transfer_relative_residual",
            "centered_logit_gap", "remainder_bound",
            "heldout_transfer_without_enlargement",
        ],
    })
    files[Path("results_manifest_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": "pldr-observable-results-manifest-v1",
        "format": "sha256-two-space-normalized-relative-path",
        "coverage": (
            "all files below results_root after immutable ledger-prefix "
            "snapshot and before final release attestation"),
        "omission_sensitive": True,
        "symbolic_links_allowed": False,
        "excluded": [
            "the manifest itself",
            "release-verification.json sealed by the executor output binding",
        ],
        "verification": "independent full re-enumeration and digest replay",
    })
    files[Path("final_decision_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "schema_version": "pldr-observable-margin-final-decision-v1",
        "algebra_is_prerequisite_not_scientific_statement": True,
        "required_statements": [
            "S1_positive_heldout_margin",
            "S2_nonvacuous_frozen_energy_and_diameter_bound",
            "S3_causal_intervention_response",
            "S4_fresh_seed_transfer_without_enlargement",
            "S5_nonempty_centered_logit_plga_margin",
        ],
        "robustness_scientific_gate": False,
    })
    files[Path("README.md")] = (
        "# Observable-margin confirmation protocols\n\n"
        "This generated directory owns the fresh-seed R/T/B/H/ROB/M/I/F "
        "confirmation. Native float32 AdamW updates consume actual sequential "
        "minibatches. The same digest-bound probe registry is evaluated at "
        "every node, so graph energies telescope on one fixed finite "
        "population. Float64 shadow arithmetic records the numerical charge "
        "without replacing the trained system.\n\n"
        "Construction seed 4444 locks the selected cell, complete source "
        "envelopes, one exact connected union edge/weight graph and R-star, "
        "ordinal held-out mapping, affine path, numerical margin floor, useful "
        "diameter target, PLGA product bound, and lower-bound rule before "
        "seeds 5555 and 6666 open. ROB is reported but is not a "
        "scientific gate. Stage I uses 24 deterministic continuation sites, "
        "12 per held-out seed. Each seed separately must pass the 9-of-12 "
        "directional threshold and fourth ascending effect floor; no "
        "p-value is reported. The executor is serial: the 44.25 GPU-hour "
        "quantity is the aggregate stage budget, while 76.6 reserved wall "
        "hours fit inside the 80-hour hard wall. No raw intermediate "
        "parameter nodes are retained.\n\n"
        "Stage F invokes `make release-check-current` in the digest-bound "
        "realized environment. Its evidence records the environment "
        "attestation digest, resolved Python import root, and normalized "
        "package-inventory count and digest. It freshly verifies the exact "
        "current repository MANIFEST target count, manifest digest, and "
        "all-target aggregate against an equal staged seal. It separately "
        "verifies the strict observable-margin manuscript export of exactly 18 regular "
        "files, including a 17-entry nonself MANIFEST, with manifest and "
        "all-target aggregate digests against its equal staged seal. Both "
        "closures are independently recomputed before and after the canonical "
        "release command, and Stage F records the pass booleans, digests, and "
        "cross-command stability.\n"
    ).encode("utf-8")
    checksums = "".join(
        f"{_digest(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda value: value.as_posix())
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
        path for path, payload in expected.items()
        if not (OUTPUT / path).is_file()
        or (OUTPUT / path).read_bytes() != payload
    ]
    unknown = sorted(actual - set(expected), key=lambda value: value.as_posix())
    if check:
        if stale or unknown:
            raise SystemExit(
                "observable-margin protocols are stale: "
                + ", ".join(
                    [path.as_posix() for path in stale]
                    + [f"unknown:{path.as_posix()}" for path in unknown]))
        print(f"observable-margin protocols: verified ({len(expected)} files)")
        return
    if unknown:
        raise SystemExit(
            "refusing to overwrite protocol directory with unknown files: "
            + ", ".join(path.as_posix() for path in unknown))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for relative, payload in expected.items():
        destination = OUTPUT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    print(f"observable-margin protocols: wrote {len(expected)} files")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    generate(parser.parse_args().check)


if __name__ == "__main__":
    main()
