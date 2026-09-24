"""Frozen v3 Q/T/N/I/A/R specifications for composite collapse theory."""

from __future__ import annotations


TIME_SEMANTICS = (
    "FINITE_IMPLEMENTED_SCHEDULE",
    "FROZEN_CHECKPOINT",
)

ARTIFACT_ROLES = (
    "stage_protocol",
    "raw_observations",
    "source_complete_state",
    "target_complete_state",
    "partition_manifest",
    "resource_metadata",
)

BLOCK_SIZES = (1, 2, 4, 8)

BLOCKING_QUALIFICATION = {
    "schema_version": "pldr-composite-blocking-qualification-v3",
    "map": "visible_null_sector_common_metric_product_convolution",
    "normalization": "construction_metric_with_correlated_vertices",
    "block_sizes": BLOCK_SIZES,
    "fine_coefficients": "RECOMPUTED_FROM_LIVE_CHECKPOINT_OPERATORS",
    "individual_fine_gain_may_exceed_one": True,
    "decision": "ORDERED_PRODUCT_FORCE_AND_QUADRATIC_TUBE_RECURRENCE",
}


def _stage(
        stage_id, title, dependencies, execution, purpose, gpu_hours,
        checks, raw_schema, *,
        time_semantics=("FINITE_IMPLEMENTED_SCHEDULE",)):
    return {
        "id": stage_id,
        "title": title,
        "dependencies": tuple(dependencies),
        "time_semantics": tuple(time_semantics),
        "fresh_execution": execution,
        "target": purpose,
        "gpu_hour_cap": gpu_hours,
        "required_checks": tuple(checks),
        "raw_schema": raw_schema,
        "analyzer":
            "experiments/analysis/composite_confirmation_analysis.py",
        "decision": (
            "The bound analyzer resolves complete v3 checkpoints and "
            "recomputes all decisions from producer-owned sufficient "
            "tensors. Caller arrays, pass flags, bounds, masks, and argmax "
            "claims are not accepted."
        ),
    }


STAGES = (
    _stage(
        "Q",
        "Instrument qualification",
        (),
        "Width 8, depth 1, float64, batch 8, context 8 on target device",
        "Qualify every row JVP, whole-tube LayerNorm floors, nonvacuous "
        "Taylor transport, exact AdamW replay, and a dense composite Gram.",
        0.25,
        (
            "all_rows_registered",
            "frozen_job_geometry",
            "dense_matrix_free_jvp",
            "moving_row_decomposition",
            "taylor_remainder_enclosed",
            "layernorm_floor_measured",
            "adamw_successor_exact",
            "energy_identity",
            "negative_work_measured",
            "positive_witness_constructed",
            "provenance_mutation_rejected",
            "taylor_enclosure_ratio",
        ),
        "pldr-q-live-record-v2",
    ),
    _stage(
        "T",
        "Complete-state training and restart",
        ("Q",),
        "Width 64, depth 4, four heads, 6144 updates and four landmarks",
        "Bind and fresh-process replay model tensors, Adam moments, clocks, "
        "realized masks, registries, RNG state, and next minibatch.",
        0.75,
        (
            "live_producer_bound",
            "primary_seed_bound",
            "architecture_and_updates",
            "four_primary_landmarks",
            "complete_checkpoint_schema",
            "optimizer_clock_consistent",
            "scheduler_coefficients_recomputed",
            "fresh_process_tensor_replay",
            "data_cursor_advances",
        ),
        "pldr-t-raw-observations-v3",
    ),
    _stage(
        "N",
        "Composite normal windows",
        ("T",),
        "First layer, post-warmup and late anchors, disjoint 32 by 256 rows",
        "Resolve row, causal-score, value and composite capacities; construct "
        "visible/null frames, charge the full edge, and test the normal tube.",
        2.0,
        (
            "partition_and_producer_bound",
            "capacity_obstructions_accounted",
            "backward_frame_recomputed",
            "composite_kernel_resolved",
            "charged_full_edge_positive",
            "sector_comparison_contracts",
            "common_metric_contracts",
            "ordered_windows_contract",
            "nonlinear_recurrence_encloses",
            "direct_validation_contracts",
        ),
        "pldr-n-raw-observations-v3",
    ),
    _stage(
        "I",
        "Decay-aware corridor intervention",
        ("N",),
        "One matched anchor and 32 paired low-rate and high-rate continuations",
        "Use the live Adam denominator and realized decay coefficient to test "
        "the full lifted Jury and common-metric ordering.",
        0.5,
        (
            "live_producer_bound",
            "matched_source_states",
            "branch_only_changes_rate",
            "jury_margins_recomputed",
            "corridor_bracketed",
            "damping_order_observed",
            "oscillation_onset_order_observed",
        ),
        "pldr-i-raw-observations-v3",
    ),
    _stage(
        "A",
        "Same-source application bridge",
        ("N",),
        "Early and late checkpoints, all layers, 32 immutable held-out pairs",
        "Propagate the row defect once through actual PLGA stage tensors and "
        "recompute the sharp logit margin and greedy decision from logits.",
        0.25,
        (
            "partition_and_producer_bound",
            "heldout_pairs_immutable",
            "all_layers_registered",
            "row_factor_applied_once",
            "stage_envelopes_recomputed",
            "absolute_differences_enclosed",
            "layernorm_floors_measured",
            "registered_margins_preserved",
        ),
        "pldr-a-raw-observations-v3",
        time_semantics=("FROZEN_CHECKPOINT",),
    ),
    _stage(
        "R",
        "Independent replay and report",
        ("I", "A"),
        "Fresh CPU process with deterministic analyzers and report build",
        "Rehash every artifact and regenerate scientific records, tables, "
        "figures, protocols, and manuscript outputs byte stably.",
        0,
        (
            "all_artifacts_resolve",
            "scientific_digests_replay",
            "records_recompute_byte_stably",
            "tables_recompute_byte_stably",
            "figures_recompute_byte_stably",
            "resource_metadata_unsigned",
        ),
        "pldr-r-raw-observations-v2",
        time_semantics=(
            "FINITE_IMPLEMENTED_SCHEDULE", "FROZEN_CHECKPOINT"),
    ),
)


ARCHITECTURE_RUNS = (
    {
        "run": "w64-d4-seed-3201-primary",
        "architecture": "w064_d04_h04",
        "seed": 3201,
        "updates": 6144,
        "schedule": "warmup_cosine",
        "landmarks": (256, 768, 3072, 5632),
        "normal_window_starts": (768, 5632),
        "gate": "PRIMARY",
    },
    {
        "run": "w64-d4-seed-3202-replication",
        "architecture": "w064_d04_h04",
        "seed": 3202,
        "updates": 6144,
        "schedule": "warmup_cosine",
        "landmarks": (256, 768, 3072, 5632),
        "normal_window_starts": (768,),
        "gate": "AFTER_PRIMARY_Q_T_N",
    },
)


RESOURCE_CAPS = {
    "gpu_count": 2,
    "gpu_bytes_each": 20 * 1024 ** 3,
    "host_bytes_total": 96 * 1024 ** 3,
    "retained_artifact_bytes": 80 * 1024 ** 3,
    "stage_gpu_hours": {
        stage["id"]: stage["gpu_hour_cap"] for stage in STAGES
    },
    "planned_gpu_hours": 4.5,
    "primary_gpu_hours": 3.75,
    "gated_replication_gpu_hours": 0.75,
    "hard_gpu_hours": 4.5,
    "hard_wall_clock_hours": 12,
    "dense_stack_parameter_derivative": "FORBIDDEN_EXCEPT_Q",
    "construction_contexts": 32,
    "validation_contexts": 32,
    "application_heldout_pairs": 32,
    "context_length": 256,
}
