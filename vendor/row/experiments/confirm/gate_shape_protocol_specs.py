"""Single source of truth for the prospective gate-shape confirmation."""

from __future__ import annotations


SCHEMA_VERSION = "pldr-gate-shape-protocol-v1"
STATUS = "DRAFT_PENDING_TWO_DEVICE_QUALIFICATION"

ARCHITECTURE = {
    "width": 64,
    "layers": 4,
    "heads": 4,
    "head_width": 16,
    "context_length": 256,
    "batch_size": 32,
    "updates": 6144,
    "row_map_residual_units": 8,
    "glu_blocks_per_residual_unit": 2,
    "glu_hidden_width": 48,
    "gate_sector_dimension_per_layer": 16,
    "lifted_gate_dimension_per_layer": 48,
    "trajectory_seeds": [3401, 3402],
    "anchor_source_steps": [768, 3072, 5632],
    "capture_update_blocks": [
        list(range(769, 777)),
        list(range(3073, 3081)),
        list(range(5633, 5641)),
    ],
    "construction_contexts_per_anchor": 4,
    "validation_contexts_per_anchor": 4,
    "construction_application_pairs": 16,
    "validation_application_pairs": 16,
    "intervention_updates": 8,
    "intervention_arms": [
        "baseline",
        "rate_0.75",
        "rate_1.25",
        "freeze_final_gate",
        "freeze_upstream_shape",
    ],
}

RESOURCE_CAPS = {
    "gpu_count": 2,
    "gpu_model": "NVIDIA RTX 4090",
    "gpu_bytes_each": 24 * 1024 ** 3,
    "host_bytes": 125 * 1024 ** 3,
    "scientific_subtotal_gpu_hours": 2.45,
    "engineering_reserve_gpu_hours": 0.40,
    "hard_gpu_hours": 3.25,
    "hard_wall_clock_hours": 4.0,
    "validation_tuning_permitted": False,
    "qualification_required_before_freeze": True,
}


def _stage(stage_id, title, dependencies, gpu_hours, purpose, checks):
    return {
        "id": stage_id,
        "title": title,
        "dependencies": list(dependencies),
        "gpu_hour_cap": float(gpu_hours),
        "purpose": purpose,
        "required_checks": list(checks),
        "decision_source": (
            "A fresh analyzer resolves normalized checkpoint-bound paths and "
            "recomputes every deciding quantity from raw evidence. Digests "
            "bind evidence but never stand in for scientific recomputation."
        ),
    }


STAGES = (
    _stage(
        "Q",
        "Two-device gate-path qualification",
        (),
        0.10,
        "Replay one CPU-realized context-256 row-map fixture on both devices.",
        (
            "same_realized_fixture",
            "context_256_row_map_complete",
            "gate_factorization_replays",
            "zero_gate_is_constant",
            "gate_scaling_identity_replays",
            "global_gate_bound_encloses_rows",
            "dense_true_hessian_replays",
            "analytic_48_by_48_adamw_block_matches_autodiff",
            "cross_device_arrays_agree_with_declared_tolerance",
            "measured_memory_and_seconds_recorded",
        ),
    ),
    _stage(
        "T",
        "Complete construction and validation trajectories",
        ("Q",),
        0.40,
        "Train seeds 3401 and 3402 with complete source and endpoint state.",
        (
            "exactly_6144_updates",
            "all_three_anchor_blocks_complete",
            "model_optimizer_scheduler_rng_and_data_cursor_bound",
            "one_resolved_lineage_root_per_ledger",
            "construction_and_validation_chunks_disjoint",
            "all_resource_events_complete",
        ),
    ),
    _stage(
        "TR",
        "Independent same-device replay",
        ("T",),
        0.15,
        "Repeat the registered primary trajectory from scratch.",
        (
            "distinct_run_and_lineage_identity",
            "scientific_checkpoint_components_equal",
            "aliased_primary_rejected",
            "registered_update_count_and_windows_equal",
        ),
    ),
    _stage(
        "G",
        "Gate-shape geometry and true-loss response",
        ("T", "TR"),
        1.15,
        "Measure every 16-dimensional layer gate on disjoint contexts.",
        (
            "row_outputs_recomputed_from_final_layernorm_preactivations",
            "q_and_energy_recomputed_from_fixed_pairs",
            "fisher_plus_signed_jet_reconstructs_true_hessian",
            "all_dense_spectra_recomputed",
            "spectral_sign_uses_residual_scaled_threshold",
            "gate_force_at_zero_recomputed",
            "no_parameter_jacobian_svd_or_lanczos_basis",
        ),
    ),
    _stage(
        "C",
        "Ordered gate capture and source-owned energy enclosure",
        ("G",),
        0.35,
        "Test exact 48-dimensional AdamW products and gate-shape work.",
        (
            "analytic_gate_blocks_match_automatic_differentiation",
            "clipping_boundary_distance_recorded",
            "direct_ordered_products_recomputed",
            "construction_block_and_remainder_lock_precedes_validation",
            "target_energy_not_read_while_constructing_envelope",
            "gate_plus_shape_work_identity_replays_at_every_node",
            "validation_nodes_lie_in_source_owned_envelope",
            "capturing_block_contains_transient_expansion",
            "construction_mechanism_label_replicates",
        ),
    ),
    _stage(
        "I",
        "Causal gate and shape interventions",
        ("C",),
        0.20,
        "Run short validation-owned rate and freeze continuations.",
        (
            "branches_share_one_complete_source_state",
            "all_five_registered_arms_complete",
            "only_registered_intervention_differs",
            "real_event_ledgers_have_one_lineage_root",
            "all_gate_shape_energies_recomputed",
            "final_gate_freeze_has_zero_gate_work",
        ),
    ),
    _stage(
        "A",
        "Exact PLGA secant transfer",
        ("G",),
        0.10,
        "Propagate fixed same-source generator pairs through PLGA.",
        (
            "checkpoint_bindings_and_logit_differences_replay",
            "iswiglu_and_positive_power_secants_replay",
            "deductive_and_centered_logit_differences_enclosed",
            "margin_claim_only_for_independently_qualified_pairs",
            "empty_qualified_set_supports_no_decision_claim",
        ),
    ),
    _stage(
        "R",
        "Fresh-process reconstruction and mutation gate",
        ("C", "I", "A"),
        0.0,
        "Recompute the campaign in a clean CPU process.",
        (
            "all_relative_paths_resolve_under_campaign_root",
            "self_pairs_and_absolute_paths_rejected",
            "every_deciding_tensor_mutation_rejected",
            "npz_payloads_have_npz_names",
            "records_rebuild_byte_stably",
            "device_host_disk_and_wall_time_recomputed",
            "hard_resource_caps_hold",
        ),
    ),
)

PREDICTIONS = (
    {
        "id": "P1",
        "name": "exact_gate_identity",
        "statement": (
            "Final-gate rescaling multiplies every row difference by the "
            "same factor; a zero gate produces the constant bias row; and "
            "the global architecture bound encloses all registered pairs."
        ),
    },
    {
        "id": "P2",
        "name": "schedule_capture",
        "statement": (
            "At least one source-fixed post-warmup eight-update block has a "
            "residual-padded ordered 48-dimensional gain below one, even if "
            "one or more constituent updates expand."
        ),
    },
    {
        "id": "P3",
        "name": "source_owned_energy_enclosure",
        "statement": (
            "Construction-owned gate, directional-shape, and remainder charges "
            "enclose every validation energy node, while forcing and direct "
            "product charges enclose their corresponding validation data."
        ),
    },
    {
        "id": "P4",
        "name": "mechanism_and_intervention_decomposition",
        "statement": (
            "The construction classification as gate-led, shape-led, or mixed "
            "replicates on the validation trajectory. Every matched "
            "continuation obeys the exact gate-shape energy decomposition, and "
            "the final-gate freeze has identically zero gate work."
        ),
    },
    {
        "id": "P5",
        "name": "plga_secant_transfer",
        "statement": (
            "Exact occupied-interval secants reconstruct the PLGA deductive "
            "difference and enclose the held-out centered-logit difference."
        ),
    },
)

RAW_REQUIRED = {
    "Q": ("fixture", "qualification_cuda0", "qualification_cuda1"),
    "T": (
        "construction_checkpoints", "validation_checkpoints",
        "construction_event_ledger", "validation_event_ledger",
        "tokens", "tokenizer", "registry",
    ),
    "TR": (
        "primary_checkpoints", "replay_checkpoints",
        "primary_event_ledger", "replay_event_ledger",
    ),
    "G": (
        "checkpoints", "tokens", "registry", "gate_shape_raw_archives",
        "producer_source",
    ),
    "C": (
        "source_checkpoints", "endpoint_checkpoints", "optimizer_snapshots",
        "tokens", "registry", "construction_lock", "capture_raw_archives",
        "producer_source",
    ),
    "I": (
        "source_checkpoint", "arm_checkpoints", "arm_event_ledgers",
        "tokens", "registry", "construction_lock",
        "intervention_geometry_raw_archives", "producer_source",
    ),
    "A": (
        "checkpoints", "tokens", "registry", "application_raw_archives",
        "construction_lock", "producer_source",
    ),
    "R": ("rebuild_manifest", "mutation_report", "resource_record"),
}

STAGE_BY_ID = {stage["id"]: stage for stage in STAGES}
