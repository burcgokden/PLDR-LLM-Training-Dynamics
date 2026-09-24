"""Frozen specifications for the comprehensive collapse confirmation."""

from __future__ import annotations


SCHEMA_VERSION = "pldr-comprehensive-collapse-protocol-v4"

ARCHITECTURE = {
    "width": 64,
    "layers": 4,
    "heads": 4,
    "head_width": 16,
    "context_length": 256,
    "batch_size": 32,
    "updates": 6144,
    "row_map_parameter_upper_bound": 38912,
    "construction_contexts": 48,
    "construction_row_output_coordinates": 49152,
    "validation_contexts": 32,
    "application_pairs": 32,
    "trajectory_seeds": [3401, 3402],
    "normal_challenge_seed": 340031,
    "cover_directions_per_row": 4,
    "finite_batch_failure_probability": 0.01,
    "optimizer_snapshot_windows": [
        list(range(769, 777)), list(range(5633, 5641)),
    ],
}

RESOURCE_CAPS = {
    "gpu_count": 2,
    "gpu_model": "NVIDIA RTX 4090",
    "gpu_bytes_each": 24 * 1024 ** 3,
    "normal_peak_allocated_bytes": 20 * 1024 ** 3,
    "host_bytes": 125 * 1024 ** 3,
    "planned_gpu_hours": 3.30,
    "hard_gpu_hours": 4.0,
    "hard_wall_clock_hours": 6.0,
    "validation_tuning_permitted": False,
}


def _stage(stage_id, title, dependencies, gpu_hours, placement, outputs, checks):
    return {
        "id": stage_id,
        "title": title,
        "dependencies": list(dependencies),
        "gpu_hour_cap": gpu_hours,
        "placement": placement,
        "primary_outputs": list(outputs),
        "required_checks": list(checks),
        "decision_source": (
            "The analyzer resolves the sealed request, checkpoints, raw archive, "
            "producer and analyzer code, partition, resource record, challenge "
            "seed, and prediction lock, then recomputes the decision."
        ),
    }


STAGES = (
    _stage(
        "Q", "Two-device instrument qualification", (), 0.10,
        "same float64 fixture on cuda:0 and cuda:1",
        (
            "finite-stencil JVP equality", "centered-logit Fisher action",
            "true-loss HVP", "full AdamW successor JVP", "peak memory",
            "complete environment record",
        ),
        (
            "context_256_complete", "peak_allocated_below_20_gib",
            "row_jvp_replays", "fisher_action_replays",
            "true_hessian_action_replays", "adamw_successor_replays",
            "cross_device_scientific_scalars_agree", "q_stage_sealed",
        ),
    ),
    _stage(
        "T", "Complete-state trajectories", ("Q",), 0.50,
        "seed 3401 on cuda:0 and seed 3402 on cuda:1",
        (
            "two 6144-update trajectories", "complete landmarks",
            "dense early and late window snapshots",
            "full Z_h vector and direct seminorm",
            "independent first and last endpoint replay",
        ),
        (
            "run_identities_unique", "lineage_is_from_scratch",
            "checkpoint_components_complete", "scheduler_uses_trainer_formula",
            "direct_observables_at_every_snapshot", "environment_fields_complete",
        ),
    ),
    _stage(
        "TR", "Independent same-device replay", ("T",), 0.25,
        "primary trajectory repeated from scratch on cuda:0",
        ("scientific state equality", "distinct run identity and event ledger"),
        (
            "replay_run_identity_differs", "replay_lineage_differs",
            "model_moments_clocks_masks_registry_next_batch_rng_equal",
            "device_and_cursor_labels_excluded_from_scientific_digest",
            "aliased_primary_checkpoint_rejected",
        ),
    ),
    _stage(
        "N", "Parameter-normal frame and balance", ("T", "TR"), 1.40,
        "one anchor per GPU; large tangent blocks staged in host memory",
        (
            "rank and basis of DZ_h", "selected pairwise normal frame",
            "true full-loss edge", "first and second balance defects",
            "finite-batch force", "joint row-direction cover radii",
            "spectral residuals",
        ),
        (
            "parameter_count_recorded", "stencil_capacity_exceeds_normal_rank",
            "normal_rank_residual_certified",
            "joint_registered_pair_cover_certified",
            "selected_edges_span_normal_dual",
            "pair_frame_edge_positive", "signed_full_loss_edge_positive",
            "balance_defects_within_lock", "force_within_schedule_envelope",
            "checkpoint_challenge_probes_replay", "synthetic_archive_rejected",
        ),
    ),
    _stage(
        "D", "Scheduled-metric trajectory enclosure", ("N",), 0.15,
        "early and late windows split across both GPUs",
        (
            "sealed scalar three-block AdamW snapshot challenges",
            "analyzer-owned successor differentiation",
            "SVD-linearized normal coordinates", "two-block scheduled metrics",
            "variable radii and construction-locked direct conversion",
        ),
        (
            "live_successor_matches_jvp", "path_metric_inequalities_hold",
            "metric_conditioning_within_lock", "radius_images_are_invariant",
            "normal_trajectory_enclosed", "direct_trajectory_enclosed",
            "transient_expansion_charged_at_block_boundary",
        ),
    ),
    _stage(
        "I", "Rate interventions", ("D",), 0.40,
        "matched 16-snapshot continuations on both rate sides",
        ("two locked-rate continuation paths",
         "finite-stencil SVD normal amplitudes", "damping order",
         "phase order", "oscillation-onset order"),
        (
            "branches_share_source_state", "only_registered_rate_changes",
            "full_second_moment_successor_used", "sealed_ordering_observed",
        ),
    ),
    _stage(
        "A", "Oriented same-source application", ("N",), 0.10,
        "32 immutable pairs split by checkpoint",
        (
            "oriented end-to-end integral", "qualified greedy margins",
            "construction-locked cell-residual ratios",
        ),
        (
            "heldout_pairs_match_partition", "subdivision_fixed_by_lock",
            "no_fitted_additive_envelope", "all_logit_differences_enclosed",
            "every_margin_qualified_decision_preserved",
        ),
    ),
    _stage(
        "S2N", "Matched normal replication", ("N",), 0.40,
        "late anchor of seed 3402 on its registered device",
        ("replicated frame", "replicated balance", "unchanged intervals"),
        (
            "prediction_lock_unchanged", "pair_frame_interval_replicates",
            "full_loss_interval_replicates", "balance_interval_replicates",
        ),
    ),
    _stage(
        "R", "Independent evidence replay", ("D", "I", "A", "S2N"), 0.0,
        "fresh CPU process",
        ("recomputed stage records", "campaign decision", "resource total"),
        (
            "all_artifacts_resolve", "every_stage_recomputed_from_evidence",
            "empty_or_typed_campaign_rejected", "records_rebuild_byte_stably",
            "resource_total_below_hard_cap",
        ),
    ),
)

PREDICTIONS = (
    {
        "id": "P1",
        "quantity": "selected_pair_and_true_loss_edges",
        "lock_rule": "construction intervals fixed before validation",
        "success": "positive at both anchors and replicated at S2N",
    },
    {
        "id": "P2",
        "quantity": "conditional_balance_defects_and_normal_force",
        "lock_rule": (
            "construction balance term plus clip-derived Bernstein radius "
            "at delta 0.01, frozen before validation"),
        "success": (
            "replayed held-out clipped force and signed defects lie within "
            "their locked envelopes"),
    },
    {
        "id": "P3",
        "quantity": "scheduled_metric_product_convolution",
        "lock_rule": "metrics, radii, pads, and block boundaries frozen",
        "success": "normal and direct trajectories are enclosed in both windows",
    },
    {
        "id": "P4",
        "quantity": "rate_intervention_order",
        "lock_rule": "damping, phase, and onset order frozen from the live lift",
        "success": "all paired continuations have the locked order",
    },
    {
        "id": "P5",
        "quantity": "oriented_same_source_logit_bound",
        "lock_rule": "construction pairs freeze subdivision and cell-residual radii",
        "success": (
            "all pairs enclosed, at least one pair qualified, and every "
            "qualified decision preserved"),
    },
)


RAW_REQUIRED = {
    "Q": ("qualification_cuda0", "qualification_cuda1"),
    "T": (
        "primary_checkpoints", "replication_checkpoints",
        "primary_direct_trajectory", "replication_direct_trajectory",
        "tokens", "registry",
        "primary_event_ledger", "replication_event_ledger",
    ),
    "TR": (
        "primary_checkpoint", "replay_checkpoint",
        "primary_event_ledger", "replay_event_ledger",
    ),
    "N": (
        "checkpoint", "tokens", "registry", "normal_metadata",
        "finite_stencil_jacobian", "normal_basis", "normal_left_basis",
        "singular_values", "normal_actions", "prediction_lock",
    ),
    "D": (
        "dynamics", "snapshots", "prediction_lock", "normal_metadata",
        "normal_left_basis", "singular_values", "normal_actions",
        "direct_trajectory",
    ),
    "I": (
        "intervention", "prediction_lock",
        "source_checkpoints", "low_checkpoints", "high_checkpoints",
        "tokens", "registry", "normal_metadata", "normal_left_basis",
        "singular_values", "low_event_ledger", "high_event_ledger",
    ),
    "A": ("application", "prediction_lock"),
    "S2N": (
        "checkpoint", "tokens", "registry", "normal_metadata",
        "finite_stencil_jacobian", "normal_basis", "normal_left_basis",
        "singular_values", "normal_actions", "prediction_lock",
    ),
    "R": ("rebuild_manifest",),
}


STAGE_BY_ID = {stage["id"]: stage for stage in STAGES}
