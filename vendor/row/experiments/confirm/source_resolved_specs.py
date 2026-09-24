"""Immutable design for source-resolved PLDR row-map confirmation."""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-source-resolved-row-map-collapse-v1"
DESIGN_SCHEMA = "pldr-source-resolved-design-v1"
QUALIFICATION_SCHEMA = "pldr-source-resolved-qualification-v1"
CHECKPOINT_BINDING_SCHEMA = "pldr-source-resolved-checkpoint-binding-v1"
EDGE_SCHEMA = "pldr-source-resolved-edge-v1"
MASK_RESPONSE_SCHEMA = "pldr-source-resolved-mask-response-v1"
BLOCK_RESPONSE_SCHEMA = "pldr-source-resolved-block-response-v1"
RANK_TUBE_SCHEMA = "pldr-source-resolved-rank-tube-v1"
TIMEPOINT_SCHEMA = "pldr-source-resolved-timepoint-v1"
TIMECOURSE_SCHEMA = "pldr-source-resolved-timecourse-v1"
INTERVENTION_PREDICTION_SCHEMA = "pldr-source-resolved-intervention-prediction-v1"
INTERVENTION_SCHEMA = "pldr-source-resolved-intervention-v1"
CONSTRUCTION_POLICY_SCHEMA = "pldr-source-resolved-construction-policy-v1"
ANALYSIS_SCHEMA = "pldr-source-resolved-analysis-v1"
SCIENTIFIC_OUTCOMES = (
    "confirmed", "not_confirmed", "unresolved", "technically_invalid"
)

ARCHITECTURE = {
    "trainable_parameters": 20_622_914,
    "decoder_layers": 3,
    "heads_per_layer": 4,
    "head_width": 64,
    "row_count": 64,
    "residual_units": 8,
    "gated_blocks_per_unit": 2,
}

OPTIMIZER = {
    "name": "AdamW",
    "batch_size": 32,
    "context_length": 256,
    "warmup_updates": 250,
    "learning_rate": 7.5e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1.0e-5,
    "gradient_clip": 1.0,
    "model_dtype": "float32",
    "analysis_dtype": "float64",
}

REGISTRY = {
    "construction_context_count": 8,
    "heldout_context_count": 16,
    "context_count": 24,
    "layers": [0, 1, 2],
    "heads": [0, 1, 2, 3],
    "rows_per_map": 64,
    "features_per_row": 64,
    "pairs_per_map": 2016,
    "map_count": 288,
    "within_map_pair_count": 580_608,
    "independent_contrast_dimension_per_map": 3_969,
    "pair_rule": "within-context-layer-head-all-unordered-pairs-v1",
    "cross_map_pairs_forbidden": True,
}

TRAJECTORIES = (
    {
        "name": "construction-seed7444",
        "role": "construction",
        "seed": 7444,
        "device": "cuda:0",
        "heldout": False,
    },
    {
        "name": "heldout-seed8444",
        "role": "heldout",
        "seed": 8444,
        "device": "cuda:0",
        "heldout": True,
    },
    {
        "name": "heldout-seed9444",
        "role": "heldout",
        "seed": 9444,
        "device": "cuda:1",
        "heldout": True,
    },
)

FROZEN_POLICY = {
    "rank_relative_threshold": 1.0e-8,
    "metric_maximum_eigenvalue_cap": 1.0e4,
    "block_length": 16,
    "reduced_krylov_rank": 8,
    "orthogonal_audit_directions": 8,
    "plga_attenuation_cap_menu": [0.05, 0.10, 0.20, 0.50],
    "regular_cadence": 16,
    "last_regular_update": 8992,
    "terminal_update": 9000,
    "intervention_window_updates": {
        "pre-onset": 1000,
        "onset": 4000,
        "late": 8000,
    },
    "regular_snapshot_count": 563,
    "total_snapshot_count": 564,
    "numerical_absolute_tolerance": 1.0e-12,
    "float32_factorization_relative_tolerance": 1.0e-6,
    "transport_relative_tolerance": 1.0e-10,
    "native_link_tolerance": 0.0,
    "intervention_prediction_relative_tolerance": 1.0e-7,
    "maximum_unresolved_adamw_coordinates": 0,
    "local_bounds": "analytic-or-outward-interval-only",
}

RESOURCE_BUDGET = {
    "working_aggregate_gpu_hours": 8.0,
    "hard_aggregate_gpu_hours": 10.0,
    "queried_device_count": 2,
    "queried_gpu_capacity_mib": 24_564,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 24 * 1024**3,
    "poll_seconds": 1.0,
    "maximum_complete_checkpoints_per_active_lineage": 16,
}

STAGES = (
    {
        "id": "Q0",
        "purpose": "kernel-schema-executor-recovery-and-real-path-qualification",
        "producer": "qualify",
        "aggregate_gpu_hours_target": 0.1,
        "depends_on": [],
    },
    {
        "id": "Q1",
        "purpose": "complete-source-transport-updates-1000-through-1004",
        "producer": "edge",
        "aggregate_gpu_hours_target": 0.1,
        "depends_on": ["Q0"],
    },
    {
        "id": "Q2",
        "purpose": "adamw-coordinate-strata-and-one-update-response",
        "producer": "mask-response",
        "aggregate_gpu_hours_target": 0.3,
        "depends_on": ["Q1"],
    },
    {
        "id": "Q3",
        "purpose": "exact-image-rank-and-diagnostic-full-state-tube-screen",
        "producer": "rank-tube-block",
        "aggregate_gpu_hours_target": 3.5,
        "depends_on": ["Q2"],
    },
    {
        "id": "Q4",
        "purpose": "construction-timecourse-and-source-mechanism",
        "producer": "timecourse",
        "aggregate_gpu_hours_target": 1.0,
        "depends_on": ["Q1"],
    },
    {
        "id": "Q5",
        "purpose": "controlled-source-interventions",
        "producer": "intervention",
        "aggregate_gpu_hours_target": 0.8,
        "depends_on": ["Q4"],
    },
    {
        "id": "Q6",
        "purpose": "two-heldout-trajectories-with-frozen-policy",
        "producer": "timecourse",
        "aggregate_gpu_hours_target": 2.0,
        "depends_on": ["Q3", "Q4", "Q5"],
    },
)

INTERVENTIONS = (
    "final-gate-freeze",
    "row-program-weight-freeze",
    "upstream-shape-freeze",
    "moment-reset",
    "learning-rate-half",
    "learning-rate-double",
)


def snapshot_updates() -> list[int]:
    updates = list(range(
        0,
        FROZEN_POLICY["last_regular_update"] + 1,
        FROZEN_POLICY["regular_cadence"],
    ))
    updates.append(FROZEN_POLICY["terminal_update"])
    return updates


def campaign_design() -> dict[str, Any]:
    """Return the canonical prospective campaign design."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "architecture": ARCHITECTURE,
        "optimizer": OPTIMIZER,
        "registry": REGISTRY,
        "trajectories": list(TRAJECTORIES),
        "frozen_policy": FROZEN_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "stages": list(STAGES),
        "interventions": list(INTERVENTIONS),
        "scientific_outcomes": list(SCIENTIFIC_OUTCOMES),
        "evidence_contract": {
            "complete_checkpoint_required": True,
            "checkpoint_content_opened": True,
            "checkpoint_optimizer_batch_cursor_bound": True,
            "source_code_tokenizer_dataset_order_bound": True,
            "producer_precedes_assembler": True,
            "validity_fields_derived_from_checks": True,
            "unknown_schema_fields_rejected": True,
            "mapwise_outcomes_retained": True,
            "scientific_failure_does_not_prune": True,
            "technical_invalidity_blocks_descendants": True,
            "complete_checkpoint_never_overwritten": True,
            "finite_window_not_asymptotic_proof": True,
        },
    }


def validate_design(value: Any) -> None:
    """Fail closed on design drift or inconsistent counts and budgets."""

    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("source-resolved campaign design is not canonical")
    registry = value["registry"]
    expected_maps = (
        registry["context_count"]
        * len(registry["layers"])
        * len(registry["heads"])
    )
    expected_pairs = (
        registry["rows_per_map"] * (registry["rows_per_map"] - 1) // 2
    )
    updates = snapshot_updates()
    if (
        registry["construction_context_count"]
        + registry["heldout_context_count"] != registry["context_count"]
        or expected_maps != registry["map_count"]
        or expected_pairs != registry["pairs_per_map"]
        or expected_maps * expected_pairs != registry["within_map_pair_count"]
        or registry["independent_contrast_dimension_per_map"]
        != (registry["rows_per_map"] - 1)
        * (registry["features_per_row"] - 1)
        or len(updates) != FROZEN_POLICY["total_snapshot_count"]
        or len(updates) - 1 != FROZEN_POLICY["regular_snapshot_count"]
        or updates[-2] != FROZEN_POLICY["last_regular_update"]
        or updates[-1] != FROZEN_POLICY["terminal_update"]
        or sum(stage["aggregate_gpu_hours_target"] for stage in STAGES)
        > RESOURCE_BUDGET["working_aggregate_gpu_hours"]
        or RESOURCE_BUDGET["working_aggregate_gpu_hours"]
        > RESOURCE_BUDGET["hard_aggregate_gpu_hours"]
    ):
        raise ValueError("source-resolved campaign counts or budgets disagree")
