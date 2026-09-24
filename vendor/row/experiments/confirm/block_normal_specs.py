"""Immutable design constants for block-normal row-map confirmation."""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-block-normal-collapse-confirmation-v1"
DESIGN_SCHEMA = "pldr-block-normal-design-v1"
REGISTRY_SCHEMA = "pldr-block-normal-registry-v1"
QUALIFICATION_SCHEMA = "pldr-block-normal-fixture-qualification-v1"
BLOCK_RECORD_SCHEMA = "pldr-block-normal-record-v1"
ANALYSIS_SCHEMA = "pldr-block-normal-analysis-v1"
SOURCE_EVIDENCE_SCHEMA = "pldr-block-normal-source-v1"
NATIVE_EVIDENCE_SCHEMA = "pldr-block-normal-native-v1"
SCIENTIFIC_OUTCOMES = ("confirmed", "not_confirmed", "unresolved")

ARCHITECTURE = {
    "trainable_parameters": 20_622_914,
    "decoder_layers": 3,
    "heads_per_layer": 4,
    "head_width": 64,
    "row_map_residual_units": 8,
    "gated_blocks_per_unit": 2,
}

OPTIMIZER = {
    "name": "AdamW",
    "batch_size": 32,
    "context_length": 256,
    "warmup_updates": 250,
    "constant_learning_rate": 7.5e-4,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1.0e-5,
    "gradient_clip": 1.0,
    "model_dtype": "float32",
    "reduction_dtype": "float64",
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

MASK_CONTRACT = {
    "padding_token": 0,
    "causal_exclusion": "strict-upper-triangular",
    "combination": "logical-or-before-dtype-conversion",
    "broadcast_order": ["batch", "head", "query", "key"],
}

SELECTION_POLICY = {
    "candidate_block_lengths": [16, 32, 64],
    "candidate_normal_ranks": [8, 16, 32, 64],
    "objective": [
        {"term": "lyapunov_residual", "weight": 1.0},
        {"term": "log_metric_condition", "weight": 0.05},
        {"term": "normal_rank", "weight": 1.0e-4},
    ],
    "tie_rule": "minimum-objective-then-shorter-block-then-lower-rank",
    "amplitude_menu": [0.03125, 0.015625, 0.0078125, 0.00390625],
    "sign_pair_fractions": [1.0, 0.5, 0.25],
    "projection_relative_tolerance": 1.0e-10,
    "rank_relative_tolerance": 1.0e-10,
    "lyapunov_relative_tolerance": 1.0e-9,
    "tube_relative_tolerance": 1.0e-8,
    "outcome_absolute_tolerance": 1.0e-12,
}

RESOURCE_BUDGET = {
    "working_aggregate_gpu_hours": 8.0,
    "hard_aggregate_gpu_hours": 10.0,
    "gpu_native_capacity_mib": 24_564,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 12 * 1024**3,
    "poll_seconds": 1.0,
    "maximum_complete_checkpoints_per_active_lineage": 2,
}

STAGES = (
    {
        "id": "Q0",
        "purpose": "deterministic-kernel-schema-executor-recovery-fixtures",
        "aggregate_gpu_hours_target": 0.1,
        "depends_on": [],
    },
    {
        "id": "Q1",
        "purpose": "four-consecutive-development-edges-1000-through-1003",
        "aggregate_gpu_hours_target": 0.3,
        "depends_on": ["Q0"],
    },
    {
        "id": "Q2",
        "purpose": "construction-timecourse-through-update-9000",
        "aggregate_gpu_hours_target": 0.8,
        "depends_on": ["Q1"],
    },
    {
        "id": "Q3",
        "purpose": "normal-response-and-lyapunov-construction",
        "aggregate_gpu_hours_target": 2.5,
        "depends_on": ["Q2"],
    },
    {
        "id": "Q4",
        "purpose": "normal-perturbations-and-optimizer-interventions",
        "aggregate_gpu_hours_target": 1.8,
        "depends_on": ["Q3"],
    },
    {
        "id": "Q5",
        "purpose": "two-locked-heldout-trajectories",
        "aggregate_gpu_hours_target": 2.5,
        "depends_on": ["Q3", "Q4"],
    },
)

ACTION_LEDGER = {
    "source": ["S1", "S2", "S3", "S4", "S5"],
    "normal": ["N1", "N2", "N3", "N4"],
    "lyapunov": ["L1", "L2", "L3"],
}


def campaign_design() -> dict[str, Any]:
    """Return the canonical prospective campaign design."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "architecture": ARCHITECTURE,
        "optimizer": OPTIMIZER,
        "registry": REGISTRY,
        "trajectories": list(TRAJECTORIES),
        "mask_contract": MASK_CONTRACT,
        "selection_policy": SELECTION_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "stages": list(STAGES),
        "action_ledger": ACTION_LEDGER,
        "scientific_outcomes": list(SCIENTIFIC_OUTCOMES),
        "policy_lock": {
            "construction_precedes_heldout": True,
            "source_prediction_precedes_native_endpoint": True,
            "source_side_quantities_only": True,
            "scientific_outcome_does_not_prune": True,
            "technical_invalidity_blocks_only_descendants": True,
            "native_and_conversion_radii_separate": True,
            "complete_checkpoint_never_overwritten": True,
        },
    }


def validate_design(value: Any) -> None:
    """Fail closed on a malformed or internally inconsistent design."""

    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("block-normal campaign design is not canonical")
    registry = value["registry"]
    expected_maps = (
        registry["context_count"]
        * len(registry["layers"])
        * len(registry["heads"])
    )
    expected_pairs = (
        registry["rows_per_map"] * (registry["rows_per_map"] - 1) // 2
    )
    if (
        registry["construction_context_count"]
        + registry["heldout_context_count"]
        != registry["context_count"]
        or expected_maps != registry["map_count"]
        or expected_pairs != registry["pairs_per_map"]
        or expected_maps * expected_pairs
        != registry["within_map_pair_count"]
        or sum(
            stage["aggregate_gpu_hours_target"]
            for stage in value["stages"]
        )
        > value["resource_budget"]["hard_aggregate_gpu_hours"]
    ):
        raise ValueError("block-normal campaign counts or budgets disagree")
