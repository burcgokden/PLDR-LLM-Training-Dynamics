"""Locked design constants for source-restoring confirmation."""

from __future__ import annotations

import json


CAMPAIGN_ID = "pldr-source-restoring-collapse-confirmation-v1"
DESIGN_SCHEMA = "pldr-source-restoring-design-v1"
REGISTRY_SCHEMA = "pldr-source-restoring-registry-v1"
PLAN_SCHEMA = "pldr-source-restoring-plan-v1"
LAUNCH_SCHEMA = "pldr-source-restoring-launch-v1"
TRAINING_SCHEMA = "pldr-source-restoring-training-campaign-v1"

ARCHITECTURE = {
    "layers": 3,
    "heads": 4,
    "head_width": 64,
    "rows_per_map": 64,
    "row_map_residual_units": 8,
    "gated_blocks_per_residual_unit": 2,
    "layernorm_epsilon": 1.0e-6,
    "plga_adjustment_epsilon": 1.0e-9,
    "parameter_count": 20_622_914,
}

REGISTRY = {
    "construction_context_count": 8,
    "heldout_context_count": 16,
    "context_count": 24,
    "layers": (0, 1, 2),
    "heads": 4,
    "rows_per_map": 64,
    "pairs_per_map": 2_016,
    "maps_per_full_registry": 288,
    "pairs_per_full_registry": 580_608,
    "pair_rule": "within-context-layer-head-all-unordered-pairs-v1",
}

OPTIMIZER = {
    "name": "AdamW",
    "learning_rate": 7.5e-4,
    "warmup_updates": 250,
    "constant_rate_after_warmup": True,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1.0e-5,
    "weight_decay": 0.1,
    "gradient_clip_kind": "value",
    "gradient_clip_value": 1.0,
    "batch_size": 32,
    "context_length": 256,
    "native_dtype": "float32",
    "certificate_dtype": "float64-outward",
}

TRAJECTORIES = (
    {
        "role": "construction",
        "name": "source-construction-s7444",
        "seed": 7444,
        "updates": 9_001,
    },
    {
        "role": "heldout",
        "name": "source-heldout-s8444",
        "seed": 8444,
        "updates": 9_001,
    },
    {
        "role": "heldout",
        "name": "source-heldout-s9444",
        "seed": 9444,
        "updates": 9_001,
    },
    {
        "role": "independent",
        "name": "source-independent-s10444",
        "seed": 10_444,
        "updates": 9_001,
    },
)

PILOT_EDGES = (1000, 1001, 1002, 1003)
CONSTRUCTION_EDGES = tuple(range(1004, 1024))
TIMECOURSE_ANCHORS = tuple(range(100, 9001, 100))
HELDOUT_ANCHORS = (1000, 3000, 5000, 7000, 9000)

MASK_CONTRACT = {
    "padding_id": 0,
    "causal_mask": "strict-upper-triangle-negative-infinity-v1",
    "padding_mask": "token-equals-padding-id-v1",
    "combination_rule": "logical-or-before-additive-conversion-v1",
    "dtype": "model-native",
    "shape": "batch-head-query-key-broadcast-v1",
    "digest_required": True,
}

CERTIFICATE_POLICY = {
    "source_direction": "complete-clipped-adamw-state-v1",
    "work_split": (
        "final-postnorm-gain",
        "nonfinal-row-map-shape",
        "upstream-query-plga-input",
    ),
    "remainder": "source-segment-validated-taylor-or-finite-telescope-v1",
    "plga_crossing": "multiplied-two-coordinate-contributions-v1",
    "native_charge": "source-program-outward-arithmetic-v1",
    "successor_values_forbidden": True,
    "exhaustive_pairs": True,
    "scientific_outcomes": ("confirmed", "not_confirmed", "unresolved"),
    "scientific_outcomes_halt_descendants": False,
    "technical_invalidity_halts": True,
}

RESOURCE_BUDGET = {
    "devices": ("cuda:0", "cuda:1"),
    "device_memory_mib_each": 24_564,
    "peak_reserved_gib_each": 20.0,
    "aggregate_gpu_hours_cap": 4.0,
    "persistent_output_gib_cap": 8.0,
    "projected_storage_checked_before_launch": True,
    "live_wall_and_memory_checked_per_node": True,
    "native_peak_memory_required_for_cuda": True,
    "checkpoint_retention": "preserve-complete-until-descendants-close",
    "model_only_copy_path_must_differ": True,
}

STAGES = (
    {
        "id": "P0",
        "purpose": "analytic fixtures, source closure, and real-path qualification",
        "budget_gpu_hours": 0.05,
    },
    {
        "id": "P1",
        "purpose": "four consecutive all-layer development edges",
        "budget_gpu_hours": 0.35,
    },
    {
        "id": "P2",
        "purpose": "twenty-edge source-cocycle construction block",
        "budget_gpu_hours": 1.2,
    },
    {
        "id": "P3",
        "purpose": "construction spatial-mechanism time course",
        "budget_gpu_hours": 0.4,
    },
    {
        "id": "P4",
        "purpose": "locked held-out spatial and temporal confirmation",
        "budget_gpu_hours": 0.4,
    },
    {
        "id": "P5",
        "purpose": "independent locked confirmation after policy freeze",
        "budget_gpu_hours": 0.8,
    },
)


def training_campaign_spec(registry_sha256: str) -> dict:
    """Return the content-addressed native-training campaign binding."""

    value = {
        "schema_version": TRAINING_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "architecture": ARCHITECTURE,
        "optimizer": OPTIMIZER,
        "trajectories": TRAJECTORIES,
        "registry_sha256": registry_sha256,
        "mask_contract": MASK_CONTRACT,
        "certificate_policy": CERTIFICATE_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "stages": STAGES,
    }
    value = json.loads(json.dumps(value, sort_keys=True))
    from confirm.confirmation_artifacts import digest_object

    value["spec_sha256"] = digest_object(value)
    return value


def validate_design() -> None:
    if REGISTRY["pairs_per_map"] != (
        REGISTRY["rows_per_map"] * (REGISTRY["rows_per_map"] - 1) // 2
    ):
        raise ValueError("the pair registry is inconsistent")
    if REGISTRY["maps_per_full_registry"] != (
        REGISTRY["context_count"] * len(REGISTRY["layers"]) * REGISTRY["heads"]
    ):
        raise ValueError("the map registry is inconsistent")
    if REGISTRY["pairs_per_full_registry"] != (
        REGISTRY["pairs_per_map"] * REGISTRY["maps_per_full_registry"]
    ):
        raise ValueError("the full pair count is inconsistent")
    if tuple(row["id"] for row in STAGES) != (
        "P0", "P1", "P2", "P3", "P4", "P5"
    ):
        raise ValueError("the stage order changed")
    if sum(row["budget_gpu_hours"] for row in STAGES) >= 4.0:
        raise ValueError("stage targets must remain below the hard GPU-hour cap")
    if len({row["seed"] for row in TRAJECTORIES}) != len(TRAJECTORIES):
        raise ValueError("trajectory seeds must be unique")
    if any(row["updates"] <= max(HELDOUT_ANCHORS) for row in TRAJECTORIES):
        raise ValueError("every trajectory must retain the final successor")
    if len(CONSTRUCTION_EDGES) != 20:
        raise ValueError("the construction block must contain twenty edges")
    if len(TIMECOURSE_ANCHORS) != 90:
        raise ValueError("the time course must contain ninety points")


validate_design()
