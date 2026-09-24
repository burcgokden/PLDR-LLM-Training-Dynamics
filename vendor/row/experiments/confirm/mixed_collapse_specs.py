"""Frozen design for the mixed-collapse confirmation campaign.

The campaign starts each trajectory from initialization.  Its explicit
chunk schedule preserves the entire predecessor training permutation and then
appends a seeded permutation of newly archived, disjoint chunks.  Every chunk
index is globally unique over the complete 65,536-update trajectory.
"""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-mixed-row-map-collapse-v1"
DESIGN_SCHEMA = "pldr-mixed-collapse-design-v1"
FULL_TIMEPOINT_SCHEMA = "pldr-mixed-collapse-full-timepoint-v1"
DENSE_TIMEPOINT_SCHEMA = "pldr-mixed-collapse-dense-timepoint-v1"
ANALYSIS_SCHEMA = "pldr-mixed-collapse-analysis-v1"
FACTORIAL_RESULT_SCHEMA = "pldr-gate-factorial-result-v1"
SOURCE_RADIUS_SCHEMA = "pldr-source-frozen-radius-v1"

REGISTRY = {
    "context_count": 24,
    "construction_context_count": 8,
    "transfer_context_count": 16,
    "layers": [0, 1, 2],
    "heads": [0, 1, 2, 3],
    "row_count": 64,
    "feature_count": 64,
    "map_count": 288,
    "map_order": "context-major-layer-major-head-major-v1",
}

OPTIMIZER = {
    "name": "AdamW",
    "batch_size": 32,
    "context_length": 256,
    "warmup_updates": 250,
    "learning_rate": 7.5e-4,
    "constant_learning_rate_after_warmup": True,
    "weight_decay": 0.1,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1.0e-5,
    "gradient_clip": 1.0,
    "model_dtype": "float32",
    "analysis_dtype": "float64",
}

TRAJECTORIES = (
    {
        "label": "A",
        "name": "construction-seed10444",
        "role": "construction",
        "seed": 10444,
        "preferred_device": "cuda:0",
        "order_key": "construction",
    },
    {
        "label": "B",
        "name": "transfer-seed11444",
        "role": "heldout",
        "seed": 11444,
        "preferred_device": "cuda:0",
        "order_key": "transfer11444",
    },
    {
        "label": "C",
        "name": "transfer-seed12444",
        "role": "heldout",
        "seed": 12444,
        "preferred_device": "cuda:1",
        "order_key": "transfer12444",
    },
)

TERMINAL_UPDATE = 65_536
REGULAR_CADENCE = 32
PROBE_RESERVE_CHUNKS = 5_120
DENSE_HALF_WIDTH = 32
DENSE_CENTERS = (1_024, 4_096, 8_192, 16_384, 32_768, 65_536)
BLOCK_ANCHORS = (1_024, 2_048, 4_096, 8_192, 16_384, 32_768, 65_536)
INTERVENTION_UPDATES = (4_096, 8_192, 16_384)
INTERVENTION_LENGTH = 64
INTERVENTION_MODES = ("full", "frozen", "decay_only", "adaptive_only")
SOURCE_RADIUS_UPDATES = (4_096, 8_192, 16_384)

# The construction arithmetic probes reached 6.69e-5.  The threshold was
# rounded upward before evaluating either heldout trajectory.
SOURCE_FROZEN_RADIUS = 1.0e-4

PERMANENT_CHECKPOINT_UPDATES = (
    1_024,
    4_096,
    8_192,
    16_384,
    18_000,
    32_768,
    65_536,
)

TOLERANCES = {
    "construction_float32_factorization_relative": 1.0e-5,
    "float64_identity_relative": 1.0e-12,
    "duhamel_reconstruction_relative": 1.0e-10,
    "source_frozen_radius": SOURCE_FROZEN_RADIUS,
}

RESOURCE_BUDGET = {
    "working_aggregate_gpu_hours": 8.0,
    "hard_aggregate_gpu_hours": 10.0,
    "queried_device_count": 2,
    "queried_gpu_capacity_mib": 24_564,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 32 * 1024**3,
}


def full_snapshot_updates() -> list[int]:
    """Return cadence-32 states plus the terminal state."""

    updates = list(range(0, TERMINAL_UPDATE, REGULAR_CADENCE))
    updates.append(TERMINAL_UPDATE)
    return updates


def dense_snapshot_updates() -> list[int]:
    """Return fixed all-map dense windows, clipped to the run horizon."""

    return sorted({
        update
        for center in DENSE_CENTERS
        for update in range(
            max(0, center - DENSE_HALF_WIDTH),
            min(TERMINAL_UPDATE, center + DENSE_HALF_WIDTH) + 1,
        )
    })


def campaign_design() -> dict[str, Any]:
    """Return the canonical scientific design before provenance signing."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "registry": REGISTRY,
        "optimizer": OPTIMIZER,
        "trajectories": list(TRAJECTORIES),
        "terminal_update": TERMINAL_UPDATE,
        "regular_cadence": REGULAR_CADENCE,
        "full_snapshot_updates": full_snapshot_updates(),
        "dense_centers": list(DENSE_CENTERS),
        "dense_half_width": DENSE_HALF_WIDTH,
        "dense_snapshot_updates": dense_snapshot_updates(),
        "block_anchors": list(BLOCK_ANCHORS),
        "intervention_updates": list(INTERVENTION_UPDATES),
        "intervention_length": INTERVENTION_LENGTH,
        "intervention_modes": list(INTERVENTION_MODES),
        "source_radius_updates": list(SOURCE_RADIUS_UPDATES),
        "source_frozen_radius": SOURCE_FROZEN_RADIUS,
        "permanent_checkpoint_updates": list(PERMANENT_CHECKPOINT_UPDATES),
        "probe_reserve_chunks": PROBE_RESERVE_CHUNKS,
        "data_order_policy": {
            "kind": "prefix-locked-disjoint-extension",
            "predecessor_training_chunks": 854_255,
            "complete_training_chunks": TERMINAL_UPDATE * OPTIMIZER["batch_size"],
            "preserve_predecessor_permutation": True,
            "extension_generator": "numpy-pcg64-seed-plus-47000003",
            "globally_unique_chunk_indices": True,
        },
        "tolerances": TOLERANCES,
        "resource_budget": RESOURCE_BUDGET,
        "evidence_contract": {
            "finite_horizon_only": True,
            "all_map_dense_windows": True,
            "per_map_factorization_decisions": True,
            "exact_excursion_distinguished_from_cadence_lower_bound": True,
            "gate_effects_aggregated_by_seed_layer_horizon": True,
            "scientific_outcome_never_controls_execution": True,
            "source_radius_fixed_before_heldout_successors": True,
            "plga_and_local_robustness_do_not_gate_upstream_analysis": True,
        },
    }


def validate_design(value: Any) -> None:
    """Fail closed on any scientific schedule or resource drift."""

    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("mixed-collapse campaign design is not canonical")
    if (
        full_snapshot_updates()[0] != 0
        or full_snapshot_updates()[-1] != TERMINAL_UPDATE
        or dense_snapshot_updates()[-1] != TERMINAL_UPDATE
        or tuple(sorted(BLOCK_ANCHORS)) != BLOCK_ANCHORS
        or BLOCK_ANCHORS[-1] != TERMINAL_UPDATE
        or REGISTRY["map_count"]
        != REGISTRY["context_count"]
        * len(REGISTRY["layers"])
        * len(REGISTRY["heads"])
        or SOURCE_FROZEN_RADIUS <= 0.0
        or RESOURCE_BUDGET["working_aggregate_gpu_hours"]
        > RESOURCE_BUDGET["hard_aggregate_gpu_hours"]
    ):
        raise ValueError("mixed-collapse schedule or resource budget drifted")
