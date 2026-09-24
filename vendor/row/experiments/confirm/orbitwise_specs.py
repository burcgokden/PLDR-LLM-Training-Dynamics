"""Immutable design for the orbitwise row-map confirmation campaign."""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-orbitwise-row-map-collapse-v2"
DESIGN_SCHEMA = "pldr-orbitwise-design-v2"
FULL_TIMEPOINT_SCHEMA = "pldr-orbitwise-full-timepoint-v2"
DENSE_TIMEPOINT_SCHEMA = "pldr-orbitwise-dense-timepoint-v2"
SOURCE_PREDICTION_SCHEMA = "pldr-orbitwise-source-prediction-v2"
SOURCE_LINK_SCHEMA = "pldr-orbitwise-source-link-v2"
GATE_PREDICTION_SCHEMA = "pldr-orbitwise-gate-prediction-v2"
GATE_RESULT_SCHEMA = "pldr-orbitwise-gate-result-v2"
QUALIFICATION_SCHEMA = "pldr-orbitwise-qualification-v2"
ANALYSIS_SCHEMA = "pldr-orbitwise-analysis-v2"
LAUNCH_PLAN_SCHEMA = "pldr-orbitwise-launch-plan-v2"

REGISTRY = {
    "context_count": 24,
    "construction_context_count": 8,
    "transfer_context_count": 16,
    "layers": [0, 1, 2],
    "heads": [0, 1, 2, 3],
    "row_count": 64,
    "feature_count": 64,
    "map_count": 288,
    "sentinel_context_ids": ["c00", "c01"],
    "sentinel_map_count": 24,
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
        "name": "construction-seed10444",
        "role": "construction",
        "seed": 10444,
        "device": "cuda:0",
        "order_key": "construction",
    },
    {
        "name": "transfer-seed11444",
        "role": "transfer",
        "seed": 11444,
        "device": "cuda:0",
        "order_key": "transfer11444",
    },
    {
        "name": "transfer-seed12444",
        "role": "transfer",
        "seed": 12444,
        "device": "cuda:1",
        "order_key": "transfer12444",
    },
)

TERMINAL_UPDATE = 18_000
REGULAR_CADENCE = 32
PROBE_RESERVE_CHUNKS = 5_120
DENSE_HALF_WIDTH = 32
DENSE_CENTERS = (1_024, 4_096, 8_192, 16_384)
INTERVENTION_UPDATES = (4_096, 8_192, 16_384)
INTERVENTION_LENGTH = 64
PERMANENT_CHECKPOINT_UPDATES = (1_024, 4_096, 8_192, 16_384, 18_000)

TOLERANCES = {
    "float32_factorization_relative": 1.0e-5,
    "float64_identity_relative": 1.0e-12,
    "source_transport_relative": 1.0e-10,
    "source_reopening_absolute": 1.0e-12,
    "gate_intervention_relative": 1.0e-6,
}

RESOURCE_BUDGET = {
    "working_aggregate_gpu_hours": 5.0,
    "hard_aggregate_gpu_hours": 7.0,
    "queried_device_count": 2,
    "queried_gpu_capacity_mib": 24_564,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 16 * 1024**3,
    "poll_seconds": 1.0,
}

STAGES = (
    {
        "id": "C0",
        "purpose": "protocol-real-path-recovery-and-resource-qualification",
        "aggregate_gpu_hours_target": 0.1,
    },
    {
        "id": "C1",
        "purpose": "three-18000-update-orbits-and-dense-sentinels",
        "aggregate_gpu_hours_target": 3.8,
    },
    {
        "id": "C2",
        "purpose": "prospective-source-resolved-reopening",
        "aggregate_gpu_hours_target": 0.4,
    },
    {
        "id": "C3",
        "purpose": "paired-final-gate-freeze-interventions",
        "aggregate_gpu_hours_target": 0.6,
    },
    {
        "id": "C4",
        "purpose": "strict-orbitwise-assembly-and-analysis",
        "aggregate_gpu_hours_target": 0.1,
    },
)


def full_snapshot_updates() -> list[int]:
    """Return the 563 regular points plus the terminal point."""

    updates = list(range(0, TERMINAL_UPDATE, REGULAR_CADENCE))
    updates.append(TERMINAL_UPDATE)
    return updates


def dense_snapshot_updates() -> list[int]:
    """Return every state in the four closed 64-edge sentinel windows."""

    return sorted({
        update
        for center in DENSE_CENTERS
        for update in range(
            center - DENSE_HALF_WIDTH,
            center + DENSE_HALF_WIDTH + 1,
        )
    })


def source_edges() -> list[tuple[int, int]]:
    """Return the two registered edges adjacent to each dense center."""

    return [
        edge
        for center in DENSE_CENTERS
        for edge in ((center - 1, center), (center, center + 1))
    ]


def construction_checkpoint_updates() -> list[int]:
    """Return permanent and temporary source checkpoints in sorted order."""

    return sorted(
        set(PERMANENT_CHECKPOINT_UPDATES)
        | {endpoint for edge in source_edges() for endpoint in edge}
    )


def campaign_design() -> dict[str, Any]:
    """Return the canonical campaign design."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "registry": REGISTRY,
        "optimizer": OPTIMIZER,
        "trajectories": list(TRAJECTORIES),
        "terminal_update": TERMINAL_UPDATE,
        "probe_reserve_chunks": PROBE_RESERVE_CHUNKS,
        "regular_cadence": REGULAR_CADENCE,
        "full_snapshot_updates": full_snapshot_updates(),
        "dense_centers": list(DENSE_CENTERS),
        "dense_half_width": DENSE_HALF_WIDTH,
        "dense_snapshot_updates": dense_snapshot_updates(),
        "source_edges": [list(edge) for edge in source_edges()],
        "permanent_checkpoint_updates": list(PERMANENT_CHECKPOINT_UPDATES),
        "construction_checkpoint_updates": construction_checkpoint_updates(),
        "intervention_updates": list(INTERVENTION_UPDATES),
        "intervention_length": INTERVENTION_LENGTH,
        "tolerances": TOLERANCES,
        "resource_budget": RESOURCE_BUDGET,
        "stages": list(STAGES),
        "evidence_contract": {
            "finite_horizon_only": True,
            "full_map_outcomes_retained": True,
            "dense_one_update_reopening": True,
            "zero_energy_ratio_forbidden": True,
            "source_prediction_precedes_replay_successor": True,
            "complete_checkpoint_content_bound": True,
            "scientific_outcome_never_controls_launch": True,
            "strict_unknown_field_rejection": True,
            "absolute_plga_response_required": True,
        },
    }


def validate_design(value: Any) -> None:
    """Fail closed on schedule, count, or resource drift."""

    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("orbitwise campaign design is not canonical")
    full = full_snapshot_updates()
    dense = dense_snapshot_updates()
    edges = source_edges()
    if (
        len(full) != 564
        or full[0] != 0
        or full[-2] != 17_984
        or full[-1] != TERMINAL_UPDATE
        or len(dense) != 4 * 65
        or len(edges) != 8
        or any(right != left + 1 for left, right in edges)
        or REGISTRY["map_count"]
        != REGISTRY["context_count"]
        * len(REGISTRY["layers"])
        * len(REGISTRY["heads"])
        or REGISTRY["sentinel_map_count"]
        != len(REGISTRY["sentinel_context_ids"])
        * len(REGISTRY["layers"])
        * len(REGISTRY["heads"])
        or sum(stage["aggregate_gpu_hours_target"] for stage in STAGES)
        > RESOURCE_BUDGET["working_aggregate_gpu_hours"]
        or RESOURCE_BUDGET["working_aggregate_gpu_hours"]
        > RESOURCE_BUDGET["hard_aggregate_gpu_hours"]
    ):
        raise ValueError("orbitwise schedule, count, or resource budget drifted")
