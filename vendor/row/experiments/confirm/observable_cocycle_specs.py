"""Frozen design for the observable full-state AdamW cocycle study."""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-observable-full-state-cocycle-v1"
DESIGN_SCHEMA = "pldr-observable-cocycle-design-v1"
QUALIFICATION_SCHEMA = "pldr-observable-cocycle-qualification-v1"
CALIBRATION_SCHEMA = "pldr-observable-cocycle-calibration-v1"
LOCK_SCHEMA = "pldr-observable-cocycle-lock-v1"
PREDICTION_SCHEMA = "pldr-observable-cocycle-prediction-v1"
VALIDATION_SCHEMA = "pldr-observable-cocycle-validation-v1"
STRATUM_SCHEMA = "pldr-observable-cocycle-stratum-v1"
ANALYSIS_SCHEMA = "pldr-observable-cocycle-analysis-v1"

ARCHITECTURE = {
    "parameter_count": 20_622_914,
    "continuous_state_dimension": 61_868_742,
    "decoder_layers": 3,
    "heads_per_layer": 4,
    "head_width": 64,
    "registered_contexts": 24,
    "registered_maps_per_layer": 96,
}

TRAJECTORIES = (
    {
        "label": "A",
        "name": "construction-seed10444",
        "seed": 10_444,
        "role": "construction",
        "device": "cuda:0",
        "order": "construction-chunk-order.npy",
    },
    {
        "label": "B",
        "name": "transfer-seed11444",
        "seed": 11_444,
        "role": "heldout",
        "device": "cuda:0",
        "order": "transfer11444-chunk-order.npy",
    },
    {
        "label": "C",
        "name": "transfer-seed12444",
        "seed": 12_444,
        "role": "heldout",
        "device": "cuda:1",
        "order": "transfer12444-chunk-order.npy",
    },
)

SOURCE_STEPS = (4_096, 8_192, 16_384)
HORIZON_GRID = (1, 2, 4, 8, 16)
DIRECTION_CLASSES = (
    "row_adjoint",
    "optimizer_displacement",
    "balanced_full_state",
)

DIRECTION_POLICY = {
    "row_adjoint_seed": 49_101,
    "balanced_seed": 49_211,
    "row_adjoint_amplitudes": [2.0**-16, 2.0**-17],
    "optimizer_displacement_amplitudes": [2.0**-10, 2.0**-11],
    "balanced_full_state_amplitudes": [2.0**-17, 2.0**-18],
    "second_moment_boundary_policy": "zero-tangent-on-exact-zero-face",
    "block_balance": "equal-relative-euclidean-norm-across-nonzero-state-blocks",
    "random_distribution": "seeded-rademacher",
}

DECISION_POLICY = {
    "maximum_centered_response_relative_residual": 0.25,
    "minimum_halving_improvement_factor": 1.05,
    "absolute_effect_floor": 1.0e-8,
    "maximum_placebo_fraction": 0.1,
    "horizon_selection": "longest-jointly-qualified-else-one",
    "heldout_amplitude": "smaller-construction-amplitude",
    "operator_contraction_from_sampled_directions": False,
    "finite_horizon_implies_asymptotic_collapse": False,
    "sampled_direction_implies_operator_norm": False,
}

STRATUM_POLICY = {
    "construction": (
        "set-selected-final-layernorm-gate-and-first-moment-to-zero"
    ),
    "second_moment": "retain-checkpoint-value",
    "other_state": "retain-checkpoint-value",
    "horizon_updates": 1,
    "observed_force_zero_tolerance": 1.0e-10,
}

NUMERICAL_POLICY = {
    "dtype": "float32-native-successor",
    "observation_storage_dtype": "float32",
    "clipping_boundary_tolerance": 1.0e-7,
    "adamw_replay_relative_tolerance": 2.0e-6,
    "zero_second_moment_tangent_tolerance": 1.0e-20,
    "deterministic_algorithms": True,
}

RESOURCE_BUDGET = {
    "working_aggregate_reserved_device_hours": 2.0,
    "hard_aggregate_reserved_device_hours": 3.0,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 2 * 1024**3,
}


def amplitudes(direction: str) -> tuple[float, float]:
    """Return the fixed larger and smaller amplitudes for one direction."""

    if direction not in DIRECTION_CLASSES:
        raise ValueError(f"unknown full-state direction: {direction}")
    values = DIRECTION_POLICY[f"{direction}_amplitudes"]
    return float(values[0]), float(values[1])


def campaign_design() -> dict[str, Any]:
    """Return the exact content-addressed design payload."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "architecture": ARCHITECTURE,
        "trajectories": list(TRAJECTORIES),
        "source_steps": list(SOURCE_STEPS),
        "horizon_grid": list(HORIZON_GRID),
        "direction_classes": list(DIRECTION_CLASSES),
        "direction_policy": DIRECTION_POLICY,
        "decision_policy": DECISION_POLICY,
        "stratum_policy": STRATUM_POLICY,
        "numerical_policy": NUMERICAL_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "execution_policy": {
            "development_qualification_precedes_confirmation": True,
            "construction_lock_precedes_heldout_prediction": True,
            "prediction_and_validation_are_distinct_nodes": True,
            "validation_cannot_write_prediction_fields": True,
            "same_device_signed_comparisons": True,
            "all_registered_units_execute_independent_of_outcome": True,
            "all_adverse_outcomes_are_retained": True,
            "analysis_nodes_are_excluded_from_gpu_resource_totals": True,
            "source_snapshot_must_match_committed_launch": True,
        },
    }


def validate_design(value: Any) -> None:
    """Reject any drift in the frozen design."""

    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("observable-cocycle campaign design is not canonical")
    if (
        len(value["trajectories"]) != 3
        or len({row["seed"] for row in value["trajectories"]}) != 3
        or tuple(value["horizon_grid"]) != HORIZON_GRID
        or value["architecture"]["continuous_state_dimension"]
        != 3 * value["architecture"]["parameter_count"]
        or value["resource_budget"][
            "working_aggregate_reserved_device_hours"
        ]
        > value["resource_budget"]["hard_aggregate_reserved_device_hours"]
    ):
        raise ValueError("observable-cocycle design counts or budgets disagree")
