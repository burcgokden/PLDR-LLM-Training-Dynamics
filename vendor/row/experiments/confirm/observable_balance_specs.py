"""Frozen design for the finite-increment observable-balance study."""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-finite-increment-observable-balance-v1"
RELEASE_ID = "rev50"
DESIGN_SCHEMA = "pldr-observable-balance-design-v1"
QUALIFICATION_SCHEMA = "pldr-observable-balance-qualification-v1"
CONSTRUCTION_SCHEMA = "pldr-observable-balance-construction-v1"
LOCK_SCHEMA = "pldr-observable-balance-lock-v1"
PREDICTION_SCHEMA = "pldr-observable-balance-prediction-v1"
VALIDATION_SCHEMA = "pldr-observable-balance-validation-v1"
STRATUM_SCHEMA = "pldr-gate-zero-controlled-gradient-v1"
ANALYSIS_SCHEMA = "pldr-observable-balance-analysis-v1"

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
        "order": "construction-chunk-order.npy",
    },
    {
        "label": "B",
        "name": "transfer-seed11444",
        "seed": 11_444,
        "role": "heldout",
        "order": "transfer11444-chunk-order.npy",
    },
    {
        "label": "C",
        "name": "transfer-seed12444",
        "seed": 12_444,
        "role": "heldout",
        "order": "transfer12444-chunk-order.npy",
    },
)

SOURCE_STEPS = (4_096, 8_192, 16_384)
HORIZON_GRID = (1, 2, 4, 8)
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
    "absolute_effect_floor": 1.0e-8,
    "maximum_reconstruction_relative_residual": 2.0e-4,
    "maximum_adamw_replay_relative_residual": 2.0e-6,
    "minimum_halving_improvement_factor": 1.05,
    "reconstruction_equality_floor": 2.0e-7,
    "minimum_cancellation_index": 10.0,
    "maximum_response_to_homogeneous_ratio": 0.1,
    "minimum_evaluable_fraction": 2.0 / 3.0,
    "campaign_support_fraction": 2.0 / 3.0,
    "campaign_support_unit": "heldout-trajectory",
    "campaign_support_rule": (
        "at-least-two-thirds-cancellation-dominant-in-each-heldout-trajectory"
    ),
    "campaign_refutation_rule": (
        "support-rule-fails-with-at-least-two-thirds-evaluable-in-both-trajectories"
    ),
    "horizon_selection": "longest-jointly-evaluable-else-one",
    "heldout_amplitude": "smaller-construction-amplitude",
    "finite_horizon_implies_asymptotic_collapse": False,
    "sampled_direction_implies_operator_norm": False,
}

STRATUM_POLICY = {
    "source": "selected-final-gate-and-first-moment-exactly-zero",
    "second_moment": "retain-checkpoint-value",
    "natural_arm": "implemented-clipped-gradient-write",
    "controlled_arm": "zero-only-selected-clipped-gate-gradient-before-moment-write",
    "horizon_updates": 1,
    "formula_match": "bitwise",
    "controlled_face_match": "bitwise-zero",
    "source_step": 4_096,
}

NUMERICAL_POLICY = {
    "dtype": "float32-native-successor",
    "observation_storage_dtype": "float32",
    "clipping_boundary_tolerance": 1.0e-7,
    "adamw_replay_relative_tolerance": 2.0e-6,
    "zero_second_moment_tangent_tolerance": 1.0e-20,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "allow_tf32": False,
}

RESOURCE_BUDGET = {
    "working_aggregate_reserved_device_hours": 2.0,
    "hard_aggregate_reserved_device_hours": 3.0,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 2 * 1024**3,
}

# Verified attempt-ledger components completed before this campaign. They are
# typed inputs so the manuscript total is generated rather than pinned as
# prose. One reserved-device hour is 3,600 wall-clock seconds for which a
# named accelerator was reserved; simultaneous reservations add.
ARCHIVE_PRIOR_ATTEMPTS = {
    "definition": (
        "sum-over-all-attempts-of-named-accelerator-reservation-seconds-divided-by-3600"
    ),
    "components": [
        {"class": "training-and-long-capture", "hours": 6.6760},
        {"class": "unsuccessful-launch", "hours": 4.4733},
        {"class": "layer-resolved-cocycle", "hours": 0.4936},
        {"class": "full-state-observable-cocycle", "hours": 0.1324},
    ],
}


def amplitudes(direction: str) -> tuple[float, float]:
    if direction not in DIRECTION_CLASSES:
        raise ValueError(f"unknown full-state direction: {direction}")
    values = DIRECTION_POLICY[f"{direction}_amplitudes"]
    return float(values[0]), float(values[1])


def campaign_design() -> dict[str, Any]:
    return {
        "schema_version": DESIGN_SCHEMA,
        "release_id": RELEASE_ID,
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
        "archive_prior_attempts": ARCHIVE_PRIOR_ATTEMPTS,
        "execution_policy": {
            "development_qualification_precedes_construction": True,
            "construction_lock_precedes_heldout_prediction": True,
            "prediction_and_validation_are_distinct_nodes": True,
            "validation_cannot_write_prediction_fields": True,
            "same_device_signed_comparisons": True,
            "all_registered_units_execute_independent_of_outcome": True,
            "all_adverse_terminal_outcomes_are_retained": True,
            "analysis_role_excluded_from_resource_totals": True,
            "planned_devices_are_immutable": True,
            "gpu_science_cannot_fallback_to_cpu": True,
            "runtime_and_incident_membership_is_sealed": True,
        },
    }


def validate_design(value: Any) -> None:
    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("observable-balance campaign design is not canonical")
    if (
        len(value["trajectories"]) != 3
        or len({row["seed"] for row in value["trajectories"]}) != 3
        or tuple(value["horizon_grid"]) != HORIZON_GRID
        or value["architecture"]["continuous_state_dimension"]
        != 3 * value["architecture"]["parameter_count"]
        or value["resource_budget"]["working_aggregate_reserved_device_hours"]
        > value["resource_budget"]["hard_aggregate_reserved_device_hours"]
    ):
        raise ValueError("observable-balance design counts or budgets disagree")
