"""Frozen design for the direct physical-work source intervention."""

from __future__ import annotations

from typing import Any


RELEASE_ID = "rev52"

CAMPAIGN_ID = "pldr-direct-work-source-intervention-v1"
DESIGN_SCHEMA = "pldr-direct-work-design-v1"
RECORD_SCHEMA = "pldr-direct-work-record-v1"
RESOURCE_SCHEMA = "pldr-direct-work-resource-record-v2"
LOCK_SCHEMA = "pldr-direct-work-construction-lock-v1"
ANALYSIS_SCHEMA = "pldr-direct-work-analysis-v1"

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
LAYERS = (0, 1, 2)
REGISTERED_CONTEXTS = 24
HEADS = 4
ROWS = 64
WIDTH = 64

DECISION_POLICY = {
    "maps_per_unit": REGISTERED_CONTEXTS * HEADS,
    "minimum_evaluable_fraction": 2.0 / 3.0,
    "minimum_same_sign_fraction": 2.0 / 3.0,
    "source_updates_required": 2,
    "source_updates_planned": len(SOURCE_STEPS),
    "effect_floor_rule": (
        "max(1e-12,20-times-qualification-duplicate-absolute-energy-difference)"
    ),
    "construction_layer_rule": (
        "at-least-two-of-three-source-updates-have-at-least-two-thirds-of-"
        "evaluable-maps-on-the-same-side-of-the-frozen-effect-floor"
    ),
    "heldout_layer_support_rule": (
        "locked-sign-in-at-least-two-thirds-of-evaluable-maps-at-two-of-three-"
        "source-updates-on-each-heldout-trajectory"
    ),
    "heldout_layer_refutation_rule": (
        "support-rule-fails-when-at-least-two-thirds-of-planned-maps-are-"
        "evaluable-on-both-heldout-trajectories"
    ),
    "unresolved_layers_have_no_pooled_fallback": True,
}

NUMERICAL_POLICY = {
    "native_model_dtype": "float32",
    "physical_observer_dtype": "float64-promoted-native-output",
    "identity_relative_tolerance": 2.0e-5,
    "gate_shape_relative_tolerance": 5.0e-5,
    "identity_absolute_tolerance": 2.0e-10,
    "optimizer_replay_absolute_tolerance": 0.0,
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "allow_tf32": False,
}

RESOURCE_BUDGET = {
    "working_aggregate_reserved_device_hours": 1.5,
    "hard_aggregate_reserved_device_hours": 2.5,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 2 * 1024**3,
    "node_wall_seconds": 900.0,
}


def campaign_design() -> dict[str, Any]:
    return {
        "schema_version": DESIGN_SCHEMA,
        "release_id": RELEASE_ID,
        "campaign_id": CAMPAIGN_ID,
        "trajectories": list(TRAJECTORIES),
        "source_steps": list(SOURCE_STEPS),
        "layers": list(LAYERS),
        "registered_maps_per_unit": REGISTERED_CONTEXTS * HEADS,
        "physical_map_shape": [ROWS, WIDTH],
        "intervention": {
            "natural_arm": "implemented-clipped-adamw-step",
            "control_arm": (
                "replace-only-selected-row-map-output-adjoint-by-its-row-"
                "constant-orthogonal-projection-before-upstream-backpropagation"
            ),
            "forward_values_identical": True,
            "row_constant_adjoint_retained": True,
            "centered_row_adjoint_removed": True,
            "paired_state": (
                "identical-parameters-moments-batch-clock-clipping-rng-and-device"
            ),
        },
        "decision_policy": DECISION_POLICY,
        "numerical_policy": NUMERICAL_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "execution_policy": {
            "qualification_precedes_construction": True,
            "construction_lock_precedes_heldout_analysis": True,
            "all_planned_units_execute_independent_of_outcome": True,
            "gpu_science_cannot_fallback_to_cpu": True,
            "one_scientific_process_per_device": True,
            "missing_driver_metadata_is_fatal": True,
            "binding_root_is_required": True,
        },
    }


def validate_design(value: Any) -> None:
    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("direct-work campaign design is not canonical")
    budget = value["resource_budget"]
    if (
        budget["working_aggregate_reserved_device_hours"]
        > budget["hard_aggregate_reserved_device_hours"]
        or value["registered_maps_per_unit"] != 96
        or len(value["trajectories"]) != 3
    ):
        raise ValueError("direct-work campaign counts or budgets disagree")
