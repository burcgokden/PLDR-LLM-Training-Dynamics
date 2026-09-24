"""Frozen design for the temporal-context direct-work replication."""

from __future__ import annotations

from typing import Any


RELEASE_ID = "rev52"
CAMPAIGN_ID = "pldr-direct-work-temporal-context-replication-v2"
DESIGN_SCHEMA = "pldr-direct-work-replication-design-v1"
RECORD_SCHEMA = "pldr-direct-work-replication-record-v1"
RESOURCE_SCHEMA = "pldr-direct-work-replication-resource-v1"
LOCK_SCHEMA = "pldr-direct-work-replication-lock-v1"
ANALYSIS_SCHEMA = "pldr-direct-work-replication-analysis-v1"
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
SOURCE_STEPS = (18_000, 32_768, 65_536)
LAYERS = (0, 1, 2)
REGISTERED_CONTEXTS = 24
HEADS = 4
MAPS_PER_UNIT = REGISTERED_CONTEXTS * HEADS
DECISION_POLICY = {
    "maps_per_unit": MAPS_PER_UNIT,
    "minimum_evaluable_fraction": 2.0 / 3.0,
    "minimum_same_sign_fraction": 2.0 / 3.0,
    "source_updates_required": 2,
    "source_updates_planned": len(SOURCE_STEPS),
    "effect_floor_rule": (
        "max(1e-12,20-times-within-device-duplicate-energy-discrepancy,"
        "20-times-cross-device-energy-discrepancy)"
    ),
    "construction_layer_rule": (
        "at-least-two-of-three-source-updates-have-at-least-two-thirds-of-"
        "primary-evaluable-maps-on-the-same-side-of-the-effect-floor"
    ),
    "heldout_layer_support_rule": (
        "locked-sign-in-at-least-two-thirds-of-primary-evaluable-maps-at-"
        "two-of-three-source-updates-on-each-heldout-trajectory"
    ),
    "heldout_layer_refutation_rule": (
        "support-rule-fails-when-at-least-two-thirds-of-planned-maps-are-"
        "primary-evaluable-on-both-heldout-trajectories"
    ),
    "unresolved_layers_have_no_pooled_fallback": True,
}
NUMERICAL_POLICY = {
    "native_model_dtype": "float32",
    "native_observer": "float64-promoted-native-output",
    "factorized_observer": "float64-product-of-promoted-captured-factors",
    "work_charge_rounding_multiplier": 128.0,
    "reconstruction_rounding_multiplier": 128.0,
    "energy_summation_rounding_multiplier": 2.0,
    "coordinate_reconstruction_bound": (
        "128-epsilon-times-sum-of-primitive-coordinate-magnitudes"
    ),
    "energy_reconstruction_bound": (
        "2-gamma-4128-times-primitive-energy-magnitude"
    ),
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "allow_tf32": False,
}
RESOURCE_BUDGET = {
    "working_aggregate_reserved_device_hours": 0.35,
    "hard_aggregate_reserved_device_hours": 0.75,
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
        "registered_maps_per_unit": MAPS_PER_UNIT,
        "intervention": {
            "primary_estimand": (
                "control-native-endpoint-energy-minus-natural-native-"
                "endpoint-energy"
            ),
            "natural_arm": "implemented-clipped-adamw-step",
            "control_arm": (
                "remove-selected-centered-incoming-row-map-adjoint-only"
            ),
            "paired_state": (
                "identical-parameters-moments-batch-clock-clipping-rng-device"
            ),
        },
        "observer_policy": {
            "native_causal_eligibility": True,
            "factorization_magnitude_is_secondary": True,
            "four_source_reconstruction_is_technical": True,
            "sources": [
                "shape",
                "gate",
                "interaction",
                "implementation-defect",
            ],
        },
        "temporal_policy": {
            "chronological_source_steps": [18_000, 32_768],
            "counterfactual_terminal_source_step": 65_536,
            "terminal_continuation_uses_registered_reserve_batch": True,
            "terminal_node_is_not_a_historical_training_successor": True,
        },
        "decision_policy": DECISION_POLICY,
        "numerical_policy": NUMERICAL_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "execution_policy": {
            "dual_device_qualification_precedes_construction": True,
            "construction_technical_closure_precedes_lock": True,
            "construction_lock_precedes_heldout_analysis": True,
            "all_science_nodes_execute_independent_of_outcome": True,
            "gpu_science_cannot_fallback_to_cpu": True,
            "one_scientific_process_per_device": True,
            "binding_root_is_required": True,
        },
    }


def validate_design(value: Any) -> None:
    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("direct-work replication design is not canonical")
    if (
        value["registered_maps_per_unit"] != 96
        or len(value["trajectories"]) != 3
        or value["resource_budget"]["working_aggregate_reserved_device_hours"]
        > value["resource_budget"]["hard_aggregate_reserved_device_hours"]
    ):
        raise ValueError("direct-work replication counts or budgets disagree")
