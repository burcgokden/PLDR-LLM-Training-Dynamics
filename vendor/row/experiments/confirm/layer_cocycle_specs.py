"""Frozen design for the layer-resolved row-map cocycle study."""

from __future__ import annotations

from typing import Any


CAMPAIGN_ID = "pldr-layer-resolved-cocycle-confirmation-v1"
DESIGN_SCHEMA = "pldr-layer-cocycle-design-v1"
PRECISION_SCHEMA = "pldr-layer-cocycle-precision-v1"
GATE_OPERATOR_SCHEMA = "pldr-layer-cocycle-gate-operator-v1"
INTERVENTION_SCHEMA = "pldr-layer-cocycle-intervention-v1"
PERTURBATION_SCHEMA = "pldr-layer-cocycle-perturbation-v1"
PLGA_SCHEMA = "pldr-layer-cocycle-plga-defect-v1"
ANALYSIS_SCHEMA = "pldr-layer-cocycle-analysis-v1"
RESOURCE_SCHEMA = "pldr-layer-cocycle-resource-v1"

ARCHITECTURE = {
    "decoder_layers": 3,
    "heads_per_layer": 4,
    "head_width": 64,
    "row_map_residual_units": 8,
    "registered_contexts": 24,
    "registered_maps": 288,
    "maps_per_layer": 96,
}

TRAJECTORIES = (
    {
        "label": "A",
        "name": "construction-seed10444",
        "seed": 10444,
        "role": "construction",
        "device": "cuda:0",
        "order": "construction-chunk-order.npy",
    },
    {
        "label": "B",
        "name": "transfer-seed11444",
        "seed": 11444,
        "role": "heldout",
        "device": "cuda:0",
        "order": "transfer11444-chunk-order.npy",
    },
    {
        "label": "C",
        "name": "transfer-seed12444",
        "seed": 12444,
        "role": "heldout",
        "device": "cuda:1",
        "order": "transfer12444-chunk-order.npy",
    },
)

PRECISION_STEPS = (16_384, 65_536)
GATE_OPERATOR_STEP = 4_096
INTERVENTION_STEP = 4_096

RESOLUTION_POLICY = {
    "finite_registry_energy_threshold": 1.0e-3,
    "arithmetic_safety_factor": 8.0,
    "relative_denominator_floor": 1.0e-30,
    "float64_factorization_relative_tolerance": 1.0e-12,
}

GATE_OPERATOR_POLICY = {
    "clipping_boundary_tolerance": 1.0e-10,
    "hessian_construction": "weighted-basiswise-forward-over-reverse",
    "autodiff_crosscheck_scope": (
        "implemented-adamw-composition-given-exact-loss-gradient-and-hessian"
    ),
    "dimensionless_probe_amplitudes": [2.0 ** -10, 2.0 ** -11],
    "direction_seed": 48_101,
    "direction_classes": ["gate", "first_moment", "second_moment", "mixed"],
    "autodiff_relative_tolerance": 2.0e-9,
    "maximum_smaller_probe_relative_residual": 1.0e-2,
    "minimum_probe_residual_reduction": 1.25,
    "scaled_invariance_defect_zero_tolerance": 1.0e-8,
}

INTERVENTION_ARMS = (
    "native",
    "noop_restamp",
    "one_ulp_placebo",
    "gate_update_removed",
    "upstream_update_removed",
    "joint_row_program_update_removed",
    "gate_decay_only",
    "gate_adaptive_only",
)

INTERVENTION_POLICY = {
    "horizon_updates": 1,
    "placebo_parameter": "selected-layer-final-gate-weight",
    "placebo_coordinate": 0,
    "placebo_direction": "positive_float32_nextafter",
    "placebo_fraction_of_treatment": 0.1,
    "energy_effect_floor": 1.0e-12,
}

PERTURBATION_POLICY = {
    "construction_amplitude": 2.0 ** -12,
    "validation_amplitude_fraction": 0.5,
    "direction_seed": 48_211,
    "direction_scope": "one-layer-upstream-row-program-parameters",
    "normalization": "global-euclidean-relative-to-parameter-norm",
    "prediction": "centered-central-secant-at-half-amplitude",
    "maximum_validation_relative_residual": 0.25,
}

PLGA_POLICY = {
    "regularization_constant": 1.0e-9,
    "row_preservation_tolerance": 1.0e-10,
    "defect_zero_tolerance": 1.0e-12,
}

RESOURCE_BUDGET = {
    "working_aggregate_gpu_hours": 2.0,
    "hard_aggregate_gpu_hours": 3.0,
    "process_reserved_memory_cap_bytes": 20 * 1024**3,
    "persistent_output_cap_bytes": 2 * 1024**3,
}


def campaign_design() -> dict[str, Any]:
    """Return the exact content-addressed design payload."""

    return {
        "schema_version": DESIGN_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "architecture": ARCHITECTURE,
        "trajectories": list(TRAJECTORIES),
        "precision_steps": list(PRECISION_STEPS),
        "gate_operator_step": GATE_OPERATOR_STEP,
        "intervention_step": INTERVENTION_STEP,
        "resolution_policy": RESOLUTION_POLICY,
        "gate_operator_policy": GATE_OPERATOR_POLICY,
        "intervention_arms": list(INTERVENTION_ARMS),
        "intervention_policy": INTERVENTION_POLICY,
        "perturbation_policy": PERTURBATION_POLICY,
        "plga_policy": PLGA_POLICY,
        "resource_budget": RESOURCE_BUDGET,
        "execution_policy": {
            "committed_clean_source_required": True,
            "source_snapshot_must_match_commit": True,
            "same_device_scientific_comparisons": True,
            "construction_prediction_precedes_validation": True,
            "contexts_and_heads_are_within_layer_subsamples": True,
            "identity_checks_are_regressions_not_scientific_decisions": True,
            "all_adverse_outcomes_reported": True,
            "long_branches_not_authorized": True,
        },
    }


def validate_design(value: Any) -> None:
    """Reject any drift in the frozen design."""

    if not isinstance(value, dict) or value != campaign_design():
        raise ValueError("layer-cocycle campaign design is not canonical")
    if (
        len(value["trajectories"]) != 3
        or len({row["seed"] for row in value["trajectories"]}) != 3
        or value["resource_budget"]["working_aggregate_gpu_hours"]
        > value["resource_budget"]["hard_aggregate_gpu_hours"]
        or value["architecture"]["registered_maps"]
        != value["architecture"]["registered_contexts"]
        * value["architecture"]["decoder_layers"]
        * value["architecture"]["heads_per_layer"]
    ):
        raise ValueError("layer-cocycle design counts or budgets disagree")
