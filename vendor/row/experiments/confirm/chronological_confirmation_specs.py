"""Single owner for the chronological-collapse confirmation design."""

from __future__ import annotations

import math


CAMPAIGN_ID = "pldr-chronological-collapse-confirmation-v1"
SCHEMA_VERSION = "pldr-chronological-confirmation-design-v1"

ARCHITECTURE = {
    "layers": 3,
    "heads": 4,
    "head_width": 64,
    "row_map_residual_units": 8,
    "parameter_count": 20_622_914,
}

OPTIMIZER = {
    "name": "AdamW",
    "learning_rate": 7.5e-4,
    "warmup_updates": 250,
    "constant_rate_after_warmup": True,
    "beta1": 0.9,
    "beta2": 0.95,
    "epsilon": 1e-5,
    "weight_decay": 0.1,
    "gradient_clip_kind": "value",
    "gradient_clip_value": 1.0,
    "batch_size": 32,
    "context_length": 256,
    "dtype": "float32",
}


TRAJECTORIES = (
    {
        "role": "development",
        "name": "chron-development-s7444-o0",
        "seed": 7444,
        "data_offset_chunks": 0,
        "updates": 9001,
    },
    {
        "role": "construction",
        "name": "chron-construction-s8444-o8192",
        "seed": 8444,
        "data_offset_chunks": 8192,
        "updates": 24001,
    },
    {
        "role": "heldout",
        "name": "chron-heldout-s9444-o24576",
        "seed": 9444,
        "data_offset_chunks": 24576,
        "updates": 24001,
    },
    {
        "role": "heldout",
        "name": "chron-heldout-s10444-o40960",
        "seed": 10444,
        "data_offset_chunks": 40960,
        "updates": 24001,
    },
    {
        "role": "heldout",
        "name": "chron-heldout-s11444-o57344",
        "seed": 11444,
        "data_offset_chunks": 57344,
        "updates": 24001,
    },
    {
        "role": "heldout",
        "name": "chron-heldout-s12444-o73728",
        "seed": 12444,
        "data_offset_chunks": 73728,
        "updates": 24001,
    },
)

REGISTRY = {
    "context_count": 24,
    "context_length": 256,
    "complete_rows_per_context": 64,
    "vertex_count": 1536,
    "pair_count": 1_178_880,
    "pair_chunk_size": 32768,
    "anchor_steps": (2000, 4000, 8000, 16000, 24000),
    "development_anchor_steps": (2000, 4000, 8000),
    "construction_block_lengths": (16, 64, 256),
    "history_length": 48,
    "intervention_anchor_step": 8000,
    "intervention_sites_per_seed": 12,
    "intervention_positive_sites_required": 10,
    "intervention_steps": 4,
    "intervention_layer_rule": "ordinal-modulo-three",
    "intervention_context_rule": "ordinal-modulo-sixteen",
    "intervention_head_rule": "context-modulo-four",
    "intervention_pair": (0, 63),
}

NUMERICAL_POLICIES = {
    "float64_algebra": {
        "dtype": "float64",
        "safety_factor": 16.0,
        "operation_count_recorded": True,
    },
    "float32_model_tensor": {
        "dtype": "float32",
        "safety_factor": 32.0,
        "promote_before_algebra": True,
    },
    "microbatch_reduction": {
        "source_dtype": "float32",
        "safety_factor": 64.0,
        "condition_scale_recorded": True,
        "residual_enters_numerical_charge": True,
    },
}

DECISION_RULES = {
    "development_gate": {
        "minimum_positive_windows": 1,
        "requires_pre_update_margin": True,
        "requires_pair_envelope_coverage": True,
        "requires_shape_remainder_coverage": True,
        "requires_diameter_contraction": True,
    },
    "construction": {
        "report_every_registered_cell": True,
        "primary_block_rule": (
            "smallest registered block with simultaneous shape and numerical "
            "coverage in the development window class"
        ),
        "constant_enlargement_after_lock": False,
        "extremal_cell_selection": False,
    },
    "heldout": {
        "required_trajectory_count": 4,
        "all_trajectories_must_cover": True,
        "all_registered_layer_anchor_cells_reported": True,
        "data_offsets_must_be_distinct": True,
    },
    "intervention": {
        "sites_per_seed": 12,
        "positive_sites_required": 10,
        "fair_sign_probability_per_seed": 79 / 4096,
        "source_anchor_step": 8000,
        "future_update_start_rule": "four-times-ordinal",
        "direction_from_source_removal_theorem": True,
        "magnitude_lower_bound_required": True,
        "seed_clustered_continuous_effect_bound": True,
    },
    "plga": {
        "positive_occupied_bases": True,
        "float64_identity_replay": True,
        "rectangular_query_key": True,
        "construction_owned_directional_bound": True,
    },
}

RESOURCE_BUDGET = {
    "aggregate_gpu_hours_cap": 10.0,
    "wall_hours_cap": 6.5,
    "peak_gpu_gib_per_job": 22.0,
    "aggregate_output_gib": 8.0,
    "estimated_gpu_hours": {
        "qualification": 0.25,
        "development": 0.65,
        "construction": 1.95,
        "heldout_training": 3.85,
        "heldout_capture": 0.90,
        "interventions": 0.60,
        "plga_and_finalization": 0.20,
    },
}

STAGES = (
    {
        "id": "Q",
        "title": "code and algebra qualification",
        "requires": (),
        "stop_on_failure": True,
    },
    {
        "id": "D",
        "title": "positive-branch development pilot",
        "requires": ("Q",),
        "stop_on_failure": True,
    },
    {
        "id": "C",
        "title": "construction and immutable lock",
        "requires": ("D",),
        "stop_on_failure": True,
    },
    {
        "id": "H",
        "title": "four held-out trajectories",
        "requires": ("C",),
        "stop_on_failure": True,
    },
    {
        "id": "I",
        "title": "theorem-directed isolated interventions",
        "requires": ("H",),
        "stop_on_failure": True,
    },
    {
        "id": "P",
        "title": "occupied-segment PLGA transfer",
        "requires": ("H",),
        "stop_on_failure": True,
    },
    {
        "id": "F",
        "title": "independent analysis and release",
        "requires": ("I", "P"),
        "stop_on_failure": True,
    },
)


def validate_design() -> None:
    names = [row["name"] for row in TRAJECTORIES]
    seeds = [row["seed"] for row in TRAJECTORIES]
    offsets = [row["data_offset_chunks"] for row in TRAJECTORIES]
    if len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
        raise ValueError("trajectory identities and seeds must be unique")
    if len(set(offsets)) != len(offsets):
        raise ValueError("every trajectory must have a distinct data offset")
    if OPTIMIZER["context_length"] != REGISTRY["context_length"]:
        raise ValueError("training and registry context lengths disagree")
    if REGISTRY["vertex_count"] != (
        REGISTRY["context_count"] * REGISTRY["complete_rows_per_context"]
    ):
        raise ValueError("registry vertex count is inconsistent")
    expected_pairs = (
        REGISTRY["vertex_count"] * (REGISTRY["vertex_count"] - 1) // 2)
    if REGISTRY["pair_count"] != expected_pairs:
        raise ValueError("registry pair count is inconsistent")
    if (
        REGISTRY["intervention_anchor_step"] not in REGISTRY["anchor_steps"]
        or REGISTRY["intervention_pair"] != (0, 63)
    ):
        raise ValueError("intervention anchor or row pair is inconsistent")
    intervention = DECISION_RULES["intervention"]
    null_rate = sum(
        math.comb(intervention["sites_per_seed"], count)
        for count in range(
            intervention["positive_sites_required"],
            intervention["sites_per_seed"] + 1)
    ) / 2 ** intervention["sites_per_seed"]
    if null_rate != intervention["fair_sign_probability_per_seed"]:
        raise ValueError("intervention null rate is inconsistent")
    if abs(
        sum(RESOURCE_BUDGET["estimated_gpu_hours"].values()) - 8.40
    ) > 1e-12:
        raise ValueError("resource estimate must total 8.40 GPU-hours")


validate_design()
