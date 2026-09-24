"""Single owner for the causal row-map confirmation campaign."""

from __future__ import annotations


CAMPAIGN_ID = "pldr-causal-row-map-confirmation-v1"
SCHEMA_VERSION = "pldr-causal-row-map-design-v1"
LOCK_SCHEMA = "pldr-causal-row-map-construction-lock-v1"

ARCHITECTURE = {
    "layers": 3,
    "heads": 4,
    "head_width": 64,
    "row_map_residual_units": 8,
    "gated_blocks_per_residual_unit": 2,
    "pre_chain_layernorm_count": 1,
    "postnorm_layernorm_count": 8,
    "layernorm_epsilon": 1.0e-6,
    "parameter_count": 20_622_914,
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
    "dtype": "float32",
}

TRAJECTORIES = (
    {
        "role": "development",
        "name": "causal-development-s7444-o0",
        "seed": 7444,
        "data_offset_chunks": 0,
        "updates": 9000,
    },
    {
        "role": "construction",
        "name": "causal-construction-s8444-o8192",
        "seed": 8444,
        "data_offset_chunks": 8192,
        "updates": 9000,
    },
    {
        "role": "heldout",
        "name": "causal-heldout-s9444-o24576",
        "seed": 9444,
        "data_offset_chunks": 24576,
        "updates": 9000,
    },
    {
        "role": "heldout",
        "name": "causal-heldout-s10444-o40960",
        "seed": 10444,
        "data_offset_chunks": 40960,
        "updates": 9000,
    },
    {
        "role": "heldout",
        "name": "causal-heldout-s11444-o57344",
        "seed": 11444,
        "data_offset_chunks": 57344,
        "updates": 9000,
    },
    {
        "role": "heldout",
        "name": "causal-heldout-s12444-o73728",
        "seed": 12444,
        "data_offset_chunks": 73728,
        "updates": 9000,
    },
)

ANCHORS = tuple(range(1000, 9001, 500))

REGISTRY = {
    "context_count": 24,
    "context_length": 256,
    "complete_rows_per_context": 64,
    "vertex_count": 1536,
    "pair_count": 1_178_880,
    "pair_chunk_size": 32768,
    "anchor_steps": ANCHORS,
    "development_anchor_steps": ANCHORS,
    "construction_block_lengths": (1,),
    "history_length": 48,
    "intervention_anchor_steps": (2000, 4000, 6000, 8000),
    "intervention_horizons": (1, 2, 3, 4),
}

NUMERICAL_POLICIES = {
    "float64_algebra": {
        "dtype": "float64",
        "forward_error_gamma": True,
        "operation_count_recorded": True,
        "safety_factor": 8.0,
    },
    "float32_execution": {
        "dtype": "float32",
        "bitwise_when_deterministic": True,
        "ulp_bound_otherwise": True,
    },
    "cross_provenance": {
        "decisive": False,
        "forward_error_allowance_required": True,
    },
    "directional_remainder": {
        "successor_values_forbidden": True,
        "complete_parameter_segment": True,
        "outward_rational_bounds": True,
        "segment_grid": (0.0, 0.5, 1.0),
        "segment_subdivisions": 2,
        "numerical_shape_jvp_error_required": True,
    },
}

RESOURCE_BUDGET = {
    "devices": ("cuda:0", "cuda:1"),
    "device_memory_mib_each": 24_564,
    "aggregate_gpu_hours_cap": 8.0,
    "wall_hours_cap": 4.0,
    "aggregate_output_gib_cap": 8.0,
    "pair_chunk_size": 32768,
    "persistent_checkpoint_kind": "model_only",
    "temporary_full_checkpoint_consumed_before_prune": True,
}


def validate_design() -> None:
    if ARCHITECTURE["row_map_residual_units"] != 8:
        raise ValueError("the implemented row map needs eight residual units")
    if ARCHITECTURE["postnorm_layernorm_count"] != 8:
        raise ValueError("every residual unit must be post-normalized")
    if REGISTRY["vertex_count"] != (
        REGISTRY["context_count"] * REGISTRY["complete_rows_per_context"]
    ):
        raise ValueError("registry vertex count is inconsistent")
    if REGISTRY["pair_count"] != (
        REGISTRY["vertex_count"] * (REGISTRY["vertex_count"] - 1) // 2
    ):
        raise ValueError("registry pair count is inconsistent")
    if len({row["seed"] for row in TRAJECTORIES}) != len(TRAJECTORIES):
        raise ValueError("trajectory seeds must be unique")
    if len({row["data_offset_chunks"] for row in TRAJECTORIES}) != len(
        TRAJECTORIES
    ):
        raise ValueError("trajectory data offsets must be unique")
    if ANCHORS[0] != 1000 or ANCHORS[-1] != 9000 or len(ANCHORS) != 17:
        raise ValueError("the time-course anchor grid changed")


validate_design()
