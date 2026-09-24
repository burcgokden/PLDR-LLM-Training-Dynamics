"""Single owner for the direct contrast-energy confirmation campaign."""

from __future__ import annotations


CAMPAIGN_ID = "pldr-contrast-energy-cocycle-confirmation-v1"
SCHEMA_VERSION = "pldr-contrast-energy-design-v1"
LOCK_SCHEMA = "pldr-contrast-energy-construction-lock-v1"

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
    "decay_mask": "all_trainable_coordinates",
    "gradient_clip_kind": "value",
    "gradient_clip_value": 1.0,
    "clip_equality_active": False,
    "batch_size": 32,
    "context_length": 256,
    "dtype": "float32",
}

TRAJECTORIES = (
    {"role": "development", "name": "contrast-development-s7444-o0",
     "seed": 7444, "data_offset_chunks": 0, "updates": 9000},
    {"role": "construction", "name": "contrast-construction-s8444-o8192",
     "seed": 8444, "data_offset_chunks": 8192, "updates": 9000},
    {"role": "heldout", "name": "contrast-heldout-s9444-o24576",
     "seed": 9444, "data_offset_chunks": 24576, "updates": 9000},
    {"role": "heldout", "name": "contrast-heldout-s10444-o40960",
     "seed": 10444, "data_offset_chunks": 40960, "updates": 9000},
    {"role": "heldout", "name": "contrast-heldout-s11444-o57344",
     "seed": 11444, "data_offset_chunks": 57344, "updates": 9000},
    {"role": "heldout", "name": "contrast-heldout-s12444-o73728",
     "seed": 12444, "data_offset_chunks": 73728, "updates": 9000},
)

ANCHORS = tuple(range(1000, 9001, 500))

REGISTRY = {
    "context_count": 24,
    "context_length": 256,
    "complete_rows_per_context": 64,
    "vertex_count": 1536,
    "pair_count": 1_178_880,
    "pair_chunk_size": 32_768,
    "anchor_steps": ANCHORS,
    "development_anchor_steps": ANCHORS,
    "layers": (0, 1, 2),
    "history_length": 48,
    "construction_block_lengths": (1,),
    "development_edges": 102,
    "heldout_edges": 204,
}

NUMERICAL_POLICIES = {
    "float32_model_tensor": {
        "safety_factor": 8.0,
        "operation_count_recorded": True,
    },
    "successor_values_forbidden_in_prediction": True,
    "bitwise_native_successor_required": True,
}

TAYLOR_POLICY = {
    "path": "native_executed_adamw_affine_parameter_segment",
    "model_order": 3,
    "initial_subdivisions": 2,
    "maximum_subdivisions": 64,
    "adaptive_rule": "bisect_largest_weighted_width_then_lexicographic",
    "positive_third_modulus_required": True,
    "explicit_layernorm_primitives": True,
    "rope_interval_enclosure": True,
    "pair_energy_formed_before_interval_widening": True,
    "successor_values_forbidden": True,
    "outward_dtype": "float64_nextafter",
}

RESOURCE_BUDGET = {
    "devices": ("cuda:0", "cuda:1"),
    "device_memory_mib_each": 24_564,
    "provisional_peak_gib_each": 20.0,
    "stage_b_enlargement": 1.15,
    "aggregate_gpu_hours_lower": 7.1,
    "aggregate_gpu_hours_upper": 9.6,
    "wall_hours_lower": 4.0,
    "wall_hours_upper": 6.0,
    "aggregate_output_gib_cap": 16.0,
    "pair_chunk_size": 32_768,
    "persistent_checkpoint_kind": "model_only",
    "temporary_full_checkpoint_consumed_before_prune": True,
}


def validate_design() -> None:
    if ARCHITECTURE["row_map_residual_units"] != 8:
        raise ValueError("the implemented row map needs eight residual units")
    if REGISTRY["vertex_count"] != (
        REGISTRY["context_count"] * REGISTRY["complete_rows_per_context"]
    ):
        raise ValueError("registry vertex count is inconsistent")
    if REGISTRY["pair_count"] != (
        REGISTRY["vertex_count"] * (REGISTRY["vertex_count"] - 1) // 2
    ):
        raise ValueError("registry pair count is inconsistent")
    if len(ANCHORS) != 17 or ANCHORS[0] != 1000 or ANCHORS[-1] != 9000:
        raise ValueError("the anchor grid changed")
    if len({row["seed"] for row in TRAJECTORIES}) != len(TRAJECTORIES):
        raise ValueError("trajectory seeds must be unique")
    if REGISTRY["development_edges"] != 2 * len(ANCHORS) * 3:
        raise ValueError("development/construction edge count changed")
    if REGISTRY["heldout_edges"] != 4 * len(ANCHORS) * 3:
        raise ValueError("held-out edge count changed")


validate_design()
