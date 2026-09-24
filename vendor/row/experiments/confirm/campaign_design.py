"""Pre-outcome design for the structured-path E0-E8 confirmation campaign."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from campaign_full_stack_design import CERTIFICATE_RULES


SCHEMA_VERSION = "pldr-row-map-confirmation-design-v3"

DATA = {
    "tokens": "experiments/data/refinedweb_tokens.npy",
    "tokens_sha256":
        "7127603cbd401986b3b9bd114b5d35723d04ddff2e9624da0d013932d04ee7d7",
    "tokens_manifest": "experiments/data/refinedweb_tokens.npy.manifest.json",
    "tokens_manifest_sha256":
        "177903068532f98aa002ae2ba88bad52f26a087450897eaffafc3b04f3c60c08",
    "tokenizer": "experiments/data/tokenizer.model",
    "tokenizer_sha256":
        "51f4369714712232bfc746188f347e550790d1d75e5e35bfa4399b784d2a666f",
    "data_offset_chunks": 0,
    "probe_region": "global",
}

ARCHITECTURES = {
    "w064_d04": {
        "model_width": 64,
        "model_depth": 4,
        "layers": 4,
        "heads": 4,
        "dk": 16,
        "adff": 42,
    },
    "w128_d08": {
        "model_width": 128,
        "model_depth": 8,
        "layers": 8,
        "heads": 4,
        "dk": 32,
        "adff": 85,
    },
    "w256_d12": {
        "model_width": 256,
        "model_depth": 12,
        "layers": 12,
        "heads": 4,
        "dk": 64,
        "adff": 170,
    },
}

SCHEDULES = {
    "short_warmup": {
        "maximum_learning_rate": 0.001,
        "warmup_steps": 250,
        "anneal_floor": 0.1,
        "hold_until": -1,
        "total_steps": 16000,
    },
    "long_warmup": {
        "maximum_learning_rate": 0.001,
        "warmup_steps": 2000,
        "anneal_floor": 0.1,
        "hold_until": -1,
        "total_steps": 16000,
    },
    "low_floor": {
        "maximum_learning_rate": 0.001,
        "warmup_steps": 1000,
        "anneal_floor": 0.01,
        "hold_until": -1,
        "total_steps": 16000,
    },
}

COMMON_TRAINING = {
    "batch": 32,
    "context": 256,
    "gradient_accumulation": 1,
    "optimizer": "adamw",
    "beta1": 0.9,
    "beta2": 0.95,
    "adam_epsilon": 1e-5,
    "weight_decay": 0.1,
    "gradient_value_clip": 1.0,
    "loss_mode": "block",
    "probe_every": 50,
    "sharp_block_every": 100,
    "sharp_every": 250,
    "sharp_pre_every": 1000,
    "sharp_full": 0,
    "generation_prompts": 16,
    "generation_continuations": 8,
    "finite_difference_steps": [0.005, 0.01, 0.02, 0.04],
    "dtype": "float32_training_float64_certificate",
}

BASELINE = {
    "architecture_cell": "w256_d12",
    "schedule": {
        "maximum_learning_rate": 0.001,
        "warmup_steps": 1000,
        "anneal_floor": 0.1,
        "hold_until": -1,
        "total_steps": 16000,
    },
    "seeds": list(range(1, 25)),
    "e0_e6_seeds": list(range(1, 11)),
    "e0_e1_e3_e5_e6_seeds": list(range(1, 9)),
    "model_checkpoint_steps": sorted(set(
        list(range(1000, 16001, 1000))
        + [step + 1 for step in range(1000, 13000, 1000)]
        + list(range(10000, 10011))
    )),
    "optimizer_checkpoint_steps": [9000, 11000, 13000],
    "transport_snapshot_steps": [4000, 8000],
    "transport_tensor_groups": ["query", "key", "value", "deductive"],
    "certificate_snapshot_plan": {
        "4000": ["query", "key", "value", "deductive"],
        "6000": ["deductive"],
        "8000": ["query", "key", "value", "deductive"],
        **{
            str(step): ["deductive"]
            for step in [
                9000, *range(10000, 10011),
                11000, 12000, 13000, 14000, 15000,
            ]
        },
    },
}

E0_COVER = {
    "center_rows_per_layer": 16,
    "heldout_rows_per_layer": 8,
    "probe_batch_size": 8,
    "probe_offset": 0,
    "coordinate_padding": 0.0,
    "cover_strategy": "registered_pairwise_segment_hull",
    "subdivisions_per_axis": 1,
    "subdivisions_per_segment": 8,
    "maximum_boxes": 4096,
    "joint_parameter_segment_updates": 1,
    "primary_domain": "registered_pairwise_segment_hull",
    "resource_action_limit": 2000000,
    "resource_storage_bytes_limit": 2147483648,
}

SCALAR_DIAGNOSTIC_RULES = {
    "row_chart_projection": {
        "row": "first_registered_e0_row_per_layer",
        "input_direction": "alternating_sign_unit_vector",
        "output_direction": "signed_integer_ramp_unit_vector",
        "dimension": "one_scalar_per_decoder_layer",
    },
    "normal_operator": {
        "rule": "positive_scalar_least_squares_on_projected_Q_and_Z",
        "ridge": 1e-6,
        "maximum_eigenvalue": 1000.0,
        "reference_forcing": "Q_minus_K_times_Z_at_registered_center",
    },
    "auxiliary_family": {
        "nominal_step": 10000,
        "nominal_steps": [10000, 10001],
        "outer_steps": list(range(10000, 10010)),
        "radius_rounding": "nextafter_positive_infinity",
    },
    "structured_family": {
        "primitive_coordinates": [
            "alpha", "decay", "normal_eigenvalue",
        ],
        "family_rule": "multiaffine_primitive_box_corner_hull",
        "corner_count": 8,
        "vertex_test": "exact_cross_metric_ldl",
        "window_length": 4,
        "path_rule": "all_registered_graph_paths",
    },
    "defect_envelope": {
        "steps": list(range(10001, 10010)),
        "geometric_rate": 0.95,
        "persistent_rule": "final_step_complete_disturbance_norm",
        "transient_rule": "maximum_required_K_over_frozen_window",
        "positive_comparison_rule": "scalar_rate_matrix",
    },
    "self_map": {
        "fit_steps": list(range(10001, 10010)),
        "heldout_step": 10010,
        "entry_radius_multiplier": 1.25,
        "successor_radius_multiplier": 1.10,
        "auxiliary_radius_multiplier": 1.25,
    },
    "finite_entry": {
        "criterion": 1.0,
        "criterion_unit": "output_row_unit_per_input_row_unit",
        "certificate_checkpoints": list(range(8000, 16000, 1000)),
        "terminal_step": 16000,
        "sustained_window_checkpoints": 2,
        "geometric_cover_rate": 0.95,
    },
}

E7_INTERVENTIONS = {
    "source_checkpoints": [9000, 11000, 13000],
    "certificate_window_offsets": [252, 253, 254, 255, 256],
    "continuation_steps": 256,
    "outcome_checkpoints": [9256, 11256, 13256],
    "terminal_snapshot_blocks": ["deductive"],
    "terminal_validation": True,
    "generation_skipped": True,
    "dose_application": (
        "multiply_the_scheduled_rate_before_each_optimizer_step"
    ),
    "arms": {
        "contraction_improving": {
            "learning_rate_multiplier": 0.5,
            "description":
                "constant loading reduction for 256 matched updates",
        },
        "defect_increasing": {
            "learning_rate_multiplier": 1.15,
            "description":
                "near-edge loading increase for 256 matched updates",
        },
        "sham": {
            "learning_rate_multiplier": 1.0,
            "description":
                "matched continuation with zero schedule dose",
        },
    },
    "avalanche_window_updates": 256,
    "avalanche_detector": {
        "signal": "gnorm",
        "support_signals": [
            "gnorm_phi", "gnorm_plga", "gnorm_rest",
        ],
        "reference_window_updates": 256,
        "threshold_mad_multiplier": 3.0,
        "minimum_relative_scale": 0.01,
        "event_rule": "maximal_contiguous_strict_threshold_exceedance",
        "size_rule": "sum_signal_over_reference_threshold",
        "support_rule": "mean_number_of_block_threshold_exceedances",
    },
}

E8_SPLIT = {
    "training_seeds": [25, 26],
    "heldout_seeds": list(range(101, 109)),
    "architecture_cells": list(ARCHITECTURES),
    "schedule_cells": list(SCHEDULES),
}
E8_SPLIT["sha256"] = hashlib.sha256(json.dumps(
    E8_SPLIT, sort_keys=True, separators=(",", ":"), allow_nan=False,
).encode("utf-8")).hexdigest()

E8 = {
    "split": E8_SPLIT,
    "checkpoint": 16000,
    "certificate_time": 8000,
    "certificate_checkpoint_steps": list(range(8000, 16001, 1000)),
    "certificate_snapshot_steps": list(range(8000, 16000, 1000)),
    "certificate_snapshot_blocks": ["deductive"],
    "block_sizes": [2, 4, 8],
    "model_arms": [
        "theory", "constant_rate", "loss_only", "unconstrained_trend",
    ],
    "heldout_partition_locked_before_fit": True,
    "baseline_fit_order": [
        "constant_rate", "loss_only", "unconstrained_trend", "theory",
    ],
    "prompt_offset": 2048,
    "cached_and_independent_inference": True,
}

RESOURCES = {
    "gpu_indices": [0, 1],
    "maximum_power_watts_per_gpu": 320,
    "minimum_free_workspace_bytes": 429496729600,
    "maximum_parallel_training_jobs": 2,
    "checkpoint_retention": "all_registered_and_branch_source_checkpoints",
    "raw_records_retention": "permanent",
    "development_runs_excluded": True,
}

DESIGN = {
    "schema_version": SCHEMA_VERSION,
    "status": "FROZEN_BEFORE_CONFIRMATORY_OUTCOMES",
    "data": DATA,
    "architectures": ARCHITECTURES,
    "schedules": SCHEDULES,
    "common_training": COMMON_TRAINING,
    "scalar_diagnostic_rules": SCALAR_DIAGNOSTIC_RULES,
    "baseline": BASELINE,
    "e0_cover": E0_COVER,
    "certificate_rules": CERTIFICATE_RULES,
    "e7_interventions": E7_INTERVENTIONS,
    "e8": E8,
    "resources": RESOURCES,
}


def canonical_json(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def design_object():
    value = deepcopy(DESIGN)
    value["design_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return value


def validate_design():
    design = design_object()
    for name, cell in design["architectures"].items():
        if cell["heads"] * cell["dk"] != cell["model_width"]:
            raise AssertionError(f"architecture width mismatch: {name}")
        if cell["layers"] != cell["model_depth"]:
            raise AssertionError(f"architecture depth mismatch: {name}")
    if (
        design["e7_interventions"]["outcome_checkpoints"]
        != [
            source + design["e7_interventions"]["continuation_steps"]
            for source in design["e7_interventions"]["source_checkpoints"]
        ]
    ):
        raise AssertionError("E7 outcome checkpoints do not follow sources")
    plan = design["baseline"]["certificate_snapshot_plan"]
    if set(plan["4000"]) != set(design["baseline"]["transport_tensor_groups"]):
        raise AssertionError("E1 tensor snapshot plan is stale")
    if sorted(int(step) for step in plan) != [
        4000, 6000, 8000, 9000, *range(10000, 10011),
        11000, 12000, 13000, 14000, 15000,
    ]:
        raise AssertionError("certificate snapshot steps are stale")
    if not set(range(10000, 10011)).issubset(
        design["baseline"]["model_checkpoint_steps"]
    ):
        raise AssertionError("dense certificate model checkpoints are stale")
    if (
        design["e0_cover"]["cover_strategy"]
        != "registered_pairwise_segment_hull"
        or design["e0_cover"]["coordinate_padding"] != 0.0
        or design["e0_cover"]["subdivisions_per_segment"] < 2
    ):
        raise AssertionError("physical chordal row cover is stale")
    if (
        design["certificate_rules"]["complete_stack"]["criterion_layers"]
        != "every_layer_zero_through_depth_minus_one"
        or design["certificate_rules"]["complete_block_family"][
            "path_length"] != 4
    ):
        raise AssertionError("complete stack or path-family rule is stale")
    if design["certificate_rules"]["entry"]["criterion"] != 1.0:
        raise AssertionError("physical direct-entry criterion is stale")
    if design["certificate_rules"]["source_ledger"]["trace_fit_allowed"]:
        raise AssertionError("a fitted source entered the primary design")
    if set(design["certificate_rules"]["time_semantics"]["allowed"]) != {
        "FINITE_IMPLEMENTED_SCHEDULE", "FROZEN_CHECKPOINT",
        "CONTROLLED_INFINITE_TAIL",
    }:
        raise AssertionError("time semantics are incomplete")
    if design["e7_interventions"]["terminal_snapshot_blocks"] != [
        "deductive",
    ]:
        raise AssertionError("E7 terminal snapshot plan is stale")
    if design["e8"]["certificate_checkpoint_steps"] != list(
        range(8000, 16001, 1000)
    ) or design["e8"]["certificate_snapshot_steps"] != list(
        range(8000, 16000, 1000)
    ):
        raise AssertionError("E8 certificate trajectory is stale")
    if design["e8"]["split"]["heldout_seeds"] != list(range(101, 109)):
        raise AssertionError("E8 held-out seeds are stale")
    if design["resources"]["maximum_power_watts_per_gpu"] > 320:
        raise AssertionError("registered GPU power exceeds the site limit")
    return True
