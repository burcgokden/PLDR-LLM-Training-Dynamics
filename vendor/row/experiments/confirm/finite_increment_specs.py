"""Single owner for the all-layer finite-increment confirmation campaign."""

from __future__ import annotations


CAMPAIGN_ID = "pldr-finite-increment-collapse-confirmation-v1"
SCHEMA_VERSION = "pldr-finite-increment-design-v1"
REGISTRY_SCHEMA = "pldr-finite-increment-registry-v1"
PLAN_SCHEMA = "pldr-finite-increment-plan-v1"
REPORT_SCHEMA = "pldr-finite-increment-edge-report-v1"

ARCHITECTURE = {
    "layers": 3,
    "heads": 4,
    "head_width": 64,
    "rows_per_map": 64,
    "row_map_residual_units": 8,
    "gated_blocks_per_residual_unit": 2,
    "pre_chain_layernorm_count": 1,
    "postnorm_layernorm_count": 8,
    "layernorm_epsilon": 1.0e-6,
    "plga_adjustment_epsilon": 1.0e-9,
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
    "clip_equality_active": False,
    "batch_size": 32,
    "context_length": 256,
    "dtype": "float32",
}

TRAJECTORIES = (
    {
        "role": "construction",
        "name": "finite-construction-s7444",
        "seed": 7444,
        "updates": 9000,
    },
    {
        "role": "heldout",
        "name": "finite-heldout-s8444",
        "seed": 8444,
        "updates": 9000,
    },
    {
        "role": "heldout",
        "name": "finite-heldout-s9444",
        "seed": 9444,
        "updates": 9000,
    },
)

CONSTRUCTION_ANCHOR = 1000
CONSECUTIVE_EDGES = tuple(range(1000, 1020))
TIME_COURSE_ANCHORS = tuple(range(100, 9001, 100))
HELDOUT_ANCHORS = (1000, 3000, 5000, 7000, 9000)

REGISTRY = {
    "construction_context_count": 8,
    "heldout_context_count": 16,
    "context_count": 24,
    "context_length": 256,
    "heads": 4,
    "layers": (0, 1, 2),
    "rows_per_map": 64,
    "pairs_per_map": 2_016,
    "maps_per_full_registry": 288,
    "pairs_per_full_registry": 580_608,
    "pair_rule": "within-context-layer-head-all-unordered-pairs-v1",
    "construction_anchor": CONSTRUCTION_ANCHOR,
    "consecutive_edges": CONSECUTIVE_EDGES,
    "time_course_anchors": TIME_COURSE_ANCHORS,
    "heldout_anchors": HELDOUT_ANCHORS,
}

FINITE_INCREMENT_POLICY = {
    "source_only_endpoint": "native-adamw-clone-before-live-step-v1",
    "layernorm": "stable-endpoint-chord-v1",
    "affine_and_multilinear": "operation-ordered-exact-telescopes-v1",
    "plga_power": "base-first-and-exponent-first-intersection-v1",
    "softmax": "endpoint-chord-on-allowed-causal-prefix-v1",
    "pair_reduction": "stream-within-map-pairs-retain-maxima-and-near-active-v1",
    "outward_backend": "float64-directed-nextafter-with-proof-obligations-v1",
    "native_rotary": "float32-cast-recorded-and-charged-v1",
    "successor_evaluation": "same-native-endpoint-or-explicit-rounding-charge-v1",
    "absolute_fallback": True,
    "successor_values_forbidden_in_prediction": True,
}

RESOURCE_BUDGET = {
    "devices": ("cuda:0", "cuda:1"),
    "device_memory_mib_each": 24_564,
    "peak_reserved_gib_each": 20.0,
    "aggregate_gpu_hours_cap": 4.0,
    "persistent_output_gib_cap": 8.0,
    "pair_chunk_size": 32_768,
    "resource_probe_failure": "fail_closed",
    "cap_check_cadence": "before_and_after_every_node",
    "checkpoint_retention": "copy_model_only_after_all_descendants",
}

STAGES = (
    {
        "id": "E0",
        "purpose": "deterministic scalar and tensor identity qualification",
        "go_condition": "all_exact_identities_and_malformed_input_rejections_pass",
    },
    {
        "id": "E1",
        "purpose": "complete native endpoint replay on each registered device",
        "go_condition": "both_devices_cover_every_endpoint_identity_below_resource_caps",
    },
    {
        "id": "E2",
        "purpose": "all-map construction anchor at update 1000",
        "go_condition": "all_layers_covered_and_every_strict_sign_correct",
    },
    {
        "id": "E3",
        "purpose": "20 consecutive all-layer source-state edges",
        "go_condition": "every_prefix_covered_by_the_product_convolution",
    },
    {
        "id": "E4",
        "purpose": "three trajectory diameter time courses",
        "go_condition": "all_orders_digests_resources_and_time_courses_complete",
    },
    {
        "id": "E5",
        "purpose": "locked held-out all-layer anchors",
        "go_condition": "coverage_and_positive_state_dominance_hold_without_policy_change",
    },
)


def validate_design() -> None:
    if ARCHITECTURE["layers"] != 3 or ARCHITECTURE["heads"] != 4:
        raise ValueError("the implemented decoder architecture changed")
    if ARCHITECTURE["row_map_residual_units"] != 8:
        raise ValueError("the implemented row map needs eight residual units")
    if REGISTRY["pairs_per_map"] != (
        REGISTRY["rows_per_map"] * (REGISTRY["rows_per_map"] - 1) // 2
    ):
        raise ValueError("within-map pair count changed")
    if REGISTRY["maps_per_full_registry"] != (
        REGISTRY["context_count"] * REGISTRY["heads"] * len(REGISTRY["layers"])
    ):
        raise ValueError("full registry map count changed")
    if REGISTRY["pairs_per_full_registry"] != (
        REGISTRY["maps_per_full_registry"] * REGISTRY["pairs_per_map"]
    ):
        raise ValueError("full registry pair count changed")
    if CONSECUTIVE_EDGES != tuple(range(1000, 1020)):
        raise ValueError("the consecutive block must contain twenty edges")
    if len({row["seed"] for row in TRAJECTORIES}) != len(TRAJECTORIES):
        raise ValueError("trajectory seeds must be unique")
    if tuple(stage["id"] for stage in STAGES) != (
        "E0", "E1", "E2", "E3", "E4", "E5"
    ):
        raise ValueError("confirmation stages must remain chronological")


validate_design()
