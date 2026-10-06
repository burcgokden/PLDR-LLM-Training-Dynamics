"""Frozen design for the observable-margin confirmation.

This module contains data, statistical, arithmetic, and resource constants
only.  Generators serialize the values without consulting experiment
outcomes.  In particular, the construction seed owns the candidate-cell
selection and the two held-out seeds are not opened before that lock exists.
"""

from __future__ import annotations


SCHEMA_VERSION = "pldr-observable-margin-protocol-v1"
CAMPAIGN_ID = "observable-margin-confirmation"
STATUS = "WAITING_FOR_FRESH_CHECKPOINTS"

ARCHITECTURE = {
    "parameter_count": 20_622_914,
    "decoder_layers": 3,
    "heads": 4,
    "model_width": 256,
    "row_width": 64,
    "row_hidden_width": 170,
    "row_residual_units": 8,
    "row_glu_blocks_per_unit": 2,
    "context_length": 256,
    "batch_size": 32,
    "optimizer": "adamw",
    "learning_rate": 7.5e-4,
    "warmup_updates": 250,
    "constant_rate_after_warmup": True,
    "weight_decay": 0.1,
    "gradient_value_clip": 1.0,
    "adam_betas": [0.9, 0.95],
    "adam_epsilon": 1e-5,
    "gate_dimension": 64,
    "training_instrumentation": {
        "probe_every": 1000000,
        "sharp_every": 1000000,
        "sharp_pre_every": 1000000,
        "sharp_full": 0,
        "sharp_block_every": 0,
        "validation_batches": 1,
        "purpose": (
            "Retain native training, complete checkpoints, validation logs, "
            "and resource telemetry while disabling unrelated legacy "
            "curvature probes during fresh trajectory construction."
        ),
    },
}

RUNS = (
    {
        "name": "om37-const-lr7.5e-4-s4444",
        "seed": 4444,
        "ownership": "construction",
        "device": "cuda:0",
    },
    {
        "name": "om37-const-lr7.5e-4-s5555",
        "seed": 5555,
        "ownership": "heldout",
        "device": "cuda:1",
    },
    {
        "name": "om37-const-lr7.5e-4-s6666",
        "seed": 6666,
        "ownership": "heldout",
        "device": "cuda:0",
    },
)

ANCHORS = {
    "complete_optimizer_steps": [2000, 4000, 8000, 16000, 24000],
    "construction_window_starts": [2000, 4000, 8000, 16000],
    "layers": [0, 1, 2],
    "candidate_block_lengths": [16, 64, 256],
    "checkpoint_pattern": "ckpt_{step}.pt",
    "required_checkpoint_state": [
        "model", "optimizer", "scheduler_clock", "torch_rng",
        "numpy_rng", "python_rng", "data_cursor", "registry",
        "code_manifest", "complete_state_digest",
    ],
    "selection_rule": (
        "Among algebraically qualified construction cells, maximize positive "
        "relative block margin; break ties by shorter block, earlier anchor, "
        "then lower layer. Lock exactly one cell before held-out evidence opens."
    ),
}

REGISTRY = {
    "schema_version": "pldr-observable-probe-registry-v1",
    "construction_contexts": 8,
    "heldout_contexts": 8,
    "context_head_block_rule": "head-equals-context-index-modulo-four-v1",
    "block_row_rule": "complete-ordered-generator-rows-0-through-63-v1",
    "graph_transfer": "single-union-ordinal-block-transfer-v1",
    "context_length": 256,
    "probe_location": "reserved-dataset-tail",
    "training_disjoint": True,
    "node_rule": (
        "Evaluate the identical hash-bound probe chunks at every parameter "
        "node. Actual sequential training minibatches alone drive updates."
    ),
    "non_telescoping_guard": (
        "An energy sequence may telescope only when every node carries the "
        "same probe_registry_sha256 and probe_chunk_ids_sha256. Changing "
        "actual update batches is expected and is recorded separately."
    ),
    "graph_rule": "symmetric-8nn-plus-euclidean-mst-uniform-v1",
    "neighbors": 8,
    "graph_scope": "one connected graph on eight ordered context-head blocks",
    "graph_weights": "uniform-1-over-edge-count",
    "graph_binding": (
        "Only context identities, head assignment, complete d-row order, "
        "ordinal role mapping, and the algorithm are prebound. The eight "
        "construction context-head blocks form one ordered 512-vertex union. "
        "Construction coordinates determine one connected edge/weight graph "
        "and exact resistance constant; that object is frozen and reused "
        "byte-for-byte on ordinal held-out block identities at every node."
    ),
    "domain_statement": (
        "The effective-resistance diameter bound controls only the finite "
        "registered union graph. It is not a continuum cover claim."
    ),
    "intervention_unit_rule": (
        "Twenty-four deterministic continuation sites, twelve per held-out "
        "seed, use distinct registered RNG substreams and nonoverlapping "
        "future update intervals. The three intervention comparisons reuse "
        "each site identity. Sites, steps, nodes, rows, and contexts are not "
        "randomized or treated as independent replicates."
    ),
    "intervention_steps": 4,
}

INTERVENTION_OPERATOR_CONTRACTS = (
    {
        "name": "aggregate_adaptive_gate_displacement_suppression",
        "operator_identity": (
            "post-native-selected-gate-aggregate-adaptive-displacement-"
            "suppression-v1"),
        "application_point": "after-native-optimizer-and-scheduler-step-v1",
        "selected_gate_moment_semantics": (
            "retain-native-selected-gate-adam-moments-and-evolve-on-"
            "modified-trajectory-v1"),
        "normalized_shape_reference_semantics": "not-applicable",
        "post_update_probe_semantics": "unclamped-fixed-registry-probe-v1",
    },
    {
        "name": "decay_source_removal",
        "operator_identity": "post-native-selected-gate-decay-removal-v1",
        "application_point": "after-native-optimizer-and-scheduler-step-v1",
        "selected_gate_moment_semantics": (
            "retain-native-selected-gate-adam-moments-and-evolve-on-"
            "modified-trajectory-v1"),
        "normalized_shape_reference_semantics": "not-applicable",
        "post_update_probe_semantics": "unclamped-fixed-registry-probe-v1",
    },
    {
        "name": "fixed_source_normalized_shape_activation_clamp",
        "operator_identity": (
            "selected-final-metric-layernorm-loss-forward-backward-"
            "activation-clamp-v1"),
        "application_point": (
            "inside-loss-forward-and-backward-before-native-step-v1"),
        "selected_gate_moment_semantics": (
            "native-adam-moments-advance-from-clamped-loss-gradient-v1"),
        "normalized_shape_reference_semantics": (
            "unadvanced-digest-bound-checkpoint-source-model-training-mode-"
            "same-minibatch-arm-rng-replay-detached-v1"),
        "post_update_probe_semantics": "unclamped-fixed-registry-probe-v1",
    },
)

INTERVENTION_COMPARISON_NAMES = tuple(
    row["name"] for row in INTERVENTION_OPERATOR_CONTRACTS)
INTERVENTION_OPERATOR_FIELDS = tuple(
    INTERVENTION_OPERATOR_CONTRACTS[0])
INTERVENTION_OPERATOR_BY_NAME = {
    row["name"]: row for row in INTERVENTION_OPERATOR_CONTRACTS}


ARITHMETIC = {
    "native_update": (
        "Training and continuation execute the registered native float32 "
        "AdamW update, including value clipping and decoupled weight decay."
    ),
    "deciding_shadow_dtype": "float64",
    "shadow_rule": (
        "Deciding gate, shape, bridge-sector, graph, and affine ledgers are "
        "recomputed in float64 from the executed native source and successor."
    ),
    "roundoff_charge": (
        "The residual between the float64 ideal update and the promoted "
        "executed native successor is measured as a nonnegative floating-"
        "point charge in every one-step and block bound."
    ),
    "float64_control": (
        "A separately labeled float64 continuation is a numerical control "
        "only and never replaces or enlarges the native-system verdict."
    ),
    "relative_identity_tolerance": 1e-8,
    "absolute_identity_tolerance": 1e-12,
    "clipping_boundary_tolerance": 1e-10,
    "native_moment_transition_relative_tolerance": 5e-7,
}

BRIDGE_SECTORS = (
    "fisher", "signed_jet", "force", "nonlinear", "clip",
    "preconditioner", "lag",
)

OBSERVABLE_CONTRACT = {
    "energy": "E = sum_j gamma_j^2 q_j on one fixed connected union graph",
    "shape_secant": (
        "q_next = q + linear_q + quadratic_q exactly; the signed linear "
        "shape gain and nonnegative quadratic shape charge are separate."
    ),
    "source_margin": (
        "Executed decay dissipation, signed Fisher and signed-jet gains, and "
        "signed-shape gain, minus the absolute force, nonlinear, clip, "
        "preconditioner, and lag projections and the finite-step, quadratic-"
        "shape, and measured floating-point charges. The aggregate adaptive "
        "gain is retained only as an exact diagnostic."
    ),
    "decay_mask": (
        "Record the binary gate-coordinate decay mask and the optimizer's "
        "effective decay coefficient at every update; replay both before "
        "using the decay dissipation."
    ),
    "bridge": (
        "Each of the seven disaggregated bridge-sector direction vectors is "
        "stored. Their sum must replay the executed adaptive direction before "
        "sector energy gains are used."
    ),
    "block": "Ordered nonautonomous affine convolution, not a mean factor",
    "attribution": ["gate-first", "symmetric-shapley"],
    "plga": (
        "Positive occupied bases, signed real exponents, exact divided "
        "differences, and derivative extrema on the occupied segment only."
    ),
    "raw_node_policy": "forbidden",
    "compact_node_arrays": [
        "gamma", "q", "linear_q", "quadratic_q",
        "normalized_edge_contrast", "adaptive_direction",
        "bridge_sector_direction", "gate_decay_mask",
        "optimizer_gate_weight_decay", "roundoff_charge", "energy",
    ],
    "single_union_sufficient_arrays": [
        "graph_edges", "graph_weights", "block_vertex_offsets",
        "block_head_index", "normalized_edge_contrast",
    ],
}

TRANSFER = {
    "construction_safety_multiplier": 1.25,
    "heldout_enlargement_allowed": False,
    "numerical_margin_floor_multiplier": 1.25,
    "diameter_endpoint_fraction_of_construction_source": 0.75,
    "diameter_bound_to_observed_ratio_max": 4.0,
    "lower_bound_rule": (
        "The simultaneous held-out lower bound is the minimum relative block "
        "margin over both preregistered seeds on the locked union graph. "
        "It must be strictly positive; nodes and context-head blocks are not "
        "resampled or treated as independent replicates."
    ),
    "frozen_arrays": [
        "affine_factor_upper_by_step", "affine_forcing_upper_by_step",
        "roundoff_charge_upper_by_step", "effective_resistance_upper",
    ],
    "frozen_scalars": [
        "effective_resistance_upper", "source_energy_upper",
        "source_diameter_bound_upper",
    ],
}


STATISTICS = {
    "trajectory_estimand": (
        "The registered probe population is fixed. Nodes, coordinates, "
        "contexts, and optimizer steps are not treated as independent "
        "replicates."
    ),
    "construction_seed": 4444,
    "heldout_seeds": [5555, 6666],
    "heldout_rule": (
        "Both held-out seeds must pass algebraic replay, graph enclosure, "
        "positive exact block margin after all charges, and endpoint "
        "contraction for the unchanged locked cell."
    ),
    "intervention_sites": 24,
    "sites_per_heldout_seed": 12,
    "intervention_comparisons": 3,
    "positive_sites_required_per_seed": 9,
    "lower_effect_order_statistic_per_seed": 4,
    "deterministic_criterion": (
        "For each held-out seed separately, at least nine of twelve fixed "
        "continuation sites must have the preregistered directional sign, "
        "and the fourth ascending directional effect must meet the frozen "
        "positive floor. This is deterministic seed-stratified coverage, "
        "not an independent-replicate significance test."
    ),
    "stopping": (
        "No optional stopping or sample-size adaptation. Failed or exhausted "
        "jobs remain in the append-only attempt ledger."
    ),
}

RESOURCE_CAPS = {
    "registered_devices": ["cuda:0", "cuda:1"],
    "peak_gpu_allocated_bytes_per_job": 23 * 1024 ** 3,
    "peak_host_rss_bytes_per_job": 64 * 1024 ** 3,
    "compact_output_bytes_per_job": 4 * 1024 ** 3,
    "aggregate_output_bytes": 32 * 1024 ** 3,
    "planned_stage_gpu_hours": {
        "R": 0.05, "T": 6.0, "B": 10.0, "H": 8.0,
        "ROB": 2.0, "M": 10.0, "I": 8.0, "F": 0.2,
    },
    "planned_gpu_hours": 44.25,
    "hard_gpu_hours": 48.0,
    "hard_wall_clock_hours": 80.0,
    "derivative_microbatch_contexts": 8,
    "simultaneous_derivative_directions": 1,
    "raw_nodes_allowed": False,
    "usage_carries_across_every_stage": True,
}


def _stage(stage_id, title, dependencies, checks, hard_gate=True):
    return {
        "id": stage_id,
        "title": title,
        "dependencies": list(dependencies),
        "hard_gate": bool(hard_gate),
        "gpu_hour_cap": RESOURCE_CAPS["planned_stage_gpu_hours"][stage_id],
        "required_checks": list(checks),
        "record_key": "STAGE:job_id",
        "attempt_rule": "append-only-positive-integer-attempts",
    }


STAGES = (
    _stage("R", "Registry, arithmetic, and resource preflight", (), (
        "fresh_run_names", "dataset_and_tokenizer_digests",
        "fixed_probe_registry", "training_probe_disjointness",
        "native_update_shadow_arithmetic", "registered_devices",
        "compact_evidence_only", "resource_meter_ready",
    )),
    _stage("T", "Fresh exact-state trajectories", ("R",), (
        "three_fresh_seeds", "native_float32_adamw", "actual_minibatches",
        "five_complete_optimizer_anchors", "exact_restart_replay",
        "data_cursor_and_rng_state", "checkpoint_registry_binding",
    )),
    _stage("B", "Construction baseline and candidate lock", ("T",), (
        "construction_only", "fixed_probe_at_every_node",
        "symmetric_8nn_mst_union_graph", "effective_resistance_enclosure",
        "exact_gate_shape_secants", "native_roundoff_charge",
        "ordered_affine_convolution", "one_locked_candidate_cell",
    )),
    _stage("H", "Executed loss-bridge sectors", ("B",), (
        "fisher", "signed_jet", "force", "nonlinear", "clip",
        "preconditioner", "lag", "sector_vector_sum",
        "actual_adaptive_direction_replay", "float64_shadow",
    )),
    _stage("ROB", "Coordinate robustness diagnostic", ("H",), (
        "full_3d_native_block", "scaled_induced_norm", "spectral_radius",
        "endpoint_conditioning", "clipping_and_dissipation",
        "diagnostic_not_scalar_margin_gate",
        "terminal_record_required_but_pass_not_required_by_M",
    ), hard_gate=False),
    _stage("M", "Held-out observable margin", ("H", "ROB"), (
        "construction_lock_precedes_opening", "both_heldout_seeds",
        "unchanged_candidate_cell", "fixed_probe_at_every_node",
        "positive_charged_margin", "endpoint_contraction",
        "graph_enclosure", "positive_base_real_exponent_plga",
    )),
    _stage("I", "Paired causal interventions", ("M",), (
        "twenty_four_preregistered_sites", "three_comparisons",
        "common_native_source", "common_actual_minibatch_sequence",
        "registered_intervention_execution_evidence",
        "seed_stratified_sign_coverage",
        "seed_stratified_effect_floor",
    )),
    _stage("F", "Fresh-process reconstruction and release", ("I",), (
        "fresh_process_reanalysis", "unique_stage_job_record_keys",
        "append_only_attempts", "schema_and_digest_replay",
        "resource_caps", "mutation_tests", "no_raw_nodes",
        "final_checksum_manifest",
    )),
)

STAGE_BY_ID = {stage["id"]: stage for stage in STAGES}

VALIDATION_FIXTURES = {
    "V5_scalar_order_sensitive_forcing": {
        "initial_energy": 0.0,
        "factors": [0.5, 0.25],
        "forcing": [1.0, 0.0],
        "endpoint": 0.25,
        "permuted_forcing": [0.0, 1.0],
        "permuted_endpoint": 1.0,
        "purpose": (
            "Detect replacement of the chronological scalar forcing "
            "convolution by an order-insensitive sum or mean."
        ),
    },
    "V7_lifted_noncommuting": {
        "initial": [1.0, -1.0],
        "operators": [
            [[1.0, 1.0], [0.0, 1.0]],
            [[1.0, 0.0], [1.0, 1.0]],
        ],
        "forcing": [[0.25, 0.0], [0.0, 0.5]],
        "purpose": (
            "Test chronological products in the lifted recursion. "
            "Noncommutativity is not a scalar V5 claim."
        ),
    },
}


STOP_GATES = (
    "missing_or_incomplete_anchor_checkpoint",
    "checkpoint_or_registry_digest_mismatch",
    "probe_registry_changes_between_nodes",
    "training_range_overlaps_probe_registry",
    "native_successor_or_bridge_sector_replay_failure",
    "nonfinite_deciding_array",
    "algebraic_residual_above_registered_tolerance",
    "graph_effective_resistance_enclosure_failure",
    "construction_lock_missing_before_heldout_open",
    "resource_cap_exceeded",
    "raw_node_artifact_detected",
)
