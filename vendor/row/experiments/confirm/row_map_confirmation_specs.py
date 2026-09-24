"""Registered constants for the archive-bound row-map confirmation."""

from __future__ import annotations


SCHEMA_VERSION = "pldr-row-map-collapse-protocol-v1"
STATUS = "REGISTERED_PENDING_EXECUTION"

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
    "lifted_gate_dimension": 192,
}

RUNS = (
    {"name": "w8-const-lr7.5e-4", "seed": 1234, "ownership": "construction"},
    {"name": "w8-const-lr7.5e-4-s2", "seed": 2222, "ownership": "validation"},
    {"name": "w8-const-lr7.5e-4-s3", "seed": 3333, "ownership": "validation"},
)

CHECKPOINTS = {
    "geometry_steps": [1000, 2000, 4000, 8000, 16000, 24000],
    "loss_sector_steps": [4000, 8000, 16000],
    "complete_optimizer_steps": [24000],
    "checkpoint_file_by_step": {
        1000: "ckpt_1000.pt",
        2000: "ckpt_2000.pt",
        4000: "ckpt_4000.pt",
        8000: "ckpt_8000.pt",
        16000: "ckpt_16000.pt",
        24000: "ckpt_final.pt",
    },
    "optimizer_state_rule": (
        "The retained intermediate checkpoints contain model state only. "
        "Complete-successor dynamics and paired continuations therefore "
        "start at retained 24000-update states; no moments are imputed."
    ),
}

FIXED_CONSTANTS = {
    "construction_contexts": 8,
    "validation_contexts": 16,
    "continuation_updates": 16,
    "material_energy_factor": 3.0,
    "construction_to_validation_factor": 2.0,
    "route_dominance_fraction": 0.625,
    "relative_identity_tolerance": 1e-8,
    "absolute_zero_tolerance": 1e-12,
    "clipping_boundary_tolerance": 1e-10,
    "trajectory_gradient_replay_tolerance": 1e-5,
    "plga_margin_rule": "sqrt(2)*B < Delta",
    "empty_qualified_set_supports_margin": False,
    "adjust_factors_after_validation": False,
}

RESOURCE_CAPS = {
    "registered_devices": ["cuda:0", "cuda:1"],
    "gpu_bytes_each": 24 * 1024 ** 3,
    "planned_gpu_hours": 2.05,
    "hard_gpu_hours": 2.4,
    "hard_wall_clock_hours": 2.0,
    "host_bytes": 125 * 1024 ** 3,
    "carry_base_usage_into_intervention_plan": True,
}


INTERVENTION_SELECTION = {
    "construction_seed": 1234,
    "primary_rule": (
        "Select every layer whose minimum construction interval energy ratio "
        "is at most 1/3."
    ),
    "fallback_rule": (
        "If no layer meets the primary rule, select the unique minimum-ratio "
        "layer with lower-index tie breaking."
    ),
    "validation_blinding": (
        "The selection consumes construction intervals only and is sealed "
        "before intervention observables are opened."
    ),
    "required_provenance": [
        "base_plan_sha256", "geometry_analysis_sha256", "selection_sha256",
    ],
}


def _stage(stage_id, title, dependencies, gpu_hours, checks):
    return {
        "id": stage_id,
        "title": title,
        "dependencies": list(dependencies),
        "gpu_hour_cap": float(gpu_hours),
        "required_checks": list(checks),
        "evidence_rule": (
            "Deciding arrays are recomputed from digest-bound checkpoint "
            "tensors and registered contexts; log summaries are cross-checks."
        ),
    }


STAGES = (
    _stage("Q", "Real two-device qualification", (), 0.05, (
        "real_20m_checkpoint", "real_cross_entropy", "nonzero_signed_jet",
        "real_adamw_moments", "complete_successor_autodiff",
        "relative_block_residual_at_most_1e-8", "both_devices_agree",
    )),
    _stage("G", "Graph and route geometry", ("Q",), 0.25, (
        "three_seeds_six_nodes_three_layers", "exact_layernorm_factorization",
        "exact_energy_identity", "connected_graph_bound",
        "covered_domain_bound", "gate_shape_log_routes", "zero_policy",
    )),
    _stage("H", "True loss sectors", ("Q",), 0.45, (
        "three_anchor_steps", "eight_streamed_construction_contexts",
        "sixteen_disjoint_validation_contexts", "fisher_psd",
        "signed_second_jet", "true_hessian_reconstruction",
        "force_at_zero", "context_mean_dispersion_and_count",
    )),
    _stage("D", "Exact retained-state dynamics", ("G", "H"), 0.85, (
        "retained_24000_optimizer_state_only", "sixteen_update_windows",
        "energy_every_node", "actual_update_shape_jvp", "segment_remainder",
        "analytic_192_block", "complete_successor_autodiff",
        "sixteen_true_hessian_blocks_per_window",
        "scaled_chronological_products", "no_raw_norm_contraction_verdict",
    )),
    _stage("I", "Paired causal arms", ("D",), 0.35, (
        "common_complete_source_state", "common_minibatch_sequence",
        "baseline_continue", "freeze_final_gate", "disable_final_gate_decay",
        "freeze_normalized_shape", "replay_physical_rows",
        "gate_and_shape_signal_retention",
    )),
    _stage("A", "Signed-power PLGA transfer", ("G", "H"), 0.10, (
        "strictly_positive_power_bases", "signed_real_exponents",
        "exact_divided_difference_reconstruction",
        "nonempty_heldout_margin_set",
        "sharp_margin_test", "fixed_factor_2_validation_transfer",
    )),
    _stage("R", "Fresh reconstruction and release gate", ("D", "I", "A"), 0.0, (
        "fresh_process_recomputation", "schema_validation", "mutation_suite",
        "resource_caps", "deterministic_records", "checksum_verification",
        "python_tests", "lean_build", "axiom_check", "tex_build",
    )),
)

CONFIRMATION_STATEMENTS = (
    {
        "id": "C1", "name": "row_map_collapse_and_graph_control",
        "rule": (
            "Report every seed, interval, and layer. A collapse cell has "
            "E_T/E_0 <= 1/3. Each validation seed must contain a collapse "
            "cell. Graph and covered-domain bounds must enclose measurements."
        ),
    },
    {
        "id": "C2", "name": "exact_route_reconstruction",
        "rule": (
            "Gate plus shape log work, direct energy, fixed-Q gate work, "
            "shape flux, and zero cases replay within 1e-8 relative error."
        ),
    },
    {
        "id": "C3", "name": "gate_mechanism",
        "rule": (
            "Decay plus loss memory reconstructs each gate step. The decay-off "
            "arm removes only direct gate decay; gate freeze removes gate "
            "work while retaining measurable shape work."
        ),
    },
    {
        "id": "C4", "name": "shape_mechanism",
        "rule": (
            "Actual-update shape JVP plus measured remainder reconstructs the "
            "increment, transfers within factor 2, and the shape clamp removes "
            "shape work while retaining measurable gate work."
        ),
    },
    {
        "id": "C5", "name": "mixed_mechanism",
        "rule": (
            "Route logs add exactly. A mechanism label requires material total "
            "log work of at least log(3) and a registered dominance fraction."
        ),
    },
    {
        "id": "C6", "name": "complete_successor_derivative",
        "rule": (
            "The analytic 192 by 192 block matches float64 autodiff of the "
            "actual-minibatch successor at each retained-state anchor within "
            "relative residual 1e-8; every continuation block is freshly "
            "reconstructed before chronological composition."
        ),
    },
    {
        "id": "C7", "name": "coordinate_aware_robustness",
        "rule": (
            "Report spectral radius, scaled induced norm, endpoint scaling "
            "condition number, gate block, moments, clipping, and dissipation."
        ),
    },
    {
        "id": "C8", "name": "plga_transfer",
        "rule": (
            "Signed-power secants reconstruct every registered difference. A "
            "margin needs a nonempty held-out set, sqrt(2)B < Delta, a "
            "construction-owned constant, and factor-2 held-out transfer."
        ),
    },
)

INTERVENTIONS = {
    "baseline_continue": "Unmodified AdamW continuation for sixteen updates.",
    "freeze_final_gate": (
        "Restore gamma and its first and second moments after every successor."
    ),
    "disable_final_gate_decay": (
        "Remove only decoupled weight decay from the selected gamma entries."
    ),
    "freeze_normalized_shape": (
        "Replay source normalized rows at the final LayerNorm interface and "
        "block their upstream gradient while retaining gate updates."
    ),
    "replay_physical_rows": (
        "Use the identical registered rows as a paired-data control."
    ),
}
