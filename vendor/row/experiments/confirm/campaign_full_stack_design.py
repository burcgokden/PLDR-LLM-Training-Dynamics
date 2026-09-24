"""Frozen primitive rules for the complete-stack confirmation campaign."""

from __future__ import annotations


CERTIFICATE_RULES = {
    "row_domain": {
        "construction_partition": "frozen_rows_only",
        "validation_partition": "disjoint_rows_never_used_for_construction",
        "primary_domain": "registered_pairwise_segment_hull",
        "membership_test": "exact_rational_union_of_boxes",
        "cover_remainder": (
            "validated_second_derivative_modulus_times_exact_cell_radius"
        ),
        "utility_test": "cover_remainder_plus_reserved_tail_below_criterion",
    },
    "complete_stack": {
        "coordinate": "direct_sum_all_layers_all_cells_vec_full_jacobian",
        "within_matrix_order": "output_major_input_minor",
        "criterion_layers": "every_layer_zero_through_depth_minus_one",
        "scalar_projection_role": "diagnostic_only",
    },
    "normal_closure": {
        "stack_derivative": "B_equals_D_theta_Psi",
        "normal_velocity_derivative": "DQ_equals_D_theta_Q",
        "right_inverse": "frozen_pseudoinverse_R_with_norm_BR_minus_I_below_one",
        "normal_operator": "K_equals_DQ_times_R",
        "closure_complement": "norm_DQ_minus_KB_carried_as_a_defect",
        "runtime_roundoff": (
            "right_inverse_closure_center_and_derivative_errors_charged"
        ),
        "reference_forcing": "Q_bar_minus_K_Psi_bar_frozen_once",
        "validation": (
            "center_first_derivative_second_derivative_residual_envelopes_"
            "without_refit"
        ),
    },
    "actual_program_state": {
        "coordinates": [
            "parameters", "first_moments", "second_moments",
            "optimizer_clock", "scheduler_phase", "learning_rate",
            "row_cover_state", "intervention_state",
        ],
        "successor": "implemented_branch_resolved_clipped_adamw_order",
        "cell_rule": "split_every_clipping_or_square_root_branch_crossing",
        "positive_map": (
            "derived_from_actual_center_image_interval_derivatives_and_"
            "exogenous_cell_radii"
        ),
        "caller_selected_radii": False,
    },
    "runtime_bridge": {
        "primitive_enclosure": "nextafter_outward_exact_binary_rational",
        "error_sources": [
            "kernel_rounding", "reduction_order", "reconstruction",
            "serialization",
        ],
        "operator_charge": "added_to_every_ordered_block_gain",
    },
    "complete_block_family": {
        "blocks": "all_lifted_stack_and_program_state_blocks",
        "cross_blocks": "every_ordered_source_target_pair_required",
        "metric_gain": "source_to_target_quadratic_metric_operator_norm",
        "path_length": 4,
        "path_rule": "all_admissible_graph_paths_in_temporal_order",
        "path_envelope": "entrywise_maximum_of_composed_positive_affine_maps",
        "tail_witness": "P_v_componentwise_below_kappa_v_with_kappa_below_one",
    },
    "source_ledger": {
        "rule": "primitive_interval_derivative_and_remainder_bounds_only",
        "trace_fit_allowed": False,
        "geometric_rate_fit_allowed": False,
        "terminal_value_as_persistent_source_allowed": False,
    },
    "time_semantics": {
        "allowed": [
            "FINITE_IMPLEMENTED_SCHEDULE",
            "FROZEN_CHECKPOINT",
            "CONTROLLED_INFINITE_TAIL",
        ],
        "finite": "claim_stops_at_last_registered_update",
        "frozen": "evaluation_only_no_optimizer_updates_identity_persistence",
        "controlled_tail": "total_successor_graph_and_positive_witness_required",
        "sentinel_observations_allowed": False,
    },
    "entry": {
        "criterion": 1.0,
        "criterion_unit": "output_row_unit_per_input_row_unit",
        "certificate_checkpoints": list(range(8000, 16000, 1000)),
        "terminal_step": 16000,
        "strict_floor_margin_required": True,
    },
}
