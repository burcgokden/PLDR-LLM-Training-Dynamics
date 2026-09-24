"""Deterministic analysis for the prospective direct-certificate E0-E8 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS / "confirm"))

import campaign_record as records  # noqa: E402
from confirmation_decision import (  # noqa: E402
    base_result,
    error_result,
    paired_sign_randomization,
)


def _values(protocol_records, name):
    result = []
    unavailable = []
    for record in protocol_records:
        row = records.measurement_map(record)[name]
        if row["status"] != "OBSERVED":
            unavailable.append({
                "run_id": record["run_id"],
                "status": row["status"],
                "reason_code": row["reason_code"],
            })
        else:
            result.append(float(row["value"]))
    return np.asarray(result, dtype=float), unavailable


def _paired_pass(result, *, alpha, threshold, direction):
    if result["one_sided_exact_p"] > alpha:
        return False
    if direction == "lower":
        return result["median_contrast"] < threshold
    if direction == "upper":
        return result["median_contrast"] > threshold
    raise ValueError("direction must be lower or upper")


def _base(pid, rows, registration):
    return base_result(pid, rows, registration, records)


def _finalize(result, criteria, summary):
    result["criteria"] = {
        name: bool(value) for name, value in criteria.items()
    }
    result["summary"] = summary
    if result["status"] == "READY":
        result["status"] = (
            "CONFIRMED" if all(criteria.values()) else "NOT_CONFIRMED"
        )
    return result


def _e0(rows, reg):
    result = _base("E0", rows, reg)
    if result["status"] != "READY":
        return result
    identity, _ = _values(rows, "cover_identity_error")
    heldout, _ = _values(rows, "heldout_jacobian_ratio")
    cover_distance, _ = _values(rows, "heldout_cover_radius_ratio")
    membership, _ = _values(rows, "validation_domain_membership_indicator")
    utility, _ = _values(rows, "cover_utility_margin")
    mixed, _ = _values(rows, "mixed_derivative_enclosure_ratio")
    third, _ = _values(rows, "third_derivative_enclosure_ratio")
    direct, _ = _values(rows, "direct_row_map_upper")
    grid, _ = _values(rows, "grid_jacobian_upper")
    cover, _ = _values(rows, "cover_remainder")
    numerical, _ = _values(rows, "numerical_error")
    threshold = reg["thresholds"]
    return _finalize(result, {
        "heldout_rows": float(np.max(heldout))
        <= threshold["heldout_jacobian_ratio_max"],
        "validation_domain_membership": float(np.min(membership))
        >= threshold["validation_domain_membership_indicator_min"],
        "cover_utility": float(np.min(utility))
        > threshold["cover_utility_margin_min"],
        "mixed_derivative": float(np.max(mixed))
        <= threshold["mixed_derivative_enclosure_ratio_max"],
        "third_derivative": float(np.max(third))
        <= threshold["third_derivative_enclosure_ratio_max"],
    }, {
        "maximum_cover_identity_error": float(np.max(np.abs(identity))),
        "maximum_heldout_jacobian_ratio": float(np.max(heldout)),
        "maximum_heldout_cover_radius_ratio":
            float(np.max(cover_distance)),
        "minimum_validation_domain_membership_indicator": float(np.min(membership)),
        "minimum_cover_utility_margin": float(np.min(utility)),
        "maximum_mixed_derivative_enclosure_ratio": float(np.max(mixed)),
        "maximum_third_derivative_enclosure_ratio": float(np.max(third)),
        "maximum_direct_upper": float(np.max(direct)),
        "maximum_grid_upper": float(np.max(grid)),
        "maximum_cover_remainder": float(np.max(cover)),
        "maximum_numerical_error": float(np.max(numerical)),
    })


def _e1(rows, reg):
    result = _base("E1", rows, reg)
    if result["status"] != "READY":
        return result
    names = (
        "adamw_ledger_residual",
        "lifted_recurrence_residual",
        "defect_recomposition_residual",
        "coordinate_motion_enclosure_ratio",
        "taylor_remainder_enclosure_ratio",
        "intervention_boundary_residual",
    )
    value = {name: _values(rows, name)[0] for name in names}
    threshold = reg["thresholds"]
    return _finalize(result, {
        "adamw_replay": float(np.max(np.abs(
            value["adamw_ledger_residual"])))
        <= threshold["adamw_ledger_residual_max"],
        "lifted_identity": float(np.max(np.abs(
            value["lifted_recurrence_residual"])))
        <= threshold["lifted_recurrence_residual_max"],
        "defect_recomposition": float(np.max(np.abs(
            value["defect_recomposition_residual"])))
        <= threshold["defect_recomposition_residual_max"],
        "intervention_boundary": float(np.max(np.abs(
            value["intervention_boundary_residual"])))
        <= threshold["intervention_boundary_residual_max"],
        "coordinate_motion_enclosure": float(np.max(
            value["coordinate_motion_enclosure_ratio"]))
        <= threshold["coordinate_motion_enclosure_ratio_max"],
        "taylor_enclosure": float(np.max(
            value["taylor_remainder_enclosure_ratio"]))
        <= threshold["taylor_remainder_enclosure_ratio_max"],
    }, {
        "maximum_values": {
            name: float(np.max(np.abs(rows_value)))
            for name, rows_value in value.items()
        },
    })


def _e2(rows, reg):
    result = _base("E2", rows, reg)
    if result["status"] != "READY":
        return result
    lower, _ = _values(rows, "lower_normal_edge")
    upper, _ = _values(rows, "upper_normal_edge")
    jury, _ = _values(rows, "jury_margin_min")
    self_adjoint, _ = _values(
        rows, "normal_operator_self_adjoint_residual")
    linearization, _ = _values(
        rows, "normal_linearization_residual_ratio")
    nonlinear, _ = _values(rows, "nonlinear_residual_enclosure_ratio")
    identity, _ = _values(rows, "normal_operator_identity_residual")
    lower_margin, _ = _values(rows, "jury_lower_margin")
    upper_margin, _ = _values(rows, "jury_upper_margin")
    loading, _ = _values(rows, "spectral_loading")
    threshold = reg["thresholds"]
    return _finalize(result, {
        "positive_lower_edge": float(np.min(lower))
        > threshold["lower_normal_edge_min"],
        "strict_jury": float(np.min(jury)) > threshold["jury_margin_min"],
        "strict_schedule_lower_margin": float(np.min(lower_margin))
        > threshold["jury_lower_margin_min"],
        "strict_schedule_upper_margin": float(np.min(upper_margin))
        > threshold["jury_upper_margin_min"],
        "subcritical_spectral_loading": float(np.max(loading))
        < threshold["spectral_loading_max"],
        "self_adjoint_operator": float(np.max(np.abs(self_adjoint)))
        <= threshold["normal_operator_self_adjoint_residual_max"],
        "normal_identity": float(np.max(np.abs(identity)))
        <= threshold["normal_operator_identity_residual_max"],
        "linearization_enclosure": float(np.max(linearization))
        <= threshold["normal_linearization_residual_ratio_max"],
        "nonlinear_enclosure": float(np.max(nonlinear))
        <= threshold["nonlinear_residual_enclosure_ratio_max"],
    }, {
        "normal_spectral_band": [
            float(np.min(lower)), float(np.max(upper))
        ],
        "minimum_jury_margin": float(np.min(jury)),
        "minimum_schedule_lower_margin": float(np.min(lower_margin)),
        "minimum_schedule_upper_margin": float(np.min(upper_margin)),
        "maximum_spectral_loading": float(np.max(loading)),
        "maximum_self_adjoint_residual":
            float(np.max(np.abs(self_adjoint))),
        "maximum_normal_identity_residual": float(np.max(np.abs(identity))),
        "maximum_normal_linearization_ratio":
            float(np.max(linearization)),
        "maximum_nonlinear_ratio": float(np.max(nonlinear)),
    })


def _e3(rows, reg):
    result = _base("E3", rows, reg)
    if result["status"] != "READY":
        return result
    residual, _ = _values(rows, "nominal_lyapunov_residual")
    metric_min, _ = _values(rows, "metric_eigenvalue_min")
    nominal, _ = _values(rows, "nominal_q")
    structured, _ = _values(rows, "structured_family_q")
    slack, _ = _values(rows, "structured_family_slack")
    corner, _ = _values(rows, "maximum_corner_h_gain")
    reconstruction, _ = _values(
        rows, "multiaffine_reconstruction_residual")
    enclosure, _ = _values(rows, "primitive_box_enclosure_ratio")
    window, _ = _values(rows, "path_window_gain")
    observed, _ = _values(rows, "observed_h_gain")
    slack_identity_error = float(
        np.max(np.abs(slack - (1.0 - structured))))
    corner_excess = float(np.max(corner - structured))
    gain_excess = float(np.max(observed - structured))
    threshold = reg["thresholds"]
    return _finalize(result, {
        "lyapunov_identity": float(np.max(np.abs(residual)))
        <= threshold["nominal_lyapunov_residual_max"],
        "positive_metric": float(np.min(metric_min))
        > threshold["metric_eigenvalue_min"],
        "structured_contraction": float(np.max(structured))
        < threshold["structured_family_q_max"],
        "structured_slack": (
            float(np.min(slack)) > threshold["structured_family_slack_min"]
            and slack_identity_error
            <= threshold["structured_slack_identity_error_max"]
        ),
        "corner_lmi_diagnostic": corner_excess
        <= threshold["corner_gain_excess_max"],
        "multiaffine_identity": float(np.max(np.abs(reconstruction)))
        <= threshold["multiaffine_reconstruction_residual_max"],
        "primitive_box": float(np.max(enclosure))
        <= threshold["primitive_box_enclosure_ratio_max"],
        "path_window_contraction": float(np.max(window))
        < threshold["path_window_gain_max"],
        "observed_gain": gain_excess
        <= threshold["observed_gain_excess_max"],
    }, {
        "maximum_lyapunov_residual": float(np.max(np.abs(residual))),
        "minimum_metric_eigenvalue": float(np.min(metric_min)),
        "maximum_nominal_q": float(np.max(nominal)),
        "maximum_structured_family_q": float(np.max(structured)),
        "minimum_structured_family_slack": float(np.min(slack)),
        "maximum_structured_slack_identity_error": slack_identity_error,
        "maximum_corner_gain_excess": corner_excess,
        "maximum_multiaffine_reconstruction_residual":
            float(np.max(np.abs(reconstruction))),
        "maximum_primitive_box_enclosure_ratio":
            float(np.max(enclosure)),
        "maximum_path_window_gain": float(np.max(window)),
        "maximum_observed_gain_excess": gain_excess,
    })


def _e4(rows, reg):
    result = _base("E4", rows, reg)
    if result["status"] != "READY":
        return result
    forcing, _ = _values(rows, "forcing_norm")
    defect, _ = _values(rows, "complete_defect_norm")
    disturbance, _ = _values(rows, "complete_disturbance_norm")
    persistent, _ = _values(rows, "persistent_disturbance_floor")
    envelope, _ = _values(rows, "disturbance_envelope")
    ratio, _ = _values(rows, "envelope_ratio")
    rate, _ = _values(rows, "geometric_rate")
    witness, _ = _values(rows, "positive_comparison_witness_ratio")
    residual, _ = _values(rows, "defect_recomposition_residual")
    if (persistent < 0.0).any() or (envelope < 0.0).any():
        result["status"] = "INCOMPLETE"
        result["reasons"].append("disturbance envelope must be nonnegative")
        return result
    threshold = reg["thresholds"]
    return _finalize(result, {
        "defect_recomposition": float(np.max(np.abs(residual)))
        <= threshold["defect_recomposition_residual_max"],
        "persistent_plus_geometric_envelope": float(np.max(ratio))
        <= threshold["envelope_ratio_max"],
        "geometric_rate": float(np.max(rate))
        < threshold["geometric_rate_max"],
        "positive_comparison_witness": float(np.max(witness))
        <= threshold["positive_comparison_witness_ratio_max"],
    }, {
        "maximum_forcing_norm": float(np.max(forcing)),
        "maximum_complete_defect_norm": float(np.max(defect)),
        "maximum_complete_disturbance_norm": float(np.max(disturbance)),
        "maximum_persistent_disturbance_floor": float(np.max(persistent)),
        "maximum_disturbance_envelope": float(np.max(envelope)),
        "maximum_envelope_ratio": float(np.max(ratio)),
        "maximum_geometric_rate": float(np.max(rate)),
        "maximum_positive_comparison_witness_ratio":
            float(np.max(witness)),
        "maximum_recomposition_residual": float(np.max(np.abs(residual))),
    })


def _e5(rows, reg):
    result = _base("E5", rows, reg)
    if result["status"] != "READY":
        return result
    initial, _ = _values(rows, "initial_membership_margin")
    lifted, _ = _values(rows, "lifted_budget_slack")
    window_length, _ = _values(rows, "path_window_length")
    window_gain, _ = _values(rows, "path_window_gain")
    block, _ = _values(rows, "block_lifted_budget_slack")
    prefix, _ = _values(rows, "prefix_budget_slack_min")
    auxiliary, _ = _values(rows, "auxiliary_budget_slack_min")
    heldout, _ = _values(rows, "heldout_image_margin")
    escapes, _ = _values(rows, "one_step_escape_count")
    enclosure, _ = _values(rows, "family_box_enclosure_ratio")
    threshold = reg["thresholds"]
    return _finalize(result, {
        "initial_membership": float(np.min(initial))
        >= threshold["initial_membership_margin_min"],
        "lifted_budget": float(np.min(lifted))
        >= threshold["lifted_budget_slack_min"],
        "registered_path_window": bool(
            np.all(window_length >= 1.0)
            and np.all(window_length == np.floor(window_length))
            and np.max(window_gain) < threshold["path_window_gain_max"]
        ),
        "block_lifted_budget": float(np.min(block))
        >= threshold["block_lifted_budget_slack_min"],
        "prefix_budgets": float(np.min(prefix))
        >= threshold["prefix_budget_slack_min"],
        "auxiliary_budgets": float(np.min(auxiliary))
        >= threshold["auxiliary_budget_slack_min"],
        "heldout_images": float(np.min(heldout))
        >= threshold["heldout_image_margin_min"],
        "zero_escapes": float(np.max(escapes))
        <= threshold["one_step_escape_count_max"],
        "family_box_enclosure": float(np.max(enclosure))
        <= threshold["family_box_enclosure_ratio_max"],
    }, {
        "minimum_initial_membership_margin": float(np.min(initial)),
        "minimum_lifted_budget_slack": float(np.min(lifted)),
        "registered_path_window_length": float(np.max(window_length)),
        "maximum_path_window_gain": float(np.max(window_gain)),
        "minimum_block_lifted_budget_slack": float(np.min(block)),
        "minimum_prefix_budget_slack": float(np.min(prefix)),
        "minimum_auxiliary_budget_slack": float(np.min(auxiliary)),
        "minimum_heldout_image_margin": float(np.min(heldout)),
        "total_one_step_escapes": float(np.sum(escapes)),
        "maximum_family_box_enclosure_ratio": float(np.max(enclosure)),
    })


def _e6(rows, reg):
    result = _base("E6", rows, reg)
    if result["status"] != "READY":
        return result
    predicted, _ = _values(rows, "predicted_entry_step")
    observed, _ = _values(rows, "observed_entry_step")
    error, _ = _values(rows, "entry_error")
    floor, _ = _values(rows, "residual_floor")
    direct, _ = _values(rows, "direct_row_map_upper")
    criterion, _ = _values(rows, "criterion")
    sustained, _ = _values(rows, "sustained_entry_indicator")
    window_length, _ = _values(rows, "path_window_length")
    window_gain, _ = _values(rows, "path_window_gain")
    block_upper, _ = _values(rows, "block_product_convolution_upper")
    schedule_upper, _ = _values(
        rows, "schedule_product_convolution_upper")
    schedule_predicted, _ = _values(
        rows, "schedule_predicted_entry_step")
    if (
        (predicted < 0.0).any()
        or (observed < 0.0).any()
        or (floor < 0.0).any()
        or (direct < 0.0).any()
        or (criterion <= 0.0).any()
        or (schedule_upper < 0.0).any()
        or (window_length < 1.0).any()
        or not np.all(window_length == np.floor(window_length))
        or (window_gain < 0.0).any()
        or (block_upper < 0.0).any()
        or (schedule_predicted < 0.0).any()
        or not np.isin(sustained, (0.0, 1.0)).all()
    ):
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            "E6 steps, norms, criteria, or indicators are outside their domain"
        )
        return result
    identity_error = float(np.max(np.abs(error - (observed - predicted))))
    coverage = float(np.mean(observed <= predicted))
    relative = np.abs(observed - predicted) / np.maximum(predicted, 1.0)
    threshold = reg["thresholds"]
    return _finalize(result, {
        "horizon_coverage": coverage >= threshold["coverage_min"],
        "entry_identity": identity_error <= 1e-10,
        "strict_floor_margin": float(np.min(criterion - floor))
        > threshold["floor_margin_min"],
        "contracting_path_windows": float(np.max(window_gain))
        < threshold["path_window_gain_max"],
        "sustained_entry": float(np.mean(sustained))
        >= threshold["sustained_entry_fraction_min"],
        "direct_criterion": bool(np.all(direct < criterion)),
    }, {
        "coverage": coverage,
        "median_absolute_relative_error": float(np.median(relative)),
        "maximum_entry_identity_error": identity_error,
        "minimum_floor_margin": float(np.min(criterion - floor)),
        "sustained_entry_fraction": float(np.mean(sustained)),
        "maximum_direct_to_criterion_ratio": float(np.max(direct / criterion)),
        "registered_path_window_length": float(np.max(window_length)),
        "maximum_path_window_gain": float(np.max(window_gain)),
        "maximum_block_product_convolution_upper":
            float(np.max(block_upper)),
        "maximum_schedule_product_convolution_upper":
            float(np.max(schedule_upper)),
        "schedule_horizon_coverage":
            float(np.mean(observed <= schedule_predicted)),
        "median_schedule_horizon_absolute_relative_error":
            float(np.median(
                np.abs(observed - schedule_predicted)
                / np.maximum(schedule_predicted, 1.0)
            )),
    })


def _arm_name(record):
    return record["intervention"]["arm"].lower().replace("-", "_")


def _matched(rows):
    groups = {}
    for row in rows:
        key = (row["seed"], row["clocks"]["global_optimizer_step"])
        arm = _arm_name(row)
        if arm in groups.setdefault(key, {}):
            raise ValueError(f"duplicate arm {arm} for matched unit {key}")
        groups[key][arm] = row
    return groups


def _matched_e8(rows):
    groups = {}
    for row in rows:
        key = (
            row["seed"],
            row["clocks"]["global_optimizer_step"],
            _direct_value(row, "model_width"),
            _direct_value(row, "model_depth"),
            _direct_value(row, "maximum_learning_rate"),
            _direct_value(row, "warmup_steps"),
            _direct_value(row, "anneal_floor"),
            _direct_value(row, "rg_block_size"),
        )
        arm = _arm_name(row)
        if arm in groups.setdefault(key, {}):
            raise ValueError(
                f"duplicate arm {arm} for E8 architecture unit {key}")
        groups[key][arm] = row
    return groups


def _direct_value(record, name):
    return float(
        records.measurement_map(record, require_observed=True)[name]["value"]
    )


def _e7(rows, reg):
    result = _base("E7", rows, reg)
    if result["status"] != "READY":
        return result
    contrasts = {
        "contraction_q": [],
        "contraction_horizon": [],
        "contraction_direct": [],
        "defect_envelope": [],
        "defect_horizon": [],
        "defect_direct": [],
        "contraction_upper_margin": [],
        "defect_upper_margin": [],
        "contraction_loading": [],
        "defect_loading": [],
        "contraction_product": [],
        "defect_product": [],
        "contraction_avalanche_rate": [],
        "defect_avalanche_rate": [],
        "contraction_avalanche_size": [],
        "defect_avalanche_size": [],
        "contraction_avalanche_support": [],
        "defect_avalanche_support": [],
    }
    loss_differences = []
    groups = _matched(rows)
    required_arms = set(reg["required_arms"])
    incomplete = [
        key for key, group in groups.items()
        if set(group) != required_arms
    ]
    if incomplete:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{len(incomplete)} E7 groups do not contain exactly the "
            "registered arms"
        )
        return result
    for group in groups.values():
        contraction = group["contraction_improving"]
        defect = group["defect_increasing"]
        sham = group["sham"]
        contrasts["contraction_q"].append(
            _direct_value(contraction, "robust_q")
            - _direct_value(sham, "robust_q")
        )
        contrasts["contraction_horizon"].append(
            _direct_value(contraction, "predicted_entry_step")
            - _direct_value(sham, "predicted_entry_step")
        )
        contrasts["contraction_direct"].append(
            _direct_value(contraction, "direct_row_map_upper")
            - _direct_value(sham, "direct_row_map_upper")
        )
        contrasts["defect_envelope"].append(
            _direct_value(defect, "complete_disturbance_envelope")
            - _direct_value(sham, "complete_disturbance_envelope")
        )
        contrasts["defect_horizon"].append(
            _direct_value(defect, "predicted_entry_step")
            - _direct_value(sham, "predicted_entry_step")
        )
        contrasts["defect_direct"].append(
            _direct_value(defect, "direct_row_map_upper")
            - _direct_value(sham, "direct_row_map_upper")
        )
        for prefix, branch in (
            ("contraction", contraction), ("defect", defect)
        ):
            contrasts[f"{prefix}_upper_margin"].append(
                _direct_value(branch, "jury_upper_margin")
                - _direct_value(sham, "jury_upper_margin")
            )
            contrasts[f"{prefix}_loading"].append(
                _direct_value(branch, "spectral_loading")
                - _direct_value(sham, "spectral_loading")
            )
            contrasts[f"{prefix}_product"].append(
                _direct_value(
                    branch, "schedule_product_convolution_upper")
                - _direct_value(
                    sham, "schedule_product_convolution_upper")
            )
            contrasts[f"{prefix}_avalanche_rate"].append(
                _direct_value(branch, "avalanche_event_rate")
                - _direct_value(sham, "avalanche_event_rate")
            )
            contrasts[f"{prefix}_avalanche_size"].append(
                _direct_value(branch, "avalanche_mean_size")
                - _direct_value(sham, "avalanche_mean_size")
            )
            contrasts[f"{prefix}_avalanche_support"].append(
                _direct_value(branch, "avalanche_mean_support")
                - _direct_value(sham, "avalanche_mean_support")
            )
        loss_differences.extend([
            _direct_value(contraction, "heldout_loss")
            - _direct_value(sham, "heldout_loss"),
            _direct_value(defect, "heldout_loss")
            - _direct_value(sham, "heldout_loss"),
        ])
    complete = len(contrasts["contraction_q"])
    minimum_sets = int(reg["sample_size"])
    if complete < minimum_sets:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{complete} complete branch sets are below "
            f"the registered minimum {minimum_sets}"
        )
        return result
    alpha = reg["familywise_alpha"] / reg["primary_tests"]
    threshold = reg["thresholds"]
    upper_names = (
        "contraction_q", "contraction_horizon", "contraction_direct"
    )
    lower_names = ("defect_envelope", "defect_horizon", "defect_direct")
    upper = {
        name: paired_sign_randomization(
            contrasts[name], direction="lower",
            null=threshold[f"{name}_contrast_max"],
        )
        for name in upper_names
    }
    lower = {
        name: paired_sign_randomization(
            contrasts[name], direction="upper",
            null=threshold[f"{name}_contrast_min"],
        )
        for name in lower_names
    }
    criteria = {
        **{
            f"{name}_direction": _paired_pass(
                upper[name], alpha=alpha,
                threshold=threshold[f"{name}_contrast_max"], direction="lower",
            )
            for name in upper_names
        },
        **{
            f"{name}_direction": _paired_pass(
                lower[name], alpha=alpha,
                threshold=threshold[f"{name}_contrast_min"], direction="upper",
            )
            for name in lower_names
        },
    }
    return _finalize(result, criteria, {
        "complete_branch_sets": complete,
        "contraction_randomization": upper,
        "defect_randomization": lower,
        "maximum_loss_difference": float(np.max(loss_differences)),
        "mean_schedule_and_avalanche_contrasts": {
            name: float(np.mean(values))
            for name, values in contrasts.items()
            if name not in upper_names and name not in lower_names
        },
    })


def _e8(rows, reg):
    result = _base("E8", rows, reg)
    if result["status"] != "READY":
        return result
    groups = _matched_e8(rows)
    score_by_baseline = {}
    loss_by_baseline = {}
    theorem_rows = []
    rg_theorem_rows = []
    required_arms = set(reg["required_arms"])
    baseline_arms = required_arms - {"theory"}
    if "theory" not in required_arms or not baseline_arms:
        raise ValueError("E8 registration must declare theory and baselines")
    registered_blocks = tuple(
        float(value) for value in reg["registered_block_sizes"])
    if (
        not registered_blocks
        or len(set(registered_blocks)) != len(registered_blocks)
        or not set(registered_blocks).issubset({2.0, 4.0, 8.0})
    ):
        raise ValueError(
            "E8 registered_block_sizes must be distinct values from 2, 4, 8")
    incomplete = [
        key for key, group in groups.items()
        if set(group) != required_arms
    ]
    if incomplete:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{len(incomplete)} E8 groups do not contain exactly the "
            "registered arms"
        )
        return result
    architecture_groups = {}
    for key, group in groups.items():
        architecture_key = key[:-1]
        block_size = key[-1]
        blocked = architecture_groups.setdefault(architecture_key, {})
        if block_size in blocked:
            raise ValueError(
                f"duplicate E8 block size {block_size} for "
                f"architecture unit {architecture_key}")
        blocked[block_size] = group
    incomplete_architectures = [
        key for key, blocked in architecture_groups.items()
        if set(blocked) != set(registered_blocks)
    ]
    if incomplete_architectures:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{len(incomplete_architectures)} E8 architecture units do "
            "not contain exactly the registered RG block sizes"
        )
        return result
    bad_indicators = 0
    for group in groups.values():
        if _direct_value(group["theory"], "baseline_indicator") != 0.0:
            bad_indicators += 1
        bad_indicators += sum(
            _direct_value(group[arm], "baseline_indicator") != 1.0
            for arm in baseline_arms
        )
    if bad_indicators:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{bad_indicators} E8 arm indicators disagree with registration"
        )
        return result
    non_rg_names = tuple(
        name for name in records.MEASUREMENT_NAMES["E8"]
        if not name.startswith("rg_")
    )
    inconsistent = 0
    reference_block = min(registered_blocks)
    for blocked in architecture_groups.values():
        reference = blocked[reference_block]
        for block_size, group in blocked.items():
            if block_size == reference_block:
                continue
            for arm in required_arms:
                for name in non_rg_names:
                    if not math.isclose(
                        _direct_value(group[arm], name),
                        _direct_value(reference[arm], name),
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    ):
                        inconsistent += 1
        group = reference
        theory = group["theory"]
        theorem_rows.append(theory)
        rg_theorem_rows.extend(
            blocked[block_size]["theory"]
            for block_size in registered_blocks
        )
        theory_score = _direct_value(theory, "model_score")
        theory_loss = _direct_value(theory, "heldout_loss")
        for arm in sorted(baseline_arms):
            baseline = group[arm]
            score_by_baseline.setdefault(arm, []).append(
                theory_score - _direct_value(baseline, "model_score")
            )
            loss_by_baseline.setdefault(arm, []).append(
                theory_loss - _direct_value(baseline, "heldout_loss")
            )
    if inconsistent:
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{inconsistent} non-RG E8 values disagree across block sizes"
        )
        return result
    minimum_groups = int(reg["sample_size"])
    if len(theorem_rows) < max(2, minimum_groups):
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            f"{len(theorem_rows)} complete held-out groups are below "
            f"the registered minimum {minimum_groups}"
        )
        return result
    alpha = (
        reg["familywise_alpha"]
        / max(reg["primary_tests"], len(score_by_baseline))
    )
    score_bounds = {}
    loss_bounds = {}
    for arm, differences in sorted(score_by_baseline.items()):
        if len(differences) < 2:
            result["status"] = "INCOMPLETE"
            result["reasons"].append(
                f"baseline {arm} has fewer than two matched pairs"
            )
            return result
        score_bounds[arm] = paired_sign_randomization(
            differences, direction="lower",
            null=reg["thresholds"]["paired_model_score_difference_max"],
        )
        loss_bounds[arm] = paired_sign_randomization(
            loss_by_baseline[arm], direction="lower", null=0.0,
        )
    bound_ratio = np.asarray([
        _direct_value(row, "theorem_bound_ratio") for row in theorem_rows
    ])
    direct = np.asarray([
        _direct_value(row, "direct_row_map_upper") for row in theorem_rows
    ])
    criterion = np.asarray([
        _direct_value(row, "criterion") for row in theorem_rows
    ])
    predicted = np.asarray([
        _direct_value(row, "predicted_entry_step") for row in theorem_rows
    ])
    observed = np.asarray([
        _direct_value(row, "observed_entry_step") for row in theorem_rows
    ])
    maximum_rate = np.asarray([
        _direct_value(row, "maximum_learning_rate")
        for row in theorem_rows
    ])
    warmup = np.asarray([
        _direct_value(row, "warmup_steps") for row in theorem_rows
    ])
    anneal_floor = np.asarray([
        _direct_value(row, "anneal_floor") for row in theorem_rows
    ])
    source_order = np.asarray([
        _direct_value(row, "source_order_parameter")
        for row in theorem_rows
    ])
    source_rmse = np.asarray([
        _direct_value(row, "source_rmse") for row in theorem_rows
    ])
    tensor_mean = np.asarray([
        _direct_value(row, "tensor_mean_abs") for row in theorem_rows
    ])
    row_spread = np.asarray([
        _direct_value(row, "row_input_spread") for row in theorem_rows
    ])
    downstream = np.asarray([
        _direct_value(row, "downstream_lipschitz")
        for row in theorem_rows
    ])
    order_upper = np.asarray([
        _direct_value(row, "order_parameter_upper")
        for row in theorem_rows
    ])
    bridge_ratio = np.asarray([
        _direct_value(row, "order_parameter_bridge_ratio")
        for row in theorem_rows
    ])
    rg_names = (
        "rg_block_size", "rg_density_identity_residual",
        "rg_layernorm_scale_residual", "rg_within_covariance_trace",
        "rg_deductive_closure_upper", "rg_deductive_closure_ratio",
        "rg_softmax_aggregation_residual",
        "rg_repeated_block_semigroup_residual",
        "rg_full_layer_closure_upper", "rg_full_layer_closure_ratio",
    )
    rg = {
        name: np.asarray([
            _direct_value(row, name) for row in rg_theorem_rows
        ])
        for name in rg_names
    }
    avalanche_names = (
        "avalanche_event_rate", "avalanche_mean_size",
        "avalanche_mean_duration", "avalanche_mean_support",
        "avalanche_tail_comparison", "finite_size_scaling_error",
    )
    avalanche = {
        name: np.asarray([
            _direct_value(row, name) for row in theorem_rows
        ])
        for name in avalanche_names
    }
    if (
        (direct < 0.0).any()
        or (criterion <= 0.0).any()
        or (predicted < 0.0).any()
        or (observed < 0.0).any()
        or (maximum_rate <= 0.0).any()
        or (warmup < 1.0).any()
        or (anneal_floor < 0.0).any()
        or (anneal_floor > 1.0).any()
        or (source_order < 0.0).any()
        or (source_rmse < 0.0).any()
        or (tensor_mean <= 0.0).any()
        or (row_spread < 0.0).any()
        or (downstream < 0.0).any()
        or (order_upper < 0.0).any()
        or (bridge_ratio < 0.0).any()
        or not np.isin(rg["rg_block_size"], registered_blocks).all()
        or any(
            (value < 0.0).any()
            for name, value in rg.items()
            if name != "rg_block_size"
        )
        or any(
            (value < 0.0).any()
            for name, value in avalanche.items()
            if name != "avalanche_tail_comparison"
        )
    ):
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            "E8 bounds, criteria, or horizons are outside their domain"
        )
        return result
    positive_upper = order_upper > 0.0
    if np.any(~positive_upper & (source_order > 0.0)):
        result["status"] = "INCOMPLETE"
        result["reasons"].append(
            "positive source order parameter has a zero bridge upper bound"
        )
        return result
    expected_ratio = np.zeros_like(source_order)
    expected_ratio[positive_upper] = (
        source_order[positive_upper] / order_upper[positive_upper]
    )
    bridge_identity_error = float(np.max(np.abs(
        bridge_ratio - expected_ratio)))
    relative = np.abs(observed - predicted) / np.maximum(predicted, 1.0)
    horizon_coverage = float(np.mean(observed <= predicted))
    threshold = reg["thresholds"]
    return _finalize(result, {
        "theorem_bound": float(np.max(bound_ratio))
        <= threshold["theorem_bound_ratio_max"],
        "heldout_direct_entry": float(np.mean(direct < criterion))
        >= threshold["heldout_entry_fraction_min"],
        "heldout_horizon": horizon_coverage
        >= threshold["horizon_coverage_min"],
        "order_parameter_bridge": float(np.max(bridge_ratio))
        <= threshold["order_parameter_bridge_ratio_max"],
        "order_parameter_bridge_identity": bridge_identity_error
        <= threshold["order_parameter_bridge_identity_error_max"],
        "rg_density_identity": float(np.max(
            rg["rg_density_identity_residual"]))
        <= threshold["rg_density_identity_residual_max"],
        "rg_layernorm_scale_identity": float(np.max(
            rg["rg_layernorm_scale_residual"]))
        <= threshold["rg_layernorm_scale_residual_max"],
        "rg_softmax_aggregation_identity": float(np.max(
            rg["rg_softmax_aggregation_residual"]))
        <= threshold["rg_softmax_aggregation_residual_max"],
        "rg_blocking_semigroup": float(np.max(
            rg["rg_repeated_block_semigroup_residual"]))
        <= threshold["rg_repeated_block_semigroup_residual_max"],
        "rg_deductive_closure": float(np.max(
            rg["rg_deductive_closure_ratio"]))
        <= threshold["rg_deductive_closure_ratio_max"],
        "rg_full_layer_closure": float(np.max(
            rg["rg_full_layer_closure_ratio"]))
        <= threshold["rg_full_layer_closure_ratio_max"],
        "collapse_prediction": all(
            _paired_pass(
                value, alpha=alpha,
                threshold=threshold["paired_model_score_difference_max"],
                direction="lower",
            )
            for value in score_bounds.values()
        ),
    }, {
        "complete_theorem_groups": len(theorem_rows),
        "complete_rg_theory_rows": len(rg_theorem_rows),
        "maximum_theorem_bound_ratio": float(np.max(bound_ratio)),
        "direct_entry_fraction": float(np.mean(direct < criterion)),
        "maximum_direct_to_criterion_ratio":
            float(np.max(direct / criterion)),
        "horizon_coverage": horizon_coverage,
        "median_absolute_relative_horizon_error":
            float(np.median(relative)),
        "maximum_source_order_parameter": float(np.max(source_order)),
        "maximum_source_rmse": float(np.max(source_rmse)),
        "minimum_tensor_mean_abs": float(np.min(tensor_mean)),
        "maximum_row_input_spread": float(np.max(row_spread)),
        "maximum_downstream_lipschitz": float(np.max(downstream)),
        "maximum_order_parameter_upper": float(np.max(order_upper)),
        "maximum_order_parameter_bridge_ratio":
            float(np.max(bridge_ratio)),
        "maximum_order_parameter_bridge_identity_error":
            bridge_identity_error,
        "rg_block_sizes": sorted(set(
            float(value) for value in rg["rg_block_size"])),
        "maximum_rg_density_identity_residual": float(np.max(
            rg["rg_density_identity_residual"])),
        "maximum_rg_layernorm_scale_residual": float(np.max(
            rg["rg_layernorm_scale_residual"])),
        "maximum_rg_within_covariance_trace": float(np.max(
            rg["rg_within_covariance_trace"])),
        "maximum_rg_deductive_closure_upper": float(np.max(
            rg["rg_deductive_closure_upper"])),
        "maximum_rg_deductive_closure_ratio": float(np.max(
            rg["rg_deductive_closure_ratio"])),
        "maximum_rg_softmax_aggregation_residual": float(np.max(
            rg["rg_softmax_aggregation_residual"])),
        "maximum_rg_repeated_block_semigroup_residual": float(np.max(
            rg["rg_repeated_block_semigroup_residual"])),
        "maximum_rg_full_layer_closure_upper": float(np.max(
            rg["rg_full_layer_closure_upper"])),
        "maximum_rg_full_layer_closure_ratio": float(np.max(
            rg["rg_full_layer_closure_ratio"])),
        "maximum_avalanche_observables": {
            name: float(np.max(value))
            for name, value in avalanche.items()
        },
        "model_score_paired_randomization": score_bounds,
        "loss_difference_paired_randomization": loss_bounds,
    })


ANALYZERS = {
    "E0": _e0,
    "E1": _e1,
    "E2": _e2,
    "E3": _e3,
    "E4": _e4,
    "E5": _e5,
    "E6": _e6,
    "E7": _e7,
    "E8": _e8,
}


def analyze(record_list, registration):
    if set(registration) != set(records.PROTOCOLS):
        raise ValueError("registration must contain exactly E0 through E8")
    by_protocol = {pid: [] for pid in records.PROTOCOLS}
    for record in record_list:
        records.validate_record(record)
        by_protocol[record["protocol_id"]].append(record)
    results = {}
    for pid in records.PROTOCOLS:
        try:
            results[pid] = ANALYZERS[pid](
                by_protocol[pid], registration[pid]
            )
        except (ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
            results[pid] = error_result(pid, by_protocol[pid], error)
    statuses = [row["status"] for row in results.values()]
    if all(value == "CONFIRMED" for value in statuses):
        program_status = "CONFIRMED"
    elif any(value == "INCOMPLETE" for value in statuses):
        program_status = "INCOMPLETE"
    elif any(value == "INFEASIBLE" for value in statuses):
        program_status = "INFEASIBLE"
    elif any(value == "NOT_CONFIRMED" for value in statuses):
        program_status = "NOT_CONFIRMED"
    else:
        program_status = "NOT_RUN"
    payload = records.strict_dumps(record_list).encode("utf-8")
    return {
        "schema_version": records.SCHEMA_VERSION,
        "analysis_status": program_status,
        "record_count": len(record_list),
        "record_bundle_sha256": hashlib.sha256(payload).hexdigest(),
        "experiments": results,
    }


def load_registration(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != records.SCHEMA_VERSION:
        raise ValueError("registration schema version is stale")
    experiments = value.get("experiments")
    if not isinstance(experiments, dict):
        raise ValueError("registration experiments are missing")
    return experiments


def load_records(raw_directory, binding_root):
    raw_directory = Path(raw_directory)
    result = []
    for path in sorted(raw_directory.rglob("record-*.json")):
        result.append(
            records.load_and_verify(
                path, binding_root=binding_root, artifact_root=path.parent
            )
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--binding-root", required=True)
    parser.add_argument(
        "--registration",
        default=str(
            EXPERIMENTS
            / "protocols"
            / "row_map_confirmation"
            / "registration.json"
        ),
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    registration = load_registration(args.registration)
    report = analyze(
        load_records(args.raw_dir, args.binding_root), registration
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(records.strict_dumps(report), encoding="utf-8")
    print(f"{report['analysis_status']}: {report['record_count']} records")


if __name__ == "__main__":
    main()
