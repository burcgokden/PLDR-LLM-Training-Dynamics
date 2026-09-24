#!/usr/bin/env python3
"""Compute E2-E6 measurements from closed direct-certificate artifacts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import campaign_design  # noqa: E402
from campaign_record import strict_dumps  # noqa: E402
from direct_certificate import (  # noqa: E402
    componentwise_self_map_budget,
    complete_defect_vector,
    lifted_block_matrix,
    monotone_entry_horizon,
    nonautonomous_schedule_bounds,
    robust_family_certificate,
    solve_discrete_lyapunov,
)
from optimizer_transport import jury_margins  # noqa: E402
from pathwise_contraction import (  # noqa: E402
    cross_metric_family_certificate,
    path_product_convolution,
    structured_family_certificate,
)
from validated_linear_algebra import graph_window_certificate  # noqa: E402


def _load(path):
    if isinstance(path, dict):
        value = path
    else:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("certificate input must be a JSON object")
    return value


def _closed(value, required, label):
    if set(value) != set(required):
        missing = sorted(set(required) - set(value))
        extra = sorted(set(value) - set(required))
        raise ValueError(
            f"{label} schema mismatch: missing={missing}, extra={extra}")
    return value


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    return result


def _scalar(value, name, *, nonnegative=False):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _integer(value, name, *, lower=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        raise ValueError(f"{name} must be an integer at least {lower}")
    return value


def _relative(actual, expected):
    actual = _array(actual, "actual")
    expected = _array(expected, "expected")
    if actual.shape != expected.shape:
        raise ValueError("relative-error shapes disagree")
    scale = max(
        float(np.linalg.norm(actual)),
        float(np.linalg.norm(expected)),
        1e-30,
    )
    return float(np.linalg.norm(actual - expected) / scale)


def _ratio(observed, bound):
    observed = _scalar(observed, "observed", nonnegative=True)
    bound = _scalar(bound, "bound", nonnegative=True)
    if bound == 0.0:
        return 0.0 if observed == 0.0 else float(np.finfo(float).max)
    return observed / bound


def _measurement(name, value, unit, method):
    return {
        "name": name,
        "value": float(value),
        "unit": unit,
        "method": method,
        "status": "OBSERVED",
        "reason_code": None,
    }


def measure_e2(value):
    required = {
        "normal_operator", "normal_velocity", "jacobian_state",
        "reference_forcing", "normal_residual",
        "linearization_residual_norm", "linearization_residual_bound",
        "nonlinear_residual_norm", "nonlinear_residual_bound",
        "alpha", "beta1", "scalar_decay",
    }
    _closed(value, required, "E2")
    operator = _array(value["normal_operator"], "normal_operator", ndim=2)
    if operator.shape[0] != operator.shape[1]:
        raise ValueError("normal_operator must be square")
    velocity = _array(value["normal_velocity"], "normal_velocity", ndim=1)
    state = _array(value["jacobian_state"], "jacobian_state", ndim=1)
    forcing = _array(value["reference_forcing"], "reference_forcing", ndim=1)
    residual = _array(value["normal_residual"], "normal_residual", ndim=1)
    if (
        operator.shape[0] != len(state)
        or velocity.shape != state.shape
        or forcing.shape != state.shape
        or residual.shape != state.shape
    ):
        raise ValueError("normal-operator dimensions disagree")
    identity = _relative(
        velocity, operator @ state + forcing + residual)
    operator_scale = max(float(np.linalg.norm(operator, ord=2)), 1e-30)
    self_adjoint_residual = float(
        np.linalg.norm(operator - operator.T, ord=2) / operator_scale)
    symmetric = 0.5 * (operator + operator.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    alpha = _scalar(value["alpha"], "alpha", nonnegative=True)
    beta1 = _scalar(value["beta1"], "beta1", nonnegative=True)
    decay = _scalar(
        value["scalar_decay"], "scalar_decay", nonnegative=True)
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    if beta1 >= 1.0:
        raise ValueError("beta1 must be below one")
    if decay >= 1.0:
        raise ValueError("scalar_decay must be below one")
    matrices = []
    for eigenvalue in eigenvalues:
        gain = (1.0 - beta1) * float(eigenvalue)
        matrices.append(np.asarray([
            [1.0 - decay - alpha * gain, -alpha * beta1],
            [gain, beta1],
        ]))
    jury_rows = [jury_margins(matrix) for matrix in matrices]
    jury_minimum = min(
        margin
        for row in jury_rows
        for margin in row["margins"].values()
    )
    lower_margin = (
        alpha * (1.0 - beta1) * float(eigenvalues[0])
        + decay * (1.0 - beta1)
    )
    upper_margin = (
        2.0 * (1.0 + beta1)
        - alpha * (1.0 - beta1) * float(eigenvalues[-1])
        - decay * (1.0 + beta1)
    )
    spectral_loading = (
        alpha * (1.0 - beta1) * float(eigenvalues[-1])
        + decay * (1.0 + beta1)
    ) / (2.0 * (1.0 + beta1))
    measurements = [
        _measurement(
            "lower_normal_edge", eigenvalues[0],
            "normal_operator_unit", "symmetric_normal_eigenvalue"),
        _measurement(
            "upper_normal_edge", eigenvalues[-1],
            "normal_operator_unit", "symmetric_normal_eigenvalue"),
        _measurement(
            "jury_margin_min", jury_minimum,
            "dimensionless", "spectrum_derived_second_order_jury_margins"),
        _measurement(
            "jury_lower_margin", lower_margin,
            "dimensionless", "worst_case_positive_one_jury_margin"),
        _measurement(
            "jury_upper_margin", upper_margin,
            "dimensionless", "worst_case_negative_one_jury_margin"),
        _measurement(
            "spectral_loading", spectral_loading,
            "dimensionless", "normalized_worst_case_upper_jury_loading"),
        _measurement(
            "normal_operator_self_adjoint_residual",
            self_adjoint_residual,
            "relative_l2", "K_minus_K_transpose"),
        _measurement(
            "normal_linearization_residual_ratio",
            _ratio(
                value["linearization_residual_norm"],
                value["linearization_residual_bound"]),
            "ratio", "primitive_normal_linearization_enclosure"),
        _measurement(
            "nonlinear_residual_enclosure_ratio",
            _ratio(
                value["nonlinear_residual_norm"],
                value["nonlinear_residual_bound"]),
            "ratio", "normal_second_derivative_enclosure"),
        _measurement(
            "normal_operator_identity_residual", identity,
            "relative_l2", "Q_minus_KZ_minus_f_minus_e_g"),
    ]
    return {
        "protocol_id": "E2",
        "alpha": alpha,
        "beta1": beta1,
        "scalar_decay": decay,
        "normal_eigenvalues": eigenvalues.tolist(),
        "spectrum_derived_scalar_mode_matrices": [
            matrix.tolist() for matrix in matrices
        ],
        "jury": jury_rows,
        "measurements": measurements,
    }


def measure_e3(value):
    required = {
        "nominal_matrix", "q_matrix", "family_matrices",
        "euclidean_interval_radius",
    }
    _closed(value, required, "E3")
    nominal = _array(value["nominal_matrix"], "nominal_matrix", ndim=2)
    q_matrix = _array(value["q_matrix"], "q_matrix", ndim=2)
    construction = solve_discrete_lyapunov(nominal, q_matrix)
    family = [
        _array(matrix, "family_matrix", ndim=2)
        for matrix in value["family_matrices"]
    ]
    robust = robust_family_certificate(
        nominal,
        family,
        construction["metric"],
        construction["nominal_q_bound"],
        euclidean_radius=value["euclidean_interval_radius"],
    )
    measurements = [
        _measurement(
            "nominal_lyapunov_residual",
            construction["relative_residual"],
            "relative_operator_norm", "discrete_lyapunov_solve"),
        _measurement(
            "metric_eigenvalue_min", construction["metric_lower"],
            "metric_unit", "symmetric_metric_eigensolve"),
        _measurement(
            "nominal_q", construction["nominal_q_bound"],
            "H_gain", "lyapunov_Q_over_H_bound"),
        _measurement(
            "family_perturbation_delta",
            robust["metric_family_radius"],
            "H_gain", "euclidean_radius_times_metric_condition"),
        _measurement(
            "robust_q", robust["robust_q"],
            "H_gain", "nominal_q_plus_family_radius"),
        _measurement(
            "observed_h_gain", robust["maximum_exact_h_gain"],
            "H_gain", "heldout_or_vertex_induced_H_norm"),
        _measurement(
            "interval_matrix_enclosure_ratio",
            _ratio(
                robust["observed_euclidean_family_radius"],
                robust["certified_euclidean_family_radius"],
            ),
            "ratio", "outward_rounded_family_matrix_enclosure"),
    ]
    return {
        "protocol_id": "E3",
        "construction": {
            **{
                name: item for name, item in construction.items()
                if name not in {"metric", "q_matrix"}
            },
            "metric": construction["metric"].tolist(),
            "q_matrix": construction["q_matrix"].tolist(),
        },
        "robust_family": robust,
        "measurements": measurements,
    }


def measure_structured_e3(value):
    required = {
        "nominal_matrix", "q_matrix", "family_matrices",
        "beta1", "window_length",
    }
    _closed(value, required, "structured E3")
    nominal = _array(value["nominal_matrix"], "nominal_matrix", ndim=2)
    q_matrix = _array(value["q_matrix"], "q_matrix", ndim=2)
    family = [
        _array(matrix, "family_matrix", ndim=2)
        for matrix in value["family_matrices"]
    ]
    certificate = structured_family_certificate(
        nominal,
        family,
        value["beta1"],
        q_matrix=q_matrix,
        window_length=value["window_length"],
    )
    construction = certificate["construction"]
    structured_q = certificate["structured_q"]
    corner_gain = certificate["maximum_corner_h_gain"]
    observed_gain = certificate["maximum_observed_h_gain"]
    measurements = [
        _measurement(
            "nominal_lyapunov_residual",
            construction["relative_residual"],
            "relative_operator_norm", "discrete_lyapunov_solve"),
        _measurement(
            "metric_eigenvalue_min", construction["metric_lower"],
            "metric_unit", "symmetric_metric_eigensolve"),
        _measurement(
            "nominal_q", construction["nominal_q_bound"],
            "H_gain", "lyapunov_Q_over_H_bound"),
        _measurement(
            "structured_family_q", structured_q,
            "H_gain", "exact_ldl_correlated_corner_lmis"),
        _measurement(
            "structured_family_slack", 1.0 - structured_q,
            "H_gain", "one_minus_structured_family_q"),
        _measurement(
            "maximum_corner_h_gain", corner_gain,
            "H_gain", "floating_corner_induced_H_norm_diagnostic"),
        _measurement(
            "multiaffine_reconstruction_residual",
            certificate["maximum_reconstruction_residual"],
            "relative_operator_norm",
            "primitive_barycentric_corner_reconstruction"),
        _measurement(
            "primitive_box_enclosure_ratio",
            certificate["maximum_primitive_box_ratio"],
            "ratio", "primitive_coordinate_box_membership"),
        _measurement(
            "path_window_gain", certificate["maximum_window_gain"],
            "H_gain", "exact_admissible_path_product"),
        _measurement(
            "observed_h_gain", observed_gain,
            "H_gain", "observed_primitive_family_induced_H_norm"),
    ]
    observed_radius = max(
        float(np.linalg.norm(matrix - nominal, ord=2))
        for matrix in family
    )
    robust_compatibility = {
        "certificate_kind": "correlated_multiaffine_primitive_box",
        "certified_euclidean_family_radius": (
            0.0 if observed_radius == 0.0
            else math.nextafter(observed_radius, math.inf)
        ),
        "observed_euclidean_family_radius": observed_radius,
        "nominal_q": construction["nominal_q_bound"],
        "robust_q": structured_q,
        "structured_q": structured_q,
        "strictly_contracting": bool(structured_q < 1.0),
        "exact_family_h_gains": certificate["family_h_gains"],
        "maximum_exact_h_gain": observed_gain,
        "corner_h_gains": certificate["corner_h_gains"],
        "maximum_corner_h_gain": corner_gain,
    }
    return {
        "protocol_id": "E3",
        "construction": {
            **{
                name: item for name, item in construction.items()
                if name not in {"metric", "q_matrix"}
            },
            "metric": construction["metric"].tolist(),
            "q_matrix": construction["q_matrix"].tolist(),
        },
        "structured_family": {
            **{
                name: item for name, item in certificate.items()
                if name not in {"construction"}
            },
        },
        "robust_family": robust_compatibility,
        "measurements": measurements,
    }


def _h_norm_rows(rows, metric):
    rows = _array(rows, "vectors", ndim=2)
    metric = _array(metric, "metric", ndim=2)
    if not len(rows):
        raise ValueError("vectors must be nonempty")
    if metric.shape != (rows.shape[1], rows.shape[1]):
        raise ValueError("metric and vector dimensions disagree")
    if not np.allclose(metric, metric.T, rtol=0.0, atol=1e-12):
        raise ValueError("metric must be symmetric")
    if np.linalg.eigvalsh(0.5 * (metric + metric.T))[0] <= 0.0:
        raise ValueError("metric must be positive definite")
    values = np.einsum("ni,ij,nj->n", rows, metric, rows)
    if (values < -1e-12).any():
        raise ValueError("metric produced a negative squared norm")
    return np.sqrt(np.maximum(values, 0.0))


def measure_e4(value):
    required = {
        "metric", "forcing_vectors", "complete_defect_vectors",
        "complete_disturbance_vectors", "tail_offsets",
        "persistent_constant", "geometric_constant", "geometric_rate",
        "comparison_matrix", "comparison_weights", "comparison_kappa",
    }
    _closed(value, required, "E4")
    metric = _array(value["metric"], "metric", ndim=2)
    forcing_vectors = _array(
        value["forcing_vectors"], "forcing_vectors", ndim=2)
    defect_vectors = _array(
        value["complete_defect_vectors"],
        "complete_defect_vectors",
        ndim=2,
    )
    disturbance_vectors = _array(
        value["complete_disturbance_vectors"],
        "complete_disturbance_vectors",
        ndim=2,
    )
    if not (
        forcing_vectors.shape
        == defect_vectors.shape
        == disturbance_vectors.shape
    ):
        raise ValueError("forcing and defect vector arrays disagree")
    forcing_norms = _h_norm_rows(forcing_vectors, metric)
    defect_norms = _h_norm_rows(defect_vectors, metric)
    disturbance_norms = _h_norm_rows(disturbance_vectors, metric)
    recomposed = forcing_vectors + defect_vectors
    residuals = [
        _relative(actual, expected)
        for actual, expected in zip(disturbance_vectors, recomposed)
    ]
    offsets = _array(value["tail_offsets"], "tail_offsets", ndim=1)
    if (
        len(offsets) != len(disturbance_norms)
        or (offsets < 0.0).any()
        or not np.equal(offsets, np.floor(offsets)).all()
        or not np.all(np.diff(offsets) > 0.0)
    ):
        raise ValueError(
            "tail_offsets must be strictly increasing nonnegative integers")
    persistent = _scalar(
        value["persistent_constant"],
        "persistent_constant",
        nonnegative=True,
    )
    constant = _scalar(
        value["geometric_constant"],
        "geometric_constant",
        nonnegative=True,
    )
    rate = _scalar(
        value["geometric_rate"], "geometric_rate", nonnegative=True)
    comparison = _array(
        value["comparison_matrix"], "comparison_matrix", ndim=2)
    weights = _array(
        value["comparison_weights"], "comparison_weights", ndim=1)
    kappa = _scalar(
        value["comparison_kappa"], "comparison_kappa", nonnegative=True)
    if (
        comparison.shape != (len(weights), len(weights))
        or not len(weights)
        or (comparison < 0.0).any()
        or (weights <= 0.0).any()
        or not kappa < 1.0
    ):
        raise ValueError(
            "positive comparison matrix, weights, or kappa are outside domain")
    product = comparison @ weights
    denominator = kappa * weights
    if kappa == 0.0:
        witness_ratio = (
            0.0 if np.all(product == 0.0) else float(np.finfo(float).max)
        )
    else:
        witness_ratio = float(np.max(product / denominator))
    envelopes = (
        persistent + constant * np.power(rate, offsets.astype(int))
    )
    if not np.isfinite(envelopes).all():
        raise ValueError("derived geometric envelopes are nonfinite")
    ratios = np.zeros_like(envelopes)
    positive = envelopes > 0.0
    ratios[positive] = disturbance_norms[positive] / envelopes[positive]
    if np.any((~positive) & (disturbance_norms != 0.0)):
        ratios[(~positive) & (disturbance_norms != 0.0)] = np.finfo(float).max
    measurements = [
        _measurement(
            "forcing_norm", np.max(forcing_norms),
            "lifted_H_unit", "maximum_forcing_H_norm"),
        _measurement(
            "complete_defect_norm", np.max(defect_norms),
            "lifted_H_unit", "maximum_complete_defect_H_norm"),
        _measurement(
            "complete_disturbance_norm", np.max(disturbance_norms),
            "lifted_H_unit", "maximum_complete_disturbance_H_norm"),
        _measurement(
            "persistent_disturbance_floor", persistent,
            "lifted_H_unit", "registered_persistent_source_H_norm"),
        _measurement(
            "disturbance_envelope", np.max(envelopes),
            "lifted_H_unit", "registered_K_times_r_to_tail_offset"),
        _measurement(
            "envelope_ratio", np.max(ratios),
            "ratio", "complete_disturbance_over_envelope"),
        _measurement(
            "geometric_rate", rate,
            "per_global_step", "registered_transient_tail_rate"),
        _measurement(
            "positive_comparison_witness_ratio", witness_ratio,
            "ratio", "maximum_Pv_over_kappa_v"),
        _measurement(
            "defect_recomposition_residual", np.max(residuals),
            "relative_l2", "W_minus_F_plus_E"),
    ]
    return {
        "protocol_id": "E4",
        "tail_offsets": offsets.astype(int).tolist(),
        "persistent_constant": persistent,
        "geometric_constant": constant,
        "geometric_rate": rate,
        "comparison_matrix_times_weights": product.tolist(),
        "positive_comparison_witness_ratio": witness_ratio,
        "derived_disturbance_envelopes": envelopes.tolist(),
        "forcing_h_norms": forcing_norms.tolist(),
        "defect_h_norms": defect_norms.tolist(),
        "disturbance_h_norms": disturbance_norms.tolist(),
        "envelope_ratios": ratios.tolist(),
        "measurements": measurements,
    }


def _path_window_budget(*, gains, disturbances, radii,
                        certified_window_gain):
    """Evaluate every registered path's prefix and closed-window budgets."""
    gains = _array(gains, "path_gains")
    disturbances = _array(disturbances, "path_disturbance_bounds")
    radii = _array(radii, "path_lifted_radii")
    if gains.ndim == 1:
        gains = gains[np.newaxis, :]
    if disturbances.ndim == 1:
        disturbances = disturbances[np.newaxis, :]
    if radii.ndim == 1:
        radii = radii[np.newaxis, :]
    if gains.ndim != 2 or disturbances.ndim != 2 or radii.ndim != 2:
        raise ValueError("path gains, disturbances, and radii must be matrices")
    if not gains.shape[0] or not gains.shape[1] \
            or gains.shape != disturbances.shape:
        raise ValueError("path gains and disturbances must be nonempty and aligned")
    if radii.shape != (gains.shape[0], gains.shape[1] + 1):
        raise ValueError(
            "path_lifted_radii must contain every endpoint of every path")
    if (gains < 0.0).any() or (disturbances < 0.0).any() \
            or (radii < 0.0).any():
        raise ValueError("path gains, disturbances, and radii must be nonnegative")
    window_gain = _scalar(
        certified_window_gain,
        "certified_path_window_gain",
        nonnegative=True,
    )
    if not window_gain < 1.0:
        raise ValueError("the certified path window must strictly contract")
    prefix_required = gains * radii[:, :-1] + disturbances
    prefix_slacks = radii[:, 1:] - prefix_required
    convolutions = [
        path_product_convolution(path_gains, path_disturbances, initial=0.0)
        for path_gains, path_disturbances in zip(gains, disturbances)
    ]
    edge_products = np.prod(gains, axis=1)
    maximum_edge_product = float(np.max(edge_products))
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(maximum_edge_product), abs(window_gain))
    if np.any(edge_products > window_gain + tolerance):
        raise ValueError(
            "certified_path_window_gain does not enclose every edge product")
    block_disturbances = np.asarray([
        value["terminal_bound"] for value in convolutions
    ], dtype=float)
    block_required = window_gain * radii[:, 0] + block_disturbances
    block_slacks = radii[:, -1] - block_required
    return {
        "window_length": gains.shape[1],
        "admissible_path_count": gains.shape[0],
        "edge_gain_product": maximum_edge_product,
        "edge_gain_products": edge_products.tolist(),
        "certified_window_gain": window_gain,
        "block_disturbance_bound": float(np.max(block_disturbances)),
        "block_disturbance_bounds": block_disturbances.tolist(),
        "block_lifted_required": float(np.max(block_required)),
        "block_lifted_required_by_path": block_required.tolist(),
        "block_lifted_slack": float(np.min(block_slacks)),
        "block_lifted_slacks": block_slacks.tolist(),
        "prefix_lifted_required": prefix_required.tolist(),
        "prefix_budget_slacks": prefix_slacks.tolist(),
        "minimum_prefix_budget_slack": float(np.min(prefix_slacks)),
        "certified": bool(
            np.all(block_slacks >= 0.0) and np.all(prefix_slacks >= 0.0)
        ),
    }


def measure_e5(value):
    required = {
        "q", "lifted_radius", "disturbance_bound",
        "next_lifted_radius", "center_errors", "lifted_lipschitz",
        "auxiliary_lipschitz", "auxiliary_radii",
        "next_auxiliary_radii", "initial_membership_margin",
        "heldout_image_margins", "family_box_enclosure_ratio",
        "path_gains", "path_disturbance_bounds", "path_lifted_radii",
        "certified_path_window_gain",
    }
    _closed(value, required, "E5")
    budget = componentwise_self_map_budget(
        q=value["q"],
        lifted_radius=value["lifted_radius"],
        disturbance_bound=value["disturbance_bound"],
        next_lifted_radius=value["next_lifted_radius"],
        center_errors=value["center_errors"],
        lifted_lipschitz=value["lifted_lipschitz"],
        auxiliary_lipschitz=value["auxiliary_lipschitz"],
        auxiliary_radii=value["auxiliary_radii"],
        next_auxiliary_radii=value["next_auxiliary_radii"],
    )
    path_budget = _path_window_budget(
        gains=value["path_gains"],
        disturbances=value["path_disturbance_bounds"],
        radii=value["path_lifted_radii"],
        certified_window_gain=value["certified_path_window_gain"],
    )
    heldout = _array(
        value["heldout_image_margins"],
        "heldout_image_margins",
        ndim=1,
    )
    if not len(heldout):
        raise ValueError("heldout_image_margins must be nonempty")
    initial = _scalar(
        value["initial_membership_margin"],
        "initial_membership_margin",
    )
    enclosure = _scalar(
        value["family_box_enclosure_ratio"],
        "family_box_enclosure_ratio",
        nonnegative=True,
    )
    escapes = int(np.count_nonzero(heldout < 0.0))
    measurements = [
        _measurement(
            "initial_membership_margin", initial,
            "scaled_state_unit", "minimum_initial_box_margin"),
        _measurement(
            "lifted_budget_slack", budget["lifted_slack"],
            "lifted_H_unit", "B_next_minus_qB_minus_w"),
        _measurement(
            "path_window_length", path_budget["window_length"],
            "optimizer_update", "registered_contracting_path_length"),
        _measurement(
            "path_window_gain", path_budget["certified_window_gain"],
            "H_gain", "exact_registered_path_product_upper"),
        _measurement(
            "block_lifted_budget_slack",
            path_budget["block_lifted_slack"],
            "lifted_H_unit", "B_block_next_minus_gamma_B_minus_Omega"),
        _measurement(
            "prefix_budget_slack_min",
            path_budget["minimum_prefix_budget_slack"],
            "lifted_H_unit", "minimum_registered_path_prefix_radius_slack"),
        _measurement(
            "auxiliary_budget_slack_min",
            budget["minimum_auxiliary_slack"],
            "scaled_auxiliary_unit", "minimum_component_radius_slack"),
        _measurement(
            "heldout_image_margin", np.min(heldout),
            "scaled_state_unit", "minimum_heldout_successor_box_margin"),
        _measurement(
            "one_step_escape_count", escapes,
            "count", "negative_heldout_image_margin_count"),
        _measurement(
            "family_box_enclosure_ratio", enclosure,
            "ratio", "outward_rounded_box_enclosure"),
    ]
    return {
        "protocol_id": "E5",
        "budget": {
            **budget,
            "auxiliary_required": budget["auxiliary_required"].tolist(),
            "auxiliary_slacks": budget["auxiliary_slacks"].tolist(),
        },
        "path_budget": path_budget,
        "heldout_image_margins": heldout.tolist(),
        "measurements": measurements,
    }


def measure_e6(value):
    required = {
        "certificate_time", "q", "r", "initial_h_norm",
        "forcing_constant", "persistent_forcing_constant",
        "path_window_length", "path_window_gain",
        "block_forcing_constant", "block_persistent_forcing_constant",
        "block_rate",
        "metric_lower", "cover_floor",
        "cover_transient_constant", "cover_rate", "criterion",
        "observed_entry_step", "direct_row_map_upper",
        "sustained_entry_indicator",
        "schedule_gains", "schedule_disturbance_bounds",
    }
    _closed(value, required, "E6")
    certificate_time = _integer(
        value["certificate_time"], "certificate_time")
    observed = _integer(
        value["observed_entry_step"], "observed_entry_step")
    criterion = _scalar(
        value["criterion"], "criterion", nonnegative=True)
    if criterion <= 0.0:
        raise ValueError("criterion must be positive")
    cover_floor = _scalar(
        value["cover_floor"], "cover_floor", nonnegative=True)
    persistent = _scalar(
        value["persistent_forcing_constant"],
        "persistent_forcing_constant", nonnegative=True)
    q = _scalar(value["q"], "q", nonnegative=True)
    r = _scalar(value["r"], "r", nonnegative=True)
    path_window_length = _integer(
        value["path_window_length"], "path_window_length", lower=1)
    path_window_gain = _scalar(
        value["path_window_gain"], "path_window_gain", nonnegative=True)
    block_forcing = _scalar(
        value["block_forcing_constant"],
        "block_forcing_constant", nonnegative=True)
    block_persistent = _scalar(
        value["block_persistent_forcing_constant"],
        "block_persistent_forcing_constant", nonnegative=True)
    block_rate = _scalar(
        value["block_rate"], "block_rate", nonnegative=True)
    metric_lower = _scalar(
        value["metric_lower"], "metric_lower", nonnegative=True)
    if (
        not r < 1.0
        or not path_window_gain < 1.0
        or not block_rate < 1.0
        or not metric_lower > 0.0
    ):
        raise ValueError("E6 block contraction or metric is outside its domain")
    complete_floor = (
        cover_floor
        + block_persistent
        / ((1.0 - path_window_gain) * math.sqrt(metric_lower))
    )
    schedule = nonautonomous_schedule_bounds(
        gains=value["schedule_gains"],
        disturbance_bounds=value["schedule_disturbance_bounds"],
        initial_h_norm=value["initial_h_norm"],
        metric_lower=metric_lower,
        cover_floor=cover_floor,
        cover_transient_constant=value["cover_transient_constant"],
        cover_rate=value["cover_rate"],
        criterion=criterion,
    )
    registered_relative_sentinel = len(schedule["gains"]) + 1
    if complete_floor < criterion:
        block_horizon = monotone_entry_horizon(
            q=path_window_gain,
            r=block_rate,
            initial_h_norm=value["initial_h_norm"],
            forcing_constant=block_forcing,
            persistent_forcing=block_persistent,
            metric_lower=metric_lower,
            cover_floor=cover_floor,
            cover_transient_constant=value["cover_transient_constant"],
            cover_rate=float(value["cover_rate"]) ** path_window_length,
            criterion=criterion,
        )
        candidate_block_horizon = int(block_horizon["horizon"])
        candidate_horizon = candidate_block_horizon * path_window_length
        horizon = {
            **block_horizon,
            "block_horizon": candidate_block_horizon,
            "candidate_update_horizon": candidate_horizon,
            "path_window_length": path_window_length,
        }
        if candidate_horizon <= len(schedule["gains"]):
            relative_horizon = candidate_horizon
            horizon["horizon"] = relative_horizon
            horizon["finite_entry"] = True
            horizon["reason_code"] = None
        else:
            relative_horizon = registered_relative_sentinel
            horizon["unbounded_candidate_horizon"] = candidate_horizon
            horizon["horizon"] = relative_horizon
            horizon["finite_entry"] = False
            horizon["reason_code"] = (
                "NO_MONOTONE_ENTRY_THROUGH_REGISTERED_TERMINAL")
    else:
        relative_horizon = registered_relative_sentinel
        horizon = {
            "horizon": relative_horizon,
            "block_horizon": None,
            "path_window_length": path_window_length,
            "finite_entry": False,
            "reason_code": "COMPLETE_PERSISTENT_FLOOR_NOT_BELOW_CRITERION",
            "complete_persistent_floor": complete_floor,
        }
    predicted = certificate_time + relative_horizon
    relative_schedule_entry = schedule["first_sustained_relative_entry"]
    schedule_finite_entry = relative_schedule_entry is not None
    if relative_schedule_entry is None:
        relative_schedule_entry = registered_relative_sentinel
    schedule_predicted = certificate_time + relative_schedule_entry
    schedule["finite_entry"] = schedule_finite_entry
    schedule["reason_code"] = (
        None if schedule_finite_entry
        else "NO_SUSTAINED_ENTRY_THROUGH_REGISTERED_TERMINAL"
    )
    direct = _scalar(
        value["direct_row_map_upper"],
        "direct_row_map_upper",
        nonnegative=True,
    )
    sustained = _scalar(
        value["sustained_entry_indicator"],
        "sustained_entry_indicator",
        nonnegative=True,
    )
    if sustained not in {0.0, 1.0}:
        raise ValueError("sustained_entry_indicator must be zero or one")
    block_upper = (
        path_window_gain * _scalar(
            value["initial_h_norm"], "initial_h_norm", nonnegative=True)
        + block_persistent
        + block_forcing
    )
    measurements = [
        _measurement(
            "predicted_entry_step", predicted,
            "global_optimizer_step",
            "monotone_convolution_horizon_or_terminal_plus_one"),
        _measurement(
            "observed_entry_step", observed,
            "global_optimizer_step", "first_sustained_all_layer_entry"),
        _measurement(
            "entry_error", observed - predicted,
            "global_optimizer_step", "observed_minus_predicted"),
        _measurement(
            "residual_floor", horizon["complete_persistent_floor"],
            "output_row_per_input_row",
            "certified_cover_plus_persistent_source_floor"),
        _measurement(
            "direct_row_map_upper", direct,
            "output_row_per_input_row", "complete_direct_cover_upper"),
        _measurement(
            "criterion", criterion,
            "output_row_per_input_row", "registered_direct_criterion"),
        _measurement(
            "sustained_entry_indicator", sustained,
            "indicator", "all_future_observed_window_entry"),
        _measurement(
            "path_window_length", path_window_length,
            "optimizer_update", "registered_contracting_path_length"),
        _measurement(
            "path_window_gain", path_window_gain,
            "H_gain", "maximum_registered_window_product"),
        _measurement(
            "block_product_convolution_upper", block_upper,
            "lifted_H_unit", "one_block_gamma_y_plus_persistent_plus_transient"),
        _measurement(
            "schedule_product_convolution_upper",
            schedule["final_product_convolution_upper"],
            "lifted_H_unit", "time_indexed_q_product_and_w_convolution"),
        _measurement(
            "schedule_predicted_entry_step", schedule_predicted,
            "global_optimizer_step",
            "first_sustained_schedule_specific_direct_bound_entry"),
    ]
    return {
        "protocol_id": "E6",
        "certificate_time": certificate_time,
        "relative_horizon": horizon,
        "pathwise_block": {
            "window_length": path_window_length,
            "window_gain": path_window_gain,
            "forcing_constant": block_forcing,
            "persistent_forcing_constant": block_persistent,
            "rate": block_rate,
            "one_block_product_convolution_upper": block_upper,
        },
        "schedule_bound": schedule,
        "predicted_entry_step": predicted,
        "schedule_predicted_entry_step": schedule_predicted,
        "finite_entry": bool(horizon["finite_entry"]),
        "schedule_finite_entry": schedule_finite_entry,
        "measurements": measurements,
    }


def _measurement_values(value):
    rows = value.get("measurements")
    if not isinstance(rows, list):
        raise ValueError("measurement artifact has no measurement array")
    result = {row["name"]: float(row["value"]) for row in rows}
    if len(result) != len(rows):
        raise ValueError("measurement artifact contains duplicate names")
    return result


def _rational_float(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("exact certificate endpoint is not rational")
    result = int(value["numerator"]) / int(value["denominator"])
    if not math.isfinite(result):
        raise OverflowError("exact certificate endpoint is outside float range")
    return result


def construct_e2_input(*, lifted_certificate, optimizer_measurement,
                       e0_measurement, layer_index):
    certificate = _load(lifted_certificate)
    optimizer = _load(optimizer_measurement)
    e0 = _load(e0_measurement)
    state = _array(
        certificate["lifted_state_before"],
        "lifted_state_before", ndim=1)
    if len(state) % 2:
        raise ValueError("lifted state must have two equal blocks")
    dimension = len(state) // 2
    layer_index = _integer(layer_index, "layer_index")
    if layer_index >= dimension:
        raise ValueError("layer index is outside the lifted chart")
    beta1 = _scalar(certificate["beta1"], "beta1", nonnegative=True)
    forcing = _array(certificate["forcing"], "forcing", ndim=1)
    z_value = float(state[layer_index])
    q_value = float(forcing[dimension + layer_index] / (1.0 - beta1))
    rule = campaign_design.SCALAR_DIAGNOSTIC_RULES["normal_operator"]
    ridge = float(rule["ridge"])
    maximum = float(rule["maximum_eigenvalue"])
    least_squares = z_value * q_value / max(z_value * z_value, 1e-30)
    eigenvalue = min(max(least_squares, ridge), maximum)
    operator = np.asarray([[eigenvalue]], dtype=float)
    z = np.asarray([z_value], dtype=float)
    q = np.asarray([q_value], dtype=float)
    reference_forcing = q - operator @ z
    residual = q - operator @ z - reference_forcing
    optimizer_values = _measurement_values(optimizer)
    layers = e0.get("layers")
    if not isinstance(layers, list):
        raise ValueError("E0 artifact has no layer certificates")
    layer = next(
        (row for row in layers if int(row["layer"]) == layer_index), None)
    if layer is None:
        raise ValueError("E0 artifact omits the selected layer")
    primitive_bound = max(
        _rational_float(item["parameter_row_mixed_upper"])
        for item in layer["exact_joint_certificates"])
    input_value = {
        "normal_operator": operator.tolist(),
        "normal_velocity": q.tolist(),
        "jacobian_state": z.tolist(),
        "reference_forcing": reference_forcing.tolist(),
        "normal_residual": residual.tolist(),
        "linearization_residual_norm": float(np.linalg.norm(residual)),
        "linearization_residual_bound": primitive_bound,
        "nonlinear_residual_norm": 0.0,
        "nonlinear_residual_bound": primitive_bound,
        "alpha": float(certificate["alpha"]),
        "beta1": beta1,
        "scalar_decay": optimizer_values["decay_coefficient"],
    }
    return input_value


def measure_live_e2(*, lifted_certificate, optimizer_measurement,
                    e0_measurement, layer_index):
    input_value = construct_e2_input(
        lifted_certificate=lifted_certificate,
        optimizer_measurement=optimizer_measurement,
        e0_measurement=e0_measurement,
        layer_index=layer_index,
    )
    result = measure_e2(input_value)
    result.update({
        "schema_version": "pldr-live-e2-measurement-v1",
        "layer_index": int(layer_index),
        "normal_operator_rule": campaign_design.SCALAR_DIAGNOSTIC_RULES[
            "normal_operator"],
        "closed_input": input_value,
    })
    return result


def _mode_matrix(e2):
    matrices = e2.get("spectrum_derived_scalar_mode_matrices")
    if not isinstance(matrices, list) or len(matrices) != 1:
        raise ValueError("live layer E2 artifact must contain one scalar mode")
    return _array(matrices[0], "scalar_mode_matrix", ndim=2)


def measure_live_e3(*, nominal_e2, family_e2, certificate_box):
    nominal_value = _load(nominal_e2)
    if not family_e2:
        raise ValueError("live E3 family is empty")
    nominal = _mode_matrix(nominal_value)
    family = [_mode_matrix(_load(value)) for value in family_e2]
    beta1 = _scalar(
        nominal_value["beta1"], "nominal beta1", nonnegative=True)
    if not beta1 < 1.0 or any(
        abs(_scalar(_load(value)["beta1"], "family beta1") - beta1) > 1e-15
        for value in family_e2
    ):
        raise ValueError("live E3 family does not share one beta1")
    input_value = {
        "nominal_matrix": nominal.tolist(),
        "q_matrix": np.eye(len(nominal)).tolist(),
        "family_matrices": [matrix.tolist() for matrix in family],
        "beta1": beta1,
        "window_length": int(
            campaign_design.SCALAR_DIAGNOSTIC_RULES["structured_family"][
                "window_length"]
        ),
    }
    result = measure_structured_e3(input_value)
    result.update({
        "schema_version": "pldr-live-e3-measurement-v2",
        "certificate_box": certificate_box,
        "nominal_matrix": nominal.tolist(),
        "family_matrices": [matrix.tolist() for matrix in family],
        "closed_input": input_value,
    })
    return result


def _layer_lifted_step(lifted_certificate, optimizer_measurement,
                       e2_measurement, layer_index):
    certificate = (
        _load(lifted_certificate)
        if not isinstance(lifted_certificate, dict)
        else lifted_certificate
    )
    optimizer = (
        _load(optimizer_measurement)
        if not isinstance(optimizer_measurement, dict)
        else optimizer_measurement
    )
    e2 = (
        _load(e2_measurement)
        if not isinstance(e2_measurement, dict)
        else e2_measurement
    )
    before = _array(
        certificate["lifted_state_before"], "lifted_state_before", ndim=1)
    after = _array(
        certificate["lifted_state_after"], "lifted_state_after", ndim=1)
    if len(before) % 2 or after.shape != before.shape:
        raise ValueError("lifted state blocks disagree")
    dimension = len(before) // 2
    layer_index = _integer(layer_index, "layer_index")
    if layer_index >= dimension:
        raise ValueError("layer index is outside the lifted state")
    indices = [layer_index, dimension + layer_index]
    y_before = before[indices]
    y_after = after[indices]
    alpha = float(certificate["alpha"])
    beta1 = float(certificate["beta1"])
    decay = _scalar(e2["scalar_decay"], "scalar_decay", nonnegative=True)
    eigenvalues = e2.get("normal_eigenvalues")
    if not isinstance(eigenvalues, list) or len(eigenvalues) != 1:
        raise ValueError("layer E2 artifact must have one normal eigenvalue")
    normal = np.asarray([[float(eigenvalues[0])]])
    matrix = lifted_block_matrix(
        normal, decay, alpha=alpha, beta1=beta1)
    original_forcing = _array(
        certificate["forcing"], "forcing", ndim=1)
    q_value = original_forcing[dimension + layer_index] / (1.0 - beta1)
    z_value = y_before[0]
    normal_residual = np.asarray([
        float(certificate["normal_residual"][layer_index])])
    reference_forcing = np.asarray([
        q_value - normal[0, 0] * z_value])
    forcing = np.asarray([
        -alpha * (1.0 - beta1) * reference_forcing[0],
        (1.0 - beta1) * reference_forcing[0],
    ])
    decay_defect = np.asarray([
        float(certificate["decay_defect"][layer_index])
        + decay * z_value
    ])
    row_motion = np.asarray([
        float(certificate["row_motion_defect"][layer_index])])
    intervention = np.asarray([
        float(certificate["intervention_defect"][layer_index])])
    taylor = np.asarray([
        float(certificate["taylor_remainder"][layer_index])])
    coordinate = np.asarray([
        float(certificate["coordinate_motion_defect"][layer_index])])
    defect = complete_defect_vector(
        alpha=alpha,
        beta1=beta1,
        normal_residual=normal_residual,
        decay_defect=decay_defect,
        row_motion_defect=row_motion,
        intervention_defect=intervention,
        taylor_remainder=taylor,
        coordinate_motion_defect=coordinate,
    )
    disturbance = forcing + defect
    predicted = matrix @ y_before + disturbance
    if _relative(y_after, predicted) > 1e-10:
        raise ArithmeticError("rebased live layer lift did not close")
    return {
        "step": int(optimizer["global_optimizer_step"]),
        "matrix": matrix,
        "state_before": y_before,
        "state_after": y_after,
        "forcing": forcing,
        "defect": defect,
        "disturbance": disturbance,
    }


def _metric_norm(vector, metric):
    vector = _array(vector, "metric_vector", ndim=1)
    metric = _array(metric, "metric", ndim=2)
    return math.sqrt(max(0.0, float(vector @ metric @ vector)))


def measure_live_e4(*, lifted_certificates, optimizer_measurements,
                    e2_measurements, e3_measurement, layer_index,
                    selected_step):
    if not (
        len(lifted_certificates)
        == len(optimizer_measurements)
        == len(e2_measurements)
        and lifted_certificates
    ):
        raise ValueError("live E4 source arrays must be nonempty and aligned")
    steps = [
        _layer_lifted_step(certificate, optimizer, e2, layer_index)
        for certificate, optimizer, e2 in zip(
            lifted_certificates, optimizer_measurements, e2_measurements)
    ]
    steps.sort(key=lambda row: row["step"])
    if len({row["step"] for row in steps}) != len(steps):
        raise ValueError("live E4 source steps are duplicated")
    selected = next(
        (row for row in steps if row["step"] == int(selected_step)), None)
    if selected is None:
        raise ValueError("selected E4 step is outside the source window")
    e3 = _load(e3_measurement)
    metric = _array(e3["construction"]["metric"], "metric", ndim=2)
    start = steps[0]["step"]
    offsets = np.asarray([row["step"] - start for row in steps], dtype=int)
    norms = np.asarray([
        _metric_norm(row["disturbance"], metric) for row in steps
    ])
    rules = campaign_design.SCALAR_DIAGNOSTIC_RULES["defect_envelope"]
    rate = float(rules["geometric_rate"])
    persistent = float(norms[-1])
    transient = max(
        (max(0.0, float(value - persistent)) / (rate ** int(offset))
         for value, offset in zip(norms, offsets)),
        default=0.0,
    )
    if persistent > 0.0:
        persistent = math.nextafter(persistent, math.inf)
    if transient > 0.0:
        transient = math.nextafter(transient, math.inf)
    selected_offset = int(selected["step"] - start)
    input_value = {
        "metric": metric.tolist(),
        "forcing_vectors": [selected["forcing"].tolist()],
        "complete_defect_vectors": [selected["defect"].tolist()],
        "complete_disturbance_vectors": [selected["disturbance"].tolist()],
        "tail_offsets": [selected_offset],
        "persistent_constant": persistent,
        "geometric_constant": transient,
        "geometric_rate": rate,
        "comparison_matrix": [[rate]],
        "comparison_weights": [1.0],
        "comparison_kappa": rate,
    }
    result = measure_e4(input_value)
    result.update({
        "schema_version": "pldr-live-e4-measurement-v1",
        "layer_index": int(layer_index),
        "selected_step": int(selected_step),
        "source_steps": [row["step"] for row in steps],
        "source_disturbance_h_norms": norms.tolist(),
        "envelope_rule": rules,
        "closed_input": input_value,
    })
    return result


def _outward_maximum(values):
    value = max(float(item) for item in values)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "outward maximum values must be finite and nonnegative")
    return 0.0 if value == 0.0 else math.nextafter(value, math.inf)


def _matrix_family_data(e3):
    robust = e3.get("robust_family")
    if not isinstance(robust, dict):
        raise ValueError("E3 artifact omits the robust family")
    structured = e3.get("structured_family")
    if not isinstance(structured, dict):
        raise ValueError("E3 artifact omits the structured family")
    nominal = _array(e3["nominal_matrix"], "nominal_matrix", ndim=2)
    if nominal.shape != (2, 2):
        raise ValueError("live E5 requires a scalar two-block lifted mode")
    radius = _scalar(
        robust["certified_euclidean_family_radius"],
        "certified_euclidean_family_radius", nonnegative=True,
    )
    q = _scalar(
        structured["structured_q"], "structured_q", nonnegative=True)
    window_length = _integer(
        structured["window_length"], "window_length", lower=1)
    window_gain = _scalar(
        structured["maximum_window_gain"],
        "maximum_window_gain", nonnegative=True,
    )
    if not window_gain < 1.0:
        raise ValueError("E3 registered path window does not contract")
    return nominal, radius, q, window_length, window_gain


def _constant_edge_block_envelope(*, q, persistent, transient, rate,
                                  window_length, window_gain):
    """Lift a constant edge envelope to one registered path block."""
    q = _scalar(q, "q", nonnegative=True)
    persistent = _scalar(
        persistent, "persistent", nonnegative=True)
    transient = _scalar(transient, "transient", nonnegative=True)
    rate = _scalar(rate, "rate", nonnegative=True)
    if not rate < 1.0:
        raise ValueError("the per-update disturbance rate must be below one")
    window_length = _integer(
        window_length, "window_length", lower=1)
    window_gain = _scalar(
        window_gain, "window_gain", nonnegative=True)
    if not window_gain < 1.0:
        raise ValueError("the registered path window must contract")
    edge_product = q ** window_length
    tolerance = 64.0 * np.finfo(float).eps * max(
        1.0, abs(edge_product), abs(window_gain))
    if edge_product > window_gain + tolerance:
        raise ValueError("the path-window certificate omits its edge product")
    gains = [q] * window_length
    persistent_bound = path_product_convolution(
        gains, [persistent] * window_length, initial=0.0
    )["terminal_bound"]
    transient_bound = path_product_convolution(
        gains,
        [transient * rate ** index for index in range(window_length)],
        initial=0.0,
    )["terminal_bound"]
    block_rate = rate ** window_length
    return {
        "window_length": window_length,
        "window_gain": window_gain,
        "edge_gain_product": edge_product,
        "persistent_forcing_constant": (
            0.0 if persistent_bound == 0.0
            else math.nextafter(float(persistent_bound), math.inf)
        ),
        "forcing_constant": (
            0.0 if transient_bound == 0.0
            else math.nextafter(float(transient_bound), math.inf)
        ),
        "rate": (
            0.0 if block_rate == 0.0
            else math.nextafter(float(block_rate), math.inf)
        ),
    }


def _coordinate_disturbance_radius(metric, disturbance_bound):
    eigenvalue = float(np.linalg.eigvalsh(metric)[0])
    if not eigenvalue > 0.0:
        raise ValueError("E5 metric must be positive definite")
    radius = disturbance_bound / math.sqrt(eigenvalue)
    return 0.0 if radius == 0.0 else math.nextafter(radius, math.inf)


def measure_live_e5(*, lifted_certificates, optimizer_measurements,
                    e2_measurements, entry_e3_measurement,
                    successor_e3_measurement, e4_measurement,
                    layer_index, certificate_box):
    """Construct one frozen E5 tube and open its held-out one-step image.

    The matrix-coordinate radii come from E3's certified Euclidean family
    ball. The disturbance-coordinate radii are the coordinate projection
    of E4's H-norm envelope. Only the lifted radius is sized from the
    registered fit window; the held-out step is not read until every center,
    radius, and budget has been fixed.
    """
    if certificate_box not in {"entry", "successor"}:
        raise ValueError("certificate_box must be entry or successor")
    if not (
        len(lifted_certificates)
        == len(optimizer_measurements)
        == len(e2_measurements)
        and lifted_certificates
    ):
        raise ValueError("live E5 source arrays must be nonempty and aligned")
    steps = [
        _layer_lifted_step(certificate, optimizer, e2, layer_index)
        for certificate, optimizer, e2 in zip(
            lifted_certificates, optimizer_measurements, e2_measurements)
    ]
    steps.sort(key=lambda row: row["step"])
    rules = campaign_design.SCALAR_DIAGNOSTIC_RULES["self_map"]
    fit_steps = [int(value) for value in rules["fit_steps"]]
    heldout_step = int(rules["heldout_step"])
    by_step = {row["step"]: row for row in steps}
    if len(by_step) != len(steps):
        raise ValueError("live E5 source steps are duplicated")
    missing = sorted(set([*fit_steps, heldout_step]) - set(by_step))
    if missing:
        raise ValueError(f"live E5 source window omits steps {missing}")
    fit = [by_step[step] for step in fit_steps]

    entry_e3 = _load(entry_e3_measurement)
    successor_e3 = _load(successor_e3_measurement)
    e4 = _load(e4_measurement)
    (
        entry_nominal,
        entry_matrix_radius,
        entry_q,
        entry_window_length,
        entry_window_gain,
    ) = _matrix_family_data(entry_e3)
    (
        successor_nominal,
        successor_matrix_radius,
        successor_q,
        successor_window_length,
        successor_window_gain,
    ) = _matrix_family_data(successor_e3)
    if entry_window_length != successor_window_length:
        raise ValueError("E5 source and successor path windows disagree")
    entry_metric = _array(
        entry_e3["construction"]["metric"], "entry_metric", ndim=2)
    successor_metric = _array(
        successor_e3["construction"]["metric"],
        "successor_metric", ndim=2)
    if entry_metric.shape != (2, 2) or successor_metric.shape != (2, 2):
        raise ValueError("E5 metrics must be two dimensional")
    source_metric = (
        entry_metric if certificate_box == "entry" else successor_metric)
    target_metric = successor_metric
    source_e3 = entry_e3 if certificate_box == "entry" else successor_e3
    source_structured = source_e3.get("structured_family")
    successor_structured = successor_e3.get("structured_family")
    if not isinstance(source_structured, dict) \
            or not isinstance(successor_structured, dict):
        raise ValueError("live E5 source artifacts omit structured families")
    source_corners = source_structured.get("corner_matrices")
    successor_edge_certificate = successor_structured.get(
        "exact_vertex_certificate")
    if not isinstance(source_corners, list) or not source_corners \
            or not isinstance(successor_edge_certificate, dict):
        raise ValueError("live E5 structured families omit exact edge data")
    if certificate_box == "entry":
        edge_certificate = cross_metric_family_certificate(
            source_corners, source_metric, target_metric)
        graph_edges = [
            {
                "edge_id": "entry_to_successor",
                "source": "entry",
                "target": "successor",
                "gain_upper": edge_certificate["gain_upper"],
            },
            {
                "edge_id": "successor_stay",
                "source": "successor",
                "target": "successor",
                "gain_upper": successor_edge_certificate["gain_upper"],
            },
        ]
    else:
        edge_certificate = successor_edge_certificate
        graph_edges = [{
            "edge_id": "successor_stay",
            "source": "successor",
            "target": "successor",
            "gain_upper": successor_edge_certificate["gain_upper"],
        }]
    window_length = entry_window_length
    graph = graph_window_certificate(graph_edges, window_length)
    q = _rational_float(edge_certificate["gain_upper"])
    window_gain = _rational_float(graph["maximum_window_gain"])
    edge_gains = {
        edge["edge_id"]: _rational_float(edge["gain_upper"])
        for edge in graph["edges"]
    }
    path_edge_ids = [
        value["edge_ids"] for value in graph["path_products"]
    ]
    path_gain_sequences = [
        [edge_gains[edge_id] for edge_id in edge_ids]
        for edge_ids in path_edge_ids
    ]
    e4_input = e4.get("closed_input")
    if not isinstance(e4_input, dict):
        raise ValueError("live E4 artifact omits its closed input")
    persistent = _scalar(
        e4_input["persistent_constant"],
        "persistent_constant", nonnegative=True)
    transient = _scalar(
        e4_input["geometric_constant"],
        "geometric_constant", nonnegative=True)
    disturbance_bound = _outward_maximum([persistent + transient])
    disturbance_coordinate_radius = _coordinate_disturbance_radius(
        target_metric, disturbance_bound)

    fit_norms = [
        _metric_norm(row["state_before"], source_metric)
        for row in fit
    ] + [
        _metric_norm(row["state_after"], target_metric)
        for row in fit
    ]
    path_disturbance_sequences = [
        [disturbance_bound] * window_length
        for _ in path_gain_sequences
    ]
    block_disturbance_bounds = [
        float(path_product_convolution(
            gains, disturbances, initial=0.0)["terminal_bound"])
        for gains, disturbances in zip(
            path_gain_sequences, path_disturbance_sequences)
    ]
    maximum_block_disturbance = _outward_maximum(
        block_disturbance_bounds)
    invariant_requirement = (
        maximum_block_disturbance / (1.0 - window_gain)
    )
    base_radius = _outward_maximum([*fit_norms, invariant_requirement])
    lifted_radius = math.nextafter(
        float(rules["entry_radius_multiplier"]) * base_radius,
        math.inf,
    )
    next_lifted_radius = lifted_radius

    source_nominal = (
        entry_nominal if certificate_box == "entry"
        else successor_nominal
    )
    source_matrix_radius = (
        entry_matrix_radius if certificate_box == "entry"
        else successor_matrix_radius
    )
    source_center = np.concatenate([
        source_nominal.reshape(-1), np.zeros(2, dtype=float),
    ])
    successor_center = np.concatenate([
        successor_nominal.reshape(-1), np.zeros(2, dtype=float),
    ])
    auxiliary_radii = np.asarray([
        *([source_matrix_radius] * 4),
        disturbance_coordinate_radius,
        disturbance_coordinate_radius,
    ])
    next_auxiliary_radii = np.asarray([
        *([successor_matrix_radius] * 4),
        disturbance_coordinate_radius,
        disturbance_coordinate_radius,
    ])
    # These coordinates are external inputs to the reduced Y recurrence on
    # the declared interval family. Their successor-family radii are the
    # center-error bounds, with zero Y/zeta derivatives.
    center_errors = next_auxiliary_radii.copy()
    zero_vector = np.zeros_like(center_errors)
    zero_matrix = np.zeros(
        (len(center_errors), len(center_errors)), dtype=float)

    # Freeze all preceding objects before opening the held-out transition.
    heldout = by_step[heldout_step]
    heldout_before = _metric_norm(heldout["state_before"], source_metric)
    heldout_after = _metric_norm(heldout["state_after"], target_metric)
    heldout_auxiliary = np.concatenate([
        heldout["matrix"].reshape(-1), heldout["disturbance"],
    ])
    source_margins = auxiliary_radii - np.abs(
        heldout_auxiliary - source_center)
    successor_margins = next_auxiliary_radii - np.abs(
        heldout_auxiliary - successor_center)
    initial_margin = min(
        lifted_radius - heldout_before,
        float(np.min(source_margins)),
    )
    image_margins = np.concatenate([
        [next_lifted_radius - heldout_after], successor_margins,
    ])
    entry_ratio = _measurement_values(entry_e3)[
        "primitive_box_enclosure_ratio"]
    successor_ratio = _measurement_values(successor_e3)[
        "primitive_box_enclosure_ratio"]
    family_ratio = max(entry_ratio, successor_ratio)
    input_value = {
        "q": q,
        "lifted_radius": lifted_radius,
        "disturbance_bound": disturbance_bound,
        "next_lifted_radius": next_lifted_radius,
        "center_errors": center_errors.tolist(),
        "lifted_lipschitz": zero_vector.tolist(),
        "auxiliary_lipschitz": zero_matrix.tolist(),
        "auxiliary_radii": auxiliary_radii.tolist(),
        "next_auxiliary_radii": next_auxiliary_radii.tolist(),
        "initial_membership_margin": initial_margin,
        "heldout_image_margins": image_margins.tolist(),
        "family_box_enclosure_ratio": family_ratio,
        "path_gains": path_gain_sequences,
        "path_disturbance_bounds": path_disturbance_sequences,
        "path_lifted_radii": [
            [lifted_radius] * (window_length + 1)
            for _ in path_gain_sequences
        ],
        "certified_path_window_gain": window_gain,
    }
    result = measure_e5(input_value)
    result.update({
        "schema_version": "pldr-live-e5-measurement-v3",
        "layer_index": int(layer_index),
        "certificate_box": certificate_box,
        "fit_steps": fit_steps,
        "heldout_step": heldout_step,
        "heldout_opened_after_fit": True,
        "auxiliary_coordinate_names": [
            "matrix_00", "matrix_01", "matrix_10", "matrix_11",
            "disturbance_0", "disturbance_1",
        ],
        "source_auxiliary_center": source_center.tolist(),
        "successor_auxiliary_center": successor_center.tolist(),
        "source_to_target_edge_certificate": edge_certificate,
        "registered_graph_window_certificate": graph,
        "registered_path_edge_ids": path_edge_ids,
        "self_map_rule": rules,
        "closed_input": input_value,
    })
    return result


def measure_live_e6(*, lifted_certificate, optimizer_measurement,
                    e2_measurements, e3_measurements, e4_measurements,
                    e0_trajectory, certificate_time):
    """Build the conservative registered finite E6 schedule envelope.

    The complete E0 direct upper, rather than a sampled projected Jacobian,
    is retained as the cover floor. This is intentionally conservative: a
    failure to enter remains a complete NOT_CONFIRMED outcome.
    """
    if not (
        len(e2_measurements)
        == len(e3_measurements)
        == len(e4_measurements)
        and e2_measurements
    ):
        raise ValueError("live E6 layer artifact arrays must be aligned")
    certificate = _load(lifted_certificate)
    optimizer = _load(optimizer_measurement)
    e2_rows = [_load(value) for value in e2_measurements]
    e3_rows = [_load(value) for value in e3_measurements]
    e4_rows = [_load(value) for value in e4_measurements]
    certificate_time = _integer(certificate_time, "certificate_time")
    if int(optimizer["global_optimizer_step"]) != certificate_time:
        raise ValueError("E6 optimizer artifact is at the wrong checkpoint")
    layer_indices = [int(value["layer_index"]) for value in e2_rows]
    if len(set(layer_indices)) != len(layer_indices):
        raise ValueError("E6 criterion layers are duplicated")

    layer_steps = [
        _layer_lifted_step(
            certificate, optimizer, e2, layer_index)
        for e2, layer_index in zip(e2_rows, layer_indices)
    ]
    metrics = [
        _array(value["construction"]["metric"], "metric", ndim=2)
        for value in e3_rows
    ]
    if any(metric.shape != (2, 2) for metric in metrics):
        raise ValueError("E6 layer metrics must be two dimensional")
    family_data = [_matrix_family_data(value) for value in e3_rows]
    q_values = [value[2] for value in family_data]
    window_lengths = {value[3] for value in family_data}
    if len(window_lengths) != 1:
        raise ValueError("E6 criterion layers use different path windows")
    window_length = window_lengths.pop()
    window_gain = max(value[4] for value in family_data)
    q = max(q_values)
    metric_lower = min(
        float(np.linalg.eigvalsh(metric)[0]) for metric in metrics)
    initial_h_norm = max(
        _metric_norm(step["state_before"], metric)
        for step, metric in zip(layer_steps, metrics)
    )

    persistent_values = []
    transient_values = []
    rates = []
    for value in e4_rows:
        closed = value.get("closed_input")
        if not isinstance(closed, dict):
            raise ValueError("live E4 artifact omits its closed input")
        persistent_values.append(_scalar(
            closed["persistent_constant"],
            "persistent_constant", nonnegative=True))
        transient_values.append(_scalar(
            closed["geometric_constant"],
            "geometric_constant", nonnegative=True))
        rates.append(_scalar(
            closed["geometric_rate"],
            "geometric_rate", nonnegative=True))
    persistent = _outward_maximum(persistent_values)
    transient = _outward_maximum(transient_values)
    rate = max(rates)
    if not rate < 1.0:
        raise ValueError("E6 disturbance rate must be below one")
    block_envelope = _constant_edge_block_envelope(
        q=q,
        persistent=persistent,
        transient=transient,
        rate=rate,
        window_length=window_length,
        window_gain=window_gain,
    )

    direct_rows = [_load(value) for value in e0_trajectory]
    direct_by_step = {}
    for value in direct_rows:
        step = int(value["checkpoint_step"])
        if step in direct_by_step:
            raise ValueError("E6 E0 trajectory contains duplicate checkpoints")
        direct_by_step[step] = _measurement_values(value)[
            "direct_row_map_upper"]
    finite_rule = campaign_design.SCALAR_DIAGNOSTIC_RULES["finite_entry"]
    terminal = int(finite_rule["terminal_step"])
    trajectory_steps = list(range(certificate_time, terminal + 1, 1000))
    missing = sorted(set(trajectory_steps) - set(direct_by_step))
    if missing:
        raise ValueError(f"E6 E0 trajectory omits checkpoints {missing}")
    criterion = float(finite_rule["criterion"])
    sustained_window = int(finite_rule["sustained_window_checkpoints"])
    observed_entry = terminal + 1
    for index, step in enumerate(trajectory_steps):
        remaining = trajectory_steps[index:]
        if (
            len(remaining) >= sustained_window
            and all(direct_by_step[item] < criterion for item in remaining)
        ):
            observed_entry = step
            break
    sustained = float(observed_entry <= terminal)
    direct = direct_by_step[certificate_time]
    cover_floor = _outward_maximum(
        direct_by_step[step] for step in trajectory_steps)
    schedule_length = terminal - certificate_time
    disturbances = [
        math.nextafter(
            persistent + transient * rate ** index, math.inf)
        for index in range(schedule_length)
    ]
    input_value = {
        "certificate_time": certificate_time,
        "q": q,
        "r": rate,
        "initial_h_norm": initial_h_norm,
        "forcing_constant": transient,
        "persistent_forcing_constant": persistent,
        "path_window_length": window_length,
        "path_window_gain": window_gain,
        "block_forcing_constant": block_envelope["forcing_constant"],
        "block_persistent_forcing_constant": block_envelope[
            "persistent_forcing_constant"],
        "block_rate": block_envelope["rate"],
        "metric_lower": metric_lower,
        "cover_floor": cover_floor,
        "cover_transient_constant": 0.0,
        "cover_rate": float(finite_rule["geometric_cover_rate"]),
        "criterion": criterion,
        "observed_entry_step": observed_entry,
        "direct_row_map_upper": direct,
        "sustained_entry_indicator": sustained,
        "schedule_gains": [q] * schedule_length,
        "schedule_disturbance_bounds": disturbances,
    }
    result = measure_e6(input_value)
    result.update({
        "schema_version": "pldr-live-e6-measurement-v2",
        "criterion_layer_indices": layer_indices,
        "trajectory_checkpoint_steps": trajectory_steps,
        "complete_e0_cover_floor_used": True,
        "block_envelope": block_envelope,
        "finite_entry_rule": finite_rule,
        "closed_input": input_value,
    })
    return result


MEASURERS = {
    "E2": measure_e2,
    "E3": measure_e3,
    "E4": measure_e4,
    "E5": measure_e5,
    "E6": measure_e6,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=tuple(MEASURERS), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = MEASURERS[args.protocol](_load(args.input))
    result["schema_version"] = "pldr-direct-certificate-measurement-v3"
    Path(args.output).write_text(strict_dumps(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
