"""Exact primitive-to-normal-stability and lifted-AdamW certificates."""

from __future__ import annotations

from fractions import Fraction
import itertools

from validated_interval import RationalInterval, as_fraction, rational_object
from validated_linear_algebra import (
    convex_family_lyapunov_certificate,
    solve_discrete_lyapunov_exact,
)


NORMAL_FORCE_TERMS = (
    "rope_frequency_remainder",
    "contextual_covariance",
    "finite_batch",
    "nonstationarity",
    "intervention",
)


def _nonnegative(value, label):
    value = as_fraction(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _positive(value, label):
    value = _nonnegative(value, label)
    if value == 0:
        raise ValueError(f"{label} must be strictly positive")
    return value


def _object_matrix(value):
    return [[rational_object(entry) for entry in row] for row in value]


def construct_normal_response(primitives):
    """Derive the PLDR normal-response edge and closed force envelope."""

    required = {
        "loss_hessian_lower", "loss_hessian_upper",
        "downstream_normal_singular_lower", "downstream_normal_norm_upper",
        "full_hessian_residual_upper", "branch_residual_upper",
        "preconditioner_motion_upper", "normal_force_terms",
        "normal_hessian_lipschitz_upper", "normal_tube_radius",
    }
    if not isinstance(primitives, dict) or set(primitives) != required:
        raise ValueError("normal-response primitives have missing or unknown fields")
    h_lower = _positive(primitives["loss_hessian_lower"], "loss Hessian lower")
    h_upper = _positive(primitives["loss_hessian_upper"], "loss Hessian upper")
    if h_lower > h_upper:
        raise ValueError("loss Hessian bounds are reversed")
    sigma_lower = _positive(
        primitives["downstream_normal_singular_lower"],
        "downstream normal singular lower")
    downstream_upper = _positive(
        primitives["downstream_normal_norm_upper"],
        "downstream normal norm upper")
    if sigma_lower > downstream_upper:
        raise ValueError("downstream normal bounds are reversed")
    hessian_residual = _nonnegative(
        primitives["full_hessian_residual_upper"],
        "full Hessian residual")
    branch_residual = _nonnegative(
        primitives["branch_residual_upper"], "branch residual")
    preconditioner_motion = _nonnegative(
        primitives["preconditioner_motion_upper"],
        "preconditioner motion")
    lower_edge = (
        h_lower * sigma_lower ** 2
        - hessian_residual - branch_residual
    )
    if lower_edge <= 0:
        raise ValueError("the derived normal-response lower edge is not positive")
    upper_edge = (
        h_upper * downstream_upper ** 2
        + hessian_residual + branch_residual
    )
    terms = primitives["normal_force_terms"]
    if not isinstance(terms, dict) or set(terms) != set(NORMAL_FORCE_TERMS):
        raise ValueError("the normal-force ledger is incomplete")
    force_terms = {
        name: _nonnegative(terms[name], f"normal force term {name}")
        for name in NORMAL_FORCE_TERMS
    }
    base_force = sum(force_terms.values(), Fraction(0))
    hessian_lipschitz = _nonnegative(
        primitives["normal_hessian_lipschitz_upper"],
        "normal Hessian Lipschitz upper")
    radius = _nonnegative(
        primitives["normal_tube_radius"], "normal tube radius")
    nonlinear_remainder = hessian_lipschitz * radius ** 2 / 2
    preconditioner_force = preconditioner_motion * radius
    return {
        "schema_version": "pldr-normal-response-certificate-v1",
        "normal_response_lower": rational_object(lower_edge),
        "normal_response_upper": rational_object(upper_edge),
        "force_terms": {
            name: rational_object(value) for name, value in force_terms.items()},
        "normal_force_upper": rational_object(base_force),
        "preconditioner_motion_force_upper": rational_object(
            preconditioner_force),
        "nonlinear_remainder_upper": rational_object(nonlinear_remainder),
        "complete_force_upper": rational_object(
            base_force + preconditioner_force + nonlinear_remainder),
        "normal_tube_radius": rational_object(radius),
        "conditions": {
            "loss_zero_sum_edge_positive": True,
            "downstream_normal_observable": True,
            "residuals_charged": True,
            "normal_force_ledger_complete": True,
            "normal_response_strictly_positive": True,
        },
        "derivation": (
            "loss_hessian_edge_times_downstream_normal_singular_edge_"
            "minus_full_hessian_and_branch_residuals_with_"
            "preconditioner_motion_charged_to_forcing"),
    }


def lifted_matrix(alpha, beta, eigenvalue, decay):
    """Return the exact scalar-mode lifted AdamW matrix."""

    alpha = as_fraction(alpha)
    beta = as_fraction(beta)
    eigenvalue = as_fraction(eigenvalue)
    decay = as_fraction(decay)
    return (
        (
            1 - decay - alpha * (1 - beta) * eigenvalue,
            -alpha * beta,
        ),
        ((1 - beta) * eigenvalue, beta),
    )


def jury_margins(alpha, beta, eigenvalue, decay):
    """Return the exact three strict Jury margins."""

    alpha = as_fraction(alpha)
    beta = as_fraction(beta)
    eigenvalue = as_fraction(eigenvalue)
    decay = as_fraction(decay)
    determinant = beta * (1 - decay)
    return (
        1 - determinant,
        (1 - beta) * (decay + alpha * eigenvalue),
        2 * (1 + beta) - decay * (1 + beta)
        - alpha * (1 - beta) * eigenvalue,
    )


def certify_lifted_family(parameter_box):
    """Construct an exact common metric for the scalar parameter box."""

    required = {"alpha", "beta", "normal_eigenvalue", "decay"}
    if not isinstance(parameter_box, dict) or set(parameter_box) != required:
        raise ValueError("lifted-family box has missing or unknown fields")
    alpha = RationalInterval.from_object(parameter_box["alpha"])
    beta = RationalInterval.from_object(parameter_box["beta"])
    eigenvalue = RationalInterval.from_object(
        parameter_box["normal_eigenvalue"])
    decay = RationalInterval.from_object(parameter_box["decay"])
    if alpha.lower <= 0:
        raise ValueError("alpha must be strictly positive on the box")
    if beta.lower < 0 or beta.upper >= 1:
        raise ValueError("beta must lie in [0,1) on the box")
    if eigenvalue.lower <= 0:
        raise ValueError("the normal eigenvalue must be strictly positive")
    if decay.lower < 0 or decay.upper >= 1:
        raise ValueError("decay must lie in [0,1) on the box")

    vertices = []
    for values in itertools.product(
        (alpha.lower, alpha.upper),
        (beta.lower, beta.upper),
        (eigenvalue.lower, eigenvalue.upper),
        (decay.lower, decay.upper),
    ):
        matrix = lifted_matrix(*values)
        margins = jury_margins(*values)
        vertices.append((values, matrix, margins))
    minimum_margins = tuple(
        min(value[2][index] for value in vertices) for index in range(3))
    if any(value <= 0 for value in minimum_margins):
        raise ValueError("the complete lifted family fails a strict Jury margin")

    center_values = tuple(interval.midpoint for interval in (
        alpha, beta, eigenvalue, decay))
    nominal = lifted_matrix(*center_values)
    metric = solve_discrete_lyapunov_exact(nominal)
    common = convex_family_lyapunov_certificate(
        tuple(vertex[1] for vertex in vertices), metric)
    return {
        "schema_version": "pldr-lifted-adamw-family-v1",
        "parameter_box": {
            name: interval.to_object() for name, interval in (
                ("alpha", alpha), ("beta", beta),
                ("normal_eigenvalue", eigenvalue), ("decay", decay))
        },
        "nominal_parameters": {
            name: rational_object(value) for name, value in zip(
                ("alpha", "beta", "normal_eigenvalue", "decay"),
                center_values)
        },
        "nominal_matrix": _object_matrix(nominal),
        "metric": _object_matrix(metric),
        "minimum_jury_margins": [
            rational_object(value) for value in minimum_margins],
        "common_metric_certificate": common,
        "vertex_count": len(vertices),
        "family_rule": (
            "multiaffine_parameter_box_image_lies_in_the_convex_hull_"
            "of_its_correlated_vertices"),
        "decision": "CERTIFIED",
    }


def ordered_product_convolution(gains, forcings, initial):
    """Compute the exact nonautonomous product-convolution envelope."""

    if not isinstance(gains, (list, tuple)) or not isinstance(
        forcings, (list, tuple),
    ) or len(gains) != len(forcings):
        raise ValueError("gains and forcings must be equal-length sequences")
    current = _nonnegative(initial, "initial norm")
    envelope = [current]
    cumulative_gain = Fraction(1)
    for gain_value, force_value in zip(gains, forcings):
        gain = _nonnegative(gain_value, "gain")
        force = _nonnegative(force_value, "forcing")
        current = gain * current + force
        cumulative_gain *= gain
        envelope.append(current)
    return {
        "schema_version": "pldr-ordered-product-convolution-v2",
        "envelope": [rational_object(value) for value in envelope],
        "final_upper": rational_object(current),
        "cumulative_gain": rational_object(cumulative_gain),
        "path_contracts_without_force": cumulative_gain < 1,
        "individual_contraction_not_required": True,
        "edge_count": len(gains),
        "derivation": (
            "exact_temporal_recurrence_y_next_equals_q_y_plus_w_"
            "with_no_per_edge_contraction_assumption"
        ),
    }
