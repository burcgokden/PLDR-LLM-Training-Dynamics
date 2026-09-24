"""Exact and adversarial Q0 fixtures for the validated certificate path."""

from __future__ import annotations

from fractions import Fraction

from assemble_confirmation import finite_scalar_projection
from check_certificate import check_small_certificate
from row_domain import disconnected_ball_counterexample
from small_pldr_certificate import build_small_certificate
from tail_comparison import finite_trace_cannot_prove_infinite_tail
from validated_interval import RationalInterval, as_fraction, exp_enclosure
from validated_linear_algebra import (
    residual_aware_lyapunov_certificate,
    spectral_norm_upper,
)
from validated_row_map import certify_row_map


def validated_certificate_fixture():
    exp_lower, exp_upper = exp_enclosure(Fraction(1))
    exp_known = exp_lower > Fraction(271828, 100000) and (
        exp_upper < Fraction(271829, 100000))
    specification = {
        "width": 2,
        "residual_units": [{
            "glu_blocks": [{
                "w1": [["1/8", "0"], ["0", "1/8"]],
                "b1": ["0", "0"],
                "w2": [["1/8", "0"], ["0", "1/8"]],
                "b2": ["1", "1"],
                "w3": [["1/8", "0"], ["0", "1/8"]],
                "b3": ["0", "0"],
            }],
            "gamma": ["1", "1"],
            "beta": ["0", "0"],
            "epsilon": "1",
        }],
    }
    row_map = certify_row_map(specification, [
        RationalInterval("-1/4", "1/4"),
        RationalInterval("-1/4", "1/4"),
    ])
    metric = residual_aware_lyapunov_certificate(
        [[Fraction(1, 2), 0], [0, Fraction(1, 3)]],
        [[1, 0], [0, 1]],
        [[RationalInterval("49/100", "51/100"), RationalInterval(0, 0)],
         [RationalInterval(0, 0), RationalInterval("32/100", "34/100")]],
    )
    small = build_small_certificate()
    replay = check_small_certificate(small)
    checks = {
        "rational_exp_enclosed": exp_known,
        "analytic_row_map_bound_positive":
            as_fraction(row_map["jacobian_operator_upper"]) > 0,
        "analytic_derivative_modulus_positive":
            as_fraction(row_map["derivative_lipschitz_upper"]) > 0,
        "residual_aware_gain_strict":
            as_fraction(metric["robust_q_upper"]) < 1,
        "small_complete_certificate_replayed": replay["status"] == "PASS",
        "small_complete_certificate_enters":
            bool(small["permanent_through_terminal"]),
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "small_entry_relative_update": small["entry_relative_update"],
        "small_graph_sha256": small["graph"]["graph_sha256"],
    }


def adversarial_fixture():
    exact = RationalInterval(
        Fraction(10_000_000_000_000_001, 10_000_000_000_000_000),
        Fraction(10_000_000_000_000_001, 10_000_000_000_000_000),
    )
    shared_float_estimate = Fraction.from_float(1.0)
    floating_agreement_is_not_enclosure = not exact.contains(
        shared_float_estimate)
    induced_upper = spectral_norm_upper([[1, 1]])
    ordinary_claim_underestimates = induced_upper > 1
    exact_fraction = exact.lower
    projected_upper = finite_scalar_projection(exact_fraction, "upper")
    projected_lower = finite_scalar_projection(exact_fraction, "lower")
    directed_projection_encloses = (
        Fraction.from_float(projected_lower) <= exact_fraction
        <= Fraction.from_float(projected_upper)
    )
    inexact_point_rejected = False
    try:
        finite_scalar_projection(exact_fraction, "point")
    except ValueError:
        inexact_point_rejected = True
    residual_rejected = False
    try:
        residual_aware_lyapunov_certificate(
            [[1, 0], [0, 1]], [[1, 0], [0, 1]],
            [[RationalInterval(1, 1), RationalInterval(0, 0)],
             [RationalInterval(0, 0), RationalInterval(1, 1)]],
        )
    except ValueError:
        residual_rejected = True
    disconnected = disconnected_ball_counterexample()
    finite_trace = finite_trace_cannot_prove_infinite_tail(
        [1, "1/2", "1/4"])
    checks = {
        "shared_float_error_exposed": floating_agreement_is_not_enclosure,
        "directed_float_projection_encloses": directed_projection_encloses,
        "inexact_point_projection_rejected": inexact_point_rejected,
        "ordinary_spectral_underestimate_exposed": ordinary_claim_underestimates,
        "residual_consumes_nominal_margin": residual_rejected,
        "disconnected_segment_rejected":
            not disconnected["global_segment_valid"],
        "finite_trace_not_promoted":
            not finite_trace["infinite_conclusion_identified"],
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
    }
