"""End-to-end rational width-two certificate used by mandatory Q0."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path

from certificate_graph import CertificateGraph, CertificateNode, digest_file
from row_domain import interval_hull, tile_interval_hull
from tail_comparison import (
    construct_positive_certificate,
    finite_entry_horizon,
)
from validated_interval import RationalInterval, as_fraction, rational_object
from validated_linear_algebra import residual_aware_lyapunov_certificate
from validated_row_map import certify_row_map


HERE = Path(__file__).resolve().parent


def _constructor_digest(file_name):
    return digest_file(HERE / file_name)


def rational_width_two_specification():
    return {
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


def exact_adamw_fixture():
    beta1 = Fraction(9, 10)
    beta2 = Fraction(99, 100)
    gradient = Fraction(1, 5)
    learning_rate = Fraction(1, 100)
    weight_decay = Fraction(1, 100)
    epsilon = Fraction(1, 1000)
    theta = Fraction(1)
    first = (1 - beta1) * gradient
    second = (1 - beta2) * gradient ** 2
    first_hat = first / (1 - beta1)
    second_hat = second / (1 - beta2)
    root = Fraction(1, 5)
    if root * root != second_hat:
        raise ArithmeticError("small AdamW fixture lost its exact square root")
    theta_next = ((1 - learning_rate * weight_decay) * theta
                  - learning_rate * first_hat / (root + epsilon))
    return {
        "operation_order": [
            "raw_gradient", "coordinatewise_clip", "first_moment",
            "second_moment", "bias_correction", "adaptive_preconditioner",
            "decoupled_decay_and_adaptive_parameter_update",
            "registered_intervention", "row_cover_motion",
        ],
        "global_optimizer_update": 1,
        "scheduler_index_applied": 0,
        "theta_before": rational_object(theta),
        "raw_and_clipped_gradient": rational_object(gradient),
        "first_moment_after": rational_object(first),
        "second_moment_after": rational_object(second),
        "bias_corrected_first": rational_object(first_hat),
        "bias_corrected_second": rational_object(second_hat),
        "theta_after": rational_object(theta_next),
        "intervention": rational_object(Fraction(0)),
        "row_cover_motion": rational_object(Fraction(0)),
        "derivation": "exact_rational_clipped_adamw_update",
    }


def build_small_certificate():
    specification = rational_width_two_specification()
    rows = ((Fraction(-1, 4), Fraction(-1, 4)),
            (Fraction(1, 4), Fraction(1, 4)))
    hull = interval_hull(rows)
    tiling = tile_interval_hull(hull, (4, 4), maximum_boxes=100)
    row_box = [RationalInterval(Fraction(-1, 4), Fraction(1, 4))] * 2
    row_map = certify_row_map(specification, row_box)
    adamw = exact_adamw_fixture()

    nominal = [[Fraction(1, 2), 0], [0, Fraction(1, 3)]]
    metric = [[1, 0], [0, 1]]
    family = [
        [RationalInterval(Fraction(49, 100), Fraction(51, 100)),
         RationalInterval(0, 0)],
        [RationalInterval(0, 0),
         RationalInterval(Fraction(32, 100), Fraction(34, 100))],
    ]
    lyapunov = residual_aware_lyapunov_certificate(
        nominal, metric, family)
    comparison = construct_positive_certificate(
        initial=[Fraction(1, 4), Fraction(1, 8), Fraction(1, 16)],
        matrix=[
            [Fraction(1, 2), Fraction(1, 16), 0],
            [0, Fraction(1, 3), Fraction(1, 16)],
            [0, 0, Fraction(1, 4)],
        ],
        forcing=[Fraction(1, 100), Fraction(1, 200), Fraction(1, 400)],
        rho=Fraction(1, 2),
        weights=[1, 1, 1],
        kappa=Fraction(9, 16),
        persistent=[
            Fraction(1, 1000), Fraction(1, 2000), Fraction(1, 4000),
        ],
        horizon=16,
    )
    robust_q = as_fraction(lyapunov["robust_q_upper"])
    gains = [robust_q] * 16
    disturbances = [
        Fraction(1, 1000) + Fraction(1, 100) * Fraction(1, 2) ** n
        for n in range(16)
    ]
    cover = [Fraction(1, 20) * Fraction(1, 2) ** n
             for n in range(17)]
    horizon = finite_entry_horizon(
        gains, disturbances,
        initial=as_fraction(row_map["jacobian_operator_upper"]),
        cover_remainders=cover,
        metric_projection=1,
        criterion=Fraction(1, 2),
    )
    if not horizon["permanent_through_terminal"]:
        raise ArithmeticError("small certificate failed to enter its criterion")

    constructor_digests = {
        name: _constructor_digest(name)
        for name in (
            "small_pldr_certificate.py",
            "row_domain.py",
            "validated_row_map.py",
            "validated_linear_algebra.py",
            "tail_comparison.py",
        )
    }
    graph = CertificateGraph()
    graph.add(CertificateNode(
        "rational_roots", "root_artifact",
        {"specification": specification,
         "rows": [[rational_object(entry) for entry in row] for row in rows]},
        "exact rational data", "width-two fixture",
        str(Path(__file__).name),
        constructor_digests["small_pldr_certificate.py"], (), {},
    ))
    graph.add(CertificateNode(
        "convex_row_domain", "exact_recurrence", tiling,
        "physical row units", "convex interval hull",
        "row_domain.tile_interval_hull",
        constructor_digests["row_domain.py"], ("rational_roots",), {},
    ))
    graph.add(CertificateNode(
        "validated_row_map", "rational_interval_evaluation", row_map,
        "output row per input row", "complete convex row domain",
        "validated_row_map.certify_row_map",
        constructor_digests["validated_row_map.py"],
        ("rational_roots", "convex_row_domain"), {},
    ))
    graph.add(CertificateNode(
        "adamw_transport", "exact_recurrence", adamw,
        "parameter units", "one registered optimizer update",
        "small_pldr_certificate.exact_adamw_fixture",
        constructor_digests["small_pldr_certificate.py"],
        ("rational_roots",), {},
    ))
    graph.add(CertificateNode(
        "common_metric", "exact_ldl_witness", lyapunov,
        "scaled lifted-state units", "complete lifted interval family",
        "validated_linear_algebra.residual_aware_lyapunov_certificate",
        constructor_digests["validated_linear_algebra.py"],
        ("adamw_transport",), {},
    ))
    graph.add(CertificateNode(
        "positive_tail", "positive_comparison", comparison,
        "registered auxiliary scales", "time-uniform auxiliary box",
        "tail_comparison.construct_positive_certificate",
        constructor_digests["tail_comparison.py"], ("common_metric",), {},
    ))
    graph.add(CertificateNode(
        "finite_direct_entry", "finite_horizon", horizon,
        "global optimizer updates", "registered finite schedule",
        "tail_comparison.finite_entry_horizon",
        constructor_digests["tail_comparison.py"],
        ("validated_row_map", "positive_tail"), {},
    ))
    graph_object = graph.to_object(outputs=("finite_direct_entry",))
    return {
        "schema_version": "pldr-small-complete-certificate-v2",
        "scope": (
            "width-two interface instantiation; not a production-model "
            "measurement"
        ),
        "graph": graph_object,
        "entry_relative_update": horizon["entry_relative_update"],
        "permanent_through_terminal": horizon["permanent_through_terminal"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = build_small_certificate()
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
