#!/usr/bin/env python3
"""Execute deterministic qualification experiments for the RG implementation.

The stochastic section tests the finite-variance log-gain fixed point on
synthetic laws.  It does not stand in for a PLDR trajectory experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import seal_record  # noqa: E402
from row_rgmap.core import (  # noqa: E402
    Edge,
    beta,
    block_edges,
    clean_correlation_scale,
    clean_flow,
    compose,
    normalize,
)
from row_rgmap.statistics import (  # noqa: E402
    normal_ks_distance,
    standardized_moments,
    wasserstein_1d,
)


SEED = 20260901
BLOCK_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _relative(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1e-300)


def algebra_qualification(generator: np.random.Generator) -> dict:
    maximum_action_residual = 0.0
    maximum_associativity_residual = 0.0
    maximum_gauge_residual = 0.0
    trials = 20000
    for _ in range(trials):
        raw = generator.normal(size=8)
        first = Edge(math.exp(0.05 * raw[0]), math.exp(raw[1] - 5.0))
        second = Edge(math.exp(0.05 * raw[2]), math.exp(raw[3] - 5.0))
        third = Edge(math.exp(0.05 * raw[4]), math.exp(raw[5] - 5.0))
        energy = math.exp(raw[6])
        sequential = second.act(first.act(energy))
        blocked = compose(second, first).act(energy)
        maximum_action_residual = max(
            maximum_action_residual, _relative(sequential, blocked)
        )
        left = compose(third, compose(second, first))
        right = compose(compose(third, second), first)
        maximum_associativity_residual = max(
            maximum_associativity_residual,
            _relative(left.gain, right.gain),
            _relative(left.source, right.source),
        )
        gauges = np.exp(generator.normal(size=3))
        normalized_composition = compose(
            normalize(second, gauges[1], gauges[2]),
            normalize(first, gauges[0], gauges[1]),
        )
        normalized_block = normalize(
            compose(second, first), gauges[0], gauges[2]
        )
        maximum_gauge_residual = max(
            maximum_gauge_residual,
            _relative(normalized_composition.gain, normalized_block.gain),
            _relative(normalized_composition.source, normalized_block.source),
        )

    edges = [
        Edge(math.exp(value), source)
        for value, source in zip(
            generator.normal(-2e-4, 0.01, size=4096),
            generator.exponential(1e-8, size=4096),
        )
    ]
    initial = 0.7
    sequential = initial
    for edge in edges:
        sequential = edge.act(sequential)
    blocked = block_edges(edges).act(initial)
    long_path_residual = _relative(sequential, blocked)
    return {
        "trials": trials,
        "long_path_edges": len(edges),
        "maximum_action_relative_residual": maximum_action_residual,
        "maximum_associativity_relative_residual": maximum_associativity_residual,
        "maximum_gauge_covariance_relative_residual": maximum_gauge_residual,
        "long_path_relative_residual": long_path_residual,
        "qualified": max(
            maximum_action_residual,
            maximum_associativity_residual,
            maximum_gauge_residual,
            long_path_residual,
        ) < 5e-13,
    }


def clean_critical_qualification() -> dict:
    distances = np.logspace(-5, -1, 17)
    gains = 1.0 - distances
    scales = np.asarray(
        [clean_correlation_scale(float(gain)) for gain in gains]
    )
    design = np.column_stack([np.ones(len(distances)), np.log(distances)])
    intercept, slope = np.linalg.lstsq(design, np.log(scales), rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    exponent = -float(slope)

    epsilon = 1e-6
    beta_rows = []
    for gain in (0.2, 0.5, 0.9, 0.99, 1.01, 1.5):
        numerical = (clean_flow(gain, math.exp(epsilon)) - gain) / epsilon
        exact = beta(gain)
        beta_rows.append(
            {
                "gain": gain,
                "analytic_beta": exact,
                "finite_difference_beta": numerical,
                "relative_residual": _relative(numerical, exact),
            }
        )
    return {
        "distance_grid": distances.tolist(),
        "fitted_parallel_exponent": exponent,
        "log_fit_intercept": float(intercept),
        "log_fit_root_mean_square_residual": float(
            np.sqrt(np.mean((np.log(scales) - predicted) ** 2))
        ),
        "beta_checks": beta_rows,
        "maximum_beta_relative_residual": max(
            row["relative_residual"] for row in beta_rows
        ),
        "qualified": (
            abs(exponent - 1.0) < 0.01
            and max(row["relative_residual"] for row in beta_rows) < 2e-5
        ),
    }


def _draw_law(
    generator: np.random.Generator,
    name: str,
    replicas: int,
    width: int,
    sigma: float,
) -> np.ndarray:
    if name == "gaussian":
        return generator.normal(0.0, sigma, size=(replicas, width))
    if name == "laplace":
        return generator.laplace(
            0.0, sigma / math.sqrt(2.0), size=(replicas, width)
        )
    if name == "uniform":
        bound = math.sqrt(3.0) * sigma
        return generator.uniform(-bound, bound, size=(replicas, width))
    if name == "rademacher":
        signs = generator.integers(0, 2, size=(replicas, width), dtype=np.int8)
        return sigma * (2.0 * signs.astype(np.float64) - 1.0)
    raise ValueError(f"unknown law {name}")


def fluctuation_qualification(generator: np.random.Generator) -> dict:
    replicas = 30000
    sigma = 0.04
    laws = ("gaussian", "laplace", "uniform", "rademacher")
    systems = []
    final_samples = {}
    for name in laws:
        increments = _draw_law(
            generator, name, replicas, BLOCK_SIZES[-1], sigma
        )
        levels = []
        for block_size in BLOCK_SIZES:
            normalized = (
                increments[:, :block_size].sum(axis=1)
                / (sigma * math.sqrt(block_size))
            )
            moments = standardized_moments(normalized)
            levels.append(
                {
                    "block_size": block_size,
                    **moments,
                    "normal_ks_distance": normal_ks_distance(normalized),
                }
            )
            if block_size == BLOCK_SIZES[-1]:
                final_samples[name] = np.sort(normalized)
        systems.append({"law": name, "levels": levels})

    cross = []
    for left_index, left in enumerate(laws):
        for right in laws[left_index + 1 :]:
            cross.append(
                {
                    "left": left,
                    "right": right,
                    "scale_256_wasserstein_1": wasserstein_1d(
                        final_samples[left], final_samples[right]
                    ),
                }
            )
    non_gaussian = [row for row in systems if row["law"] != "gaussian"]
    initial_non_gaussian_ks = max(
        row["levels"][0]["normal_ks_distance"] for row in non_gaussian
    )
    final_non_gaussian_ks = max(
        row["levels"][-1]["normal_ks_distance"] for row in non_gaussian
    )
    final_cross = max(row["scale_256_wasserstein_1"] for row in cross)
    return {
        "replicas_per_law": replicas,
        "increment_standard_deviation": sigma,
        "block_sizes": list(BLOCK_SIZES),
        "systems": systems,
        "cross_law_distances": cross,
        "maximum_initial_non_gaussian_ks_distance": initial_non_gaussian_ks,
        "maximum_final_non_gaussian_ks_distance": final_non_gaussian_ks,
        "maximum_scale_256_cross_law_wasserstein_1": final_cross,
        "qualified": (
            final_non_gaussian_ks < 0.035
            and final_non_gaussian_ks < 0.15 * initial_non_gaussian_ks
            and final_cross < 0.04
        ),
    }


def execute() -> dict:
    generator = np.random.default_rng(SEED)
    algebra = algebra_qualification(generator)
    clean = clean_critical_qualification()
    fluctuations = fluctuation_qualification(generator)
    record = {
        "schema_version": "row-rgmap-synthetic-qualification-v1",
        "seed": SEED,
        "interpretation": {
            "algebra_and_clean_flow": "implementation qualification",
            "fluctuation_fixed_point": "synthetic finite-variance qualification",
            "pldr_empirical_claim": False,
        },
        "algebra": algebra,
        "clean_critical_flow": clean,
        "finite_variance_fluctuations": fluctuations,
        "decision": (
            "QUALIFIED"
            if algebra["qualified"] and clean["qualified"]
            and fluctuations["qualified"]
            else "NOT_QUALIFIED"
        ),
    }
    return seal_record(record)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = execute()
    destination = Path(arguments.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if result["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
