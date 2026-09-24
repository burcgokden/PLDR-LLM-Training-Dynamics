"""Constrained metric projection and normal-chart diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite(value: Any, name: str, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != ndim or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite rank-{ndim} array")
    return result


def metric_projection(
    point: Any,
    constraint_jacobian: Any,
    target: Any,
    metric: Any,
    *,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Project a point onto a linearized constraint in a positive metric."""

    source = _finite(point, "point", 1)
    jacobian = _finite(constraint_jacobian, "constraint_jacobian", 2)
    rhs = _finite(target, "target", 1)
    weight = _finite(metric, "metric", 2)
    dimension = source.size
    if (
        jacobian.shape[1] != dimension
        or rhs.shape != (jacobian.shape[0],)
        or weight.shape != (dimension, dimension)
    ):
        raise ValueError("projection dimensions disagree")
    if not np.isfinite(relative_tolerance) or relative_tolerance <= 0.0:
        raise ValueError("projection tolerance must be finite and positive")
    symmetry = float(np.linalg.norm(weight - weight.T, ord=2))
    symmetric = 0.5 * (weight + weight.T)
    metric_eigenvalues = np.linalg.eigvalsh(symmetric)
    if metric_eigenvalues[0] <= 0.0:
        raise ValueError("projection metric is not positive definite")
    inverse_jacobian_transpose = np.linalg.solve(symmetric, jacobian.T)
    gram = jacobian @ inverse_jacobian_transpose
    left, singular, right = np.linalg.svd(gram, full_matrices=False)
    threshold = (
        relative_tolerance * singular[0]
        if singular.size and singular[0] > 0.0
        else relative_tolerance
    )
    rank = int(np.sum(singular > threshold))
    inverse = np.zeros_like(singular)
    inverse[:rank] = 1.0 / singular[:rank]
    gram_pseudoinverse = (right.T * inverse) @ left.T
    residual = jacobian @ source - rhs
    multiplier = gram_pseudoinverse @ residual
    projected = source - inverse_jacobian_transpose @ multiplier
    primal = jacobian @ projected - rhs
    stationarity = (
        symmetric @ (projected - source) + jacobian.T @ multiplier
    )
    range_residual = residual - gram @ multiplier
    objective = 0.5 * float(
        (projected - source) @ symmetric @ (projected - source)
    )
    condition = (
        float(singular[0] / singular[rank - 1])
        if rank and singular[rank - 1] > 0.0
        else float("inf")
    )
    return {
        "projected": projected,
        "multiplier": multiplier,
        "rank": rank,
        "rank_threshold": float(threshold),
        "singular_values": singular,
        "condition_estimate": condition,
        "metric_eigenvalue_min": float(metric_eigenvalues[0]),
        "metric_eigenvalue_max": float(metric_eigenvalues[-1]),
        "metric_symmetry_residual": symmetry,
        "primal_residual_norm": float(np.linalg.norm(primal)),
        "kkt_stationarity_residual_norm": float(
            np.linalg.norm(stationarity)
        ),
        "constraint_range_residual_norm": float(
            np.linalg.norm(range_residual)
        ),
        "objective": objective,
    }


def projection_valid(
    result: dict[str, Any],
    *,
    absolute_tolerance: float,
    required_rank: int | None = None,
) -> bool:
    """Recompute projection validity from numeric diagnostics."""

    if not np.isfinite(absolute_tolerance) or absolute_tolerance <= 0.0:
        raise ValueError("absolute tolerance must be finite and positive")
    rank_valid = (
        required_rank is None or int(result.get("rank", -1)) == required_rank
    )
    return bool(
        rank_valid
        and result.get("metric_eigenvalue_min", 0.0) > 0.0
        and result.get("metric_symmetry_residual", np.inf)
        <= absolute_tolerance
        and result.get("primal_residual_norm", np.inf)
        <= absolute_tolerance
        and result.get("kkt_stationarity_residual_norm", np.inf)
        <= absolute_tolerance
        and result.get("constraint_range_residual_norm", np.inf)
        <= absolute_tolerance
    )


def sign_paired_perturbations(
    direction: Any,
    amplitude: float,
    *,
    fractions: tuple[float, ...] = (1.0, 0.5, 0.25),
) -> np.ndarray:
    """Construct the frozen positive and negative normal perturbation menu."""

    vector = _finite(direction, "direction", 1)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("normal perturbation direction must be nonzero")
    if not np.isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("normal perturbation amplitude must be positive")
    if (
        not fractions
        or any(not np.isfinite(value) or value <= 0.0 for value in fractions)
        or tuple(sorted(fractions, reverse=True)) != fractions
    ):
        raise ValueError("perturbation fractions must decrease and be positive")
    unit = vector / norm
    rows = []
    for fraction in fractions:
        rows.extend((amplitude * fraction * unit, -amplitude * fraction * unit))
    return np.stack(rows)


def quadratic_defect_profile(
    function,
    source: Any,
    operator: Any,
    force: Any,
    perturbations: Any,
) -> dict[str, Any]:
    """Measure finite nonlinear defects around a source-frozen block map."""

    origin = _finite(source, "source", 1)
    linear = _finite(operator, "operator", 2)
    constant = _finite(force, "force", 1)
    probes = _finite(perturbations, "perturbations", 2)
    if (
        linear.shape != (origin.size, origin.size)
        or constant.shape != origin.shape
        or probes.shape[1] != origin.size
    ):
        raise ValueError("quadratic defect dimensions disagree")
    defects = []
    coefficients = []
    for perturbation in probes:
        observed = _finite(function(origin + perturbation), "block output", 1)
        predicted = function(origin) + linear @ perturbation
        defect = observed - predicted
        amplitude = float(np.linalg.norm(perturbation))
        defects.append(float(np.linalg.norm(defect)))
        coefficients.append(
            float(np.linalg.norm(defect) / amplitude**2)
            if amplitude > 0.0
            else float("inf")
        )
    return {
        "amplitudes": np.linalg.norm(probes, axis=1),
        "defect_norms": np.asarray(defects),
        "quadratic_coefficients": np.asarray(coefficients),
        "coefficient_upper": float(
            np.nextafter(max(coefficients), np.inf)
        ),
        "source_force_residual": float(
            np.linalg.norm(_finite(function(origin), "block output", 1)
                           - constant)
        ),
    }

