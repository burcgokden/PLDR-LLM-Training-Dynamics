"""Exact LayerNorm chords and path metrics for the PLDR post-norm chain."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "pldr-postnorm-geometry-v1"


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1 or array.size < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    return np.ascontiguousarray(array)


def _matrix(value: Any, name: str, dimension: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[0] != array.shape[1]
        or not np.isfinite(array).all()
        or dimension is not None and array.shape != (dimension, dimension)
    ):
        raise ValueError(f"{name} must be a finite square matrix")
    return np.ascontiguousarray(array)


def centering_matrix(dimension: int) -> np.ndarray:
    count = int(dimension)
    if count < 1:
        raise ValueError("dimension must be positive")
    return np.eye(count) - np.ones((count, count)) / count


def normalized_layernorm(value: Any, epsilon: float = 1.0e-6) -> np.ndarray:
    vector = _vector(value, "value")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("LayerNorm epsilon must be positive")
    centered = vector - np.mean(vector)
    denominator = math.sqrt(
        epsilon + float(np.dot(centered, centered)) / vector.size)
    return centered / denominator


def layernorm_jacobian(value: Any, epsilon: float = 1.0e-6) -> np.ndarray:
    """Return the exact derivative of stabilized normalized LayerNorm."""

    vector = _vector(value, "value")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("LayerNorm epsilon must be positive")
    projection = centering_matrix(vector.size)
    centered = projection @ vector
    denominator = math.sqrt(
        epsilon + float(np.dot(centered, centered)) / vector.size)
    return (
        projection / denominator
        - np.outer(centered, centered)
        / (vector.size * denominator ** 3)
    )


def affine_layernorm_jacobian(
    value: Any, gamma: Any, epsilon: float = 1.0e-6,
) -> np.ndarray:
    vector = _vector(value, "value")
    gain = _vector(gamma, "gamma")
    if gain.shape != vector.shape:
        raise ValueError("LayerNorm value and gain widths disagree")
    return gain[:, None] * layernorm_jacobian(vector, epsilon)


def layernorm_chord_operator(
    left: Any,
    right: Any,
    *,
    gamma: Any | None = None,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Return an exact analytic chord for normalized or affine LayerNorm.

    The stable divided difference
    ``-1/(s_left*s_right*(s_left+s_right))`` avoids cancellation when the
    two centered variances are close.
    """

    x = _vector(left, "left")
    y = _vector(right, "right")
    if x.shape != y.shape:
        raise ValueError("LayerNorm chord endpoints disagree")
    epsilon = float(epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("LayerNorm epsilon must be positive")
    projection = centering_matrix(x.size)
    px = projection @ x
    py = projection @ y
    sx = math.sqrt(epsilon + float(np.dot(px, px)) / x.size)
    sy = math.sqrt(epsilon + float(np.dot(py, py)) / x.size)
    if np.array_equal(x, y):
        chord = layernorm_jacobian(x, epsilon)
    else:
        reciprocal_secant = -1.0 / (sx * sy * (sx + sy))
        chord = (
            projection / sx
            + reciprocal_secant / x.size
            * np.outer(py, projection @ (x + y))
        )
    if gamma is not None:
        gain = _vector(gamma, "gamma")
        if gain.shape != x.shape:
            raise ValueError("LayerNorm chord and gain widths disagree")
        chord = gain[:, None] * chord
    return np.ascontiguousarray(chord)


def chord_action_residual(
    left: Any,
    right: Any,
    chord: Any,
    *,
    gamma: Any | None = None,
    beta: Any | None = None,
    epsilon: float = 1.0e-6,
) -> float:
    """Measure the relative endpoint residual of a supplied LayerNorm chord."""

    x = _vector(left, "left")
    y = _vector(right, "right")
    operator = _matrix(chord, "chord", x.size)
    left_output = normalized_layernorm(x, epsilon)
    right_output = normalized_layernorm(y, epsilon)
    if gamma is not None:
        gain = _vector(gamma, "gamma")
        if gain.shape != x.shape:
            raise ValueError("LayerNorm gain width disagrees")
        shift = (
            np.zeros_like(gain) if beta is None else _vector(beta, "beta")
        )
        if shift.shape != gain.shape:
            raise ValueError("LayerNorm shift width disagrees")
        left_output = shift + gain * left_output
        right_output = shift + gain * right_output
    realized = left_output - right_output
    predicted = operator @ (x - y)
    scale = max(
        float(np.linalg.norm(realized)),
        float(np.linalg.norm(predicted)),
        np.finfo(np.float64).tiny,
    )
    return float(np.linalg.norm(realized - predicted)) / scale


def compose_factors(factors: Sequence[Any]) -> np.ndarray:
    """Compose factors in execution order: first factor acts first."""

    if not factors:
        raise ValueError("at least one factor is required")
    parsed = [_matrix(value, f"factor[{index}]")
              for index, value in enumerate(factors)]
    dimension = parsed[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in parsed):
        raise ValueError("path factors have inconsistent dimensions")
    product = np.eye(dimension)
    for factor in parsed:
        product = factor @ product
    return np.ascontiguousarray(product)


def backward_path_metrics(
    factors: Sequence[Any], *, regularization: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Construct positive path metrics by a backward Lyapunov recursion."""

    if not factors:
        raise ValueError("at least one path factor is required")
    parsed = [_matrix(value, f"factor[{index}]")
              for index, value in enumerate(factors)]
    dimension = parsed[0].shape[0]
    if any(value.shape != (dimension, dimension) for value in parsed):
        raise ValueError("path factors have inconsistent dimensions")
    charge = float(regularization)
    if not math.isfinite(charge) or charge <= 0.0:
        raise ValueError("regularization must be positive")
    metrics: list[np.ndarray] = [np.empty((0, 0))] * (len(parsed) + 1)
    metrics[-1] = charge * np.eye(dimension)
    gain_squared = np.empty(len(parsed), dtype=np.float64)
    for index in range(len(parsed) - 1, -1, -1):
        transported = parsed[index].T @ metrics[index + 1] @ parsed[index]
        metrics[index] = charge * np.eye(dimension) + transported
        maximum = float(np.linalg.eigvalsh(metrics[index])[-1])
        gain_squared[index] = max(0.0, 1.0 - charge / maximum)
    return metrics, np.sqrt(gain_squared)


def verify_path_metric(
    factors: Sequence[Any],
    metrics: Sequence[Any],
    gains: Any,
    *,
    tolerance: float = 512.0 * np.finfo(np.float64).eps,
) -> dict[str, Any]:
    """Verify every quadratic matrix inequality and its Euclidean bound."""

    parsed = [_matrix(value, f"factor[{index}]")
              for index, value in enumerate(factors)]
    if not parsed:
        raise ValueError("at least one factor is required")
    dimension = parsed[0].shape[0]
    metric_values = [_matrix(value, f"metric[{index}]", dimension)
                     for index, value in enumerate(metrics)]
    gain = _vector(gains, "gains")
    if len(metric_values) != len(parsed) + 1 or gain.shape != (len(parsed),):
        raise ValueError("path metrics and gains have invalid counts")
    if np.any(gain < 0.0):
        raise ValueError("path gains must be nonnegative")
    minimum_metric = math.inf
    maximum_metric = 0.0
    margins = []
    passed = True
    for metric in metric_values:
        eigenvalues = np.linalg.eigvalsh((metric + metric.T) / 2.0)
        minimum_metric = min(minimum_metric, float(eigenvalues[0]))
        maximum_metric = max(maximum_metric, float(eigenvalues[-1]))
        passed = bool(passed and eigenvalues[0] > 0.0)
    for index, factor in enumerate(parsed):
        residual = (
            gain[index] ** 2 * metric_values[index]
            - factor.T @ metric_values[index + 1] @ factor
        )
        margin = float(np.linalg.eigvalsh((residual + residual.T) / 2.0)[0])
        scale = max(
            float(np.linalg.norm(residual, ord=2)),
            float(np.linalg.norm(metric_values[index], ord=2)),
            1.0,
        )
        margins.append(margin)
        passed = bool(passed and margin >= -float(tolerance) * scale)
    endpoint_lower = float(np.linalg.eigvalsh(metric_values[-1])[0])
    source_upper = float(np.linalg.eigvalsh(metric_values[0])[-1])
    physical_coefficient = (
        math.sqrt(source_upper / endpoint_lower)
        * float(np.prod(gain))
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "stage_count": len(parsed),
        "dimension": dimension,
        "quadratic_inequalities_hold": passed,
        "minimum_stage_margin": min(margins),
        "stage_margins": margins,
        "minimum_metric_eigenvalue": minimum_metric,
        "maximum_metric_eigenvalue": maximum_metric,
        "endpoint_metric_lower": endpoint_lower,
        "source_metric_upper": source_upper,
        "gain_product": float(np.prod(gain)),
        "physical_euclidean_coefficient": physical_coefficient,
    }


__all__ = [
    "SCHEMA_VERSION",
    "affine_layernorm_jacobian",
    "backward_path_metrics",
    "centering_matrix",
    "chord_action_residual",
    "compose_factors",
    "layernorm_chord_operator",
    "layernorm_jacobian",
    "normalized_layernorm",
    "verify_path_metric",
]
