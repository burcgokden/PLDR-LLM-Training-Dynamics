"""Composite PLGA secants and row-centered causal-score quotients."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np


SCHEMA_VERSION = "pldr-composite-plga-transfer-v1"


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.ascontiguousarray(array)


def iswiglu(value: Any) -> np.ndarray:
    """Implemented identity-weight SwiGLU: ``z^2 sigmoid(z)``."""

    z = _array(value, "value")
    sigmoid = np.empty_like(z)
    positive = z >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return z * z * sigmoid


def iswiglu_derivative(value: Any) -> np.ndarray:
    z = _array(value, "value")
    sigmoid = np.empty_like(z)
    positive = z >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return 2.0 * z * sigmoid + z * z * sigmoid * (1.0 - sigmoid)


def positive_real_power(base: Any, exponent: Any) -> np.ndarray:
    value = _array(base, "base")
    power = _array(exponent, "exponent")
    if value.shape != power.shape or np.any(value <= 0.0):
        raise ValueError("positive real powers need matching positive bases")
    result = np.exp(power * np.log(value))
    if not np.isfinite(result).all():
        raise FloatingPointError("positive real power overflowed")
    return result


def composite_secant(
    left: Any,
    right: Any,
    exponent: Any,
    *,
    epsilon: float = 1.0e-9,
) -> np.ndarray:
    """Signed divided difference of ``(iSwiGLU(z)+epsilon)^p``."""

    x = _array(left, "left")
    y = _array(right, "right")
    power = _array(exponent, "exponent")
    if x.shape != y.shape or x.shape != power.shape:
        raise ValueError("composite secant arrays disagree")
    floor = float(epsilon)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("the PLGA adjustment epsilon must be positive")
    base_x = iswiglu(x) + floor
    base_y = iswiglu(y) + floor
    output_x = positive_real_power(base_x, power)
    output_y = positive_real_power(base_y, power)
    equal = x == y
    coefficient = np.empty_like(x)
    coefficient[equal] = (
        power[equal]
        * positive_real_power(base_x[equal], power[equal] - 1.0)
        * iswiglu_derivative(x[equal])
    )
    unequal = ~equal
    coefficient[unequal] = (
        output_y[unequal] - output_x[unequal]
    ) / (y[unequal] - x[unequal])
    if not np.isfinite(coefficient).all():
        raise FloatingPointError("composite secant overflowed")
    return coefficient


def row_center(value: Any) -> tuple[np.ndarray, np.ndarray]:
    matrix = _array(value, "value", 2)
    mean_row = np.mean(matrix, axis=0)
    constant_rows = np.broadcast_to(mean_row, matrix.shape).copy()
    return matrix - constant_rows, constant_rows


def row_diameter_squared(value: Any) -> tuple[float, tuple[int, int]]:
    matrix = _array(value, "value", 2)
    if matrix.shape[0] < 2:
        raise ValueError("row diameter needs at least two rows")
    maximum = -math.inf
    pair = (0, 1)
    for left in range(matrix.shape[0] - 1):
        difference = matrix[left + 1:] - matrix[left]
        energy = np.einsum("pd,pd->p", difference, difference)
        offset = int(np.argmax(energy))
        candidate = float(energy[offset])
        candidate_pair = (left, left + 1 + offset)
        if candidate > maximum or candidate == maximum and candidate_pair < pair:
            maximum = candidate
            pair = candidate_pair
    return maximum, pair


def plga_transfer(
    value: Any,
    reference: Any,
    weight: Any,
    bias: Any,
    exponent: Any,
    coupling: Any,
    coupling_bias: Any,
    *,
    epsilon: float = 1.0e-9,
) -> dict[str, np.ndarray | float]:
    """Evaluate the exact composite secant identity for one PLGA head."""

    matrix = _array(value, "value", 2)
    baseline = _array(reference, "reference", 2)
    W = _array(weight, "weight", 2)
    b = _array(bias, "bias", 2)
    power = _array(exponent, "exponent", 2)
    a = _array(coupling, "coupling", 2)
    ba = _array(coupling_bias, "coupling_bias", 2)
    if (
        matrix.shape != baseline.shape
        or matrix.shape[0] != matrix.shape[1]
        or any(item.shape != matrix.shape for item in (W, b, power, a, ba))
    ):
        raise ValueError("PLGA transfer arrays must be matching square matrices")
    z_value = W @ matrix + b
    z_reference = W @ baseline + b
    secant = composite_secant(
        z_reference, z_value, power, epsilon=epsilon)
    powered_value = positive_real_power(iswiglu(z_value) + epsilon, power)
    powered_reference = positive_real_power(
        iswiglu(z_reference) + epsilon, power)
    realized = a @ powered_value + ba - (a @ powered_reference + ba)
    predicted = a @ (secant * (W @ (matrix - baseline)))
    scale = max(
        float(np.linalg.norm(realized)),
        float(np.linalg.norm(predicted)),
        np.finfo(np.float64).tiny,
    )
    return {
        "z_value": z_value,
        "z_reference": z_reference,
        "composite_secant": secant,
        "realized_difference": realized,
        "predicted_difference": predicted,
        "identity_relative_residual": float(
            np.linalg.norm(realized - predicted)) / scale,
        "minimum_positive_base": float(min(
            np.min(iswiglu(z_value) + epsilon),
            np.min(iswiglu(z_reference) + epsilon),
        )),
        "minimum_exponent": float(np.min(power)),
        "maximum_exponent": float(np.max(power)),
        "negative_exponent_fraction": float(np.mean(power < 0.0)),
    }


def _center_rows(value: np.ndarray) -> np.ndarray:
    return value - np.mean(value, axis=0, keepdims=True)


def _causal_center(value: np.ndarray) -> np.ndarray:
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("causal centering requires a square score matrix")
    result = np.zeros_like(value)
    for row in range(value.shape[0]):
        allowed = value[row, :row + 1]
        result[row, :row + 1] = allowed - np.mean(allowed)
    return result


def _power_iteration(
    operator: Callable[[np.ndarray], np.ndarray],
    adjoint: Callable[[np.ndarray], np.ndarray],
    shape: tuple[int, int],
    *,
    iterations: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(int(seed))
    vector = _center_rows(rng.standard_normal(shape))
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ArithmeticError("power iteration initialized at zero")
    vector /= norm
    residual = math.inf
    eigenvalue = 0.0
    for _ in range(int(iterations)):
        image = operator(vector)
        normal = _center_rows(adjoint(image))
        next_eigenvalue = float(np.vdot(vector, normal).real)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm == 0.0:
            return 0.0, 0.0
        next_vector = normal / normal_norm
        residual = float(np.linalg.norm(normal - next_eigenvalue * vector))
        vector = next_vector
        eigenvalue = max(0.0, next_eigenvalue)
    return math.sqrt(eigenvalue), residual


def restricted_operator_coefficients(
    *,
    composite_secant_value: Any,
    weight: Any,
    coupling: Any,
    query: Any,
    key: Any,
    iterations: int = 200,
    seed: int = 4040,
) -> dict[str, float]:
    """Compute row-centered PLGA and causal-score restricted coefficients."""

    secant = _array(composite_secant_value, "composite_secant", 2)
    W = _array(weight, "weight", 2)
    a = _array(coupling, "coupling", 2)
    Q = _array(query, "query", 2)
    K = _array(key, "key", 2)
    dimension = secant.shape[0]
    if (
        secant.shape != (dimension, dimension)
        or W.shape != secant.shape
        or a.shape != secant.shape
        or Q.shape[1] != dimension
        or K.shape != Q.shape
        or Q.shape[0] < 1
        or int(iterations) < 1
    ):
        raise ValueError("restricted PLGA operator dimensions disagree")

    def transfer(value: np.ndarray) -> np.ndarray:
        return a @ (secant * (W @ value))

    def transfer_adjoint(value: np.ndarray) -> np.ndarray:
        return W.T @ (secant * (a.T @ value))

    def centered_transfer(value: np.ndarray) -> np.ndarray:
        return transfer(_center_rows(value))

    def centered_transfer_adjoint(value: np.ndarray) -> np.ndarray:
        return _center_rows(transfer_adjoint(value))

    plga_norm, plga_residual = _power_iteration(
        centered_transfer,
        centered_transfer_adjoint,
        secant.shape,
        iterations=iterations,
        seed=seed,
    )
    scale = math.sqrt(dimension)

    def causal_score(value: np.ndarray) -> np.ndarray:
        return _causal_center(Q @ transfer(_center_rows(value)) @ K.T / scale)

    def causal_score_adjoint(value: np.ndarray) -> np.ndarray:
        centered = _causal_center(value)
        pulled = Q.T @ centered @ K / scale
        return _center_rows(transfer_adjoint(pulled))

    score_norm, score_residual = _power_iteration(
        causal_score,
        causal_score_adjoint,
        secant.shape,
        iterations=iterations,
        seed=seed + 1,
    )
    raw_bound = (
        float(np.linalg.norm(a, ord=2))
        * float(np.max(np.abs(secant)))
        * float(np.linalg.norm(W, ord=2))
    )
    return {
        "raw_separate_maximum_coefficient": raw_bound,
        "row_centered_composite_coefficient": plga_norm,
        "causal_score_quotient_coefficient": score_norm,
        "plga_power_iteration_residual": plga_residual,
        "score_power_iteration_residual": score_residual,
    }


def row_variance_identity(value: Any) -> dict[str, float | list[int]]:
    matrix = _array(value, "value", 2)
    centered, _constant = row_center(matrix)
    left = float(np.vdot(centered, centered).real)
    pair_sum = 0.0
    for first in range(matrix.shape[0] - 1):
        difference = matrix[first + 1:] - matrix[first]
        pair_sum += float(np.vdot(difference, difference).real)
    right = pair_sum / matrix.shape[0]
    diameter_sq, pair = row_diameter_squared(matrix)
    bound = (matrix.shape[0] - 1.0) * diameter_sq / 2.0
    scale = max(abs(left), abs(right), np.finfo(np.float64).tiny)
    return {
        "centered_frobenius_squared": left,
        "pair_average_squared": right,
        "identity_relative_residual": abs(left - right) / scale,
        "diameter_squared": diameter_sq,
        "diameter_pair": list(pair),
        "diameter_upper_squared": bound,
    }


__all__ = [
    "SCHEMA_VERSION",
    "composite_secant",
    "iswiglu",
    "iswiglu_derivative",
    "plga_transfer",
    "positive_real_power",
    "restricted_operator_coefficients",
    "row_center",
    "row_diameter_squared",
    "row_variance_identity",
]
