"""Explicit LayerNorm primitives and analytic affine-path derivatives."""

from __future__ import annotations

from typing import Any

import numpy as np


SCHEMA_VERSION = "pldr-explicit-layernorm-v1"


def _array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite non-scalar array")
    return np.ascontiguousarray(array)


def explicit_layer_norm(
    value: Any,
    gain: Any,
    bias: Any,
    *,
    epsilon: float = 1.0e-6,
) -> np.ndarray:
    """Evaluate affine LayerNorm from mean, square, reciprocal, and sqrt."""

    x = _array(value, "value")
    gamma = _array(gain, "gain")
    beta = _array(bias, "bias")
    if gamma.shape != (x.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("gain and bias must match the final input dimension")
    eps = float(epsilon)
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("epsilon must be finite and positive")
    centered = x - np.mean(x, axis=-1, keepdims=True)
    radius_squared = eps + np.mean(centered * centered, axis=-1, keepdims=True)
    return gamma * centered / np.sqrt(radius_squared) + beta


def affine_path_coefficients(
    value: Any,
    direction: Any,
    gain: Any,
    bias: Any,
    *,
    position: float = 0.0,
    epsilon: float = 1.0e-6,
) -> dict[str, np.ndarray]:
    """Return value and first three derivatives on ``x + s direction``.

    The calculation differentiates the explicit centered variance and
    reciprocal square root.  It never nests automatic differentiation
    through a native LayerNorm kernel.
    """

    x0 = _array(value, "value")
    dx = _array(direction, "direction")
    gamma = _array(gain, "gain")
    beta = _array(bias, "bias")
    if x0.shape != dx.shape:
        raise ValueError("value and direction shapes disagree")
    if gamma.shape != (x0.shape[-1],) or beta.shape != gamma.shape:
        raise ValueError("gain and bias must match the final input dimension")
    s = float(position)
    eps = float(epsilon)
    if not np.isfinite(s) or not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("position and epsilon are invalid")

    centered0 = x0 - np.mean(x0, axis=-1, keepdims=True)
    centered_direction = dx - np.mean(dx, axis=-1, keepdims=True)
    centered = centered0 + s * centered_direction
    q = eps + np.mean(centered * centered, axis=-1, keepdims=True)
    q1 = 2.0 * np.mean(
        centered * centered_direction, axis=-1, keepdims=True)
    q2 = 2.0 * np.mean(
        centered_direction * centered_direction, axis=-1, keepdims=True)
    if np.any(q <= 0.0):
        raise AssertionError("regularized LayerNorm radius lost positivity")

    inverse = q ** -0.5
    inverse1 = -0.5 * q ** -1.5 * q1
    inverse2 = 0.75 * q ** -2.5 * q1 * q1 - 0.5 * q ** -1.5 * q2
    inverse3 = (
        -1.875 * q ** -3.5 * q1 * q1 * q1
        + 2.25 * q ** -2.5 * q1 * q2
    )
    normalized = centered * inverse
    derivative1 = centered_direction * inverse + centered * inverse1
    derivative2 = 2.0 * centered_direction * inverse1 + centered * inverse2
    derivative3 = 3.0 * centered_direction * inverse2 + centered * inverse3
    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "value": gamma * normalized + beta,
        "first": gamma * derivative1,
        "second": gamma * derivative2,
        "third": gamma * derivative3,
        "regularized_radius_squared": q,
    }
