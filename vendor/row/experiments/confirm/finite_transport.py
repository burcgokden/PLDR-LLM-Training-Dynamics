"""Exact finite endpoint transport for the implemented row program."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite array")
    return result


def _same_shape(*values: np.ndarray) -> None:
    if len({value.shape for value in values}) != 1:
        raise ValueError("endpoint arrays must have one common shape")


def affine_increment(
    weight0: Any,
    weight1: Any,
    input0: Any,
    input1: Any,
    bias0: Any,
    bias1: Any,
) -> dict[str, Any]:
    """Return the exact finite increment of an affine primitive."""

    w0 = _array(weight0, "weight0")
    w1 = _array(weight1, "weight1")
    x0 = _array(input0, "input0")
    x1 = _array(input1, "input1")
    b0 = _array(bias0, "bias0")
    b1 = _array(bias1, "bias1")
    if w0.shape != w1.shape or x0.shape != x1.shape or b0.shape != b1.shape:
        raise ValueError("affine endpoints disagree in shape")
    direct = w1 @ x1 + b1 - (w0 @ x0 + b0)
    state = w1 @ (x1 - x0)
    weight = (w1 - w0) @ x0
    bias = b1 - b0
    reconstructed = state + weight + bias
    return {
        "increment": direct,
        "state_contribution": state,
        "weight_contribution": weight,
        "bias_contribution": bias,
        "reconstruction": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(direct - reconstructed))),
    }


def product_increment(
    left0: Any, left1: Any, right0: Any, right1: Any
) -> dict[str, Any]:
    """Return the exact endpoint-ordered increment of an entrywise product."""

    a0 = _array(left0, "left0")
    a1 = _array(left1, "left1")
    b0 = _array(right0, "right0")
    b1 = _array(right1, "right1")
    _same_shape(a0, a1, b0, b1)
    direct = a1 * b1 - a0 * b0
    right = a1 * (b1 - b0)
    left = (a1 - a0) * b0
    reconstructed = right + left
    return {
        "increment": direct,
        "left_contribution": left,
        "right_contribution": right,
        "reconstruction": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(direct - reconstructed))),
    }


def divided_difference(
    function: Callable[[np.ndarray], np.ndarray],
    derivative: Callable[[np.ndarray], np.ndarray],
    left: Any,
    right: Any,
) -> np.ndarray:
    """Evaluate a secant with the true derivative on its diagonal."""

    x0 = _array(left, "left")
    x1 = _array(right, "right")
    _same_shape(x0, x1)
    f0 = _array(function(x0), "function(left)")
    f1 = _array(function(x1), "function(right)")
    _same_shape(x0, f0, f1)
    result = np.empty_like(x0)
    different = x0 != x1
    result[different] = (
        (f1[different] - f0[different])
        / (x1[different] - x0[different])
    )
    if np.any(~different):
        diagonal = _array(derivative(x0), "derivative(left)")
        _same_shape(x0, diagonal)
        result[~different] = diagonal[~different]
    if not np.isfinite(result).all():
        raise FloatingPointError("divided difference is not finite")
    return result


def layernorm_increment(
    input0: Any,
    input1: Any,
    gain0: Any,
    gain1: Any,
    bias0: Any,
    bias1: Any,
    *,
    epsilon: float,
) -> dict[str, Any]:
    """Reconstruct the exact regularized LayerNorm endpoint increment."""

    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("LayerNorm epsilon must be finite and positive")
    x0 = _array(input0, "input0")
    x1 = _array(input1, "input1")
    g0 = _array(gain0, "gain0")
    g1 = _array(gain1, "gain1")
    b0 = _array(bias0, "bias0")
    b1 = _array(bias1, "bias1")
    if x0.shape != x1.shape or x0.shape[-1:] != g0.shape:
        raise ValueError("LayerNorm input and gain shapes disagree")
    _same_shape(g0, g1, b0, b1)

    def normalize(value: np.ndarray) -> np.ndarray:
        centered = value - np.mean(value, axis=-1, keepdims=True)
        radius = np.sqrt(
            epsilon + np.mean(centered * centered, axis=-1, keepdims=True)
        )
        return centered / radius

    n0 = normalize(x0)
    n1 = normalize(x1)
    direct = g1 * n1 + b1 - (g0 * n0 + b0)
    state = g1 * (n1 - n0)
    gain = (g1 - g0) * n0
    bias = b1 - b0
    reconstructed = state + gain + bias
    return {
        "increment": direct,
        "state_contribution": state,
        "gain_contribution": gain,
        "bias_contribution": bias,
        "reconstruction": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(direct - reconstructed))),
    }


def sigmoid(value: Any) -> np.ndarray:
    value = _array(value, "sigmoid input")
    positive = value >= 0.0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def plga_power(value: Any, exponent: Any, *, epsilon: float) -> np.ndarray:
    z = _array(value, "PLGA value")
    p = np.broadcast_to(_array(exponent, "PLGA exponent"), z.shape)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("PLGA adjustment must be finite and positive")
    base = z * z * sigmoid(z) + epsilon
    result = np.power(base, p)
    if not np.isfinite(result).all():
        raise FloatingPointError("implemented PLGA power is not finite")
    return result


def plga_z_derivative(
    value: Any, exponent: Any, *, epsilon: float
) -> np.ndarray:
    z = _array(value, "PLGA value")
    p = np.broadcast_to(_array(exponent, "PLGA exponent"), z.shape)
    s = sigmoid(z)
    base = z * z * s + epsilon
    return (
        p
        * np.power(base, p - 1.0)
        * (2.0 * z * s + z * z * s * (1.0 - s))
    )


def plga_p_derivative(
    value: Any, exponent: Any, *, epsilon: float
) -> np.ndarray:
    z = _array(value, "PLGA value")
    p = np.broadcast_to(_array(exponent, "PLGA exponent"), z.shape)
    base = z * z * sigmoid(z) + epsilon
    return plga_power(z, p, epsilon=epsilon) * np.log(base)


def plga_multiplied_fields(
    value0: Any,
    value1: Any,
    exponent0: Any,
    exponent1: Any,
    *,
    epsilon: float = 1.0e-9,
) -> dict[str, Any]:
    """Return both exact coordinate orders as multiplied endpoint fields."""

    z0 = _array(value0, "value0")
    z1 = _array(value1, "value1")
    p0 = np.broadcast_to(_array(exponent0, "exponent0"), z0.shape)
    p1 = np.broadcast_to(_array(exponent1, "exponent1"), z0.shape)
    _same_shape(z0, z1, p0, p1)

    def z_secant(left: np.ndarray, right: np.ndarray, power: np.ndarray):
        return divided_difference(
            lambda value: plga_power(value, power, epsilon=epsilon),
            lambda value: plga_z_derivative(value, power, epsilon=epsilon),
            left,
            right,
        )

    def p_secant(left: np.ndarray, right: np.ndarray, value: np.ndarray):
        return divided_difference(
            lambda power: plga_power(value, power, epsilon=epsilon),
            lambda power: plga_p_derivative(value, power, epsilon=epsilon),
            left,
            right,
        )

    direct = (
        plga_power(z1, p1, epsilon=epsilon)
        - plga_power(z0, p0, epsilon=epsilon)
    )
    z_first = z_secant(z0, z1, p1) * (z1 - z0)
    p_after = p_secant(p0, p1, z0) * (p1 - p0)
    p_first = p_secant(p0, p1, z1) * (p1 - p0)
    z_after = z_secant(z0, z1, p0) * (z1 - z0)
    route_one = z_first + p_after
    route_two = p_first + z_after
    return {
        "increment": direct,
        "z_at_endpoint_exponent": z_first,
        "p_at_source_value": p_after,
        "p_at_endpoint_value": p_first,
        "z_at_source_exponent": z_after,
        "route_z_then_p": route_one,
        "route_p_then_z": route_two,
        "maximum_abs_route_one_residual": float(
            np.max(np.abs(direct - route_one))
        ),
        "maximum_abs_route_two_residual": float(
            np.max(np.abs(direct - route_two))
        ),
        "maximum_abs_route_disagreement": float(
            np.max(np.abs(route_one - route_two))
        ),
    }


def chronological_unroll(
    maps: Sequence[Any], sources: Sequence[Any], initial: Any
) -> dict[str, Any]:
    """Unroll a finite affine recurrence in actual execution order."""

    if len(maps) != len(sources):
        raise ValueError("maps and local sources must have equal length")
    state0 = _array(initial, "initial")
    if state0.ndim != 1:
        raise ValueError("the finite recurrence state must be one-dimensional")
    dimension = state0.size
    matrices = tuple(_array(value, "map") for value in maps)
    local = tuple(_array(value, "source") for value in sources)
    if any(value.shape != (dimension, dimension) for value in matrices):
        raise ValueError("every finite transport map must be square")
    if any(value.shape != (dimension,) for value in local):
        raise ValueError("every local source must match the state dimension")
    direct = state0.copy()
    states = [direct.copy()]
    for matrix, source in zip(matrices, local, strict=True):
        direct = matrix @ direct + source
        states.append(direct.copy())
    suffix = np.eye(dimension)
    source_contributions: list[np.ndarray] = []
    for matrix, source in zip(
        reversed(matrices), reversed(local), strict=True
    ):
        source_contributions.append(suffix @ source)
        suffix = suffix @ matrix
    source_contributions.reverse()
    homogeneous = suffix @ state0
    reconstructed = homogeneous + sum(
        source_contributions, start=np.zeros_like(state0)
    )
    return {
        "endpoint": direct,
        "states": np.stack(states),
        "homogeneous": homogeneous,
        "source_contributions": np.stack(source_contributions)
        if source_contributions
        else np.empty((0, dimension), dtype=np.float64),
        "reconstruction": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(direct - reconstructed))),
    }


def centered_energy(rows: Any) -> dict[str, Any]:
    """Return centered rows, centered energy, and all-pair diameter."""

    value = _array(rows, "rows")
    if value.ndim < 2 or value.shape[-2] < 2:
        raise ValueError("row maps need at least two rows")
    centered = value - np.mean(value, axis=-2, keepdims=True)
    energy = np.sum(centered * centered, axis=(-2, -1))
    row_count = value.shape[-2]
    pairs = np.asarray(
        [(left, right) for left in range(row_count)
         for right in range(left + 1, row_count)],
        dtype=np.int64,
    )
    contrasts = (
        value[..., pairs[:, 0], :] - value[..., pairs[:, 1], :]
    )
    pair_energy = np.sum(contrasts * contrasts, axis=-1)
    diameter_squared = np.max(pair_energy, axis=-1)
    return {
        "centered_rows": centered,
        "energy": energy,
        "diameter_squared": diameter_squared,
        "pair_energy": pair_energy,
        "pair_indices": pairs,
        "pair_identity_residual": np.sum(pair_energy, axis=-1)
        - row_count * energy,
    }


def exact_energy_balance(rows0: Any, rows1: Any) -> dict[str, Any]:
    """Evaluate exact centered-energy polarization at two endpoints."""

    source = _array(rows0, "rows0")
    endpoint = _array(rows1, "rows1")
    _same_shape(source, endpoint)
    source_geometry = centered_energy(source)
    endpoint_geometry = centered_energy(endpoint)
    centered_increment = (
        endpoint_geometry["centered_rows"]
        - source_geometry["centered_rows"]
    )
    linear = 2.0 * np.sum(
        source_geometry["centered_rows"] * centered_increment,
        axis=(-2, -1),
    )
    quadratic = np.sum(
        centered_increment * centered_increment, axis=(-2, -1)
    )
    direct = endpoint_geometry["energy"] - source_geometry["energy"]
    return {
        "source_energy": source_geometry["energy"],
        "endpoint_energy": endpoint_geometry["energy"],
        "centered_increment": centered_increment,
        "linear_work": linear,
        "quadratic_charge": quadratic,
        "energy_increment": direct,
        "reconstruction": linear + quadratic,
        "maximum_abs_residual": float(
            np.max(np.abs(direct - linear - quadratic))
        ),
    }


def named_transport_ledger(
    rows0: Any,
    rows1: Any,
    contributions: Mapping[str, Any],
) -> dict[str, Any]:
    """Check a disjoint named centered-row source partition."""

    if not contributions:
        raise ValueError("finite transport needs at least one named source")
    source = _array(rows0, "rows0")
    endpoint = _array(rows1, "rows1")
    _same_shape(source, endpoint)
    names = tuple(contributions)
    values = tuple(
        _array(contributions[name], f"contribution {name}") for name in names
    )
    if any(value.shape != source.shape for value in values):
        raise ValueError("a named source has the wrong row-map shape")
    reconstructed_increment = sum(values, start=np.zeros_like(source))
    native_increment = endpoint - source
    source_centered = centered_energy(source)["centered_rows"]
    centered_values = tuple(
        value - np.mean(value, axis=-2, keepdims=True) for value in values
    )
    linear_work = {
        name: 2.0 * np.sum(source_centered * value, axis=(-2, -1))
        for name, value in zip(names, centered_values, strict=True)
    }
    balance = exact_energy_balance(source, endpoint)
    reconstruction_residual = native_increment - reconstructed_increment
    centered_residual = (
        reconstruction_residual
        - np.mean(reconstruction_residual, axis=-2, keepdims=True)
    )
    return {
        "source_names": list(names),
        "reconstructed_increment": reconstructed_increment,
        "native_increment": native_increment,
        "maximum_abs_transport_residual": float(
            np.max(np.abs(reconstruction_residual))
        ),
        "centered_transport_residual_norm": float(
            np.linalg.norm(centered_residual)
        ),
        "linear_work": linear_work,
        "quadratic_charge": balance["quadratic_charge"],
        "maximum_abs_energy_residual": balance["maximum_abs_residual"],
    }

