"""Crossing-safe finite-increment kernels for PLDR row-map collapse.

The functions evaluate exact endpoint telescopes in float64 and provide
serial, directed ``nextafter`` enclosures for pair energies, diameters, and
source-state margins.  The campaign producer records their proof obligations.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable

import numpy as np


SCHEMA_VERSION = "pldr-finite-increment-kernel-v1"


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.ascontiguousarray(array)


def _nonnegative(value: Any, name: str) -> np.ndarray:
    array = _array(value, name)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _same_shape(*arrays: np.ndarray) -> None:
    if not arrays or any(array.shape != arrays[0].shape for array in arrays[1:]):
        raise ValueError("finite-increment arrays must have matching shapes")


def iswiglu(value: Any) -> np.ndarray:
    """Evaluate the implemented identity-weight SwiGLU, ``z^2 sigmoid(z)``."""

    z = _array(value, "value")
    sigmoid = np.empty_like(z)
    positive = z >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return z * z * sigmoid


def iswiglu_derivative(value: Any) -> np.ndarray:
    """Derivative of :func:`iswiglu` with a cancellation-safe sigmoid."""

    z = _array(value, "value")
    sigmoid = np.empty_like(z)
    positive = z >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exponential = np.exp(z[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return 2.0 * z * sigmoid + z * z * sigmoid * (1.0 - sigmoid)


def positive_power(base: Any, exponent: Any) -> np.ndarray:
    """Evaluate ``exp(exponent * log(base))`` on a strictly positive base."""

    value = _array(base, "base")
    power = _array(exponent, "exponent")
    _same_shape(value, power)
    if np.any(value <= 0.0):
        raise ValueError("positive_power requires a strictly positive base")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.exp(power * np.log(value))
        except FloatingPointError as error:
            raise FloatingPointError("positive real power overflowed") from error
    if not np.isfinite(result).all():
        raise FloatingPointError("positive real power is nonfinite")
    return result


def _secant(
    left: np.ndarray,
    right: np.ndarray,
    left_value: np.ndarray,
    right_value: np.ndarray,
    diagonal: np.ndarray,
) -> np.ndarray:
    _same_shape(left, right, left_value, right_value, diagonal)
    equal = left == right
    coefficient = np.empty_like(left)
    coefficient[equal] = diagonal[equal]
    coefficient[~equal] = (
        (right_value[~equal] - left_value[~equal])
        / (right[~equal] - left[~equal])
    )
    if not np.isfinite(coefficient).all():
        raise FloatingPointError("a divided difference is nonfinite")
    return coefficient


def two_variable_power_increment(
    preactivation_source: Any,
    preactivation_endpoint: Any,
    exponent_source: Any,
    exponent_endpoint: Any,
    *,
    epsilon: float = 1.0e-9,
) -> dict[str, np.ndarray | float | int]:
    """Evaluate both exact telescopes for the implemented PLGA power.

    The base-first telescope changes the preactivation at the endpoint
    exponent and then changes the exponent at the source preactivation.
    The exponent-first telescope uses the opposite path.  Both remain finite
    when the preactivation segment crosses zero because ``epsilon`` keeps the
    powered base strictly positive.
    """

    z0 = _array(preactivation_source, "preactivation_source")
    z1 = _array(preactivation_endpoint, "preactivation_endpoint")
    p0 = _array(exponent_source, "exponent_source")
    p1 = _array(exponent_endpoint, "exponent_endpoint")
    _same_shape(z0, z1, p0, p1)
    floor = float(epsilon)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("the PLGA adjustment epsilon must be positive")

    base0 = iswiglu(z0) + floor
    base1 = iswiglu(z1) + floor
    phi00 = positive_power(base0, p0)
    phi01 = positive_power(base0, p1)
    phi10 = positive_power(base1, p0)
    phi11 = positive_power(base1, p1)

    dz = z1 - z0
    dp = p1 - p0
    sz_at_p1 = _secant(
        z0,
        z1,
        phi01,
        phi11,
        p1 * positive_power(base0, p1 - 1.0) * iswiglu_derivative(z0),
    )
    sp_at_z0 = _secant(
        p0,
        p1,
        phi00,
        phi01,
        phi00 * np.log(base0),
    )
    sp_at_z1 = _secant(
        p0,
        p1,
        phi10,
        phi11,
        phi10 * np.log(base1),
    )
    sz_at_p0 = _secant(
        z0,
        z1,
        phi00,
        phi10,
        p0 * positive_power(base0, p0 - 1.0) * iswiglu_derivative(z0),
    )

    base_first_z = sz_at_p1 * dz
    base_first_p = sp_at_z0 * dp
    exponent_first_p = sp_at_z1 * dp
    exponent_first_z = sz_at_p0 * dz
    direct = phi11 - phi00
    base_first = base_first_z + base_first_p
    exponent_first = exponent_first_p + exponent_first_z
    scale = max(float(np.max(np.abs(direct))), 1.0)

    return {
        "source": phi00,
        "endpoint": phi11,
        "direct_increment": direct,
        "base_first_z_secant": sz_at_p1,
        "base_first_p_secant": sp_at_z0,
        "base_first_z_contribution": base_first_z,
        "base_first_p_contribution": base_first_p,
        "base_first_increment": base_first,
        "exponent_first_p_secant": sp_at_z1,
        "exponent_first_z_secant": sz_at_p0,
        "exponent_first_p_contribution": exponent_first_p,
        "exponent_first_z_contribution": exponent_first_z,
        "exponent_first_increment": exponent_first,
        "base_first_max_abs_residual": float(
            np.max(np.abs(base_first - direct))
        ),
        "exponent_first_max_abs_residual": float(
            np.max(np.abs(exponent_first - direct))
        ),
        "relative_residual_scale": scale,
        "strict_crossing_count": int(np.count_nonzero(z0 * z1 < 0.0)),
        "touching_zero_count": int(np.count_nonzero((z0 == 0.0) | (z1 == 0.0))),
        "minimum_base": float(min(np.min(base0), np.min(base1))),
        "maximum_abs_z_secant": float(max(
            np.max(np.abs(sz_at_p1)), np.max(np.abs(sz_at_p0))
        )),
        "maximum_abs_p_secant": float(max(
            np.max(np.abs(sp_at_z0)), np.max(np.abs(sp_at_z1))
        )),
        "maximum_abs_multiplied_contribution": float(max(
            np.max(np.abs(base_first_z)),
            np.max(np.abs(base_first_p)),
            np.max(np.abs(exponent_first_z)),
            np.max(np.abs(exponent_first_p)),
        )),
    }


def affine_increment(
    input_source: Any,
    input_endpoint: Any,
    weight_source: Any,
    weight_endpoint: Any,
    bias_source: Any,
    bias_endpoint: Any,
) -> dict[str, np.ndarray | float]:
    """Exact finite increment of ``weight @ input + bias``."""

    x0 = _array(input_source, "input_source")
    x1 = _array(input_endpoint, "input_endpoint")
    w0 = _array(weight_source, "weight_source")
    w1 = _array(weight_endpoint, "weight_endpoint")
    b0 = _array(bias_source, "bias_source")
    b1 = _array(bias_endpoint, "bias_endpoint")
    direct = np.matmul(w1, x1) + b1 - (np.matmul(w0, x0) + b0)
    input_contribution = np.matmul(w1, x1 - x0)
    weight_contribution = np.matmul(w1 - w0, x0)
    bias_contribution = b1 - b0
    reconstructed = input_contribution + weight_contribution + bias_contribution
    return {
        "direct_increment": direct,
        "input_contribution": input_contribution,
        "weight_contribution": weight_contribution,
        "bias_contribution": bias_contribution,
        "reconstructed_increment": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(reconstructed - direct))),
    }


def layernorm_shape(value: Any, *, epsilon: float = 1.0e-6) -> np.ndarray:
    """Explicit LayerNorm shape with the implemented population variance."""

    x = _array(value, "value")
    if x.ndim < 1 or x.shape[-1] < 1:
        raise ValueError("LayerNorm needs a nonempty final dimension")
    floor = float(epsilon)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("LayerNorm epsilon must be positive")
    centered = x - np.mean(x, axis=-1, keepdims=True)
    radius = np.sqrt(floor + np.mean(centered * centered, axis=-1, keepdims=True))
    return centered / radius


def stable_layernorm_chord(
    source: Any,
    endpoint: Any,
    *,
    epsilon: float = 1.0e-6,
) -> dict[str, np.ndarray | float]:
    """Construct the exact stable LayerNorm chord on the final dimension."""

    x0 = _array(source, "source")
    x1 = _array(endpoint, "endpoint")
    _same_shape(x0, x1)
    if x0.ndim != 1 or len(x0) < 1:
        raise ValueError("stable_layernorm_chord expects two vectors")
    dimension = len(x0)
    floor = float(epsilon)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("LayerNorm epsilon must be positive")
    projection = np.eye(dimension) - np.ones((dimension, dimension)) / dimension
    c0 = projection @ x0
    c1 = projection @ x1
    s0 = math.sqrt(floor + float(c0 @ c0) / dimension)
    s1 = math.sqrt(floor + float(c1 @ c1) / dimension)
    chord = (
        projection / s1
        - np.outer(c0, c1 + c0) @ projection
        / (dimension * s1 * s0 * (s1 + s0))
    )
    direct = c1 / s1 - c0 / s0
    reconstructed = chord @ (x1 - x0)
    return {
        "chord": chord,
        "direct_increment": direct,
        "reconstructed_increment": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(reconstructed - direct))),
        "source_radius": s0,
        "endpoint_radius": s1,
    }


def affine_layernorm_increment(
    preactivation_source: Any,
    preactivation_endpoint: Any,
    gain_source: Any,
    gain_endpoint: Any,
    bias_source: Any,
    bias_endpoint: Any,
    *,
    epsilon: float = 1.0e-6,
) -> dict[str, np.ndarray | float]:
    """Exact endpoint split of affine LayerNorm into shape, gain, and bias."""

    w0 = _array(preactivation_source, "preactivation_source", 1)
    w1 = _array(preactivation_endpoint, "preactivation_endpoint", 1)
    gamma0 = _array(gain_source, "gain_source", 1)
    gamma1 = _array(gain_endpoint, "gain_endpoint", 1)
    beta0 = _array(bias_source, "bias_source", 1)
    beta1 = _array(bias_endpoint, "bias_endpoint", 1)
    _same_shape(w0, w1, gamma0, gamma1, beta0, beta1)
    chord = stable_layernorm_chord(w0, w1, epsilon=epsilon)
    shape0 = layernorm_shape(w0, epsilon=epsilon)
    shape1 = layernorm_shape(w1, epsilon=epsilon)
    direct = gamma1 * shape1 + beta1 - (gamma0 * shape0 + beta0)
    shape_contribution = gamma1 * np.asarray(chord["reconstructed_increment"])
    gain_contribution = (gamma1 - gamma0) * shape0
    bias_contribution = beta1 - beta0
    reconstructed = shape_contribution + gain_contribution + bias_contribution
    return {
        "direct_increment": direct,
        "shape_contribution": shape_contribution,
        "gain_contribution": gain_contribution,
        "bias_contribution": bias_contribution,
        "reconstructed_increment": reconstructed,
        "maximum_abs_residual": float(np.max(np.abs(reconstructed - direct))),
        "shape_chord_residual": float(chord["maximum_abs_residual"]),
    }


def finite_source_unroll(
    initial_increment: Any,
    linear_maps: Iterable[Any],
    local_sources: Iterable[Any],
) -> dict[str, np.ndarray | list[np.ndarray]]:
    """Execute and explicitly unroll a chronological affine recurrence."""

    initial = _array(initial_increment, "initial_increment", 1)
    maps = [_array(value, "linear_map", 2) for value in linear_maps]
    sources = [_array(value, "local_source", 1) for value in local_sources]
    if len(maps) != len(sources):
        raise ValueError("the finite source chain needs one source per map")
    if any(
        matrix.shape != (len(initial), len(initial)) for matrix in maps
    ) or any(source.shape != initial.shape for source in sources):
        raise ValueError("finite source chain dimensions disagree")

    states = [initial]
    for matrix, source in zip(maps, sources, strict=True):
        states.append(matrix @ states[-1] + source)

    homogeneous = initial.copy()
    for matrix in maps:
        homogeneous = matrix @ homogeneous
    transported = []
    for index, source in enumerate(sources):
        value = source.copy()
        for matrix in maps[index + 1:]:
            value = matrix @ value
        transported.append(value)
    unrolled = homogeneous + sum(transported, start=np.zeros_like(initial))
    return {
        "states": states,
        "homogeneous": homogeneous,
        "transported_sources": transported,
        "unrolled": unrolled,
        "recursive": states[-1],
    }


def unordered_pair_indices(row_count: int) -> np.ndarray:
    """Return every unordered row pair in deterministic lexicographic order."""

    count = int(row_count)
    if count < 2:
        raise ValueError("a row map needs at least two rows")
    left, right = np.triu_indices(count, k=1)
    return np.column_stack((left, right)).astype(np.int64, copy=False)


def pair_energies(rows: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return the within-map pair registry and squared pair energies."""

    value = _array(rows, "rows", 2)
    pairs = unordered_pair_indices(value.shape[0])
    contrasts = value[pairs[:, 0]] - value[pairs[:, 1]]
    energies = np.einsum("pd,pd->p", contrasts, contrasts, optimize=True)
    return pairs, energies


def _round_down(value: Any) -> np.ndarray:
    """Move a finite float64 result one representable number downward."""

    result = np.nextafter(np.asarray(value, dtype=np.float64), -np.inf)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "directed lower rounding produced a nonfinite value"
        )
    return result


def _round_up(value: Any) -> np.ndarray:
    """Move a finite float64 result one representable number upward."""

    result = np.nextafter(np.asarray(value, dtype=np.float64), np.inf)
    if not np.isfinite(result).all():
        raise FloatingPointError(
            "directed upper rounding produced a nonfinite value"
        )
    return result


def _interval_subtract(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        _round_down(left_lower - right_upper),
        _round_up(left_upper - right_lower),
    )


def _interval_multiply(
    left_lower: np.ndarray,
    left_upper: np.ndarray,
    right_lower: np.ndarray,
    right_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    products = np.stack((
        left_lower * right_lower,
        left_lower * right_upper,
        left_upper * right_lower,
        left_upper * right_upper,
    ))
    return _round_down(np.min(products, axis=0)), _round_up(
        np.max(products, axis=0)
    )


def _interval_square(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if np.any(lower > upper):
        raise ValueError(
            "an interval lower endpoint exceeds its upper endpoint"
        )
    crosses_zero = (lower <= 0.0) & (upper >= 0.0)
    minimum_magnitude = np.minimum(np.abs(lower), np.abs(upper))
    minimum_magnitude = np.where(crosses_zero, 0.0, minimum_magnitude)
    maximum_magnitude = np.maximum(np.abs(lower), np.abs(upper))
    raw_lower = minimum_magnitude * minimum_magnitude
    squared_lower = np.where(
        minimum_magnitude == 0.0,
        0.0,
        _round_down(raw_lower),
    )
    return squared_lower, _round_up(
        maximum_magnitude * maximum_magnitude
    )


def _interval_sum_last(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if lower.shape != upper.shape or lower.ndim < 1:
        raise ValueError(
            "interval sums need matching non-scalar arrays"
        )
    sum_lower = np.zeros(lower.shape[:-1], dtype=np.float64)
    sum_upper = np.zeros(upper.shape[:-1], dtype=np.float64)
    for index in range(lower.shape[-1]):
        sum_lower = np.maximum(
            _round_down(sum_lower + lower[..., index]), 0.0
        )
        sum_upper = _round_up(sum_upper + upper[..., index])
    return sum_lower, sum_upper


def _interval_energy(
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    squared_lower, squared_upper = _interval_square(lower, upper)
    return _interval_sum_last(squared_lower, squared_upper)


def _pair_contrast_interval(
    rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pairs = unordered_pair_indices(rows.shape[0])
    left = rows[pairs[:, 0]]
    right = rows[pairs[:, 1]]
    return _interval_subtract(left, left, right, right)


def outward_pair_energy_bounds(rows: Any) -> dict[str, Any]:
    """Enclose every within-map pair energy by directed float64 arithmetic.

    The kernel uses only scalar IEEE-754 binary64 operations followed by one
    outward nextafter step. Reductions are performed in a fixed serial order,
    never by BLAS or an implementation-dependent parallel reduction.
    """

    value = _array(rows, "rows", 2)
    pairs = unordered_pair_indices(value.shape[0])
    contrast_lower, contrast_upper = _pair_contrast_interval(value)
    energy_lower, energy_upper = _interval_energy(
        contrast_lower, contrast_upper
    )
    if (
        np.any(energy_lower > energy_upper)
        or np.any(energy_lower < 0.0)
    ):
        raise AssertionError(
            "directed pair-energy intervals are invalid"
        )
    return {
        "pairs": pairs,
        "lower": energy_lower,
        "upper": energy_upper,
        "diameter_lower": float(np.max(energy_lower)),
        "diameter_upper": float(np.max(energy_upper)),
        "backend": "float64-directed-nextafter-v1",
    }


def directed_state_margin_bounds(
    shape_source_rows: Any,
    shape_endpoint_rows: Any,
    gain_source: Any,
    gain_endpoint: Any,
    *,
    source_rows: Any | None = None,
    endpoint_rows: Any | None = None,
) -> dict[str, Any]:
    """Outward enclosure of the finite gate-shape endpoint margin."""

    shape0 = _array(shape_source_rows, "shape_source_rows", 2)
    shape1 = _array(shape_endpoint_rows, "shape_endpoint_rows", 2)
    gamma0 = _array(gain_source, "gain_source", 1)
    gamma1 = _array(gain_endpoint, "gain_endpoint", 1)
    _same_shape(shape0, shape1)
    if shape0.shape[1] != len(gamma0) or gamma0.shape != gamma1.shape:
        raise ValueError("shape and gain dimensions disagree")

    contrast0_lower, contrast0_upper = _pair_contrast_interval(shape0)
    contrast1_lower, contrast1_upper = _pair_contrast_interval(shape1)
    gamma0_rows = np.broadcast_to(gamma0, contrast0_lower.shape)
    gamma1_rows = np.broadcast_to(gamma1, contrast0_lower.shape)
    if source_rows is None:
        physical0_lower, physical0_upper = _interval_multiply(
            contrast0_lower, contrast0_upper, gamma0_rows, gamma0_rows
        )
    else:
        source_physical = _array(source_rows, "source_rows", 2)
        _same_shape(source_physical, shape0)
        physical0_lower, physical0_upper = _pair_contrast_interval(
            source_physical
        )
    if endpoint_rows is None:
        physical1_lower, physical1_upper = _interval_multiply(
            contrast1_lower, contrast1_upper, gamma1_rows, gamma1_rows
        )
    else:
        endpoint_physical = _array(endpoint_rows, "endpoint_rows", 2)
        _same_shape(endpoint_physical, shape1)
        physical1_lower, physical1_upper = _pair_contrast_interval(
            endpoint_physical
        )
    fixed_lower, fixed_upper = _interval_multiply(
        contrast0_lower, contrast0_upper, gamma1_rows, gamma1_rows
    )
    residual_lower, residual_upper = _interval_subtract(
        physical1_lower,
        physical1_upper,
        fixed_lower,
        fixed_upper,
    )

    source_lower, source_upper = _interval_energy(
        physical0_lower, physical0_upper
    )
    endpoint_lower, endpoint_upper = _interval_energy(
        physical1_lower, physical1_upper
    )
    fixed_energy_lower, fixed_energy_upper = _interval_energy(
        fixed_lower, fixed_upper
    )
    residual_energy_lower, residual_energy_upper = _interval_energy(
        residual_lower, residual_upper
    )

    source_diameter_lower = float(np.max(source_lower))
    source_diameter_upper = float(np.max(source_upper))
    endpoint_diameter_lower = float(np.max(endpoint_lower))
    endpoint_diameter_upper = float(np.max(endpoint_upper))
    gap_lower = _round_down(source_diameter_lower - source_upper)
    gap_lower = np.maximum(gap_lower, 0.0)
    gate_decrement_lower = _round_down(
        source_lower - fixed_energy_upper
    )

    fixed_radius_upper = _round_up(np.sqrt(fixed_energy_upper))
    residual_radius_upper = _round_up(
        np.sqrt(residual_energy_upper)
    )
    cross_upper = _round_up(
        fixed_radius_upper * residual_radius_upper
    )
    cross_upper = _round_up(2.0 * cross_upper)
    shape_bound_upper = _round_up(
        cross_upper + residual_energy_upper
    )
    partial_margin_lower = _round_down(
        gap_lower + gate_decrement_lower
    )
    margin_lower = _round_down(
        partial_margin_lower - shape_bound_upper
    )
    minimum_margin_lower = float(np.min(margin_lower))
    state_bound_upper = float(_round_up(
        source_diameter_upper - minimum_margin_lower
    ))
    alpha_lower = None
    if minimum_margin_lower > 0.0 and source_diameter_upper > 0.0:
        alpha_lower = float(_round_down(
            minimum_margin_lower / source_diameter_upper
        ))

    return {
        "source_energy_lower": source_lower,
        "source_energy_upper": source_upper,
        "endpoint_energy_lower": endpoint_lower,
        "endpoint_energy_upper": endpoint_upper,
        "fixed_energy_lower": fixed_energy_lower,
        "fixed_energy_upper": fixed_energy_upper,
        "residual_energy_lower": residual_energy_lower,
        "residual_energy_upper": residual_energy_upper,
        "source_gap_lower": gap_lower,
        "gate_decrement_lower": gate_decrement_lower,
        "shape_bound_upper": shape_bound_upper,
        "state_margin_lower": margin_lower,
        "minimum_state_margin_lower": minimum_margin_lower,
        "source_diameter_squared_lower": source_diameter_lower,
        "source_diameter_squared_upper": source_diameter_upper,
        "endpoint_diameter_squared_lower": endpoint_diameter_lower,
        "endpoint_diameter_squared_upper": endpoint_diameter_upper,
        "state_bound_endpoint_squared_upper": state_bound_upper,
        "state_bound_covers_endpoint": (
            endpoint_diameter_upper <= state_bound_upper
        ),
        "strict_contraction_certified": minimum_margin_lower > 0.0,
        "alpha_star_zero_forcing_lower": alpha_lower,
        "backend": "float64-directed-nextafter-v1",
        "backend_obligations": [
            "IEEE-754 binary64 basic operations are correctly rounded",
            "numpy.nextafter returns the adjacent binary64 number",
            "numpy.sqrt is correctly rounded",
            "all inputs and intermediate interval endpoints are finite",
        ],
    }


def per_map_diameters(rows: Any) -> dict[str, Any]:
    """Compute one diameter for each map without ever forming cross-map pairs."""

    maps = _array(rows, "rows", 3)
    if maps.shape[1] < 2:
        raise ValueError("every map needs at least two rows")
    diameter_squared = []
    maximizing_pairs = []
    for matrix in maps:
        pairs, energies = pair_energies(matrix)
        maximum = float(np.max(energies))
        candidates = np.flatnonzero(energies == maximum)
        index = int(candidates[0])
        diameter_squared.append(maximum)
        maximizing_pairs.append([int(value) for value in pairs[index]])
    return {
        "diameter_squared": np.asarray(diameter_squared),
        "maximizing_pairs": maximizing_pairs,
        "aggregate_diameter_squared": float(max(diameter_squared)),
        "map_count": int(len(maps)),
        "pairs_per_map": int(maps.shape[1] * (maps.shape[1] - 1) // 2),
        "cross_map_pairs_formed": False,
    }


def exact_diameter_slack(source_rows: Any, endpoint_rows: Any) -> dict[str, Any]:
    """Evaluate the exact finite maximum-level slack law for one row map."""

    source = _array(source_rows, "source_rows", 2)
    endpoint = _array(endpoint_rows, "endpoint_rows", 2)
    _same_shape(source, endpoint)
    pairs, source_energy = pair_energies(source)
    endpoint_pairs, endpoint_energy = pair_energies(endpoint)
    if not np.array_equal(pairs, endpoint_pairs):
        raise AssertionError("source and endpoint pair registries changed")
    source_diameter = float(np.max(source_energy))
    endpoint_diameter = float(np.max(endpoint_energy))
    source_gap = source_diameter - source_energy
    decrement = source_energy - endpoint_energy
    slack = source_gap + decrement
    minimum_slack = float(np.min(slack))
    return {
        "pairs": pairs,
        "source_energy": source_energy,
        "endpoint_energy": endpoint_energy,
        "source_gap": source_gap,
        "pair_decrement": decrement,
        "slack": slack,
        "source_diameter_squared": source_diameter,
        "endpoint_diameter_squared": endpoint_diameter,
        "minimum_slack": minimum_slack,
        "identity_residual": float(abs(
            endpoint_diameter - (source_diameter - minimum_slack)
        )),
        "strict_contraction": endpoint_diameter < source_diameter,
    }


def gate_shape_state_margin(
    shape_source_rows: Any,
    shape_endpoint_rows: Any,
    gain_source: Any,
    gain_endpoint: Any,
    *,
    decay_multiplier: float,
    arithmetic_charge: Any = 0.0,
) -> dict[str, Any]:
    """Evaluate the exact gate terms and the finite shape-work upper bound.

    ``innovation`` is defined by ``gain_endpoint = decay_multiplier *
    gain_source - innovation``.  This makes endpoint rounding part of the
    executed innovation and keeps the gate balance exact.
    """

    shape0 = _array(shape_source_rows, "shape_source_rows", 2)
    shape1 = _array(shape_endpoint_rows, "shape_endpoint_rows", 2)
    gamma0 = _array(gain_source, "gain_source", 1)
    gamma1 = _array(gain_endpoint, "gain_endpoint", 1)
    _same_shape(shape0, shape1)
    if shape0.shape[1] != len(gamma0) or gamma0.shape != gamma1.shape:
        raise ValueError("shape and gain dimensions disagree")
    decay = float(decay_multiplier)
    if not math.isfinite(decay):
        raise ValueError("decay_multiplier must be finite")

    pairs = unordered_pair_indices(shape0.shape[0])
    contrast0 = shape0[pairs[:, 0]] - shape0[pairs[:, 1]]
    contrast1 = shape1[pairs[:, 0]] - shape1[pairs[:, 1]]
    physical0 = contrast0 * gamma0
    physical1 = contrast1 * gamma1
    fixed_endpoint_gate = contrast0 * gamma1
    source_energy = np.einsum("pd,pd->p", physical0, physical0, optimize=True)
    endpoint_energy = np.einsum("pd,pd->p", physical1, physical1, optimize=True)
    fixed_energy = np.einsum(
        "pd,pd->p", fixed_endpoint_gate, fixed_endpoint_gate, optimize=True
    )
    source_diameter = float(np.max(source_energy))
    source_gap = source_diameter - source_energy

    decayed_gain = decay * gamma0
    innovation = decayed_gain - gamma1
    weights = contrast0 * contrast0
    decay_dissipation = np.einsum(
        "pd,d->p", weights, gamma0 * gamma0 - decayed_gain * decayed_gain,
        optimize=True,
    )
    adaptive_alignment = 2.0 * np.einsum(
        "pd,d->p", weights, decayed_gain * innovation, optimize=True
    )
    innovation_energy = np.einsum(
        "pd,d->p", weights, innovation * innovation, optimize=True
    )
    gate_decrement = source_energy - fixed_energy
    gate_balance = decay_dissipation + adaptive_alignment - innovation_energy

    weighted_shape_increment = gamma1 * (contrast1 - contrast0)
    shape_radius = np.linalg.norm(weighted_shape_increment, axis=1)
    shape_work = endpoint_energy - fixed_energy
    shape_work_upper = 2.0 * np.sqrt(np.maximum(fixed_energy, 0.0)) * shape_radius
    shape_work_upper += shape_radius * shape_radius
    charge = _nonnegative(arithmetic_charge, "arithmetic_charge")
    try:
        charge = np.broadcast_to(charge, source_energy.shape)
    except ValueError as error:
        raise ValueError("arithmetic_charge does not broadcast over pairs") from error
    margin = (
        source_gap
        + decay_dissipation
        + adaptive_alignment
        - innovation_energy
        - shape_work_upper
        - charge
    )
    minimum_margin = float(np.min(margin))
    endpoint_diameter = float(np.max(endpoint_energy))
    alpha_star = (
        minimum_margin / source_diameter if source_diameter > 0.0 else None
    )
    return {
        "pairs": pairs,
        "source_energy": source_energy,
        "endpoint_energy": endpoint_energy,
        "fixed_source_shape_endpoint_energy": fixed_energy,
        "source_gap": source_gap,
        "decay_dissipation": decay_dissipation,
        "adaptive_alignment": adaptive_alignment,
        "innovation_energy": innovation_energy,
        "gate_decrement": gate_decrement,
        "gate_balance": gate_balance,
        "gate_balance_max_abs_residual": float(np.max(np.abs(
            gate_decrement - gate_balance
        ))),
        "shape_work": shape_work,
        "shape_radius": shape_radius,
        "shape_work_upper": shape_work_upper,
        "shape_bound_minimum_slack": float(np.min(
            shape_work_upper - shape_work
        )),
        "arithmetic_charge": np.ascontiguousarray(charge),
        "state_margin": margin,
        "minimum_state_margin": minimum_margin,
        "source_diameter_squared": source_diameter,
        "endpoint_diameter_squared": endpoint_diameter,
        "alpha_star_zero_forcing": alpha_star,
        "state_bound_endpoint_squared": source_diameter - minimum_margin,
        "state_bound_covers_endpoint": bool(
            endpoint_diameter <= source_diameter - minimum_margin
            + 64.0 * np.finfo(np.float64).eps * max(source_diameter, 1.0)
        ),
        "strict_contraction_certified": minimum_margin > 0.0,
    }


def absolute_diameter_enclosure(
    pair_lower: Any,
    pair_upper: Any,
    *,
    source_lower: float | None = None,
    source_upper: float | None = None,
) -> dict[str, float | bool | None]:
    """Take a finite pair maximum and retain absolute bounds near collapse."""

    lower = _array(pair_lower, "pair_lower", 1)
    upper = _array(pair_upper, "pair_upper", 1)
    _same_shape(lower, upper)
    if not len(lower) or np.any(lower > upper):
        raise ValueError("pair bounds must be nonempty and ordered")
    diameter_lower = max(0.0, float(np.max(lower)))
    diameter_upper = max(0.0, float(np.max(upper)))
    lower_ratio = None
    upper_ratio = None
    denominator_valid = False
    if source_lower is not None or source_upper is not None:
        if source_lower is None or source_upper is None:
            raise ValueError("both source diameter bounds are required")
        lo = float(source_lower)
        hi = float(source_upper)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError("source diameter bounds are invalid")
        denominator_valid = lo > 0.0
        if denominator_valid:
            lower_ratio = max(0.0, float(np.nextafter(
                diameter_lower / hi, -np.inf
            )))
            upper_ratio = float(np.nextafter(
                diameter_upper / lo, np.inf
            ))
    return {
        "absolute_lower": diameter_lower,
        "absolute_upper": diameter_upper,
        "ratio_denominator_valid": denominator_valid,
        "lower_ratio": lower_ratio,
        "upper_ratio": upper_ratio,
    }


def product_convolution(
    initial: float,
    gains: Iterable[float],
    forcings: Iterable[float],
) -> dict[str, list[float] | float]:
    """Evaluate the chronological affine product-convolution recurrence."""

    start = float(initial)
    q = [float(value) for value in gains]
    beta = [float(value) for value in forcings]
    if (
        not math.isfinite(start)
        or start < 0.0
        or len(q) != len(beta)
        or any(not math.isfinite(value) or value < 0.0 for value in q + beta)
    ):
        raise ValueError("product-convolution inputs must be finite and nonnegative")
    values = [start]
    homogeneous = [start]
    current = start
    product = 1.0
    for gain, forcing in zip(q, beta, strict=True):
        current = gain * current + forcing
        product *= gain
        values.append(current)
        homogeneous.append(product * start)
    return {
        "initial": start,
        "gains": q,
        "forcings": beta,
        "envelope": values,
        "homogeneous": homogeneous,
        "final": values[-1],
    }


def representative_crossing_qualification() -> dict[str, Any]:
    """Run deterministic scalar and matrix identities used by campaign E0."""

    z0 = np.asarray([-2.0e-5, -1.0e-8, 0.0, 3.0e-5])
    z1 = np.asarray([3.0e-5, 2.0e-8, -2.0e-8, -4.0e-5])
    p0 = np.asarray([-0.10, 0.04, 0.0, 0.11])
    p1 = np.asarray([-0.09, 0.03, 0.01, 0.10])
    power = two_variable_power_increment(z0, z1, p0, p1)

    rng = np.random.default_rng(42001)
    x0 = rng.normal(size=7)
    x1 = x0 + rng.normal(scale=0.05, size=7)
    gamma0 = rng.normal(size=7)
    gamma1 = gamma0 + rng.normal(scale=0.01, size=7)
    beta0 = rng.normal(size=7)
    beta1 = beta0 + rng.normal(scale=0.01, size=7)
    layernorm = affine_layernorm_increment(
        x0, x1, gamma0, gamma1, beta0, beta1
    )

    u0 = rng.normal(size=4)
    u1 = u0 + rng.normal(scale=0.04, size=4)
    v0 = rng.normal(size=3)
    v1 = v0 + rng.normal(scale=0.04, size=3)
    bilinear_direct = np.outer(u1, v1) - np.outer(u0, v0)
    bilinear_reconstructed = (
        np.outer(u1, v1 - v0) + np.outer(u1 - u0, v0)
    )
    bilinear_residual = float(np.max(np.abs(
        bilinear_reconstructed - bilinear_direct
    )))

    maps = [rng.normal(scale=0.2, size=(7, 7)) for _ in range(8)]
    sources = [rng.normal(scale=0.02, size=7) for _ in range(8)]
    chain = finite_source_unroll(rng.normal(size=7), maps, sources)
    chain_residual = float(np.max(np.abs(
        np.asarray(chain["recursive"]) - np.asarray(chain["unrolled"])
    )))

    shape = rng.normal(size=(6, 7))
    gate0 = np.ones(7)
    gate1 = np.full(7, 0.98)
    rows0 = shape * gate0
    rows1 = shape * gate1
    diameter = exact_diameter_slack(rows0, rows1)
    state = gate_shape_state_margin(
        shape, shape, gate0, gate1, decay_multiplier=0.98
    )
    directed = directed_state_margin_bounds(
        shape,
        shape,
        gate0,
        gate1,
        source_rows=rows0,
        endpoint_rows=rows1,
    )
    convolution = product_convolution(1.0, [0.5, 0.4], [0.1, 0.2])
    convolution_residual = abs(float(convolution["final"]) - 0.44)

    malformed = (
        lambda: positive_power(np.asarray([0.0]), np.asarray([1.0])),
        lambda: per_map_diameters(np.zeros((1, 1, 2))),
        lambda: finite_source_unroll(
            np.zeros(2), [np.eye(3)], [np.zeros(2)]
        ),
        lambda: two_variable_power_increment(
            np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), epsilon=0.0
        ),
    )
    malformed_rejection_count = 0
    for operation in malformed:
        try:
            operation()
        except (ValueError, FloatingPointError):
            malformed_rejection_count += 1

    tolerance = 256.0 * np.finfo(np.float64).eps
    passed = bool(
        float(power["base_first_max_abs_residual"])
        <= tolerance * float(power["relative_residual_scale"])
        and float(power["exponent_first_max_abs_residual"])
        <= tolerance * float(power["relative_residual_scale"])
        and float(layernorm["maximum_abs_residual"])
        <= tolerance * max(float(np.max(np.abs(
            np.asarray(layernorm["direct_increment"])
        ))), 1.0)
        and chain_residual
        <= tolerance * max(float(np.max(np.abs(
            np.asarray(chain["recursive"])
        ))), 1.0)
        and bilinear_residual
        <= tolerance * max(float(np.max(np.abs(bilinear_direct))), 1.0)
        and diameter["identity_residual"]
        <= tolerance * max(diameter["source_diameter_squared"], 1.0)
        and state["gate_balance_max_abs_residual"]
        <= tolerance * max(state["source_diameter_squared"], 1.0)
        and state["state_bound_covers_endpoint"]
        and directed["state_bound_covers_endpoint"]
        and directed["strict_contraction_certified"]
        and convolution_residual <= tolerance
        and malformed_rejection_count == len(malformed)
    )
    return {
        "schema_version": "pldr-finite-increment-e0-v1",
        "passed": passed,
        "tolerance_factor_float64_epsilon": 256.0,
        "plga": {
            name: value for name, value in power.items()
            if isinstance(value, (float, int))
        },
        "layernorm_max_abs_residual": float(
            layernorm["maximum_abs_residual"]
        ),
        "bilinear_max_abs_residual": bilinear_residual,
        "eight_unit_unroll_max_abs_residual": chain_residual,
        "diameter_slack_identity_residual": float(
            diameter["identity_residual"]
        ),
        "gate_balance_max_abs_residual": float(
            state["gate_balance_max_abs_residual"]
        ),
        "directed_state_bound_covers_endpoint": bool(
            directed["state_bound_covers_endpoint"]
        ),
        "directed_strict_state_dominance": bool(
            directed["strict_contraction_certified"]
        ),
        "product_convolution_residual": convolution_residual,
        "malformed_rejection_count": malformed_rejection_count,
    }


def write_json_atomic(path: Any, payload: dict[str, Any]) -> None:
    """Write a JSON report atomically, including failure reports."""

    import json
    import os
    from pathlib import Path
    import tempfile

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)
