"""Source-state certificates for finite-registry PLDR row-map dynamics.

The deciding path in this module consumes only source rows, the executed
optimizer tangent, and certified row remainder radii. Successor rows are an
optional coverage input and never alter a prediction.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Mapping

import numpy as np


SCHEMA_VERSION = "pldr-causal-row-map-kernel-v1"


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


def forward_error_gamma(operation_count: int, dtype: Any = np.float64) -> float:
    """Return the standard ``k*u/(1-k*u)`` forward-error coefficient."""

    count = int(operation_count)
    if count < 0:
        raise ValueError("operation_count must be nonnegative")
    unit = float(np.finfo(np.dtype(dtype)).eps) / 2.0
    product = count * unit
    if product >= 1.0:
        raise ValueError("operation count is too large for a gamma bound")
    return product / (1.0 - product)


def unordered_pair_chunks(
    vertex_count: int, chunk_size: int,
) -> Iterator[np.ndarray]:
    """Yield every unordered pair once in deterministic lexicographic order."""

    count = int(vertex_count)
    size = int(chunk_size)
    if count < 2:
        raise ValueError("the registry needs at least two rows")
    if size < 1:
        raise ValueError("chunk_size must be positive")
    left_buffer: list[int] = []
    right_buffer: list[int] = []
    for left in range(count - 1):
        for right in range(left + 1, count):
            left_buffer.append(left)
            right_buffer.append(right)
            if len(left_buffer) == size:
                yield np.column_stack((left_buffer, right_buffer)).astype(
                    np.int64, copy=False)
                left_buffer.clear()
                right_buffer.clear()
    if left_buffer:
        yield np.column_stack((left_buffer, right_buffer)).astype(
            np.int64, copy=False)


def pair_energy_enclosure(
    source_difference: Any,
    directional_difference: Any,
    remainder_radius: Any,
    *,
    arithmetic_charge: Any = 0.0,
) -> dict[str, np.ndarray]:
    """Compute the two-sided finite-step pair-energy enclosure.

    For ``successor = source + directional + remainder`` with
    ``||remainder|| <= radius``, the returned lower and upper increments are
    rigorous in real arithmetic. ``arithmetic_charge`` enlarges both sides
    for analyzer roundoff or cross-provenance allowances.
    """

    source = _array(source_difference, "source_difference", 2)
    directional = _array(
        directional_difference, "directional_difference", 2)
    radius = _nonnegative(remainder_radius, "remainder_radius")
    charge = _nonnegative(arithmetic_charge, "arithmetic_charge")
    if directional.shape != source.shape:
        raise ValueError("source and directional differences disagree")
    pair_count = source.shape[0]
    try:
        radius = np.broadcast_to(radius, (pair_count,)).astype(
            np.float64, copy=False)
        charge = np.broadcast_to(charge, (pair_count,)).astype(
            np.float64, copy=False)
    except ValueError as error:
        raise ValueError("pair radii or charges do not broadcast") from error

    source_energy = np.einsum("pd,pd->p", source, source, optimize=True)
    directional_energy = np.einsum(
        "pd,pd->p", directional, directional, optimize=True)
    linear_work = 2.0 * np.einsum(
        "pd,pd->p", source, directional, optimize=True)
    source_norm = np.sqrt(np.maximum(source_energy, 0.0))
    directional_norm = np.sqrt(np.maximum(directional_energy, 0.0))
    remainder_cross = 2.0 * (
        source_norm + directional_norm) * radius
    common = linear_work + directional_energy
    lower_increment = common - remainder_cross - charge
    upper_increment = (
        common + remainder_cross + radius * radius + charge)
    values = (
        source_energy,
        directional_energy,
        linear_work,
        lower_increment,
        upper_increment,
    )
    if not all(np.isfinite(value).all() for value in values):
        raise FloatingPointError("pair-energy enclosure overflowed")
    return {
        "source_energy": source_energy,
        "linear_work": linear_work,
        "directional_energy": directional_energy,
        "remainder_radius": radius,
        "arithmetic_charge": charge,
        "lower_increment": lower_increment,
        "upper_increment": upper_increment,
        "successor_lower": source_energy + lower_increment,
        "successor_upper": source_energy + upper_increment,
    }


def _lexicographic_max(
    values: np.ndarray,
    pairs: np.ndarray,
    current_value: float,
    current_pair: tuple[int, int] | None,
) -> tuple[float, tuple[int, int] | None]:
    index = int(np.argmax(values))
    candidate = float(values[index])
    pair = (int(pairs[index, 0]), int(pairs[index, 1]))
    if (
        candidate > current_value
        or candidate == current_value
        and (current_pair is None or pair < current_pair)
    ):
        return candidate, pair
    return current_value, current_pair


def stream_registry_prediction(
    source_rows: Any,
    row_jvp: Any,
    row_remainder_radius: Any,
    *,
    pair_chunk_size: int = 32768,
    arithmetic_safety_factor: float = 8.0,
    extra_energy_charge: float = 0.0,
    successor_rows: Any | None = None,
) -> dict[str, Any]:
    """Stream the causal all-pairs diameter prediction.

    A pair remainder is bounded by the sum of its two row radii. The
    successor, when supplied, is checked only after all source-state maxima
    and the decision have been formed.
    """

    rows = _array(source_rows, "source_rows", 2)
    jvp = _array(row_jvp, "row_jvp", 2)
    radius = _nonnegative(row_remainder_radius, "row_remainder_radius")
    if rows.shape != jvp.shape or radius.shape != (rows.shape[0],):
        raise ValueError("row registry, JVP, and radius dimensions disagree")
    if rows.shape[0] < 2 or rows.shape[1] < 1:
        raise ValueError("the row registry has invalid dimensions")
    safety = float(arithmetic_safety_factor)
    extra = float(extra_energy_charge)
    if not math.isfinite(safety) or safety < 1.0:
        raise ValueError("arithmetic_safety_factor must be at least one")
    if not math.isfinite(extra) or extra < 0.0:
        raise ValueError("extra_energy_charge must be nonnegative")
    successor = None
    if successor_rows is not None:
        successor = _array(successor_rows, "successor_rows", 2)
        if successor.shape != rows.shape:
            raise ValueError("source and successor row registries disagree")

    dimension = rows.shape[1]
    gamma = forward_error_gamma(12 * dimension + 32)
    realized_gamma = forward_error_gamma(3 * dimension + 4)
    source_nominal_max = -math.inf
    source_nominal_pair = None
    source_lower_max = -math.inf
    source_lower_pair = None
    source_upper_max = -math.inf
    source_upper_pair = None
    upper_max = -math.inf
    upper_pair = None
    lower_max = -math.inf
    lower_pair = None
    realized_max = -math.inf
    realized_pair = None
    realized_lower_max = -math.inf
    realized_lower_pair = None
    realized_upper_max = -math.inf
    realized_upper_pair = None
    coverage_pass = True
    maximum_upper_excess = -math.inf
    maximum_lower_excess = -math.inf
    pair_count = 0

    for pairs in unordered_pair_chunks(rows.shape[0], pair_chunk_size):
        left = pairs[:, 0]
        right = pairs[:, 1]
        source_difference = rows[left] - rows[right]
        directional_difference = jvp[left] - jvp[right]
        pair_radius = radius[left] + radius[right]

        provisional = pair_energy_enclosure(
            source_difference,
            directional_difference,
            pair_radius,
        )
        magnitude = (
            provisional["source_energy"]
            + np.abs(provisional["linear_work"])
            + provisional["directional_energy"]
            + 2.0 * (
                np.linalg.norm(source_difference, axis=1)
                + np.linalg.norm(directional_difference, axis=1)
            ) * pair_radius
            + pair_radius * pair_radius
        )
        arithmetic_charge = safety * gamma * magnitude + extra
        enclosure = pair_energy_enclosure(
            source_difference,
            directional_difference,
            pair_radius,
            arithmetic_charge=arithmetic_charge,
        )
        source_energy = enclosure["source_energy"]
        source_lower = source_energy - arithmetic_charge
        source_upper = source_energy + arithmetic_charge
        source_nominal_max, source_nominal_pair = _lexicographic_max(
            source_energy, pairs, source_nominal_max, source_nominal_pair)
        source_lower_max, source_lower_pair = _lexicographic_max(
            source_lower, pairs, source_lower_max, source_lower_pair)
        source_upper_max, source_upper_pair = _lexicographic_max(
            source_upper, pairs, source_upper_max, source_upper_pair)
        upper_max, upper_pair = _lexicographic_max(
            enclosure["successor_upper"], pairs, upper_max, upper_pair)
        lower_max, lower_pair = _lexicographic_max(
            enclosure["successor_lower"], pairs, lower_max, lower_pair)

        if successor is not None:
            realized_difference = successor[left] - successor[right]
            realized = np.einsum(
                "pd,pd->p", realized_difference, realized_difference,
                optimize=True,
            )
            realized_charge = (
                safety * realized_gamma * np.abs(realized) + extra)
            realized_lower = realized - realized_charge
            realized_upper = realized + realized_charge
            realized_max, realized_pair = _lexicographic_max(
                realized, pairs, realized_max, realized_pair)
            realized_lower_max, realized_lower_pair = _lexicographic_max(
                realized_lower, pairs, realized_lower_max,
                realized_lower_pair)
            realized_upper_max, realized_upper_pair = _lexicographic_max(
                realized_upper, pairs, realized_upper_max,
                realized_upper_pair)
            upper_excess = (
                realized_upper - enclosure["successor_upper"])
            lower_excess = (
                enclosure["successor_lower"] - realized_lower)
            maximum_upper_excess = max(
                maximum_upper_excess,
                float(np.max(upper_excess, initial=-math.inf)),
            )
            maximum_lower_excess = max(
                maximum_lower_excess,
                float(np.max(lower_excess, initial=-math.inf)),
            )
            coverage_pass = bool(
                coverage_pass
                and np.all(upper_excess <= 0.0)
                and np.all(lower_excess <= 0.0)
            )
        pair_count += len(pairs)

    source_certified_lower = max(0.0, source_lower_max)
    contraction = upper_max < source_certified_lower
    expansion = lower_max > source_upper_max
    if contraction and expansion:
        raise ArithmeticError("inconsistent upper and lower certificates")
    if contraction:
        decision = "CONTRACTION_CERTIFIED"
    elif expansion:
        decision = "REEXPANSION_CERTIFIED"
    else:
        decision = "UNRESOLVED"
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "vertex_count": int(rows.shape[0]),
        "row_width": int(dimension),
        "pair_count": int(pair_count),
        "pair_chunk_size": int(pair_chunk_size),
        "source_diameter_squared": source_nominal_max,
        "source_diameter_pair": list(source_nominal_pair),
        "source_diameter_lower_squared": source_certified_lower,
        "source_diameter_lower_pair": list(source_lower_pair),
        "source_diameter_upper_squared": source_upper_max,
        "source_diameter_upper_pair": list(source_upper_pair),
        "predicted_successor_upper_squared": upper_max,
        "predicted_upper_pair": list(upper_pair),
        "predicted_successor_lower_max_squared": lower_max,
        "predicted_lower_pair": list(lower_pair),
        "contraction_margin_squared": source_certified_lower - upper_max,
        "reexpansion_margin_squared": lower_max - source_upper_max,
        "diameter_ratio_upper": (
            math.sqrt(max(upper_max, 0.0) / source_certified_lower)
            if source_certified_lower > 0.0 else None
        ),
        "decision": decision,
        "decision_uses_successor": False,
        "arithmetic": {
            "dtype": "float64",
            "operation_count": 12 * dimension + 32,
            "forward_error_gamma": gamma,
            "safety_factor": safety,
            "extra_energy_charge": extra,
            "source_interval_used_for_decision": True,
            "contraction_boundary": "source_diameter_lower_squared",
            "reexpansion_boundary": "source_diameter_upper_squared",
        },
    }
    if successor is not None:
        result["coverage"] = {
            "successor_opened_after_prediction": True,
            "realized_successor_diameter_squared": realized_max,
            "realized_successor_pair": list(realized_pair),
            "realized_successor_diameter_lower_squared": realized_lower_max,
            "realized_successor_lower_pair": list(realized_lower_pair),
            "realized_successor_diameter_upper_squared": realized_upper_max,
            "realized_successor_upper_pair": list(realized_upper_pair),
            "realized_energy_interval_charged": True,
            "realized_energy_operation_count": 3 * dimension + 4,
            "realized_energy_forward_error_gamma": realized_gamma,
            "all_pair_enclosures_hold": coverage_pass,
            "maximum_upper_excess": maximum_upper_excess,
            "maximum_lower_excess": maximum_lower_excess,
        }
    else:
        result["coverage"] = None
    return result


def decompose_linear_routes(
    source_difference: Any,
    route_jvps: Mapping[str, Any],
    total_jvp: Any,
    *,
    tolerance: float = 256.0 * np.finfo(np.float64).eps,
) -> dict[str, Any]:
    """Attribute linear work while retaining the total quadratic mixed work."""

    source = _array(source_difference, "source_difference", 2)
    total = _array(total_jvp, "total_jvp", 2)
    if source.shape != total.shape or not route_jvps:
        raise ValueError("route attribution requires matching nonempty arrays")
    parsed = {
        name: _array(value, f"route_jvps[{name!r}]", 2)
        for name, value in route_jvps.items()
    }
    if any(value.shape != total.shape for value in parsed.values()):
        raise ValueError("route JVP shapes disagree")
    reconstructed = np.sum(np.stack(tuple(parsed.values())), axis=0)
    scale = np.maximum(
        np.linalg.norm(total, axis=1), np.finfo(np.float64).tiny)
    residual = np.linalg.norm(reconstructed - total, axis=1) / scale
    if float(np.max(residual, initial=0.0)) > float(tolerance):
        raise ArithmeticError("route JVPs do not reconstruct the total JVP")
    linear = {
        name: 2.0 * np.einsum(
            "pd,pd->p", source, value, optimize=True)
        for name, value in parsed.items()
    }
    quadratic = np.einsum("pd,pd->p", total, total, optimize=True)
    return {
        "linear_work": linear,
        "linear_work_sum": np.sum(np.stack(tuple(linear.values())), axis=0),
        "total_directional_energy_with_cross_terms": quadratic,
        "maximum_jvp_reconstruction_relative_residual": float(
            np.max(residual, initial=0.0)),
    }


def iterate_affine_diameter_bound(
    initial_squared_diameter: float,
    contraction_factors: Any,
    forcing: Any,
) -> np.ndarray:
    """Iterate ``D_next^2 <= q^2 D^2 + f`` in chronological order."""

    initial = float(initial_squared_diameter)
    factors = _nonnegative(contraction_factors, "contraction_factors")
    source = _nonnegative(forcing, "forcing")
    if not math.isfinite(initial) or initial < 0.0:
        raise ValueError("initial squared diameter must be nonnegative")
    if factors.ndim != 1 or source.shape != factors.shape:
        raise ValueError("block factors and forcing must be matching vectors")
    envelope = np.empty(len(factors) + 1, dtype=np.float64)
    envelope[0] = initial
    for step, (factor, local_force) in enumerate(
        zip(factors, source, strict=True)
    ):
        envelope[step + 1] = factor * factor * envelope[step] + local_force
    if not np.isfinite(envelope).all():
        raise FloatingPointError("the affine diameter envelope overflowed")
    return envelope


def geometric_convolution(beta: float, sigma: float, horizon: int) -> float:
    """Return ``sum_{r=0}^h beta^(h-r) sigma^r`` stably."""

    beta = float(beta)
    sigma = float(sigma)
    count = int(horizon)
    if not 0.0 <= beta < 1.0 or not 0.0 <= sigma < 1.0 or count < 0:
        raise ValueError("geometric rates or horizon are invalid")
    if beta == sigma:
        return (count + 1) * beta ** count
    return (beta ** (count + 1) - sigma ** (count + 1)) / (beta - sigma)


def adamw_gate_route_envelope(
    *,
    initial_gate: float,
    initial_first_moment: float,
    gradient_envelope: float,
    gradient_rate: float,
    beta1: float,
    learning_rate_upper: float,
    adam_epsilon: float,
    bias_correction_floor: float,
    decay_multiplier_upper: float,
    horizon: int,
) -> dict[str, np.ndarray]:
    """Explicit gradient-to-moment-to-innovation-to-gate upper envelope."""

    values = (
        initial_gate,
        initial_first_moment,
        gradient_envelope,
        learning_rate_upper,
        adam_epsilon,
        bias_correction_floor,
        decay_multiplier_upper,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("gate route inputs must be finite")
    if (
        gradient_envelope < 0.0
        or learning_rate_upper < 0.0
        or adam_epsilon <= 0.0
        or not 0.0 < bias_correction_floor <= 1.0
        or not 0.0 <= beta1 < 1.0
        or not 0.0 <= gradient_rate < 1.0
        or not 0.0 <= decay_multiplier_upper < 1.0
        or int(horizon) < 0
    ):
        raise ValueError("gate route constants are outside their domains")
    steps = int(horizon)
    moment = np.empty(steps, dtype=np.float64)
    innovation = np.empty(steps, dtype=np.float64)
    gate = np.empty(steps + 1, dtype=np.float64)
    gate[0] = abs(float(initial_gate))
    for local in range(steps):
        moment[local] = (
            beta1 ** (local + 1) * abs(float(initial_first_moment))
            + (1.0 - beta1) * gradient_envelope
            * geometric_convolution(beta1, gradient_rate, local)
        )
        innovation[local] = (
            learning_rate_upper * moment[local]
            / (adam_epsilon * bias_correction_floor)
        )
        gate[local + 1] = (
            decay_multiplier_upper * gate[local] + innovation[local])
    return {
        "moment_upper": moment,
        "innovation_upper": innovation,
        "gate_upper": gate,
    }


__all__ = [
    "SCHEMA_VERSION",
    "adamw_gate_route_envelope",
    "decompose_linear_routes",
    "forward_error_gamma",
    "geometric_convolution",
    "iterate_affine_diameter_bound",
    "pair_energy_enclosure",
    "stream_registry_prediction",
    "unordered_pair_chunks",
]
