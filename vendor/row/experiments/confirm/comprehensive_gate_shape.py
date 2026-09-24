"""Exact kernels for comprehensive PLDR gate-shape collapse.

The primary path is endpoint exact. Directional derivatives and native
arithmetic comparisons are diagnostics and never enter the physical sign
decision.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "pldr-comprehensive-gate-shape-kernel-v1"


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.ascontiguousarray(array)


def coordinate_oscillations(gate: Any, normalized_rows: Any) -> dict[str, Any]:
    """Compute the exact coordinate carriers and diameter sandwich."""

    gamma = _array(gate, "gate", 1)
    rows = _array(normalized_rows, "normalized_rows", 2)
    if rows.shape[1] != gamma.size or rows.shape[0] < 2:
        raise ValueError("gate and finite row registry dimensions disagree")
    shape_min = np.min(rows, axis=0)
    shape_max = np.max(rows, axis=0)
    shape_oscillation = shape_max - shape_min
    carrier = np.abs(gamma) * shape_oscillation
    return {
        "shape_min": shape_min,
        "shape_max": shape_max,
        "shape_oscillation": shape_oscillation,
        "gate_magnitude": np.abs(gamma),
        "carrier": carrier,
        "carrier_linf": float(np.max(carrier, initial=0.0)),
        "carrier_l2": float(np.linalg.norm(carrier)),
    }


def _pair_chunks(vertex_count: int, chunk_size: int):
    if vertex_count < 2 or chunk_size < 1:
        raise ValueError("invalid finite registry or pair chunk size")
    for left in range(vertex_count - 1):
        for start in range(left + 1, vertex_count, chunk_size):
            right = np.arange(
                start, min(start + chunk_size, vertex_count), dtype=np.int64)
            yield left, right


def finite_registry_geometry(
    gate: Any,
    normalized_rows: Any,
    *,
    pair_chunk_size: int = 32768,
) -> dict[str, Any]:
    """Stream the exact all-pairs diameter without a dense pair ledger."""

    gamma = _array(gate, "gate", 1)
    rows = _array(normalized_rows, "normalized_rows", 2)
    coordinate = coordinate_oscillations(gamma, rows)
    maximum = -math.inf
    maximizer = (-1, -1)
    pair_count = 0
    for left, right in _pair_chunks(rows.shape[0], int(pair_chunk_size)):
        contrast = rows[left] - rows[right]
        energy = np.einsum(
            "pd,d,pd->p", contrast, gamma * gamma, contrast, optimize=True)
        local = int(np.argmax(energy))
        if float(energy[local]) > maximum:
            maximum = float(energy[local])
            maximizer = (left, int(right[local]))
        pair_count += len(right)
    diameter = math.sqrt(max(maximum, 0.0))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(
        diameter, coordinate["carrier_l2"], 1.0)
    return {
        **coordinate,
        "pair_count": pair_count,
        "diameter_squared": maximum,
        "diameter": diameter,
        "diameter_maximizer": maximizer,
        "sandwich_lower_slack": diameter - coordinate["carrier_linf"],
        "sandwich_upper_slack": coordinate["carrier_l2"] - diameter,
        "sandwich_valid": (
            coordinate["carrier_linf"] <= diameter + tolerance
            and diameter <= coordinate["carrier_l2"] + tolerance
        ),
    }


def diameter_slack_summary(
    gate_before: Any,
    rows_before: Any,
    gate_after: Any,
    rows_after: Any,
    *,
    pair_chunk_size: int = 32768,
) -> dict[str, Any]:
    """Compute exact maximum-level slack and changing maximizers."""

    gamma0 = _array(gate_before, "gate_before", 1)
    gamma1 = _array(gate_after, "gate_after", 1)
    source = _array(rows_before, "rows_before", 2)
    target = _array(rows_after, "rows_after", 2)
    if source.shape != target.shape or gamma0.shape != gamma1.shape:
        raise ValueError("diameter-slack endpoint dimensions disagree")
    if source.shape[1] != gamma0.size:
        raise ValueError("row width and gate width disagree")

    source_max = -math.inf
    target_max = -math.inf
    source_pair = (-1, -1)
    target_pair = (-1, -1)
    target_source_energy = math.nan
    expanding_pairs = 0
    maximum_pair_expansion = 0.0
    pair_count = 0
    for left, right in _pair_chunks(source.shape[0], int(pair_chunk_size)):
        contrast0 = source[left] - source[right]
        contrast1 = target[left] - target[right]
        energy0 = np.einsum(
            "pd,d,pd->p", contrast0, gamma0 * gamma0, contrast0,
            optimize=True)
        energy1 = np.einsum(
            "pd,d,pd->p", contrast1, gamma1 * gamma1, contrast1,
            optimize=True)
        local0 = int(np.argmax(energy0))
        if float(energy0[local0]) > source_max:
            source_max = float(energy0[local0])
            source_pair = (left, int(right[local0]))
        local1 = int(np.argmax(energy1))
        if float(energy1[local1]) > target_max:
            target_max = float(energy1[local1])
            target_pair = (left, int(right[local1]))
            target_source_energy = float(energy0[local1])
        expansion = energy1 - energy0
        expanding_pairs += int(np.count_nonzero(expansion > 0.0))
        maximum_pair_expansion = max(
            maximum_pair_expansion, float(np.max(expansion, initial=0.0)))
        pair_count += len(right)

    minimum_slack = source_max - target_max
    source_gap_at_target_max = source_max - target_source_energy
    decrement_at_target_max = target_source_energy - target_max
    residual = (
        minimum_slack
        - source_gap_at_target_max
        - decrement_at_target_max
    )
    scale = max(
        abs(minimum_slack),
        abs(source_gap_at_target_max),
        abs(decrement_at_target_max),
        1.0,
    )
    return {
        "pair_count": pair_count,
        "source_diameter_squared": source_max,
        "target_diameter_squared": target_max,
        "source_diameter": math.sqrt(max(source_max, 0.0)),
        "target_diameter": math.sqrt(max(target_max, 0.0)),
        "source_maximizer": source_pair,
        "target_maximizer": target_pair,
        "maximizer_changed": source_pair != target_pair,
        "minimum_exact_slack": minimum_slack,
        "source_gap_at_target_maximizer": source_gap_at_target_max,
        "decrement_at_target_maximizer": decrement_at_target_max,
        "slack_identity_relative_residual": abs(residual) / scale,
        "expanding_pair_count": expanding_pairs,
        "maximum_pair_expansion": maximum_pair_expansion,
    }


def adamw_gate_duhamel(
    gate: Any, multipliers: Any, innovations: Any,
) -> dict[str, Any]:
    """Replay gamma_(n+1) = a_n gamma_n - innovation_n exactly."""

    initial = _array(gate, "gate", 1)
    factor = _array(multipliers, "multipliers", 2)
    force = _array(innovations, "innovations", 2)
    if factor.shape != force.shape or factor.shape[1] != initial.size:
        raise ValueError("gate Duhamel dimensions disagree")
    trajectory = np.empty((factor.shape[0] + 1, initial.size))
    trajectory[0] = initial
    for step in range(factor.shape[0]):
        trajectory[step + 1] = (
            factor[step] * trajectory[step] - force[step])

    transport = np.ones(initial.size)
    convolution = np.zeros(initial.size)
    convolution_bound = np.zeros(initial.size)
    for step in range(factor.shape[0] - 1, -1, -1):
        contribution = transport * force[step]
        convolution += contribution
        convolution_bound += np.abs(contribution)
        transport *= factor[step]
    endpoint = transport * initial - convolution
    scale = np.maximum.reduce((
        np.abs(endpoint),
        np.abs(trajectory[-1]),
        np.ones_like(endpoint),
    ))
    return {
        "trajectory": trajectory,
        "homogeneous_product": transport,
        "signed_forcing_convolution": convolution,
        "forcing_upper": convolution_bound,
        "endpoint": endpoint,
        "identity_relative_residual": float(np.max(
            np.abs(endpoint - trajectory[-1]) / scale, initial=0.0)),
    }


def coordinate_block_recurrence(
    gate_before: Any,
    rows_before: Any,
    gate_after: Any,
    rows_after: Any,
    *,
    multipliers: Any | None = None,
    innovations: Any | None = None,
    zero_tolerance: float = 0.0,
) -> dict[str, Any]:
    """Assemble q and f from gate Duhamel and exact shape factors."""

    gamma0 = _array(gate_before, "gate_before", 1)
    gamma1 = _array(gate_after, "gate_after", 1)
    start = coordinate_oscillations(gamma0, rows_before)
    end = coordinate_oscillations(gamma1, rows_after)
    g0 = start["gate_magnitude"]
    g1 = end["gate_magnitude"]
    w0 = start["shape_oscillation"]
    w1 = end["shape_oscillation"]
    z0 = start["carrier"]
    z1 = end["carrier"]

    if (multipliers is None) != (innovations is None):
        raise ValueError("gate multipliers and innovations are one input")
    if multipliers is None:
        gate_gain = np.zeros_like(g0)
        gate_force = np.zeros_like(g0)
        positive_gate = g0 > zero_tolerance
        gate_gain[positive_gate] = g1[positive_gate] / g0[positive_gate]
        gate_force[~positive_gate] = g1[~positive_gate]
        duhamel_residual = 0.0
    else:
        replay = adamw_gate_duhamel(gamma0, multipliers, innovations)
        endpoint_scale = np.maximum.reduce((
            np.abs(gamma1), np.abs(replay["endpoint"]), np.ones_like(gamma1)))
        duhamel_residual = float(np.max(
            np.abs(gamma1 - replay["endpoint"]) / endpoint_scale,
            initial=0.0,
        ))
        gate_gain = np.abs(replay["homogeneous_product"])
        gate_force = replay["forcing_upper"]

    shape_gain = np.zeros_like(w0)
    shape_force = np.zeros_like(w0)
    positive_shape = w0 > zero_tolerance
    shape_gain[positive_shape] = w1[positive_shape] / w0[positive_shape]
    shape_force[~positive_shape] = w1[~positive_shape]
    coefficient = gate_gain * shape_gain
    forcing = (
        gate_gain * g0 * shape_force
        + shape_gain * w0 * gate_force
        + gate_force * shape_force
    )
    bound = coefficient * z0 + forcing
    tolerance = 128.0 * np.finfo(np.float64).eps * np.maximum.reduce((
        np.abs(bound), np.abs(z1), np.ones_like(z1)))

    gate_log = np.full_like(z0, np.nan)
    shape_log = np.full_like(z0, np.nan)
    total_log = np.full_like(z0, np.nan)
    positive = (
        (g0 > zero_tolerance) & (g1 > zero_tolerance)
        & (w0 > zero_tolerance) & (w1 > zero_tolerance)
    )
    gate_log[positive] = np.log(g0[positive] / g1[positive])
    shape_log[positive] = np.log(w0[positive] / w1[positive])
    total_log[positive] = np.log(z0[positive] / z1[positive])
    log_residual = np.full_like(z0, np.nan)
    log_residual[positive] = (
        total_log[positive] - gate_log[positive] - shape_log[positive])
    return {
        "gate_start": g0,
        "gate_end": g1,
        "shape_start": w0,
        "shape_end": w1,
        "carrier_start": z0,
        "carrier_end": z1,
        "gate_gain": gate_gain,
        "gate_force": gate_force,
        "shape_gain": shape_gain,
        "shape_force": shape_force,
        "coefficient": coefficient,
        "forcing": forcing,
        "carrier_bound": bound,
        "carrier_covered": z1 <= bound + tolerance,
        "duhamel_endpoint_relative_residual": duhamel_residual,
        "positive_log_mask": positive,
        "gate_log_decrement": gate_log,
        "shape_log_decrement": shape_log,
        "total_log_decrement": total_log,
        "log_identity_residual": log_residual,
    }


def exact_pair_decomposition(
    gate_before: Any,
    contrast_before: Any,
    gate_after: Any,
    contrast_after: Any,
    *,
    decayed_gate: Any | None = None,
    innovation: Any | None = None,
) -> dict[str, Any]:
    """Exact gate then shape decomposition for supplied row pairs."""

    gamma0 = _array(gate_before, "gate_before", 1)
    gamma1 = _array(gate_after, "gate_after", 1)
    c0 = _array(contrast_before, "contrast_before", 2)
    c1 = _array(contrast_after, "contrast_after", 2)
    if c0.shape != c1.shape or c0.shape[1] != gamma0.size:
        raise ValueError("pair decomposition dimensions disagree")
    energy0 = np.einsum("pd,d,pd->p", c0, gamma0 * gamma0, c0)
    fixed = np.einsum("pd,d,pd->p", c0, gamma1 * gamma1, c0)
    energy1 = np.einsum("pd,d,pd->p", c1, gamma1 * gamma1, c1)
    gate_decrement = energy0 - fixed
    shape_work = energy1 - fixed
    pair_decrement = energy0 - energy1
    result = {
        "energy_before": energy0,
        "energy_after_fixed_shape": fixed,
        "energy_after": energy1,
        "gate_decrement": gate_decrement,
        "shape_work": shape_work,
        "pair_decrement": pair_decrement,
        "decomposition_residual": pair_decrement - gate_decrement + shape_work,
    }
    if (decayed_gate is None) != (innovation is None):
        raise ValueError("decayed gate and innovation are one input")
    if decayed_gate is not None:
        decayed = _array(decayed_gate, "decayed_gate", 1)
        update = _array(innovation, "innovation", 1)
        metric = c0 * c0
        decay_energy = np.einsum("pd,d->p", metric, decayed * decayed)
        decay_dissipation = energy0 - decay_energy
        alignment = 2.0 * np.einsum(
            "pd,d,d->p", metric, decayed, update)
        finite_charge = np.einsum("pd,d->p", metric, update * update)
        result.update({
            "decay_dissipation": decay_dissipation,
            "chronological_alignment": alignment,
            "finite_gate_charge": finite_charge,
            "gate_balance_residual": (
                gate_decrement
                - decay_dissipation - alignment + finite_charge
            ),
        })
    return result


def observed_jvp_remainder(
    rows_before: Any, rows_after: Any, directional_increment: Any,
) -> dict[str, Any]:
    """Return an a-posteriori JVP residual, not a source-state bound."""

    source = _array(rows_before, "rows_before", 2)
    target = _array(rows_after, "rows_after", 2)
    tangent = _array(directional_increment, "directional_increment", 2)
    if source.shape != target.shape or source.shape != tangent.shape:
        raise ValueError("observed JVP arrays disagree")
    residual = target - source - tangent
    return {
        "residual": residual,
        "row_norm": np.linalg.norm(residual, axis=1),
        "maximum_row_norm": float(np.max(
            np.linalg.norm(residual, axis=1), initial=0.0)),
    }


def two_arm_shape_response(
    source_contrast: Any,
    baseline_gate: Any,
    baseline_contrast: Any,
    removal_gate: Any,
    removal_contrast: Any,
) -> dict[str, Any]:
    """Separate fixed-source-shape gate response and arm shape correction."""

    source = _array(source_contrast, "source_contrast", 2)
    baseline_shape = _array(baseline_contrast, "baseline_contrast", 2)
    removal_shape = _array(removal_contrast, "removal_contrast", 2)
    gamma_b = _array(baseline_gate, "baseline_gate", 1)
    gamma_r = _array(removal_gate, "removal_gate", 1)
    if source.shape != baseline_shape.shape or source.shape != removal_shape.shape:
        raise ValueError("two-arm contrast shapes disagree")
    baseline_fixed = np.einsum(
        "pd,d,pd->p", source, gamma_b * gamma_b, source)
    removal_fixed = np.einsum(
        "pd,d,pd->p", source, gamma_r * gamma_r, source)
    baseline_live = np.einsum(
        "pd,d,pd->p", baseline_shape, gamma_b * gamma_b, baseline_shape)
    removal_live = np.einsum(
        "pd,d,pd->p", removal_shape, gamma_r * gamma_r, removal_shape)
    fixed_response = removal_fixed - baseline_fixed
    baseline_work = baseline_live - baseline_fixed
    removal_work = removal_live - removal_fixed
    correction = removal_work - baseline_work
    live_response = removal_live - baseline_live
    return {
        "fixed_shape_response": fixed_response,
        "baseline_shape_work": baseline_work,
        "removal_shape_work": removal_work,
        "shape_correction": correction,
        "live_response": live_response,
        "identity_residual": live_response - fixed_response - correction,
    }


def iswiglu(value: Any) -> np.ndarray:
    x = _array(value, "value")
    sigmoid = np.where(
        x >= 0.0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return x * x * sigmoid


def iswiglu_secant(left: Any, right: Any) -> np.ndarray:
    """Exact occupied-segment secant of the implemented iSwiGLU."""

    x = _array(left, "left")
    y = _array(right, "right")
    if x.shape != y.shape:
        raise ValueError("iSwiGLU secant endpoints disagree")
    result = np.empty_like(x)
    equal = x == y
    sigmoid = np.where(
        x >= 0.0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    result[equal] = (
        2.0 * x[equal] * sigmoid[equal]
        + x[equal] * x[equal] * sigmoid[equal] * (1.0 - sigmoid[equal])
    )
    unequal = ~equal
    result[unequal] = (
        iswiglu(y[unequal]) - iswiglu(x[unequal])
    ) / (y[unequal] - x[unequal])
    return result


def occupied_power_secant(
    left: Any, right: Any, exponent: Any,
) -> np.ndarray:
    x = _array(left, "left")
    y = _array(right, "right")
    power = _array(exponent, "exponent")
    if x.shape != y.shape or x.shape != power.shape:
        raise ValueError("power secant endpoints disagree")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("power bases must be strictly positive")
    result = np.empty_like(x)
    equal = x == y
    result[equal] = power[equal] * x[equal] ** (power[equal] - 1.0)
    unequal = ~equal
    result[unequal] = (
        y[unequal] ** power[unequal] - x[unequal] ** power[unequal]
    ) / (y[unequal] - x[unequal])
    return result


def permuted_chunk_order(
    chunk_count: int,
    seed: int,
    *,
    reserved_chunks: Any = (),
) -> dict[str, Any]:
    """Generate a seed-keyed permutation with reserved probes excluded."""

    count = int(chunk_count)
    if count < 1:
        raise ValueError("chunk_count must be positive")
    reserved = np.asarray(tuple(reserved_chunks), dtype=np.int64)
    if reserved.size and (
        np.any(reserved < 0)
        or np.any(reserved >= count)
        or np.unique(reserved).size != reserved.size
    ):
        raise ValueError("reserved chunks are invalid")
    allowed = np.setdiff1d(
        np.arange(count, dtype=np.int64), reserved, assume_unique=False)
    order = np.random.Generator(np.random.PCG64(int(seed))).permutation(allowed)
    digest = hashlib.sha256(order.astype("<i8", copy=False).tobytes()).hexdigest()
    return {
        "order": order,
        "sha256": digest,
        "chunk_count": int(order.size),
        "seed": int(seed),
    }


__all__ = [
    "SCHEMA_VERSION",
    "adamw_gate_duhamel",
    "coordinate_block_recurrence",
    "coordinate_oscillations",
    "diameter_slack_summary",
    "exact_pair_decomposition",
    "finite_registry_geometry",
    "iswiglu",
    "iswiglu_secant",
    "observed_jvp_remainder",
    "occupied_power_secant",
    "permuted_chunk_order",
    "two_arm_shape_response",
]
