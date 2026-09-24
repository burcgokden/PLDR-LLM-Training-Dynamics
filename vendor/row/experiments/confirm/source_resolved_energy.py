"""Source-resolved physical-energy kernels for PLDR row-map collapse.

The functions in this module implement finite identities and fail-closed
decision rules.  They do not infer asymptotic collapse from a finite record.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from itertools import combinations
import math
from typing import Any

import numpy as np


def _array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite array")
    return result


def _rows(value: Any, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.ndim < 2 or result.shape[-2] < 2:
        raise ValueError(f"{name} must contain at least two rows")
    return result


def _feature_vector(value: Any, features: int, name: str) -> np.ndarray:
    result = _array(value, name)
    if result.shape != (features,):
        raise ValueError(f"{name} must have shape ({features},)")
    return result


def center_rows(rows: Any) -> np.ndarray:
    """Project a row map onto the row-constant quotient."""

    value = _rows(rows, "rows")
    return value - np.mean(value, axis=-2, keepdims=True)


def gate_shape_energy(shape: Any, gate: Any) -> dict[str, Any]:
    """Evaluate the exact final-LayerNorm gate-shape factorization."""

    value = _rows(shape, "shape")
    gain = _feature_vector(gate, value.shape[-1], "gate")
    centered_shape = center_rows(value)
    coordinate_shape_energy = np.sum(
        centered_shape * centered_shape, axis=-2
    )
    coordinate_energy = coordinate_shape_energy * gain * gain
    centered_rows = centered_shape * gain
    energy = np.sum(coordinate_energy, axis=-1)
    direct = np.sum(centered_rows * centered_rows, axis=(-2, -1))
    return {
        "centered_shape": centered_shape,
        "centered_rows": centered_rows,
        "coordinate_shape_energy": coordinate_shape_energy,
        "coordinate_energy": coordinate_energy,
        "energy": energy,
        "maximum_abs_factorization_residual": float(
            np.max(np.abs(energy - direct))
        ),
    }


def effective_gate_shape(
    coordinate_shape_energy: Any,
    gate: Any,
) -> dict[str, Any]:
    """Factor total energy into shape mass and a shape-weighted squared gate."""

    shape = _array(coordinate_shape_energy, "coordinate_shape_energy")
    if shape.ndim < 1 or np.any(shape < 0.0):
        raise ValueError("coordinate shape energy must be nonnegative")
    gain = np.asarray(gate, dtype=np.float64)
    try:
        gain = np.broadcast_to(gain, shape.shape)
    except ValueError as error:
        raise ValueError("gate does not broadcast over shape coordinates") from error
    if not np.isfinite(gain).all():
        raise ValueError("gate must be finite")
    shape_energy = np.sum(shape, axis=-1)
    coordinate_energy = shape * gain * gain
    physical_energy = np.sum(coordinate_energy, axis=-1)
    occupied = shape_energy > 0.0
    occupancy = np.divide(
        shape,
        shape_energy[..., None],
        out=np.zeros_like(shape),
        where=occupied[..., None],
    )
    effective_gate_squared = np.divide(
        physical_energy,
        shape_energy,
        out=np.zeros_like(physical_energy),
        where=occupied,
    )
    gate_squared = gain * gain
    lower = np.min(gate_squared, axis=-1) * shape_energy
    upper = np.max(gate_squared, axis=-1) * shape_energy
    return {
        "coordinate_shape_energy": shape,
        "coordinate_energy": coordinate_energy,
        "shape_energy": shape_energy,
        "physical_energy": physical_energy,
        "shape_occupied": occupied,
        "shape_occupancy": occupancy,
        "effective_gate_squared": effective_gate_squared,
        "gate_lower_energy": lower,
        "gate_upper_energy": upper,
    }


def endpoint_gate_shape_decomposition(
    coordinate_shape_energy0: Any,
    coordinate_shape_energy1: Any,
    gate0: Any,
    gate1: Any,
) -> dict[str, Any]:
    """Decompose one positive endpoint gain into shape, gate, and alignment."""

    source = effective_gate_shape(coordinate_shape_energy0, gate0)
    endpoint = effective_gate_shape(coordinate_shape_energy1, gate1)
    if source["coordinate_shape_energy"].shape != endpoint[
        "coordinate_shape_energy"
    ].shape:
        raise ValueError("gate-shape endpoints must have the same shape")
    shape = source["coordinate_shape_energy"]
    gain1 = np.asarray(gate1, dtype=np.float64)
    try:
        gain1 = np.broadcast_to(gain1, shape.shape)
    except ValueError as error:
        raise ValueError("endpoint gate does not broadcast") from error
    gate_only_endpoint = np.sum(
        source["shape_occupancy"] * gain1 * gain1, axis=-1
    )
    required_positive = (
        source["shape_energy"] > 0.0
    ) & (
        endpoint["shape_energy"] > 0.0
    ) & (
        source["effective_gate_squared"] > 0.0
    ) & (
        endpoint["effective_gate_squared"] > 0.0
    ) & (
        gate_only_endpoint > 0.0
    )
    if not np.all(required_positive):
        raise ValueError(
            "endpoint decomposition requires positive shape and gate factors"
        )
    shape_gain = endpoint["shape_energy"] / source["shape_energy"]
    gate_only_gain = (
        gate_only_endpoint / source["effective_gate_squared"]
    )
    alignment_gain = (
        endpoint["effective_gate_squared"] / gate_only_endpoint
    )
    physical_gain = (
        endpoint["physical_energy"] / source["physical_energy"]
    )
    reconstruction = shape_gain * gate_only_gain * alignment_gain
    return {
        "source": source,
        "endpoint": endpoint,
        "shape_gain": shape_gain,
        "gate_only_gain": gate_only_gain,
        "alignment_gain": alignment_gain,
        "physical_gain": physical_gain,
        "maximum_abs_gain_residual": float(
            np.max(np.abs(physical_gain - reconstruction))
        ),
        "maximum_abs_log_gain_residual": float(np.max(np.abs(
            np.log(physical_gain)
            - np.log(shape_gain)
            - np.log(gate_only_gain)
            - np.log(alignment_gain)
        ))),
    }


def reopening_budget_envelope(energy: Any) -> dict[str, Any]:
    """Bound a finite nonnegative path by its accumulated positive variation."""

    value = _array(energy, "energy")
    if value.ndim < 1 or np.any(value < 0.0):
        raise ValueError("energy must have a nonnegative leading time axis")
    increments = np.diff(value, axis=0)
    reopening = np.maximum(increments, 0.0)
    prefix = np.concatenate((
        np.zeros((1,) + value.shape[1:], dtype=np.float64),
        np.cumsum(reopening, axis=0),
    ), axis=0)
    envelope = value[0] + prefix
    tail = np.concatenate((
        np.flip(np.cumsum(np.flip(reopening, axis=0), axis=0), axis=0),
        np.zeros((1,) + value.shape[1:], dtype=np.float64),
    ), axis=0)
    return {
        "energy": value,
        "increments": increments,
        "reopening": reopening,
        "reopening_prefix": prefix,
        "reopening_tail": tail,
        "energy_envelope": envelope,
        "minimum_envelope_slack": float(np.min(envelope - value)),
    }


def symmetric_gate_shape_transport(
    shape0: Any,
    shape1: Any,
    gate0: Any,
    gate1: Any,
) -> dict[str, Any]:
    """Split a finite row increment symmetrically into shape and gate parts."""

    source = _rows(shape0, "shape0")
    endpoint = _rows(shape1, "shape1")
    if source.shape != endpoint.shape:
        raise ValueError("shape endpoints must have the same shape")
    features = source.shape[-1]
    gain0 = _feature_vector(gate0, features, "gate0")
    gain1 = _feature_vector(gate1, features, "gate1")
    centered0 = center_rows(source)
    centered1 = center_rows(endpoint)
    row0 = centered0 * gain0
    row1 = centered1 * gain1
    shape_increment = (centered1 - centered0) * (
        0.5 * (gain1 + gain0)
    )
    gate_increment = (0.5 * (centered1 + centered0)) * (gain1 - gain0)
    increment = row1 - row0
    reconstruction = shape_increment + gate_increment

    axes = (-2, -1)
    source_energy = np.sum(row0 * row0, axis=axes)
    endpoint_energy = np.sum(row1 * row1, axis=axes)
    shape_work = 2.0 * np.sum(row0 * shape_increment, axis=axes)
    gate_work = 2.0 * np.sum(row0 * gate_increment, axis=axes)
    shape_charge = np.sum(shape_increment * shape_increment, axis=axes)
    gate_charge = np.sum(gate_increment * gate_increment, axis=axes)
    interaction_charge = 2.0 * np.sum(
        shape_increment * gate_increment, axis=axes
    )
    ledger = (
        shape_work
        + gate_work
        + shape_charge
        + gate_charge
        + interaction_charge
    )
    return {
        "source_centered_rows": row0,
        "endpoint_centered_rows": row1,
        "centered_increment": increment,
        "shape_increment": shape_increment,
        "gate_increment": gate_increment,
        "source_energy": source_energy,
        "endpoint_energy": endpoint_energy,
        "shape_work": shape_work,
        "gate_work": gate_work,
        "shape_charge": shape_charge,
        "gate_charge": gate_charge,
        "interaction_charge": interaction_charge,
        "maximum_abs_transport_residual": float(
            np.max(np.abs(increment - reconstruction))
        ),
        "maximum_abs_ledger_residual": float(
            np.max(np.abs(endpoint_energy - source_energy - ledger))
        ),
    }


def coordinate_gate_shape_cocycle(
    shape0: Any,
    shape1: Any,
    gate0: Any,
    gate1: Any,
) -> dict[str, Any]:
    """Resolve nonzero coordinate-energy changes into exact log increments."""

    source = gate_shape_energy(shape0, gate0)
    endpoint = gate_shape_energy(shape1, gate1)
    e0 = source["coordinate_energy"]
    e1 = endpoint["coordinate_energy"]
    s0 = source["coordinate_shape_energy"]
    s1 = endpoint["coordinate_shape_energy"]
    g0 = _feature_vector(gate0, e0.shape[-1], "gate0")
    g1 = _feature_vector(gate1, e0.shape[-1], "gate1")
    occupied = (e0 > 0.0) & (e1 > 0.0)
    total_log = np.zeros_like(e0)
    gate_log = np.zeros_like(e0)
    shape_log = np.zeros_like(e0)
    total_log[occupied] = np.log(e1[occupied] / e0[occupied])
    safe_gate0 = np.where(g0 != 0.0, g0 * g0, 1.0)
    gate_ratio = np.broadcast_to((g1 * g1) / safe_gate0, e0.shape)
    shape_ratio = s1 / np.where(s0 > 0.0, s0, 1.0)
    gate_occupied = occupied & (g0 != 0.0) & (g1 != 0.0)
    shape_occupied = occupied & (s0 > 0.0) & (s1 > 0.0)
    gate_log[gate_occupied] = np.log(gate_ratio[gate_occupied])
    shape_log[shape_occupied] = np.log(shape_ratio[shape_occupied])
    decomposition_defined = gate_occupied & shape_occupied
    residual = np.zeros_like(e0)
    residual[decomposition_defined] = (
        total_log[decomposition_defined]
        - gate_log[decomposition_defined]
        - shape_log[decomposition_defined]
    )
    return {
        "source_coordinate_energy": e0,
        "endpoint_coordinate_energy": e1,
        "occupied_both_endpoints": occupied,
        "decomposition_defined": decomposition_defined,
        "closed_at_endpoint": (e0 > 0.0) & (e1 == 0.0),
        "reopened_at_endpoint": (e0 == 0.0) & (e1 > 0.0),
        "total_log_increment": total_log,
        "gate_log_increment": gate_log,
        "shape_log_increment": shape_log,
        "maximum_abs_log_residual": float(
            np.max(np.abs(residual[decomposition_defined]))
            if np.any(decomposition_defined)
            else 0.0
        ),
    }


def permutation_symmetric_attribution(
    source_names: Sequence[str],
    evaluate: Callable[[frozenset[str]], Any],
) -> dict[str, Any]:
    """Compute exact Shapley attribution for a finite macro-source set."""

    names = tuple(source_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("source names must be nonempty and unique")
    if len(names) > 8:
        raise ValueError("at most eight macro sources are supported")
    values: dict[frozenset[str], np.ndarray] = {}
    for count in range(len(names) + 1):
        for subset in combinations(names, count):
            key = frozenset(subset)
            values[key] = _array(evaluate(key), f"coalition {sorted(key)}")
    shape = values[frozenset()].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("all coalition evaluations must share one shape")

    count = len(names)
    contributions: dict[str, np.ndarray] = {}
    for name in names:
        contribution = np.zeros(shape, dtype=np.float64)
        others = tuple(item for item in names if item != name)
        for size in range(len(others) + 1):
            weight = (
                math.factorial(size)
                * math.factorial(count - size - 1)
                / math.factorial(count)
            )
            for subset in combinations(others, size):
                key = frozenset(subset)
                contribution += weight * (
                    values[key | {name}] - values[key]
                )
        contributions[name] = contribution
    baseline = values[frozenset()]
    endpoint = values[frozenset(names)]
    reconstruction = baseline + sum(
        contributions.values(), start=np.zeros_like(baseline)
    )
    return {
        "baseline": baseline,
        "endpoint": endpoint,
        "coalition_values": values,
        "contributions": contributions,
        "maximum_abs_efficiency_residual": float(
            np.max(np.abs(endpoint - reconstruction))
        ),
    }


def source_work_gram(
    source_rows: Any,
    contributions: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve exact source gain into signed work and a complete Gram matrix."""

    source = center_rows(source_rows)
    if not contributions:
        raise ValueError("at least one source contribution is required")
    names = tuple(contributions)
    if len(set(names)) != len(names):
        raise ValueError("source contribution names must be unique")
    centered: dict[str, np.ndarray] = {}
    for name, value in contributions.items():
        candidate = _rows(value, f"contribution {name}")
        if candidate.shape != source.shape:
            raise ValueError("source contributions must match the row shape")
        centered[name] = center_rows(candidate)
    total = sum(centered.values(), start=np.zeros_like(source))
    axes = (-2, -1)
    energy = np.sum(source * source, axis=axes)
    predicted_energy = np.sum((source + total) ** 2, axis=axes)
    defined = energy > 0.0
    safe_energy = np.where(defined, energy, 1.0)
    work = np.stack([
        -2.0 * np.sum(source * centered[name], axis=axes) / safe_energy
        for name in names
    ], axis=-1)
    gram = np.empty(energy.shape + (len(names), len(names)))
    for left, left_name in enumerate(names):
        for right, right_name in enumerate(names):
            gram[..., left, right] = np.sum(
                centered[left_name] * centered[right_name], axis=axes
            ) / safe_energy
    source_gain = predicted_energy / safe_energy
    reconstructed_gain = 1.0 - np.sum(work, axis=-1) + np.sum(
        gram, axis=(-2, -1)
    )
    work[~defined] = 0.0
    gram[~defined] = 0.0
    source_gain = np.where(defined, source_gain, 0.0)
    reconstructed_gain = np.where(defined, reconstructed_gain, 0.0)
    return {
        "source_names": names,
        "source_energy": energy,
        "predicted_energy": predicted_energy,
        "gain_defined": defined,
        "source_gain": source_gain,
        "radial_dissipation": work,
        "charge_gram": gram,
        "maximum_abs_gain_residual": float(
            np.max(np.abs(
                source_gain[defined] - reconstructed_gain[defined]
            )) if np.any(defined) else 0.0
        ),
    }


def source_native_energy_envelope(
    predicted_centered_rows: Any,
    native_radius: Any,
) -> dict[str, Any]:
    """Apply the outward source-to-native energy inequality map by map."""

    predicted = _rows(predicted_centered_rows, "predicted_centered_rows")
    radius = np.asarray(native_radius, dtype=np.float64)
    target_shape = predicted.shape[:-2]
    try:
        radius = np.broadcast_to(radius, target_shape)
    except ValueError as error:
        raise ValueError("native radius does not match the map population") from error
    if not np.isfinite(radius).all() or np.any(radius < 0.0):
        raise ValueError("native radius must be finite and nonnegative")
    norm = np.sqrt(np.sum(predicted * predicted, axis=(-2, -1)))
    predicted_energy = norm * norm
    charge = 2.0 * norm * radius + radius * radius
    return {
        "predicted_energy": predicted_energy,
        "native_energy_charge": charge,
        "native_energy_upper": predicted_energy + charge,
        "exact_when_zero_radius": radius == 0.0,
    }


def ordered_energy_envelope(
    gains: Any,
    charges: Any,
    initial_energy: Any,
) -> dict[str, Any]:
    """Compute the exact chronological affine energy envelope."""

    gain = _array(gains, "gains")
    charge = _array(charges, "charges")
    if gain.shape != charge.shape or gain.ndim < 1:
        raise ValueError("gains and charges must share a nonempty shape")
    if np.any(gain < 0.0) or np.any(charge < 0.0):
        raise ValueError("gains and charges must be nonnegative")
    initial = np.asarray(initial_energy, dtype=np.float64)
    population = gain.shape[1:]
    try:
        initial = np.broadcast_to(initial, population)
    except ValueError as error:
        raise ValueError("initial energy does not match the population") from error
    if not np.isfinite(initial).all() or np.any(initial < 0.0):
        raise ValueError("initial energy must be finite and nonnegative")

    steps = gain.shape[0]
    prefix = np.ones((steps + 1,) + population)
    homogeneous = np.empty_like(prefix)
    forcing = np.zeros_like(prefix)
    envelope = np.empty_like(prefix)
    homogeneous[0] = initial
    envelope[0] = initial
    for step in range(steps):
        prefix[step + 1] = prefix[step] * gain[step]
        homogeneous[step + 1] = prefix[step + 1] * initial
        forcing[step + 1] = gain[step] * forcing[step] + charge[step]
        envelope[step + 1] = homogeneous[step + 1] + forcing[step + 1]
    return {
        "prefix_products": prefix,
        "homogeneous_contribution": homogeneous,
        "transported_native_charge": forcing,
        "energy_envelope": envelope,
    }


def finite_log_gain_summary(gains: Any) -> dict[str, Any]:
    """Report a finite gain record without turning it into an asymptotic claim."""

    gain = _array(gains, "gains")
    if gain.ndim < 1 or np.any(gain < 0.0):
        raise ValueError("gains must have a leading time axis and be nonnegative")
    zero = gain == 0.0
    logs = np.zeros_like(gain)
    positive = ~zero
    logs[positive] = np.log(gain[positive])
    cumulative = np.cumsum(logs, axis=0)
    zero_seen = np.cumsum(zero.astype(np.int64), axis=0) > 0
    prefix = np.exp(np.where(zero_seen, 0.0, cumulative))
    prefix[zero_seen] = 0.0
    return {
        "log_gain": logs,
        "cumulative_log_gain": cumulative,
        "zero_gain_seen": zero_seen,
        "prefix_gain": prefix,
        "fraction_expanding_per_step": np.mean(gain > 1.0, axis=tuple(
            range(1, gain.ndim)
        )) if gain.ndim > 1 else (gain > 1.0).astype(np.float64),
    }


def partition_adamw_coordinates(
    updated_second_moment: Any,
    structurally_inactive: Any,
) -> dict[str, Any]:
    """Partition AdamW coordinates before any square-root division."""

    moment = _array(updated_second_moment, "updated_second_moment")
    inactive_proof = np.asarray(structurally_inactive, dtype=bool)
    if inactive_proof.shape != moment.shape:
        raise ValueError("structural inactivity mask has the wrong shape")
    if np.any(moment < 0.0):
        raise ValueError("updated second moments cannot be negative")
    active = moment > 0.0
    inactive = (moment == 0.0) & inactive_proof
    unresolved = (moment == 0.0) & ~inactive_proof
    if np.any((active.astype(int) + inactive.astype(int)
               + unresolved.astype(int)) != 1):
        raise AssertionError("AdamW coordinate strata are not exhaustive")
    return {
        "active": active,
        "structurally_inactive": inactive,
        "unresolved": unresolved,
        "active_count": int(np.count_nonzero(active)),
        "structurally_inactive_count": int(np.count_nonzero(inactive)),
        "unresolved_count": int(np.count_nonzero(unresolved)),
        "smooth_full_state_eligible": not np.any(unresolved),
    }


def inactive_adamw_position_multiplier(
    decay_mask: Any,
    *,
    learning_rate: float,
    weight_decay: float,
) -> np.ndarray:
    """Return the exact position derivative on a proved inactive coordinate."""

    mask = np.asarray(decay_mask)
    if mask.size == 0 or not np.isin(mask, [0, 1, False, True]).all():
        raise ValueError("decay mask must be a nonempty zero-one array")
    if (
        not np.isfinite(learning_rate)
        or not np.isfinite(weight_decay)
        or learning_rate < 0.0
        or weight_decay < 0.0
    ):
        raise ValueError("AdamW rate and decay must be finite and nonnegative")
    return 1.0 - learning_rate * weight_decay * mask.astype(np.float64)


def metric_nonvacuity(
    maximum_metric_eigenvalue: float,
    observable_envelope: Any,
    source_diameter: Any,
    layernorm_baseline: Any,
    *,
    eigenvalue_cap: float = 1.0e4,
) -> dict[str, Any]:
    """Apply conditioning and physical-baseline gates to a scheduled metric."""

    values = np.asarray(
        [maximum_metric_eigenvalue, eigenvalue_cap], dtype=np.float64
    )
    if not np.isfinite(values).all() or np.any(values <= 0.0):
        raise ValueError("metric eigenvalues and cap must be finite and positive")
    envelope = _array(observable_envelope, "observable_envelope")
    source = np.broadcast_to(_array(source_diameter, "source_diameter"), envelope.shape)
    baseline = np.broadcast_to(
        _array(layernorm_baseline, "layernorm_baseline"), envelope.shape
    )
    if np.any(envelope < 0.0) or np.any(source < 0.0) or np.any(baseline < 0.0):
        raise ValueError("physical comparison bounds must be nonnegative")
    conditioning_pass = maximum_metric_eigenvalue <= eigenvalue_cap
    observable_pass = (envelope <= source) & (envelope <= baseline)
    return {
        "conditioning_pass": bool(conditioning_pass),
        "metric_norm_conversion": float(np.sqrt(maximum_metric_eigenvalue)),
        "observable_pass": observable_pass,
        "eligible": observable_pass & conditioning_pass,
    }


def layernorm_hessian_bound(feature_count: int, epsilon: float) -> float:
    """Return a global conservative Euclidean Hessian bound for regularized LN."""

    if feature_count < 2 or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("LayerNorm feature count and epsilon are invalid")
    return float(6.0 / (math.sqrt(feature_count) * epsilon))


def plga_derivative_bounds(
    z_lower: float,
    z_upper: float,
    p_lower: float,
    p_upper: float,
    *,
    epsilon: float,
) -> dict[str, float]:
    """Conservatively bound PLGA z derivatives on a rectangular tube."""

    values = np.asarray(
        [z_lower, z_upper, p_lower, p_upper, epsilon], dtype=np.float64
    )
    if (
        not np.isfinite(values).all()
        or z_lower > z_upper
        or p_lower > p_upper
        or epsilon <= 0.0
    ):
        raise ValueError("PLGA derivative interval is invalid")
    z_abs = max(abs(z_lower), abs(z_upper))
    base_lower = epsilon
    base_upper = z_abs * z_abs + epsilon

    def power_bound(exponent_lower: float, exponent_upper: float) -> float:
        candidates = [
            base_lower ** exponent_lower,
            base_lower ** exponent_upper,
            base_upper ** exponent_lower,
            base_upper ** exponent_upper,
        ]
        if base_lower <= 1.0 <= base_upper:
            candidates.append(1.0)
        return float(max(candidates))

    p_abs = max(abs(p_lower), abs(p_upper))
    p_pair_abs = max(
        abs(p_lower * (p_lower - 1.0)),
        abs(p_lower * (p_upper - 1.0)),
        abs(p_upper * (p_lower - 1.0)),
        abs(p_upper * (p_upper - 1.0)),
    )
    base_prime = 2.0 * z_abs + 0.25 * z_abs * z_abs
    base_second = 2.0 + z_abs + 0.1 * z_abs * z_abs
    first = p_abs * power_bound(p_lower - 1.0, p_upper - 1.0) * base_prime
    second = (
        p_pair_abs
        * power_bound(p_lower - 2.0, p_upper - 2.0)
        * base_prime * base_prime
        + p_abs
        * power_bound(p_lower - 1.0, p_upper - 1.0)
        * base_second
    )
    return {
        "base_lower": float(base_lower),
        "base_upper": float(base_upper),
        "z_derivative_abs_upper": float(first),
        "z_second_derivative_abs_upper": float(second),
    }


def admissible_quadratic_radius(
    gain: float,
    quadratic_coefficient: float,
    force: float,
) -> dict[str, Any]:
    """Solve ``gain*R + c*R^2 + force <= R`` without fitting R."""

    values = np.asarray([gain, quadratic_coefficient, force], dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("radius coefficients must be finite and nonnegative")
    margin = 1.0 - gain
    if quadratic_coefficient == 0.0:
        if margin <= 0.0:
            return {
                "eligible": bool(force == 0.0 and gain == 1.0),
                "radius_lower": 0.0,
                "radius_upper": math.inf if force == 0.0 and gain <= 1.0 else 0.0,
                "discriminant": margin * margin,
            }
        return {
            "eligible": True,
            "radius_lower": float(force / margin),
            "radius_upper": math.inf,
            "discriminant": margin * margin,
        }
    discriminant = margin * margin - 4.0 * quadratic_coefficient * force
    if margin <= 0.0 or discriminant < 0.0:
        return {
            "eligible": False,
            "radius_lower": 0.0,
            "radius_upper": 0.0,
            "discriminant": float(discriminant),
        }
    root = math.sqrt(discriminant)
    return {
        "eligible": True,
        "radius_lower": float(
            (margin - root) / (2.0 * quadratic_coefficient)
        ),
        "radius_upper": float(
            (margin + root) / (2.0 * quadratic_coefficient)
        ),
        "discriminant": float(discriminant),
    }


def select_plga_attenuation_cap(
    construction_ratios: Any,
    *,
    menu: Sequence[float] = (0.05, 0.10, 0.20, 0.50),
) -> dict[str, Any]:
    """Select the smallest predeclared cap covering construction ratios."""

    ratio = _array(construction_ratios, "construction_ratios")
    if np.any(ratio < 0.0):
        raise ValueError("PLGA quotient ratios must be nonnegative")
    caps = tuple(float(value) for value in menu)
    if (
        not caps
        or any(not np.isfinite(value) or value <= 0.0 or value >= 1.0
               for value in caps)
        or tuple(sorted(set(caps))) != caps
    ):
        raise ValueError("PLGA cap menu must be strictly increasing in (0, 1)")
    maximum = float(np.max(ratio))
    selected = next((value for value in caps if maximum <= value), None)
    return {
        "construction_maximum_ratio": maximum,
        "selected_cap": selected,
        "attenuation_admitted": selected is not None,
        "menu": caps,
    }


def apply_frozen_plga_cap(
    ratios: Any,
    selected_cap: float | None,
) -> np.ndarray:
    """Apply a construction-frozen PLGA attenuation cap map by map."""

    ratio = _array(ratios, "ratios")
    if np.any(ratio < 0.0):
        raise ValueError("PLGA quotient ratios must be nonnegative")
    if selected_cap is None:
        return np.zeros(ratio.shape, dtype=bool)
    if (
        not np.isfinite(selected_cap)
        or selected_cap <= 0.0
        or selected_cap >= 1.0
    ):
        raise ValueError("selected PLGA cap must lie strictly between zero and one")
    return ratio <= selected_cap
