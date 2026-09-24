"""Executed-scale kernels for chronological PLDR row-map collapse.

These pure NumPy float64 replays construct predictions only from pre-update
state. Live adapters supply the gate, AdamW state, clipped-gradient history,
registered normalized rows, directional shape JVP, and a justified remainder
radius.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "pldr-chronological-collapse-kernel-v1"


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.ascontiguousarray(array)


def _scalar(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def unordered_pairs(vertex_count: int) -> np.ndarray:
    """Every unordered pair in deterministic lexicographic order."""

    count = int(vertex_count)
    if count < 2:
        raise ValueError("the row registry needs at least two vertices")
    left, right = np.triu_indices(count, k=1)
    return np.column_stack((left, right)).astype(np.int64, copy=False)


def pair_contrasts(
    normalized_rows: Any, pairs: Any | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct registered normalized-shape contrasts."""

    rows = _array(normalized_rows, "normalized_rows", 2)
    pair_index = (
        unordered_pairs(rows.shape[0])
        if pairs is None else np.asarray(pairs, dtype=np.int64)
    )
    if (
        pair_index.ndim != 2
        or pair_index.shape[1] != 2
        or pair_index.size == 0
        or np.any(pair_index < 0)
        or np.any(pair_index >= rows.shape[0])
        or np.any(pair_index[:, 0] >= pair_index[:, 1])
    ):
        raise ValueError("pairs must be ordered distinct vertex pairs")
    if len({tuple(row) for row in pair_index.tolist()}) != len(pair_index):
        raise ValueError("pair registry contains duplicates")
    contrast = rows[pair_index[:, 0]] - rows[pair_index[:, 1]]
    return np.ascontiguousarray(contrast), np.ascontiguousarray(pair_index)


def pair_energies(gate: Any, contrasts: Any) -> np.ndarray:
    """Squared physical-row distance for every supplied pair."""

    gamma = _array(gate, "gate", 1)
    contrast = _array(contrasts, "contrasts", 2)
    if contrast.shape[1] != gamma.size:
        raise ValueError("gate and pair widths disagree")
    return np.einsum(
        "pd,d,pd->p", contrast, gamma * gamma, contrast, optimize=True)


def adamw_chronological_direction(
    clipped_gradient_history: Any,
    first_moment_tail: Any,
    second_moment_before: Any,
    *,
    beta1: float,
    beta2: float,
    epsilon: float,
    optimizer_step_after: int,
) -> dict[str, Any]:
    """Resolve the exact bias-corrected AdamW direction chronologically.

    History is newest first. For length K ending at optimizer step n,
    first_moment_tail is m_(n-K), so the stored first moment is the signed
    history sum plus beta1 to the K times that tail.
    """

    history = _array(clipped_gradient_history, "gradient_history", 2)
    tail = _array(first_moment_tail, "first_moment_tail", 1)
    second_before = _array(second_moment_before, "second_moment_before", 1)
    if history.shape[1] != tail.size or second_before.shape != tail.shape:
        raise ValueError("AdamW chronological array widths disagree")
    length = history.shape[0]
    step = int(optimizer_step_after)
    beta1 = _scalar(beta1, "beta1")
    beta2 = _scalar(beta2, "beta2")
    epsilon = _scalar(epsilon, "epsilon")
    if not 0.0 <= beta1 < 1.0 or not 0.0 <= beta2 < 1.0:
        raise ValueError("Adam coefficients must lie in [0, 1)")
    if epsilon <= 0.0 or length < 1 or length > step:
        raise ValueError("invalid Adam epsilon, history length, or step")
    if np.any(second_before < 0.0):
        raise ValueError("the second moment must be nonnegative")

    powers = beta1 ** np.arange(length, dtype=np.float64)
    first_history = (1.0 - beta1) * powers[:, None] * history
    first_tail = beta1 ** length * tail
    first_after = np.sum(first_history, axis=0) + first_tail
    current = history[0]
    second_after = (
        beta2 * second_before + (1.0 - beta2) * current * current)
    first_bias = 1.0 - beta1 ** step
    second_bias = 1.0 - beta2 ** step
    first_hat = first_after / first_bias
    second_hat = second_after / second_bias
    preconditioner = 1.0 / (np.sqrt(second_hat) + epsilon)
    history_direction = preconditioner[None, :] * first_history / first_bias
    tail_direction = preconditioner * first_tail / first_bias
    direction = preconditioner * first_hat
    reconstructed = np.sum(history_direction, axis=0) + tail_direction
    scale = max(
        float(np.linalg.norm(direction)),
        float(np.linalg.norm(reconstructed)),
        np.finfo(np.float64).tiny,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "history_length": length,
        "optimizer_step_after": step,
        "first_moment_after": first_after,
        "second_moment_after": second_after,
        "first_moment_hat_after": first_hat,
        "second_moment_hat_after": second_hat,
        "preconditioner": preconditioner,
        "history_direction": history_direction,
        "tail_direction": tail_direction,
        "direction": direction,
        "reconstruction_relative_residual": (
            float(np.linalg.norm(direction - reconstructed)) / scale),
    }


def executed_pair_gate_balance(
    gate: Any,
    contrasts: Any,
    adaptive_direction: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    decay_mask: Any | None = None,
) -> dict[str, Any]:
    """Exact fixed-shape AdamW balance for every registered pair."""

    gamma = _array(gate, "gate", 1)
    contrast = _array(contrasts, "contrasts", 2)
    direction = _array(adaptive_direction, "adaptive_direction", 1)
    eta = _scalar(learning_rate, "learning_rate")
    decay = _scalar(weight_decay, "weight_decay")
    if contrast.shape[1] != gamma.size or direction.shape != gamma.shape:
        raise ValueError("pair gate balance dimensions disagree")
    if eta < 0.0 or decay < 0.0:
        raise ValueError("learning rate and decay must be nonnegative")
    mask = (
        np.ones_like(gamma)
        if decay_mask is None else _array(decay_mask, "decay_mask", 1)
    )
    if mask.shape != gamma.shape or np.any((mask < 0.0) | (mask > 1.0)):
        raise ValueError("decay mask entries must lie in [0, 1]")

    decayed_gate = (1.0 - eta * decay * mask) * gamma
    gate_next = decayed_gate - eta * direction
    metric = contrast * contrast
    energy_before = np.einsum(
        "pd,d->p", metric, gamma * gamma, optimize=True)
    energy_decay = np.einsum(
        "pd,d->p", metric, decayed_gate * decayed_gate, optimize=True)
    energy_next = np.einsum(
        "pd,d->p", metric, gate_next * gate_next, optimize=True)
    decay_dissipation = energy_before - energy_decay
    alignment = 2.0 * eta * np.einsum(
        "pd,d,d->p", metric, decayed_gate, direction, optimize=True)
    finite_charge = eta * eta * np.einsum(
        "pd,d->p", metric, direction * direction, optimize=True)
    margin = decay_dissipation + alignment - finite_charge
    residual = energy_next - energy_before + margin
    scale = np.maximum.reduce((
        np.abs(energy_next),
        np.abs(energy_before),
        np.abs(margin),
        np.full_like(margin, np.finfo(np.float64).tiny),
    ))
    return {
        "energy_before": energy_before,
        "energy_after_fixed_shape": energy_next,
        "decayed_gate": decayed_gate,
        "gate_next": gate_next,
        "decay_dissipation": decay_dissipation,
        "chronological_alignment": alignment,
        "finite_step_charge": finite_charge,
        "gate_margin": margin,
        "identity_relative_residual": np.abs(residual) / scale,
    }


def directional_shape_enclosure(
    contrasts: Any,
    directional_increment: Any,
    remainder_radius: Any,
    gate_next: Any,
) -> dict[str, Any]:
    """Upper-enclose shape work from a pre-update JVP and radius."""

    contrast = _array(contrasts, "contrasts", 2)
    increment = _array(directional_increment, "directional_increment", 2)
    radius = _array(remainder_radius, "remainder_radius", 1)
    gamma_next = _array(gate_next, "gate_next", 1)
    if (
        increment.shape != contrast.shape
        or radius.shape != (contrast.shape[0],)
        or gamma_next.size != contrast.shape[1]
    ):
        raise ValueError("shape enclosure dimensions disagree")
    if np.any(radius < 0.0):
        raise ValueError("shape remainder radii must be nonnegative")
    source = contrast * gamma_next
    linear_endpoint = (contrast + increment) * gamma_next
    linear_work = (
        np.einsum("pd,pd->p", linear_endpoint, linear_endpoint)
        - np.einsum("pd,pd->p", source, source)
    )
    gate_inf = float(np.max(np.abs(gamma_next), initial=0.0))
    remainder_charge = (
        2.0 * np.linalg.norm(linear_endpoint, axis=1) * gate_inf * radius
        + gate_inf * gate_inf * radius * radius
    )
    return {
        "linear_shape_work": linear_work,
        "shape_remainder_charge": remainder_charge,
        "shape_work_upper": linear_work + remainder_charge,
    }


def chronological_pair_certificate(
    gate: Any,
    contrasts: Any,
    adaptive_direction: Any,
    directional_shape_increment: Any,
    shape_remainder_radius: Any,
    numerical_charge: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    decay_mask: Any | None = None,
) -> dict[str, Any]:
    """Construct the pre-update lower margin and direct pair envelope."""

    gate_balance = executed_pair_gate_balance(
        gate,
        contrasts,
        adaptive_direction,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        decay_mask=decay_mask,
    )
    shape = directional_shape_enclosure(
        contrasts,
        directional_shape_increment,
        shape_remainder_radius,
        gate_balance["gate_next"],
    )
    numerical = _array(numerical_charge, "numerical_charge", 1)
    energy = gate_balance["energy_before"]
    if numerical.shape != energy.shape or np.any(numerical < 0.0):
        raise ValueError("numerical charges must be nonnegative per pair")
    lower_margin = (
        gate_balance["gate_margin"] - shape["shape_work_upper"] - numerical)
    factor = np.zeros_like(energy)
    forcing = np.zeros_like(energy)
    positive = energy > 0.0
    factor[positive] = np.maximum(
        0.0, 1.0 - lower_margin[positive] / energy[positive])
    forcing[~positive] = np.maximum(0.0, -lower_margin[~positive])
    return {
        **gate_balance,
        **shape,
        "numerical_charge": numerical,
        "chronological_margin_lower": lower_margin,
        "affine_factor": factor,
        "affine_forcing": forcing,
        "positive_margin": lower_margin > 0.0,
    }


def verify_shape_enclosure(
    contrasts: Any,
    realized_increment: Any,
    enclosure: dict[str, Any],
    gate_next: Any,
) -> dict[str, Any]:
    """Check the shape bound after opening a realized successor."""

    contrast = _array(contrasts, "contrasts", 2)
    increment = _array(realized_increment, "realized_increment", 2)
    gamma_next = _array(gate_next, "gate_next", 1)
    if increment.shape != contrast.shape or gamma_next.size != contrast.shape[1]:
        raise ValueError("realized shape dimensions disagree")
    source = contrast * gamma_next
    target = (contrast + increment) * gamma_next
    work = (
        np.einsum("pd,pd->p", target, target)
        - np.einsum("pd,pd->p", source, source)
    )
    upper = _array(enclosure["shape_work_upper"], "shape_work_upper", 1)
    if upper.shape != work.shape:
        raise ValueError("shape enclosure pair count disagrees")
    return {
        "realized_shape_work": work,
        "covered": work <= upper,
        "coverage_slack": upper - work,
    }


def iterate_pair_envelope(
    initial_energy: Any, factors: Any, forcing: Any,
) -> np.ndarray:
    """Iterate each pair chronologically before taking a maximum."""

    initial = _array(initial_energy, "initial_energy", 1)
    coefficient = _array(factors, "factors", 2)
    source = _array(forcing, "forcing", 2)
    if coefficient.shape != source.shape or coefficient.shape[1] != initial.size:
        raise ValueError("pair envelope dimensions disagree")
    if np.any(coefficient < 0.0) or np.any(source < 0.0):
        raise ValueError("pair factors and forcing must be nonnegative")
    envelope = np.empty((coefficient.shape[0] + 1, initial.size))
    envelope[0] = initial
    for step in range(coefficient.shape[0]):
        envelope[step + 1] = (
            coefficient[step] * envelope[step] + source[step])
    return envelope


def diameter_upper_from_pair_envelope(envelope: Any) -> np.ndarray:
    """Convert squared all-pairs envelopes to diameter bounds."""

    values = _array(envelope, "envelope", 2)
    if np.any(values < -64.0 * np.finfo(np.float64).eps):
        raise ValueError("squared pair envelope is negative")
    return np.sqrt(np.maximum(0.0, np.max(values, axis=1)))


def source_removal_response(
    gate_next_baseline: Any,
    contrasts: Any,
    source_direction: Any,
    *,
    learning_rate: float,
) -> dict[str, Any]:
    """Exact fixed-shape response to removing one adaptive source."""

    baseline = _array(gate_next_baseline, "gate_next_baseline", 1)
    contrast = _array(contrasts, "contrasts", 2)
    source = _array(source_direction, "source_direction", 1)
    eta = _scalar(learning_rate, "learning_rate")
    if source.shape != baseline.shape or contrast.shape[1] != baseline.size:
        raise ValueError("source-removal response dimensions disagree")
    metric = contrast * contrast
    linear = 2.0 * eta * np.einsum(
        "pd,d,d->p", metric, baseline, source, optimize=True)
    quadratic = eta * eta * np.einsum(
        "pd,d->p", metric, source * source, optimize=True)
    response = linear + quadratic
    return {
        "linear_response": linear,
        "quadratic_response": quadratic,
        "predicted_energy_response": response,
        "predicted_direction": np.sign(response).astype(np.int64),
        "predicted_magnitude": np.abs(response),
    }


def occupied_power_secant(left: Any, right: Any, exponent: Any) -> dict[str, Any]:
    """Exact positive-base real-power secant on occupied endpoints."""

    x = _array(left, "left")
    y = _array(right, "right")
    power = _array(exponent, "exponent")
    if x.shape != y.shape or x.shape != power.shape:
        raise ValueError("power secant endpoint shapes disagree")
    if np.any(x <= 0.0) or np.any(y <= 0.0):
        raise ValueError("power secant bases must be strictly positive")
    equal = x == y
    coefficient = np.empty_like(x)
    coefficient[equal] = power[equal] * x[equal] ** (power[equal] - 1.0)
    unequal = ~equal
    coefficient[unequal] = (
        y[unequal] ** power[unequal] - x[unequal] ** power[unequal]
    ) / (y[unequal] - x[unequal])
    low = np.minimum(x, y)
    high = np.maximum(x, y)
    derivative_bound = np.abs(power) * np.maximum(
        low ** (power - 1.0), high ** (power - 1.0))
    prediction = coefficient * (y - x)
    realized = y ** power - x ** power
    scale = np.maximum.reduce((
        np.abs(prediction),
        np.abs(realized),
        np.full_like(realized, np.finfo(np.float64).tiny),
    ))
    return {
        "coefficient": coefficient,
        "derivative_bound": derivative_bound,
        "identity_relative_residual": np.abs(realized - prediction) / scale,
    }


def rectangular_score_transfer(
    query_reference: Any,
    query_candidate: Any,
    generator_reference: Any,
    generator_candidate: Any,
    key_reference: Any,
    key_candidate: Any,
) -> dict[str, Any]:
    """Dimension-correct chronological telescoping of Q G K transpose."""

    q0 = _array(query_reference, "query_reference", 2)
    q1 = _array(query_candidate, "query_candidate", 2)
    g0 = _array(generator_reference, "generator_reference", 2)
    g1 = _array(generator_candidate, "generator_candidate", 2)
    k0 = _array(key_reference, "key_reference", 2)
    k1 = _array(key_candidate, "key_candidate", 2)
    if (
        q0.shape != q1.shape
        or k0.shape != k1.shape
        or g0.shape != g1.shape
        or g0.shape[0] != g0.shape[1]
        or q0.shape[1] != g0.shape[0]
        or k0.shape[1] != g0.shape[1]
    ):
        raise ValueError("rectangular score transfer dimensions disagree")
    scale = math.sqrt(g0.shape[0])
    reference = q0 @ g0 @ k0.T / scale
    candidate = q1 @ g1 @ k1.T / scale
    query_term = (q1 - q0) @ g0 @ k0.T / scale
    generator_term = q1 @ (g1 - g0) @ k0.T / scale
    key_term = q1 @ g1 @ (k1 - k0).T / scale
    prediction = query_term + generator_term + key_term
    realized = candidate - reference
    denominator = max(
        float(np.linalg.norm(realized)),
        float(np.linalg.norm(prediction)),
        np.finfo(np.float64).tiny,
    )
    return {
        "reference_score": reference,
        "candidate_score": candidate,
        "query_term": query_term,
        "generator_term": generator_term,
        "key_term": key_term,
        "predicted_difference": prediction,
        "realized_difference": realized,
        "identity_relative_residual": (
            float(np.linalg.norm(realized - prediction)) / denominator),
    }


def centered_argmax_margin(logits: Any, candidate: int) -> float:
    """Sharp reference margin for one candidate token."""

    value = _array(logits, "logits", 1)
    candidate = int(candidate)
    if not 0 <= candidate < value.size:
        raise ValueError("candidate index is outside the vocabulary")
    return float(value[candidate] - np.max(np.delete(value, candidate)))


def provenance_tolerance(
    dtype: str, operation_count: int, scale: float, safety: float = 16.0,
) -> float:
    """Derived absolute tolerance for one arithmetic provenance."""

    try:
        epsilon = np.finfo(np.dtype(dtype)).eps
    except (TypeError, ValueError) as error:
        raise ValueError("dtype lacks floating-point precision") from error
    count = int(operation_count)
    scale = _scalar(scale, "scale")
    safety = _scalar(safety, "safety")
    if count < 1 or scale < 0.0 or safety < 1.0:
        raise ValueError("invalid tolerance inputs")
    return float(safety * count * epsilon * max(scale, 1.0))


__all__ = [
    "SCHEMA_VERSION",
    "adamw_chronological_direction",
    "centered_argmax_margin",
    "chronological_pair_certificate",
    "diameter_upper_from_pair_envelope",
    "directional_shape_enclosure",
    "executed_pair_gate_balance",
    "iterate_pair_envelope",
    "occupied_power_secant",
    "pair_contrasts",
    "pair_energies",
    "provenance_tolerance",
    "rectangular_score_transfer",
    "source_removal_response",
    "unordered_pairs",
    "verify_shape_enclosure",
]

