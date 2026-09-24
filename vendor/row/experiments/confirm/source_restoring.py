"""Source-state certificate kernels for PLDR row-map collapse.

The deciding arrays contain only source rows, source directional derivatives,
validated source-segment remainder intervals, and source-derived native
arithmetic charges. Native successor rows are accepted only by the separate
coverage function.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


SCHEMA_VERSION = "pldr-source-restoring-kernel-v1"
CERTIFICATE_SCHEMA = "pldr-source-restoring-certificate-v1"
COVERAGE_SCHEMA = "pldr-source-restoring-coverage-v1"
TIMECOURSE_SCHEMA = "pldr-source-restoring-timecourse-v1"
SCIENTIFIC_OUTCOMES = ("confirmed", "not_confirmed", "unresolved")


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.ascontiguousarray(array)


def _broadcast_nonnegative(
    value: Any,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _array(value, name)
    try:
        result = np.broadcast_to(array, shape).astype(np.float64, copy=True)
    except ValueError as error:
        raise ValueError(f"{name} cannot broadcast to {shape}") from error
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def unordered_pair_indices(row_count: int) -> np.ndarray:
    """Return every unordered row pair in lexicographic order."""

    count = int(row_count)
    if count < 2:
        raise ValueError("a row map needs at least two rows")
    left, right = np.triu_indices(count, k=1)
    return np.column_stack((left, right)).astype(np.int64, copy=False)


def pair_contrasts(rows: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return within-map row contrasts without forming cross-map pairs."""

    value = _array(rows, "rows", 3)
    pairs = unordered_pair_indices(value.shape[1])
    contrasts = value[:, pairs[:, 0], :] - value[:, pairs[:, 1], :]
    return pairs, np.ascontiguousarray(contrasts)


def pair_energies(rows: Any) -> tuple[np.ndarray, np.ndarray]:
    pairs, contrasts = pair_contrasts(rows)
    energies = np.einsum(
        "mpd,mpd->mp", contrasts, contrasts, optimize=True
    )
    return pairs, energies


def _directed_sum(
    values: tuple[np.ndarray, ...], direction: float
) -> np.ndarray:
    if not values:
        raise ValueError("a directed sum needs at least one term")
    result = np.zeros_like(values[0])
    target = np.full_like(result, direction)
    for value in values:
        result = np.nextafter(result + value, target)
    return result


def _interval_product(
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
    lower = np.nextafter(
        np.min(products, axis=0), np.full_like(left_lower, -np.inf)
    )
    upper = np.nextafter(
        np.max(products, axis=0), np.full_like(left_upper, np.inf)
    )
    return lower, upper


def _contrast_coordinate(
    rows: np.ndarray, pairs: np.ndarray, coordinate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nominal = (
        rows[:, pairs[:, 0], coordinate]
        - rows[:, pairs[:, 1], coordinate]
    )
    return (
        nominal,
        np.nextafter(nominal, np.full_like(nominal, -np.inf)),
        np.nextafter(nominal, np.full_like(nominal, np.inf)),
    )


def pair_energy_work_intervals(
    source_rows: Any, velocity_rows: tuple[Any, ...] = ()
) -> dict[str, Any]:
    """Evaluate pair energies and signed work with directed endpoints."""

    source = _array(source_rows, "source_rows", 3)
    components = tuple(
        _array(value, f"velocity_rows_{index}", 3)
        for index, value in enumerate(velocity_rows)
    )
    if any(value.shape != source.shape for value in components):
        raise ValueError("source rows and velocity blocks must have one shape")
    pairs = unordered_pair_indices(source.shape[1])
    shape = (source.shape[0], len(pairs))
    energy = np.zeros(shape, dtype=np.float64)
    energy_lower = np.zeros_like(energy)
    energy_upper = np.zeros_like(energy)
    work = [np.zeros_like(energy) for _value in components]
    work_lower = [np.zeros_like(energy) for _value in components]
    work_upper = [np.zeros_like(energy) for _value in components]
    direct_total = np.zeros_like(energy)
    negative_infinity = np.full(shape, -np.inf)
    positive_infinity = np.full(shape, np.inf)
    with np.errstate(over="raise", invalid="raise"):
        for coordinate in range(source.shape[2]):
            source_nominal, source_lower, source_upper = _contrast_coordinate(
                source, pairs, coordinate
            )
            square_lower, square_upper = _interval_product(
                source_lower, source_upper, source_lower, source_upper
            )
            square_lower = np.maximum(0.0, square_lower)
            energy = energy + source_nominal * source_nominal
            energy_lower = np.nextafter(
                energy_lower + square_lower, negative_infinity
            )
            energy_upper = np.nextafter(
                energy_upper + square_upper, positive_infinity
            )
            component_nominals = []
            for index, component in enumerate(components):
                nominal, lower, upper = _contrast_coordinate(
                    component, pairs, coordinate
                )
                component_nominals.append(nominal)
                product_lower, product_upper = _interval_product(
                    source_lower, source_upper, lower, upper
                )
                product_lower = np.nextafter(
                    2.0 * product_lower, negative_infinity
                )
                product_upper = np.nextafter(
                    2.0 * product_upper, positive_infinity
                )
                work[index] = (
                    work[index] + 2.0 * source_nominal * nominal
                )
                work_lower[index] = np.nextafter(
                    work_lower[index] + product_lower, negative_infinity
                )
                work_upper[index] = np.nextafter(
                    work_upper[index] + product_upper, positive_infinity
                )
            if component_nominals:
                total_nominal = sum(
                    component_nominals, start=np.zeros_like(source_nominal)
                )
                direct_total = (
                    direct_total + 2.0 * source_nominal * total_nominal
                )
    energy_lower = np.maximum(0.0, energy_lower)
    total_work = (
        sum(work, start=np.zeros_like(energy))
        if work else np.zeros_like(energy)
    )
    total_lower = (
        _directed_sum(tuple(work_lower), -np.inf)
        if work_lower else np.zeros_like(energy)
    )
    total_upper = (
        _directed_sum(tuple(work_upper), np.inf)
        if work_upper else np.zeros_like(energy)
    )
    return {
        "pairs": pairs,
        "source_energy": energy,
        "source_energy_lower": energy_lower,
        "source_energy_upper": energy_upper,
        "signed_work": tuple(work),
        "signed_work_lower": tuple(work_lower),
        "signed_work_upper": tuple(work_upper),
        "signed_total_work": total_work,
        "signed_total_work_lower": total_lower,
        "signed_total_work_upper": total_upper,
        "work_additivity_residual": direct_total - total_work,
    }


def pair_energy_intervals(rows: Any) -> dict[str, Any]:
    """Return nominal pair energies and directed lower and upper endpoints."""

    return pair_energy_work_intervals(rows)


def centered_row_geometry(rows: Any) -> dict[str, np.ndarray | bool | float]:
    """Evaluate the exact centering identity and finite norm equivalence."""

    value = _array(rows, "rows", 3)
    pairs, energies = pair_energies(value)
    centered = value - np.mean(value, axis=1, keepdims=True)
    centered_energy = np.einsum(
        "mrd,mrd->m", centered, centered, optimize=True
    )
    unordered_sum = np.sum(energies, axis=1)
    row_count = value.shape[1]
    identity_residual = unordered_sum - row_count * centered_energy
    diameter_squared = np.max(energies, axis=1)
    lower_slack = 2.0 * centered_energy - diameter_squared
    upper_slack = (
        (row_count - 1.0) * diameter_squared - 2.0 * centered_energy
    )
    scale = np.maximum.reduce((
        np.ones_like(centered_energy),
        np.abs(unordered_sum),
        row_count * np.abs(centered_energy),
    ))
    tolerance = 256.0 * np.finfo(np.float64).eps * scale
    valid = bool(
        np.all(np.abs(identity_residual) <= tolerance)
        and np.all(lower_slack >= -tolerance)
        and np.all(upper_slack >= -tolerance)
    )
    return {
        "pairs": pairs,
        "centered_energy": centered_energy,
        "unordered_pair_energy_sum": unordered_sum,
        "diameter_squared": diameter_squared,
        "identity_residual": identity_residual,
        "lower_equivalence_slack": lower_slack,
        "upper_equivalence_slack": upper_slack,
        "valid": valid,
        "maximum_abs_identity_residual": float(
            np.max(np.abs(identity_residual))
        ),
    }


def signed_source_work(
    source_rows: Any,
    gate_velocity_rows: Any,
    shape_velocity_rows: Any,
    input_velocity_rows: Any,
) -> dict[str, np.ndarray | float]:
    """Split the source energy derivative into three additive blocks."""

    source = _array(source_rows, "source_rows", 3)
    components = [
        _array(gate_velocity_rows, "gate_velocity_rows", 3),
        _array(shape_velocity_rows, "shape_velocity_rows", 3),
        _array(input_velocity_rows, "input_velocity_rows", 3),
    ]
    if any(component.shape != source.shape for component in components):
        raise ValueError("source rows and velocity blocks must have one shape")
    pairs, source_contrast = pair_contrasts(source)
    velocity_contrasts = [pair_contrasts(value)[1] for value in components]
    work = [
        2.0 * np.einsum(
            "mpd,mpd->mp", source_contrast, contrast, optimize=True
        )
        for contrast in velocity_contrasts
    ]
    total_contrast = sum(
        velocity_contrasts, start=np.zeros_like(source_contrast)
    )
    total = 2.0 * np.einsum(
        "mpd,mpd->mp", source_contrast, total_contrast, optimize=True
    )
    residual = total - sum(work, start=np.zeros_like(total))
    return {
        "pairs": pairs,
        "source_contrasts": source_contrast,
        "gate_velocity_contrasts": velocity_contrasts[0],
        "shape_velocity_contrasts": velocity_contrasts[1],
        "input_velocity_contrasts": velocity_contrasts[2],
        "total_velocity_contrasts": total_contrast,
        "gate_work": work[0],
        "shape_work": work[1],
        "input_work": work[2],
        "total_work": total,
        "additivity_residual": residual,
        "maximum_abs_additivity_residual": float(
            np.max(np.abs(residual))
        ),
    }


def symmetric_taylor_remainder(
    energy_curvature_abs_upper: Any,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert an absolute second-derivative bound to a Taylor interval."""

    curvature = _broadcast_nonnegative(
        energy_curvature_abs_upper,
        "energy_curvature_abs_upper",
        shape,
    )
    radius = 0.5 * curvature
    return -radius, radius


def native_energy_charge(
    endpoint_contrast_norm_upper: Any,
    native_contrast_radius: Any,
    shape: tuple[int, int],
) -> np.ndarray:
    """Bound energy change caused by a source-derived contrast perturbation."""

    norm_upper = _broadcast_nonnegative(
        endpoint_contrast_norm_upper,
        "endpoint_contrast_norm_upper",
        shape,
    )
    radius = _broadcast_nonnegative(
        native_contrast_radius,
        "native_contrast_radius",
        shape,
    )
    with np.errstate(over="raise", invalid="raise"):
        cross = np.nextafter(
            norm_upper * radius, np.full(shape, np.inf)
        )
        cross = np.nextafter(2.0 * cross, np.full(shape, np.inf))
        square = np.nextafter(
            radius * radius, np.full(shape, np.inf)
        )
        result = np.nextafter(cross + square, np.full(shape, np.inf))
    if not np.isfinite(result).all():
        raise ValueError("native energy charge overflowed")
    return result


def two_channel_spatial_bound(
    gamma_infinity_norm: Any,
    shape_chord_coefficient: Any,
    input_diameter: Any,
    feature_dimension: int,
) -> dict[str, np.ndarray]:
    """Return universal-gain, occupied-chord, and combined spatial bounds."""

    dimension = int(feature_dimension)
    if dimension < 1:
        raise ValueError("feature_dimension must be positive")
    gamma = _array(gamma_infinity_norm, "gamma_infinity_norm")
    coefficient = _array(
        shape_chord_coefficient, "shape_chord_coefficient"
    )
    incoming = _array(input_diameter, "input_diameter")
    try:
        gamma, coefficient, incoming = np.broadcast_arrays(
            gamma, coefficient, incoming
        )
    except ValueError as error:
        raise ValueError("spatial factors cannot be broadcast together") from error
    if (
        np.any(gamma < 0.0)
        or np.any(coefficient < 0.0)
        or np.any(incoming < 0.0)
    ):
        raise ValueError("spatial factors must be nonnegative")
    universal = 2.0 * math.sqrt(dimension) * gamma
    chord = gamma * coefficient * incoming
    return {
        "universal_gain_bound": np.asarray(universal, dtype=np.float64),
        "occupied_chord_bound": np.asarray(chord, dtype=np.float64),
        "combined_bound": np.minimum(universal, chord),
    }


def source_restoring_certificate(
    source_rows: Any,
    gate_velocity_rows: Any,
    shape_velocity_rows: Any,
    input_velocity_rows: Any,
    *,
    remainder_lower: Any,
    remainder_upper: Any,
    endpoint_contrast_norm_upper: Any,
    native_contrast_radius: Any,
) -> dict[str, Any]:
    """Construct the complete source-only directed pair certificate.

    The remainder endpoints enclose the nonlinear remainder on the
    source-defined parameter segment. No successor value is an input.
    """

    source = _array(source_rows, "source_rows", 3)
    interval = pair_energy_work_intervals(
        source,
        (gate_velocity_rows, shape_velocity_rows, input_velocity_rows),
    )
    pairs = np.asarray(interval["pairs"])
    energy = np.asarray(interval["source_energy"])
    energy_lower = np.asarray(interval["source_energy_lower"])
    energy_upper = np.asarray(interval["source_energy_upper"])
    shape = energy.shape
    lower = _array(remainder_lower, "remainder_lower")
    upper = _array(remainder_upper, "remainder_upper")
    try:
        lower = np.broadcast_to(lower, shape).astype(np.float64, copy=True)
        upper = np.broadcast_to(upper, shape).astype(np.float64, copy=True)
    except ValueError as error:
        raise ValueError(
            "remainder intervals do not match the pair registry"
        ) from error
    if np.any(lower > upper):
        raise ValueError("a remainder lower endpoint exceeds its upper endpoint")
    charge = native_energy_charge(
        endpoint_contrast_norm_upper,
        native_contrast_radius,
        shape,
    )
    work = tuple(np.asarray(value) for value in interval["signed_work"])
    work_lower = tuple(
        np.asarray(value) for value in interval["signed_work_lower"]
    )
    work_upper = tuple(
        np.asarray(value) for value in interval["signed_work_upper"]
    )
    total_work = np.asarray(interval["signed_total_work"])
    total_work_lower = np.asarray(interval["signed_total_work_lower"])
    total_work_upper = np.asarray(interval["signed_total_work_upper"])
    diameter_squared = np.max(energy, axis=1)
    diameter_lower = np.max(energy_lower, axis=1)
    diameter_upper = np.max(energy_upper, axis=1)
    negative_infinity = np.full(shape, -np.inf)
    gap_nominal = diameter_squared[:, None] - energy
    gap_lower = np.nextafter(
        diameter_lower[:, None] - energy_upper, negative_infinity
    )
    upper_energy = np.maximum(0.0, _directed_sum((
        energy_upper, total_work_upper, upper, charge
    ), np.inf))
    lower_energy = np.maximum(0.0, _directed_sum((
        energy_lower, total_work_lower, lower, -charge
    ), -np.inf))
    margin_lower = _directed_sum((
        gap_lower, -total_work_upper, -upper, -charge
    ), -np.inf)
    minimum_margin = np.min(margin_lower, axis=1)
    predicted_upper = np.max(upper_energy, axis=1)
    predicted_lower = np.max(lower_energy, axis=1)
    upper_multiplier = np.full_like(diameter_squared, np.nan)
    positive = diameter_lower > 0.0
    ratios = predicted_upper[positive] / diameter_lower[positive]
    upper_multiplier[positive] = np.where(
        predicted_upper[positive] == 0.0,
        0.0,
        np.nextafter(ratios, np.full_like(ratios, np.inf)),
    )
    restoring_coefficient = np.zeros_like(diameter_squared)
    restoring = positive & (upper_multiplier < 1.0)
    restoring_coefficient[restoring] = np.clip(
        np.nextafter(
            1.0 - upper_multiplier[restoring],
            np.full(np.count_nonzero(restoring), -np.inf),
        ),
        0.0,
        1.0,
    )
    outcomes = []
    for index in range(len(diameter_squared)):
        if predicted_upper[index] < diameter_lower[index]:
            outcomes.append("confirmed")
        elif predicted_lower[index] > diameter_upper[index]:
            outcomes.append("not_confirmed")
        else:
            outcomes.append("unresolved")
    finite_arrays = (
        energy, energy_lower, energy_upper, diameter_squared,
        diameter_lower, diameter_upper, gap_nominal, gap_lower,
        *work, *work_lower, *work_upper, total_work, total_work_lower,
        total_work_upper, lower, upper, charge, lower_energy, upper_energy,
        margin_lower, minimum_margin, predicted_lower, predicted_upper,
        restoring_coefficient,
    )
    if (
        any(not np.isfinite(value).all() for value in finite_arrays)
        or np.any(energy_lower > energy_upper)
        or np.any(lower_energy > upper_energy)
        or np.any(diameter_lower > diameter_upper)
        or any(outcome not in SCIENTIFIC_OUTCOMES for outcome in outcomes)
    ):
        raise ValueError("directed source certificate is numerically invalid")
    return {
        "schema_version": CERTIFICATE_SCHEMA,
        "successor_values_used": False,
        "cross_map_pairs_formed": False,
        "map_count": int(source.shape[0]),
        "rows_per_map": int(source.shape[1]),
        "feature_dimension": int(source.shape[2]),
        "pairs_per_map": int(len(pairs)),
        "pairs": pairs,
        "source_energy": energy,
        "source_energy_lower": energy_lower,
        "source_energy_upper": energy_upper,
        "source_diameter_squared": diameter_squared,
        "source_diameter_squared_lower": diameter_lower,
        "source_diameter_squared_upper": diameter_upper,
        "source_gap_nominal": gap_nominal,
        "source_gap": gap_lower,
        "source_gap_lower": gap_lower,
        "signed_gate_work": work[0],
        "signed_gate_work_lower": work_lower[0],
        "signed_gate_work_upper": work_upper[0],
        "signed_shape_work": work[1],
        "signed_shape_work_lower": work_lower[1],
        "signed_shape_work_upper": work_upper[1],
        "signed_input_work": work[2],
        "signed_input_work_lower": work_lower[2],
        "signed_input_work_upper": work_upper[2],
        "signed_total_work": total_work,
        "signed_total_work_lower": total_work_lower,
        "signed_total_work_upper": total_work_upper,
        "work_additivity_residual": np.asarray(
            interval["work_additivity_residual"]
        ),
        "remainder_lower": lower,
        "remainder_upper": upper,
        "native_arithmetic_charge": charge,
        "successor_energy_lower": lower_energy,
        "successor_energy_upper": upper_energy,
        "source_restoring_margin": margin_lower,
        "source_restoring_margin_lower": margin_lower,
        "minimum_source_restoring_margin": minimum_margin,
        "predicted_diameter_squared_lower": predicted_lower,
        "predicted_diameter_squared_upper": predicted_upper,
        "source_upper_multiplier": upper_multiplier,
        "source_restoring_coefficient": restoring_coefficient,
        "scientific_outcome": tuple(outcomes),
        "technical_valid": True,
    }


def cover_native_successor(
    certificate: dict[str, Any],
    successor_rows: Any,
) -> dict[str, Any]:
    """Open a native successor only after the source certificate is frozen."""

    if (
        certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("successor_values_used") is not False
        or certificate.get("technical_valid") is not True
    ):
        raise ValueError("coverage requires a valid frozen source certificate")
    successor = _array(successor_rows, "successor_rows", 3)
    expected = (
        int(certificate["map_count"]),
        int(certificate["rows_per_map"]),
        int(certificate["feature_dimension"]),
    )
    if successor.shape != expected:
        raise ValueError("native successor rows disagree with the certificate")
    native = pair_energy_intervals(successor)
    pairs = np.asarray(native["pairs"])
    energy = np.asarray(native["source_energy"])
    energy_lower = np.asarray(native["source_energy_lower"])
    energy_upper = np.asarray(native["source_energy_upper"])
    if not np.array_equal(pairs, certificate["pairs"]):
        raise ValueError("native successor pair registry changed")
    lower = _array(
        certificate["successor_energy_lower"],
        "successor_energy_lower",
        2,
    )
    upper = _array(
        certificate["successor_energy_upper"],
        "successor_energy_upper",
        2,
    )
    covered = (lower <= energy_lower) & (energy_upper <= upper)
    technical_valid = bool(np.all(covered))
    diameter = np.max(energy, axis=1)
    diameter_lower = np.max(energy_lower, axis=1)
    diameter_upper = np.max(energy_upper, axis=1)
    observed_outcomes = []
    source_lower = np.asarray(certificate["source_diameter_squared_lower"])
    source_upper = np.asarray(certificate["source_diameter_squared_upper"])
    for before_lower, before_upper, after_lower, after_upper in zip(
        source_lower,
        source_upper,
        diameter_lower,
        diameter_upper,
        strict=True,
    ):
        if after_upper < before_lower:
            observed_outcomes.append("contracted")
        elif after_lower > before_upper:
            observed_outcomes.append("expanded")
        else:
            observed_outcomes.append("unresolved")
    return {
        "schema_version": COVERAGE_SCHEMA,
        "certificate_schema_version": certificate["schema_version"],
        "certificate_was_source_only": True,
        "technical_valid": technical_valid,
        "coverage_fraction": float(np.mean(covered)),
        "covered_pair_count": int(np.count_nonzero(covered)),
        "pair_count": int(covered.size),
        "covered": covered,
        "native_energy": energy,
        "native_energy_lower": energy_lower,
        "native_energy_upper": energy_upper,
        "native_diameter_squared": diameter,
        "native_diameter_squared_lower": diameter_lower,
        "native_diameter_squared_upper": diameter_upper,
        "observed_outcome": tuple(observed_outcomes),
    }


def source_multiplier_cocycle(
    initial_diameter_squared: Any,
    source_upper_multipliers: Any,
) -> dict[str, np.ndarray]:
    """Compose map-specific source multipliers in chronological order."""

    initial = _array(
        initial_diameter_squared, "initial_diameter_squared", 1
    )
    multiplier = _array(
        source_upper_multipliers, "source_upper_multipliers", 2
    )
    if multiplier.shape[1] != initial.shape[0]:
        raise ValueError("one multiplier column is required per map")
    if np.any(initial < 0.0) or np.any(multiplier < 0.0):
        raise ValueError("diameters and upper multipliers must be nonnegative")
    prefix = np.ones((multiplier.shape[0] + 1, multiplier.shape[1]))
    with np.errstate(over="raise", invalid="raise"):
        for edge in range(multiplier.shape[0]):
            product = prefix[edge] * multiplier[edge]
            prefix[edge + 1] = np.where(
                product == 0.0,
                0.0,
                np.nextafter(product, np.full_like(product, np.inf)),
            )
        raw_bound = prefix * initial[None, :]
        bound = np.where(
            raw_bound == 0.0,
            0.0,
            np.nextafter(raw_bound, np.full_like(raw_bound, np.inf)),
        )
    log_increment = np.full_like(multiplier, -np.inf)
    positive = multiplier > 0.0
    raw_log = np.log(multiplier[positive])
    log_increment[positive] = np.nextafter(
        raw_log, np.full_like(raw_log, np.inf)
    )
    cumulative_log = np.zeros_like(prefix)
    for edge in range(multiplier.shape[0]):
        raw_sum = cumulative_log[edge] + log_increment[edge]
        cumulative_log[edge + 1] = np.where(
            np.isneginf(raw_sum),
            -np.inf,
            np.nextafter(raw_sum, np.full_like(raw_sum, np.inf)),
        )
    return {
        "prefix_product": prefix,
        "diameter_squared_bound": bound,
        "log_multiplier": log_increment,
        "cumulative_log_product": cumulative_log,
    }


def partial_timecourse(
    steps: Any,
    values: Any,
    expected_steps: Any,
    *,
    reason: str | None = None,
) -> dict[str, Any]:
    """Preserve every available time point and state completeness explicitly."""

    observed_steps = np.asarray(steps, dtype=np.int64)
    expected = np.asarray(expected_steps, dtype=np.int64)
    series = _array(values, "values")
    if observed_steps.ndim != 1 or expected.ndim != 1:
        raise ValueError("time-course steps must be one-dimensional")
    if series.shape[0] != len(observed_steps):
        raise ValueError("one value row is required per observed step")
    if len(observed_steps) and (
        np.any(np.diff(observed_steps) <= 0)
        or len(set(observed_steps.tolist())) != len(observed_steps)
    ):
        raise ValueError("observed steps must be strictly increasing")
    complete = bool(np.array_equal(observed_steps, expected))
    if complete and reason is not None:
        raise ValueError("a complete time course cannot carry a stop reason")
    if not complete and not reason:
        raise ValueError("an incomplete time course needs a reason")
    return {
        "schema_version": TIMECOURSE_SCHEMA,
        "steps": observed_steps,
        "values": series,
        "expected_steps": expected,
        "complete": complete,
        "reason": reason,
    }
