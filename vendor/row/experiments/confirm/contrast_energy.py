"""Source-only direct contrast-energy certificates and diameter cocycles."""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np


SCHEMA_VERSION = "pldr-contrast-energy-kernel-v1"


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return np.ascontiguousarray(array)


def _nonnegative(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = _array(value, name, ndim)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return array


def _outward_lower(value: np.ndarray) -> np.ndarray:
    return np.nextafter(value, -np.inf)


def _outward_upper(value: np.ndarray) -> np.ndarray:
    return np.nextafter(value, np.inf)


def cell_weights(knots: Any) -> np.ndarray:
    """Return exact integral weights for ``(1-s)`` on consecutive cells."""

    points = _array(knots, "knots", 1)
    if (
        len(points) < 2 or points[0] != 0.0 or points[-1] != 1.0
        or np.any(np.diff(points) <= 0.0)
    ):
        raise ValueError("knots must strictly partition the unit segment")
    left = points[:-1]
    right = points[1:]
    weights = (right - left) * (1.0 - 0.5 * (left + right))
    if np.any(weights <= 0.0):
        raise AssertionError("unit-segment cell weights must be positive")
    return weights


def cell_second_bounds(
    endpoint_lower: Any,
    endpoint_upper: Any,
    third_modulus: Any,
    knots: Any,
) -> dict[str, np.ndarray]:
    """Turn endpoint enclosures and a positive cell modulus into cell bounds.

    If ``|e'''| <= T`` on a cell, every point is at most half a cell width
    from one endpoint.  The construction therefore uses both endpoints and
    charges ``T * width / 2``.  The final operations are rounded outward.
    """

    lower = _array(endpoint_lower, "endpoint_lower", 2)
    upper = _array(endpoint_upper, "endpoint_upper", 2)
    points = _array(knots, "knots", 1)
    if lower.shape != upper.shape or lower.shape[1] != len(points):
        raise ValueError("endpoint arrays and knots disagree")
    if np.any(lower > upper):
        raise ValueError("an endpoint lower bound exceeds its upper bound")
    modulus = _nonnegative(third_modulus, "third_modulus")
    cells = len(points) - 1
    try:
        modulus = np.broadcast_to(modulus, (lower.shape[0], cells)).astype(
            np.float64, copy=False)
    except ValueError as error:
        raise ValueError("third_modulus does not broadcast over pairs and cells") from error
    if np.any(modulus <= 0.0):
        raise ValueError("every certified third-derivative modulus must be positive")
    widths = np.diff(points)[None, :]
    charge = _outward_upper(0.5 * widths * modulus)
    cell_lower = _outward_lower(
        np.minimum(lower[:, :-1], lower[:, 1:]) - charge)
    cell_upper = _outward_upper(
        np.maximum(upper[:, :-1], upper[:, 1:]) + charge)
    if np.any(cell_lower > cell_upper):
        raise AssertionError("constructed cell bounds are inverted")
    return {
        "cell_lower": cell_lower,
        "cell_upper": cell_upper,
        "modulus_charge": charge,
        "third_modulus": np.ascontiguousarray(modulus),
        "weights": cell_weights(points),
    }


def integrate_pair_energy(
    source_energy: Any,
    source_slope: Any,
    cell_lower: Any,
    cell_upper: Any,
    knots: Any,
    *,
    source_charge: Any = 0.0,
    arithmetic_charge: Any = 0.0,
) -> dict[str, np.ndarray]:
    """Integrate signed second-derivative bounds for every registered pair."""

    energy = _nonnegative(source_energy, "source_energy", 1)
    slope = _array(source_slope, "source_slope", 1)
    lower = _array(cell_lower, "cell_lower", 2)
    upper = _array(cell_upper, "cell_upper", 2)
    weights = cell_weights(knots)
    pair_count = len(energy)
    if (
        slope.shape != energy.shape or lower.shape != upper.shape
        or lower.shape != (pair_count, len(weights))
        or np.any(lower > upper)
    ):
        raise ValueError("pair energies, slopes, and cell bounds disagree")
    source_allowance = _nonnegative(source_charge, "source_charge")
    arithmetic = _nonnegative(arithmetic_charge, "arithmetic_charge")
    try:
        source_allowance = np.broadcast_to(source_allowance, (pair_count,))
        arithmetic = np.broadcast_to(arithmetic, (pair_count,))
    except ValueError as error:
        raise ValueError("energy charges do not broadcast over pairs") from error

    weighted_lower = np.einsum("c,pc->p", weights, lower, optimize=True)
    weighted_upper = np.einsum("c,pc->p", weights, upper, optimize=True)
    successor_lower = _outward_lower(
        energy + slope + weighted_lower - arithmetic)
    successor_upper = _outward_upper(
        energy + slope + weighted_upper + arithmetic)
    source_lower = _outward_lower(energy - source_allowance)
    source_upper = _outward_upper(energy + source_allowance)
    if np.any(successor_lower > successor_upper):
        raise AssertionError("integrated pair-energy bounds are inverted")
    return {
        "source_energy": energy,
        "source_lower": source_lower,
        "source_upper": source_upper,
        "source_slope": slope,
        "weighted_second_lower": weighted_lower,
        "weighted_second_upper": weighted_upper,
        "successor_lower": successor_lower,
        "successor_upper": successor_upper,
        "source_charge": np.ascontiguousarray(source_allowance),
        "arithmetic_charge": np.ascontiguousarray(arithmetic),
        "weights": weights,
    }


def unordered_pair_chunks(vertex_count: int, chunk_size: int) -> Iterator[np.ndarray]:
    """Yield all unordered pairs once in deterministic lexicographic order."""

    count = int(vertex_count)
    size = int(chunk_size)
    if count < 2 or size < 1:
        raise ValueError("a pair registry needs at least two vertices and a positive chunk")
    left: list[int] = []
    right: list[int] = []
    for first in range(count - 1):
        for second in range(first + 1, count):
            left.append(first)
            right.append(second)
            if len(left) == size:
                yield np.column_stack((left, right)).astype(np.int64, copy=False)
                left.clear()
                right.clear()
    if left:
        yield np.column_stack((left, right)).astype(np.int64, copy=False)


def pair_source_terms(rows: Any, row_direction: Any) -> dict[str, np.ndarray]:
    """Form direct pair energies and slopes from physical rows and their JVP."""

    source = _array(rows, "rows", 2)
    direction = _array(row_direction, "row_direction", 2)
    if source.shape != direction.shape or source.shape[0] < 2:
        raise ValueError("row source and direction registries disagree")
    pairs = np.concatenate(list(unordered_pair_chunks(source.shape[0], 32_768)))
    contrast = source[pairs[:, 0]] - source[pairs[:, 1]]
    tangent = direction[pairs[:, 0]] - direction[pairs[:, 1]]
    energy = np.einsum("pd,pd->p", contrast, contrast, optimize=True)
    slope = 2.0 * np.einsum("pd,pd->p", contrast, tangent, optimize=True)
    return {"pairs": pairs, "source_energy": energy, "source_slope": slope}


def _maximum(values: np.ndarray, pairs: np.ndarray) -> tuple[float, list[int]]:
    maximum = float(np.max(values))
    candidates = np.flatnonzero(values == maximum)
    if not len(candidates):
        raise AssertionError("a finite maximum has no maximizing pair")
    index = min(
        map(int, candidates),
        key=lambda item: (int(pairs[item, 0]), int(pairs[item, 1])),
    )
    return maximum, [int(pairs[index, 0]), int(pairs[index, 1])]


def finite_registry_prediction(
    pairs: Any,
    enclosure: dict[str, Any],
    *,
    successor_energy: Any | None = None,
) -> dict[str, Any]:
    """Take the exact finite maximum and decide strict source-only signs."""

    registry = np.asarray(pairs, dtype=np.int64)
    if (
        registry.ndim != 2 or registry.shape[1] != 2
        or np.any(registry[:, 0] < 0)
        or np.any(registry[:, 0] >= registry[:, 1])
        or len(np.unique(registry, axis=0)) != len(registry)
    ):
        raise ValueError("pairs must be unique ordered unordered-pair indices")
    fields = {}
    for name in ("source_lower", "source_upper", "successor_lower", "successor_upper"):
        fields[name] = _array(enclosure[name], name, 1)
        if fields[name].shape != (len(registry),):
            raise ValueError("finite-registry enclosure length changed")
    if (
        np.any(fields["source_lower"] > fields["source_upper"])
        or np.any(fields["successor_lower"] > fields["successor_upper"])
    ):
        raise ValueError("finite-registry bounds are inverted")

    maxima = {}
    for name, values in fields.items():
        value, pair = _maximum(values, registry)
        maxima[name] = {"value": value, "pair": pair}
    contraction_margin = (
        maxima["source_lower"]["value"]
        - maxima["successor_upper"]["value"]
    )
    expansion_margin = (
        maxima["successor_lower"]["value"]
        - maxima["source_upper"]["value"]
    )
    if contraction_margin > 0.0 and expansion_margin > 0.0:
        raise ValueError("inconsistent enclosures certify opposite strict signs")
    if contraction_margin > 0.0:
        decision = "contraction"
        strict_margin = contraction_margin
    elif expansion_margin > 0.0:
        decision = "reexpansion"
        strict_margin = expansion_margin
    else:
        decision = "unresolved"
        strict_margin = max(contraction_margin, expansion_margin)

    denominator_lower = maxima["source_lower"]["value"]
    denominator_upper = maxima["source_upper"]["value"]
    lower_multiplier = (
        max(0.0, maxima["successor_lower"]["value"]) / denominator_upper
        if denominator_upper > 0.0 else None
    )
    upper_multiplier = (
        max(0.0, maxima["successor_upper"]["value"]) / denominator_lower
        if denominator_lower > 0.0 else None
    )
    coverage = None
    if successor_energy is not None:
        realized = _nonnegative(successor_energy, "successor_energy", 1)
        if realized.shape != (len(registry),):
            raise ValueError("successor energy length changed")
        covered = (
            fields["successor_lower"] <= realized
        ) & (realized <= fields["successor_upper"])
        realized_maximum, realized_pair = _maximum(realized, registry)
        realized_decision = (
            "contraction" if realized_maximum < maxima["source_lower"]["value"]
            else "reexpansion" if realized_maximum > maxima["source_upper"]["value"]
            else "unresolved"
        )
        coverage = {
            "all_pairs_enclosed": bool(np.all(covered)),
            "enclosed_pair_count": int(np.count_nonzero(covered)),
            "pair_count": int(len(covered)),
            "realized_diameter_squared": realized_maximum,
            "realized_maximizing_pair": realized_pair,
            "realized_decision": realized_decision,
            "strict_sign_correct": (
                decision == "unresolved" or decision == realized_decision
            ),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "pair_count": int(len(registry)),
        "maxima": maxima,
        "decision": decision,
        "decision_uses_successor": False,
        "strict_margin_squared": float(strict_margin),
        "contraction_margin_squared": float(contraction_margin),
        "reexpansion_margin_squared": float(expansion_margin),
        "lower_multiplier": lower_multiplier,
        "upper_multiplier": upper_multiplier,
        "ratio_denominator_valid": denominator_lower > 0.0,
        "coverage": coverage,
    }


def aggregate_cocycle(edges: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate chronologically ordered multiplier intervals."""

    if not edges:
        raise ValueError("a diameter cocycle needs at least one edge")
    lowers: list[float] = []
    uppers: list[float] = []
    for index, edge in enumerate(edges):
        lower = edge.get("lower_multiplier")
        upper = edge.get("upper_multiplier")
        if (
            isinstance(lower, bool) or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) < 0.0 or float(lower) > float(upper)
        ):
            raise ValueError(f"edge {index} has an invalid multiplier interval")
        lowers.append(float(lower))
        uppers.append(float(upper))
    lower_products = []
    upper_products = []
    cumulative_logs = []
    lower_product = 1.0
    upper_product = 1.0
    cumulative_log = 0.0
    for lower, upper in zip(lowers, uppers, strict=True):
        lower_product *= lower
        upper_product *= upper
        cumulative_log = (
            -math.inf if upper == 0.0 or cumulative_log == -math.inf
            else cumulative_log + math.log(upper)
        )
        lower_products.append(lower_product)
        upper_products.append(upper_product)
        cumulative_logs.append(cumulative_log)
    return {
        "schema_version": "pldr-diameter-cocycle-analysis-v1",
        "edge_count": len(edges),
        "lower_multipliers": lowers,
        "upper_multipliers": uppers,
        "lower_products": lower_products,
        "upper_products": upper_products,
        "cumulative_log_upper": cumulative_logs,
        "temporary_reexpansion_count": sum(lower > 1.0 for lower in lowers),
        "certified_contraction_count": sum(upper < 1.0 for upper in uppers),
    }
