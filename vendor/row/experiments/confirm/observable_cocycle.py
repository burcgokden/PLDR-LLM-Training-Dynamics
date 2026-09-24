"""Pure numerical decisions for the observable full-state cocycle study."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np

from confirm.observable_cocycle_specs import (
    DECISION_POLICY,
    DIRECTION_CLASSES,
    HORIZON_GRID,
)


def canonical_json(value: Any) -> bytes:
    """Encode JSON with the campaign's unique deterministic representation."""

    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    """Return the digest of the compact canonical scientific object."""

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def centered_rows(value: np.ndarray) -> np.ndarray:
    """Project a (..., row, coordinate) tensor to the row quotient."""

    array = np.asarray(value)
    if array.ndim < 2 or not np.isfinite(array).all():
        raise ValueError("row observation must be finite and at least rank two")
    return array - np.mean(array, axis=-2, keepdims=True)


def aggregate_norm(value: np.ndarray) -> float:
    """Use one Frobenius norm over the fixed within-unit registry."""

    array = np.asarray(value, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("observable response contains nonfinite values")
    return float(np.linalg.norm(array.reshape(-1)))


def relative_response_residual(
    predicted: np.ndarray,
    observed: np.ndarray,
    *,
    effect_floor: float,
) -> dict[str, float]:
    """Compare two centered response vectors with an explicit zero policy."""

    prediction = np.asarray(predicted, dtype=np.float64)
    measurement = np.asarray(observed, dtype=np.float64)
    floor = float(effect_floor)
    if (
        prediction.shape != measurement.shape
        or not np.isfinite(prediction).all()
        or not np.isfinite(measurement).all()
        or not math.isfinite(floor)
        or floor <= 0.0
    ):
        raise ValueError("response residual inputs are invalid")
    absolute = aggregate_norm(prediction - measurement)
    predicted_norm = aggregate_norm(prediction)
    observed_norm = aggregate_norm(measurement)
    denominator = max(predicted_norm, observed_norm, floor)
    return {
        "absolute_residual": absolute,
        "relative_residual": absolute / denominator,
        "predicted_norm": predicted_norm,
        "observed_norm": observed_norm,
        "effect_resolved": max(predicted_norm, observed_norm) > floor,
    }


def directional_gain(
    source_action: np.ndarray,
    endpoint_action: np.ndarray,
    *,
    effect_floor: float,
) -> dict[str, float | bool]:
    """Measure visible gain without promoting it to an operator norm."""

    floor = float(effect_floor)
    source = aggregate_norm(source_action)
    endpoint = aggregate_norm(endpoint_action)
    resolved = source > floor
    return {
        "source_action_norm": source,
        "endpoint_action_norm": endpoint,
        "visible_directional_gain": endpoint / source if resolved else math.inf,
        "source_action_resolved": resolved,
        "directionally_attenuating": resolved and endpoint < source,
    }


def calibration_cell(
    *,
    larger_relative_residual: float,
    smaller_relative_residual: float,
    smaller_response_norm: float,
    placebo_norm: float,
    effect_floor: float | None = None,
    maximum_residual: float | None = None,
    minimum_improvement: float | None = None,
    maximum_placebo_fraction: float | None = None,
) -> dict[str, float | bool]:
    """Apply the fixed scale, convergence, and placebo gates to one cell."""

    floor = float(
        DECISION_POLICY["absolute_effect_floor"]
        if effect_floor is None
        else effect_floor
    )
    residual_limit = float(
        DECISION_POLICY["maximum_centered_response_relative_residual"]
        if maximum_residual is None
        else maximum_residual
    )
    improvement = float(
        DECISION_POLICY["minimum_halving_improvement_factor"]
        if minimum_improvement is None
        else minimum_improvement
    )
    placebo_limit = float(
        DECISION_POLICY["maximum_placebo_fraction"]
        if maximum_placebo_fraction is None
        else maximum_placebo_fraction
    )
    values = (
        larger_relative_residual,
        smaller_relative_residual,
        smaller_response_norm,
        placebo_norm,
        floor,
        residual_limit,
        improvement,
        placebo_limit,
    )
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise ValueError("calibration statistics must be finite and nonnegative")
    resolved = smaller_response_norm > floor
    placebo_fraction = (
        placebo_norm / smaller_response_norm if resolved else math.inf
    )
    residual_pass = smaller_relative_residual <= residual_limit
    improvement_pass = (
        larger_relative_residual
        >= improvement * smaller_relative_residual
        or (larger_relative_residual == smaller_relative_residual == 0.0)
    )
    placebo_pass = resolved and placebo_fraction <= placebo_limit
    return {
        "effect_resolved": resolved,
        "residual_pass": residual_pass,
        "halving_improves": improvement_pass,
        "placebo_pass": placebo_pass,
        "placebo_fraction": placebo_fraction,
        "qualified": resolved and residual_pass and improvement_pass and placebo_pass,
    }


def select_layer_horizon(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select the longest horizon qualified at every source and direction."""

    records = list(rows)
    expected = {
        (step, direction, horizon)
        for step in (4_096, 8_192, 16_384)
        for direction in DIRECTION_CLASSES
        for horizon in HORIZON_GRID
    }
    actual = {
        (int(row["source_step"]), str(row["direction"]), int(row["horizon"]))
        for row in records
    }
    if actual != expected or len(records) != len(expected):
        raise ValueError("construction calibration grid is incomplete or duplicated")
    by_horizon = {
        horizon: all(
            bool(row["qualified"])
            for row in records
            if int(row["horizon"]) == horizon
        )
        for horizon in HORIZON_GRID
    }
    qualified = [horizon for horizon in HORIZON_GRID if by_horizon[horizon]]
    selected = max(qualified) if qualified else min(HORIZON_GRID)
    return {
        "selected_horizon": selected,
        "jointly_qualified": bool(qualified),
        "qualified_horizons": qualified,
        "qualification_by_horizon": {
            str(horizon): by_horizon[horizon] for horizon in HORIZON_GRID
        },
        "fallback_used": not qualified,
    }
