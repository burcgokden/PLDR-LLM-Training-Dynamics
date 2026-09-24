"""Decision kernels for exact finite-increment observable balance.

The live producer writes three vectors for a signed perturbation:

``homogeneous``
    The transported tangent observed at the endpoint.
``state_correction``
    The endpoint observation derivative applied to the difference between
    the finite state secant and the transported tangent.
``observation_correction``
    The finite observation secant minus the derivative applied to the finite
    state secant.

Their sum is the measured finite observation response.  This module keeps the
identity, numerical qualification, and campaign decision rules independent of
PyTorch and of the live checkpoint boundary.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import numpy as np

from confirm.observable_balance_specs import (
    DECISION_POLICY,
    DIRECTION_CLASSES,
    HORIZON_GRID,
    SOURCE_STEPS,
)


def _finite_array(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite array")
    return result


def vector_norm(value: Any) -> float:
    """Return one Euclidean norm over the fixed registered observation."""

    array = _finite_array(value, "observable vector")
    return float(np.linalg.norm(array.reshape(-1)))


def relative_residual(actual: Any, reconstructed: Any, *, floor: float) -> float:
    """Symmetric relative residual with a declared absolute floor."""

    left = _finite_array(actual, "actual response")
    right = _finite_array(reconstructed, "reconstructed response")
    if left.shape != right.shape:
        raise ValueError("response and reconstruction shapes disagree")
    denominator = max(vector_norm(left), vector_norm(right), float(floor))
    return vector_norm(left - right) / denominator


def balance_metrics(
    homogeneous: Any,
    state_correction: Any,
    observation_correction: Any,
    observed: Any,
    *,
    effect_floor: float | None = None,
) -> dict[str, Any]:
    """Evaluate the exact three-term balance and its cancellation geometry."""

    terms = [
        _finite_array(homogeneous, "homogeneous term"),
        _finite_array(state_correction, "state correction"),
        _finite_array(observation_correction, "observation correction"),
    ]
    response = _finite_array(observed, "observed response")
    if any(term.shape != response.shape for term in terms):
        raise ValueError("observable-balance terms have different shapes")
    floor = float(
        DECISION_POLICY["absolute_effect_floor"]
        if effect_floor is None
        else effect_floor
    )
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("effect floor must be finite and positive")

    reconstructed = terms[0] + terms[1] + terms[2]
    norms = np.asarray([vector_norm(term) for term in terms], dtype=np.float64)
    response_norm = vector_norm(response)
    gram = np.empty((3, 3), dtype=np.float64)
    flattened = [term.reshape(-1) for term in terms]
    for row in range(3):
        for column in range(3):
            gram[row, column] = float(flattened[row] @ flattened[column])
    correction = terms[1] + terms[2]
    homogeneous_correction_inner = float(
        flattened[0] @ correction.reshape(-1)
    )
    reconstruction_relative = relative_residual(
        response, reconstructed, floor=floor
    )
    cancellation_index = float(np.sum(norms) / max(response_norm, floor))
    response_to_homogeneous = response_norm / max(float(norms[0]), floor)
    cancellation_dominant = bool(
        response_norm > floor
        and cancellation_index
        >= float(DECISION_POLICY["minimum_cancellation_index"])
        and response_to_homogeneous
        <= float(DECISION_POLICY["maximum_response_to_homogeneous_ratio"])
        and homogeneous_correction_inner < 0.0
    )
    return {
        "term_norms": norms,
        "response_norm": response_norm,
        "reconstructed_response": reconstructed,
        "reconstruction_relative_residual": reconstruction_relative,
        "cross_gram": gram,
        "homogeneous_correction_inner_product": homogeneous_correction_inner,
        "cancellation_index": cancellation_index,
        "response_to_homogeneous_ratio": response_to_homogeneous,
        "cancellation_dominant": cancellation_dominant,
        "dominant_correction": (
            "state"
            if norms[1] > norms[2]
            else "observation" if norms[2] > norms[1] else "tie"
        ),
    }


def cell_evaluable(
    *,
    effect_norm: float,
    reconstruction_relative_residual: float,
    replay_relative_residual: float,
    finite: bool = True,
) -> bool:
    """Apply the predeclared per-amplitude numerical gates."""

    values = (
        float(effect_norm),
        float(reconstruction_relative_residual),
        float(replay_relative_residual),
    )
    return bool(
        finite
        and all(math.isfinite(value) for value in values)
        and values[0] > float(DECISION_POLICY["absolute_effect_floor"])
        and values[1]
        <= float(DECISION_POLICY["maximum_reconstruction_relative_residual"])
        and values[2]
        <= float(DECISION_POLICY["maximum_adamw_replay_relative_residual"])
    )


def halving_stable(larger_residual: float, smaller_residual: float) -> bool:
    """Require improvement under halving, up to the equality floor."""

    larger = float(larger_residual)
    smaller = float(smaller_residual)
    if not math.isfinite(larger) or not math.isfinite(smaller):
        return False
    equality = float(DECISION_POLICY["reconstruction_equality_floor"])
    factor = float(DECISION_POLICY["minimum_halving_improvement_factor"])
    return bool(smaller <= equality or smaller * factor <= larger)


def select_layer_horizon(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Select the longest horizon valid for every construction source/direction."""

    materialized = list(rows)
    expected_pairs = {
        (int(source), str(direction))
        for source in SOURCE_STEPS
        for direction in DIRECTION_CLASSES
    }
    by_key: dict[tuple[int, str, int], dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in materialized:
        key = (
            int(row["source_step"]),
            str(row["direction"]),
            int(row["horizon"]),
        )
        amplitude_index = int(row["amplitude_index"])
        if amplitude_index not in (0, 1) or amplitude_index in by_key[key]:
            raise ValueError("construction amplitude rows are duplicated or invalid")
        by_key[key][amplitude_index] = row

    qualification: dict[int, bool] = {}
    for horizon in HORIZON_GRID:
        valid = True
        for source, direction in expected_pairs:
            pair = by_key.get((source, direction, int(horizon)), {})
            if set(pair) != {0, 1}:
                valid = False
                break
            larger, smaller = pair[0], pair[1]
            if not bool(larger["evaluable"]) or not bool(smaller["evaluable"]):
                valid = False
                break
            if not halving_stable(
                float(larger["reconstruction_relative_residual"]),
                float(smaller["reconstruction_relative_residual"]),
            ):
                valid = False
                break
        qualification[int(horizon)] = valid
    selected = max(
        (horizon for horizon, valid in qualification.items() if valid),
        default=min(HORIZON_GRID),
    )
    return {
        "selected_horizon": int(selected),
        "jointly_evaluable": bool(qualification[int(selected)]),
        "fallback_used": not any(qualification.values()),
        "horizon_qualification": {
            str(horizon): bool(qualification[int(horizon)])
            for horizon in HORIZON_GRID
        },
    }


def campaign_classification(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Classify held-out cancellation support separately for each trajectory."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["trajectory_label"])].append(row)
    if set(grouped) != {"B", "C"}:
        raise ValueError("held-out classification requires trajectories B and C")
    required_fraction = float(DECISION_POLICY["campaign_support_fraction"])
    minimum_fraction = float(DECISION_POLICY["minimum_evaluable_fraction"])
    summaries = {}
    trajectory_support = []
    sufficient = True
    for label in ("B", "C"):
        local = grouped[label]
        evaluable = [row for row in local if bool(row["evaluable"])]
        dominant = [row for row in evaluable if bool(row["cancellation_dominant"])]
        evaluable_fraction = len(evaluable) / max(len(local), 1)
        support_fraction = len(dominant) / max(len(evaluable), 1)
        enough = evaluable_fraction >= minimum_fraction
        support = enough and support_fraction >= required_fraction
        sufficient = sufficient and enough
        trajectory_support.append(support)
        summaries[label] = {
            "total": len(local),
            "evaluable": len(evaluable),
            "cancellation_dominant": len(dominant),
            "evaluable_fraction": evaluable_fraction,
            "support_fraction": support_fraction,
            "support": support,
        }
    if not sufficient:
        outcome = "insufficient"
    elif all(trajectory_support):
        outcome = "supported"
    else:
        outcome = "refuted"
    return {"outcome": outcome, "trajectories": summaries}
