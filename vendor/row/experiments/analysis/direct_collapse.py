"""Numerical helpers for the direct observable collapse certificate."""

import math

import numpy as np


def _finite_scalar(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite scalar")
    return float(value)


def positive_part(value):
    """Return the one-sided part of a signed gate coefficient."""
    return max(_finite_scalar(value, "value"), 0.0)


def gate_sign_branch(interval):
    """Classify a simultaneous interval without changing its orientation."""
    if not isinstance(interval, (list, tuple)) or len(interval) != 2:
        raise ValueError("gate interval must have two endpoints")
    lower = _finite_scalar(interval[0], "lower gate endpoint")
    upper = _finite_scalar(interval[1], "upper gate endpoint")
    if lower > upper:
        raise ValueError("gate interval endpoints are reversed")
    if upper <= 0.0:
        return "NONPOSITIVE"
    if lower > 0.0:
        return "POSITIVE"
    return "SIGN_UNRESOLVED"


def signed_gate_step(jet, contraction, eta, delta, gate, ceiling=1.0):
    """Apply a clipped signed-gate step and its adverse upper envelope."""
    jet = _finite_scalar(jet, "jet")
    contraction = _finite_scalar(contraction, "contraction")
    eta = _finite_scalar(eta, "eta")
    delta = _finite_scalar(delta, "delta")
    gate = _finite_scalar(gate, "gate")
    ceiling = _finite_scalar(ceiling, "ceiling")
    if jet < 0.0 or contraction < 0.0 or eta < 0.0 or gate < 0.0:
        raise ValueError("jet, contraction, eta, and gate must be nonnegative")
    if ceiling < 0.0:
        raise ValueError("ceiling must be nonnegative")
    raw = contraction * jet + eta * delta * gate
    realized = min(max(raw, 0.0), ceiling)
    upper = contraction * jet + eta * positive_part(delta) * gate
    if realized > upper + 64.0 * np.finfo(float).eps * max(1.0, abs(upper)):
        raise ArithmeticError("signed-gate upper envelope was violated")
    return {
        "raw": raw,
        "realized": realized,
        "upper": upper,
        "positive_gate_coefficient": positive_part(delta),
    }


def trajectory_guard_tube(
    score_initial,
    threshold_initial,
    score_increment_upper,
    threshold_increment_lower,
):
    """Propagate the direct headroom lower bound along one registered tube."""
    score_initial = np.asarray(score_initial, dtype=float)
    threshold_initial = np.asarray(threshold_initial, dtype=float)
    score_increment_upper = np.asarray(score_increment_upper, dtype=float)
    threshold_increment_lower = np.asarray(
        threshold_increment_lower, dtype=float
    )
    if score_initial.shape != threshold_initial.shape:
        raise ValueError("initial score and threshold shapes differ")
    if score_increment_upper.shape != threshold_increment_lower.shape:
        raise ValueError("increment-bound shapes differ")
    if score_increment_upper.ndim < 1:
        raise ValueError("increment arrays need a time axis")
    if score_increment_upper.shape[1:] != score_initial.shape:
        raise ValueError("increment and initial layer shapes differ")
    arrays = (
        score_initial,
        threshold_initial,
        score_increment_upper,
        threshold_increment_lower,
    )
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("guard tube inputs must be finite")
    initial_margin = threshold_initial - score_initial
    surplus = threshold_increment_lower - score_increment_upper
    margins = np.concatenate(
        [
            initial_margin[None, ...],
            initial_margin[None, ...] + np.cumsum(surplus, axis=0),
        ],
        axis=0,
    )
    return {
        "margins": margins,
        "minimum_margin": float(np.min(margins)),
        "closed": bool(np.all(margins >= 0.0)),
        "stepwise_dominance": bool(np.all(surplus >= 0.0)),
    }


def geometric_tail_bound(jet_initial, contraction, steps):
    """Return the direct geometric tail envelope at integer horizons."""
    jet_initial = _finite_scalar(jet_initial, "jet_initial")
    contraction = _finite_scalar(contraction, "contraction")
    steps = np.asarray(steps)
    if jet_initial < 0.0 or not 0.0 <= contraction < 1.0:
        raise ValueError("need jet_initial >= 0 and 0 <= contraction < 1")
    if not np.issubdtype(steps.dtype, np.integer) or np.any(steps < 0):
        raise ValueError("steps must be nonnegative integers")
    return jet_initial * np.power(contraction, steps)


def common_entry_horizon(jet_max, criterion, contraction):
    """Compute the common finite horizon of the uniform direct envelope."""
    jet_max = _finite_scalar(jet_max, "jet_max")
    criterion = _finite_scalar(criterion, "criterion")
    contraction = _finite_scalar(contraction, "contraction")
    if jet_max < 0.0 or criterion <= 0.0:
        raise ValueError("need jet_max >= 0 and criterion > 0")
    if not 0.0 <= contraction < 1.0:
        raise ValueError("contraction must lie in [0, 1)")
    if jet_max <= criterion:
        return 0
    if contraction == 0.0:
        return 1
    return int(math.ceil(math.log(jet_max / criterion)
                         / abs(math.log(contraction))))
