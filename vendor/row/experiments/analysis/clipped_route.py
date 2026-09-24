"""Exact route observables for clipped, time-varying factor dynamics.

The route selector consumes realized positive factor values.  It never
substitutes an unclipped affine proposal for the transition that was
actually applied.  Cumulative log contraction is therefore meaningful
under schedules, activity-gated rebuild, and upper or lower clipping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RouteStep:
    before: float
    proposal: float
    after: float
    ratio: float
    log_contraction: float
    lower_clipped: bool
    upper_clipped: bool


def clipped_affine_route_step(
    value: float,
    *,
    eta: float,
    theta: float,
    rebuild: float = 0.0,
    lower: float = 0.0,
    upper: float = 1.0,
) -> RouteStep:
    """Apply the transition and report its exact realized route ratio.

    The proposal is ``(1 - eta*theta)*value + eta*rebuild``.  The
    realized next value is its projection to ``[lower, upper]``.  Log
    contraction requires both endpoint values to be strictly positive;
    zero is an absorbing/crossing endpoint and is handled by a floor
    crossing rather than by an infinite floating-point rate.
    """
    vals = (value, eta, theta, rebuild, lower, upper)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("route-step inputs must be finite")
    if value <= 0.0:
        raise ValueError("route ratio needs a positive starting value")
    if eta < 0.0 or theta < 0.0 or rebuild < 0.0:
        raise ValueError("eta, theta, and rebuild must be nonnegative")
    if not 0.0 <= lower < upper or not lower <= value <= upper:
        raise ValueError("need 0 <= lower < upper and value in the box")

    proposal = (1.0 - eta * theta) * value + eta * rebuild
    after = min(max(proposal, lower), upper)
    if after <= 0.0:
        raise ValueError(
            "the realized endpoint is zero; use a positive route floor"
        )
    ratio = after / value
    return RouteStep(
        before=value,
        proposal=proposal,
        after=after,
        ratio=ratio,
        log_contraction=-math.log(ratio),
        lower_clipped=proposal < lower,
        upper_clipped=proposal > upper,
    )


def realized_route_ratios(values: Iterable[float]) -> tuple[float, ...]:
    """Return consecutive realized ratios ``F[t+1] / F[t]``."""
    path = tuple(float(v) for v in values)
    if len(path) < 2:
        raise ValueError("a route path needs at least two values")
    if not all(math.isfinite(v) and v > 0.0 for v in path):
        raise ValueError("route values must be finite and positive")
    return tuple(b / a for a, b in zip(path[:-1], path[1:]))


def cumulative_log_contraction(values: Iterable[float]) -> tuple[float, ...]:
    """Cumulative time-varying rate with the exact telescoping identity.

    The returned element at horizon ``n`` is

        ``C(n) = sum(t < n, -log(F[t+1]/F[t])) = log(F[0]/F[n])``.

    No constant-rate or nonsaturation assumption is used.
    """
    path = tuple(float(v) for v in values)
    ratios = realized_route_ratios(path)
    total = 0.0
    out = [0.0]
    for ratio in ratios:
        total -= math.log(ratio)
        out.append(total)
    for n, got in enumerate(out):
        expected = math.log(path[0] / path[n])
        if not math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ArithmeticError("route-rate telescoping identity failed")
    return tuple(out)


def first_route_floor_crossing(
    values: Iterable[float], floor: float
) -> int | None:
    """Return the first observed index at or below a positive floor."""
    path = tuple(float(v) for v in values)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("route floor must be finite and positive")
    if not path or not all(math.isfinite(v) and v > 0.0 for v in path):
        raise ValueError("route values must be finite and positive")
    return next((i for i, value in enumerate(path) if value <= floor), None)


def cumulative_route_dominance(
    left: Iterable[float], right: Iterable[float]
) -> tuple[float, ...]:
    """Return ``C_left(n) - C_right(n)`` on a shared horizon.

    A positive value says that the left route has accumulated more
    realized contraction by that horizon.  Pointwise rate ordering is
    neither required nor inferred.
    """
    left_c = cumulative_log_contraction(left)
    right_c = cumulative_log_contraction(right)
    if len(left_c) != len(right_c):
        raise ValueError("route paths must share a horizon")
    return tuple(a - b for a, b in zip(left_c, right_c))
