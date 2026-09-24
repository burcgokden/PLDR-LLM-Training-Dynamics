"""Finite positive-comparison witnesses for finite and infinite tails."""

from __future__ import annotations

from fractions import Fraction

from validated_interval import as_fraction, rational_object


def _vector(value, *, nonnegative=False, positive=False):
    value = tuple(as_fraction(entry) for entry in value)
    if not value:
        raise ValueError("comparison vector must be nonempty")
    if nonnegative and any(entry < 0 for entry in value):
        raise ValueError("comparison vector must be nonnegative")
    if positive and any(entry <= 0 for entry in value):
        raise ValueError("comparison weights must be positive")
    return value


def _matrix(value, size=None):
    value = tuple(tuple(as_fraction(entry) for entry in row) for row in value)
    if not value or any(len(row) != len(value) for row in value):
        raise ValueError("comparison matrix must be square")
    if size is not None and len(value) != size:
        raise ValueError("comparison matrix and vector dimensions disagree")
    if any(entry < 0 for row in value for entry in row):
        raise ValueError("comparison matrix must be entrywise nonnegative")
    return value


def matrix_vector(matrix, vector):
    matrix = _matrix(matrix, len(vector))
    vector = _vector(vector)
    return tuple(sum((entry * coordinate for entry, coordinate in zip(row, vector)),
                     Fraction(0)) for row in matrix)


def weighted_max_norm(vector, weights):
    vector = _vector(vector, nonnegative=True)
    weights = _vector(weights, positive=True)
    if len(vector) != len(weights):
        raise ValueError("weighted norm dimensions disagree")
    return max(entry / weight for entry, weight in zip(vector, weights))


def verify_positive_witness(matrix, weights, kappa):
    weights = _vector(weights, positive=True)
    matrix = _matrix(matrix, len(weights))
    kappa = as_fraction(kappa)
    if not 0 <= kappa < 1:
        raise ValueError("positive comparison gain must lie in [0, 1)")
    product = matrix_vector(matrix, weights)
    if any(entry > kappa * weight
           for entry, weight in zip(product, weights)):
        raise ValueError("P v <= kappa v fails")
    return product


def _horizon(value, name):
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def comparison_bound(initial, matrix, forcing, rho, weights, kappa, steps,
                     *, persistent=None):
    """Unroll ``x[n+1] <= P x[n] + d + b rho^n`` exactly."""

    initial = _vector(initial, nonnegative=True)
    forcing = _vector(forcing, nonnegative=True)
    persistent = (
        tuple(Fraction(0) for _ in initial)
        if persistent is None
        else _vector(persistent, nonnegative=True)
    )
    weights = _vector(weights, positive=True)
    if not len(initial) == len(forcing) == len(persistent) == len(weights):
        raise ValueError("positive comparison dimensions disagree")
    matrix = _matrix(matrix, len(initial))
    rho = as_fraction(rho)
    if not 0 <= rho < 1:
        raise ValueError("schedule envelope rate must lie in [0, 1)")
    verify_positive_witness(matrix, weights, kappa)
    kappa = as_fraction(kappa)
    steps = _horizon(steps, "comparison horizon")
    initial_norm = weighted_max_norm(initial, weights)
    forcing_norm = weighted_max_norm(forcing, weights)
    persistent_norm = weighted_max_norm(persistent, weights)
    bounds = []
    for n in range(steps + 1):
        convolution = sum((
            kappa ** (n - 1 - index) * rho ** index
            for index in range(n)
        ), Fraction(0))
        persistent_sum = sum(
            (kappa ** index for index in range(n)), Fraction(0))
        bounds.append(
            kappa ** n * initial_norm
            + persistent_norm * persistent_sum
            + forcing_norm * convolution
        )
    return bounds


def construct_positive_certificate(initial, matrix, forcing, rho,
                                   weights, kappa, *, persistent=None,
                                   horizon):
    initial = _vector(initial, nonnegative=True)
    forcing = _vector(forcing, nonnegative=True)
    persistent = (
        tuple(Fraction(0) for _ in initial)
        if persistent is None
        else _vector(persistent, nonnegative=True)
    )
    weights = _vector(weights, positive=True)
    if not len(initial) == len(forcing) == len(persistent) == len(weights):
        raise ValueError("positive comparison dimensions disagree")
    matrix = _matrix(matrix, len(initial))
    horizon = _horizon(horizon, "certificate horizon")
    product = verify_positive_witness(matrix, weights, kappa)
    bounds = comparison_bound(
        initial, matrix, forcing, rho, weights, kappa, horizon,
        persistent=persistent,
    )
    persistent_norm = weighted_max_norm(persistent, weights)
    forcing_norm = weighted_max_norm(forcing, weights)
    kappa = as_fraction(kappa)
    return {
        "schema_version": "pldr-positive-tail-comparison-v2",
        "initial": [rational_object(entry) for entry in initial],
        "matrix": [[rational_object(entry) for entry in row] for row in matrix],
        "forcing": [rational_object(entry) for entry in forcing],
        "persistent_forcing": [
            rational_object(entry) for entry in persistent
        ],
        "rho": rational_object(as_fraction(rho)),
        "weights": [rational_object(entry) for entry in weights],
        "kappa": rational_object(kappa),
        "matrix_times_weights": [rational_object(entry) for entry in product],
        "initial_weighted_norm": rational_object(
            weighted_max_norm(initial, weights)),
        "persistent_weighted_norm": rational_object(persistent_norm),
        "vanishing_weighted_norm": rational_object(forcing_norm),
        "persistent_floor_weighted": rational_object(
            persistent_norm / (1 - kappa)),
        "horizon": horizon,
        "weighted_bounds": [rational_object(entry) for entry in bounds],
        "all_future_conclusion": (
            "weighted auxiliary state is bounded by the persistent floor "
            "plus decaying initial and geometric-source transients at every "
            "future update"
        ),
        "zero_limit_if_persistent_forcing_zero": all(
            entry == 0 for entry in persistent),
        "derivation": "entrywise_monotonicity_and_weighted_geometric_convolution",
    }


def check_positive_certificate(certificate):
    rebuilt = construct_positive_certificate(
        certificate["initial"], certificate["matrix"], certificate["forcing"],
        certificate["rho"], certificate["weights"], certificate["kappa"],
        persistent=certificate["persistent_forcing"],
        horizon=certificate["horizon"],
    )
    if rebuilt != certificate:
        raise ValueError("positive comparison certificate does not replay exactly")
    return True


def nonautonomous_upper(gains, disturbances, initial):
    gains = tuple(as_fraction(value) for value in gains)
    disturbances = tuple(as_fraction(value) for value in disturbances)
    initial = as_fraction(initial)
    if len(gains) != len(disturbances):
        raise ValueError("gain and disturbance schedules must have equal length")
    if initial < 0 or any(not 0 <= q < 1 for q in gains):
        raise ValueError("nonautonomous gains or initial value are outside domain")
    if any(value < 0 for value in disturbances):
        raise ValueError("disturbance bounds must be nonnegative")
    result = [initial]
    for gain, disturbance in zip(gains, disturbances):
        result.append(gain * result[-1] + disturbance)
    return tuple(result)


def finite_entry_horizon(gains, disturbances, initial, cover_remainders,
                         metric_projection, criterion):
    lifted = nonautonomous_upper(gains, disturbances, initial)
    gains = tuple(as_fraction(value) for value in gains)
    disturbances = tuple(as_fraction(value) for value in disturbances)
    cover_remainders = tuple(as_fraction(value) for value in cover_remainders)
    if len(cover_remainders) != len(lifted):
        raise ValueError("cover remainder schedule must include the initial update")
    if any(value < 0 for value in cover_remainders):
        raise ValueError("cover remainder schedule must be nonnegative")
    metric_projection = as_fraction(metric_projection)
    criterion = as_fraction(criterion)
    if metric_projection < 0 or criterion <= 0:
        raise ValueError("finite-entry projection and criterion are outside domain")
    direct = tuple(metric_projection * value + remainder
                   for value, remainder in zip(lifted, cover_remainders))
    entry = None
    for index in range(len(direct)):
        if all(value < criterion for value in direct[index:]):
            entry = index
            break
    return {
        "schema_version": "pldr-finite-registered-horizon-v1",
        "horizon_type": "FINITE_REGISTERED_SCHEDULE",
        "gains": [rational_object(value) for value in gains],
        "disturbances": [rational_object(value) for value in disturbances],
        "initial": rational_object(as_fraction(initial)),
        "cover_remainders": [
            rational_object(value) for value in cover_remainders
        ],
        "metric_projection": rational_object(metric_projection),
        "terminal_relative_update": len(gains),
        "entry_relative_update": entry,
        "lifted_upper": [rational_object(value) for value in lifted],
        "direct_upper": [rational_object(value) for value in direct],
        "criterion": rational_object(criterion),
        "permanent_through_terminal": entry is not None,
    }


def check_finite_entry_horizon(certificate):
    rebuilt = finite_entry_horizon(
        certificate["gains"], certificate["disturbances"],
        certificate["initial"], certificate["cover_remainders"],
        certificate["metric_projection"], certificate["criterion"],
    )
    if rebuilt != certificate:
        raise ValueError(
            "finite registered-horizon certificate does not replay exactly")
    return True


def finite_trace_cannot_prove_infinite_tail(trace):
    """Construct two continuations sharing a finite trace but different limits."""

    trace = tuple(as_fraction(value) for value in trace)
    if not trace:
        raise ValueError("finite trace must be nonempty")
    zero_continuation = trace + tuple(Fraction(0) for _ in range(4))
    persistent_continuation = trace + tuple(trace[-1] + 1 for _ in range(4))
    return {
        "common_prefix_length": len(trace),
        "zero_continuation": [rational_object(value)
                              for value in zero_continuation],
        "persistent_continuation": [rational_object(value)
                                    for value in persistent_continuation],
        "infinite_conclusion_identified": False,
    }
