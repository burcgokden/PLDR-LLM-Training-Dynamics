"""Upper-edge event semantics for nonnegative transported doses."""

import numpy as np


def certificate_margin_nearest(sigma, lower_edge, upper_edge):
    """Two-sided containment margin.  This is not offspring headroom."""
    sigma = np.asarray(sigma, dtype=float)
    return np.minimum(sigma - lower_edge, upper_edge - sigma)


def topple_headroom_upper(state, upper_threshold):
    """Oriented distance to the positive-dose upper guard."""
    return np.asarray(upper_threshold, dtype=float) - np.asarray(
        state, dtype=float)


def strict_upper_child(state, dose, upper_threshold):
    """A child occurs exactly when a nonnegative dose crosses strictly."""
    state = np.asarray(state, dtype=float)
    dose = np.asarray(dose, dtype=float)
    if np.any(dose < 0):
        raise ValueError("positive-dose child semantics require dose >= 0")
    return state + dose > np.asarray(upper_threshold, dtype=float)


def strict_empirical_upper_cdf(headrooms, dose):
    """Empirical P(headroom < dose), with equality a non-crossing."""
    values = np.asarray(headrooms, dtype=float)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("headrooms must be a nonempty finite array")
    if not np.isfinite(dose) or dose < 0:
        raise ValueError("dose must be finite and nonnegative")
    return float(np.mean(values < dose))
