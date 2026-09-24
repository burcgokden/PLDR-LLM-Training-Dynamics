"""Finite numerical boundaries for independent scientific verifiers.

These checks precede subtraction, tolerance arithmetic and error aggregation.
They deliberately do not replace native-state or logical-byte admission.
"""
import json
import math
from numbers import Real

import numpy as np


def finite_scalar(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError('Invalid numerical scalar: ' + label)
    try:
        valid = math.isfinite(value)
    except (ValueError, TypeError, OverflowError):
        valid = False
    if not valid:
        raise ValueError('Nonfinite numerical scalar: ' + label)
    return value


def finite_array(value, label):
    array = np.asarray(value)
    if array.dtype.kind not in 'iuf' or not np.isfinite(array).all():
        raise ValueError('Invalid finite numerical array: ' + label)
    return array


def discrepancy(actual, claimed, label, *, atol=1e-12, rtol=0.):
    finite_scalar(actual, 'reconstructed ' + label)
    finite_scalar(claimed, 'claimed ' + label)
    finite_scalar(atol, 'absolute tolerance')
    finite_scalar(rtol, 'relative tolerance')
    if atol < 0 or rtol < 0:
        raise ValueError('Negative tolerance')
    error = abs(actual - claimed)
    tolerance = atol + rtol * abs(actual)
    finite_scalar(error, 'discrepancy ' + label)
    finite_scalar(tolerance, 'tolerance ' + label)
    if error > tolerance:
        raise ValueError('Independent reduction differs: ' + label)
    return float(error)


def array_discrepancy(actual, claimed, label, *, atol=1e-12, rtol=0.):
    actual = finite_array(actual, 'reconstructed ' + label)
    claimed = finite_array(claimed, 'claimed ' + label)
    if actual.shape != claimed.shape or actual.size == 0:
        raise ValueError('Numerical array shape differs: ' + label)
    finite_scalar(atol, 'absolute tolerance')
    finite_scalar(rtol, 'relative tolerance')
    if atol < 0 or rtol < 0:
        raise ValueError('Negative tolerance')
    with np.errstate(over='ignore', invalid='ignore'):
        errors = np.abs(actual.astype(float) - claimed.astype(float))
        tolerances = atol + rtol * np.abs(actual.astype(float))
    finite_array(errors, 'discrepancy ' + label)
    finite_array(tolerances, 'tolerance ' + label)
    if np.any(errors > tolerances):
        raise ValueError('Independent reduction differs: ' + label)
    return float(errors.max())


def load_json_strict(text):
    """Reject JSON constants and overflowing exponents; preserve null statuses."""
    def constant(value):
        raise ValueError('Nonfinite numerical scalar in JSON: ' + value)

    def floating(value):
        return finite_scalar(float(value), 'JSON number')

    return json.loads(text, parse_constant=constant, parse_float=floating)
