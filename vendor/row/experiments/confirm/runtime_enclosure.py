"""Outward bridge from runtime floats to exact rational certificate inputs."""

from __future__ import annotations

from fractions import Fraction
import math

import numpy as np

from validated_interval import RationalInterval


def outward_runtime_interval(value, absolute_error=0.0):
    value = float(value)
    error = float(absolute_error)
    if not math.isfinite(value) or not math.isfinite(error) or error < 0:
        raise ValueError("runtime values and error bounds must be finite")
    lower = math.nextafter(value - error, -math.inf)
    upper = math.nextafter(value + error, math.inf)
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise OverflowError("outward runtime interval is not finite")
    return RationalInterval(Fraction.from_float(lower), Fraction.from_float(upper))


def enclose_runtime_array(values, absolute_errors=0.0):
    values = np.asarray(values, dtype=float)
    errors = np.asarray(absolute_errors, dtype=float)
    if errors.ndim == 0:
        errors = np.full(values.shape, float(errors))
    if values.shape != errors.shape:
        raise ValueError("runtime values and errors have different shapes")
    if not np.isfinite(values).all() or not np.isfinite(errors).all():
        raise ValueError("runtime arrays contain a nonfinite value")
    if (errors < 0).any():
        raise ValueError("runtime error bounds must be nonnegative")
    flat = [
        outward_runtime_interval(value, error).to_object()
        for value, error in zip(values.reshape(-1), errors.reshape(-1))
    ]
    return {
        "schema_version": "pldr-outward-runtime-array-v1",
        "shape": list(values.shape),
        "intervals": flat,
        "error_sources": [
            "kernel_rounding", "reduction_order", "reconstruction",
            "serialization",
        ],
    }


def verify_runtime_array(record, values):
    if record.get("schema_version") != "pldr-outward-runtime-array-v1":
        raise ValueError("unknown runtime enclosure record")
    values = np.asarray(values, dtype=float)
    if list(values.shape) != record.get("shape"):
        raise ValueError("runtime enclosure shape disagrees with the array")
    intervals = record.get("intervals")
    if not isinstance(intervals, list) or len(intervals) != values.size:
        raise ValueError("runtime enclosure does not contain every coordinate")
    for interval, value in zip(intervals, values.reshape(-1)):
        if not RationalInterval.from_object(interval).contains(
            Fraction.from_float(float(value))
        ):
            raise ValueError("runtime primitive lies outside its exact interval")
    return True


def charged_gain(exact_gain_upper, runtime_operator_error):
    gain = float(exact_gain_upper)
    error = float(runtime_operator_error)
    if (
        not math.isfinite(gain) or gain < 0
        or not math.isfinite(error) or error < 0
    ):
        raise ValueError("gain and runtime charge must be nonnegative")
    accepted = math.nextafter(gain + error, math.inf)
    return {
        "exact_gain_upper": gain,
        "runtime_operator_error": error,
        "accepted_gain_upper": accepted,
        "strict_contraction": bool(accepted < 1.0),
    }
