"""Prelaunch resource gate for complete streamed energy certificates."""

from __future__ import annotations

import math


SCHEMA_VERSION = "pldr-program-energy-resource-gate-v1"
REGISTERED_ARCHITECTURES = {
    "w064_d04": {
        "layers": 4, "width": 64, "dk": 16,
        "parameter_count": 4_551_680, "cells_per_layer": 960,
    },
    "w128_d08": {
        "layers": 8, "width": 128, "dk": 32,
        "parameter_count": 11_577_600, "cells_per_layer": 960,
    },
    "w256_d12": {
        "layers": 12, "width": 256, "dk": 64,
        "parameter_count": 36_392_960, "cells_per_layer": 960,
    },
}
DEFAULT_CAPS = {
    "gpu_bytes": 20 * 1024 ** 3,
    "host_bytes_per_job": 48 * 1024 ** 3,
    "retained_bytes_total": 150 * 1024 ** 3,
    "concurrent_jobs": 2,
}


def architecture_cost(name, *, bytes_per_scalar=8, retained_copies=3):
    if name not in REGISTERED_ARCHITECTURES:
        raise ValueError("unknown registered architecture")
    row = REGISTERED_ARCHITECTURES[name]
    stack_dimension = (
        row["layers"] * row["cells_per_layer"] * row["dk"] ** 2)
    stack_bytes = stack_dimension * bytes_per_scalar
    dense_derivative_bytes = (
        stack_dimension * row["parameter_count"] * bytes_per_scalar)
    streamed_peak = max(
        stack_bytes,
        row["cells_per_layer"] * row["dk"] ** 2 * bytes_per_scalar * 4,
    )
    retained = retained_copies * stack_bytes
    return {
        **row,
        "architecture": name,
        "stack_dimension": stack_dimension,
        "stack_bytes": stack_bytes,
        "streamed_peak_bytes": streamed_peak,
        "retained_bytes": retained,
        "dense_stack_parameter_derivative_bytes": dense_derivative_bytes,
        "dense_derivative_forbidden": True,
    }


def qualify_resources(measured, caps=None):
    """Check measured profiling data against the frozen two-GPU caps."""

    required = {
        "architecture", "peak_gpu_bytes", "peak_host_bytes",
        "retained_bytes", "projected_certificate_seconds",
        "dense_derivative_requested",
    }
    if not isinstance(measured, dict) or set(measured) != required:
        raise ValueError("resource measurement has missing or unknown fields")
    caps = dict(DEFAULT_CAPS if caps is None else caps)
    if set(caps) != set(DEFAULT_CAPS):
        raise ValueError("resource caps have missing or unknown fields")
    estimate = architecture_cost(measured["architecture"])
    values = {}
    for name in (
        "peak_gpu_bytes", "peak_host_bytes", "retained_bytes",
        "projected_certificate_seconds",
    ):
        value = measured[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
        values[name] = float(value)
    if not isinstance(measured["dense_derivative_requested"], bool):
        raise TypeError("dense_derivative_requested must be Boolean")
    conditions = {
        "dense_derivative_forbidden":
            not measured["dense_derivative_requested"],
        "gpu_cap": values["peak_gpu_bytes"] <= caps["gpu_bytes"],
        "host_cap": values["peak_host_bytes"] <= caps["host_bytes_per_job"],
        "storage_cap": values["retained_bytes"] <= caps["retained_bytes_total"],
        "analytical_streamed_peak_below_cap":
            estimate["streamed_peak_bytes"] <= caps["gpu_bytes"],
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture_cost": estimate,
        "measured": measured,
        "caps": caps,
        "conditions": conditions,
        "decision": "QUALIFIED" if all(conditions.values()) else "INFEASIBLE",
    }
