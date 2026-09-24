"""Residual interval linking full Hessian robustness to primary GN stress."""

import math


def certificate(boundary_result, checkpoint_batteries, relative_limit):
    values = []
    log_value = (boundary_result or {}).get(
        "gn_full_residual_relative_max")
    if log_value is None or not math.isfinite(float(log_value)):
        return {"certificate_ready": False, "pass": None, "values": []}
    values.append(float(log_value))
    for battery in checkpoint_batteries.values():
        rows = (battery or {}).get("ckpts") or []
        primary_rows = [row for row in rows if row.get("primary_defined")]
        if not primary_rows:
            return {
                "certificate_ready": False,
                "pass": None,
                "values": values,
            }
        for row in primary_rows:
            value = row.get("gn_full_residual_relative")
            if value is None or not math.isfinite(float(value)):
                return {
                    "certificate_ready": False,
                    "pass": None,
                    "values": values,
                }
            values.append(float(value))
    maximum = max(values)
    return {
        "certificate_ready": True,
        "relative_limit": float(relative_limit),
        "relative_residual_max": maximum,
        "pass": maximum <= float(relative_limit),
        "values": values,
    }
