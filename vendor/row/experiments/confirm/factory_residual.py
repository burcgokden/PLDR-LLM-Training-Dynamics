"""Euler factory residual for the manufactured curvature decomposition."""

from bisect import bisect_right
import math


def _interp(t, axis, values):
    pairs = sorted((float(x), float(y)) for x, y in zip(axis, values)
                   if math.isfinite(float(x)) and math.isfinite(float(y)))
    if not pairs:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    t = float(t)
    if t <= xs[0]:
        return ys[0]
    if t >= xs[-1]:
        return ys[-1]
    right = bisect_right(xs, t)
    x0, x1 = xs[right - 1], xs[right]
    y0, y1 = ys[right - 1], ys[right]
    return y0 + (y1 - y0) * (t - x0) / (x1 - x0)


def certificate(boundary_result, checkpoint_battery, relative_limit,
                base_degree=2.0):
    """Evaluate h = radial(x) - [2 p kappa u^2 + m_b b]."""
    series = (boundary_result or {}).get("primary_series") or {}
    kappa = ((boundary_result or {}).get("kappa") or {}).get("kappa")
    checkpoints = (checkpoint_battery or {}).get("ckpts") or []
    if kappa is None or not math.isfinite(float(kappa)):
        return {"certificate_ready": False, "pass": False, "rows": []}
    kappa = float(kappa)
    rows = []
    for checkpoint in checkpoints:
        if not checkpoint.get("primary_defined"):
            continue
        step = checkpoint.get("step")
        x = _interp(step, series.get("t", []), series.get("x", []))
        a = _interp(step, series.get("factor_t", []), series.get("a", []))
        jet = _interp(step, series.get("factor_t", []), series.get("J", []))
        if x is None or a is None or jet is None:
            continue
        u2 = (a * jet) ** 2
        base = x - kappa * u2
        for degree in (2, 3):
            measured = checkpoint.get(f"radial_deriv_p{degree}")
            if measured is None or not math.isfinite(float(measured)):
                continue
            expected = 2.0 * degree * kappa * u2 + base_degree * base
            residual = float(measured) - expected
            scale = max(abs(float(measured)), abs(expected), 1e-12)
            rows.append({
                "step": int(step),
                "path_degree": degree,
                "measured_radial": float(measured),
                "expected_radial": expected,
                "residual": residual,
                "relative_residual": abs(residual) / scale,
            })
    maximum = max((row["relative_residual"] for row in rows), default=None)
    return {
        "certificate_ready": bool(rows),
        "relative_limit": float(relative_limit),
        "relative_residual_max": maximum,
        "pass": bool(rows and maximum <= float(relative_limit)),
        "rows": rows,
    }
