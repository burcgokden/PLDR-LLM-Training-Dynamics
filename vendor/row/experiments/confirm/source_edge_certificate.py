"""Finite-horizon source-versus-threshold interval certificates."""

import math


def _physical_upper(window, one_minus_beta1):
    edge = window.get("band_hi")
    eta = window.get("eta")
    if edge is None or eta is None:
        return None
    edge = float(edge)
    eta = float(eta)
    if not math.isfinite(edge) or not math.isfinite(eta) or eta <= 0:
        return None
    return edge / (one_minus_beta1 * eta)


def finite_horizon_certificate(windows, loading_result, one_minus_beta1,
                               classifier, minimum_pairs=4):
    """Compare every eligible exact source step with upper-edge motion.

    The interval endpoints are uniform envelopes over the eligible observed
    horizon, so the two intervals are simultaneous by construction.  Source
    rows use the exact clipped or unclipped factor-law increment stored at
    index three by ``e0_measure.loading_slope``.  No first-order proxy is
    substituted.  The result is a finite-horizon measurement and makes no
    claim about unobserved future steps.
    """
    if not (math.isfinite(float(one_minus_beta1))
            and 0 < float(one_minus_beta1) <= 1):
        raise ValueError("one_minus_beta1 must lie in (0,1]")

    by_window = {
        int(window["w"]): window for window in windows
        if not window.get("excluded") and not window.get("band_empty")
    }
    exact_source = {}
    for family in ("closed", "open"):
        rows = (loading_result.get(family) or {}).get("windows") or []
        for row in rows:
            if len(row) >= 4 and math.isfinite(float(row[3])):
                exact_source[int(row[0])] = float(row[3])

    records = []
    for index in sorted(exact_source):
        current = by_window.get(index)
        following = by_window.get(index + 1)
        if current is None or following is None:
            continue
        c0 = _physical_upper(current, one_minus_beta1)
        c1 = _physical_upper(following, one_minus_beta1)
        t0 = 0.5 * (float(current["lo"]) + float(current["hi"]))
        t1 = 0.5 * (float(following["lo"]) + float(following["hi"]))
        if c0 is None or c1 is None or not t1 > t0:
            continue
        records.append({
            "window": index,
            "threshold_increment_per_step": (c1 - c0) / (t1 - t0),
            "complete_source_increment_per_step": exact_source[index],
        })

    if len(records) < int(minimum_pairs):
        return {
            "certificate_ready": False,
            "classification": None,
            "n_pairs": len(records),
            "scope": "finite_observed_horizon",
            "records": records,
        }

    threshold_values = [
        row["threshold_increment_per_step"] for row in records]
    source_values = [
        row["complete_source_increment_per_step"] for row in records]
    threshold_interval = [min(threshold_values), max(threshold_values)]
    source_interval = [min(source_values), max(source_values)]
    classification = classifier(threshold_interval, source_interval)
    return {
        "certificate_ready": True,
        "classification": classification,
        "n_pairs": len(records),
        "scope": "finite_observed_horizon",
        "simultaneous_rule": "uniform envelope over all eligible pairs",
        "threshold_increment_interval": threshold_interval,
        "complete_source_increment_interval": source_interval,
        "records": records,
    }
