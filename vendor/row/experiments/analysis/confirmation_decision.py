"""Total exact-set and finite-sample decision helpers."""

from __future__ import annotations

import math


TERMINAL_STATUSES = (
    "CONFIRMED", "NOT_CONFIRMED", "INCOMPLETE", "NOT_RUN", "INFEASIBLE",
)


def _empty(pid, minimum, unit_minimum, records_per_unit):
    return {
        "protocol_id": pid,
        "status": "NOT_RUN",
        "n_records": 0,
        "registered_minimum": minimum,
        "registered_unit_minimum": unit_minimum,
        "records_per_unit": records_per_unit,
        "criteria": {},
        "summary": {},
        "reason_codes": ["NO_RAW_RECORDS"],
        "reasons": ["no raw records"],
        "offending_unit_keys": [],
    }


def base_result(pid, rows, registration, records_module):
    unit_minimum = int(registration["sample_size"])
    records_per_unit = len(registration.get(
        "required_arms", ["observational"]
    )) * len(registration.get("registered_block_sizes", [1]))
    expected_rows = registration.get("expected_unit_keys")
    minimum = (
        len(expected_rows)
        if expected_rows is not None
        else unit_minimum * records_per_unit
    )
    if not rows:
        return _empty(pid, minimum, unit_minimum, records_per_unit)

    reasons = []
    reason_codes = []
    offending = []
    infeasible = False
    if expected_rows is not None:
        expected = {
            records_module.canonical_unit_key(value) for value in expected_rows
        }
        if len(expected) != len(expected_rows):
            raise ValueError(f"{pid} registration contains duplicate unit keys")
        realized_list = [
            records_module.canonical_unit_key(row["unit_key"]) for row in rows
        ]
        realized = set(realized_list)
        duplicates = sorted({
            key for key in realized if realized_list.count(key) > 1
        })
        missing = sorted(expected - realized)
        unexpected = sorted(realized - expected)
        if duplicates:
            reason_codes.append("DUPLICATE_UNIT_KEY")
            reasons.append(f"{len(duplicates)} exact unit keys are duplicated")
            offending.extend(duplicates)
        if missing:
            reason_codes.append("MISSING_REGISTERED_UNIT_KEY")
            reasons.append(f"{len(missing)} registered unit keys are missing")
            offending.extend(missing)
        if unexpected:
            reason_codes.append("UNREGISTERED_UNIT_KEY")
            reasons.append(f"{len(unexpected)} unit keys are not registered")
            offending.extend(unexpected)
    elif len(rows) < minimum:
        reason_codes.append("BELOW_REGISTERED_MINIMUM")
        reasons.append(
            f"{len(rows)} records are below the registered minimum {minimum}"
        )

    for name in registration["required_measurements"]:
        unavailable = []
        for row in rows:
            measurement = records_module.measurement_map(row)[name]
            if measurement["status"] != "OBSERVED":
                unavailable.append(measurement)
        if unavailable:
            codes = {value["reason_code"] for value in unavailable}
            if any(code and code.startswith("INFEASIBLE_") for code in codes):
                infeasible = True
                reason_codes.append("REGISTERED_CONSTRUCTION_INFEASIBLE")
            else:
                reason_codes.append("REQUIRED_MEASUREMENT_UNAVAILABLE")
            reasons.append(f"{name} unavailable in {len(unavailable)} records")

    if infeasible:
        status = "INFEASIBLE"
    elif reasons:
        status = "INCOMPLETE"
    else:
        status = "READY"
    return {
        "protocol_id": pid,
        "status": status,
        "n_records": len(rows),
        "registered_minimum": minimum,
        "registered_unit_minimum": unit_minimum,
        "records_per_unit": records_per_unit,
        "criteria": {},
        "summary": {},
        "reason_codes": sorted(set(reason_codes)),
        "reasons": reasons,
        "offending_unit_keys": sorted(set(offending)),
    }


def paired_sign_randomization(values, *, direction, null=0.0):
    """Exact one-sided paired sign randomization result.

    Zeros are conservatively retained as unfavorable.  No Gaussian or
    large-sample approximation is used.
    """

    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("paired randomization needs finite paired contrasts")
    if direction == "lower":
        favorable = sum(value < null for value in values)
    elif direction == "upper":
        favorable = sum(value > null for value in values)
    else:
        raise ValueError("direction must be lower or upper")
    n = len(values)
    tail = sum(math.comb(n, k) for k in range(favorable, n + 1))
    p_value = tail / (2 ** n)
    ordered = sorted(values)
    middle = n // 2
    median = (
        ordered[middle]
        if n % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "n_pairs": n,
        "favorable_pairs": favorable,
        "median_contrast": median,
        "one_sided_exact_p": p_value,
        "procedure": "exact_paired_sign_randomization",
    }


def error_result(pid, rows, error):
    return {
        "protocol_id": pid,
        "status": "INCOMPLETE",
        "n_records": len(rows),
        "registered_minimum": 0,
        "registered_unit_minimum": 0,
        "records_per_unit": 0,
        "criteria": {},
        "summary": {},
        "reason_codes": ["ANALYSIS_INPUT_ERROR"],
        "reasons": [f"analysis input error: {type(error).__name__}: {error}"],
        "offending_unit_keys": [],
    }
