"""Total decision for the prospective direct row-map closure experiment."""

from enum import Enum
import math


SCHEMA_VERSION = "pldr-e0-prime-observable-v2"
BINDINGS = (
    "protocol_sha256",
    "schema_sha256",
    "source_sha256",
    "calibration_sha256",
    "data_order_sha256",
    "tokenizer_sha256",
)
AVAILABILITY_CHECKS = (
    "bindings",
    "fresh_namespace",
    "strict_json",
    "global_checkpoint_clock",
    "averaging_preconditioner_metadata",
    "eligible_unit_ledger",
)
CORE_CHECKS = (
    "direct_row_map_probe",
    "quiet_tail_contraction",
    "trajectory_guard_tube",
    "ordered_adamw_ledger",
)
MECHANISM_CHECKS = (
    "anchor_chart",
    "normalized_stress_path",
    "fresh_box_band",
    "gauss_newton_bridge",
    "curvature_factory",
    "fiber_convexity",
    "crossing_signature",
)
FAMILIES = (
    "direct_tail_contraction",
    "guard_tube_closure",
    "capture_quench_contrast",
)
REGISTERED_MINIMUMS = {
    "direct_tail_contraction": 80,
    "guard_tube_closure": 80,
    "capture_quench_contrast": 32,
}


class Outcome(Enum):
    PASS = "PASS"
    NOT_CONFIRMED = "NOT_CONFIRMED"
    INDETERMINATE = "INDETERMINATE"
    MODEL_REJECTED = "MODEL_REJECTED"
    INCOMPLETE = "INCOMPLETE"


def _finite(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _exact_mapping(value, names, label, reasons):
    if not isinstance(value, dict):
        reasons.append(f"{label} is not a mapping")
        return {}
    missing = sorted(set(names) - set(value))
    extra = sorted(set(value) - set(names))
    if missing:
        reasons.append(f"{label} is missing: {', '.join(missing)}")
    if extra:
        reasons.append(f"{label} has unknown fields: {', '.join(extra)}")
    return value


def _result(outcome, reasons, requirements=None, mechanism=None):
    return {
        "schema_version": SCHEMA_VERSION,
        "outcome": outcome.value,
        "reasons": list(reasons),
        "requirements": requirements or {},
        "mechanism_discrimination": mechanism or {},
    }


def holm(p_values, alpha=0.05):
    if set(p_values) != set(FAMILIES):
        raise ValueError("Holm input must contain exactly three families")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    rows = []
    passed = True
    for index, (name, value) in enumerate(ordered):
        threshold = alpha / (len(ordered) - index)
        row_pass = bool(passed and value <= threshold)
        rows.append({
            "family": name,
            "p_value": value,
            "threshold": threshold,
            "pass": row_pass,
        })
        passed = row_pass
    return all(row["pass"] for row in rows), rows


def _decide(record):
    reasons = []
    requirements = {}
    if not isinstance(record, dict):
        return _result(Outcome.INCOMPLETE, ["record is not a mapping"])
    if record.get("schema_version") != SCHEMA_VERSION:
        reasons.append("schema version is missing or stale")
    campaign = record.get("campaign_id")
    if not isinstance(campaign, str) or not campaign:
        reasons.append("campaign identifier is missing")
    bindings = _exact_mapping(record.get("bindings"), BINDINGS,
                              "bindings", reasons)
    for name in BINDINGS:
        value = bindings.get(name)
        if not (
            isinstance(value, str)
            and len(value) == 64
            and all(char in "0123456789abcdef" for char in value)
        ):
            reasons.append(f"binding {name} is not a lowercase SHA-256")

    availability = _exact_mapping(
        record.get("availability"), AVAILABILITY_CHECKS,
        "availability checks", reasons,
    )
    core = _exact_mapping(
        record.get("core_checks"), CORE_CHECKS, "core checks", reasons,
    )
    mechanism = _exact_mapping(
        record.get("mechanism_checks"), MECHANISM_CHECKS,
        "mechanism checks", reasons,
    )
    for name, mapping, allowed in (
        ("availability", availability, {"PASS", "MISSING", "ERROR"}),
        ("core", core, {"PASS", "FAIL", "UNRESOLVED", "MISSING", "ERROR"}),
        (
            "mechanism",
            mechanism,
            {"PASS", "FAIL", "UNRESOLVED", "NOT_APPLICABLE"},
        ),
    ):
        for key, value in mapping.items():
            if value not in allowed:
                reasons.append(f"{name} check {key} has invalid status")

    eligibility = _exact_mapping(
        record.get("eligibility"), FAMILIES, "eligibility", reasons,
    )
    for name in FAMILIES:
        row = eligibility.get(name)
        if not isinstance(row, dict):
            reasons.append(f"eligibility {name} is not a mapping")
            continue
        if row.get("status") not in {
            "ELIGIBLE", "MAX_CAP_UNRESOLVED", "MISSING", "ERROR"
        }:
            reasons.append(f"eligibility {name} has invalid status")
        for field in ("n", "minimum"):
            value = row.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                reasons.append(f"eligibility {name}.{field} is invalid")
        if row.get("minimum") != REGISTERED_MINIMUMS[name]:
            reasons.append(
                f"eligibility {name}.minimum is not the registered value"
            )

    gate = record.get("gate")
    if not isinstance(gate, dict):
        reasons.append("gate estimate is missing")
        gate = {}
    estimate = gate.get("estimate")
    interval = gate.get("interval")
    if not _finite(estimate):
        reasons.append("gate estimate is not finite")
    if not (
        isinstance(interval, list)
        and len(interval) == 2
        and all(_finite(value) for value in interval)
        and interval[0] <= interval[1]
    ):
        reasons.append("gate interval is invalid")
        interval = [0.0, 0.0]
    elif _finite(estimate) and not interval[0] <= estimate <= interval[1]:
        reasons.append("gate estimate lies outside its interval")

    pilot = record.get("branch_pilot")
    if not isinstance(pilot, dict):
        reasons.append("branch pilot is missing")
        pilot = {}
    pilot_status = pilot.get("status")
    if pilot_status not in {
        "NONPOSITIVE_COLLAPSE_RESOLVED",
        "POSITIVE_CAPTURE_RESOLVED",
        "SIGN_UNRESOLVED",
        "UNRESOLVED",
        "MISSING",
        "ERROR",
    }:
        reasons.append("branch pilot status is invalid")

    families = _exact_mapping(
        record.get("families"), FAMILIES, "inferential families", reasons,
    )
    p_values = {}
    directions = {}
    family_unresolved = []
    for name in FAMILIES:
        row = families.get(name)
        if not isinstance(row, dict):
            reasons.append(f"family {name} is not a mapping")
            continue
        status = row.get("status")
        if status == "CONCLUSIVE":
            if not _finite(row.get("p_value")) or not (
                0.0 <= float(row["p_value"]) <= 1.0
            ):
                reasons.append(f"family {name} p-value is invalid")
            elif not isinstance(row.get("direction_pass"), bool):
                reasons.append(f"family {name} direction is invalid")
            else:
                p_values[name] = float(row["p_value"])
                directions[name] = row["direction_pass"]
        elif status == "UNRESOLVED":
            family_unresolved.append(name)
        elif status in {"MISSING", "ERROR"}:
            reasons.append(f"family {name} is {status.lower()}")
        else:
            reasons.append(f"family {name} has invalid status")

    if reasons:
        return _result(Outcome.INCOMPLETE, reasons, requirements, mechanism)

    unavailable = [
        name for name in AVAILABILITY_CHECKS
        if availability[name] != "PASS"
    ]
    unavailable.extend(
        f"core:{name}" for name in CORE_CHECKS
        if core[name] in {"MISSING", "ERROR"}
    )
    unavailable.extend(
        f"eligibility:{name}" for name in FAMILIES
        if eligibility[name]["status"] in {"MISSING", "ERROR"}
    )
    lower, upper = map(float, interval)
    if upper <= 0.0:
        expected_pilot = "NONPOSITIVE_COLLAPSE_RESOLVED"
    elif lower > 0.0:
        expected_pilot = "POSITIVE_CAPTURE_RESOLVED"
    else:
        expected_pilot = "SIGN_UNRESOLVED"
    if pilot_status in {"MISSING", "ERROR"} or (
        pilot_status not in {expected_pilot, "UNRESOLVED"}
    ):
        unavailable.append("gate_branch_pilot")
    if unavailable:
        return _result(
            Outcome.INCOMPLETE,
            ["unavailable required evidence: " + ", ".join(unavailable)],
            requirements,
            mechanism,
        )

    failures = [name for name in CORE_CHECKS if core[name] == "FAIL"]
    requirements.update({
        f"core_{name}": core[name] == "PASS" for name in CORE_CHECKS
    })
    if failures:
        return _result(
            Outcome.MODEL_REJECTED,
            ["failed direct-closure checks: " + ", ".join(failures)],
            requirements,
            mechanism,
        )

    unresolved = [
        name for name in CORE_CHECKS if core[name] == "UNRESOLVED"
    ]
    unresolved.extend(
        f"eligibility:{name}" for name in FAMILIES
        if eligibility[name]["status"] == "MAX_CAP_UNRESOLVED"
    )
    unresolved.extend(f"family:{name}" for name in family_unresolved)
    if expected_pilot == "SIGN_UNRESOLVED":
        unresolved.append("gate_sign")
    if pilot_status == "UNRESOLVED":
        unresolved.append("gate_branch_pilot")
    if unresolved:
        return _result(
            Outcome.INDETERMINATE,
            ["resolved evidence is insufficient: " + ", ".join(unresolved)],
            requirements,
            mechanism,
        )

    for name in FAMILIES:
        requirements[f"minimum_{name}"] = (
            eligibility[name]["n"] >= eligibility[name]["minimum"]
        )
    if set(p_values) != set(FAMILIES):
        return _result(
            Outcome.INCOMPLETE,
            ["conclusive family inputs are incomplete"],
            requirements,
            mechanism,
        )
    holm_pass, rows = holm(p_values)
    requirements["holm_three_families"] = holm_pass
    requirements["holm_rows"] = rows
    requirements.update({
        f"direction_{name}": directions[name] for name in FAMILIES
    })
    failed = [
        name for name, value in requirements.items()
        if name != "holm_rows" and value is False
    ]
    if failed:
        return _result(
            Outcome.NOT_CONFIRMED,
            ["failed confirmatory requirements: " + ", ".join(failed)],
            requirements,
            mechanism,
        )
    return _result(Outcome.PASS, [], requirements, mechanism)


def decide(record):
    """Return exactly one typed outcome for every Python input."""
    try:
        return _decide(record)
    except Exception as error:
        return _result(
            Outcome.INCOMPLETE,
            [
                "decision input could not be evaluated: "
                f"{type(error).__name__}: {error}"
            ],
        )


def authorize_downstream(record, expected_bindings):
    decision = decide(record)
    if decision["outcome"] != Outcome.PASS.value:
        raise RuntimeError(
            "downstream launch denied: "
            f"E0-prime outcome {decision['outcome']}"
        )
    if (
        not isinstance(expected_bindings, dict)
        or set(expected_bindings) != set(BINDINGS)
        or record.get("bindings") != expected_bindings
    ):
        raise RuntimeError("downstream launch denied: bindings are stale")
    return decision
