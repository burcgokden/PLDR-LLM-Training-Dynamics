"""Qualified adapters for the prospective observable-closure experiment."""

import json
import math
from pathlib import Path
import re


PRECONDITIONER_METADATA = {"operator": "inverse", "window": "trailing"}

REASON_CODES = {
    "CAP_EXHAUSTED",
    "INSUFFICIENT_ELIGIBLE_UNITS",
    "MEASUREMENT_ERROR",
    "NOT_APPLICABLE_BRANCH",
    "SIGN_UNRESOLVED",
}


def global_checkpoint_step(checkpoint_path, payload, chain_meta=None):
    """Resolve global time and reject disagreement between valid sources."""
    checkpoint_path = Path(checkpoint_path)
    match = re.fullmatch(r"ckpt_(\d+)\.pt", checkpoint_path.name)
    filename_step = int(match.group(1)) if match else None
    internal = payload.get("step") if isinstance(payload, dict) else None
    if internal is not None and (
        isinstance(internal, bool)
        or not isinstance(internal, int)
        or internal < 0
    ):
        raise ValueError("segment-local checkpoint step is invalid")
    offset_step = None
    if isinstance(chain_meta, dict):
        offset = chain_meta.get("global_step_offset")
        if offset is not None and (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
        ):
            raise ValueError("global checkpoint offset is invalid")
        if offset is not None and internal is None:
            raise ValueError("checkpoint offset requires a local step")
        if offset is not None:
            offset_step = offset + internal
    candidates = [
        value for value in (filename_step, offset_step)
        if value is not None
    ]
    if candidates and len(set(candidates)) != 1:
        raise ValueError("checkpoint filename and chain offset disagree")
    if candidates:
        global_step = candidates[0]
        source = (
            "filename_and_chain_offset"
            if filename_step is not None and offset_step is not None
            else "filename" if filename_step is not None
            else "chain_offset"
        )
    elif isinstance(internal, int):
        global_step = internal
        source = "monolithic_internal_step"
    else:
        raise ValueError("no valid global checkpoint step source")
    if global_step < 0:
        raise ValueError("global checkpoint step is negative")
    return {
        "global_step": global_step,
        "segment_local_step": internal,
        "source": source,
    }


def assemble_preconditioner_metadata(boundary, control):
    """Accept one summary field only when both fresh cells agree."""
    values = [
        (boundary or {}).get("averaging_preconditioner"),
        (control or {}).get("averaging_preconditioner"),
    ]
    if values != [PRECONDITIONER_METADATA, PRECONDITIONER_METADATA]:
        return {
            "status": "MISSING",
            "value": None,
            "reason_code": "MEASUREMENT_ERROR",
        }
    return {
        "status": "PASS",
        "value": PRECONDITIONER_METADATA,
        "reason_code": None,
    }


def gate_pilot_route(estimate, interval):
    """Return an executable or terminal route for every finite sign interval."""
    if (
        isinstance(estimate, bool)
        or not isinstance(estimate, (int, float))
        or not math.isfinite(float(estimate))
        or not isinstance(interval, (list, tuple))
        or len(interval) != 2
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in interval
        )
        or interval[0] > estimate
        or estimate > interval[1]
    ):
        return {
            "status": "ERROR",
            "reason_code": "MEASUREMENT_ERROR",
            "run": None,
        }
    lower, upper = map(float, interval)
    if upper <= 0.0:
        return {
            "status": "RUN_NONPOSITIVE_COLLAPSE",
            "reason_code": None,
            "run": "nonpositive_collapse",
        }
    if lower > 0.0:
        return {
            "status": "RUN_POSITIVE_CAPTURE",
            "reason_code": None,
            "run": "positive_capture",
        }
    return {
        "status": "NOT_RUN_SIGN_UNRESOLVED",
        "reason_code": "SIGN_UNRESOLVED",
        "run": None,
    }


def nullable_measurement(value, reason_code=None):
    """Represent unavailable values without nonstandard JSON numbers."""
    finite = (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )
    if finite:
        if reason_code is not None:
            raise ValueError("finite values cannot carry a missing reason")
        return {"value": float(value), "reason_code": None}
    if reason_code not in REASON_CODES:
        raise ValueError("unavailable values require a registered reason")
    return {"value": None, "reason_code": reason_code}


def strict_json_text(value):
    """Serialize the record using the strict JSON numeric domain."""
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
