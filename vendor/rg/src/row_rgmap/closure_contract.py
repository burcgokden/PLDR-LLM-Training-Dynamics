"""Canonical branch and decision contracts for the v3 closure experiment."""

from __future__ import annotations

import copy
from typing import Any


BRANCH_SPECS: dict[str, dict[str, Any]] = {
    "baseline-a": {"operation": "none", "role": "repeatability-control"},
    "baseline-b": {"operation": "none", "role": "repeatability-control"},
    "first-moment-0.999": {
        "operation": "scale-first-moment",
        "factor": 0.999,
        "role": "same-fiber-closure-test",
    },
    "first-moment-0.9": {
        "operation": "scale-first-moment",
        "factor": 0.9,
        "role": "same-fiber-closure-test",
    },
    "zero-first-moment": {
        "operation": "scale-first-moment",
        "factor": 0.0,
        "role": "same-fiber-closure-test",
    },
    "second-moment-0.999": {
        "operation": "scale-second-moment",
        "factor": 0.999,
        "role": "same-fiber-closure-test",
    },
    "second-moment-0.5": {
        "operation": "scale-second-moment",
        "factor": 0.5,
        "role": "same-fiber-closure-test",
    },
    "second-moment-2": {
        "operation": "scale-second-moment",
        "factor": 2.0,
        "role": "same-fiber-closure-test",
    },
    "zero-second-moment": {
        "operation": "scale-second-moment",
        "factor": 0.0,
        "role": "same-fiber-closure-test",
    },
    "first-moment-one-ulp": {
        "operation": "one-ulp-first-moment",
        "role": "represented-resolution-control",
    },
    "layernorm-bias-shift": {
        "operation": "final-layernorm-bias-shift",
        "offset": 0.01,
        "role": "model-coordinate-sensitivity-control",
    },
    "sign-gauge-consistent": {
        "operation": "sign-gauge",
        "layers": [0, 1, 2],
        "transform_parameters": True,
        "co_transform_first_moment": True,
        "role": "gauge-orbit-null-control",
    },
    "sign-gauge-params-only": {
        "operation": "sign-gauge",
        "layers": [0, 1, 2],
        "transform_parameters": True,
        "co_transform_first_moment": False,
        "role": "augmented-state-closure-test",
    },
    "sign-gauge-moments-only": {
        "operation": "sign-gauge",
        "layers": [0, 1, 2],
        "transform_parameters": False,
        "co_transform_first_moment": True,
        "role": "paired-gauge-control",
    },
    "sign-gauge-layer0-consistent": {
        "operation": "sign-gauge",
        "layers": [0],
        "transform_parameters": True,
        "co_transform_first_moment": True,
        "role": "gauge-orbit-null-control",
    },
    "sign-gauge-layer0-params-only": {
        "operation": "sign-gauge",
        "layers": [0],
        "transform_parameters": True,
        "co_transform_first_moment": False,
        "role": "augmented-state-closure-test",
    },
}


_MEASUREMENT_KEYS = {
    "changed_tensor_count",
    "changed_element_count",
    "model_keys",
    "model_keys_sha256",
    "selected_element",
}


def branch_specification(branch: str) -> dict[str, Any]:
    """Return a detached copy of one canonical branch specification."""

    try:
        return copy.deepcopy(BRANCH_SPECS[branch])
    except KeyError as error:
        raise ValueError(f"unknown closure branch: {branch}") from error


def validate_serialized_branch_specification(
    branch: str, serialized: Any
) -> dict[str, Any]:
    """Validate canonical fields and allowed measured annotations."""

    if not isinstance(serialized, dict):
        raise ValueError(f"branch specification is not an object: {branch}")
    expected = BRANCH_SPECS.get(branch)
    if expected is None:
        raise ValueError(f"unknown closure branch: {branch}")
    if any(serialized.get(key) != value for key, value in expected.items()):
        raise ValueError(f"branch specification differs from registry: {branch}")
    unknown = set(serialized) - set(expected) - _MEASUREMENT_KEYS
    if unknown:
        raise ValueError(
            f"branch specification has undeclared keys for {branch}: {sorted(unknown)}"
        )
    return branch_specification(branch)


DECISION_STATUSES = ("REJECTED", "NOT_REJECTED", "NOT_EVALUABLE")


def decision_status(*, eligible: bool, rejected: bool) -> str:
    """Map eligibility and an observed counterexample to the public vocabulary."""

    if not eligible:
        return "NOT_EVALUABLE"
    return "REJECTED" if rejected else "NOT_REJECTED"
