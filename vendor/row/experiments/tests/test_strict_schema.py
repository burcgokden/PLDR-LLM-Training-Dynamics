"""Fail-closed tests for scientific artifact validation."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.strict_schema import validate  # noqa: E402


SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "value": {"type": "number", "minimum": 0.0},
        "flags": {
            "type": "array",
            "items": {"type": "boolean"},
            "minItems": 1,
        },
    },
    "required": ["name", "value", "flags"],
    "additionalProperties": False,
}


def test_strict_schema_accepts_only_the_declared_finite_record():
    validate({"name": "map", "value": 1.0, "flags": [True]}, SCHEMA)
    with pytest.raises(ValueError, match="unknown fields"):
        validate(
            {"name": "map", "value": 1.0, "flags": [True], "extra": 1},
            SCHEMA,
        )
    with pytest.raises(ValueError, match="wrong type"):
        validate({"name": "map", "value": "1", "flags": [True]}, SCHEMA)
    with pytest.raises(ValueError, match="wrong type"):
        validate({"name": "map", "value": float("nan"), "flags": [True]}, SCHEMA)
    with pytest.raises(ValueError, match="missing fields"):
        validate({"name": "map", "value": 1.0}, SCHEMA)
