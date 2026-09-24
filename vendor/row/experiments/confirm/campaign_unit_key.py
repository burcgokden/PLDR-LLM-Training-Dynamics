"""Canonical exact unit keys for the confirmation campaign."""

from __future__ import annotations

import json


UNIT_KEY_NAMES = (
    "seed", "checkpoint", "layer_bundle", "tensor_group",
    "certificate_box", "schedule_cell", "architecture_cell",
    "block_size", "arm",
)
OPTIONAL_TAGS = (
    "layer_bundle", "tensor_group", "certificate_box", "schedule_cell",
    "architecture_cell",
)


def validate_unit_key(value, record, error_type=ValueError):
    if not isinstance(value, dict):
        raise error_type("unit_key must be an object")
    missing = sorted(set(UNIT_KEY_NAMES) - set(value))
    extra = sorted(set(value) - set(UNIT_KEY_NAMES))
    if missing or extra:
        raise error_type(
            f"unit_key fields disagree: missing={missing}, unknown={extra}"
        )
    for name in ("seed", "checkpoint"):
        item = value[name]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise error_type(f"unit_key.{name} must be a nonnegative integer")
    for name in OPTIONAL_TAGS:
        item = value[name]
        if item is not None and (not isinstance(item, str) or not item):
            raise error_type(f"unit_key.{name} must be null or nonempty text")
    block_size = value["block_size"]
    if block_size is not None and (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise error_type("unit_key.block_size must be null or a positive integer")
    if not isinstance(value["arm"], str) or not value["arm"]:
        raise error_type("unit_key.arm must be nonempty text")
    if value["seed"] != record["seed"]:
        raise error_type("unit_key.seed disagrees with seed")
    if value["checkpoint"] != record["clocks"]["global_optimizer_step"]:
        raise error_type("unit_key.checkpoint disagrees with global clock")
    if value["arm"] != record["intervention"]["arm"]:
        raise error_type("unit_key.arm disagrees with intervention arm")
    return value


def canonical_unit_key(value):
    """Return a stable representation for exact registered-set comparison."""
    if not isinstance(value, dict) or set(value) != set(UNIT_KEY_NAMES):
        raise ValueError("unit_key is not closed")
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def schema_object():
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(UNIT_KEY_NAMES),
        "properties": {
            "seed": {"type": "integer", "minimum": 0},
            "checkpoint": {"type": "integer", "minimum": 0},
            **{
                name: {"type": ["string", "null"]}
                for name in OPTIONAL_TAGS
            },
            "block_size": {
                "type": ["integer", "null"], "minimum": 1,
            },
            "arm": {"type": "string", "minLength": 1},
        },
    }
