"""Small fail-closed validator for the campaign's generated JSON schemas."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported schema type {expected}")


def validate(value: Any, schema: dict[str, Any], *, path: str = "$") -> None:
    """Validate the strict subset emitted by the protocol generator."""

    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema node is not an object")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"{path}: value differs from schema constant")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: value is outside the schema enumeration")
    expected = schema.get("type")
    if expected is not None:
        choices = expected if isinstance(expected, list) else [expected]
        if not any(_matches_type(value, choice) for choice in choices):
            raise ValueError(f"{path}: value has the wrong type")

    if isinstance(value, dict) and expected == "object":
        properties = schema.get("properties")
        required = schema.get("required")
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError(f"{path}: object schema is incomplete")
        missing = set(required) - set(value)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}")
        unknown = set(value) - set(properties)
        if unknown and schema.get("additionalProperties") is False:
            raise ValueError(f"{path}: unknown fields {sorted(unknown)}")
        for name, child in value.items():
            if name in properties:
                validate(child, properties[name], path=f"{path}.{name}")
    elif isinstance(value, list) and expected == "array":
        minimum = schema.get("minItems", 0)
        if len(value) < minimum:
            raise ValueError(f"{path}: array has fewer than {minimum} items")
        maximum = schema.get("maxItems")
        if maximum is not None and len(value) > maximum:
            raise ValueError(f"{path}: array has more than {maximum} items")
        item_schema = schema.get("items")
        if not isinstance(item_schema, dict):
            raise ValueError(f"{path}: array schema has no item schema")
        for index, child in enumerate(value):
            validate(child, item_schema, path=f"{path}[{index}]")
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{path}: string is too short")
        pattern = schema.get("pattern")
        if pattern is not None and re.fullmatch(pattern, value) is None:
            raise ValueError(f"{path}: string does not match its pattern")
    elif (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and "minimum" in schema
        and value < schema["minimum"]
    ):
        raise ValueError(f"{path}: value is below its minimum")


def load_json(path: str | Path) -> Any:
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"nonfinite JSON token {token}")
        ),
    )


def validate_file(value_path: str | Path, schema_path: str | Path) -> Any:
    value = load_json(value_path)
    schema = load_json(schema_path)
    validate(value, schema)
    return value
