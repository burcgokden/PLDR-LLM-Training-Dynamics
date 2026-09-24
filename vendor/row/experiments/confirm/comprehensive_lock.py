"""Immutable construction-lock schema for the comprehensive campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .comprehensive_campaign import campaign_spec, digest_object


SCHEMA_VERSION = "pldr-comprehensive-construction-lock-v1"


def seal_lock(unsigned: dict[str, Any]) -> dict[str, Any]:
    value = dict(unsigned)
    value.pop("lock_sha256", None)
    value["lock_sha256"] = digest_object(value)
    validate_lock(value)
    return value


def validate_lock(value: dict[str, Any]) -> bool:
    spec = campaign_spec()
    required = {
        "schema_version", "campaign_id", "campaign_spec_sha256",
        "primary_cells", "diameter_ratio_limit_by_layer", "route_bounds",
        "shape_correction_ratio_limit", "plga_secant_product_maximum",
        "normalized_effect_floor", "identity_relative_tolerance",
        "native_float32_replay_tolerance", "source_report_sha256",
        "constant_enlargement_after_lock", "lock_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("construction lock has missing or unknown fields")
    if (
        value["schema_version"] != SCHEMA_VERSION
        or value["campaign_id"] != spec["campaign_id"]
        or value["campaign_spec_sha256"] != spec["spec_sha256"]
        or value["constant_enlargement_after_lock"] is not False
    ):
        raise ValueError("construction lock identity or mutability is invalid")
    recorded = value["lock_sha256"]
    unsigned = dict(value)
    unsigned.pop("lock_sha256")
    if recorded != digest_object(unsigned):
        raise ValueError("construction lock digest does not replay")
    cells = value["primary_cells"]
    if (
        not isinstance(cells, list)
        or len(cells) != 3
        or {row.get("layer") for row in cells} != {0, 1, 2}
        or any(
            not isinstance(row, dict)
            or set(row) != {"layer", "anchor", "updates"}
            or row["anchor"] not in spec["trajectories"]["anchors"]
            or row["updates"] != spec["trajectories"]["confirmation_block_updates"]
            for row in cells
        )
    ):
        raise ValueError("construction lock primary cells are malformed")
    limits = value["diameter_ratio_limit_by_layer"]
    if (
        not isinstance(limits, dict)
        or set(limits) != {"0", "1", "2"}
        or any(
            not math.isfinite(float(bound)) or not 0.0 < float(bound) < 1.0
            for bound in limits.values()
        )
    ):
        raise ValueError("construction diameter limits are invalid")
    routes = value["route_bounds"]
    if not isinstance(routes, dict) or set(routes) != {"0", "1", "2"}:
        raise ValueError("construction route bounds are malformed")
    for route in routes.values():
        if (
            not isinstance(route, dict)
            or set(route) != {"gate_abs", "shape_abs", "total_abs"}
            or any(
                not math.isfinite(float(bound)) or float(bound) < 0.0
                for bound in route.values()
            )
        ):
            raise ValueError("construction route bound is invalid")
    for name in (
        "shape_correction_ratio_limit", "plga_secant_product_maximum",
        "normalized_effect_floor", "identity_relative_tolerance",
        "native_float32_replay_tolerance",
    ):
        number = float(value[name])
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"construction lock {name} is invalid")
    sources = value["source_report_sha256"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            for digest in sources
        )
        or len(set(sources)) != len(sources)
    ):
        raise ValueError("construction lock source digests are malformed")
    if (
        float(value["normalized_effect_floor"])
        != float(spec["decisions"]["normalized_effect_floor"])
        or float(value["identity_relative_tolerance"])
        != float(spec["decisions"]["identity_relative_tolerance"])
        or float(value["native_float32_replay_tolerance"])
        != float(spec["decisions"]["native_float32_replay_tolerance"])
    ):
        raise ValueError("construction lock changed a qualification tolerance")
    return True


def load_lock(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    validate_lock(value)
    return value


__all__ = ["SCHEMA_VERSION", "load_lock", "seal_lock", "validate_lock"]
