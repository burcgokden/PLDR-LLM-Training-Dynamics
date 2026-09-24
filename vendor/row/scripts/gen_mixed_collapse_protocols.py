#!/usr/bin/env python3
"""Generate strict protocols for the mixed-collapse confirmation campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "protocols" / "mixed_collapse_confirmation"
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.mixed_collapse_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    DENSE_TIMEPOINT_SCHEMA,
    FACTORIAL_RESULT_SCHEMA,
    FULL_TIMEPOINT_SCHEMA,
    INTERVENTION_MODES,
    REGISTRY,
    SOURCE_RADIUS_SCHEMA,
    campaign_design,
    validate_design,
)


def canonical(value: Any) -> bytes:
    """Encode one JSON artifact canonically for deterministic generation."""

    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def digest_object(value: Any) -> str:
    """Hash the compact canonical JSON representation of an object."""

    compact = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(compact).hexdigest()


def _object(title: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "title": title,
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _array(items: dict[str, Any], count: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "array", "items": items}
    if count is not None:
        result.update({"minItems": count, "maxItems": count})
    return result


DIGEST = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
STRING = {"type": "string", "minLength": 1}
INTEGER = {"type": "integer"}
NONNEGATIVE_INTEGER = {"type": "integer", "minimum": 0}
NUMBER = {"type": "number"}
NONNEGATIVE = {"type": "number", "minimum": 0.0}
BOOLEAN = {"type": "boolean"}


def _timepoint(schema: str) -> dict[str, Any]:
    maps = REGISTRY["map_count"]
    features = REGISTRY["feature_count"]
    coordinate = _array(NONNEGATIVE, features)
    pair = _array(NONNEGATIVE_INTEGER, 2)
    return _object(
        "All-map mixed-collapse timepoint",
        {
            "schema_version": {"const": schema},
            "campaign_id": {"const": CAMPAIGN_ID},
            "step": NONNEGATIVE_INTEGER,
            "registry_sha256": DIGEST,
            "map_order": {"const": REGISTRY["map_order"]},
            "energy": _array(NONNEGATIVE, maps),
            "diameter_squared": _array(NONNEGATIVE, maps),
            "maximizing_pairs": _array(pair, maps),
            "gate_shape_coordinate_energy": _array(coordinate, maps),
            "normalized_shape_coordinate_energy": _array(coordinate, maps),
            "layer_final_gate": _array(
                _array(NUMBER, features), len(REGISTRY["layers"])
            ),
            "coordinate_factorization_relative_residual": NONNEGATIVE,
            "plga_quotient_ratio": _array(NONNEGATIVE, maps),
            "plga_absolute_response": _array(NONNEGATIVE, maps),
            "plga_force": _array(NONNEGATIVE, maps),
            "cross_map_pairs_formed": {"const": False},
        },
    )


def schemas() -> dict[str, dict[str, Any]]:
    """Return strict output schemas used by producers and analysis."""

    layer_effect = _object(
        "One seed-layer-horizon factorial effect",
        {
            "trajectory": STRING,
            "seed": INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "layer": NONNEGATIVE_INTEGER,
            "mode": {"enum": list(INTERVENTION_MODES)},
            "energy_ratio_to_control": NONNEGATIVE,
            "energy_difference_from_control": NUMBER,
        },
    )
    factorial = _object(
        "Hierarchy-aware final-gate factorial result",
        {
            "schema_version": {"const": FACTORIAL_RESULT_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "source_log_sha256": _array(DIGEST),
            "analysis_unit": {"const": "seed-layer-horizon"},
            "effects": _array(layer_effect),
            "all_registered_arms_present": BOOLEAN,
        },
    )
    radius_map = _object(
        "One predeclared-radius map result",
        {
            "map_id": STRING,
            "source_energy": NONNEGATIVE,
            "predicted_energy": NONNEGATIVE,
            "comparison_energy": NONNEGATIVE,
            "realized_radius": NONNEGATIVE,
            "predeclared_radius": NONNEGATIVE,
            "energy_upper": NONNEGATIVE,
            "energy_slack": NUMBER,
            "radius_slack": NUMBER,
        },
    )
    radius = _object(
        "Source-frozen cross-arithmetic radius result",
        {
            "schema_version": {"const": SOURCE_RADIUS_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "trajectory": STRING,
            "seed": INTEGER,
            "source_step": NONNEGATIVE_INTEGER,
            "endpoint_step": NONNEGATIVE_INTEGER,
            "checkpoint_sha256": DIGEST,
            "data_order_sha256": DIGEST,
            "prediction_device": STRING,
            "comparison_device": STRING,
            "predeclared_radius": NONNEGATIVE,
            "map_records": _array(radius_map, REGISTRY["map_count"]),
            "checkpoint_opened": BOOLEAN,
            "population_complete": BOOLEAN,
            "radius_contract_closed": BOOLEAN,
            "energy_envelope_closed": BOOLEAN,
            "technical_valid": BOOLEAN,
        },
    )
    analysis = _object(
        "Finite mixed-collapse analysis artifact",
        {
            "schema_version": {"const": ANALYSIS_SCHEMA},
            "campaign_id": {"const": CAMPAIGN_ID},
            "input_manifest_sha256": DIGEST,
            "trajectory_count": {"const": 3},
            "map_count": {"const": REGISTRY["map_count"]},
            "finite_horizon_only": {"const": True},
            "results_sha256": DIGEST,
        },
    )
    return {
        "full_timepoint.schema.json": _timepoint(FULL_TIMEPOINT_SCHEMA),
        "dense_timepoint.schema.json": _timepoint(DENSE_TIMEPOINT_SCHEMA),
        "factorial_result.schema.json": factorial,
        "source_radius.schema.json": radius,
        "analysis.schema.json": analysis,
    }


def generated_files() -> dict[Path, bytes]:
    """Return the complete deterministic protocol payload."""

    design = campaign_design()
    validate_design(design)
    signed = dict(design)
    signed["spec_sha256"] = digest_object(design)
    files = {
        Path("training_campaign_spec.json"): canonical(signed),
        Path("README.md"): (
            "# Mixed-collapse confirmation protocol\n\n"
            "Generated by `scripts/gen_mixed_collapse_protocols.py`.  "
            "The campaign is finite-horizon, uses all-map dense windows, "
            "a fixed dyadic block sequence, hierarchy-aware gate factorial "
            "arms, and a construction-frozen arithmetic radius.\n"
        ).encode("utf-8"),
    }
    files.update({Path(name): canonical(schema)
                  for name, schema in schemas().items()})
    checksum_lines = [
        f"{hashlib.sha256(payload).hexdigest()}  {path.as_posix()}"
        for path, payload in sorted(files.items(), key=lambda item: str(item[0]))
    ]
    files[Path("CHECKSUMS.sha256")] = (
        "\n".join(checksum_lines) + "\n"
    ).encode("ascii")
    return files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    arguments = parser.parse_args()
    files = generated_files()
    stale = []
    for relative, payload in files.items():
        destination = arguments.output / relative
        if arguments.check:
            if not destination.is_file() or destination.read_bytes() != payload:
                stale.append(relative.as_posix())
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    if stale:
        raise SystemExit("mixed-collapse protocols are stale: " + ", ".join(stale))
    action = "verified" if arguments.check else "generated"
    print(f"mixed-collapse protocols: {action} ({len(files)} files)")


if __name__ == "__main__":
    main()
