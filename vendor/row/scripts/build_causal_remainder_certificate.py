#!/usr/bin/env python3
"""Build source-frozen row error radii from validated directional bounds."""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "confirm"))

from directional_taylor import certify_directional_segment  # noqa: E402


INPUT_SCHEMA = "pldr-causal-validated-jet-bounds-v1"
OUTPUT_SCHEMA = "pldr-causal-remainder-certificate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _fraction(value: Any) -> Fraction:
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        return Fraction(int(value["numerator"]), int(value["denominator"]))
    return Fraction(str(value))


def _rational(value: Fraction) -> dict[str, int]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def build(path: str | Path) -> dict[str, Any]:
    source_path = Path(path).resolve()
    value = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != INPUT_SCHEMA
        or value.get("successor_values_present") is not False
        or value.get("bounds_valid_on_complete_parameter_segment") is not True
        or not isinstance(value.get("rows"), list)
        or not value["rows"]
    ):
        raise ValueError("validated jet-bound input has an invalid contract")
    source_state_digest = value.get("source_state_digest")
    if (
        not isinstance(source_state_digest, str)
        or len(source_state_digest) != 64
        or any(character not in "0123456789abcdef"
               for character in source_state_digest)
    ):
        raise ValueError("validated bounds need an exact source-state digest")

    certificates = []
    taylor_radii = []
    jvp_error_radii = []
    total_radii = []
    expected_fields = {
        "point_second_directional_uppers",
        "third_directional_modulus",
        "subdivisions",
        "jvp_error_radius",
    }
    for index, row in enumerate(value["rows"]):
        if not isinstance(row, dict) or set(row) != expected_fields:
            raise ValueError(f"validated row {index} has an invalid schema")
        subdivisions = int(row["subdivisions"])
        if subdivisions != 2:
            raise ValueError(
                f"validated row {index} changed the registered segment grid")
        certificate = certify_directional_segment(
            point_second_directional_uppers=row[
                "point_second_directional_uppers"],
            third_directional_modulus=row["third_directional_modulus"],
            subdivisions=subdivisions,
        )
        taylor_radius = _fraction(certificate["taylor_remainder_upper"])
        jvp_error_radius = _fraction(row["jvp_error_radius"])
        if taylor_radius < 0 or jvp_error_radius < 0:
            raise ValueError(f"validated row {index} has a negative radius")
        total_radius = taylor_radius + jvp_error_radius
        certificates.append(certificate)
        taylor_radii.append(taylor_radius)
        jvp_error_radii.append(jvp_error_radius)
        total_radii.append(total_radius)

    float_radii = np.asarray([float(radius) for radius in total_radii])
    if not np.isfinite(float_radii).all():
        raise ValueError("validated row radii exceed float64 range")
    float_radii = np.nextafter(float_radii, np.full_like(float_radii, np.inf))
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "source_bounds_path": str(source_path),
        "source_bounds_sha256": _sha256(source_path),
        "source_state_digest": source_state_digest,
        "successor_values_present": False,
        "frozen_before_successor": True,
        "row_count": len(total_radii),
        "row_taylor_remainder_radius_rational": [
            _rational(radius) for radius in taylor_radii
        ],
        "row_jvp_error_radius_rational": [
            _rational(radius) for radius in jvp_error_radii
        ],
        "row_remainder_radius_rational": [
            _rational(radius) for radius in total_radii
        ],
        "row_remainder_radius_float64_outward": float_radii.tolist(),
        "row_certificates": certificates,
    }
    unsigned = json.dumps(
        output, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    output["certificate_sha256"] = hashlib.sha256(unsigned).hexdigest()
    return output


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    certificate = build(arguments.input)
    write_json(arguments.output, certificate)
    print(json.dumps({
        "certificate_sha256": certificate["certificate_sha256"],
        "output": str(Path(arguments.output).resolve()),
        "row_count": certificate["row_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
