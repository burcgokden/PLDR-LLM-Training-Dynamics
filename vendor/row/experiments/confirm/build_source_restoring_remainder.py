#!/usr/bin/env python3
"""Reduce validated source-segment curvature bounds to Taylor remainders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.finite_increment_live import write_npz_atomic  # noqa: E402
from confirm.source_restoring_live import SOURCE_SCHEMA  # noqa: E402
from confirm.source_restoring_specs import CAMPAIGN_ID, REGISTRY  # noqa: E402


INPUT_SCHEMA = "pldr-source-restoring-validated-curvature-v1"
OUTPUT_SCHEMA = "pldr-source-restoring-remainder-enclosure-v1"
INPUT_FIELDS = frozenset({
    "schema_version",
    "campaign_id",
    "parent_source_sha256",
    "source_parameter_sha256",
    "successor_values_used",
    "energy_curvature_abs_upper",
    "endpoint_contrast_norm_upper",
    "native_contrast_radius",
    "metadata_json",
})


def scalar(record: Any, name: str, kind: type) -> Any:
    if name not in record:
        raise ValueError(f"validated curvature artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def validated_taylor_remainder(
    energy_curvature_abs_upper: Any,
    endpoint_contrast_norm_upper: Any,
    native_contrast_radius: Any,
) -> dict[str, np.ndarray]:
    """Apply integral Taylor one-half with directed float64 endpoints."""

    curvature = np.asarray(energy_curvature_abs_upper, dtype=np.float64)
    endpoint = np.asarray(endpoint_contrast_norm_upper, dtype=np.float64)
    native = np.asarray(native_contrast_radius, dtype=np.float64)
    if (
        curvature.ndim != 2
        or endpoint.shape != curvature.shape
        or native.shape != curvature.shape
        or any(
            not np.isfinite(value).all()
            for value in (curvature, endpoint, native)
        )
        or any(
            np.any(value < 0.0)
            for value in (curvature, endpoint, native)
        )
    ):
        raise ValueError("validated curvature arrays are malformed")
    radius = 0.5 * curvature
    lower = np.where(
        radius == 0.0,
        0.0,
        np.nextafter(-radius, np.full_like(radius, -np.inf)),
    )
    upper = np.where(
        radius == 0.0,
        0.0,
        np.nextafter(radius, np.full_like(radius, np.inf)),
    )
    return {
        "remainder_lower": lower,
        "remainder_upper": upper,
        "endpoint_contrast_norm_upper": endpoint,
        "native_contrast_radius": native,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--validated-curvature", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    source_path = arguments.source.resolve()
    source_digest = sha256_path(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        if scalar(source, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown source-state artifact schema")
        source_metadata = json.loads(scalar(source, "metadata_json", str))
    if (
        not isinstance(source_metadata, dict)
        or source_metadata.get("campaign_id") != CAMPAIGN_ID
        or source_metadata.get("successor_values_used") is not False
        or source_metadata.get("successor_checkpoint_opened") is not False
        or not isinstance(source_metadata.get("created_at_ns"), int)
    ):
        raise ValueError("Taylor reduction needs a bound source-only artifact")

    input_path = arguments.validated_curvature.resolve()
    with np.load(input_path, allow_pickle=False) as record:
        if set(record.files) != INPUT_FIELDS:
            raise ValueError("validated curvature has missing or unknown fields")
        if (
            scalar(record, "schema_version", str) != INPUT_SCHEMA
            or scalar(record, "campaign_id", str) != CAMPAIGN_ID
            or scalar(record, "parent_source_sha256", str) != source_digest
            or scalar(record, "source_parameter_sha256", str)
            != source_metadata.get("source_parameter_sha256")
            or scalar(record, "successor_values_used", bool) is not False
        ):
            raise ValueError("validated curvature is not source-bound")
        backend_metadata = json.loads(scalar(record, "metadata_json", str))
        arrays = validated_taylor_remainder(
            record["energy_curvature_abs_upper"],
            record["endpoint_contrast_norm_upper"],
            record["native_contrast_radius"],
        )
    expected_shape = (
        REGISTRY["maps_per_full_registry"],
        REGISTRY["pairs_per_map"],
    )
    if any(value.shape != expected_shape for value in arrays.values()):
        raise ValueError("validated curvature omits registered pairs")
    if (
        not isinstance(backend_metadata, dict)
        or backend_metadata.get("validated") is not True
        or backend_metadata.get("successor_checkpoint_opened") is not False
        or not isinstance(backend_metadata.get("backend_name"), str)
        or not backend_metadata["backend_name"]
        or not isinstance(backend_metadata.get("backend_version"), str)
        or not backend_metadata["backend_version"]
        or not isinstance(backend_metadata.get("backend_manifest_sha256"), str)
        or len(backend_metadata["backend_manifest_sha256"]) != 64
        or not isinstance(backend_metadata.get("proof_obligations"), list)
        or not backend_metadata["proof_obligations"]
    ):
        raise ValueError("validated curvature backend metadata is incomplete")
    metadata = {
        "backend_name": backend_metadata["backend_name"],
        "backend_version": backend_metadata["backend_version"],
        "backend_manifest_sha256": backend_metadata[
            "backend_manifest_sha256"
        ],
        "proof_obligations": backend_metadata["proof_obligations"],
        "validated": True,
        "successor_checkpoint_opened": False,
        "created_at_ns": max(
            time.time_ns(), int(source_metadata["created_at_ns"]) + 1
        ),
        "validated_curvature_path": str(input_path),
        "validated_curvature_sha256": sha256_path(input_path),
    }
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(OUTPUT_SCHEMA),
        campaign_id=np.asarray(CAMPAIGN_ID),
        parent_source_sha256=np.asarray(source_digest),
        source_parameter_sha256=np.asarray(
            source_metadata["source_parameter_sha256"]
        ),
        successor_values_used=np.asarray(False),
        validation_method=np.asarray("directed-interval-taylor-v1"),
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )),
        **arrays,
    )
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "sha256": sha256_path(arguments.output),
        "pair_count": int(np.prod(expected_shape)),
        "successor_values_used": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
