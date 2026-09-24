#!/usr/bin/env python3
"""Combine a source JVP artifact with an independently validated remainder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from analysis.analyze_source_restoring_confirmation import (  # noqa: E402
    source_prediction_summary,
    write_json_atomic,
)
from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.finite_increment_live import write_npz_atomic  # noqa: E402
from confirm.source_restoring import source_restoring_certificate  # noqa: E402
from confirm.source_restoring_live import SOURCE_SCHEMA  # noqa: E402
from confirm.source_restoring_specs import CAMPAIGN_ID, REGISTRY  # noqa: E402


REMAINDER_SCHEMA = "pldr-source-restoring-remainder-enclosure-v1"
SOURCE_FIELDS = frozenset({
    "schema_version",
    "metadata_json",
    "source_rows",
    "gate_velocity_rows",
    "shape_velocity_rows",
    "input_velocity_rows",
})
REMAINDER_FIELDS = frozenset({
    "schema_version",
    "campaign_id",
    "parent_source_sha256",
    "source_parameter_sha256",
    "successor_values_used",
    "validation_method",
    "remainder_lower",
    "remainder_upper",
    "endpoint_contrast_norm_upper",
    "native_contrast_radius",
    "metadata_json",
})


def scalar(record: object, name: str, kind: type) -> object:
    if name not in record:
        raise ValueError(f"artifact omits {name}")
    value = np.asarray(record[name])
    if value.shape != ():
        raise ValueError(f"{name} must be scalar")
    return kind(value.item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--remainder", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    arguments = parser.parse_args()
    source_path = arguments.source.resolve()
    remainder_path = arguments.remainder.resolve()
    source_digest = sha256_path(source_path)
    with np.load(source_path, allow_pickle=False) as source:
        if set(source.files) != SOURCE_FIELDS:
            raise ValueError("source artifact has missing or unknown fields")
        if scalar(source, "schema_version", str) != SOURCE_SCHEMA:
            raise ValueError("unknown source-state artifact schema")
        source_metadata = json.loads(scalar(source, "metadata_json", str))
        if (
            not isinstance(source_metadata, dict)
            or source_metadata.get("campaign_id") != CAMPAIGN_ID
            or source_metadata.get("successor_checkpoint_opened") is not False
            or source_metadata.get("successor_values_used") is not False
            or source_metadata.get("map_order")
            != "context-major-layer-major-head-major-v1"
            or source_metadata.get("registry_sha256") is None
            or source_metadata.get("source_parameter_sha256") is None
            or not isinstance(source_metadata.get("created_at_ns"), int)
        ):
            raise ValueError("source artifact is not a bound source-only record")
        arrays = {
            name: np.asarray(source[name], dtype=np.float64)
            for name in (
                "source_rows",
                "gate_velocity_rows",
                "shape_velocity_rows",
                "input_velocity_rows",
            )
        }
    expected_source_shape = (
        REGISTRY["maps_per_full_registry"],
        REGISTRY["rows_per_map"],
        64,
    )
    if any(
        value.shape != expected_source_shape or not np.isfinite(value).all()
        for value in arrays.values()
    ):
        raise ValueError("source artifact does not cover the exact map registry")
    with np.load(remainder_path, allow_pickle=False) as remainder:
        if set(remainder.files) != REMAINDER_FIELDS:
            raise ValueError("remainder artifact has missing or unknown fields")
        if scalar(remainder, "schema_version", str) != REMAINDER_SCHEMA:
            raise ValueError("unknown remainder enclosure schema")
        if scalar(remainder, "campaign_id", str) != CAMPAIGN_ID:
            raise ValueError("remainder enclosure names another campaign")
        if scalar(remainder, "parent_source_sha256", str) != source_digest:
            raise ValueError("remainder enclosure is linked to another source")
        if (
            scalar(remainder, "source_parameter_sha256", str)
            != source_metadata["source_parameter_sha256"]
            or scalar(remainder, "successor_values_used", bool) is not False
        ):
            raise ValueError("remainder enclosure is not source-parameter bound")
        method = scalar(remainder, "validation_method", str)
        if method not in {
            "directed-interval-taylor-v1",
            "operation-ordered-finite-telescope-v1",
        }:
            raise ValueError("remainder method is not an accepted validated method")
        validation_metadata = json.loads(
            scalar(remainder, "metadata_json", str)
        )
        if (
            not isinstance(validation_metadata, dict)
            or validation_metadata.get("validated") is not True
            or validation_metadata.get("successor_checkpoint_opened") is not False
            or not isinstance(validation_metadata.get("backend_name"), str)
            or not isinstance(validation_metadata.get("backend_version"), str)
            or not isinstance(validation_metadata.get("created_at_ns"), int)
        ):
            raise ValueError("remainder validation metadata is incomplete")
        remainder_arrays = {
            name: np.asarray(remainder[name], dtype=np.float64)
            for name in (
                "remainder_lower",
                "remainder_upper",
                "endpoint_contrast_norm_upper",
                "native_contrast_radius",
            )
        }
    pair_shape = (
        REGISTRY["maps_per_full_registry"],
        REGISTRY["pairs_per_map"],
    )
    if any(
        value.shape != pair_shape or not np.isfinite(value).all()
        for value in remainder_arrays.values()
    ):
        raise ValueError("remainder artifact does not cover every registered pair")
    certificate = source_restoring_certificate(
        arrays["source_rows"],
        arrays["gate_velocity_rows"],
        arrays["shape_velocity_rows"],
        arrays["input_velocity_rows"],
        **remainder_arrays,
    )
    certificate_created_at_ns = max(
        time.time_ns(), int(source_metadata["created_at_ns"]) + 1
    )
    metadata = {
        "schema_version": "pldr-source-restoring-certificate-metadata-v1",
        "campaign_id": CAMPAIGN_ID,
        "created_at_ns": certificate_created_at_ns,
        "source_path": str(source_path),
        "source_sha256": source_digest,
        "remainder_path": str(remainder_path),
        "remainder_sha256": sha256_path(remainder_path),
        "remainder_validation_method": method,
        "remainder_validation_metadata": validation_metadata,
        "source_metadata": source_metadata,
        "successor_values_used": False,
        "technical_valid": True,
    }
    write_npz_atomic(
        arguments.certificate,
        metadata_json=np.asarray(json.dumps(
            metadata, sort_keys=True, separators=(",", ":")
        )),
        **{
            key: np.asarray(value)
            for key, value in certificate.items()
            if key not in {
                "schema_version",
                "successor_values_used",
                "cross_map_pairs_formed",
                "technical_valid",
            }
        },
        schema_version=np.asarray(certificate["schema_version"]),
        successor_values_used=np.asarray(False),
        cross_map_pairs_formed=np.asarray(False),
        technical_valid=np.asarray(True),
    )
    prediction = source_prediction_summary(
        certificate, certificate_path=arguments.certificate
    )
    prediction.update({
        "created_at_ns": max(time.time_ns(), certificate_created_at_ns + 1),
        "source_path": str(source_path),
        "source_sha256": source_digest,
        "remainder_sha256": metadata["remainder_sha256"],
    })
    prediction.pop("prediction_sha256", None)
    prediction["prediction_sha256"] = digest_object(prediction)
    write_json_atomic(arguments.prediction, prediction)
    print(json.dumps({
        "certificate": str(arguments.certificate.resolve()),
        "prediction": str(arguments.prediction.resolve()),
        "outcome_counts": prediction["outcome_counts"],
        "successor_values_used": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
