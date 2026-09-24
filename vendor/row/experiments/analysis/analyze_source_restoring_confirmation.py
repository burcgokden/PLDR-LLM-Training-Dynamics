#!/usr/bin/env python3
"""Analyze source-restoring predictions and chronological native coverage."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import digest_object, sha256_path  # noqa: E402
from confirm.source_restoring import (  # noqa: E402
    CERTIFICATE_SCHEMA,
    SCIENTIFIC_OUTCOMES,
    source_multiplier_cocycle,
)
from confirm.source_restoring_specs import CAMPAIGN_ID  # noqa: E402


EDGE_SCHEMA = "pldr-source-restoring-edge-v1"
AGGREGATE_SCHEMA = "pldr-source-restoring-aggregate-v1"
PREDICTION_SCHEMA = "pldr-source-restoring-prediction-v1"


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def source_prediction_summary(
    certificate: dict[str, Any],
    *,
    certificate_path: str | Path | None = None,
) -> dict[str, Any]:
    """Freeze a compact scientific decision without consulting a successor."""

    if (
        certificate.get("schema_version") != CERTIFICATE_SCHEMA
        or certificate.get("successor_values_used") is not False
        or certificate.get("technical_valid") is not True
    ):
        raise ValueError("prediction requires a valid source-only certificate")
    outcomes = list(certificate["scientific_outcome"])
    if any(value not in SCIENTIFIC_OUTCOMES for value in outcomes):
        raise ValueError("certificate contains an invalid scientific outcome")
    counts = {value: outcomes.count(value) for value in SCIENTIFIC_OUTCOMES}
    result: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "successor_values_used": False,
        "technical_valid": True,
        "map_count": int(certificate["map_count"]),
        "pairs_per_map": int(certificate["pairs_per_map"]),
        "outcome_counts": counts,
        "per_map_outcome": outcomes,
        "source_diameter_squared": [
            float(value)
            for value in certificate["source_diameter_squared"]
        ],
        "source_diameter_squared_lower": [
            float(value)
            for value in certificate["source_diameter_squared_lower"]
        ],
        "source_diameter_squared_upper": [
            float(value)
            for value in certificate["source_diameter_squared_upper"]
        ],
        "source_upper_multiplier": [
            None if not np.isfinite(value) else float(value)
            for value in certificate["source_upper_multiplier"]
        ],
        "source_successor_diameter_squared_upper": [
            float(value)
            for value in certificate["predicted_diameter_squared_upper"]
        ],
        "minimum_source_restoring_margin": [
            float(value)
            for value in certificate["minimum_source_restoring_margin"]
        ],
    }
    if certificate_path is not None:
        path = Path(certificate_path).resolve()
        result["certificate_path"] = str(path)
        result["certificate_sha256"] = sha256_path(path)
    result["prediction_sha256"] = digest_object(result)
    return result


def validate_edge(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "step",
        "source_artifact_path",
        "source_artifact_sha256",
        "registry_sha256",
        "data_order_sha256",
        "certificate_created_at_ns",
        "prediction_created_at_ns",
        "successor_opened_at_ns",
        "source_checkpoint_path",
        "source_checkpoint_sha256",
        "native_successor_checkpoint_path",
        "native_successor_checkpoint_sha256",
        "prediction_sha256",
        "certificate_sha256",
        "technical_valid",
        "scientific_outcome",
        "source_diameter_squared",
        "source_diameter_squared_lower",
        "source_diameter_squared_upper",
        "source_upper_multiplier",
        "source_successor_diameter_squared_upper",
        "native_successor_diameter_squared",
        "native_successor_diameter_squared_lower",
        "native_successor_diameter_squared_upper",
        "certificate_coverage_fraction",
        "native_peak_memory_reserved_bytes",
    }
    if set(record) != required:
        raise ValueError("edge record has missing or unknown fields")
    if (
        record["schema_version"] != EDGE_SCHEMA
        or record["campaign_id"] != CAMPAIGN_ID
        or record["technical_valid"] is not True
        or record["certificate_coverage_fraction"] != 1.0
    ):
        raise ValueError("edge record is technically invalid")
    digest_fields = (
        "source_artifact_sha256",
        "registry_sha256",
        "data_order_sha256",
        "source_checkpoint_sha256",
        "native_successor_checkpoint_sha256",
        "prediction_sha256",
        "certificate_sha256",
    )
    timestamps = (
        record["certificate_created_at_ns"],
        record["prediction_created_at_ns"],
        record["successor_opened_at_ns"],
    )
    if (
        not isinstance(record["step"], int)
        or isinstance(record["step"], bool)
        or record["step"] < 0
        or any(not _is_sha256(record[name]) for name in digest_fields)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in timestamps
        )
        or not timestamps[0] < timestamps[1] < timestamps[2]
        or not Path(record["source_artifact_path"]).is_absolute()
        or not Path(record["source_checkpoint_path"]).is_absolute()
        or not Path(record["native_successor_checkpoint_path"]).is_absolute()
        or not isinstance(record["native_peak_memory_reserved_bytes"], int)
        or record["native_peak_memory_reserved_bytes"] < 0
    ):
        raise ValueError("edge provenance or chronology is malformed")
    outcomes = record["scientific_outcome"]
    if any(value not in SCIENTIFIC_OUTCOMES for value in outcomes):
        raise ValueError("edge record contains an invalid scientific outcome")
    source = np.asarray(record["source_diameter_squared"], dtype=np.float64)
    source_lower = np.asarray(
        record["source_diameter_squared_lower"], dtype=np.float64
    )
    source_upper = np.asarray(
        record["source_diameter_squared_upper"], dtype=np.float64
    )
    upper = np.asarray(
        record["source_successor_diameter_squared_upper"], dtype=np.float64
    )
    successor = np.asarray(
        record["native_successor_diameter_squared"], dtype=np.float64
    )
    successor_lower = np.asarray(
        record["native_successor_diameter_squared_lower"], dtype=np.float64
    )
    successor_upper = np.asarray(
        record["native_successor_diameter_squared_upper"], dtype=np.float64
    )
    if (
        source.ndim != 1
        or source_lower.shape != source.shape
        or source_upper.shape != source.shape
        or upper.shape != source.shape
        or successor.shape != source.shape
        or successor_lower.shape != source.shape
        or successor_upper.shape != source.shape
        or any(not np.isfinite(value).all() for value in (
            source, source_lower, source_upper, upper, successor,
            successor_lower, successor_upper,
        ))
        or any(np.any(value < 0.0) for value in (
            source, source_lower, source_upper, upper, successor,
            successor_lower, successor_upper,
        ))
        or np.any(source_lower > source)
        or np.any(source > source_upper)
        or np.any(successor_lower > successor)
        or np.any(successor > successor_upper)
        or len(outcomes) != len(source)
    ):
        raise ValueError("edge arrays are malformed")
    raw_multiplier = record["source_upper_multiplier"]
    if not isinstance(raw_multiplier, list) or len(raw_multiplier) != len(source):
        raise ValueError("edge multiplier array is malformed")
    multiplier = np.full_like(source, np.nan)
    for index, value in enumerate(raw_multiplier):
        if value is None:
            if source_lower[index] != 0.0:
                raise ValueError("a positive source diameter needs a multiplier")
            continue
        multiplier[index] = float(value)
        if (
            not np.isfinite(multiplier[index])
            or multiplier[index] < 0.0
            or source_lower[index] <= 0.0
        ):
            raise ValueError("a source multiplier is invalid")
        product = multiplier[index] * source_lower[index]
        scale = max(1.0, upper[index], product)
        tolerance = 128.0 * np.finfo(np.float64).eps * scale
        if product + tolerance < upper[index]:
            raise ValueError("a multiplier is not linked to its absolute bound")
    scale = np.maximum.reduce((np.ones_like(upper), upper, successor_upper))
    tolerance = 1024.0 * np.finfo(np.float64).eps * scale
    if np.any(successor_upper > upper + tolerance):
        raise ValueError("a technically valid edge violates its frozen enclosure")


def aggregate_chronological_edges(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate consecutive trainer-native edges under frozen source bounds."""

    if not records:
        raise ValueError("at least one chronological edge is required")
    ordered = sorted(records, key=lambda value: int(value["step"]))
    for record in ordered:
        validate_edge(record)
    steps = [int(record["step"]) for record in ordered]
    if steps != list(range(steps[0], steps[0] + len(steps))):
        raise ValueError("edge steps are not consecutive")
    for previous, current in zip(ordered, ordered[1:], strict=False):
        if (
            previous["native_successor_checkpoint_path"]
            != current["source_checkpoint_path"]
            or previous["native_successor_checkpoint_sha256"]
            != current["source_checkpoint_sha256"]
            or previous["registry_sha256"] != current["registry_sha256"]
            or previous["data_order_sha256"] != current["data_order_sha256"]
            or previous["successor_opened_at_ns"]
            >= current["certificate_created_at_ns"]
        ):
            raise ValueError("trainer-native checkpoint linkage is not exact")
    source_series = np.asarray(
        [record["source_diameter_squared"] for record in ordered],
        dtype=np.float64,
    )
    source_upper_series = np.asarray(
        [record["source_diameter_squared_upper"] for record in ordered],
        dtype=np.float64,
    )
    source = source_upper_series[0]
    multipliers = np.asarray(
        [
            [
                np.nan if value is None else float(value)
                for value in record["source_upper_multiplier"]
            ]
            for record in ordered
        ],
        dtype=np.float64,
    )
    absolute_upper = np.asarray(
        [record["source_successor_diameter_squared_upper"] for record in ordered],
        dtype=np.float64,
    )
    observed_nominal = np.asarray(
        [record["native_successor_diameter_squared"] for record in ordered],
        dtype=np.float64,
    )
    observed = np.asarray(
        [record["native_successor_diameter_squared_upper"] for record in ordered],
        dtype=np.float64,
    )
    if len(ordered) > 1:
        linkage_scale = np.maximum.reduce((
            np.ones_like(source_series[1:]),
            source_series[1:],
            observed_nominal[:-1],
        ))
        linkage_tolerance = 1024.0 * np.finfo(np.float64).eps * linkage_scale
        if np.any(
            np.abs(source_series[1:] - observed_nominal[:-1]) > linkage_tolerance
        ):
            raise ValueError("linked checkpoints disagree on their row-map diameter")
    bounds = np.empty((len(ordered) + 1, len(source)), dtype=np.float64)
    bounds[0] = source
    for edge in range(len(ordered)):
        defined = np.isfinite(multipliers[edge])
        product = multipliers[edge, defined] * bounds[edge, defined]
        bounds[edge + 1, defined] = np.where(
            product == 0.0,
            0.0,
            np.nextafter(product, np.full_like(product, np.inf)),
        )
        bounds[edge + 1, ~defined] = absolute_upper[edge, ~defined]
    successor_bounds = bounds[1:]
    tolerance = (
        1024.0
        * np.finfo(np.float64).eps
        * np.maximum.reduce((np.ones_like(observed), observed, successor_bounds))
    )
    prefix_covered = observed <= successor_bounds + tolerance
    product_defined = np.all(np.isfinite(multipliers), axis=0)
    final_log = np.full(len(source), np.nan, dtype=np.float64)
    if np.any(product_defined):
        cocycle = source_multiplier_cocycle(
            source[product_defined], multipliers[:, product_defined]
        )
        final_log[product_defined] = cocycle["cumulative_log_product"][-1]
    if bool(
        np.all(prefix_covered)
        and np.all(product_defined)
        and np.all(final_log < 0.0)
    ):
        outcome = "confirmed"
    else:
        outcome = "unresolved"
    result: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "technical_valid": True,
        "start_step": steps[0],
        "registry_sha256": ordered[0]["registry_sha256"],
        "data_order_sha256": ordered[0]["data_order_sha256"],
        "edge_count": len(ordered),
        "map_count": int(source.shape[0]),
        "exact_native_checkpoint_linkage": True,
        "all_source_products_defined": bool(np.all(product_defined)),
        "all_prefix_bounds_cover_native": bool(np.all(prefix_covered)),
        "minimum_prefix_coverage_fraction": float(
            np.min(np.mean(prefix_covered, axis=1))
        ),
        "final_cumulative_log_product": [
            None if not np.isfinite(value) else float(value)
            for value in final_log
        ],
        "scientific_outcome": outcome,
        "scientific_outcomes_halted_descendants": False,
        "edge_prediction_sha256": [
            record["prediction_sha256"] for record in ordered
        ],
        "edge_certificate_sha256": [
            record["certificate_sha256"] for record in ordered
        ],
    }
    result["aggregate_sha256"] = digest_object(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edges", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in arguments.edges
    ]
    result = aggregate_chronological_edges(records)
    write_json_atomic(arguments.output, result)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "technical_valid": result["technical_valid"],
        "scientific_outcome": result["scientific_outcome"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
