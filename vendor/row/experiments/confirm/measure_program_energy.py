#!/usr/bin/env python3
"""Build owned energy primitives from streamed row-stack measurements.

The live training adapter writes three vectors to one NPZ artifact: the
complete source stack, complete target stack, and the matrix-free JVP along
the implemented update. This constructor derives all bookkeeping, file
bindings, completeness coordinates, and primitive error radii. It never
accepts a comparison operator, contraction factor, forcing vector, direct
bound, target radius, or verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from artifact_binding import sha256_path  # noqa: E402
from full_stack import validate_registry  # noqa: E402
from program_energy import (  # noqa: E402
    SCHEMA_VERSION,
    runtime_reduction_bound,
)


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _positive(value, name, *, allow_zero=True):
    value = float(value)
    if not np.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def build_primitive_bundle(
        *, edge_id, source_checkpoint, target_checkpoint, measurement_npz,
        registry_json, metric_json, error_ingredients_json,
        construction_partition, validation_partition):
    registry = _load_json(registry_json)
    validate_registry(registry)
    metrics = _load_json(metric_json)
    if set(metrics) != {"schema_version", "metric_blocks"} or (
        metrics["schema_version"] != "pldr-program-energy-metrics-v1"
    ):
        raise ValueError("metric artifact has missing or unknown fields")
    ingredients = _load_json(error_ingredients_json)
    required_ingredients = {
        "schema_version", "taylor_directional_second_upper",
        "row_cover_motion_upper", "metric_conversion_energy_upper",
        "exogenous_input_upper", "operation_count", "derivations",
    }
    if set(ingredients) != required_ingredients or (
        ingredients["schema_version"]
        != "pldr-program-energy-error-ingredients-v1"
    ):
        raise ValueError("error-ingredient artifact has missing or unknown fields")
    if not isinstance(ingredients["operation_count"], int) or (
        ingredients["operation_count"] < 1
    ):
        raise ValueError("operation_count must be a positive integer")
    if not isinstance(ingredients["derivations"], dict) or set(
        ingredients["derivations"]
    ) != {"taylor", "cover", "metric", "exogenous"}:
        raise ValueError("every error ingredient needs an owned derivation")

    with np.load(measurement_npz, allow_pickle=False) as measured:
        if set(measured.files) != {
            "source_stack", "target_stack", "actual_update_jvp",
        }:
            raise ValueError("measurement NPZ has missing or unknown arrays")
        source = np.asarray(measured["source_stack"], dtype=float)
        target = np.asarray(measured["target_stack"], dtype=float)
        jvp = np.asarray(measured["actual_update_jvp"], dtype=float)
    dimension = registry["dimension"]
    if any(array.shape != (dimension,) for array in (source, target, jvp)):
        raise ValueError("measurement vectors disagree with the registry")
    if any(not np.isfinite(array).all() for array in (source, target, jvp)):
        raise ValueError("measurement vectors must be finite")

    roundoff = runtime_reduction_bound(
        np.concatenate((source, target, jvp)),
        ingredients["operation_count"],
    )
    taylor_upper = (
        0.5 * _positive(
            ingredients["taylor_directional_second_upper"],
            "taylor_directional_second_upper",
        ) + roundoff
    )
    remainder_observed = float(np.linalg.norm(target - source - jvp))
    zero = {"observed_norm": 0.0}
    primitive_errors = {
        "taylor_remainder": {
            "observed_norm": remainder_observed,
            "certified_upper": taylor_upper,
            "derivation": ingredients["derivations"]["taylor"],
        },
        "row_cover_motion": {
            **zero,
            "certified_upper": _positive(
                ingredients["row_cover_motion_upper"],
                "row_cover_motion_upper",
            ),
            "derivation": ingredients["derivations"]["cover"],
        },
        "metric_conversion": {
            **zero,
            "certified_upper": _positive(
                ingredients["metric_conversion_energy_upper"],
                "metric_conversion_energy_upper",
            ),
            "derivation": ingredients["derivations"]["metric"],
        },
        "runtime_roundoff": {
            **zero,
            "certified_upper": roundoff,
            "derivation": "binary64_operation_count_and_absolute_input_sum",
        },
        "exogenous_input": {
            **zero,
            "certified_upper": _positive(
                ingredients["exogenous_input_upper"],
                "exogenous_input_upper",
            ),
            "derivation": ingredients["derivations"]["exogenous"],
        },
    }
    completeness = [
        [block["layer"], block["cell_id"], output, input_index]
        for block in registry["blocks"]
        for output in range(block["width"])
        for input_index in range(block["width"])
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "edge_id": edge_id,
        "source_state_sha256": sha256_path(source_checkpoint),
        "target_state_sha256": sha256_path(target_checkpoint),
        "successor_owner_sha256": sha256_path(Path(__file__)),
        "registry": registry,
        "source_stack": source.tolist(),
        "target_stack": target.tolist(),
        "actual_update_jvp": jvp.tolist(),
        "metric_blocks": metrics["metric_blocks"],
        "primitive_errors": primitive_errors,
        "block_completeness": completeness,
        "construction_partition_sha256": sha256_path(construction_partition),
        "validation_partition_sha256": sha256_path(validation_partition),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-id", required=True)
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--measurement-npz", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--error-ingredients", required=True)
    parser.add_argument("--construction-partition", required=True)
    parser.add_argument("--validation-partition", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = build_primitive_bundle(
        edge_id=arguments.edge_id,
        source_checkpoint=arguments.source_checkpoint,
        target_checkpoint=arguments.target_checkpoint,
        measurement_npz=arguments.measurement_npz,
        registry_json=arguments.registry,
        metric_json=arguments.metrics,
        error_ingredients_json=arguments.error_ingredients,
        construction_partition=arguments.construction_partition,
        validation_partition=arguments.validation_partition,
    )
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
