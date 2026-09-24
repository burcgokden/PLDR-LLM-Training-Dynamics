#!/usr/bin/env python3
"""Independent replay of a program-bound energy-edge record.

This module does not import the constructor or its decision function.  It
recomputes registry dimensions, quadratic energies, the polarization
identity, primitive-error inclusion, the derived affine step, and the
record digest from serialized primitives.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from full_stack import validate_registry


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _metric_rows(registry, rows):
    if not isinstance(rows, list) or len(rows) != registry["block_count"]:
        raise ValueError("independent replay found an incomplete metric family")
    result = []
    for block, row in zip(registry["blocks"], rows):
        if not isinstance(row, dict) or row.get("block_index") != block["block_index"]:
            raise ValueError("independent replay found a reordered metric")
        if row.get("kind") == "scaled_identity" and set(row) == {
            "block_index", "kind", "scale",
        }:
            scale = float(row["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("independent replay found an invalid scale")
            result.append(("scaled_identity", scale))
        elif row.get("kind") == "diagonal" and set(row) == {
            "block_index", "kind", "diagonal",
        }:
            diagonal = np.asarray(row["diagonal"], dtype=float)
            if (
                diagonal.shape != (block["size"],)
                or not np.isfinite(diagonal).all()
                or np.any(diagonal <= 0)
            ):
                raise ValueError("independent replay found an invalid diagonal")
            result.append(("diagonal", diagonal))
        elif row.get("kind") == "dense_tiny_fixture" and set(row) == {
            "block_index", "kind", "matrix",
        } and block["size"] <= 256:
            matrix = np.asarray(row["matrix"], dtype=float)
            if (
                matrix.shape != (block["size"], block["size"])
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.T)
                or np.linalg.eigvalsh(matrix)[0] <= 0
            ):
                raise ValueError("independent replay found an invalid dense metric")
            result.append(("dense_tiny_fixture", matrix))
        else:
            raise ValueError("independent replay found a malformed metric")
    return result


def _quadratic(metric, left, right=None):
    right = left if right is None else right
    kind, data = metric
    if kind == "scaled_identity":
        return data * float(left @ right)
    if kind == "diagonal":
        return float((left * data) @ right)
    return float(left @ data @ right)


def _energy(registry, vector, metrics):
    value = 0.0
    for block, metric in zip(registry["blocks"], metrics):
        start = block["offset"]
        stop = start + block["size"]
        piece = vector[start:stop]
        value += _quadratic(metric, piece)
    return value


def _maximum_metric_value(metric):
    kind, data = metric
    if kind == "scaled_identity":
        return float(data)
    if kind == "diagonal":
        return float(np.max(data))
    return float(np.linalg.eigvalsh(data)[-1])


def check_record(record):
    if not isinstance(record, dict):
        raise TypeError("energy record must be a JSON object")
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        raise ValueError("energy record digest does not replay")
    if record.get("schema_version") != "pldr-program-energy-edge-v1":
        raise ValueError("unknown energy-edge schema")
    registry = record["registry"]
    validate_registry(registry)
    dimension = registry["dimension"]
    source = np.asarray(record["source_stack"], dtype=float)
    target = np.asarray(record["target_stack"], dtype=float)
    jvp = np.asarray(record["actual_update_jvp"], dtype=float)
    if any(value.shape != (dimension,) for value in (source, target, jvp)):
        raise ValueError("independent replay found a stack shape mismatch")
    if any(not np.isfinite(value).all() for value in (source, target, jvp)):
        raise ValueError("independent replay found a nonfinite stack")
    metrics = _metric_rows(registry, record["metric_blocks"])
    increment = target - source
    remainder = increment - jvp
    source_energy = _energy(registry, source, metrics)
    target_energy = _energy(registry, target, metrics)
    ideal_energy = _energy(registry, source + jvp, metrics)
    increment_energy = _energy(registry, increment, metrics)
    cross = 0.0
    for block, metric in zip(registry["blocks"], metrics):
        start = block["offset"]
        stop = start + block["size"]
        cross += _quadratic(metric, source[start:stop], increment[start:stop])
    left = target_energy - source_energy
    right = 2.0 * cross + increment_energy
    identity = record["energy_identity"]
    tolerance = identity["derived_roundoff_bound"]
    comparison = record["derived_comparison"]
    serialized_source_energy = identity["source_energy"]
    source_energy_lower = identity["source_energy_lower"]
    serialized_ideal_energy = comparison["ideal_actual_jvp_energy"]
    maximum_metric = max(_maximum_metric_value(row) for row in metrics)
    metric_remainder = (
        math.sqrt(maximum_metric) * record["jvp_remainder_upper"]
    )
    energy_charge = (
        record["primitive_errors"]["metric_conversion"]["certified_upper"]
        + 2.0 * math.sqrt(serialized_ideal_energy) * metric_remainder
        + metric_remainder ** 2
    )
    if source_energy_lower > 0:
        expected_q_squared = serialized_ideal_energy / source_energy_lower
        expected_affine = energy_charge
    else:
        expected_q_squared = 0.0
        expected_affine = serialized_ideal_energy + energy_charge
    comparison_tolerance = max(
        tolerance, 1e-12 * max(
            1.0, abs(serialized_source_energy), abs(serialized_ideal_energy),
            abs(energy_charge),
        ),
    )
    checks = {
        "source_energy": abs(source_energy - identity["source_energy"]) <= tolerance,
        "source_energy_interval": (
            identity["source_energy_lower"] <= source_energy
            <= identity["source_energy"]
        ),
        "target_energy": abs(target_energy - identity["target_energy"]) <= tolerance,
        "polarization": abs(left - right) <= tolerance,
        "jvp_remainder": (
            np.linalg.norm(remainder) <= record["jvp_remainder_upper"]
        ),
        "ideal_actual_jvp_energy": abs(
            ideal_energy - serialized_ideal_energy
        ) <= comparison_tolerance,
        "maximum_metric_eigenvalue": abs(
            maximum_metric - comparison["maximum_metric_eigenvalue"]
        ) <= comparison_tolerance,
        "metric_remainder_conversion": abs(
            metric_remainder - comparison["metric_remainder_upper"]
        ) <= comparison_tolerance,
        "jvp_derived_q_squared": abs(
            expected_q_squared - comparison["q_squared"]
        ) <= comparison_tolerance,
        "closed_affine_charge": abs(
            expected_affine - comparison["affine_source"]
        ) <= comparison_tolerance,
        "negative_work_upper": (
            serialized_ideal_energy - source_energy_lower + expected_affine
            <= comparison["negative_work_upper"] + comparison_tolerance
        ),
        "primitive_errors": all(
            row["observed_norm"] <= row["certified_upper"]
            for row in record["primitive_errors"].values()
        ),
        "derived_affine_step": target_energy <= (
            comparison["q_squared"] * source_energy
            + comparison["affine_source"]
            + tolerance
        ),
        "comparison_not_caller_supplied":
            comparison["caller_supplied"] is False,
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise ValueError("independent energy replay failed: " + ", ".join(failed))
    return {
        "schema_version": "pldr-independent-energy-replay-v1",
        "edge_id": record["edge_id"],
        "checks": checks,
        "decision": "REPLAYED",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    with Path(arguments.record).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    result = check_record(record)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
