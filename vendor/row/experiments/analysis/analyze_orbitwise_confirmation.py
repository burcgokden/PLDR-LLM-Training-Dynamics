#!/usr/bin/env python3
"""Analyze full gate-shape channels and dense absolute reopening windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.orbitwise_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    CAMPAIGN_ID,
    DENSE_CENTERS,
    DENSE_HALF_WIDTH,
    INTERVENTION_UPDATES,
    REGISTRY,
    TOLERANCES,
    TRAJECTORIES,
    dense_snapshot_updates,
    full_snapshot_updates,
    source_edges,
)
from confirm.resource_executor import write_json_atomic  # noqa: E402
from confirm.strict_schema import validate  # noqa: E402


DEFAULT_PROTOCOL = (
    ROOT / "experiments" / "protocols" / "orbitwise_confirmation"
)


def _strict_line(line: str) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant {value}")

    result = json.loads(
        line,
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(result, dict):
        raise ValueError("trainer log row is not an object")
    return result


def _schema(protocol: Path, name: str) -> dict[str, Any]:
    value = json.loads((protocol / name).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"schema {name} is not an object")
    return value


def _load_points(
    log_path: Path,
    field: str,
    schema: dict[str, Any],
    expected: list[int],
) -> tuple[list[dict[str, Any]], str]:
    points: dict[int, dict[str, Any]] = {}
    run_ids: set[str] = set()
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = _strict_line(line)
            point = row.get(field)
            if point is None:
                continue
            validate(point, schema)
            step = int(point["step"])
            if row.get("step") != step:
                raise ValueError("trainer and orbitwise point clocks disagree")
            if step in points:
                raise ValueError(f"duplicate orbitwise point at update {step}")
            points[step] = point
            run_ids.add(str(row.get("run_id", "")))
    if sorted(points) != expected:
        missing = sorted(set(expected) - set(points))
        extra = sorted(set(points) - set(expected))
        raise ValueError(
            f"orbitwise cadence differs; missing={missing}, extra={extra}"
        )
    if len(run_ids) != 1 or "" in run_ids:
        raise ValueError("orbitwise points do not have one resolved run id")
    return [points[step] for step in expected], run_ids.pop()


def _summary(value: np.ndarray) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("summary needs a nonempty finite array")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _source_gain(
    endpoint: np.ndarray,
    source: np.ndarray,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if (
        endpoint.shape != source.shape
        or np.any(source < 0.0)
        or np.any(endpoint < 0.0)
    ):
        raise ValueError(f"{name} needs shape-matched nonnegative values")
    defined = source > 0.0
    result = np.divide(
        endpoint,
        source,
        out=np.zeros_like(endpoint),
        where=defined,
    )
    if np.any(result < 0.0) or not np.isfinite(result).all():
        raise ValueError(f"{name} produced an invalid gain")
    return result, defined


def _defined_summary(
    value: np.ndarray,
    defined: np.ndarray,
) -> dict[str, float]:
    if np.any(defined):
        return _summary(value[defined])
    return _summary(np.asarray([0.0], dtype=np.float64))


def _dense_blocks(
    points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    updates = dense_snapshot_updates()
    energy = np.asarray([point["energy"] for point in points], dtype=np.float64)
    index = {update: position for position, update in enumerate(updates)}
    blocks = []
    maximum_violation = 0.0
    for center in DENSE_CENTERS:
        left = center - DENSE_HALF_WIDTH
        right = center + DENSE_HALF_WIDTH
        path = energy[index[left]:index[right] + 1]
        if path.shape != (65, REGISTRY["sentinel_map_count"]):
            raise ValueError("dense block has an unexpected shape")
        anchor = path[0]
        reopening = np.sum(np.maximum(np.diff(path, axis=0), 0.0), axis=0)
        block_maximum = np.max(path, axis=0)
        raw_slack = anchor + reopening - block_maximum
        violation = np.maximum(-raw_slack, 0.0)
        maximum_violation = max(maximum_violation, float(np.max(violation)))
        blocks.append({
            "center": int(center),
            "edge_count": 64,
            "map_count": REGISTRY["sentinel_map_count"],
            "anchor_energy": _summary(anchor),
            "reopening_budget": _summary(reopening),
            "block_maximum": _summary(block_maximum),
            "envelope_slack": _summary(np.maximum(raw_slack, 0.0)),
        })
    return blocks, maximum_violation


def analyze_trajectory(
    log_path: Path,
    *,
    trajectory: str,
    role: str,
    seed: int,
    protocol: Path,
) -> dict[str, Any]:
    full, run_id = _load_points(
        log_path,
        "orbitwise_full_timepoint",
        _schema(protocol, "full_timepoint.schema.json"),
        full_snapshot_updates(),
    )
    dense, dense_run_id = _load_points(
        log_path,
        "orbitwise_dense_timepoint",
        _schema(protocol, "dense_timepoint.schema.json"),
        dense_snapshot_updates(),
    )
    if run_id != trajectory or dense_run_id != trajectory:
        raise ValueError("declared trajectory differs from native run id")

    physical = np.asarray(
        [point["energy"] for point in full], dtype=np.float64
    )
    shape_coordinates = np.asarray(
        [point["normalized_shape_coordinate_energy"] for point in full],
        dtype=np.float64,
    )
    shape = np.sum(shape_coordinates, axis=2)
    gates = np.asarray(
        [point["layer_final_gate"] for point in full], dtype=np.float64
    )
    response = np.asarray(
        [point["plga_absolute_response"] for point in full], dtype=np.float64
    )
    force = np.asarray(
        [point["plga_force"] for point in full], dtype=np.float64
    )
    factorization = np.asarray([
        point["coordinate_factorization_relative_residual"] for point in full
    ], dtype=np.float64)
    layers = np.tile(
        np.repeat(np.asarray(REGISTRY["layers"]), len(REGISTRY["heads"])),
        REGISTRY["context_count"],
    )
    if (
        physical.shape != (len(full_snapshot_updates()), REGISTRY["map_count"])
        or shape_coordinates.shape
        != (
            len(full_snapshot_updates()),
            REGISTRY["map_count"],
            REGISTRY["feature_count"],
        )
        or gates.shape
        != (
            len(full_snapshot_updates()),
            len(REGISTRY["layers"]),
            REGISTRY["feature_count"],
        )
        or not all(np.isfinite(value).all() for value in (
            physical, shape_coordinates, gates, response, force, factorization
        ))
    ):
        raise ValueError("orbitwise full arrays are malformed or nonfinite")

    physical_gain, physical_defined = _source_gain(
        physical[-1], physical[0], "physical"
    )
    shape_gain, shape_defined = _source_gain(
        shape[-1], shape[0], "shape"
    )
    initial_occupancy = np.divide(
        shape_coordinates[0],
        shape[0, :, None],
        out=np.zeros_like(shape_coordinates[0]),
        where=shape[0, :, None] > 0.0,
    )
    terminal_occupancy = np.divide(
        shape_coordinates[-1],
        shape[-1, :, None],
        out=np.zeros_like(shape_coordinates[-1]),
        where=shape[-1, :, None] > 0.0,
    )
    initial_gate_squared = gates[0, layers] ** 2
    terminal_gate_squared = gates[-1, layers] ** 2
    initial_effective = np.sum(
        initial_occupancy * initial_gate_squared, axis=1
    )
    counterfactual = np.sum(
        initial_occupancy * terminal_gate_squared, axis=1
    )
    terminal_effective = np.sum(
        terminal_occupancy * terminal_gate_squared, axis=1
    )
    gate_only_gain, gate_defined = _source_gain(
        counterfactual, initial_effective, "gate-only"
    )
    alignment_gain, alignment_defined = _source_gain(
        terminal_effective, counterfactual, "alignment"
    )
    identity_defined = (
        physical_defined
        & shape_defined
        & gate_defined
        & alignment_defined
    )
    positive = identity_defined & (
        (physical_gain > 0.0)
        & (shape_gain > 0.0)
        & (gate_only_gain > 0.0)
        & (alignment_gain > 0.0)
    )
    log_residual = np.zeros_like(physical_gain)
    log_residual[positive] = np.abs(
        np.log(physical_gain[positive])
        - np.log(shape_gain[positive])
        - np.log(gate_only_gain[positive])
        - np.log(alignment_gain[positive])
    )
    gain_residual = np.abs(
        physical_gain - shape_gain * gate_only_gain * alignment_gain
    )
    maximum_gain_residual = (
        float(np.max(gain_residual[identity_defined]))
        if np.any(identity_defined) else 0.0
    )
    if np.any(identity_defined) and not np.allclose(
        physical_gain[identity_defined],
        (
            shape_gain
            * gate_only_gain
            * alignment_gain
        )[identity_defined],
        rtol=TOLERANCES["float32_factorization_relative"],
        atol=TOLERANCES["float64_identity_relative"],
    ):
        raise ValueError("three-channel endpoint identity did not replay")

    blocks, envelope_violation = _dense_blocks(dense)
    maximum_factorization = float(np.max(factorization))
    maximum_log_residual = float(np.max(log_residual))
    technical_valid = bool(
        maximum_factorization
        <= TOLERANCES["float32_factorization_relative"]
        and maximum_gain_residual
        <= TOLERANCES["float32_factorization_relative"]
        and maximum_log_residual
        <= TOLERANCES["float32_factorization_relative"]
        and envelope_violation
        <= TOLERANCES["float64_identity_relative"]
    )
    return {
        "trajectory": trajectory,
        "seed": int(seed),
        "role": role,
        "full_snapshot_count": len(full),
        "dense_snapshot_count": len(dense),
        "map_count": REGISTRY["map_count"],
        "terminal_update": int(full_snapshot_updates()[-1]),
        "terminal_physical_gain": _defined_summary(
            physical_gain, physical_defined
        ),
        "terminal_shape_gain": _defined_summary(shape_gain, shape_defined),
        "terminal_gate_only_gain": _defined_summary(
            gate_only_gain, gate_defined
        ),
        "terminal_alignment_gain": _defined_summary(
            alignment_gain, alignment_defined
        ),
        "gain_defined_map_count": int(np.sum(identity_defined)),
        "log_gain_defined_map_count": int(np.sum(positive)),
        "zero_initial_physical_energy_count": int(
            np.sum(~physical_defined)
        ),
        "initial_plga_absolute_response": _summary(response[0]),
        "terminal_plga_absolute_response": _summary(response[-1]),
        "initial_plga_force": _summary(force[0]),
        "terminal_plga_force": _summary(force[-1]),
        "maximum_factorization_relative_residual": maximum_factorization,
        "maximum_three_channel_gain_residual": maximum_gain_residual,
        "maximum_three_channel_log_residual": maximum_log_residual,
        "maximum_reopening_envelope_violation": envelope_violation,
        "dense_blocks": blocks,
        "technical_valid": technical_valid,
    }


def _trajectory_argument(value: str) -> tuple[str, str, int, Path]:
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "trajectory must be NAME:ROLE:SEED:LOG"
        )
    name, role, seed, path = parts
    if role not in {"construction", "transfer"}:
        raise argparse.ArgumentTypeError("role must be construction or transfer")
    try:
        parsed_seed = int(seed)
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed must be an integer") from error
    return name, role, parsed_seed, Path(path)


def _load_artifact(
    path: Path,
    schema: dict[str, Any],
) -> dict[str, Any]:
    value = _strict_line(path.read_text(encoding="utf-8"))
    validate(value, schema)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory", action="append", type=_trajectory_argument, required=True
    )
    parser.add_argument("--source-links", nargs="*", type=Path, default=[])
    parser.add_argument("--gate-interventions", nargs="*", type=Path, default=[])
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-valid", action="store_true")
    arguments = parser.parse_args()
    if len(arguments.trajectory) != 3:
        raise ValueError("complete orbitwise analysis needs three trajectories")
    declared = [
        (name, role, seed)
        for name, role, seed, _path in arguments.trajectory
    ]
    expected_trajectories = [
        (str(item["name"]), str(item["role"]), int(item["seed"]))
        for item in TRAJECTORIES
    ]
    if declared != expected_trajectories:
        raise ValueError("trajectory identities differ from the frozen design")
    trajectory_paths = [
        path.resolve() for _name, _role, _seed, path in arguments.trajectory
    ]
    if len(set(trajectory_paths)) != len(trajectory_paths):
        raise ValueError("trajectory logs must be distinct")
    reports = [
        analyze_trajectory(
            path.resolve(),
            trajectory=name,
            role=role,
            seed=seed,
            protocol=arguments.protocol_dir.resolve(),
        )
        for name, role, seed, path in arguments.trajectory
    ]
    source_paths = [path.resolve() for path in arguments.source_links]
    gate_paths = [path.resolve() for path in arguments.gate_interventions]
    if len(set(source_paths)) != 8 or len(set(gate_paths)) != 3:
        raise ValueError("source and gate artifact populations are incomplete")
    source_values = [
        _load_artifact(
            path,
            _schema(arguments.protocol_dir.resolve(), "source_link.schema.json"),
        )
        for path in source_paths
    ]
    if {
        (int(value["source_step"]), int(value["endpoint_step"]))
        for value in source_values
    } != set(source_edges()):
        raise ValueError("source-link edges differ from the frozen design")
    gate_values = [
        _load_artifact(
            path,
            _schema(arguments.protocol_dir.resolve(), "gate_result.schema.json"),
        )
        for path in gate_paths
    ]
    if {
        int(value["source_step"]) for value in gate_values
    } != set(INTERVENTION_UPDATES):
        raise ValueError("gate interventions differ from the frozen design")
    source_records = [
        record
        for value in source_values
        for record in value["map_records"]
    ]
    gate_records = [
        record
        for value in gate_values
        for record in value["map_records"]
    ]
    all_inputs_technical = all(
        value["technical_valid"]
        for value in [*source_values, *gate_values]
    )
    source_bound_holds = all(
        value["scientific_checks"]["source_reopening_bound_closed"]
        for value in source_values
    )
    source_minimum_slack = min(
        float(record["bound_slack"]) for record in source_records
    )
    gate_support_count = sum(
        bool(record["supports_registered_direction"])
        for record in gate_records
    )
    result = {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "finite_horizon_only": True,
        "trajectory_count": 3,
        "trajectories": reports,
        "source_links": [str(path) for path in source_paths],
        "source_link_map_record_count": len(source_records),
        "source_reopening_bound_holds": source_bound_holds,
        "source_reopening_minimum_slack": source_minimum_slack,
        "gate_interventions": [str(path) for path in gate_paths],
        "gate_intervention_map_record_count": len(gate_records),
        "gate_registered_direction_support_count": gate_support_count,
        "gate_registered_direction_support_fraction": (
            gate_support_count / len(gate_records)
        ),
        "all_input_artifacts_technical": all_inputs_technical,
        "technical_valid": bool(
            all(report["technical_valid"] for report in reports)
            and all_inputs_technical
        ),
    }
    validate(
        result,
        _schema(arguments.protocol_dir.resolve(), "analysis.schema.json"),
    )
    write_json_atomic(arguments.output, result)
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "technical_valid": result["technical_valid"],
        "trajectory_count": 3,
    }, sort_keys=True))
    if arguments.require_valid and not result["technical_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
