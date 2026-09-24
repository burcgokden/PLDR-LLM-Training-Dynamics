#!/usr/bin/env python3
"""Analyze bounded-gate shape collapse and renewal blocks for revision 46.

This is a finite-record analyzer.  It verifies exact factorization consequences
and reports orbitwise block gains, but it never promotes a finite timecourse to
an asymptotic conclusion.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_BLOCK_SPANS = (16, 64, 256, 512, 1024, 2048, 4096)
CHECKPOINT_UPDATES = (0, 256, 512, 1024, 2048, 4096, 8192, 9000)
FLOAT32_RELATIVE_TOLERANCE = 1.0e-5


def _summary(values: np.ndarray) -> dict[str, float]:
    value = np.asarray(values, dtype=np.float64)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError("summary needs a nonempty finite array")
    return {
        "minimum": float(np.min(value)),
        "median": float(np.median(value)),
        "p90": float(np.quantile(value, 0.90)),
        "maximum": float(np.max(value)),
    }


def _positive_ratio(endpoint: np.ndarray, source: np.ndarray) -> np.ndarray:
    if endpoint.shape != source.shape:
        raise ValueError("ratio endpoints have different shapes")
    if np.any(endpoint <= 0.0) or np.any(source <= 0.0):
        raise ValueError("rev46 log-ratio analysis requires positive values")
    return endpoint / source


def _block_statistics(
    energy: np.ndarray,
    updates: np.ndarray,
    span: int,
) -> dict[str, Any]:
    index = {int(update): position for position, update in enumerate(updates)}
    pairs = [
        (position, index[int(update) + span])
        for position, update in enumerate(updates)
        if int(update) + span in index
    ]
    if not pairs:
        raise ValueError(f"no exact block endpoints for span {span}")
    ratios = np.stack([
        _positive_ratio(energy[right], energy[left])
        for left, right in pairs
    ])
    all_map_window = np.all(ratios < 1.0, axis=1)
    return {
        "span_updates": int(span),
        "window_count": len(pairs),
        "map_window_count": int(ratios.size),
        "contracting_map_window_fraction": float(np.mean(ratios < 1.0)),
        "all_map_contracting_window_fraction": float(np.mean(all_map_window)),
        "maximum_contracting_map_count": int(np.max(np.sum(ratios < 1.0, axis=1))),
        "minimum_contracting_map_count": int(np.min(np.sum(ratios < 1.0, axis=1))),
        "gain": _summary(ratios),
    }


def _reopening_statistics(
    energy: np.ndarray,
    updates: np.ndarray,
    span: int,
) -> dict[str, Any]:
    """Summarize observed positive variation on fixed nonoverlapping blocks."""

    index = {int(update): position for position, update in enumerate(updates)}
    boundaries = [
        int(update) for update in updates
        if int(update) % span == 0 and int(update) + span in index
    ]
    if not boundaries:
        raise ValueError(f"no nonoverlapping reopening blocks for span {span}")
    initial = energy[0]
    reopening = []
    envelope_slack = []
    endpoint = []
    maximum = []
    for update in boundaries:
        left = index[update]
        right = index[update + span]
        path = energy[left:right + 1]
        positive_variation = np.sum(
            np.maximum(np.diff(path, axis=0), 0.0), axis=0
        )
        envelope = energy[left] + positive_variation
        path_maximum = np.max(path, axis=0)
        reopening.append(positive_variation / initial)
        envelope_slack.append((envelope - path_maximum) / initial)
        endpoint.append(energy[right] / initial)
        maximum.append(path_maximum / initial)
    reopening_array = np.stack(reopening)
    slack_array = np.stack(envelope_slack)
    endpoint_array = np.stack(endpoint)
    maximum_array = np.stack(maximum)
    return {
        "span_updates": int(span),
        "block_count": len(boundaries),
        "observed_snapshot_reopening_over_initial": _summary(reopening_array),
        "endpoint_energy_over_initial": _summary(endpoint_array),
        "maximum_energy_over_initial": _summary(maximum_array),
        "minimum_reopening_envelope_slack_over_initial": float(
            np.min(slack_array)
        ),
        "snapshot_cadence_limited": True,
    }


def _checkpoint_statistics(
    energy: np.ndarray,
    shape: np.ndarray,
    effective_gate_squared: np.ndarray,
    updates: np.ndarray,
) -> list[dict[str, Any]]:
    index = {int(update): position for position, update in enumerate(updates)}
    result = []
    for update in CHECKPOINT_UPDATES:
        if update not in index:
            continue
        position = index[update]
        result.append({
            "update": update,
            "physical_over_initial": _summary(energy[position] / energy[0]),
            "shape_over_initial": _summary(shape[position] / shape[0]),
            "effective_gate_over_initial": _summary(
                effective_gate_squared[position] / effective_gate_squared[0]
            ),
        })
    return result


def analyze_timecourse(
    path: str | Path,
    *,
    block_spans: tuple[int, ...] = DEFAULT_BLOCK_SPANS,
) -> dict[str, Any]:
    """Analyze one complete source-resolved timecourse."""

    input_path = Path(path).resolve()
    with input_path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not value.get("technical_valid"):
        raise ValueError("timecourse is not technically valid")

    updates = np.asarray(value["updates"], dtype=np.int64)
    records = value["map_records"]
    if not updates.size or len(records) % updates.size:
        raise ValueError("timecourse map records do not form a rectangular array")
    map_count = len(records) // updates.size
    identities = [record["map_id"] for record in records[:map_count]]
    layers = np.asarray(
        [record["layer"] for record in records[:map_count]], dtype=np.int64
    )
    if any(
        record["map_id"] != identities[index % map_count]
        or int(record["update"]) != int(updates[index // map_count])
        for index, record in enumerate(records)
    ):
        raise ValueError("timecourse record ordering changed")

    shape = np.fromiter(
        (
            sum(record["normalized_shape_coordinate_energy"])
            for record in records
        ),
        dtype=np.float64,
        count=len(records),
    ).reshape(updates.size, map_count)
    physical = np.fromiter(
        (record["energy"] for record in records),
        dtype=np.float64,
        count=len(records),
    ).reshape(updates.size, map_count)
    coordinate_sum = np.fromiter(
        (sum(record["gate_shape_coordinate_energy"]) for record in records),
        dtype=np.float64,
        count=len(records),
    ).reshape(updates.size, map_count)
    plga_ratio = np.fromiter(
        (record["plga_quotient_ratio"] for record in records),
        dtype=np.float64,
        count=len(records),
    ).reshape(updates.size, map_count)
    plga_force = np.fromiter(
        (record["plga_force"] for record in records),
        dtype=np.float64,
        count=len(records),
    ).reshape(updates.size, map_count)
    gates = np.asarray(
        [record["layer_final_gate"] for record in value["gate_records"]],
        dtype=np.float64,
    )
    initial_shape_coordinates = np.asarray([
        record["normalized_shape_coordinate_energy"]
        for record in records[:map_count]
    ], dtype=np.float64)
    terminal_shape_coordinates = np.asarray([
        record["normalized_shape_coordinate_energy"]
        for record in records[-map_count:]
    ], dtype=np.float64)
    if (
        shape.shape != physical.shape
        or gates.shape[0] != updates.size
        or gates.ndim != 3
        or np.any(shape <= 0.0)
        or np.any(physical <= 0.0)
        or not all(np.isfinite(item).all() for item in (
            shape, physical, gates, plga_ratio, plga_force
        ))
    ):
        raise ValueError("timecourse arrays are invalid for rev46 analysis")

    absolute_gates = np.abs(gates)
    layer_lower = np.min(absolute_gates, axis=2) ** 2
    layer_upper = np.max(absolute_gates, axis=2) ** 2
    lower = shape * layer_lower[:, layers]
    upper = shape * layer_upper[:, layers]
    scale = np.maximum(physical, 1.0)
    lower_violation = np.maximum(lower - physical, 0.0) / scale
    upper_violation = np.maximum(physical - upper, 0.0) / scale
    factorization_residual = np.abs(coordinate_sum - physical) / scale

    physical_gain = _positive_ratio(physical[-1], physical[0])
    shape_gain = _positive_ratio(shape[-1], shape[0])
    effective_gate_squared = physical / shape
    effective_gate_gain = _positive_ratio(
        effective_gate_squared[-1], effective_gate_squared[0]
    )
    initial_occupancy = initial_shape_coordinates / shape[0, :, None]
    terminal_occupancy = terminal_shape_coordinates / shape[-1, :, None]
    initial_gate_squared = gates[0, layers] ** 2
    terminal_gate_squared = gates[-1, layers] ** 2
    initial_effective_gate = np.sum(
        initial_occupancy * initial_gate_squared, axis=1
    )
    gate_only_endpoint = np.sum(
        initial_occupancy * terminal_gate_squared, axis=1
    )
    terminal_effective_gate = np.sum(
        terminal_occupancy * terminal_gate_squared, axis=1
    )
    gate_only_gain = _positive_ratio(
        gate_only_endpoint, initial_effective_gate
    )
    alignment_gain = _positive_ratio(
        terminal_effective_gate, gate_only_endpoint
    )
    log_physical = np.log(physical_gain)
    log_shape = np.log(shape_gain)
    log_gate = np.log(effective_gate_gain)
    log_residual = log_physical - log_shape - log_gate
    three_channel_log_residual = (
        log_physical
        - log_shape
        - np.log(gate_only_gain)
        - np.log(alignment_gain)
    )

    initial_relative = physical / physical[0]
    plga_response = plga_ratio * np.sqrt(physical)
    plga_response_gain = _positive_ratio(
        plga_response[-1], plga_response[0]
    )
    plga_force_gain = _positive_ratio(plga_force[-1], plga_force[0])
    blocks = [
        _block_statistics(physical, updates, span)
        for span in block_spans
    ]
    reopening = [
        _reopening_statistics(physical, updates, span)
        for span in (256, 512, 1024)
    ]
    result = {
        "schema_version": "pldr-rev46-orbitwise-analysis-v1",
        "input_path": str(input_path),
        "trajectory": str(value["trajectory"]),
        "seed": int(value["seed"]),
        "role": str(value["role"]),
        "finite_horizon_only": True,
        "float32_relative_tolerance": FLOAT32_RELATIVE_TOLERANCE,
        "update_count": int(updates.size),
        "map_count": int(map_count),
        "terminal_update": int(updates[-1]),
        "gate_absolute": _summary(absolute_gates),
        "gate_condition_number": float(
            np.max(absolute_gates) / np.min(absolute_gates)
        ),
        "maximum_relative_factorization_residual": float(
            np.max(factorization_residual)
        ),
        "maximum_relative_gate_lower_bound_violation": float(
            np.max(lower_violation)
        ),
        "maximum_relative_gate_upper_bound_violation": float(
            np.max(upper_violation)
        ),
        "terminal_physical_gain": _summary(physical_gain),
        "terminal_shape_gain": _summary(shape_gain),
        "terminal_effective_gate_gain": _summary(effective_gate_gain),
        "terminal_gate_only_gain": _summary(gate_only_gain),
        "terminal_shape_gate_alignment_gain": _summary(alignment_gain),
        "terminal_log_physical_gain": _summary(log_physical),
        "terminal_log_shape_gain": _summary(log_shape),
        "terminal_log_effective_gate_gain": _summary(log_gate),
        "maximum_abs_terminal_log_decomposition_residual": float(
            np.max(np.abs(log_residual))
        ),
        "maximum_abs_terminal_three_channel_log_residual": float(
            np.max(np.abs(three_channel_log_residual))
        ),
        "terminal_shape_contracted_map_count": int(np.sum(shape_gain < 1.0)),
        "terminal_physical_contracted_map_count": int(
            np.sum(physical_gain < 1.0)
        ),
        "terminal_gate_helped_map_count": int(
            np.sum(effective_gate_gain < 1.0)
        ),
        "terminal_gate_only_contracted_map_count": int(
            np.sum(gate_only_gain < 1.0)
        ),
        "terminal_alignment_helped_map_count": int(
            np.sum(alignment_gain < 1.0)
        ),
        "terminal_plga_response_gain": _summary(plga_response_gain),
        "terminal_plga_force_gain": _summary(plga_force_gain),
        "terminal_plga_response_contracted_map_count": int(
            np.sum(plga_response_gain < 1.0)
        ),
        "terminal_plga_force_contracted_map_count": int(
            np.sum(plga_force_gain < 1.0)
        ),
        "maximum_plga_quotient_ratio": float(np.max(plga_ratio)),
        "maximum_energy_relative_to_initial": _summary(
            np.max(initial_relative, axis=0)
        ),
        "checkpoints": _checkpoint_statistics(
            physical, shape, effective_gate_squared, updates
        ),
        "block_statistics": blocks,
        "reopening_statistics": reopening,
        "technical_valid": bool(
            np.max(factorization_residual) <= FLOAT32_RELATIVE_TOLERANCE
            and np.max(lower_violation) <= FLOAT32_RELATIVE_TOLERANCE
            and np.max(upper_violation) <= FLOAT32_RELATIVE_TOLERANCE
            and np.max(np.abs(log_residual)) <= 1.0e-12
            and np.max(np.abs(three_channel_log_residual))
            <= FLOAT32_RELATIVE_TOLERANCE
        ),
    }
    del (
        value, records, shape, physical, coordinate_sum, gates,
        plga_ratio, plga_force, plga_response,
    )
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timecourses", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    reports = [analyze_timecourse(path) for path in arguments.timecourses]
    result = {
        "schema_version": "pldr-rev46-orbitwise-campaign-analysis-v1",
        "finite_horizon_only": True,
        "trajectory_count": len(reports),
        "trajectories": reports,
        "technical_valid": all(report["technical_valid"] for report in reports),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({
        "output": str(arguments.output.resolve()),
        "technical_valid": result["technical_valid"],
        "trajectory_count": result["trajectory_count"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
