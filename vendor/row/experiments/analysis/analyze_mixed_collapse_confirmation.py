#!/usr/bin/env python3
"""Analyze the finite mixed-collapse campaign without asymptotic promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.mixed_collapse_specs import (  # noqa: E402
    BLOCK_ANCHORS,
    CAMPAIGN_ID,
    DENSE_CENTERS,
    DENSE_HALF_WIDTH,
    INTERVENTION_LENGTH,
    INTERVENTION_MODES,
    INTERVENTION_UPDATES,
    OPTIMIZER,
    REGISTRY,
    SOURCE_RADIUS_UPDATES,
    TERMINAL_UPDATE,
    TOLERANCES,
    TRAJECTORIES,
    dense_snapshot_updates,
    full_snapshot_updates,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _summary(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("summary input must be nonempty and finite")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
    }


def factorization_residual(timepoint: dict[str, Any]) -> np.ndarray:
    """Return the fail-capable relative residual for every individual map."""

    energy = np.asarray(timepoint["energy"], dtype=np.float64)
    shape = np.asarray(
        timepoint["normalized_shape_coordinate_energy"], dtype=np.float64
    )
    gates = np.asarray(timepoint["layer_final_gate"], dtype=np.float64)
    map_gates = np.broadcast_to(
        gates[None, :, None, :],
        (
            REGISTRY["context_count"],
            len(REGISTRY["layers"]),
            len(REGISTRY["heads"]),
            REGISTRY["feature_count"],
        ),
    ).reshape(REGISTRY["map_count"], REGISTRY["feature_count"])
    predicted = np.sum(shape * map_gates * map_gates, axis=1)
    denominator = np.maximum(energy, np.finfo(np.float64).tiny)
    return np.abs(energy - predicted) / denominator


def observed_block_diagnostics(
    steps: np.ndarray,
    energy: np.ndarray,
    anchors: tuple[int, ...] = BLOCK_ANCHORS,
) -> list[dict[str, Any]]:
    """Compute cadence-observed lower bounds for fixed growing blocks."""

    steps = np.asarray(steps, dtype=np.int64)
    energy = np.asarray(energy, dtype=np.float64)
    if energy.ndim != 2 or energy.shape[0] != len(steps):
        raise ValueError("block energy must have shape (time, map)")
    if np.any(np.diff(steps) <= 0) or np.any(energy < 0.0):
        raise ValueError("block steps and energies are invalid")
    index = {int(step): position for position, step in enumerate(steps)}
    results = []
    for left, right in zip(anchors, anchors[1:]):
        if left not in index or right not in index:
            raise ValueError(f"missing fixed block anchor {left} or {right}")
        begin, end = index[left], index[right]
        block = energy[begin:end + 1]
        anchor = block[0]
        excursion = np.maximum(np.max(block, axis=0) - anchor, 0.0)
        budget = np.sum(np.maximum(np.diff(block, axis=0), 0.0), axis=0)
        scale = np.maximum(anchor, np.finfo(np.float64).tiny)
        if np.any(excursion > budget + 1.0e-10 * np.maximum(scale, 1.0)):
            raise RuntimeError("observed excursion exceeds positive variation")
        results.append({
            "left": int(left),
            "right": int(right),
            "cadence": int(np.max(np.diff(steps[begin:end + 1]))),
            "anchor_energy": _summary(anchor),
            "endpoint_energy": _summary(block[-1]),
            "observed_excursion": _summary(excursion),
            "observed_positive_variation": _summary(budget),
            "observed_excursion_over_anchor": _summary(excursion / scale),
            "observed_positive_variation_over_anchor": _summary(budget / scale),
            "maps_with_observed_reopening": int(np.sum(budget > 0.0)),
            "interpretation": "cadence-lower-bound",
        })
    return results


def exact_dense_diagnostics(
    energy_by_step: dict[int, np.ndarray],
    centers: tuple[int, ...] = DENSE_CENTERS,
) -> list[dict[str, Any]]:
    """Compute true update-wise forward excursion in each captured window."""

    results = []
    for center in centers:
        left = max(0, center - DENSE_HALF_WIDTH)
        right = min(TERMINAL_UPDATE, center + DENSE_HALF_WIDTH)
        required = list(range(left, right + 1))
        missing = [step for step in required if step not in energy_by_step]
        if missing:
            raise ValueError(
                f"dense window {center} is missing {len(missing)} updates"
            )
        block = np.stack([energy_by_step[step] for step in required])
        anchor = block[0]
        excursion = np.maximum(np.max(block, axis=0) - anchor, 0.0)
        budget = np.sum(np.maximum(np.diff(block, axis=0), 0.0), axis=0)
        scale = np.maximum(anchor, np.finfo(np.float64).tiny)
        results.append({
            "left": int(left),
            "center": int(center),
            "right": int(right),
            "edge_count": int(right - left),
            "exact_excursion": _summary(excursion),
            "exact_positive_variation": _summary(budget),
            "exact_excursion_over_anchor": _summary(excursion / scale),
            "exact_positive_variation_over_anchor": _summary(budget / scale),
            "maps_with_reopening": int(np.sum(budget > 0.0)),
            "interpretation": "update-wise-exact-on-declared-window",
        })
    return results


def duhamel_diagnostics(
    gates: np.ndarray,
    learning_rates: np.ndarray,
    anchors: tuple[int, ...] = BLOCK_ANCHORS,
    *,
    weight_decay: float = float(OPTIMIZER["weight_decay"]),
    terminal_step: int | None = None,
) -> dict[str, Any]:
    """Replay the exact gate recurrence and absolute innovation convolution."""

    gates = np.asarray(gates, dtype=np.float64)
    learning_rates = np.asarray(learning_rates, dtype=np.float64)
    if gates.ndim != 3 or gates.shape[1:] != (
        len(REGISTRY["layers"]), REGISTRY["feature_count"]
    ):
        raise ValueError("gate history has the wrong shape")
    if len(learning_rates) != len(gates) or not np.isnan(learning_rates[0]):
        raise ValueError("learning-rate history must align with gate states")
    terminal = len(gates) - 1 if terminal_step is None else int(terminal_step)
    if terminal != len(gates) - 1:
        raise ValueError("terminal step must match the final gate state")
    recorded_anchors = tuple(
        int(step) for step in anchors if 0 <= int(step) <= terminal
    )
    previous_anchors = [step for step in recorded_anchors if step < terminal]
    reference = max(previous_anchors, default=0)
    homogeneous = gates[0].copy()
    forcing = np.zeros_like(homogeneous)
    absolute = np.zeros_like(homogeneous)
    maximum_residual = 0.0
    maximum_bound_violation = 0.0
    anchor_set = set(recorded_anchors) | {0, terminal}
    records = {
        "0": {
            "gate_abs": _summary(np.abs(gates[0])),
            "homogeneous_abs": _summary(np.abs(homogeneous)),
            "signed_forcing_abs": _summary(np.abs(forcing)),
            "absolute_innovation_convolution": _summary(absolute),
        }
    }
    for step in range(1, len(gates)):
        rate = learning_rates[step]
        if not np.isfinite(rate) or rate <= 0.0:
            raise ValueError(f"invalid applied learning rate at step {step}")
        multiplier = 1.0 - rate * weight_decay
        innovation = multiplier * gates[step - 1] - gates[step]
        homogeneous = multiplier * homogeneous
        forcing = multiplier * forcing + innovation
        absolute = abs(multiplier) * absolute + np.abs(innovation)
        residual = np.max(np.abs(gates[step] - (homogeneous - forcing)))
        violation = np.max(
            np.abs(gates[step]) - (np.abs(homogeneous) + absolute)
        )
        maximum_residual = max(maximum_residual, float(residual))
        maximum_bound_violation = max(maximum_bound_violation, float(violation))
        if step in anchor_set:
            records[str(step)] = {
                "gate_abs": _summary(np.abs(gates[step])),
                "homogeneous_abs": _summary(np.abs(homogeneous)),
                "signed_forcing_abs": _summary(np.abs(forcing)),
                "absolute_innovation_convolution": _summary(absolute),
            }
    return {
        "maximum_reconstruction_residual": maximum_residual,
        "maximum_absolute_bound_violation": maximum_bound_violation,
        "anchors": records,
        "absolute_convolution_terminal_over_previous_anchor_median": (
            records[str(terminal)]["absolute_innovation_convolution"]["median"]
            / max(
                records[str(reference)]["absolute_innovation_convolution"]["median"],
                np.finfo(np.float64).tiny,
            )
        ),
        "previous_anchor": reference,
    }


def _read_trajectory(log_path: Path) -> dict[str, Any]:
    expected_full = set(full_snapshot_updates())
    expected_dense = set(dense_snapshot_updates()) - expected_full
    gates = np.full(
        (TERMINAL_UPDATE + 1, len(REGISTRY["layers"]), REGISTRY["feature_count"]),
        np.nan,
        dtype=np.float64,
    )
    rates = np.full(TERMINAL_UPDATE + 1, np.nan, dtype=np.float64)
    full_energy: dict[int, np.ndarray] = {}
    dense_energy: dict[int, np.ndarray] = {}
    anchor_points: dict[int, dict[str, Any]] = {}
    factorization_max = 0.0
    factorization_states_above = 0
    factorization_map_values_above = 0
    threshold = TOLERANCES["construction_float32_factorization_relative"]
    run_id = None
    lineage_root = None
    resource_summary = None
    config_count = 0

    with log_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"malformed JSON at {log_path}:{line_number}"
                ) from error
            if row.get("event") == "config":
                config_count += 1
                binding = row.get("campaign_binding")
                if (
                    not isinstance(binding, dict)
                    or binding.get("campaign_id") != CAMPAIGN_ID
                ):
                    raise ValueError(
                        f"wrong campaign binding at {log_path}:{line_number}"
                    )
            run_id = run_id or row.get("run_id")
            lineage_root = lineage_root or row.get("lineage_root")
            if row.get("run_id") != run_id or row.get("lineage_root") != lineage_root:
                raise ValueError("run identity changes inside one log")
            step = int(row.get("step", -1))
            if "final_gate_state" in row:
                value = np.asarray(row["final_gate_state"], dtype=np.float64)
                if value.shape != gates.shape[1:]:
                    raise ValueError(f"gate state has wrong shape at step {step}")
                if np.isfinite(gates[step]).any():
                    raise ValueError(f"duplicate gate state at step {step}")
                gates[step] = value
                if step > 0:
                    rates[step] = float(row["learning_rate_applied"])
            for key, destination in (
                ("mixed_collapse_full_timepoint", full_energy),
                ("mixed_collapse_dense_timepoint", dense_energy),
            ):
                if key not in row:
                    continue
                point = row[key]
                if point.get("campaign_id") != CAMPAIGN_ID:
                    raise ValueError("timepoint has the wrong campaign id")
                point_step = int(point["step"])
                if point_step != step:
                    raise ValueError("timepoint and event step disagree")
                energy = np.asarray(point["energy"], dtype=np.float64)
                if energy.shape != (REGISTRY["map_count"],):
                    raise ValueError("timepoint map population changed")
                if point_step in destination:
                    raise ValueError(f"duplicate timepoint at step {point_step}")
                destination[point_step] = energy
                residual = factorization_residual(point)
                factorization_max = max(factorization_max, float(np.max(residual)))
                factorization_states_above += int(np.any(residual > threshold))
                factorization_map_values_above += int(np.sum(residual > threshold))
                if point_step in set(BLOCK_ANCHORS) | {0, 18_000, TERMINAL_UPDATE}:
                    anchor_points[point_step] = point
            if row.get("event") == "resource_summary":
                resource_summary = row
    if config_count != 1:
        raise ValueError("trajectory log must contain exactly one config row")
    if set(full_energy) != expected_full:
        raise ValueError(
            f"full timepoint schedule mismatch: missing={len(expected_full-set(full_energy))}, "
            f"extra={len(set(full_energy)-expected_full)}"
        )
    if set(dense_energy) != expected_dense:
        raise ValueError(
            f"dense timepoint schedule mismatch: missing={len(expected_dense-set(dense_energy))}, "
            f"extra={len(set(dense_energy)-expected_dense)}"
        )
    if not np.isfinite(gates).all() or np.isnan(rates[1:]).any():
        raise ValueError("gate or learning-rate history is incomplete")
    if resource_summary is None:
        raise ValueError("producer-side resource summary is absent")

    full_steps = np.asarray(sorted(full_energy), dtype=np.int64)
    energy = np.stack([full_energy[int(step)] for step in full_steps])
    dense_all = dict(full_energy)
    dense_all.update(dense_energy)
    anchor = anchor_points[BLOCK_ANCHORS[0]]
    terminal = anchor_points[TERMINAL_UPDATE]
    anchor_energy = np.asarray(anchor["energy"], dtype=np.float64)
    terminal_energy = np.asarray(terminal["energy"], dtype=np.float64)
    endpoint_gain = terminal_energy / np.maximum(
        anchor_energy, np.finfo(np.float64).tiny
    )

    coordinate = {}
    cocycle_maximum_residual = 0.0
    for step in BLOCK_ANCHORS:
        point = anchor_points[step]
        coordinate_energy = np.asarray(
            point["gate_shape_coordinate_energy"], dtype=np.float64
        )
        coordinate[str(step)] = {
            "total_energy": _summary(np.sum(coordinate_energy, axis=1)),
            "positive_coordinate_count": int(np.sum(coordinate_energy > 0.0)),
            "maximum_coordinate_energy": float(np.max(coordinate_energy)),
        }
    for left, right in zip(BLOCK_ANCHORS, BLOCK_ANCHORS[1:]):
        left_point = anchor_points[left]
        right_point = anchor_points[right]
        e0 = np.asarray(left_point["gate_shape_coordinate_energy"], dtype=np.float64)
        e1 = np.asarray(right_point["gate_shape_coordinate_energy"], dtype=np.float64)
        s0 = np.asarray(
            left_point["normalized_shape_coordinate_energy"], dtype=np.float64
        )
        s1 = np.asarray(
            right_point["normalized_shape_coordinate_energy"], dtype=np.float64
        )
        g0 = np.asarray(left_point["layer_final_gate"], dtype=np.float64)
        g1 = np.asarray(right_point["layer_final_gate"], dtype=np.float64)
        map_g0 = np.broadcast_to(
            g0[None, :, None, :], (REGISTRY["context_count"], 3, 4, 64)
        ).reshape(REGISTRY["map_count"], 64)
        map_g1 = np.broadcast_to(
            g1[None, :, None, :], (REGISTRY["context_count"], 3, 4, 64)
        ).reshape(REGISTRY["map_count"], 64)
        positive = (
            (e0 > 0.0) & (e1 > 0.0) & (s0 > 0.0) & (s1 > 0.0)
            & (np.abs(map_g0) > 0.0) & (np.abs(map_g1) > 0.0)
        )
        residual = (
            np.log(e1[positive] / e0[positive])
            - 2.0 * np.log(np.abs(map_g1[positive]) / np.abs(map_g0[positive]))
            - np.log(s1[positive] / s0[positive])
        )
        if len(residual):
            cocycle_maximum_residual = max(
                cocycle_maximum_residual, float(np.max(np.abs(residual)))
            )

    initial_force = np.asarray(anchor_points[0]["plga_force"], dtype=np.float64)
    terminal_force = np.asarray(terminal["plga_force"], dtype=np.float64)
    initial_response = np.asarray(
        anchor_points[0]["plga_absolute_response"], dtype=np.float64
    )
    terminal_response = np.asarray(
        terminal["plga_absolute_response"], dtype=np.float64
    )
    terminal_ratio = np.asarray(
        terminal["plga_quotient_ratio"], dtype=np.float64
    )
    return {
        "run_id": run_id,
        "lineage_root": lineage_root,
        "log_sha256": _sha256(log_path),
        "full_snapshot_count": len(full_energy),
        "dense_snapshot_count": len(dense_energy),
        "endpoint_gain_1024_to_terminal": _summary(endpoint_gain),
        "maps_with_endpoint_contraction": int(np.sum(endpoint_gain < 1.0)),
        "factorization": {
            "construction_frozen_tolerance": threshold,
            "maximum_per_map_relative_residual": factorization_max,
            "timepoints_with_any_map_above_tolerance": factorization_states_above,
            "map_timepoints_above_tolerance": factorization_map_values_above,
        },
        "blocks": observed_block_diagnostics(full_steps, energy),
        "dense_windows": exact_dense_diagnostics(dense_all),
        "duhamel": duhamel_diagnostics(gates, rates),
        "coordinate_anchors": coordinate,
        "coordinate_log_cocycle_maximum_residual": cocycle_maximum_residual,
        "plga": {
            "force_increased_count": int(np.sum(terminal_force > initial_force)),
            "force_initial": _summary(initial_force),
            "force_terminal": _summary(terminal_force),
            "response_decreased_count": int(
                np.sum(terminal_response < initial_response)
            ),
            "response_terminal": _summary(terminal_response),
            "endpoint_ratio_terminal": _summary(terminal_ratio),
        },
        "resource": {
            key: resource_summary[key]
            for key in (
                "resolved_device", "elapsed_seconds",
                "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
                "peak_host_rss_bytes",
            )
        },
    }


def _endpoint_energy(log_path: Path, endpoint_step: int) -> np.ndarray:
    result = None
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            point = row.get("mixed_collapse_full_timepoint")
            if point is not None and int(point["step"]) == endpoint_step:
                result = np.asarray(point["energy"], dtype=np.float64)
    if result is None:
        raise ValueError(f"missing endpoint {endpoint_step}: {log_path}")
    return result


def _factorial(bundle: Path) -> dict[str, Any]:
    effects = []
    interactions = []
    digests = []
    for trajectory in TRAJECTORIES:
        seed = int(trajectory["seed"])
        for source_step in INTERVENTION_UPDATES:
            endpoint = source_step + INTERVENTION_LENGTH
            mode_energy = {}
            for mode in INTERVENTION_MODES:
                path = bundle / "runs" / f"gate-{seed}-{source_step}-{mode}" / "log.jsonl"
                digests.append(_sha256(path))
                mode_energy[mode] = _endpoint_energy(path, endpoint).reshape(
                    REGISTRY["context_count"], 3, 4
                ).sum(axis=(0, 2))
            control = mode_energy["full"]
            for layer in REGISTRY["layers"]:
                for mode in INTERVENTION_MODES:
                    value = mode_energy[mode][layer]
                    effects.append({
                        "trajectory": str(trajectory["label"]),
                        "seed": seed,
                        "source_step": int(source_step),
                        "endpoint_step": int(endpoint),
                        "layer": int(layer),
                        "mode": mode,
                        "energy_ratio_to_control": float(value / control[layer]),
                        "energy_difference_from_control": float(
                            value - control[layer]
                        ),
                    })
                interaction = (
                    mode_energy["full"][layer]
                    - mode_energy["decay_only"][layer]
                    - mode_energy["adaptive_only"][layer]
                    + mode_energy["frozen"][layer]
                )
                interactions.append({
                    "trajectory": str(trajectory["label"]),
                    "seed": seed,
                    "source_step": int(source_step),
                    "layer": int(layer),
                    "interaction_energy": float(interaction),
                    "interaction_over_control": float(interaction / control[layer]),
                })
    return {
        "analysis_unit": "seed-layer-horizon",
        "source_log_sha256": sorted(digests),
        "effects": effects,
        "interactions": interactions,
        "all_registered_arms_present": len(effects)
        == len(TRAJECTORIES) * len(INTERVENTION_UPDATES) * 3 * len(INTERVENTION_MODES),
    }


def _source_radius(bundle: Path) -> dict[str, Any]:
    records = []
    for trajectory in TRAJECTORIES:
        seed = int(trajectory["seed"])
        for step in SOURCE_RADIUS_UPDATES:
            path = bundle / "records" / f"source-radius-{seed}-{step}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            maps = value["map_records"]
            records.append({
                "trajectory": str(trajectory["label"]),
                "seed": seed,
                "source_step": int(step),
                "record_sha256": _sha256(path),
                "radius_contract_closed": bool(value["radius_contract_closed"]),
                "energy_envelope_closed": bool(value["energy_envelope_closed"]),
                "realized_radius": _summary([
                    row["realized_radius"] for row in maps
                ]),
                "radius_slack": _summary([
                    row["radius_slack"] for row in maps
                ]),
                "energy_slack": _summary([
                    row["energy_slack"] for row in maps
                ]),
            })
    return {
        "predeclared_radius": TOLERANCES["source_frozen_radius"],
        "records": records,
        "all_radius_contracts_closed": all(
            row["radius_contract_closed"] for row in records
        ),
        "all_energy_envelopes_closed": all(
            row["energy_envelope_closed"] for row in records
        ),
    }




def _bundle_manifest(bundle: Path) -> bytes:
    files = sorted(
        path for path in bundle.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )
    return ("\n".join(
        f"{_sha256(path)}  {path.relative_to(bundle).as_posix()}"
        for path in files
    ) + "\n").encode("ascii")






