#!/usr/bin/env python3
"""Independent float64 analysis of exact comprehensive endpoint ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.comprehensive_campaign import campaign_spec  # noqa: E402
from confirm.comprehensive_gate_shape import (  # noqa: E402
    coordinate_block_recurrence,
    diameter_slack_summary,
    exact_pair_decomposition,
    finite_registry_geometry,
)
from confirm.comprehensive_lock import load_lock  # noqa: E402
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402


LEDGER_SCHEMA = "pldr-comprehensive-endpoint-ledger-v1"
REPORT_SCHEMA = "pldr-comprehensive-endpoint-report-v1"


def _scalar(archive: dict[str, np.ndarray], name: str) -> Any:
    value = np.asarray(archive[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _pair(value: Any) -> list[int]:
    return [int(value[0]), int(value[1])]


def _relative(value: np.ndarray, scale: np.ndarray) -> float:
    return float(np.max(
        np.abs(value) / np.maximum(np.abs(scale), 1.0), initial=0.0))


def load_ledger(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        value = {name: source[name] for name in source.files}
    required = {
        "schema_version", "campaign_id", "campaign_spec_sha256", "role",
        "seed", "step_start", "layer", "normalized_rows", "gate",
        "gate_multiplier", "gate_innovation", "learning_rate",
        "weight_decay", "optimizer_step_after", "update_chunk_indices",
        "update_batch_sha256", "native_gate_replay_relative_residual",
        "checkpoint_sha256", "registry_sha256", "data_order_sha256",
        "context_chunk_size", "resource_elapsed_seconds", "resource_device",
        "resource_peak_gpu_allocated_bytes", "resource_peak_gpu_reserved_bytes",
        "resource_peak_host_rss_bytes",
    }
    if set(value) != required:
        raise ValueError(
            "endpoint ledger schema mismatch: "
            f"missing={sorted(required - set(value))}, "
            f"unknown={sorted(set(value) - required)}"
        )
    if _scalar(value, "schema_version") != LEDGER_SCHEMA:
        raise ValueError("unknown endpoint-ledger schema")
    return value


def analyze(
    path: str | Path, lock_path: str | Path | None = None,
) -> dict[str, Any]:
    archive = load_ledger(path)
    spec = campaign_spec()
    if _scalar(archive, "campaign_id") != spec["campaign_id"]:
        raise ValueError("ledger campaign identity disagrees")
    if _scalar(archive, "campaign_spec_sha256") != spec["spec_sha256"]:
        raise ValueError("ledger was not captured under the frozen specification")
    role = str(_scalar(archive, "role"))
    seed = int(_scalar(archive, "seed"))
    layer = int(_scalar(archive, "layer"))
    step_start = int(_scalar(archive, "step_start"))
    lock = load_lock(lock_path) if lock_path is not None else None
    if (
        role not in {"development", "construction", "heldout"}
        or seed not in spec["trajectories"][role]
        or layer not in spec["trajectories"]["layer_indices"]
    ):
        raise ValueError("endpoint role, seed, or layer is outside the campaign")
    if role == "heldout":
        if lock is None:
            raise ValueError("held-out endpoint analysis requires the construction lock")
    elif lock is not None:
        raise ValueError("development and construction analysis must precede the lock")
    rows = np.asarray(archive["normalized_rows"], dtype=np.float64)
    gates = np.asarray(archive["gate"], dtype=np.float64)
    multipliers = np.asarray(archive["gate_multiplier"], dtype=np.float64)
    innovations = np.asarray(archive["gate_innovation"], dtype=np.float64)
    updates = rows.shape[0] - 1
    expected_updates = int(
        spec["trajectories"]["development_block_updates"]
        if role == "development"
        else spec["trajectories"]["confirmation_block_updates"]
    )
    allowed_anchors = (
        spec["trajectories"]["anchors"][:3]
        if role == "development" else spec["trajectories"]["anchors"]
    )
    if updates != expected_updates or step_start not in allowed_anchors:
        raise ValueError("endpoint block or anchor is outside the campaign")
    expected_shapes = (
        (updates + 1, spec["registry"]["row_count"],
         spec["model"]["head_width"]),
        (updates + 1, spec["model"]["head_width"]),
        (updates, spec["model"]["head_width"]),
    )
    if (
        rows.shape != expected_shapes[0]
        or gates.shape != expected_shapes[1]
        or multipliers.shape != expected_shapes[2]
        or innovations.shape != expected_shapes[2]
    ):
        raise ValueError("endpoint ledger dimensions disagree with the campaign")
    if not all(np.isfinite(value).all() for value in (
        rows, gates, multipliers, innovations)):
        raise ValueError("endpoint ledger contains nonfinite tensors")

    pair_chunk = int(spec["registry"]["pair_chunk_size"])
    identity_tolerance = float(
        spec["decisions"]["identity_relative_tolerance"])
    native_tolerance = float(
        spec["decisions"]["native_float32_replay_tolerance"])
    geometries = [finite_registry_geometry(
        gates[index], rows[index], pair_chunk_size=pair_chunk)
        for index in range(updates + 1)]
    endpoint_rows = [{
        "offset": index,
        "diameter": float(value["diameter"]),
        "diameter_squared": float(value["diameter_squared"]),
        "carrier_linf": float(value["carrier_linf"]),
        "carrier_l2": float(value["carrier_l2"]),
        "diameter_maximizer": _pair(value["diameter_maximizer"]),
        "sandwich_lower_slack": float(value["sandwich_lower_slack"]),
        "sandwich_upper_slack": float(value["sandwich_upper_slack"]),
        "sandwich_valid": bool(value["sandwich_valid"]),
    } for index, value in enumerate(geometries)]

    step_rows = []
    worst_slack_identity = 0.0
    worst_pair_identity = 0.0
    worst_gate_balance = 0.0
    pair_expansion_count = 0
    maximizer_changes = 0
    for index in range(updates):
        slack = diameter_slack_summary(
            gates[index], rows[index], gates[index + 1], rows[index + 1],
            pair_chunk_size=pair_chunk)
        left, right = slack["target_maximizer"]
        contrast0 = (rows[index, left] - rows[index, right])[None, :]
        contrast1 = (rows[index + 1, left] - rows[index + 1, right])[None, :]
        decomposition = exact_pair_decomposition(
            gates[index],
            contrast0,
            gates[index + 1],
            contrast1,
            decayed_gate=multipliers[index] * gates[index],
            innovation=innovations[index],
        )
        pair_scale = np.maximum.reduce((
            np.abs(decomposition["pair_decrement"]),
            np.abs(decomposition["gate_decrement"]),
            np.abs(decomposition["shape_work"]),
            np.ones(1),
        ))
        gate_scale = np.maximum.reduce((
            np.abs(decomposition["gate_decrement"]),
            np.abs(decomposition["decay_dissipation"]),
            np.abs(decomposition["chronological_alignment"]),
            np.abs(decomposition["finite_gate_charge"]),
            np.ones(1),
        ))
        pair_residual = _relative(
            decomposition["decomposition_residual"], pair_scale)
        gate_residual = _relative(
            decomposition["gate_balance_residual"], gate_scale)
        worst_slack_identity = max(
            worst_slack_identity,
            float(slack["slack_identity_relative_residual"]),
        )
        worst_pair_identity = max(worst_pair_identity, pair_residual)
        worst_gate_balance = max(worst_gate_balance, gate_residual)
        pair_expansion_count += int(slack["expanding_pair_count"])
        maximizer_changes += int(slack["maximizer_changed"])
        step_rows.append({
            "offset": index,
            "source_maximizer": _pair(slack["source_maximizer"]),
            "target_maximizer": _pair(slack["target_maximizer"]),
            "maximizer_changed": bool(slack["maximizer_changed"]),
            "minimum_exact_slack": float(slack["minimum_exact_slack"]),
            "source_gap_at_target_maximizer": float(
                slack["source_gap_at_target_maximizer"]),
            "decrement_at_target_maximizer": float(
                slack["decrement_at_target_maximizer"]),
            "expanding_pair_count": int(slack["expanding_pair_count"]),
            "maximum_pair_expansion": float(slack["maximum_pair_expansion"]),
            "gate_decrement": float(decomposition["gate_decrement"][0]),
            "shape_work": float(decomposition["shape_work"][0]),
            "decay_dissipation": float(
                decomposition["decay_dissipation"][0]),
            "chronological_alignment": float(
                decomposition["chronological_alignment"][0]),
            "finite_gate_charge": float(
                decomposition["finite_gate_charge"][0]),
            "slack_identity_relative_residual": float(
                slack["slack_identity_relative_residual"]),
            "pair_decomposition_relative_residual": pair_residual,
            "gate_balance_relative_residual": gate_residual,
        })

    block = coordinate_block_recurrence(
        gates[0], rows[0], gates[-1], rows[-1],
        multipliers=multipliers, innovations=innovations)
    log_mask = block["positive_log_mask"]
    log_residual = block["log_identity_residual"][log_mask]
    worst_log_identity = float(np.max(
        np.abs(log_residual), initial=0.0))
    replay_reported = np.asarray(
        archive["native_gate_replay_relative_residual"], dtype=np.float64)
    checks = {
        "all_diameter_sandwiches": all(
            row["sandwich_valid"] for row in endpoint_rows),
        "diameter_slack_identity": worst_slack_identity <= identity_tolerance,
        "pair_endpoint_identity": worst_pair_identity <= identity_tolerance,
        "gate_balance_identity": worst_gate_balance <= identity_tolerance,
        "coordinate_block_coverage": bool(np.all(block["carrier_covered"])),
        "coordinate_log_identity": worst_log_identity <= identity_tolerance,
        "float32_gate_replay": bool(
            np.max(replay_reported, initial=0.0) <= native_tolerance
            and block["duhamel_endpoint_relative_residual"] <= native_tolerance
        ),
    }
    exact_checks_pass = all(checks.values())
    diameter_ratio = float(
        geometries[-1]["diameter"] / geometries[0]["diameter"]
        if geometries[0]["diameter"] > 0.0 else np.inf)
    accumulated_minimum_slack = float(sum(
        row["minimum_exact_slack"] for row in step_rows))
    primary_cell = False
    locked_checks = None
    heldout_primary_qualified = None
    if role == "heldout":
        primary_cell = any(
            int(cell["layer"]) == layer
            and int(cell["anchor"]) == step_start
            and int(cell["updates"]) == updates
            for cell in lock["primary_cells"]
        )
        route_bounds = lock["route_bounds"][str(layer)]
        locked_checks = {
            "diameter_ratio_limit": bool(
                diameter_ratio
                <= float(lock["diameter_ratio_limit_by_layer"][str(layer)])),
            "gate_route_envelope": bool(
                abs(float(np.nansum(block["gate_log_decrement"])))
                <= float(route_bounds["gate_abs"])),
            "shape_route_envelope": bool(
                abs(float(np.nansum(block["shape_log_decrement"])))
                <= float(route_bounds["shape_abs"])),
            "total_route_envelope": bool(
                abs(float(np.nansum(block["total_log_decrement"])))
                <= float(route_bounds["total_abs"])),
        }
        heldout_primary_qualified = bool(
            primary_cell and exact_checks_pass
            and locked_checks["diameter_ratio_limit"])
    development_positive = bool(
        exact_checks_pass and role == "development"
        and accumulated_minimum_slack > 0.0 and diameter_ratio < 1.0)
    if role == "heldout":
        decision = (
            "QUALIFIED" if heldout_primary_qualified
            else "NOT_QUALIFIED" if primary_cell else "REPORTED_NONPRIMARY")
    else:
        decision = "DEVELOPMENT_POSITIVE" if development_positive else "ANALYZED"
    report = {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "source_ledger": str(Path(path).resolve()),
        "role": role,
        "seed": seed,
        "layer": layer,
        "step_start": step_start,
        "step_stop": step_start + updates,
        "updates": updates,
        "diameter_ratio": diameter_ratio,
        "diameter_contracted": diameter_ratio < 1.0,
        "endpoints": endpoint_rows,
        "steps": step_rows,
        "block": {
            "carrier_coverage_fraction": float(np.mean(
                block["carrier_covered"])),
            "duhamel_endpoint_relative_residual": float(
                block["duhamel_endpoint_relative_residual"]),
            "positive_coordinate_count": int(np.count_nonzero(log_mask)),
            "gate_log_decrement_sum": float(np.nansum(
                block["gate_log_decrement"])),
            "shape_log_decrement_sum": float(np.nansum(
                block["shape_log_decrement"])),
            "total_log_decrement_sum": float(np.nansum(
                block["total_log_decrement"])),
            "worst_log_identity_residual": worst_log_identity,
            "minimum_exact_slack_sum": accumulated_minimum_slack,
            "gate_decrement_sum": float(sum(
                row["gate_decrement"] for row in step_rows)),
            "shape_work_sum": float(sum(
                row["shape_work"] for row in step_rows)),
            "decay_dissipation_sum": float(sum(
                row["decay_dissipation"] for row in step_rows)),
            "chronological_alignment_sum": float(sum(
                row["chronological_alignment"] for row in step_rows)),
            "finite_gate_charge_sum": float(sum(
                row["finite_gate_charge"] for row in step_rows)),
            "pair_expansion_count": pair_expansion_count,
            "maximizer_change_count": maximizer_changes,
        },
        "numerics": {
            "identity_relative_tolerance": identity_tolerance,
            "native_float32_replay_tolerance": native_tolerance,
            "worst_slack_identity_relative_residual": worst_slack_identity,
            "worst_pair_identity_relative_residual": worst_pair_identity,
            "worst_gate_balance_relative_residual": worst_gate_balance,
            "maximum_native_gate_replay_relative_residual": float(
                np.max(replay_reported, initial=0.0)),
        },
        "checks": checks,
        "exact_checks_pass": exact_checks_pass,
        "development_positive": development_positive,
        "primary_cell": primary_cell,
        "locked_checks": locked_checks,
        "heldout_primary_qualified": heldout_primary_qualified,
        "decision": decision,
        "lock_sha256": lock["lock_sha256"] if lock is not None else "",
        "resources": {
            name.removeprefix("resource_"): _scalar(archive, name)
            for name in archive if name.startswith("resource_")
        },
        "provenance": {
            key: str(_scalar(archive, key))
            for key in (
                "checkpoint_sha256", "registry_sha256", "data_order_sha256")
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--lock")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    report = analyze(arguments.ledger, arguments.lock)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "exact_checks_pass": report["exact_checks_pass"],
        "diameter_ratio": report["diameter_ratio"],
        "development_positive": report["development_positive"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
