#!/usr/bin/env python3
"""Summarize the completed orbitwise campaign for the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.orbitwise_specs import (  # noqa: E402
    CAMPAIGN_ID,
    DENSE_CENTERS,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.resource_executor import write_json_atomic  # noqa: E402


DEFAULT_BUNDLE = (
    ROOT.parent
    / "experiment-data"
    / "manuscript-revisions"
    / "rev46"
    / "orbitwise-row-map-confirmation"
)
DEFAULT_OUTPUT = DEFAULT_BUNDLE / "reports" / "manuscript-summary.json"
DEFAULT_TEX_OUTPUT = (
    ROOT / "docs" / "figures" / "orbitwise_confirmation_results.tex"
)
SUMMARY_SCHEMA = "pldr-orbitwise-manuscript-summary-v1"


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
    ).encode("utf-8")


def _strict_json(path: Path) -> dict[str, Any]:
    def pairs(pairs_value: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs_value:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            ValueError(f"{path}: nonfinite JSON token {token}")
        ),
    )
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _summary(value: Any) -> dict[str, float]:
    array = np.asarray(value, dtype=np.float64)
    if not array.size or not np.isfinite(array).all():
        raise ValueError("summary population must be finite and nonempty")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _full_endpoints(
    log_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, np.ndarray]]:
    first = None
    last = None
    dense: dict[int, np.ndarray] = {}
    with log_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(
                line,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"nonfinite log token {token}")
                ),
            )
            full = row.get("orbitwise_full_timepoint")
            if full is not None:
                if first is None:
                    first = full
                last = full
            point = row.get("orbitwise_dense_timepoint")
            if point is not None:
                step = int(point["step"])
                if step in dense:
                    raise ValueError(f"duplicate dense point {step}")
                dense[step] = np.asarray(point["energy"], dtype=np.float64)
    if first is None or last is None:
        raise ValueError(f"{log_path}: missing orbitwise endpoints")
    return first, last, dense


def _late_window(
    initial_energy: np.ndarray,
    dense: dict[int, np.ndarray],
) -> dict[str, Any]:
    center = int(DENSE_CENTERS[-1])
    path = np.stack([
        dense[step] for step in range(center - 32, center + 33)
    ])
    anchor = path[0]
    reopening = np.maximum(np.diff(path, axis=0), 0.0).sum(axis=0)
    envelope = anchor + reopening
    if (
        initial_energy.shape != (REGISTRY["sentinel_map_count"],)
        or np.any(initial_energy <= 0.0)
    ):
        raise ValueError("late-window normalization needs positive sentinels")
    return {
        "center": center,
        "map_count": REGISTRY["sentinel_map_count"],
        "anchor_over_initial": _summary(anchor / initial_energy),
        "reopening_budget_over_initial": _summary(
            reopening / initial_energy
        ),
        "envelope_over_initial": _summary(envelope / initial_energy),
        "maximum_raw_envelope_violation": float(
            np.max(np.maximum(np.max(path, axis=0) - envelope, 0.0))
        ),
    }


def summarize(bundle: Path) -> dict[str, Any]:
    """Return a deterministic summary of one completed campaign bundle."""

    bundle = bundle.resolve()
    final_path = bundle / "reports" / "final-analysis.json"
    final = _strict_json(final_path)
    if (
        final.get("campaign_id") != CAMPAIGN_ID
        or final.get("technical_valid") is not True
        or final.get("trajectory_count") != 3
    ):
        raise ValueError("final campaign analysis is absent or invalid")

    final_by_name = {
        row["trajectory"]: row for row in final["trajectories"]
    }
    endpoints = []
    late_windows = []
    plga_internal = []
    for trajectory in TRAJECTORIES:
        name = str(trajectory["name"])
        report = final_by_name[name]
        first, last, dense = _full_endpoints(
            bundle / "runs" / name / "log.jsonl"
        )
        initial_energy = np.asarray(
            first["energy"][:REGISTRY["sentinel_map_count"]],
            dtype=np.float64,
        )
        endpoints.append({
            "trajectory": name,
            "role": str(trajectory["role"]),
            "seed": int(trajectory["seed"]),
            "terminal_physical_gain": report[
                "terminal_physical_gain"
            ],
            "terminal_shape_gain": report["terminal_shape_gain"],
            "terminal_gate_only_gain": report[
                "terminal_gate_only_gain"
            ],
            "terminal_alignment_gain": report[
                "terminal_alignment_gain"
            ],
        })
        late = _late_window(initial_energy, dense)
        late["trajectory"] = name
        late_windows.append(late)

        response0 = np.asarray(
            first["plga_absolute_response"], dtype=np.float64
        )
        response1 = np.asarray(
            last["plga_absolute_response"], dtype=np.float64
        )
        force0 = np.asarray(first["plga_force"], dtype=np.float64)
        force1 = np.asarray(last["plga_force"], dtype=np.float64)
        plga_internal.append({
            "trajectory": name,
            "absolute_response_decrease_count": int(
                np.sum(response1 < response0)
            ),
            "absolute_response_increase_count": int(
                np.sum(response1 > response0)
            ),
            "force_decrease_count": int(np.sum(force1 < force0)),
            "force_increase_count": int(np.sum(force1 > force0)),
            "map_count": REGISTRY["map_count"],
        })

    source_rows = []
    source_artifacts = []
    for path in sorted(
        bundle.glob("records/C2/source-link-*.json"),
        key=lambda item: tuple(
            map(int, item.stem.removeprefix("source-link-").split("-"))
        ),
    ):
        value = _strict_json(path)
        records = value["map_records"]
        reopening = np.asarray(
            [row["realized_reopening"] for row in records],
            dtype=np.float64,
        )
        slack = np.asarray(
            [row["bound_slack"] for row in records],
            dtype=np.float64,
        )
        positive = reopening > 0.0
        source_rows.append({
            "source_step": int(value["source_step"]),
            "endpoint_step": int(value["endpoint_step"]),
            "map_count": len(records),
            "positive_reopening_count": int(np.sum(positive)),
            "positive_reopening": _summary(
                reopening[positive]
                if np.any(positive) else np.asarray([0.0])
            ),
            "minimum_bound_slack": float(np.min(slack)),
            "bound_holds_within_tolerance": bool(
                value["scientific_checks"][
                    "source_reopening_bound_closed"
                ]
            ),
        })
        source_artifacts.append({
            "path": str(path),
            "sha256": sha256_path(path),
        })

    gate_rows = []
    gate_artifacts = []
    for path in sorted(
        bundle.glob("records/C3/gate-result-*.json"),
        key=lambda item: int(item.stem.removeprefix("gate-result-")),
    ):
        value = _strict_json(path)
        difference = np.asarray([
            row["frozen_minus_control_energy"]
            for row in value["map_records"]
        ], dtype=np.float64)
        gate_rows.append({
            "source_step": int(value["source_step"]),
            "map_count": len(difference),
            "registered_direction_support_count": int(
                np.sum(difference > 0.0)
            ),
            "registered_direction_support_fraction": float(
                np.mean(difference > 0.0)
            ),
            "frozen_minus_control_energy": _summary(difference),
        })
        gate_artifacts.append({
            "path": str(path),
            "sha256": sha256_path(path),
        })

    resource_rows = [
        _strict_json(path)
        for path in sorted(bundle.glob("runtime/*/resource.json"))
    ]
    if len(resource_rows) != 59 or not all(
        row.get("technical_valid") is True for row in resource_rows
    ):
        raise ValueError("campaign resource population is incomplete")
    cuda_rows = [
        row for row in resource_rows
        if str(row["device"]).startswith("cuda")
    ]
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "final_analysis": {
            "path": str(final_path),
            "sha256": sha256_path(final_path),
        },
        "trajectory_count": 3,
        "endpoint_map_count": 3 * REGISTRY["map_count"],
        "full_snapshot_count": sum(
            row["full_snapshot_count"] for row in final["trajectories"]
        ),
        "dense_snapshot_count": sum(
            row["dense_snapshot_count"] for row in final["trajectories"]
        ),
        "dense_map_snapshot_count": (
            sum(row["dense_snapshot_count"] for row in final["trajectories"])
            * REGISTRY["sentinel_map_count"]
        ),
        "all_terminal_physical_gains_below_one": all(
            row["terminal_physical_gain"]["maximum"] < 1.0
            for row in endpoints
        ),
        "all_terminal_shape_gains_below_one": all(
            row["terminal_shape_gain"]["maximum"] < 1.0
            for row in endpoints
        ),
        "all_terminal_gate_only_gains_below_one": all(
            row["terminal_gate_only_gain"]["maximum"] < 1.0
            for row in endpoints
        ),
        "maximum_factorization_relative_residual": max(
            row["maximum_factorization_relative_residual"]
            for row in final["trajectories"]
        ),
        "maximum_three_channel_gain_residual": max(
            row["maximum_three_channel_gain_residual"]
            for row in final["trajectories"]
        ),
        "maximum_three_channel_log_residual": max(
            row["maximum_three_channel_log_residual"]
            for row in final["trajectories"]
        ),
        "endpoints": endpoints,
        "late_dense_windows": late_windows,
        "source_edges": source_rows,
        "source_edge_artifacts": source_artifacts,
        "source_record_count": sum(
            row["map_count"] for row in source_rows
        ),
        "source_positive_reopening_count": sum(
            row["positive_reopening_count"] for row in source_rows
        ),
        "source_reopening_bound_holds": all(
            row["bound_holds_within_tolerance"] for row in source_rows
        ),
        "minimum_source_bound_slack": min(
            row["minimum_bound_slack"] for row in source_rows
        ),
        "gate_internal_diagnostic": gate_rows,
        "gate_artifacts": gate_artifacts,
        "gate_registered_direction_support_count": sum(
            row["registered_direction_support_count"] for row in gate_rows
        ),
        "gate_record_count": sum(row["map_count"] for row in gate_rows),
        "plga_internal_diagnostic": plga_internal,
        "resource_node_count": len(resource_rows),
        "cuda_node_count": len(cuda_rows),
        "aggregate_gpu_hours": float(
            sum(row["wall_seconds"] for row in cuda_rows) / 3600.0
        ),
        "persistent_bundle_bytes": sum(
            path.stat().st_size
            for path in bundle.rglob("*")
            if path.is_file()
            and path != bundle / "reports" / "manuscript-summary.json"
        ),
        "technical_valid": True,
    }
    return result


def _number(value: float) -> str:
    if value != 0.0 and abs(value) < 1.0e-4:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10.0 ** exponent)
        body = rf"{mantissa:.3f}\times10^{{{exponent}}}"
    else:
        body = f"{value:.6f}".rstrip("0").rstrip(".")
    return "$" + body + "$"






