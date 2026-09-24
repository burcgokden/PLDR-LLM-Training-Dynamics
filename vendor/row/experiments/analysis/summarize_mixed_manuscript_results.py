#!/usr/bin/env python3
"""Generate manuscript mechanism tables from the sealed mixed-collapse analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BUNDLE = (
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev47"
    / "mixed-row-map-collapse-confirmation"
)
DEFAULT_OUTPUT = (
    ROOT / "docs" / "figures" / "mixed_collapse_mechanism_results.tex"
)
EXPECTED_SCHEMA = "pldr-mixed-collapse-full-analysis-v1"
EXPECTED_SUMMARY_SCHEMA = "pldr-mixed-collapse-manuscript-summary-v1"
SHAPE_SUPPLEMENT_SCHEMA = "pldr-mixed-collapse-shape-supplement-v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _fmt(value: float) -> str:
    value = float(value)
    absolute = abs(value)
    if value != value or absolute == float("inf"):
        raise ValueError("nonfinite manuscript value")
    if absolute != 0.0 and (absolute < 1.0e-3 or absolute >= 1.0e4):
        return f"{value:.3e}"
    return f"{value:.4g}"


def _tex_int(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _median(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty population")
    return float(statistics.median(float(value) for value in values))


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _finite_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    converted = [float(value) for value in values]
    if not all(math.isfinite(value) for value in converted):
        raise ValueError("shape-gain summary contains a nonfinite value")
    return {
        "minimum": min(converted),
        "median": _median(converted),
        "maximum": max(converted),
    }


def _shape_gain_summary(
    source_shape: list[float],
    terminal_shape: list[float],
) -> dict[str, Any]:
    if not source_shape or len(source_shape) != len(terminal_shape):
        raise ValueError("shape endpoint populations disagree")
    source = [float(value) for value in source_shape]
    terminal = [float(value) for value in terminal_shape]
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in source + terminal
    ):
        raise ValueError("shape endpoints must be finite and nonnegative")
    positive_gain = [
        endpoint / start
        for start, endpoint in zip(source, terminal, strict=True)
        if start > 0.0
    ]
    return {
        "map_count": len(source),
        "positive_source_count": sum(value > 0.0 for value in source),
        "zero_source_count": sum(value == 0.0 for value in source),
        "zero_source_reopened_count": sum(
            start == 0.0 and endpoint > 0.0
            for start, endpoint in zip(source, terminal, strict=True)
        ),
        "maps_with_shape_contraction": sum(
            endpoint < start
            for start, endpoint in zip(source, terminal, strict=True)
        ),
        "maps_with_shape_expansion": sum(
            endpoint > start
            for start, endpoint in zip(source, terminal, strict=True)
        ),
        "positive_source_gain": _finite_summary(positive_gain),
    }


def _factorization_capture_point(
    point: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return absolute residual, relative residual, and physical energy."""

    energy = np.asarray(point["energy"], dtype=np.float64)
    shape = np.asarray(
        point["normalized_shape_coordinate_energy"], dtype=np.float64
    )
    gates = np.asarray(point["layer_final_gate"], dtype=np.float64)
    if energy.shape != (288,) or shape.shape != (288, 64):
        raise ValueError("factorization capture map population drifted")
    if gates.shape != (3, 64):
        raise ValueError("factorization capture gate population drifted")
    if (
        not np.isfinite(energy).all()
        or not np.isfinite(shape).all()
        or not np.isfinite(gates).all()
        or np.any(energy < 0.0)
        or np.any(shape < 0.0)
    ):
        raise ValueError("factorization capture contains invalid values")
    map_gates = np.broadcast_to(
        gates[None, :, None, :], (24, 3, 4, 64)
    ).reshape(288, 64)
    predicted = np.sum(shape * map_gates * map_gates, axis=1)
    absolute = np.abs(energy - predicted)
    relative = absolute / np.maximum(energy, np.finfo(np.float64).tiny)
    return absolute, relative, energy


def _shape_supplement(
    bundle: Path,
    result: dict[str, Any],
    digest: str,
) -> dict[str, Any]:
    source_step = 1_024
    terminal_step = int(result["terminal_update"])
    map_count = int(result["map_count"])
    trajectories = {}
    for label in ("A", "B", "C"):
        trajectory = result["trajectories"][label]
        log_path = (
            bundle / "runs" / str(trajectory["trajectory"]) / "log.jsonl"
        )
        log_digest = _sha256(log_path)
        if log_digest != trajectory["log_sha256"]:
            raise ValueError(f"{label}: trajectory log digest drifted")
        points: dict[int, dict[str, Any]] = {}
        seen_factorization_steps: set[int] = set()
        maximum_absolute_residual = 0.0
        failure_energy_values: list[float] = []
        failing_timepoints = 0
        failing_map_timepoints = 0
        threshold = float(
            trajectory["factorization"]["construction_frozen_tolerance"]
        )
        with log_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                full_point = row.get("mixed_collapse_full_timepoint")
                dense_point = row.get("mixed_collapse_dense_timepoint")
                if full_point is not None and dense_point is not None:
                    raise ValueError(f"{label}: duplicate timepoint kinds")
                point = full_point if full_point is not None else dense_point
                if not isinstance(point, dict):
                    continue
                step = int(point["step"])
                if step in seen_factorization_steps:
                    raise ValueError(f"{label}: duplicate factorization step {step}")
                seen_factorization_steps.add(step)
                absolute, relative, point_energy = _factorization_capture_point(
                    point
                )
                maximum_absolute_residual = max(
                    maximum_absolute_residual, float(np.max(absolute))
                )
                failed = relative > threshold
                failed_count = int(np.sum(failed))
                failing_map_timepoints += failed_count
                failing_timepoints += int(failed_count > 0)
                failure_energy_values.extend(point_energy[failed].tolist())
                if full_point is not None and step in {
                    source_step, terminal_step
                }:
                    if step in points:
                        raise ValueError(
                            f"{label}: duplicate shape endpoint {step}"
                        )
                    points[step] = point
        if set(points) != {source_step, terminal_step}:
            raise ValueError(f"{label}: shape endpoints are incomplete")
        expected_factorization = trajectory["factorization"]
        if (
            failing_timepoints
            != int(expected_factorization["timepoints_with_any_map_above_tolerance"])
            or failing_map_timepoints
            != int(expected_factorization["map_timepoints_above_tolerance"])
        ):
            raise ValueError(
                f"{label}: factorization supplement disagrees with analysis"
            )

        totals = {}
        for step, point in points.items():
            coordinates = point["normalized_shape_coordinate_energy"]
            if len(coordinates) != map_count:
                raise ValueError(f"{label}: shape map population drifted")
            values = []
            for coordinate in coordinates:
                if not isinstance(coordinate, list) or len(coordinate) != 64:
                    raise ValueError(f"{label}: shape coordinate row is invalid")
                converted = [float(value) for value in coordinate]
                if not all(
                    math.isfinite(value) and value >= 0.0
                    for value in converted
                ):
                    raise ValueError(
                        f"{label}: shape coordinate value is invalid"
                    )
                values.append(sum(converted))
            totals[step] = values
        trajectories[label] = {
            "trajectory": str(trajectory["trajectory"]),
            "log_sha256": log_digest,
            "factorization_capture": {
                "timepoint_count": len(seen_factorization_steps),
                "maximum_absolute_residual": maximum_absolute_residual,
                "relative_failure_count": failing_map_timepoints,
                "timepoints_with_relative_failure": failing_timepoints,
                "relative_failure_energy": _finite_summary(
                    failure_energy_values
                ),
            },
            **_shape_gain_summary(
                totals[source_step],
                totals[terminal_step],
            ),
        }
    return {
        "schema_version": SHAPE_SUPPLEMENT_SCHEMA,
        "producer_sha256": _sha256(Path(__file__).resolve()),
        "final_analysis_sha256": digest,
        "source_update": source_step,
        "terminal_update": terminal_step,
        "trajectories": trajectories,
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


def _fmt_optional(summary: dict[str, float] | None, key: str) -> str:
    return "--" if summary is None else _fmt(summary[key])

def _factorial_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    factorial = result["gate_factorial"]
    effects = factorial["effects"]
    interactions = factorial["interactions"]
    rows = []
    steps = sorted({int(row["source_step"]) for row in effects})
    for step in steps:
        by_mode = {}
        for mode in ("frozen", "decay_only", "adaptive_only"):
            values = [
                float(row["energy_ratio_to_control"])
                for row in effects
                if int(row["source_step"]) == step and row["mode"] == mode
            ]
            if len(values) != 9:
                raise ValueError(f"factorial unit count drifted: {step} {mode}")
            by_mode[mode] = {
                "median": _median(values),
                "above_control": sum(value > 1.0 for value in values),
            }
        interaction = [
            float(row["interaction_over_control"])
            for row in interactions
            if int(row["source_step"]) == step
        ]
        if len(interaction) != 9:
            raise ValueError(f"factorial interaction count drifted: {step}")
        rows.append({
            "source_step": step,
            "modes": by_mode,
            "interaction_median": _median(interaction),
        })
    return rows


def _bound_result(bundle: Path) -> tuple[dict[str, Any], str]:
    final_path = bundle / "reports" / "final-analysis.json"
    summary_path = bundle / "reports" / "manuscript-summary.json"
    result = _load(final_path)
    summary = _load(summary_path)
    digest = _sha256(final_path)
    if result.get("schema_version") != EXPECTED_SCHEMA:
        raise ValueError("mixed-collapse analysis schema drifted")
    if summary.get("schema_version") != EXPECTED_SUMMARY_SCHEMA:
        raise ValueError("mixed-collapse summary schema drifted")
    if summary.get("final_analysis_sha256") != digest:
        raise ValueError("manuscript summary is not bound to final analysis")
    if result.get("finite_horizon_only") is not True:
        raise ValueError("finite-horizon scope marker is absent")
    if set(result.get("trajectories", {})) != {"A", "B", "C"}:
        raise ValueError("trajectory population drifted")
    return result, digest






