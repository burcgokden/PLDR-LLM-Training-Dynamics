#!/usr/bin/env python3
"""Independent reducer for the observer-stratum and block-gain analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


INVENTORY_SCHEMA = "pldr-observer-block-inventory-v2"
SUMMARY_SCHEMA = "pldr-observer-block-analysis-v1"
CHECK_SCHEMA = "pldr-observer-block-independent-check-v2"
HORIZONS = (1, 2, 4, 8, 16, 32)
ABSOLUTE_TOLERANCE = 1.0e-14
RELATIVE_TOLERANCE = 1.0e-12


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) + "\n").encode("utf-8")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def scalar_census(inventory: dict[str, Any]) -> dict[str, Any]:
    floor = float(inventory["effect_floor"])
    aggregate: dict[str, dict[str, int]] = {}
    target = None
    cells = []
    for row in inventory["energy_records"]:
        path = Path(row["path"]).resolve(strict=True)
        if sha256_path(path) != row["sha256"]:
            raise ValueError(f"independent record digest changed: {path}")
        with np.load(path, allow_pickle=False) as record:
            energy = np.asarray(record["source_energy"], dtype=np.float64)
            centered = np.asarray(record["source_z"], dtype=np.float64)
        exact = censored = resolved = 0
        positives = []
        for value in energy.flat:
            scalar = float(value)
            if not np.isfinite(scalar) or scalar < 0.0:
                raise ValueError(f"invalid stored energy: {path}")
            if scalar == 0.0:
                exact += 1
            elif scalar <= floor:
                censored += 1
                positives.append(scalar)
            else:
                resolved += 1
                positives.append(scalar)
        if exact == energy.size:
            stratum = "exact-face"
        elif censored == energy.size:
            stratum = "floor-censored-positive"
        elif resolved == energy.size:
            stratum = "resolved-positive"
        else:
            stratum = "mixed-observer-strata"
        cell = {
            "collection": row["collection"],
            "trajectory": row["trajectory"],
            "step": int(row["step"]),
            "layer": int(row["layer"]),
            "planned_maps": int(energy.size),
            "exact_zero_maps": exact,
            "floor_censored_positive_maps": censored,
            "resolved_positive_maps": resolved,
            "stratum": stratum,
            "minimum_positive_energy": min(positives) if positives else None,
            "maximum_positive_energy": max(positives) if positives else None,
            "maximum_absolute_centered_coordinate": float(
                max(abs(float(value)) for value in centered.flat)
            ),
            "all_centered_coordinates_float32_roundtrip": all(
                float(np.float32(value)) == float(value) for value in centered.flat
            ),
        }
        cells.append(cell)
        collection = aggregate.setdefault(row["collection"], {
            "map_count": 0,
            "exact_zero_maps": 0,
            "floor_censored_positive_maps": 0,
            "resolved_positive_maps": 0,
        })
        collection["map_count"] += int(energy.size)
        collection["exact_zero_maps"] += exact
        collection["floor_censored_positive_maps"] += censored
        collection["resolved_positive_maps"] += resolved
        if (
            row["collection"] == "holdout"
            and row["trajectory"] == "C"
            and int(row["step"]) == 65_536
            and int(row["layer"]) == 0
        ):
            target = dict(cell)
    if target is None:
        raise ValueError("independent target cell not found")
    target["effect_floor_over_maximum_energy"] = (
        floor / target["maximum_positive_energy"]
    )
    return {
        "collections": aggregate,
        "all_exact_zero_maps": sum(
            cell["exact_zero_maps"] for cell in cells
        ),
        "all_maps": sum(cell["planned_maps"] for cell in cells),
        "target": target,
        "cells": cells,
    }


def dense_sequences(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    dense = inventory["dense_archive"]
    spec_path = Path(dense["specification"]["path"]).resolve(strict=True)
    if sha256_path(spec_path) != dense["specification"]["sha256"]:
        raise ValueError("independent dense specification changed")
    spec = read_object(spec_path)
    expected = {int(value) for value in spec["dense_snapshot_updates"]}
    centers = tuple(int(value) for value in spec["dense_centers"])
    width = int(spec["dense_half_width"])
    map_count = int(spec["registry"]["map_count"])
    windows = []
    for log in dense["logs"]:
        log_path = Path(log["path"]).resolve(strict=True)
        if sha256_path(log_path) != log["sha256"]:
            raise ValueError(f"independent dense log changed: {log_path}")
        points = {}
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if "mixed_collapse_dense_timepoint" not in line and (
                "mixed_collapse_full_timepoint" not in line
            ):
                continue
            outer = json.loads(line)
            point = outer.get("mixed_collapse_dense_timepoint")
            if point is None:
                point = outer.get("mixed_collapse_full_timepoint")
            step = int(point.get("update_index", point.get("step")))
            if step in expected:
                vector = np.asarray(point["energy"], dtype=np.float64).reshape(-1)
                if vector.size != map_count or step in points:
                    raise ValueError("independent dense point is malformed")
                points[step] = vector
        if set(points) != expected:
            raise ValueError("independent dense schedule is incomplete")
        for center in centers:
            selected = [
                step for step in sorted(points)
                if abs(step - center) <= width
            ]
            if len(selected) > 1:
                windows.append({
                    "trajectory": log["trajectory"],
                    "center": center,
                    "energy": np.asarray([points[step] for step in selected]),
                })
    return windows


def type7_quantile(values: list[float], probability: float) -> float:
    """Return the Hyndman-Fan type-7 quantile using scalar Python arithmetic."""
    if not values:
        raise ValueError("cannot summarize an empty population")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability is outside [0, 1]")
    ordered = sorted(float(value) for value in values)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("non-finite value in quantile population")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def scalar_quantiles(values: list[float]) -> dict[str, float]:
    return {
        "p05": type7_quantile(values, 0.05),
        "median": type7_quantile(values, 0.50),
        "p95": type7_quantile(values, 0.95),
        "maximum": max(values),
    }


def scalar_sign_counts(values: list[float]) -> dict[str, int]:
    return {
        "contracting": sum(value < 1.0 for value in values),
        "preserving": sum(value == 1.0 for value in values),
        "expanding": sum(value > 1.0 for value in values),
    }


def independent_blocks(
    inventory: dict[str, Any], windows: list[dict[str, Any]],
) -> dict[str, Any]:
    floor = float(inventory["effect_floor"])
    output: dict[str, Any] = {
        "window_count": len(windows),
        "all_stored_energies_strictly_positive": all(
            float(value) > 0.0
            for window in windows
            for value in window["energy"].flat
        ),
        "horizons": {},
    }
    for horizon in HORIZONS:
        ratios: list[float] = []
        resolved_ratios: list[float] = []
        censored_ratios: list[float] = []
        excursions: list[float] = []
        expanding_windows = 0
        window_rows = []
        for window in windows:
            energy = window["energy"]
            window_ratios: list[float] = []
            for start in range(0, energy.shape[0] - horizon):
                for map_index in range(energy.shape[1]):
                    source = float(energy[start, map_index])
                    endpoint = float(energy[start + horizon, map_index])
                    if source <= 0.0 or endpoint <= 0.0:
                        raise ValueError("independent block endpoint is not positive")
                    ratio = endpoint / source
                    ratios.append(ratio)
                    window_ratios.append(ratio)
                    block_maximum = max(
                        float(energy[offset, map_index])
                        for offset in range(start, start + horizon + 1)
                    )
                    excursions.append(block_maximum / source - 1.0)
                    if source > floor and endpoint > floor:
                        resolved_ratios.append(ratio)
                    else:
                        censored_ratios.append(ratio)
            has_expansion = any(value > 1.0 for value in window_ratios)
            expanding_windows += int(has_expansion)
            window_rows.append({
                "trajectory": window["trajectory"],
                "center": window["center"],
                "comparisons": len(window_ratios),
                "expanding": sum(value > 1.0 for value in window_ratios),
                "contains_expansion": has_expansion,
                "maximum_gain": max(window_ratios),
            })
        counts = scalar_sign_counts(ratios)
        resolved_counts = scalar_sign_counts(resolved_ratios)
        censored_counts = scalar_sign_counts(censored_ratios)
        output["horizons"][str(horizon)] = {
            "comparisons": len(ratios),
            **counts,
            "contracting_fraction": counts["contracting"] / len(ratios),
            "endpoint_gain_quantiles": scalar_quantiles(ratios),
            "maximum_relative_excursion_quantiles": scalar_quantiles(excursions),
            "resolved": {
                "comparisons": len(resolved_ratios),
                **resolved_counts,
                "endpoint_gain_quantiles": scalar_quantiles(resolved_ratios),
            },
            "censored_endpoint_population": {
                "comparisons": len(censored_ratios),
                **censored_counts,
            },
            "contributing_windows": len(window_rows),
            "windows_with_expansion": expanding_windows,
            "every_window_contains_expansion": (
                expanding_windows == len(window_rows)
            ),
            "uniform_stored_block_contraction_observed": all(
                value < 1.0 for value in ratios
            ),
            "windows": window_rows,
        }
    return output


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(
        abs(left), abs(right)
    )


def compare(inventory: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("independent inventory schema changed")
    if summary.get("schema_version") != SUMMARY_SCHEMA:
        raise ValueError("primary summary schema changed")
    census = scalar_census(inventory)
    blocks = independent_blocks(inventory, dense_sequences(inventory))
    primary_census = summary["observer_census"]
    checked_discrete = 0
    checked_numeric = 0
    numeric_differences = []

    def check_equal(left: Any, right: Any, label: str) -> None:
        nonlocal checked_discrete
        if left != right:
            raise ValueError(f"independent discrete mismatch: {label}")
        checked_discrete += 1

    def check_numeric(left: Any, right: Any, label: str) -> None:
        nonlocal checked_numeric
        if left is None or right is None:
            check_equal(left, right, label)
            return
        difference = abs(float(left) - float(right))
        numeric_differences.append(difference)
        if not _close(float(left), float(right)):
            raise ValueError(f"independent numeric mismatch: {label}")
        checked_numeric += 1

    for key in ("all_exact_zero_maps", "all_maps"):
        check_equal(census[key], primary_census[key], f"observer/{key}")
    for collection, row in census["collections"].items():
        primary = primary_census["collections"][collection]
        for key, value in row.items():
            check_equal(value, primary[key], f"collection/{collection}/{key}")

    if len(census["cells"]) != len(primary_census["cells"]):
        raise ValueError("observer per-cell table length changed")
    for index, (cell, primary) in enumerate(zip(
        census["cells"], primary_census["cells"], strict=True,
    )):
        for key in (
            "collection", "trajectory", "step", "layer", "planned_maps",
            "exact_zero_maps", "floor_censored_positive_maps",
            "resolved_positive_maps", "stratum",
            "all_centered_coordinates_float32_roundtrip",
        ):
            check_equal(cell[key], primary[key], f"cell/{index}/{key}")
        for key in (
            "minimum_positive_energy", "maximum_positive_energy",
            "maximum_absolute_centered_coordinate",
        ):
            check_numeric(cell[key], primary[key], f"cell/{index}/{key}")

    target = census["target"]
    primary_target = primary_census["target_floor_censored_cell"]
    for key in (
        "planned_maps", "exact_zero_maps", "floor_censored_positive_maps",
        "resolved_positive_maps", "stratum",
        "all_centered_coordinates_float32_roundtrip",
    ):
        check_equal(target[key], primary_target[key], f"target/{key}")
    for key in (
        "minimum_positive_energy", "maximum_positive_energy",
        "maximum_absolute_centered_coordinate",
        "effect_floor_over_maximum_energy",
    ):
        check_numeric(target[key], primary_target[key], f"target/{key}")

    primary_blocks = summary["block_gain_census"]
    for key in ("window_count", "all_stored_energies_strictly_positive"):
        check_equal(blocks[key], primary_blocks[key], f"blocks/{key}")
    for horizon, row in blocks["horizons"].items():
        primary = primary_blocks["horizons"][horizon]
        for key in (
            "comparisons", "contracting", "preserving", "expanding",
            "contributing_windows", "windows_with_expansion",
            "every_window_contains_expansion",
            "uniform_stored_block_contraction_observed",
        ):
            check_equal(row[key], primary[key], f"block/{horizon}/{key}")
        check_numeric(
            row["contracting_fraction"], primary["contracting_fraction"],
            f"block/{horizon}/contracting_fraction",
        )
        for population in (
            "endpoint_gain_quantiles", "maximum_relative_excursion_quantiles",
        ):
            for key in ("p05", "median", "p95", "maximum"):
                check_numeric(
                    row[population][key], primary[population][key],
                    f"block/{horizon}/{population}/{key}",
                )
        for population in ("resolved", "censored_endpoint_population"):
            for key in (
                "comparisons", "contracting", "preserving", "expanding",
            ):
                check_equal(
                    row[population][key], primary[population][key],
                    f"block/{horizon}/{population}/{key}",
                )
        for key in ("p05", "median", "p95", "maximum"):
            check_numeric(
                row["resolved"]["endpoint_gain_quantiles"][key],
                primary["resolved"]["endpoint_gain_quantiles"][key],
                f"block/{horizon}/resolved/endpoint_gain_quantiles/{key}",
            )
        if len(row["windows"]) != len(primary["windows"]):
            raise ValueError(f"block window table length changed: {horizon}")
        for index, (window, primary_window) in enumerate(zip(
            row["windows"], primary["windows"], strict=True,
        )):
            for key in (
                "trajectory", "center", "comparisons", "expanding",
                "contains_expansion",
            ):
                check_equal(
                    window[key], primary_window[key],
                    f"block/{horizon}/window/{index}/{key}",
                )
            check_numeric(
                window["maximum_gain"], primary_window["maximum_gain"],
                f"block/{horizon}/window/{index}/maximum_gain",
            )

    return {
        "schema_version": CHECK_SCHEMA,
        "primary_summary_sha256": hashlib.sha256(
            canonical_bytes(summary)
        ).hexdigest(),
        "independent_implementation": True,
        "checked_discrete_fields": checked_discrete,
        "checked_numeric_fields": checked_numeric,
        "quantile_definition": "Hyndman-Fan-type-7-scalar-interpolation",
        "maximum_absolute_numeric_difference": max(
            numeric_differences, default=0.0,
        ),
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "all_checks_pass": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    inventory = read_object(arguments.inventory.resolve(strict=True))
    summary_path = arguments.summary.resolve(strict=True)
    summary = read_object(summary_path)
    payload = canonical_bytes(compare(inventory, summary))
    output = arguments.output.resolve(strict=False)
    if arguments.check:
        if not output.is_file() or output.read_bytes() != payload:
            raise SystemExit("observer-block independent check is stale")
    else:
        write_bytes(output, payload)
    print(json.dumps({
        "all_checks_pass": True,
        "output": str(output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
