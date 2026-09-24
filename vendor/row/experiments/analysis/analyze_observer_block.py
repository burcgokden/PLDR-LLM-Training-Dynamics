#!/usr/bin/env python3
"""Analyze observer strata and finite block gains from sealed records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


INVENTORY_SCHEMA = "pldr-observer-block-inventory-v2"
ANALYSIS_SCHEMA = "pldr-observer-block-analysis-v1"
MAP_SHAPE = (24, 4)
ROW_MAP_SHAPE = (24, 4, 64, 64)
BLOCK_LENGTHS = (1, 2, 4, 8, 16, 32)


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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def classify_energies(
    values: np.ndarray, effect_floor: float,
) -> dict[str, np.ndarray]:
    """Return the exact observer partition for nonnegative stored energies."""

    energies = np.asarray(values, dtype=np.float64)
    if effect_floor <= 0.0 or not np.isfinite(effect_floor):
        raise ValueError("effect floor must be finite and strictly positive")
    if energies.shape != MAP_SHAPE:
        raise ValueError(f"unexpected energy shape: {energies.shape}")
    if not np.all(np.isfinite(energies)) or np.any(energies < 0.0):
        raise ValueError("stored energies must be finite and nonnegative")
    exact = energies == 0.0
    censored = (energies > 0.0) & (energies <= effect_floor)
    resolved = energies > effect_floor
    if not np.all(exact | censored | resolved):
        raise AssertionError("observer partition is not exhaustive")
    if np.any((exact & censored) | (exact & resolved) | (censored & resolved)):
        raise AssertionError("observer partition is not disjoint")
    return {"exact": exact, "censored": censored, "resolved": resolved}


def cell_stratum(counts: dict[str, int], planned: int) -> str:
    if counts["exact"] == planned:
        return "exact-face"
    if counts["censored"] == planned:
        return "floor-censored-positive"
    if counts["resolved"] == planned:
        return "resolved-positive"
    return "mixed-observer-strata"


def _record_census(row: dict[str, Any], effect_floor: float) -> dict[str, Any]:
    path = Path(str(row["path"])).resolve(strict=True)
    if sha256_path(path) != row.get("sha256"):
        raise ValueError(f"observer record digest changed: {path}")
    with np.load(path, allow_pickle=False) as record:
        energies = np.asarray(record["source_energy"], dtype=np.float64)
        centered = np.asarray(record["source_z"], dtype=np.float64)
        if centered.shape != ROW_MAP_SHAPE:
            raise ValueError(f"unexpected centered-row shape: {centered.shape}")
        for field, expected in (
            ("trajectory", row["trajectory"]),
            ("source_step", int(row["step"])),
            ("layer", int(row["layer"])),
        ):
            actual = record[field].item()
            if actual != expected:
                raise ValueError(f"observer record identity changed: {path}/{field}")

    partition = classify_energies(energies, effect_floor)
    counts = {
        name: int(np.count_nonzero(mask)) for name, mask in partition.items()
    }
    positive = energies[energies > 0.0]
    roundtrip = centered.astype(np.float32).astype(np.float64)
    return {
        "collection": row["collection"],
        "trajectory": row["trajectory"],
        "step": int(row["step"]),
        "layer": int(row["layer"]),
        "planned_maps": int(energies.size),
        "exact_zero_maps": counts["exact"],
        "floor_censored_positive_maps": counts["censored"],
        "resolved_positive_maps": counts["resolved"],
        "mathematical_radial_maps": int(np.count_nonzero(energies > 0.0)),
        "policy_radial_maps": counts["resolved"],
        "stratum": cell_stratum(counts, int(energies.size)),
        "minimum_positive_energy": (
            float(np.min(positive)) if positive.size else None
        ),
        "maximum_positive_energy": (
            float(np.max(positive)) if positive.size else None
        ),
        "maximum_absolute_centered_coordinate": float(np.max(np.abs(centered))),
        "all_centered_coordinates_float32_roundtrip": bool(
            np.array_equal(centered, roundtrip)
        ),
        "record_sha256": row["sha256"],
    }


def observer_census(
    records: list[dict[str, Any]], effect_floor: float,
) -> dict[str, Any]:
    cells = [_record_census(row, effect_floor) for row in records]
    collections: dict[str, Any] = {}
    for collection in sorted({row["collection"] for row in cells}):
        selected = [row for row in cells if row["collection"] == collection]
        positive_minima = [
            row["minimum_positive_energy"] for row in selected
            if row["minimum_positive_energy"] is not None
        ]
        positive_maxima = [
            row["maximum_positive_energy"] for row in selected
            if row["maximum_positive_energy"] is not None
        ]
        collections[collection] = {
            "cell_count": len(selected),
            "map_count": sum(row["planned_maps"] for row in selected),
            "exact_zero_maps": sum(row["exact_zero_maps"] for row in selected),
            "floor_censored_positive_maps": sum(
                row["floor_censored_positive_maps"] for row in selected
            ),
            "resolved_positive_maps": sum(
                row["resolved_positive_maps"] for row in selected
            ),
            "mathematical_radial_maps": sum(
                row["mathematical_radial_maps"] for row in selected
            ),
            "policy_radial_maps": sum(
                row["policy_radial_maps"] for row in selected
            ),
            "minimum_positive_energy": float(min(positive_minima)),
            "maximum_positive_energy": float(max(positive_maxima)),
            "stratum_cell_counts": {
                name: sum(row["stratum"] == name for row in selected)
                for name in (
                    "exact-face", "floor-censored-positive",
                    "resolved-positive", "mixed-observer-strata",
                )
            },
            "float32_roundtrip_cells": sum(
                row["all_centered_coordinates_float32_roundtrip"]
                for row in selected
            ),
        }

    target_rows = [
        row for row in cells
        if row["collection"] == "holdout"
        and row["trajectory"] == "C"
        and row["step"] == 65_536
        and row["layer"] == 0
    ]
    if len(target_rows) != 1:
        raise ValueError("target floor-censored cell is not unique")
    target = dict(target_rows[0])
    maximum = target["maximum_positive_energy"]
    if maximum is None or maximum <= 0.0:
        raise ValueError("target cell has no positive energy")
    target["effect_floor_over_maximum_energy"] = float(effect_floor / maximum)

    return {
        "effect_floor": effect_floor,
        "collections": collections,
        "target_floor_censored_cell": target,
        "all_exact_zero_maps": sum(row["exact_zero_maps"] for row in cells),
        "all_maps": sum(row["planned_maps"] for row in cells),
        "cells": cells,
    }


def _dense_points(
    log: dict[str, Any], expected: set[int], map_count: int,
) -> dict[int, np.ndarray]:
    path = Path(str(log["path"])).resolve(strict=True)
    if sha256_path(path) != log.get("sha256"):
        raise ValueError(f"dense log digest changed: {path}")
    points: dict[int, np.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if (
                '"mixed_collapse_dense_timepoint"' not in line
                and '"mixed_collapse_full_timepoint"' not in line
            ):
                continue
            payload = json.loads(line)
            point = payload.get("mixed_collapse_dense_timepoint") or payload.get(
                "mixed_collapse_full_timepoint"
            )
            step = int(point.get("update_index", point.get("step")))
            if step not in expected:
                continue
            if step in points:
                raise ValueError(f"duplicate dense step: {path}/{step}")
            values = np.asarray(point["energy"], dtype=np.float64).reshape(-1)
            if values.shape != (map_count,) or not np.all(np.isfinite(values)):
                raise ValueError(f"malformed dense energy vector: {path}/{step}")
            if np.any(values < 0.0):
                raise ValueError(f"negative dense energy: {path}/{step}")
            points[step] = values
    if set(points) != expected:
        missing = sorted(expected - set(points))
        raise ValueError(f"dense schedule is incomplete: {path}/{missing}")
    return points


def load_windows(dense: dict[str, Any]) -> list[dict[str, Any]]:
    specification_path = Path(
        str(dense["specification"]["path"])
    ).resolve(strict=True)
    if sha256_path(specification_path) != dense["specification"]["sha256"]:
        raise ValueError("dense specification digest changed")
    specification = load_json(specification_path)
    centers = [int(value) for value in specification["dense_centers"]]
    half_width = int(specification["dense_half_width"])
    expected = {int(value) for value in specification["dense_snapshot_updates"]}
    map_count = int(specification["registry"]["map_count"])
    windows = []
    for log in dense["logs"]:
        points = _dense_points(log, expected, map_count)
        for center in centers:
            steps = [
                step for step in sorted(points)
                if center - half_width <= step <= center + half_width
            ]
            if len(steps) >= 2:
                windows.append({
                    "trajectory": log["trajectory"],
                    "center": center,
                    "steps": steps,
                    "energy": np.stack([points[step] for step in steps]),
                })
    return windows


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty population")
    return {
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(np.max(values)),
    }


def _sign_counts(values: np.ndarray) -> dict[str, int]:
    return {
        "contracting": int(np.count_nonzero(values < 1.0)),
        "preserving": int(np.count_nonzero(values == 1.0)),
        "expanding": int(np.count_nonzero(values > 1.0)),
    }


def block_gain_census(
    windows: list[dict[str, Any]], effect_floor: float,
) -> dict[str, Any]:
    if len(windows) != 18:
        raise ValueError(f"expected 18 dense windows, found {len(windows)}")
    output: dict[str, Any] = {
        "window_count": len(windows),
        "all_stored_energies_strictly_positive": bool(all(
            np.all(row["energy"] > 0.0) for row in windows
        )),
        "horizons": {},
    }
    for horizon in BLOCK_LENGTHS:
        all_ratios = []
        resolved_ratios = []
        censored_ratios = []
        excursions = []
        windows_with_expansion = 0
        window_rows = []
        for row in windows:
            sequence = row["energy"]
            if sequence.shape[0] - 1 < horizon:
                continue
            row_ratios = []
            for start in range(sequence.shape[0] - horizon):
                source = sequence[start]
                endpoint = sequence[start + horizon]
                if np.any(source <= 0.0) or np.any(endpoint <= 0.0):
                    raise ValueError("block gain requires positive stored endpoints")
                ratio = endpoint / source
                excursion = (
                    np.max(sequence[start:start + horizon + 1], axis=0)
                    / source - 1.0
                )
                resolved = (source > effect_floor) & (endpoint > effect_floor)
                all_ratios.append(ratio)
                resolved_ratios.append(ratio[resolved])
                censored_ratios.append(ratio[~resolved])
                excursions.append(excursion)
                row_ratios.append(ratio)
            flattened = np.concatenate(row_ratios)
            expands = bool(np.any(flattened > 1.0))
            windows_with_expansion += int(expands)
            window_rows.append({
                "trajectory": row["trajectory"],
                "center": row["center"],
                "comparisons": int(flattened.size),
                "expanding": int(np.count_nonzero(flattened > 1.0)),
                "contains_expansion": expands,
                "maximum_gain": float(np.max(flattened)),
            })
        ratios = np.concatenate(all_ratios)
        resolved = np.concatenate(resolved_ratios)
        censored_parts = [part for part in censored_ratios if part.size]
        censored = (
            np.concatenate(censored_parts) if censored_parts
            else np.asarray([], dtype=np.float64)
        )
        excursion_values = np.concatenate(excursions)
        counts = _sign_counts(ratios)
        resolved_counts = _sign_counts(resolved)
        output["horizons"][str(horizon)] = {
            "comparisons": int(ratios.size),
            **counts,
            "contracting_fraction": float(counts["contracting"] / ratios.size),
            "endpoint_gain_quantiles": _quantiles(ratios),
            "maximum_relative_excursion_quantiles": _quantiles(excursion_values),
            "resolved": {
                "comparisons": int(resolved.size),
                **resolved_counts,
                "endpoint_gain_quantiles": _quantiles(resolved),
            },
            "censored_endpoint_population": {
                "comparisons": int(censored.size),
                **(_sign_counts(censored) if censored.size else {
                    "contracting": 0, "preserving": 0, "expanding": 0,
                }),
            },
            "contributing_windows": len(window_rows),
            "windows_with_expansion": windows_with_expansion,
            "every_window_contains_expansion": (
                windows_with_expansion == len(window_rows)
            ),
            "uniform_stored_block_contraction_observed": bool(
                np.all(ratios < 1.0)
            ),
            "windows": window_rows,
        }
    return output


def build_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("observer-block inventory schema changed")
    effect_floor = float(inventory["effect_floor"])
    census = observer_census(inventory["energy_records"], effect_floor)
    block = block_gain_census(load_windows(inventory["dense_archive"]), effect_floor)
    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "target_release_id": inventory["target_release_id"],
        "input_inventory_sha256": inventory["content_sha256"],
        "observer_census": census,
        "block_gain_census": block,
        "inferential_scope": {
            "observer_partition_is_exact_for_stored_energies": True,
            "effect_floor_is_an_eligibility_rule_not_an_exact_zero": True,
            "block_windows_overlap": True,
            "block_counts_are_descriptive_not_independent_samples": True,
            "finite_windows_do_not_establish_an_all_future_premise": True,
        },
    }
    summary["content_sha256"] = hashlib.sha256(canonical_bytes(summary)).hexdigest()
    return summary


def _tex_integer(value: int) -> str:
    return f"{value:,}"


def _tex_scientific(value: float, digits: int = 6) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0 ** exponent
    return f"{coefficient:.{digits}f}\\times10^{{{exponent}}}"








