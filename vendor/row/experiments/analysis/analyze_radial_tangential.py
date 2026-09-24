#!/usr/bin/env python3
"""Analyze exact radial--tangential identities in sealed row-map records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from analysis.analyze_direct_work_precision import (  # noqa: E402
    array,
    canonical_bytes,
    scalar,
    sha256_path,
    write_bytes,
)
from analysis.analyze_direct_work_replication import (  # noqa: E402
    MAP_SHAPE,
    arm_metrics,
    scalar_invariants,
)
from analysis.radial_tangential import (  # noqa: E402
    COMPONENT_NAMES,
    closing_masks,
    paired_decomposition,
    radial_decomposition,
    source_resolved_decomposition,
)


SCHEMA = "pldr-radial-tangential-analysis-v1"
INVENTORY_SCHEMA = "pldr-radial-tangential-inventory-v1"
TERM_NAMES = (
    "radial_linear",
    "radial_charge",
    "tangent_interaction",
    "tangent_charge",
)
STEP_PATTERN = re.compile(r'"step"\s*:\s*(\d+)')


def _write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, canonical_bytes(value))


def _finite_scalar(value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("analysis produced a nonfinite scalar")
    return result


def _quantiles(values: np.ndarray) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("quantiles require finite data")
    return {
        "minimum": _finite_scalar(np.min(finite)),
        "p05": _finite_scalar(np.quantile(finite, 0.05)),
        "median": _finite_scalar(np.median(finite)),
        "p95": _finite_scalar(np.quantile(finite, 0.95)),
        "maximum": _finite_scalar(np.max(finite)),
    }


def _sign(values: np.ndarray, floor: float) -> np.ndarray:
    result = np.zeros(np.asarray(values).shape, dtype=np.int8)
    result[np.asarray(values) > floor] = 1
    result[np.asarray(values) < -floor] = -1
    return result


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _verify_inventory(inventory_path: Path) -> dict[str, Any]:
    value = json.loads(inventory_path.read_text(encoding="utf-8"))
    if value.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("unknown radial inventory schema")
    for row in value["science_records"]:
        path = Path(row["path"]).resolve(strict=True)
        if path.is_symlink() or sha256_path(path) != row["sha256"]:
            raise ValueError(f"radial science input changed: {path}")
    dense = value["dense_archive"]
    for row in [dense["specification"], *dense["logs"]]:
        path = Path(row["path"]).resolve(strict=True)
        if path.is_symlink() or sha256_path(path) != row["sha256"]:
            raise ValueError(f"dense product input changed: {path}")
    return value


def _identity_bound(*magnitudes: np.ndarray, operations: int) -> np.ndarray:
    epsilon = np.finfo(np.float64).eps
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    total = np.add.reduce([np.abs(value) for value in magnitudes])
    return 128.0 * gamma * total + np.finfo(np.float64).tiny


def _nullable(value: float, eligible: bool) -> float | None:
    return _finite_scalar(value) if eligible else None


def _ledger_line(
    *,
    record: dict[str, Any],
    index: int,
    source_energy: np.ndarray,
    arm_results: dict[str, dict[str, Any]],
    source_results: dict[str, dict[str, Any]],
    paired: dict[str, np.ndarray],
    paired_eligible: np.ndarray,
    stored_contrast: np.ndarray,
    effect_floor: float,
) -> dict[str, Any]:
    eligible = bool(paired_eligible[index])
    arms: dict[str, Any] = {}
    for arm in ("natural", "control"):
        radial = arm_results[arm]["radial"]
        resolved = source_results[arm]
        arm_eligible = bool(arm_results[arm]["eligible"][index])
        masks = arm_results[arm]["masks"]
        label = "ineligible"
        for name in ("contract", "preserve", "reopen"):
            if bool(masks[name][index]):
                label = name
        arms[arm] = {
            "eligible": arm_eligible,
            "class": label,
            "alpha": _nullable(radial["alpha"][index], arm_eligible),
            "tau_squared": _nullable(radial["tau_squared"][index], arm_eligible),
            "gain_squared": _nullable(radial["gain_squared"][index], arm_eligible),
            "direct_gain": _nullable(radial["direct_gain"][index], arm_eligible),
            "orthogonality_residual": _nullable(
                radial["orthogonality_residual"][index], arm_eligible
            ),
            "gain_residual": _nullable(radial["gain_residual"][index], arm_eligible),
            "source_alpha": (
                [_finite_scalar(value) for value in resolved["alpha"][index]]
                if arm_eligible else None
            ),
            "tangent_gram_normalized": (
                [[_finite_scalar(value) for value in row]
                 for row in resolved["tangent_gram_normalized"][index]]
                if arm_eligible else None
            ),
        }
    paired_terms = paired["normalized_terms"][index]
    radial_energy = paired["radial_normalized"][index] * source_energy[index]
    full_floor_sign = int(_sign(np.asarray([stored_contrast[index]]), effect_floor)[0])
    radial_floor_sign = int(_sign(np.asarray([radial_energy]), effect_floor)[0]) if eligible else 0
    full_raw_sign = int(np.sign(stored_contrast[index])) if eligible else 0
    radial_raw_sign = int(np.sign(radial_energy)) if eligible else 0
    dominant = TERM_NAMES[int(np.argmax(np.abs(paired_terms)))] if eligible else None
    return {
        "trajectory": record["trajectory"],
        "step": record["step"],
        "layer": record["layer"],
        "context": index // 4,
        "head": index % 4,
        "source_energy": _finite_scalar(source_energy[index]),
        "arms": arms,
        "paired": {
            "eligible": eligible,
            "native_contrast": _finite_scalar(stored_contrast[index]),
            "normalized_terms": (
                [_finite_scalar(value) for value in paired_terms] if eligible else None
            ),
            "normalized_contrast": _nullable(
                paired["normalized_contrast"][index], eligible
            ),
            "identity_residual": _nullable(
                paired["identity_residual"][index], eligible
            ),
            "radial_energy": _nullable(radial_energy, eligible),
            "tangent_energy": _nullable(
                paired["tangent_normalized"][index] * source_energy[index], eligible
            ),
            "full_raw_sign": full_raw_sign,
            "radial_raw_sign": radial_raw_sign,
            "raw_sign_agreement": bool(
                eligible and full_raw_sign != 0 and radial_raw_sign == full_raw_sign
            ),
            "full_floor_sign": full_floor_sign,
            "radial_floor_sign": radial_floor_sign,
            "floor_sign_agreement": bool(
                eligible and full_floor_sign != 0
                and radial_floor_sign == full_floor_sign
            ),
            "dominant_term": dominant,
        },
    }


def _record_analysis(row: dict[str, Any], effect_floor: float) -> dict[str, Any]:
    path = Path(row["path"])
    with np.load(path, allow_pickle=False) as record:
        identity = {
            "trajectory": scalar(record, "trajectory", str),
            "step": scalar(record, "source_step", int),
            "layer": scalar(record, "layer", int),
            "campaign_id": scalar(record, "campaign_id", str),
            "release_id": scalar(record, "release_id", str),
            "schema_version": scalar(record, "schema_version", str),
        }
        expected = {
            name: row[name] for name in (
                "trajectory", "step", "layer", "campaign_id", "release_id",
                "schema_version",
            )
        }
        if identity != expected:
            raise ValueError(f"radial record identity changed: {path}")
        source = array(record, "source_z", MAP_SHAPE)
        source_flat = source.reshape(96, -1)
        stored_energy = array(record, "source_energy", (24, 4)).reshape(-1)
        computed_energy = np.einsum("mi,mi->m", source_flat, source_flat)
        scalar_pass = scalar_invariants(record)
        arms: dict[str, dict[str, Any]] = {}
        sources: dict[str, dict[str, Any]] = {}
        for arm in ("natural", "control"):
            technical = arm_metrics(record, arm, source)
            increment = technical["increment"].reshape(96, -1)
            radial = radial_decomposition(
                source_flat, increment, energy_floor=effect_floor
            )
            eligible = technical["technical_mask"] & radial["eligible"]
            if not scalar_pass:
                eligible[:] = False
            masks = closing_masks(radial)
            for name in masks:
                masks[name] &= eligible
            source_defect = array(
                record, f"{arm}_source_factorization_residual", MAP_SHAPE
            )
            endpoint_defect = array(
                record, f"{arm}_endpoint_factorization_residual", MAP_SHAPE
            )
            components = np.stack(
                (
                    array(record, f"{arm}_shape_secant", MAP_SHAPE),
                    array(record, f"{arm}_gate_secant", MAP_SHAPE),
                    array(record, f"{arm}_interaction_secant", MAP_SHAPE),
                    endpoint_defect - source_defect,
                ),
                axis=2,
            ).reshape(96, 4, -1)
            resolved = source_resolved_decomposition(
                source_flat, components, energy_floor=effect_floor
            )
            arms[arm] = {
                "technical": technical,
                "radial": radial,
                "eligible": eligible,
                "masks": masks,
                "increment": increment,
            }
            sources[arm] = resolved
        stored_contrast = array(record, "intervention_contrast", (24, 4)).reshape(-1)
        paired = paired_decomposition(
            source_flat,
            arms["natural"]["increment"],
            arms["control"]["increment"],
            energy_floor=effect_floor,
        )
        paired_eligible = (
            arms["natural"]["eligible"]
            & arms["control"]["eligible"]
            & paired["eligible"]
        )
        ledger = [
            _ledger_line(
                record=identity,
                index=index,
                source_energy=stored_energy,
                arm_results=arms,
                source_results=sources,
                paired=paired,
                paired_eligible=paired_eligible,
                stored_contrast=stored_contrast,
                effect_floor=effect_floor,
            )
            for index in range(96)
        ]
        return {
            "identity": identity,
            "sha256": row["sha256"],
            "source_energy": stored_energy,
            "computed_energy": computed_energy,
            "scalar_pass": scalar_pass,
            "arms": arms,
            "sources": sources,
            "paired": paired,
            "paired_eligible": paired_eligible,
            "stored_contrast": stored_contrast,
            "ledger": ledger,
        }


def _dense_points(log_path: Path, expected_steps: set[int], map_count: int) -> dict[int, np.ndarray]:
    points: dict[int, np.ndarray] = {}
    with log_path.open(encoding="utf-8") as stream:
        for line in stream:
            if '"mixed_collapse_dense_timepoint"' not in line and '"mixed_collapse_full_timepoint"' not in line:
                continue
            match = STEP_PATTERN.search(line)
            if match is None or int(match.group(1)) not in expected_steps:
                continue
            payload = json.loads(line)
            point = payload.get("mixed_collapse_dense_timepoint")
            if point is None:
                point = payload.get("mixed_collapse_full_timepoint")
            if not isinstance(point, dict):
                continue
            step = int(point["step"])
            if step not in expected_steps:
                continue
            energy = np.asarray(point["energy"], dtype=np.float64)
            if energy.shape != (map_count,) or np.any(energy < 0.0) or not np.all(np.isfinite(energy)):
                raise ValueError("dense product energy vector is malformed")
            if step in points:
                raise ValueError("dense product schedule contains duplicates")
            points[step] = energy
    if set(points) != expected_steps:
        raise ValueError("dense product schedule is incomplete")
    return points


def _product_analysis(dense: dict[str, Any]) -> dict[str, Any]:
    specification = json.loads(Path(dense["specification"]["path"]).read_text(encoding="utf-8"))
    registry = specification["registry"]
    expected_steps = {int(value) for value in specification["dense_snapshot_updates"]}
    centers = [int(value) for value in specification["dense_centers"]]
    half_width = int(specification["dense_half_width"])
    terminal = int(specification["terminal_update"])
    map_count = int(registry["map_count"])
    positive_residuals: list[np.ndarray] = []
    affine_residuals: list[np.ndarray] = []
    positive_map_windows = 0
    face_map_windows = 0
    positive_edges = 0
    face_edges = 0
    rows = []
    for log in dense["logs"]:
        points = _dense_points(Path(log["path"]), expected_steps, map_count)
        for center in centers:
            left = max(0, center - half_width)
            right = min(terminal, center + half_width)
            block = np.stack([points[step] for step in range(left, right + 1)])
            source = block[:-1]
            endpoint = block[1:]
            positive = np.all(source > 0.0, axis=0)
            gain = np.divide(endpoint, source, out=np.zeros_like(source), where=source > 0.0)
            reopening = np.where(source > 0.0, 0.0, endpoint)
            direct_product = np.prod(gain[:, positive], axis=0) if np.any(positive) else np.empty(0)
            reconstructed = block[0, positive] * direct_product
            positive_error = np.abs(reconstructed - block[-1, positive])
            positive_scale = np.maximum.reduce((
                np.abs(reconstructed), np.abs(block[-1, positive]),
                np.full(reconstructed.shape, np.finfo(np.float64).tiny),
            ))
            positive_relative = positive_error / positive_scale
            affine = block[0].copy()
            for edge in range(gain.shape[0]):
                affine = gain[edge] * affine + reopening[edge]
            affine_error = np.abs(affine - block[-1])
            affine_scale = np.maximum.reduce((
                np.abs(affine), np.abs(block[-1]),
                np.full(affine.shape, np.finfo(np.float64).tiny),
            ))
            affine_relative = affine_error / affine_scale
            positive_residuals.append(positive_relative)
            affine_residuals.append(affine_relative)
            positive_map_windows += int(np.count_nonzero(positive))
            face_map_windows += int(np.count_nonzero(~positive))
            positive_edges += int(np.count_nonzero(source > 0.0))
            face_edges += int(np.count_nonzero(source == 0.0))
            rows.append({
                "trajectory": log["trajectory"],
                "center": center,
                "edge_count": right - left,
                "positive_map_windows": int(np.count_nonzero(positive)),
                "face_map_windows": int(np.count_nonzero(~positive)),
                "positive_product_max_relative_residual": (
                    _finite_scalar(np.max(positive_relative)) if len(positive_relative) else 0.0
                ),
                "affine_face_max_relative_residual": _finite_scalar(np.max(affine_relative)),
            })
    positive_all = np.concatenate(positive_residuals) if positive_residuals else np.empty(0)
    affine_all = np.concatenate(affine_residuals)
    return {
        "window_count": len(rows),
        "map_windows": len(rows) * map_count,
        "positive_map_windows": positive_map_windows,
        "face_map_windows": face_map_windows,
        "positive_source_edges": positive_edges,
        "zero_face_edges": face_edges,
        "positive_product_max_relative_residual": (
            _finite_scalar(np.max(positive_all)) if len(positive_all) else 0.0
        ),
        "affine_face_max_relative_residual": _finite_scalar(np.max(affine_all)),
        "all_finite_products_reconstructed": bool(
            np.max(positive_all, initial=0.0) <= 1.0e-12
            and np.max(affine_all, initial=0.0) <= 1.0e-12
        ),
        "rows": rows,
        "scope": "retained-scalar-energy-only",
        "radial_components_available": False,
    }


def _aggregate(records: list[dict[str, Any]], product: dict[str, Any], effect_floor: float) -> dict[str, Any]:
    arm_alpha = []
    arm_tau = []
    arm_gain = []
    arm_counts = {name: 0 for name in ("eligible", "contract", "preserve", "reopen", "radial_failure", "tangent_excess")}
    source_alpha: dict[str, list[np.ndarray]] = {name: [] for name in ("natural", "control")}
    source_gram: dict[str, list[np.ndarray]] = {name: [] for name in ("natural", "control")}
    maximum = {
        "source_energy_disagreement": 0.0,
        "orthogonality_residual": 0.0,
        "gain_identity_residual": 0.0,
        "source_alpha_recomposition": 0.0,
        "source_tangent_recomposition": 0.0,
        "source_gram_recomposition": 0.0,
        "paired_identity_residual": 0.0,
        "stored_contrast_disagreement": 0.0,
    }
    dominance = {name: 0 for name in TERM_NAMES}
    raw_agreements = 0
    raw_comparisons = 0
    floor_agreements = 0
    floor_comparisons = 0
    paired_eligible_count = 0
    cells = []
    all_scalar_pass = True
    all_native_technical = True
    for record in records:
        maximum["source_energy_disagreement"] = max(
            maximum["source_energy_disagreement"],
            _finite_scalar(np.max(np.abs(record["source_energy"] - record["computed_energy"]))),
        )
        all_scalar_pass &= bool(record["scalar_pass"])
        cell: dict[str, Any] = {
            **{name: record["identity"][name] for name in ("trajectory", "step", "layer")},
            "planned_maps": 96,
            "source_eligible_maps": int(np.count_nonzero(record["source_energy"] > effect_floor)),
            "arms": {},
        }
        for arm in ("natural", "control"):
            values = record["arms"][arm]
            radial = values["radial"]
            eligible = values["eligible"]
            masks = values["masks"]
            all_native_technical &= bool(np.all(values["technical"]["technical_mask"]))
            count = int(np.count_nonzero(eligible))
            arm_counts["eligible"] += count
            for name in ("contract", "preserve", "reopen", "radial_failure", "tangent_excess"):
                arm_counts[name] += int(np.count_nonzero(masks[name]))
            arm_alpha.append(radial["alpha"][eligible])
            arm_tau.append(radial["tau_squared"][eligible])
            arm_gain.append(radial["gain_squared"][eligible])
            if count:
                maximum["orthogonality_residual"] = max(
                    maximum["orthogonality_residual"],
                    _finite_scalar(np.max(np.abs(radial["orthogonality_residual"][eligible]))),
                )
                maximum["gain_identity_residual"] = max(
                    maximum["gain_identity_residual"],
                    _finite_scalar(np.max(np.abs(radial["gain_residual"][eligible]))),
                )
                resolved = record["sources"][arm]
                source_alpha[arm].append(resolved["alpha"][eligible])
                source_gram[arm].append(resolved["tangent_gram_normalized"][eligible])
                alpha_error = np.abs(resolved["total_alpha"][eligible] - radial["alpha"][eligible])
                tangent_error = np.linalg.norm(
                    resolved["total_tangent"][eligible] - radial["tangent"][eligible], axis=1
                )
                gram_error = np.abs(
                    np.sum(resolved["tangent_gram_normalized"][eligible], axis=(1, 2))
                    - radial["tau_squared"][eligible]
                )
                maximum["source_alpha_recomposition"] = max(
                    maximum["source_alpha_recomposition"], _finite_scalar(np.max(alpha_error))
                )
                maximum["source_tangent_recomposition"] = max(
                    maximum["source_tangent_recomposition"], _finite_scalar(np.max(tangent_error))
                )
                maximum["source_gram_recomposition"] = max(
                    maximum["source_gram_recomposition"], _finite_scalar(np.max(gram_error))
                )
            cell["arms"][arm] = {
                "eligible": count,
                **{name: int(np.count_nonzero(masks[name])) for name in (
                    "contract", "preserve", "reopen", "radial_failure", "tangent_excess"
                )},
            }
        eligible = record["paired_eligible"]
        paired = record["paired"]
        count = int(np.count_nonzero(eligible))
        paired_eligible_count += count
        terms = paired["normalized_terms"][eligible]
        if count:
            indices = np.argmax(np.abs(terms), axis=1)
            for index, name in enumerate(TERM_NAMES):
                dominance[name] += int(np.count_nonzero(indices == index))
            residual = record["stored_contrast"][eligible] / record["source_energy"][eligible] - np.sum(terms, axis=1)
            maximum["paired_identity_residual"] = max(
                maximum["paired_identity_residual"], _finite_scalar(np.max(np.abs(residual)))
            )
            maximum["stored_contrast_disagreement"] = max(
                maximum["stored_contrast_disagreement"],
                _finite_scalar(np.max(np.abs(record["stored_contrast"][eligible] - paired["native_contrast"][eligible]))),
            )
            radial_energy = paired["radial_normalized"][eligible] * record["source_energy"][eligible]
            full_raw_sign = np.sign(record["stored_contrast"][eligible])
            radial_raw_sign = np.sign(radial_energy)
            raw_comparable = (full_raw_sign != 0) & (radial_raw_sign != 0)
            raw_local_agree = raw_comparable & (full_raw_sign == radial_raw_sign)
            raw_comparisons += int(np.count_nonzero(raw_comparable))
            raw_agreements += int(np.count_nonzero(raw_local_agree))
            full_floor_sign = _sign(record["stored_contrast"][eligible], effect_floor)
            radial_floor_sign = _sign(radial_energy, effect_floor)
            floor_comparable = (full_floor_sign != 0) & (radial_floor_sign != 0)
            floor_local_agree = floor_comparable & (full_floor_sign == radial_floor_sign)
            floor_comparisons += int(np.count_nonzero(floor_comparable))
            floor_agreements += int(np.count_nonzero(floor_local_agree))
        else:
            raw_comparable = np.empty(0, dtype=bool)
            raw_local_agree = np.empty(0, dtype=bool)
            floor_comparable = np.empty(0, dtype=bool)
            floor_local_agree = np.empty(0, dtype=bool)
        cell["paired"] = {
            "eligible": count,
            "raw_sign_comparisons": int(np.count_nonzero(raw_comparable)),
            "raw_sign_agreements": int(np.count_nonzero(raw_local_agree)),
            "raw_agreement_fraction": (
                _finite_scalar(np.count_nonzero(raw_local_agree) / np.count_nonzero(raw_comparable))
                if np.count_nonzero(raw_comparable) else None
            ),
            "floor_sign_comparisons": int(np.count_nonzero(floor_comparable)),
            "floor_sign_agreements": int(np.count_nonzero(floor_local_agree)),
            "floor_agreement_fraction": (
                _finite_scalar(np.count_nonzero(floor_local_agree) / np.count_nonzero(floor_comparable))
                if np.count_nonzero(floor_comparable) else None
            ),
            "dominant_terms": {
                name: int(np.count_nonzero(np.argmax(np.abs(terms), axis=1) == index))
                if count else 0
                for index, name in enumerate(TERM_NAMES)
            },
        }
        cells.append(cell)
    alpha_values = np.concatenate(arm_alpha)
    tau_values = np.concatenate(arm_tau)
    gain_values = np.concatenate(arm_gain)
    source_summary = {}
    mixed_sign = 0
    for arm in ("natural", "control"):
        alpha = np.concatenate(source_alpha[arm])
        gram = np.concatenate(source_gram[arm])
        mixed_sign += int(np.count_nonzero(
            (np.any(alpha > 0.0, axis=1) & np.any(alpha < 0.0, axis=1))
        ))
        source_summary[arm] = {
            "alpha_quantiles": {
                name: _quantiles(alpha[:, index])
                for index, name in enumerate(COMPONENT_NAMES)
            },
            "mean_tangent_gram_normalized": [
                [_finite_scalar(value) for value in row] for row in np.mean(gram, axis=0)
            ],
            "median_tangent_gram_normalized": [
                [_finite_scalar(value) for value in row] for row in np.median(gram, axis=0)
            ],
        }
    technical_pass = bool(
        all_scalar_pass
        and all_native_technical
        and maximum["gain_identity_residual"] <= 1.0e-12
        and maximum["paired_identity_residual"] <= 1.0e-12
        and maximum["source_alpha_recomposition"] <= 1.0e-12
        and maximum["source_gram_recomposition"] <= 1.0e-12
        and product["all_finite_products_reconstructed"]
    )
    return {
        "records": len(records),
        "planned_maps": len(records) * 96,
        "arm_map_states": 2 * len(records) * 96,
        "arm_counts": arm_counts,
        "alpha_quantiles": _quantiles(alpha_values),
        "tau_squared_quantiles": _quantiles(tau_values),
        "gain_squared_quantiles": _quantiles(gain_values),
        "source_resolved": {
            "component_order": list(COMPONENT_NAMES),
            "arm_summaries": source_summary,
            "mixed_radial_sign_states": mixed_sign,
        },
        "paired": {
            "eligible": paired_eligible_count,
            "dominant_terms": dominance,
            "raw_sign_comparisons": raw_comparisons,
            "raw_sign_agreements": raw_agreements,
            "raw_sign_disagreements": raw_comparisons - raw_agreements,
            "raw_agreement_fraction": _finite_scalar(raw_agreements / raw_comparisons),
            "floor_sign_comparisons": floor_comparisons,
            "floor_sign_agreements": floor_agreements,
            "floor_sign_disagreements": floor_comparisons - floor_agreements,
            "floor_agreement_fraction": _finite_scalar(floor_agreements / floor_comparisons),
        },
        "maximum_absolute_residuals": maximum,
        "technical": {
            "scalar_invariants_pass": all_scalar_pass,
            "native_four_source_pass": all_native_technical,
            "radial_source_and_product_checks_pass": technical_pass,
        },
        "cells": cells,
        "positive_excursion_product": product,
    }


def _independent_contract(aggregates: dict[str, Any]) -> dict[str, Any]:
    return {
        "records": aggregates["records"],
        "planned_maps": aggregates["planned_maps"],
        "arm_counts": aggregates["arm_counts"],
        "alpha_quantiles": aggregates["alpha_quantiles"],
        "tau_squared_quantiles": aggregates["tau_squared_quantiles"],
        "gain_squared_quantiles": aggregates["gain_squared_quantiles"],
        "paired": aggregates["paired"],
        "maximum_absolute_residuals": aggregates["maximum_absolute_residuals"],
        "cells": aggregates["cells"],
        "positive_excursion_product": aggregates["positive_excursion_product"],
    }


def build_report(inventory_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inventory = _verify_inventory(inventory_path)
    effect_floor = float(inventory["effect_floor"])
    records = [_record_analysis(row, effect_floor) for row in inventory["science_records"]]
    product = _product_analysis(inventory["dense_archive"])
    aggregates = _aggregate(records, product, effect_floor)
    report = {
        "schema_version": SCHEMA,
        "target_release_id": inventory["target_release_id"],
        "source_campaign_id": inventory["source_campaign_id"],
        "input_inventory_sha256": sha256_path(inventory_path),
        "effect_floor": effect_floor,
        "analysis_role": "sealed-retained-record-radial-tangential-measurement",
        "inferential_scope": {
            "identities_are_exact_not_empirical_hypotheses": True,
            "dominance_and_sign_agreement_are_descriptive": True,
            "eventual_contraction_is_not_inferred": True,
            "zero_face_reopenings_are_explicit": True,
        },
        "aggregates": aggregates,
        "record_sha256": {
            Path(row["path"]).name: row["sha256"] for row in inventory["science_records"]
        },
    }
    report["independent_contract"] = _independent_contract(aggregates)
    report["content_sha256"] = _sha256_json(report)
    ledger = [line for record in records for line in record["ledger"]]
    return report, ledger


def _tex_integer(value: int) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _tex_scientific(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / 10.0**exponent
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"










