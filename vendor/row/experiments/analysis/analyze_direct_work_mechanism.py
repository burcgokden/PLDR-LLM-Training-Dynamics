#!/usr/bin/env python3
"""Analyze the exact state-conditional paired direct-work contrast."""

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

from analysis.analyze_direct_work_precision import (  # noqa: E402
    array,
    canonical_bytes,
    scalar,
    sha256_path,
    write_bytes,
)
from analysis.sign_semantics import summarize_sign_counts  # noqa: E402


CAMPAIGN_ID = "pldr-direct-work-temporal-context-replication-v2"
RECORD_SCHEMA = "pldr-direct-work-replication-record-v1"
ANALYSIS_SCHEMA = "pldr-direct-work-mechanism-analysis-v1"
INVENTORY_SCHEMA = "pldr-direct-work-mechanism-inventory-v1"
MAP_SHAPE = (24, 4, 64, 64)
MAPS_PER_UNIT = 96
SOURCE_STEPS = (18_000, 32_768, 65_536)
TRAJECTORIES = ("A", "B", "C")
LAYERS = (0, 1, 2)
COMPONENTS = ("shape", "gate", "interaction", "implementation_defect")
EFFECT_FLOOR = 1.0e-12


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    write_bytes(path, canonical_bytes(value))


def component_arrays(record: Any, arm: str) -> np.ndarray:
    shape = array(record, f"{arm}_shape_secant", MAP_SHAPE)
    gate = array(record, f"{arm}_gate_secant", MAP_SHAPE)
    interaction = array(record, f"{arm}_interaction_secant", MAP_SHAPE)
    source_defect = array(
        record, f"{arm}_source_factorization_residual", MAP_SHAPE
    )
    endpoint_defect = array(
        record, f"{arm}_endpoint_factorization_residual", MAP_SHAPE
    )
    defect = endpoint_defect - source_defect
    return np.stack((shape, gate, interaction, defect), axis=2).reshape(
        MAPS_PER_UNIT, 4, -1
    )


def majority(contrast: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = contrast[mask]
    positive = int(np.count_nonzero(selected > EFFECT_FLOOR))
    negative = int(np.count_nonzero(selected < -EFFECT_FLOOR))
    neutral = int(len(selected) - positive - negative)
    return summarize_sign_counts(
        planned=int(len(contrast)), evaluable=int(len(selected)),
        positive=positive, negative=negative, neutral=neutral,
    )


def load_unit(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["path"])).resolve(strict=True)
    if sha256_path(path) != row.get("sha256"):
        raise ValueError(f"mechanism input changed: {path}")
    with np.load(path, allow_pickle=False) as record:
        if (
            scalar(record, "schema_version", str) != RECORD_SCHEMA
            or scalar(record, "campaign_id", str) != CAMPAIGN_ID
            or scalar(record, "trajectory", str) != row.get("trajectory")
            or scalar(record, "source_step", int) != row.get("step")
            or scalar(record, "layer", int) != row.get("layer")
        ):
            raise ValueError(f"mechanism input identity changed: {path}")
        source = array(record, "source_z", MAP_SHAPE).reshape(
            MAPS_PER_UNIT, -1
        )
        natural_endpoint = array(
            record, "natural_endpoint_z", MAP_SHAPE
        ).reshape(MAPS_PER_UNIT, -1)
        control_endpoint = array(
            record, "control_endpoint_z", MAP_SHAPE
        ).reshape(MAPS_PER_UNIT, -1)
        contrast = array(
            record, "intervention_contrast", (24, 4)
        ).reshape(-1)
        source_energy = array(record, "source_energy", (24, 4)).reshape(-1)
        natural_components = component_arrays(record, "natural")
        control_components = component_arrays(record, "control")

    natural_increment = natural_endpoint - source
    paired_increment = control_endpoint - natural_endpoint
    source_projection = 2.0 * np.einsum(
        "ma,ma->m", source, paired_increment
    )
    natural_interaction = 2.0 * np.einsum(
        "ma,ma->m", natural_increment, paired_increment
    )
    pair_charge = np.einsum(
        "ma,ma->m", paired_increment, paired_increment
    )
    paired_residual = (
        contrast - source_projection - natural_interaction - pair_charge
    )
    energy_magnitude = (
        np.sum(np.abs(control_endpoint * control_endpoint), axis=1)
        + np.sum(np.abs(natural_endpoint * natural_endpoint), axis=1)
        + np.abs(source_projection)
        + np.abs(natural_interaction)
        + pair_charge
        + np.abs(contrast)
    )
    epsilon = np.finfo(np.float64).eps
    operations = source.shape[1] + 32
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    paired_bound = 2.0 * gamma * energy_magnitude + np.finfo(np.float64).tiny
    if np.any(np.abs(paired_residual) > paired_bound):
        raise ArithmeticError(f"paired contrast identity failed: {path}")

    source_difference = control_components - natural_components
    four_source_increment = np.sum(source_difference, axis=1)
    four_source_residual = paired_increment - four_source_increment
    coordinate_magnitude = (
        np.abs(control_endpoint)
        + np.abs(natural_endpoint)
        + np.sum(np.abs(source_difference), axis=1)
        + np.abs(paired_increment)
    )
    coordinate_bound = (
        128.0 * epsilon * coordinate_magnitude + np.finfo(np.float64).tiny
    )
    if np.any(np.abs(four_source_residual) > coordinate_bound):
        raise ArithmeticError(f"paired four-source closure failed: {path}")

    effect = (source_energy > EFFECT_FLOOR) | (
        np.abs(contrast) > EFFECT_FLOOR
    )
    terms = np.stack(
        (source_projection, natural_interaction, pair_charge), axis=1
    )
    paired_norm = np.linalg.norm(paired_increment, axis=1)
    source_norm = np.linalg.norm(source_difference, axis=2)
    source_ratio = np.divide(
        source_norm,
        paired_norm[:, None],
        out=np.zeros_like(source_norm),
        where=paired_norm[:, None] > 0.0,
    )
    selected_terms = terms[effect]
    selected_contrast = contrast[effect]
    cancellation = np.divide(
        np.sum(np.abs(selected_terms), axis=1),
        np.maximum(np.abs(selected_contrast), np.finfo(np.float64).tiny),
    )
    summary = majority(contrast, effect)
    summary.update({
        "trajectory": row["trajectory"],
        "role": row["role"],
        "step": int(row["step"]),
        "layer": int(row["layer"]),
        "median_contrast": (
            float(np.median(selected_contrast)) if len(selected_contrast) else 0.0
        ),
        "median_source_projection": (
            float(np.median(selected_terms[:, 0])) if len(selected_terms) else 0.0
        ),
        "median_natural_interaction": (
            float(np.median(selected_terms[:, 1])) if len(selected_terms) else 0.0
        ),
        "median_pair_charge": (
            float(np.median(selected_terms[:, 2])) if len(selected_terms) else 0.0
        ),
        "median_cancellation_index": (
            float(np.median(cancellation)) if len(cancellation) else 0.0
        ),
        "record_sha256": row["sha256"],
    })
    return {
        "summary": summary,
        "contrast": contrast,
        "effect": effect,
        "terms": terms,
        "cancellation": cancellation,
        "source_ratio": source_ratio,
        "paired_residual": paired_residual,
        "paired_bound": paired_bound,
        "four_source_residual": four_source_residual,
        "coordinate_bound": coordinate_bound,
    }


def sign_agreement(contrast: np.ndarray, term: np.ndarray) -> dict[str, Any]:
    selected = (np.abs(contrast) > EFFECT_FLOOR) & (
        np.abs(term) > EFFECT_FLOOR
    )
    count = int(np.count_nonzero(selected))
    agreeing = int(np.count_nonzero(
        np.signbit(contrast[selected]) == np.signbit(term[selected])
    ))
    return {
        "eligible": count,
        "agreeing": agreeing,
        "fraction": float(agreeing / count) if count else 0.0,
    }


def mechanism_summary(inventory: dict[str, Any]) -> dict[str, Any]:
    if (
        inventory.get("schema_version") != INVENTORY_SCHEMA
        or inventory.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("mechanism inventory identity changed")
    units = [load_unit(row) for row in inventory["science_records"]]
    contrast = np.concatenate([unit["contrast"] for unit in units])
    effect = np.concatenate([unit["effect"] for unit in units])
    terms = np.concatenate([unit["terms"] for unit in units], axis=0)
    source_ratios = np.concatenate(
        [unit["source_ratio"] for unit in units], axis=0
    )
    selected_terms = terms[effect]
    selected_contrast = contrast[effect]
    dominant = np.argmax(np.abs(selected_terms), axis=1)
    cancellation = np.concatenate([unit["cancellation"] for unit in units])
    terminal = [
        unit for unit in units if unit["summary"]["step"] == 65_536
    ]
    terminal_ratios = np.concatenate(
        [unit["source_ratio"] for unit in terminal], axis=0
    )
    summaries = [unit["summary"] for unit in units]
    heldout_layer2 = [
        row for row in summaries
        if row["role"] == "heldout" and row["layer"] == 2
    ]
    reversal_cells = []
    for trajectory in TRAJECTORIES:
        for layer in LAYERS:
            signs = [
                row["majority_sign"]
                for row in summaries
                if row["trajectory"] == trajectory and row["layer"] == layer
                and row["majority_sign"] is not None
                and row["majority_fraction"] >= 2.0 / 3.0
            ]
            if len(set(signs)) > 1:
                reversal_cells.append({
                    "trajectory": trajectory,
                    "layer": layer,
                    "qualifying_signs": signs,
                })
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_campaign_root": inventory["source_campaign_root"],
        "science_record_count": len(units),
        "planned_maps": int(len(contrast)),
        "evaluable_maps": int(np.count_nonzero(effect)),
        "paired_contrast_max_abs_residual": float(max(
            np.max(np.abs(unit["paired_residual"])) for unit in units
        )),
        "paired_contrast_max_bound_fraction": float(max(
            np.max(np.divide(
                np.abs(unit["paired_residual"]),
                unit["paired_bound"],
            )) for unit in units
        )),
        "paired_four_source_max_abs_residual": float(max(
            np.max(np.abs(unit["four_source_residual"])) for unit in units
        )),
        "paired_four_source_max_bound_fraction": float(max(
            np.max(np.divide(
                np.abs(unit["four_source_residual"]),
                unit["coordinate_bound"],
            )) for unit in units
        )),
        "term_order": [
            "source_projection", "natural_interaction", "pair_charge"
        ],
        "term_sign_agreement": {
            "source_projection": sign_agreement(
                selected_contrast, selected_terms[:, 0]
            ),
            "natural_interaction": sign_agreement(
                selected_contrast, selected_terms[:, 1]
            ),
            "pair_charge": sign_agreement(
                selected_contrast, selected_terms[:, 2]
            ),
        },
        "dominant_absolute_term_counts": {
            "source_projection": int(np.count_nonzero(dominant == 0)),
            "natural_interaction": int(np.count_nonzero(dominant == 1)),
            "pair_charge": int(np.count_nonzero(dominant == 2)),
        },
        "cancellation_index_median": float(np.median(cancellation)),
        "cancellation_index_p95": float(np.quantile(cancellation, 0.95)),
        "cancellation_index_maximum": float(np.max(cancellation)),
        "paired_source_order": list(COMPONENTS),
        "paired_source_norm_ratio_median": [
            float(value) for value in np.median(source_ratios, axis=0)
        ],
        "paired_source_norm_ratio_p95": [
            float(value) for value in np.quantile(source_ratios, 0.95, axis=0)
        ],
        "terminal_paired_source_norm_ratio_median": [
            float(value) for value in np.median(terminal_ratios, axis=0)
        ],
        "terminal_defect_dominant_maps": int(np.count_nonzero(
            np.argmax(terminal_ratios, axis=1) == 3
        )),
        "temporal_sign_reversal_cells": reversal_cells,
        "unit_summaries": summaries,
        "heldout_layer2_summaries": heldout_layer2,
        "interpretation": (
            "descriptive exact decomposition after the locked replication; "
            "it does not alter the refuted campaign outcome"
        ),
    }


def format_float(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    coefficient = value / 10.0**exponent
    return f"{coefficient:.3f}\\times10^{{{exponent}}}"


def format_math(value: float) -> str:
    return f"\\({format_float(value)}\\)"








