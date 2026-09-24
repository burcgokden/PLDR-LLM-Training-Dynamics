#!/usr/bin/env python3
"""Independent occupied-segment PLGA and fixed-query-key analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_collapse import rectangular_score_transfer  # noqa: E402
from confirm.comprehensive_campaign import campaign_spec  # noqa: E402
from confirm.comprehensive_lock import load_lock  # noqa: E402
from confirm.gate_shape import plga_secant_chain  # noqa: E402
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402


RECORD_SCHEMA = "pldr-comprehensive-plga-ledger-v1"
REPORT_SCHEMA = "pldr-comprehensive-plga-report-v1"


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _scalar(record: Any, name: str) -> Any:
    value = np.asarray(record[name])
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    return value.item()


def _array(record: Any, name: str, ndim: int) -> np.ndarray:
    value = np.asarray(record[name], dtype=np.float64)
    if value.ndim != ndim or not np.isfinite(value).all():
        raise ValueError(f"{name} has invalid dimensions or values")
    return value


def _relative(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(
        float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1.0))


def analyze_record(path: str | Path, lock: dict[str, Any] | None) -> dict[str, Any]:
    spec = campaign_spec()
    source = Path(path).resolve()
    with np.load(source, allow_pickle=False) as record:
        if _scalar(record, "schema_version") != RECORD_SCHEMA:
            raise ValueError("unknown comprehensive PLGA schema")
        if (
            _scalar(record, "campaign_id") != spec["campaign_id"]
            or _scalar(record, "campaign_spec_sha256") != spec["spec_sha256"]
        ):
            raise ValueError("PLGA campaign identity changed")
        role = str(_scalar(record, "role"))
        seed = int(_scalar(record, "seed"))
        anchor = int(_scalar(record, "anchor"))
        layer = int(_scalar(record, "layer"))
        lock_sha256 = str(_scalar(record, "lock_sha256"))
        positive_floor = float(_scalar(record, "positive_floor"))
        generator = _array(record, "plga_generator", 3)
        collapsed = _array(record, "plga_collapsed_anchor_generator", 3)
        weight = _array(record, "plga_weight", 3)
        bias = _array(record, "plga_bias", 3)
        exponent = _array(record, "plga_exponent", 3)
        coupling = _array(record, "plga_coupling", 3)
        coupling_bias = _array(record, "plga_coupling_bias", 3)
        captured_left = _array(record, "plga_curvature_left", 3)
        captured_right = _array(record, "plga_curvature_right", 3)
        query_left = _array(record, "plga_reference_query", 3)
        query_right = _array(record, "plga_candidate_query", 3)
        key_left = _array(record, "plga_reference_key", 3)
        key_right = _array(record, "plga_candidate_key", 3)
        score_left = _array(record, "plga_reference_pre_mask_score", 3)
        score_right = _array(record, "plga_candidate_pre_mask_score", 3)
        logits_left = _array(record, "plga_reference_centered_logits", 2)
        logits_right = _array(record, "plga_candidate_centered_logits", 2)
        cache_residual = _array(record, "plga_candidate_cache_replay_residual", 1)
        resources = {
            name.removeprefix("resource_"): _scalar(record, name)
            for name in record.files if name.startswith("resource_")
        }
    if role not in {"construction", "heldout"}:
        raise ValueError("PLGA role is invalid")
    if role == "heldout":
        if lock is None or lock_sha256 != lock["lock_sha256"]:
            raise ValueError("held-out PLGA record is not lock-bound")
    elif lock is not None or lock_sha256:
        raise ValueError("construction PLGA record must precede the lock")
    expected_cells = 8 if role == "construction" else 16
    if (
        seed not in spec["trajectories"][role]
        or anchor not in spec["trajectories"]["anchors"]
        or layer not in spec["trajectories"]["layer_indices"]
        or positive_floor != 1e-9
        or generator.shape != (expected_cells, 64, 64)
        or collapsed.shape != generator.shape
    ):
        raise ValueError("PLGA metadata or generator dimensions changed")

    algebra_tolerance = float(spec["decisions"]["identity_relative_tolerance"])
    native_tolerance = float(spec["decisions"]["native_float32_replay_tolerance"])
    cells = []
    maximum_product = 0.0
    maximum_transfer = 0.0
    all_qualified = True
    margin_invoked_count = 0
    for index in range(expected_cells):
        replay = plga_secant_chain(
            collapsed[index], generator[index], weight[index], bias[index],
            exponent[index], coupling[index], coupling_bias[index],
            positive_floor=positive_floor)
        activation_max = float(np.max(np.abs(replay["activation_secant"])))
        power_max = float(np.max(np.abs(replay["power_secant"])))
        secant_product = activation_max * power_max
        maximum_product = max(maximum_product, secant_product)
        maximum_transfer = max(maximum_transfer, float(replay["operator_bound"]))
        curvature_residual = max(
            _relative(replay["curvature_left"], captured_left[index]),
            _relative(replay["curvature_right"], captured_right[index]),
        )
        score = rectangular_score_transfer(
            query_left[index], query_right[index],
            replay["curvature_left"], replay["curvature_right"],
            key_left[index], key_right[index])
        score_capture = max(
            _relative(score["reference_score"], score_left[index]),
            _relative(score["candidate_score"], score_right[index]),
        )
        construction_secant_ratio = None
        fixed_query_key = bool(
            _relative(query_left[index], query_right[index]) <= native_tolerance
            and _relative(key_left[index], key_right[index]) <= native_tolerance)
        identity_scale = max(
            float(np.linalg.norm(replay["realized_difference"])), 1.0)
        checks = {
            "positive_occupied_bases": bool(
                np.min(replay["metric_left"]) > 0.0
                and np.min(replay["metric_right"]) > 0.0),
            "iswiglu_power_secant_identity": bool(
                replay["identity_residual"] / identity_scale <= algebra_tolerance),
            "curvature_capture_replay": curvature_residual <= native_tolerance,
            "fixed_query_key": fixed_query_key,
            "rectangular_score_identity": (
                score["identity_relative_residual"] <= algebra_tolerance),
            "rectangular_score_capture": score_capture <= native_tolerance,
            "candidate_cache_replay": bool(cache_residual[index] <= native_tolerance),
        }
        if lock is not None:
            checks["construction_secant_maximum"] = (
                secant_product <= float(lock["plga_secant_product_maximum"]))
        centered_left = logits_left[index] - np.mean(logits_left[index])
        centered_right = logits_right[index] - np.mean(logits_right[index])
        winner = int(np.argmax(centered_left))
        margin = float(centered_left[winner] - np.max(np.delete(centered_left, winner)))
        delta = float(np.linalg.norm(centered_right - centered_left))
        margin_invoked = bool(margin > 0.0 and math.sqrt(2.0) * delta < margin)
        if margin_invoked:
            margin_invoked_count += 1
            checks["winner_if_margin_invoked"] = int(np.argmax(centered_right)) == winner
        qualified = all(checks.values())
        all_qualified = all_qualified and qualified
        cells.append({
            "cell": index,
            "activation_secant_max": activation_max,
            "power_secant_max": power_max,
            "activation_power_secant_product": secant_product,
            "transfer_coefficient": float(replay["operator_bound"]),
            "centered_margin": margin,
            "centered_delta_norm": delta,
            "margin_invoked": margin_invoked,
            "checks": checks,
            "qualified": qualified,
        })
    return {
        "source": {"path": str(source), "sha256": _sha256(source)},
        "role": role,
        "seed": seed,
        "anchor": anchor,
        "layer": layer,
        "maxima": {
            "activation_power_secant_product": maximum_product,
            "transfer_coefficient": maximum_transfer,
        },
        "margin_invoked_cell_count": margin_invoked_count,
        "cells": cells,
        "qualified": all_qualified,
        "resources": resources,
    }


def analyze_campaign(paths: list[str], lock_path: str | Path | None) -> dict[str, Any]:
    spec = campaign_spec()
    lock = load_lock(lock_path) if lock_path else None
    reports = [analyze_record(path, lock) for path in paths]
    if not reports:
        raise ValueError("PLGA analysis needs records")
    role = reports[0]["role"]
    if any(row["role"] != role for row in reports):
        raise ValueError("PLGA campaign cannot mix roles")
    expected_seeds = spec["trajectories"][role]
    expected = {
        (seed, anchor, layer)
        for seed in expected_seeds
        for anchor in spec["trajectories"]["anchors"]
        for layer in spec["trajectories"]["layer_indices"]
    }
    actual = {(row["seed"], row["anchor"], row["layer"]) for row in reports}
    checks = {
        "complete_registered_grid": actual == expected and len(reports) == len(expected),
        "all_identities_qualified": all(row["qualified"] for row in reports),
    }
    decision = (
        "CONSTRUCTION_SUMMARY" if role == "construction" else "QUALIFIED"
    ) if all(checks.values()) else (
        "CONSTRUCTION_INVALID" if role == "construction" else "NOT_QUALIFIED")
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "lock_sha256": lock["lock_sha256"] if lock is not None else "",
        "role": role,
        "record_count": len(reports),
        "checks": checks,
        "maxima": {
            "activation_power_secant_product": max(
                row["maxima"]["activation_power_secant_product"] for row in reports),
            "transfer_coefficient": max(
                row["maxima"]["transfer_coefficient"] for row in reports),
        },
        "margin_invoked_cell_count": sum(
            row["margin_invoked_cell_count"] for row in reports),
        "reports": reports,
        "decision": decision,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("records", nargs="+")
    parser.add_argument("--lock")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = analyze_campaign(arguments.records, arguments.lock)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"], "record_count": report["record_count"]
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] not in {
        "QUALIFIED", "CONSTRUCTION_SUMMARY",
    }:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
