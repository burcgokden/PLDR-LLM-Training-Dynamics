#!/usr/bin/env python3
"""Freeze construction-only constants for held-out comprehensive runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.comprehensive_campaign import campaign_spec  # noqa: E402
from confirm.comprehensive_lock import SCHEMA_VERSION, seal_lock  # noqa: E402
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402


ENDPOINT_REPORT_SCHEMA = "pldr-comprehensive-endpoint-report-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("construction report must be a JSON object")
    return value


def build_lock(arguments: argparse.Namespace) -> dict[str, Any]:
    spec = campaign_spec()
    development_paths = list(arguments.development_reports)
    construction_paths = list(arguments.construction_reports)
    development = [_load(path) for path in development_paths]
    construction = [_load(path) for path in construction_paths]
    for role, reports in (("development", development), ("construction", construction)):
        if any(
            row.get("schema_version") != ENDPOINT_REPORT_SCHEMA
            or row.get("campaign_spec_sha256") != spec["spec_sha256"]
            or row.get("role") != role
            or not row.get("exact_checks_pass", False)
            for row in reports
        ):
            raise ValueError(f"{role} endpoint reports are invalid")
    cells = []
    ratios: dict[str, float] = {}
    routes: dict[str, dict[str, float]] = {}
    native = float(spec["decisions"]["native_float32_replay_tolerance"])
    for layer in spec["trajectories"]["layer_indices"]:
        candidates = sorted(
            (
                row for row in development
                if int(row["layer"]) == layer and row["development_positive"]
            ),
            key=lambda row: int(row["step_start"]),
        )
        if not candidates:
            raise ValueError(f"development found no contracting cell for layer {layer}")
        anchor = int(candidates[0]["step_start"])
        selected = [
            row for row in construction
            if int(row["layer"]) == layer and int(row["step_start"]) == anchor
        ]
        if len(selected) != 1 or not selected[0]["diameter_contracted"]:
            raise ValueError(
                f"construction did not confirm the fixed cell for layer {layer}")
        row = selected[0]
        limit = float(row["diameter_ratio"]) + 4.0 * native
        if not 0.0 < limit < 1.0:
            raise ValueError("padded construction diameter ratio is not contractive")
        key = str(layer)
        ratios[key] = limit
        block = row["block"]
        routes[key] = {
            "gate_abs": 1.05 * abs(float(block["gate_log_decrement_sum"])) + native,
            "shape_abs": 1.05 * abs(float(block["shape_log_decrement_sum"])) + native,
            "total_abs": 1.05 * abs(float(block["total_log_decrement_sum"])) + native,
        }
        cells.append({
            "layer": layer,
            "anchor": anchor,
            "updates": spec["trajectories"]["confirmation_block_updates"],
        })

    intervention = _load(arguments.intervention_report)
    if (
        intervention.get("role") != "construction"
        or intervention.get("decision") != "CONSTRUCTION_SUMMARY"
    ):
        raise ValueError("construction intervention summary is invalid")
    shape_limit = 1.05 * float(
        intervention["maxima"]["shape_correction_ratio"]
    ) + native
    plga = _load(arguments.plga_report)
    if (
        plga.get("role") != "construction"
        or plga.get("decision") != "CONSTRUCTION_SUMMARY"
    ):
        raise ValueError("construction PLGA summary is invalid")
    secant_maximum = 1.05 * float(
        plga["maxima"]["activation_power_secant_product"]
    ) + native
    sources = [
        *development_paths,
        *construction_paths,
        arguments.intervention_report,
        arguments.plga_report,
    ]
    unsigned = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "primary_cells": cells,
        "diameter_ratio_limit_by_layer": ratios,
        "route_bounds": routes,
        "shape_correction_ratio_limit": shape_limit,
        "plga_secant_product_maximum": secant_maximum,
        "normalized_effect_floor": float(
            spec["decisions"]["normalized_effect_floor"]),
        "identity_relative_tolerance": float(
            spec["decisions"]["identity_relative_tolerance"]),
        "native_float32_replay_tolerance": native,
        "source_report_sha256": sorted({_sha256(path) for path in sources}),
        "constant_enlargement_after_lock": False,
    }
    return seal_lock(unsigned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-reports", nargs="+", required=True)
    parser.add_argument("--construction-reports", nargs="+", required=True)
    parser.add_argument("--intervention-report", required=True)
    parser.add_argument("--plga-report", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    lock = build_lock(arguments)
    write_json_atomic(arguments.output, lock)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "lock_sha256": lock["lock_sha256"],
        "primary_cells": lock["primary_cells"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
