#!/usr/bin/env python3
"""Aggregate the complete causal row-map campaign without re-fitting rules."""

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

from confirm.causal_confirmation_specs import (  # noqa: E402
    ANCHORS,
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402


ROW_SCHEMA = "pldr-causal-row-analysis-v1"
PLGA_SCHEMA = "pldr-causal-plga-analysis-v1"
INTERVENTION_SCHEMA = "pldr-causal-intervention-analysis-v1"
OUTPUT_SCHEMA = "pldr-causal-row-map-final-report-v1"
QUALIFICATION_SCHEMA = "pldr-causal-row-map-qualification-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).resolve()
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"report is not an object: {source}")
    return source, value


def _cell(report: dict[str, Any]) -> tuple[str, int, int, int, int]:
    metadata = report.get("metadata", {})
    return (
        str(metadata.get("role")),
        int(metadata.get("seed", -1)),
        int(metadata.get("data_offset_chunks", -1)),
        int(metadata.get("layer", -1)),
        int(metadata.get("global_step_before", -1)),
    )


def _expected_cells() -> set[tuple[str, int, int, int, int]]:
    return {
        (str(row["role"]), int(row["seed"]),
         int(row["data_offset_chunks"]), layer, anchor)
        for row in TRAJECTORIES
        for layer in range(3)
        for anchor in ANCHORS
    }


def _prediction_core(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in prediction.items()
        if key != "coverage"
    }


def _gate_route_valid(report: dict[str, Any]) -> bool:
    route = report.get("gate_route")
    if not isinstance(route, dict) or route.get("qualified") is not True:
        return False
    checks = route.get("checks")
    expected_decay = (
        1.0
        - float(OPTIMIZER["learning_rate"])
        * float(OPTIMIZER["weight_decay"])
    )
    return bool(
        isinstance(checks, dict)
        and checks
        and all(value is True for value in checks.values())
        and int(route.get("history_length", -1))
        == int(REGISTRY["history_length"])
        and abs(float(route.get("decay_multiplier_minimum", -1.0))
                - expected_decay) <= 1.0e-15
        and abs(float(route.get("decay_multiplier_maximum", -1.0))
                - expected_decay) <= 1.0e-15
    )


def build(
    prediction_paths: list[str | Path],
    coverage_paths: list[str | Path],
    *,
    construction_plga_path: str | Path,
    heldout_plga_path: str | Path,
    intervention_path: str | Path,
    lock_path: str | Path,
    qualification_path: str | Path,
) -> dict[str, Any]:
    lock_file, lock = _load(lock_path)
    unsigned_lock = dict(lock)
    lock_digest = unsigned_lock.pop("lock_sha256", None)
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("campaign_id") != CAMPAIGN_ID
        or lock.get("constant_enlargement_after_lock") is not False
        or lock_digest != digest_object(unsigned_lock)
    ):
        raise ValueError("causal construction lock is invalid")

    predictions: dict[tuple[str, int, int, int, int], tuple[Path, dict[str, Any]]] = {}
    for path in prediction_paths:
        source, report = _load(path)
        prediction = report.get("prediction", {})
        cell = _cell(report)
        if (
            report.get("schema_version") != ROW_SCHEMA
            or report.get("valid") is not True
            or prediction.get("decision_uses_successor") is not False
            or prediction.get("coverage") is not None
            or report.get("successor_attachment") is not None
            or report.get("gate_successor_coverage") is not None
            or report.get("metadata", {}).get("lock_sha256")
            != (lock_digest if cell[0] == "heldout" else "")
            or int(prediction.get("vertex_count", -1))
            != int(REGISTRY["vertex_count"])
            or int(prediction.get("pair_count", -1))
            != int(REGISTRY["pair_count"])
            or int(prediction.get("row_width", -1)) != 64
            or int(prediction.get("pair_chunk_size", -1))
            != int(lock["pair_chunk_size"])
            or prediction.get("arithmetic", {}).get("safety_factor")
            != float(lock["arithmetic_safety_factor"])
            or prediction.get("arithmetic", {}).get(
                "source_interval_used_for_decision") is not True
            or prediction.get("decision") not in {
                "CONTRACTION_CERTIFIED",
                "REEXPANSION_CERTIFIED",
                "UNRESOLVED",
            }
            or not _gate_route_valid(report)
            or cell in predictions
        ):
            raise ValueError(f"source prediction contract failed: {source}")
        predictions[cell] = (source, report)

    coverages: dict[tuple[str, int, int, int, int], tuple[Path, dict[str, Any]]] = {}
    for path in coverage_paths:
        source, report = _load(path)
        prediction = report.get("prediction", {})
        coverage = prediction.get("coverage")
        cell = _cell(report)
        attachment = report.get("successor_attachment")
        if (
            report.get("schema_version") != ROW_SCHEMA
            or report.get("valid") is not True
            or prediction.get("decision_uses_successor") is not False
            or not isinstance(coverage, dict)
            or coverage.get("all_pair_enclosures_hold") is not True
            or not isinstance(attachment, dict)
            or not isinstance(report.get("gate_successor_coverage"), dict)
            or report["gate_successor_coverage"].get("qualified") is not True
            or not _gate_route_valid(report)
            or cell in coverages
        ):
            raise ValueError(f"coverage report contract failed: {source}")
        coverages[cell] = (source, report)

    expected = _expected_cells()
    if set(predictions) != expected or set(coverages) != expected:
        raise ValueError("causal row-report grid is incomplete")
    causal_links = True
    frozen_predictions = True
    for cell in sorted(expected):
        prediction_path, source_report = predictions[cell]
        _coverage_path, coverage_report = coverages[cell]
        attachment = coverage_report["successor_attachment"]
        causal_links = bool(
            causal_links
            and attachment.get("prediction_report_sha256")
            == _sha256(prediction_path)
            and isinstance(attachment.get("successor_chronology_sha256"), str)
            and len(attachment["successor_chronology_sha256"]) == 64
        )
        frozen_predictions = bool(
            frozen_predictions
            and source_report["metadata"] == coverage_report["metadata"]
            and source_report["certificate"] == coverage_report["certificate"]
            and source_report["gate_route"] == coverage_report["gate_route"]
            and _prediction_core(source_report["prediction"])
            == _prediction_core(coverage_report["prediction"])
        )

    construction_plga_file, construction_plga = _load(
        construction_plga_path)
    heldout_plga_file, heldout_plga = _load(heldout_plga_path)
    intervention_file, intervention = _load(intervention_path)
    qualification_file, qualification = _load(qualification_path)
    if (
        construction_plga.get("schema_version") != PLGA_SCHEMA
        or construction_plga.get("decision") != "CONSTRUCTION_SUMMARY"
        or construction_plga.get("record_count") != 51
        or heldout_plga.get("schema_version") != PLGA_SCHEMA
        or heldout_plga.get("decision") != "QUALIFIED"
        or heldout_plga.get("record_count") != 204
        or heldout_plga.get("lock_sha256") != lock_digest
        or intervention.get("schema_version") != INTERVENTION_SCHEMA
        or intervention.get("decision") != "QUALIFIED"
        or intervention.get("record_count") != 64
        or intervention.get("lock_sha256") != lock_digest
        or qualification.get("schema_version") != QUALIFICATION_SCHEMA
        or qualification.get("status") != "QUALIFIED"
        or not isinstance(qualification.get("checks"), dict)
        or not qualification["checks"]
        or not all(value is True for value in qualification["checks"].values())
        or qualification.get("postnorm", {}).get("width") != 64
        or qualification.get("postnorm", {}).get("hidden") != 170
        or qualification.get("postnorm", {}).get("residual_units") != 8
        or qualification.get("postnorm", {}).get(
            "gated_blocks_per_unit") != 2
    ):
        raise ValueError(
            "qualification, PLGA, or intervention aggregate is invalid")

    heldout_decisions = [
        report["prediction"]["decision"]
        for cell, (_path, report) in predictions.items()
        if cell[0] == "heldout"
    ]
    all_decisions = [
        report["prediction"]["decision"]
        for _path, report in predictions.values()
    ]
    counts = {
        name: all_decisions.count(name)
        for name in (
            "CONTRACTION_CERTIFIED",
            "REEXPANSION_CERTIFIED",
            "UNRESOLVED",
        )
    }
    heldout_counts = {
        name: heldout_decisions.count(name)
        for name in counts
    }
    checks = {
        "complete_306_edge_grid": len(expected) == 306,
        "all_successors_causally_linked": causal_links,
        "all_source_predictions_immutable": frozen_predictions,
        "all_gate_routes_qualified": all(
            _gate_route_valid(report)
            for _path, report in predictions.values()),
        "all_frozen_enclosures_cover": all(
            report["prediction"]["coverage"]["all_pair_enclosures_hold"]
            for _path, report in coverages.values()),
        "all_native_successor_gates_replay": all(
            report["gate_successor_coverage"]["qualified"]
            for _path, report in coverages.values()),
        "heldout_contraction_observed": (
            heldout_counts["CONTRACTION_CERTIFIED"] > 0),
        "heldout_reexpansion_observed": (
            heldout_counts["REEXPANSION_CERTIFIED"] > 0),
        "construction_plga_complete": (
            construction_plga["decision"] == "CONSTRUCTION_SUMMARY"),
        "heldout_plga_qualified": heldout_plga["decision"] == "QUALIFIED",
        "heldout_interventions_qualified": (
            intervention["decision"] == "QUALIFIED"),
        "implemented_row_map_qualified": (
            qualification["status"] == "QUALIFIED"),
    }
    output: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "lock": {"path": str(lock_file), "sha256": _sha256(lock_file)},
        "edge_count": len(expected),
        "source_prediction_counts": counts,
        "heldout_prediction_counts": heldout_counts,
        "checks": checks,
        "component_reports": {
            "construction_plga": {
                "path": str(construction_plga_file),
                "sha256": _sha256(construction_plga_file),
            },
            "heldout_plga": {
                "path": str(heldout_plga_file),
                "sha256": _sha256(heldout_plga_file),
            },
            "heldout_intervention": {
                "path": str(intervention_file),
                "sha256": _sha256(intervention_file),
            },
            "qualification": {
                "path": str(qualification_file),
                "sha256": _sha256(qualification_file),
            },
        },
        "decision": "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED",
    }
    output["report_sha256"] = digest_object(output)
    return output


def _write(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--coverage", nargs="+", required=True)
    parser.add_argument("--construction-plga", required=True)
    parser.add_argument("--heldout-plga", required=True)
    parser.add_argument("--intervention", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--qualification", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = build(
        arguments.predictions,
        arguments.coverage,
        construction_plga_path=arguments.construction_plga,
        heldout_plga_path=arguments.heldout_plga,
        intervention_path=arguments.intervention,
        lock_path=arguments.lock,
        qualification_path=arguments.qualification,
    )
    _write(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "edge_count": report["edge_count"],
        "output": str(Path(arguments.output).resolve()),
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
