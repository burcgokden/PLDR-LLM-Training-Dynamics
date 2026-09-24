#!/usr/bin/env python3
"""Analyze source-only predictions and native successor coverage."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
)
from confirm.finite_increment import (  # noqa: E402
    exact_diameter_slack,
    product_convolution,
    write_json_atomic,
)
from confirm.finite_increment_live import SUCCESSOR_SCHEMA  # noqa: E402
from confirm.finite_increment_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    CONSECUTIVE_EDGES,
    REGISTRY,
    TIME_COURSE_ANCHORS,
)
from confirm.produce_finite_increment_certificate import (  # noqa: E402
    CERTIFICATE_SCHEMA,
)


PREDICTION_SCHEMA = "pldr-finite-increment-prediction-v1"
COVERAGE_SCHEMA = "pldr-finite-increment-coverage-v1"
AGGREGATE_SCHEMA = "pldr-finite-increment-product-convolution-v1"
LOCK_SCHEMA = "pldr-finite-increment-construction-lock-v1"
TIMECOURSE_SCHEMA = "pldr-finite-increment-timecourse-v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _load_certificate(path: str | Path) -> tuple[Path, dict[str, Any]]:
    certificate_path = Path(path).resolve()
    value = _load_json(certificate_path)
    if (
        value.get("schema_version") != CERTIFICATE_SCHEMA
        or value.get("successor_values_used") is not False
        or value.get("map_count") != int(REGISTRY["maps_per_full_registry"])
        or value.get("pair_count") != int(REGISTRY["pairs_per_full_registry"])
        or value.get("cross_map_pairs_formed") is not False
    ):
        raise ValueError("finite-increment certificate is invalid")
    source = Path(value["source_path"]).resolve()
    if sha256_path(source) != value.get("source_sha256"):
        raise ValueError("certificate-bound source artifact changed")
    return certificate_path, value


def predict(arguments: argparse.Namespace) -> int:
    certificate_path, certificate = _load_certificate(arguments.certificate)
    maps = certificate["maps"]
    prediction: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "certificate_path": str(certificate_path),
        "certificate_sha256": sha256_path(certificate_path),
        "source_path": certificate["source_path"],
        "source_sha256": certificate["source_sha256"],
        "role": certificate["role"],
        "seed": certificate["seed"],
        "step": certificate["step"],
        "registry_sha256": certificate["registry_sha256"],
        "successor_values_used": False,
        "valid": bool(certificate["all_identities_valid"]),
        "scientific_prediction": {
            "all_strict_state_dominance": certificate[
                "all_strict_state_dominance"
            ],
            "all_strict_contraction": certificate[
                "all_predicted_contractions"
            ],
            "minimum_state_margin": certificate["minimum_state_margin"],
            "per_map": [{
                "context": row["context"],
                "layer": row["layer"],
                "head": row["head"],
                "source_diameter_squared": row["source_diameter_squared"],
                "predicted_diameter_squared": row[
                    "predicted_diameter_squared"
                ],
                "state_bound_endpoint_squared": row[
                    "state_bound_endpoint_squared"
                ],
                "minimum_state_margin": row["minimum_state_margin"],
                "strict_state_dominance": row["strict_state_dominance"],
                "strict_contraction": row["realized_prediction_contracts"],
            } for row in maps],
        },
    }
    prediction["prediction_sha256"] = digest_object(prediction)
    write_json_atomic(arguments.output, prediction)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "valid": prediction["valid"],
        "all_strict_state_dominance": prediction[
            "scientific_prediction"
        ]["all_strict_state_dominance"],
    }, sort_keys=True))
    return 0 if prediction["valid"] else 1


def _successor(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    successor_path = Path(path).resolve()
    with np.load(successor_path, allow_pickle=False) as record:
        schema = np.asarray(record["schema_version"])
        if schema.shape != () or str(schema.item()) != SUCCESSOR_SCHEMA:
            raise ValueError("unknown finite-increment successor schema")
        metadata_value = np.asarray(record["metadata_json"])
        if metadata_value.shape != ():
            raise ValueError("successor metadata must be scalar JSON")
        metadata = json.loads(str(metadata_value.item()))
        actual = np.asarray(record["actual_rows"], dtype=np.float64)
    expected = (
        int(REGISTRY["context_count"]),
        int(ARCHITECTURE["layers"]),
        int(ARCHITECTURE["heads"]),
        int(ARCHITECTURE["rows_per_map"]),
        int(ARCHITECTURE["head_width"]),
    )
    if actual.shape != expected or not np.isfinite(actual).all():
        raise ValueError("successor rows have invalid shape or values")
    if metadata.get("successor_evaluated") is not True:
        raise ValueError("successor artifact did not execute the update")
    return metadata, actual


def coverage(arguments: argparse.Namespace) -> int:
    certificate_path, certificate = _load_certificate(arguments.certificate)
    prediction_path = Path(arguments.prediction).resolve()
    prediction = _load_json(prediction_path)
    if (
        prediction.get("schema_version") != PREDICTION_SCHEMA
        or prediction.get("certificate_sha256") != sha256_path(certificate_path)
        or prediction.get("source_sha256") != certificate["source_sha256"]
        or prediction.get("successor_values_used") is not False
        or prediction.get("valid") is not True
    ):
        raise ValueError("coverage input is not the frozen source prediction")
    metadata, actual = _successor(arguments.successor)
    if (
        metadata.get("prediction_sha256") != sha256_path(prediction_path)
        or metadata.get("source_sha256") != certificate["source_sha256"]
        or metadata.get("predicted_parameter_sha256")
        != certificate["predicted_parameter_sha256"]
        or metadata.get("actual_parameter_sha256")
        != certificate["predicted_parameter_sha256"]
    ):
        raise ValueError("native successor is not bound to this prediction")
    source_path = Path(certificate["source_path"])
    with np.load(source_path, allow_pickle=False) as source:
        source_rows = np.asarray(source["source_rows"], dtype=np.float64)
        predicted_rows = np.asarray(source["predicted_rows"], dtype=np.float64)
    native_max_abs_residual = float(np.max(np.abs(actual - predicted_rows)))
    map_reports = []
    index = 0
    for context in range(actual.shape[0]):
        for layer in range(actual.shape[1]):
            for head in range(actual.shape[2]):
                observed = exact_diameter_slack(
                    source_rows[context, layer, head],
                    actual[context, layer, head],
                )
                frozen = certificate["maps"][index]
                observed_endpoint = float(observed["endpoint_diameter_squared"])
                scale = max(observed_endpoint, 1.0)
                equality_tolerance = 8192.0 * np.finfo(np.float64).eps * scale
                bound_covers = bool(
                    observed_endpoint
                    <= float(frozen["state_bound_endpoint_squared"])
                    + equality_tolerance
                )
                sign_correct = bool(
                    bool(observed["strict_contraction"])
                    == bool(frozen["realized_prediction_contracts"])
                )
                map_reports.append({
                    "context": context,
                    "layer": layer,
                    "head": head,
                    "observed_diameter_squared": observed_endpoint,
                    "predicted_diameter_squared": frozen[
                        "predicted_diameter_squared"
                    ],
                    "prediction_max_abs_error": abs(
                        observed_endpoint
                        - float(frozen["predicted_diameter_squared"])
                    ),
                    "state_bound_covers_observed": bound_covers,
                    "contraction_sign_correct": sign_correct,
                    "observed_strict_contraction": bool(
                        observed["strict_contraction"]
                    ),
                })
                index += 1
    technical_coverage = bool(
        native_max_abs_residual == 0.0
        and metadata.get("native_parameter_max_abs_residual") == 0.0
        and metadata.get("native_row_max_abs_residual") == 0.0
        and all(row["prediction_max_abs_error"] == 0.0 for row in map_reports)
        and all(row["state_bound_covers_observed"] for row in map_reports)
        and all(row["contraction_sign_correct"] for row in map_reports)
    )
    scientific_pass = bool(
        technical_coverage
        and certificate["all_strict_state_dominance"]
        and all(row["observed_strict_contraction"] for row in map_reports)
    )
    report: dict[str, Any] = {
        "schema_version": COVERAGE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "certificate_path": str(certificate_path),
        "certificate_sha256": sha256_path(certificate_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_path(prediction_path),
        "successor_path": str(Path(arguments.successor).resolve()),
        "successor_sha256": sha256_path(arguments.successor),
        "role": certificate["role"],
        "seed": certificate["seed"],
        "step": certificate["step"],
        "registry_sha256": certificate["registry_sha256"],
        "map_count": len(map_reports),
        "pair_count": certificate["pair_count"],
        "cross_map_pairs_formed": False,
        "native_max_abs_residual": native_max_abs_residual,
        "technical_coverage": technical_coverage,
        "all_strict_state_dominance": certificate[
            "all_strict_state_dominance"
        ],
        "scientific_pass": scientific_pass,
        "maps": map_reports,
    }
    report["report_sha256"] = digest_object(report)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "technical_coverage": technical_coverage,
        "scientific_pass": scientific_pass,
    }, sort_keys=True))
    if arguments.require_coverage and not scientific_pass:
        return 1
    return 0


def aggregate(arguments: argparse.Namespace) -> int:
    paths = sorted(Path(path).resolve() for path in glob.glob(
        arguments.reports_glob
    ))
    if len(paths) != len(CONSECUTIVE_EDGES):
        raise ValueError("aggregate needs exactly twenty consecutive reports")
    reports = [_load_json(path) for path in paths]
    if any(row.get("schema_version") != COVERAGE_SCHEMA for row in reports):
        raise ValueError("aggregate input has an unknown coverage schema")
    reports.sort(key=lambda row: int(row["step"]))
    if tuple(int(row["step"]) for row in reports) != CONSECUTIVE_EDGES:
        raise ValueError("coverage reports do not form the registered block")
    if len({(row["role"], row["seed"], row["registry_sha256"])
            for row in reports}) != 1:
        raise ValueError("consecutive reports do not share one trajectory")
    certificates = [
        _load_json(row["certificate_path"]) for row in reports
    ]
    per_map = []
    for map_index in range(int(REGISTRY["maps_per_full_registry"])):
        initial = float(certificates[0]["maps"][map_index][
            "source_diameter_squared"
        ])
        gains = []
        forcings = []
        observed = []
        source_link_residuals = []
        for edge_index, certificate in enumerate(certificates):
            row = certificate["maps"][map_index]
            source = float(row["source_diameter_squared"])
            margin = float(row["minimum_state_margin"])
            alpha = 0.0 if source <= 0.0 else min(
                1.0, max(0.0, margin / source)
            )
            forcing = max(0.0, alpha * source - margin)
            gains.append(1.0 - alpha)
            forcings.append(forcing)
            endpoint = float(reports[edge_index]["maps"][map_index][
                "observed_diameter_squared"
            ])
            observed.append(endpoint)
            if edge_index + 1 < len(certificates):
                next_source = float(certificates[edge_index + 1]["maps"][
                    map_index
                ]["source_diameter_squared"])
                source_link_residuals.append(abs(endpoint - next_source))
        convolution = product_convolution(initial, gains, forcings)
        scale = max(initial, *observed, 1.0)
        tolerance = 8192.0 * np.finfo(np.float64).eps * scale
        prefix_covered = all(
            value <= bound + tolerance
            for value, bound in zip(
                observed, convolution["envelope"][1:], strict=True
            )
        )
        per_map.append({
            "map_index": map_index,
            "initial_diameter_squared": initial,
            "gains": gains,
            "forcings": forcings,
            "envelope": convolution["envelope"],
            "observed": observed,
            "maximum_source_link_residual": max(
                source_link_residuals, default=0.0
            ),
            "all_prefixes_covered": prefix_covered,
        })
    report: dict[str, Any] = {
        "schema_version": AGGREGATE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "role": reports[0]["role"],
        "seed": reports[0]["seed"],
        "registry_sha256": reports[0]["registry_sha256"],
        "steps": [row["step"] for row in reports],
        "coverage_report_sha256": [sha256_path(path) for path in paths],
        "map_count": len(per_map),
        "all_technical_coverage": all(
            row["technical_coverage"] for row in reports
        ),
        "all_edge_scientific_pass": all(
            row["scientific_pass"] for row in reports
        ),
        "all_prefixes_covered": all(
            row["all_prefixes_covered"] for row in per_map
        ),
        "all_links_exact": all(
            row["maximum_source_link_residual"] == 0.0 for row in per_map
        ),
        "maps": per_map,
    }
    report["scientific_pass"] = bool(
        report["all_technical_coverage"]
        and report["all_edge_scientific_pass"]
        and report["all_prefixes_covered"]
        and report["all_links_exact"]
    )
    report["report_sha256"] = digest_object(report)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "scientific_pass": report["scientific_pass"],
    }, sort_keys=True))
    return 0 if report["scientific_pass"] else 1


def lock(arguments: argparse.Namespace) -> int:
    aggregate_path = Path(arguments.aggregate).resolve()
    aggregate_value = _load_json(aggregate_path)
    if (
        aggregate_value.get("schema_version") != AGGREGATE_SCHEMA
        or aggregate_value.get("scientific_pass") is not True
    ):
        raise ValueError("construction lock requires a passing product convolution")
    value: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": sha256_path(aggregate_path),
        "registry_sha256": aggregate_value["registry_sha256"],
        "locked": True,
        "policy_changes_after_lock_forbidden": True,
        "source_only_prediction_required": True,
        "cross_map_pairs_forbidden": True,
        "finite_increment_policy": (
            "exact-endpoint-telescopes-plus-state-dominance-v1"
        ),
    }
    value["lock_sha256"] = digest_object(value)
    write_json_atomic(arguments.output, value)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "lock_sha256": value["lock_sha256"],
    }, sort_keys=True))
    return 0


def time_course(arguments: argparse.Namespace) -> int:
    run_root = Path(arguments.run_root).resolve()
    log_path = run_root / "log.jsonl"
    if not log_path.is_file():
        raise FileNotFoundError("trajectory log is missing")
    registry = _load_json(arguments.registry)
    points = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        point = record.get("finite_increment_timecourse")
        if point is not None:
            points.append(point)
    points.sort(key=lambda row: int(row["step"]))
    expected_steps = list(TIME_COURSE_ANCHORS)
    if [int(row["step"]) for row in points] != expected_steps:
        raise ValueError("trajectory log has an incomplete time course")
    if any(
        row.get("registry_sha256") != registry.get("registry_sha256")
        or len(row.get("diameter_squared", []))
        != int(REGISTRY["maps_per_full_registry"])
        or row.get("cross_map_pairs_formed") is not False
        for row in points
    ):
        raise ValueError("a time-course point has invalid provenance or geometry")
    initial = np.asarray(points[0]["diameter_squared"], dtype=np.float64)
    final = np.asarray(points[-1]["diameter_squared"], dtype=np.float64)
    report: dict[str, Any] = {
        "schema_version": TIMECOURSE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "run_root": str(run_root),
        "log_sha256": sha256_path(log_path),
        "registry_sha256": registry["registry_sha256"],
        "steps": expected_steps,
        "map_count": int(REGISTRY["maps_per_full_registry"]),
        "points": points,
        "strict_net_contraction_count": int(np.count_nonzero(final < initial)),
        "initial_aggregate_diameter_squared": float(np.max(initial)),
        "final_aggregate_diameter_squared": float(np.max(final)),
    }
    report["report_sha256"] = digest_object(report)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "points": len(points),
    }, sort_keys=True))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--certificate", required=True)
    prediction.add_argument("--output", required=True)
    prediction.set_defaults(function=predict)
    covered = commands.add_parser("coverage")
    covered.add_argument("--certificate", required=True)
    covered.add_argument("--successor", required=True)
    covered.add_argument("--prediction", required=True)
    covered.add_argument("--output", required=True)
    covered.add_argument("--require-coverage", action="store_true")
    covered.set_defaults(function=coverage)
    combined = commands.add_parser("aggregate")
    combined.add_argument("--reports-glob", required=True)
    combined.add_argument("--output", required=True)
    combined.set_defaults(function=aggregate)
    locked = commands.add_parser("lock")
    locked.add_argument("--aggregate", required=True)
    locked.add_argument("--output", required=True)
    locked.set_defaults(function=lock)
    course = commands.add_parser("time-course")
    course.add_argument("--run-root", required=True)
    course.add_argument("--registry", required=True)
    course.add_argument("--output", required=True)
    course.set_defaults(function=time_course)
    arguments = parser.parse_args()
    raise SystemExit(arguments.function(arguments))


if __name__ == "__main__":
    main()
