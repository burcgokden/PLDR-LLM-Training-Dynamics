#!/usr/bin/env python3
"""Independently finalize the comprehensive gate-shape campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.comprehensive_campaign import campaign_spec  # noqa: E402
from confirm.comprehensive_lock import load_lock  # noqa: E402
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402


ENDPOINT_SCHEMA = "pldr-comprehensive-endpoint-report-v1"
INTERVENTION_SCHEMA = "pldr-comprehensive-intervention-report-v1"
PLGA_SCHEMA = "pldr-comprehensive-plga-report-v1"
REPORT_SCHEMA = "pldr-comprehensive-final-report-v1"


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path, schema: str) -> dict[str, Any]:
    source = Path(path).resolve()
    with source.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"unexpected report schema: {source}")
    return value


def _expected_endpoint_cells(spec: dict[str, Any]) -> set[tuple[str, int, int, int]]:
    cells: set[tuple[str, int, int, int]] = set()
    for role in ("development", "construction", "heldout"):
        anchors = (
            spec["trajectories"]["anchors"][:3]
            if role == "development" else spec["trajectories"]["anchors"]
        )
        for seed in spec["trajectories"][role]:
            for anchor in anchors:
                for layer in spec["trajectories"]["layer_indices"]:
                    cells.add((role, int(seed), int(anchor), int(layer)))
    return cells


def _elapsed_seconds(resources: Iterable[dict[str, Any]]) -> float:
    total = 0.0
    for row in resources:
        device = str(row.get("device", row.get("resolved_device", "")))
        elapsed = float(row.get("elapsed_seconds", -1.0))
        if elapsed < 0.0:
            raise ValueError("resource record has an invalid elapsed time")
        if device.startswith("cuda"):
            total += elapsed
    return total


def _training_resources(
    campaign_root: Path, spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    resources: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for role in ("development", "construction", "heldout"):
        for seed in spec["trajectories"][role]:
            path = campaign_root / "runs" / f"cgs-{role}-s{seed}" / "log.jsonl"
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"missing native training log: {path}")
            rows = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    value = json.loads(line)
                    if value.get("event") == "resource_summary":
                        rows.append(value)
            if not rows:
                raise ValueError(f"training log has no resource summary: {path}")
            resources.extend(rows)
            sources.append({"path": str(path.resolve()), "sha256": _sha256(path)})
    return resources, sources


def _tree_size(path: Path) -> tuple[int, bool]:
    total = 0
    symlink_free = True
    for member in path.rglob("*"):
        if member.is_symlink():
            symlink_free = False
        elif member.is_file():
            total += member.stat().st_size
    return total, symlink_free


def finalize(arguments: argparse.Namespace) -> dict[str, Any]:
    spec = campaign_spec()
    lock = load_lock(arguments.lock)
    root = Path(arguments.campaign_root).resolve()
    endpoint_paths = [Path(path).resolve() for path in arguments.endpoint_reports]
    endpoint = [_load(path, ENDPOINT_SCHEMA) for path in endpoint_paths]
    intervention_paths = [
        Path(path).resolve() for path in arguments.intervention_reports]
    intervention = [
        _load(path, INTERVENTION_SCHEMA) for path in intervention_paths]
    plga_paths = [Path(path).resolve() for path in arguments.plga_reports]
    plga = [_load(path, PLGA_SCHEMA) for path in plga_paths]

    for report in [*endpoint, *intervention, *plga]:
        if (
            report.get("campaign_id") != spec["campaign_id"]
            or report.get("campaign_spec_sha256") != spec["spec_sha256"]
        ):
            raise ValueError("a report is not bound to the frozen campaign")

    endpoint_cells = {
        (str(row["role"]), int(row["seed"]), int(row["step_start"]),
         int(row["layer"]))
        for row in endpoint
    }
    expected_cells = _expected_endpoint_cells(spec)
    if len(endpoint_cells) != len(endpoint):
        raise ValueError("endpoint reports contain duplicate registered cells")
    heldout = [row for row in endpoint if row["role"] == "heldout"]
    expected_primary = {
        (int(seed), int(cell["anchor"]), int(cell["layer"]))
        for seed in spec["trajectories"]["heldout"]
        for cell in lock["primary_cells"]
    }
    actual_primary = {
        (int(row["seed"]), int(row["step_start"]), int(row["layer"]))
        for row in heldout if row.get("primary_cell")
    }

    intervention_by_role = {str(row.get("role")): row for row in intervention}
    plga_by_role = {str(row.get("role")): row for row in plga}
    capture_resources = [row["resources"] for row in endpoint]
    capture_resources.extend(
        record["resources"]
        for report in intervention for record in report["records"])
    capture_resources.extend(
        record["resources"]
        for report in plga for record in report["reports"])
    training_resources, training_sources = _training_resources(root, spec)
    gpu_seconds = (
        _elapsed_seconds(training_resources)
        + _elapsed_seconds(capture_resources)
    )
    gpu_hours = gpu_seconds / 3600.0
    output_bytes, symlink_free = _tree_size(root)

    checks = {
        "complete_endpoint_grid": endpoint_cells == expected_cells,
        "all_endpoint_identities": all(
            bool(row.get("exact_checks_pass")) for row in endpoint),
        "heldout_lock_binding": all(
            row.get("lock_sha256") == lock["lock_sha256"] for row in heldout),
        "complete_primary_grid": actual_primary == expected_primary,
        "all_primary_cells_qualified": all(
            bool(row.get("heldout_primary_qualified"))
            for row in heldout if row.get("primary_cell")),
        "construction_intervention_valid": (
            intervention_by_role.get("construction", {}).get("decision")
            == "CONSTRUCTION_SUMMARY"),
        "heldout_intervention_qualified": (
            intervention_by_role.get("heldout", {}).get("decision")
            == "QUALIFIED"
            and intervention_by_role["heldout"].get("lock_sha256")
            == lock["lock_sha256"]),
        "construction_plga_valid": (
            plga_by_role.get("construction", {}).get("decision")
            == "CONSTRUCTION_SUMMARY"),
        "heldout_plga_qualified": (
            plga_by_role.get("heldout", {}).get("decision") == "QUALIFIED"
            and plga_by_role["heldout"].get("lock_sha256")
            == lock["lock_sha256"]),
        "gpu_hour_cap": gpu_hours <= float(
            spec["resource_contract"]["gpu_hour_cap"]),
        "compressed_output_cap": output_bytes <= int(
            spec["resource_contract"]["compressed_output_cap_bytes"]),
        "campaign_tree_symlink_free": symlink_free,
    }
    decision = "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED"
    input_paths = [
        *endpoint_paths, *intervention_paths, *plga_paths, Path(arguments.lock),
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "lock_sha256": lock["lock_sha256"],
        "checks": checks,
        "decision": decision,
        "counts": {
            "endpoint_reports": len(endpoint),
            "heldout_endpoint_reports": len(heldout),
            "heldout_primary_cells": len(actual_primary),
            "intervention_reports": len(intervention),
            "plga_reports": len(plga),
        },
        "resources": {
            "aggregate_gpu_seconds": gpu_seconds,
            "aggregate_gpu_hours": gpu_hours,
            "gpu_hour_cap": float(spec["resource_contract"]["gpu_hour_cap"]),
            "campaign_output_bytes": output_bytes,
            "compressed_output_cap_bytes": int(
                spec["resource_contract"]["compressed_output_cap_bytes"]),
        },
        "input_reports": [
            {"path": str(path), "sha256": _sha256(path)}
            for path in input_paths
        ],
        "training_logs": training_sources,
        "heldout_primary": sorted(actual_primary),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint-reports", nargs="+", required=True)
    parser.add_argument("--intervention-reports", nargs="+", required=True)
    parser.add_argument("--plga-reports", nargs="+", required=True)
    parser.add_argument("--lock", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = finalize(arguments)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "aggregate_gpu_hours": report["resources"]["aggregate_gpu_hours"],
        "campaign_output_bytes": report["resources"]["campaign_output_bytes"],
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
