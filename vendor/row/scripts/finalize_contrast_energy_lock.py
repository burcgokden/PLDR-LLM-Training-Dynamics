#!/usr/bin/env python3
"""Freeze the contrast-energy construction policy after complete source coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import digest_object  # noqa: E402
from confirm.contrast_energy_specs import (  # noqa: E402
    ANCHORS,
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    REGISTRY,
    RESOURCE_BUDGET,
    TAYLOR_POLICY,
)


REPORT_SCHEMA = "pldr-contrast-energy-analysis-v1"


def _load_spec(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(value)
    recorded = unsigned.pop("spec_sha256", None)
    if (
        value.get("campaign_id") != CAMPAIGN_ID
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("campaign specification digest does not replay")
    return value


def _load_resource_lock(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    unsigned = dict(value)
    recorded = unsigned.pop("resource_lock_sha256", None)
    if (
        value.get("schema_version")
        != "pldr-contrast-energy-resource-lock-v1"
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("passed") is not True
        or value.get("status") != "measured_caps_sealed"
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("Stage B resource lock does not replay")
    return value


def build_lock(
    campaign_spec: str | Path,
    resource_lock: str | Path,
    reports: list[str | Path],
) -> tuple[dict[str, Any], bool, list[str]]:
    _load_spec(campaign_spec)
    resources = _load_resource_lock(resource_lock)
    expected = {
        (role, seed, anchor, layer)
        for role, seed in (("development", 7444), ("construction", 8444))
        for anchor in ANCHORS
        for layer in REGISTRY["layers"]
    }
    seen = set()
    strict = {(role, seed, layer): 0 for role, seed, _, layer in expected}
    backend_manifests: set[str] = set()
    expected_taylor_policy = digest_object(TAYLOR_POLICY)
    failures: list[str] = []
    for raw_path in reports:
        path = Path(raw_path).resolve()
        value = json.loads(path.read_text(encoding="utf-8"))
        prediction = value.get("prediction", {}) if isinstance(value, dict) else {}
        coverage = prediction.get("coverage") if isinstance(prediction, dict) else None
        metadata = value.get("metadata", {})
        producer_metadata = (
            metadata.get("taylor_producer_metadata", {})
            if isinstance(metadata, dict) else {}
        )
        backend_manifest = producer_metadata.get("backend_manifest_sha256")
        policy_digest = producer_metadata.get("taylor_policy_sha256")
        if (
            not isinstance(backend_manifest, str)
            or len(backend_manifest) != 64
            or any(character not in "0123456789abcdef"
                   for character in backend_manifest)
            or policy_digest != expected_taylor_policy
        ):
            failures.append(f"invalid Taylor backend binding {path}")
        else:
            backend_manifests.add(backend_manifest)
        key = (
            metadata.get("role", metadata.get("trajectory")),
            int(metadata.get("seed", -1)),
            int(metadata.get("anchor", metadata.get("global_step_before", -1))),
            int(metadata.get("layer", -1)),
        )
        if key in seen:
            failures.append(f"duplicate grid key {key}")
        seen.add(key)
        if (
            value.get("schema_version") != REPORT_SCHEMA
            or value.get("valid") is not True
            or prediction.get("decision_uses_successor") is not False
            or not isinstance(coverage, dict)
            or coverage.get("all_pairs_enclosed") is not True
            or coverage.get("strict_sign_correct") is not True
            or coverage.get("pair_count") != REGISTRY["pair_count"]
        ):
            failures.append(f"invalid construction coverage {path}")
        if key in expected and prediction.get("decision") in {
            "contraction", "reexpansion",
        }:
            strict[(key[0], key[1], key[3])] += 1
    if seen != expected:
        failures.append(
            f"construction grid mismatch missing={len(expected - seen)} "
            f"extra={len(seen - expected)}")
    for key, count in strict.items():
        if count < 1:
            failures.append(f"no strict edge for {key}")
    if len(backend_manifests) != 1:
        failures.append(
            "construction reports do not share one Taylor backend manifest")
    locked_backend = (
        next(iter(backend_manifests)) if len(backend_manifests) == 1 else "")
    passed = not failures
    value: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "constant_policy_after_lock": True,
        "primary_block_length": 1,
        "history_length": int(REGISTRY["history_length"]),
        "segment_grid": [0.0, 0.5, 1.0],
        "shape_safety_factor": 2.0,
        "numerical_relative_coefficient": (
            8.0 * 64.0 * float(np.finfo(np.float32).eps)),
        "numerical_absolute_charge": 0.0,
        "taylor_policy_sha256": expected_taylor_policy,
        "taylor_backend_manifest_sha256": locked_backend,
        "resource_lock_sha256": resources["resource_lock_sha256"],
        "resource_policy": {
            "measured_peak_mib": resources["measured_peak_mib"],
            "sealed_peak_cap_mib": resources["sealed_peak_cap_mib"],
            "stage_b_enlargement": resources["enlargement"],
            "aggregate_output_gib_cap": RESOURCE_BUDGET[
                "aggregate_output_gib_cap"],
        },
    }
    value["lock_sha256"] = digest_object(value)
    return value, passed, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-spec", required=True)
    parser.add_argument("--resource-lock", required=True)
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    value, passed, failures = build_lock(
        arguments.campaign_spec, arguments.resource_lock, arguments.reports)
    if arguments.require_pass and not passed:
        raise SystemExit("construction lock gate failed: " + "; ".join(failures))
    target = Path(arguments.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(json.dumps({
        "failures": failures, "output": str(target), "passed": passed
    }, sort_keys=True))


if __name__ == "__main__":
    main()
