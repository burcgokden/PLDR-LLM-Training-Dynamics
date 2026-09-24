#!/usr/bin/env python3
"""Bind sealed observable-cocycle results into manuscript TeX artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BINDING_ROOT = ROOT.parents[1]
BUNDLE = (
    BINDING_ROOT
    / "experiment-data"
    / "manuscript-revisions"
    / "rev49"
    / "observable-full-state-cocycle-confirmation"
)
OUTPUT = ROOT / "docs" / "figures"
DIRECTIONS = ("row_adjoint", "optimizer_displacement", "balanced_full_state")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"unsafe evidence path: {value}")
    return Path(*candidate.parts)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify_bundle(bundle: Path) -> dict[str, Any]:
    manifest = bundle / "reports" / "EVIDENCE.sha256"
    if not manifest.is_file():
        raise FileNotFoundError("observable-cocycle evidence manifest is absent")
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = bundle / _safe_relative(relative)
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"observable-cocycle evidence drifted: {relative}")
    result = _load_json(bundle / "reports" / "final-analysis.json")
    if (
        result.get("schema_version") != "pldr-observable-cocycle-analysis-v1"
        or result.get("campaign_id")
        != "pldr-observable-full-state-cocycle-v1"
        or result["resource"]["hard_budget_pass"] is not True
    ):
        raise ValueError("observable-cocycle analysis schema or resource gate failed")
    return result


def _number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        exponent = int(math.floor(math.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.3f}\mathord{{\times}}10^{{{exponent}}}"
    return f"{value:.4f}"


def _runtime_summary(bundle: Path) -> dict[str, Any]:
    plan = _load_json(bundle / "protocol" / "launch_plan.json")
    nodes = {str(node["id"]): node for node in plan["nodes"]}
    seconds = 0.0
    counts: dict[str, int] = {}
    attempts = 0
    for path in sorted((bundle / "runtime").glob("*/attempt-*/attempt.json")):
        node = nodes[path.parent.parent.name]
        role = str(node.get("role", node["stage"]))
        device = str(node["device"])
        if role == "analysis" or not device.startswith("cuda:"):
            continue
        attempt = _load_json(path)
        if int(attempt["exit_code"]) != 0:
            seconds += float(attempt["elapsed_seconds"])
        attempts += 1
        counts[device] = counts.get(device, 0) + 1
    maximum_reserved = 0
    for path in sorted((bundle / "runtime").glob("*/completed.json")):
        node = nodes[path.parent.name]
        if str(node.get("role", node["stage"])) == "analysis":
            continue
        completion = _load_json(path)
        seconds += float(completion.get("resource", {}).get("reserved_device_seconds", 0.0))
        maximum_reserved = max(
            maximum_reserved,
            int(completion.get("resource", {}).get("peak_gpu_reserved_bytes", 0)),
        )
    design = _load_json(bundle / "protocol" / "campaign-design.json")
    budget = design["resource_budget"]
    return {
        "hours": seconds / 3600.0,
        "attempts": attempts,
        "device_attempts": counts,
        "maximum_reserved": maximum_reserved,
        "working_hours": float(budget["working_aggregate_reserved_device_hours"]),
        "hard_hours": float(budget["hard_aggregate_reserved_device_hours"]),
    }








