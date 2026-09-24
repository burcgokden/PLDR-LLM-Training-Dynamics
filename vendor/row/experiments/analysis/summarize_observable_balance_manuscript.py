#!/usr/bin/env python3
"""Bind sealed observable-balance results into manuscript TeX artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[2]
BINDING_ROOT = ROOT.parents[1]
BUNDLE = (
    BINDING_ROOT
    / "experiment-data"
    / "manuscript-revisions"
    / "rev50"
    / "finite-increment-observable-balance"
)
OUTPUT = ROOT / "docs" / "figures"
CAMPAIGN_ID = "pldr-finite-increment-observable-balance-v1"


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


def _verify_manifest(bundle: Path, path: Path) -> set[str]:
    members: set[str] = set()
    for line in path.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in members:
            raise ValueError(f"duplicate manifest member: {relative}")
        members.add(relative)
        target = bundle / _safe_relative(relative)
        if not target.is_file() or sha256_path(target) != digest:
            raise ValueError(f"sealed evidence drifted: {relative}")
    return members


def verify_bundle(bundle: Path) -> dict[str, object]:
    evidence = bundle / "reports" / "EVIDENCE.sha256"
    runtime = bundle / "RUNTIME.sha256"
    provenance_path = bundle / "reports" / "provenance-seal.json"
    for path in (evidence, runtime, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"sealed campaign artifact is absent: {path.name}")
    _verify_manifest(bundle, bundle / "MANIFEST.sha256")
    _verify_manifest(bundle, evidence)
    runtime_members = _verify_manifest(bundle, runtime)
    actual_runtime = {
        path.relative_to(bundle).as_posix()
        for root in (bundle / "runtime", bundle / "incidents")
        for path in root.rglob("*")
        if path.is_file()
    }
    if runtime_members != actual_runtime:
        raise ValueError("runtime membership changed after sealing")

    result = json.loads(
        (bundle / "reports" / "final-analysis.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    if (
        result.get("schema_version") != "pldr-observable-balance-analysis-v1"
        or result.get("campaign_id") != CAMPAIGN_ID
        or result["resource"]["within_limits"] is not True
        or provenance.get("campaign_id") != CAMPAIGN_ID
        or provenance.get("evidence_manifest_sha256") != sha256_path(evidence)
        or provenance.get("runtime_manifest_sha256") != sha256_path(runtime)
        or provenance.get("static_manifest_sha256")
        != sha256_path(bundle / "MANIFEST.sha256")
        or int(provenance.get("completed_plan_nodes", -1)) != 57
    ):
        raise ValueError("observable-balance analysis or provenance gate failed")
    return provenance






