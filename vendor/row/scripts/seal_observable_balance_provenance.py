#!/usr/bin/env python3
"""Seal complete runtime and incident membership after campaign execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"unsafe relative path: {value}")
    return Path(*candidate.parts)


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _verify_digest_manifest(bundle: Path, manifest: Path) -> int:
    count = 0
    members: set[str] = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in members:
            raise ValueError(f"duplicate manifest member: {relative}")
        members.add(relative)
        if _sha256(bundle / _safe_relative(relative)) != digest:
            raise ValueError(f"manifest digest mismatch: {relative}")
        count += 1
    return count


def _verify_completions(bundle: Path, plan: dict[str, Any]) -> tuple[int, int]:
    completed_count = 0
    attempt_count = 0
    for node in plan["nodes"]:
        completion_path = bundle / "runtime" / str(node["id"]) / "completed.json"
        if not completion_path.is_file():
            raise ValueError(f"planned node is incomplete: {node['id']}")
        completion = _load_json(completion_path)
        if (
            completion.get("node") != node["id"]
            or completion.get("role") != node["role"]
            or completion.get("planned_device") != node["device"]
        ):
            raise ValueError(f"completion identity drifted: {node['id']}")
        expected = set(str(value) for value in node["expected_outputs"])
        if set(completion.get("outputs", {})) != expected:
            raise ValueError(f"completion output membership drifted: {node['id']}")
        for relative, digest in completion["outputs"].items():
            if _sha256(bundle / _safe_relative(relative)) != digest:
                raise ValueError(f"completion output digest drifted: {relative}")
        attempt_path = bundle / _safe_relative(completion["attempt"])
        if _sha256(attempt_path / "attempt.json") != completion["attempt_sha256"]:
            raise ValueError(f"completion attempt digest drifted: {node['id']}")
        completed_count += 1
    for attempt_path in (bundle / "runtime").glob("*/attempt-*/attempt.json"):
        attempt = _load_json(attempt_path)
        node_id = attempt_path.parent.parent.name
        if attempt.get("node") != node_id:
            raise ValueError(f"attempt identity drifted: {attempt_path}")
        environment = attempt_path.parent / "environment.json"
        log = attempt_path.parent / "stdout.log"
        if (
            _sha256(environment) != attempt["environment_sha256"]
            or _sha256(log) != attempt["stdout_sha256"]
        ):
            raise ValueError(f"attempt sidecar digest drifted: {attempt_path}")
        attempt_count += 1
    return completed_count, attempt_count


def _verify_incidents(bundle: Path, campaign_id: str) -> int:
    root = bundle / "incidents"
    index_path = root / "index.json"
    index = _load_json(index_path)
    if (
        index.get("schema_version")
        != "pldr-observable-balance-incident-index-v1"
        or index.get("campaign_id") != campaign_id
        or not isinstance(index.get("incidents"), list)
    ):
        raise ValueError("incident index is malformed")
    registered = {str(value) for value in index["incidents"]}
    for relative in registered:
        _safe_relative(relative)
        if not relative.startswith("incidents/") or relative == "incidents/index.json":
            raise ValueError(f"invalid incident member: {relative}")
    actual = {
        path.relative_to(bundle).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path != index_path
    }
    if registered != actual:
        raise ValueError("incident index membership disagrees with incident files")
    return len(actual)


def _runtime_files(bundle: Path) -> list[Path]:
    return sorted(
        [
            path
            for root in (bundle / "runtime", bundle / "incidents")
            for path in root.rglob("*")
            if path.is_file()
        ],
        key=lambda path: path.relative_to(bundle).as_posix(),
    )






