"""Strict evidence binding for the prospective gate-shape campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np


REQUEST_SCHEMA = "pldr-gate-shape-request-v1"
RECORD_SCHEMA = "pldr-gate-shape-record-v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value):
    """Convert NumPy scalars and emit the one canonical JSON encoding."""

    def convert(item):
        if isinstance(item, dict):
            return {str(key): convert(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [convert(value) for value in item]
        if isinstance(item, np.ndarray):
            return convert(item.tolist())
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return item.as_posix()
        return item

    return json.dumps(
        convert(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")


def digest_object(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path, value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    temporary.replace(destination)


def write_npz_atomic(path, **arrays):
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("NumPy archives must be written under .npz names")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(destination)


def load_npz(path, *, required=()):
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("a NumPy archive must have a .npz name")
    with np.load(path, allow_pickle=False) as archive:
        available = set(archive.files)
        missing = set(required) - available
        if missing:
            raise ValueError(
                "NumPy archive is missing arrays: " + ", ".join(sorted(missing)))
        return {name: archive[name].copy() for name in archive.files}


def _normalized_relative(value):
    relative = Path(value)
    if (
        relative.is_absolute() or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("artifact paths must be normalized and relative")
    return relative


def _resolve_under(root, relative):
    root = Path(root).resolve()
    target = (root / relative).resolve()
    if target == root or root not in target.parents:
        raise ValueError("artifact path escapes its campaign root")
    return target


def canonical_request_digest(request):
    unsigned = dict(request)
    unsigned.pop("request_sha256", None)
    return digest_object(unsigned)


def seal_request(stage, campaign_id, artifact_root, bindings, protocol_path):
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    artifacts = {}
    resolved_targets = set()
    for raw in bindings:
        role, separator, value = raw.partition("=")
        if not separator or not role or role in artifacts:
            raise ValueError("artifacts must be unique ROLE=PATH bindings")
        relative = _normalized_relative(value)
        target = _resolve_under(root, relative)
        if not target.is_file():
            raise ValueError("an artifact binding does not resolve to a file")
        if target in resolved_targets:
            raise ValueError("one artifact file cannot fill two evidence roles")
        resolved_targets.add(target)
        artifacts[role] = {
            "path": relative.as_posix(),
            "sha256": sha256_path(target),
        }
    if not artifacts:
        raise ValueError("a stage request cannot have an empty artifact set")
    protocol = Path(protocol_path).resolve()
    if not protocol.is_file():
        raise ValueError("stage protocol does not exist")
    request = {
        "schema_version": REQUEST_SCHEMA,
        "campaign_id": str(campaign_id),
        "stage": str(stage),
        "artifact_root": str(root),
        "protocol": {"path": str(protocol), "sha256": sha256_path(protocol)},
        "artifacts": artifacts,
    }
    request["request_sha256"] = canonical_request_digest(request)
    return request


def load_request(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        request = json.load(stream)
    required = {
        "schema_version", "campaign_id", "stage", "artifact_root",
        "protocol", "artifacts", "request_sha256",
    }
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("evidence request has missing or unknown fields")
    if request["schema_version"] != REQUEST_SCHEMA:
        raise ValueError("unknown gate-shape request schema")
    if request["request_sha256"] != canonical_request_digest(request):
        raise ValueError("evidence request digest does not replay")
    root = Path(request["artifact_root"]).resolve()
    if not root.is_dir():
        raise ValueError("evidence root no longer resolves")
    protocol = request["protocol"]
    if not isinstance(protocol, dict) or set(protocol) != {"path", "sha256"}:
        raise ValueError("protocol binding is malformed")
    protocol_path = Path(protocol["path"]).resolve()
    if (
        not protocol_path.is_file()
        or sha256_path(protocol_path) != protocol["sha256"]
    ):
        raise ValueError("protocol binding changed after sealing")
    resolved = {}
    resolved_targets = set()
    for role, binding in request["artifacts"].items():
        if (
            not isinstance(role, str) or not role
            or not isinstance(binding, dict)
            or set(binding) != {"path", "sha256"}
            or not _SHA256.fullmatch(str(binding["sha256"]))
        ):
            raise ValueError("an artifact binding is malformed")
        relative = _normalized_relative(binding["path"])
        target = _resolve_under(root, relative)
        if not target.is_file() or sha256_path(target) != binding["sha256"]:
            raise ValueError("artifact binding changed after sealing")
        if target in resolved_targets:
            raise ValueError("one artifact file fills two evidence roles")
        resolved_targets.add(target)
        resolved[role] = target
    if not resolved:
        raise ValueError("evidence request has no artifacts")
    return request, resolved, protocol_path


def seal_record(record):
    value = dict(record)
    value.setdefault("schema_version", RECORD_SCHEMA)
    value.pop("record_sha256", None)
    value["record_sha256"] = digest_object(value)
    return value


def verify_record(record):
    if not isinstance(record, dict):
        raise ValueError("sealed record must be an object")
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    if digest != digest_object(unsigned):
        raise ValueError("sealed record digest does not replay")
    return True


def load_event_ledger(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"event ledger line {line_number} is not JSON") from error
            if not isinstance(row, dict):
                raise ValueError("event ledger rows must be objects")
            rows.append(row)
    if not rows:
        raise ValueError("event ledger cannot be empty")
    return rows


def ledger_identity(path, *, expected_updates=None):
    rows = load_event_ledger(path)
    run_ids = {row.get("run_id") for row in rows}
    lineage_roots = {row.get("lineage_root") for row in rows}
    if len(run_ids) != 1 or None in run_ids or "" in run_ids:
        raise ValueError("event ledger does not carry one resolved run id")
    if (
        len(lineage_roots) != 1 or None in lineage_roots
        or "" in lineage_roots
    ):
        raise ValueError("event ledger does not carry one resolved lineage root")
    step_rows = [row for row in rows if "step" in row and row.get("event") != "config"]
    steps = sorted({int(row["step"]) for row in step_rows})
    if expected_updates is not None:
        expected_updates = int(expected_updates)
        if expected_updates < 1 or not steps or steps[-1] != expected_updates:
            raise ValueError("event ledger does not reach the registered update count")
    return {
        "run_id": next(iter(run_ids)),
        "lineage_root": next(iter(lineage_roots)),
        "rows": len(rows),
        "maximum_step": max(steps) if steps else None,
    }
