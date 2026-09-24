"""Immutable artifact and constructor-ownership checks."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath


HEX64 = re.compile(r"^[0-9a-f]{64}$")
REQUEST_SCHEMA_VERSION = "pldr-program-energy-request-v1"


def sha256_path(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_relative(root, relative):
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path must be a nonempty relative path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError("artifact path must be normalized and relative")
    root = Path(root).resolve()
    path = (root / Path(*pure.parts)).resolve()
    if path != root and root not in path.parents:
        raise ValueError("artifact path escapes its immutable root")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_artifacts(root, rows, required_roles):
    if not isinstance(rows, list):
        raise ValueError("artifacts must be an array")
    required_roles = set(required_roles)
    result = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "role", "path", "sha256",
        }:
            raise ValueError("artifact binding has missing or unknown fields")
        role = row["role"]
        if not isinstance(role, str) or not role or role in result:
            raise ValueError("artifact roles must be unique nonempty strings")
        if not isinstance(row["sha256"], str) or not HEX64.fullmatch(
            row["sha256"]
        ):
            raise ValueError("artifact digest must be lowercase hexadecimal")
        path = resolve_relative(root, row["path"])
        observed = sha256_path(path)
        if observed != row["sha256"]:
            raise ValueError(f"artifact digest mismatch for role {role}")
        result[role] = {
            "path": row["path"],
            "sha256": observed,
            "resolved_path": str(path),
        }
    if set(result) != required_roles:
        raise ValueError(
            "artifact roles disagree with the frozen protocol: "
            f"missing={sorted(required_roles - set(result))}, "
            f"extra={sorted(set(result) - required_roles)}"
        )
    return result


def validate_request(request):
    required = {
        "schema_version", "campaign_id", "stage", "time_semantics",
        "artifacts",
    }
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("request has missing or unknown fields")
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ValueError("unknown program-energy request schema")
    for name in ("campaign_id", "stage", "time_semantics"):
        if not isinstance(request[name], str) or not request[name]:
            raise ValueError(f"{name} must be a nonempty string")
    forbidden = {
        "complete_operator", "comparison_operator", "forcing",
        "direct_row_map_upper", "target_radius", "radius_multiplier",
        "decision", "validation_membership", "cover_utility_margin",
    }
    if forbidden & set(request):
        raise ValueError("request contains a caller-supplied scientific attestation")
    return True
