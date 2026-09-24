"""Strict evidence binding and replay semantics for the rev34 program."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from confirmation_artifacts import (
    digest_object,
    nested_state_digest,
    sha256_path,
)


REQUEST_SCHEMA = "pldr-evidence-request-v4"
STAGE_RECORD_SCHEMA = "pldr-evidence-stage-record-v4"
CAMPAIGN_RECORD_SCHEMA = "pldr-evidence-campaign-record-v4"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_request_digest(request):
    unsigned = dict(request)
    unsigned.pop("request_sha256", None)
    return digest_object(unsigned)


def seal_request(stage, campaign_id, artifact_root, bindings, protocol_path):
    root = Path(artifact_root).resolve()
    if not root.is_dir():
        raise ValueError("artifact root must be an existing directory")
    artifacts = {}
    for raw in bindings:
        role, separator, value = raw.partition("=")
        if not separator or not role or role in artifacts:
            raise ValueError("artifacts must be unique ROLE=PATH bindings")
        relative = Path(value)
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise ValueError("artifact bindings must use normalized relative paths")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("an artifact binding does not resolve inside its root")
        artifacts[role] = {
            "path": relative.as_posix(),
            "sha256": sha256_path(path),
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
        raise ValueError("stage request has missing or unknown fields")
    if request["schema_version"] != REQUEST_SCHEMA:
        raise ValueError("unknown stage request schema")
    if request["request_sha256"] != canonical_request_digest(request):
        raise ValueError("stage request digest does not replay")
    root = Path(request["artifact_root"]).resolve()
    if not root.is_dir():
        raise ValueError("stage request artifact root no longer resolves")
    protocol = request["protocol"]
    if set(protocol) != {"path", "sha256"}:
        raise ValueError("stage request protocol binding is malformed")
    protocol_path = Path(protocol["path"]).resolve()
    if not protocol_path.is_file() or sha256_path(protocol_path) != protocol["sha256"]:
        raise ValueError("stage protocol binding does not resolve")
    resolved = {}
    for role, binding in request["artifacts"].items():
        if not isinstance(role, str) or not role or set(binding) != {"path", "sha256"}:
            raise ValueError("stage artifact binding is malformed")
        if not _SHA256.fullmatch(binding["sha256"]):
            raise ValueError("stage artifact digest is malformed")
        relative = Path(binding["path"])
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise ValueError("stage artifact path is unsafe")
        target = (root / relative).resolve()
        if root not in target.parents or not target.is_file():
            raise ValueError("stage artifact path does not resolve")
        if sha256_path(target) != binding["sha256"]:
            raise ValueError("stage artifact digest changed after sealing")
        resolved[role] = target
    if not resolved:
        raise ValueError("stage request has no resolved evidence")
    return request, resolved, protocol_path


def scientific_checkpoint_components(checkpoint):
    """Digest scientific state while excluding invocation-only labels."""

    data = checkpoint["data_state"]
    branch = checkpoint["branch_state"]
    components = {
        "model": nested_state_digest(checkpoint["model"]),
        "optimizer": nested_state_digest(checkpoint["opt"]),
        "scheduler": nested_state_digest(checkpoint["scheduler"]),
        "rng": nested_state_digest(checkpoint["rng_states"]),
        "optimizer_config": digest_object(checkpoint["optimizer_config"]),
        "model_config": digest_object(checkpoint["model_config"]),
        "operation_order": digest_object(checkpoint["operation_order"]),
        "branch_scientific": nested_state_digest({
            "clip_mask": branch["clip_mask"],
            "decay_mask": branch["decay_mask"],
            "dtype": branch["dtype"],
            "scheduler_phase": branch["scheduler_phase"],
        }),
        "registry": digest_object(checkpoint["measurement_registry"]),
        "intervention": digest_object(checkpoint["intervention_state"]),
        "next_minibatch": digest_object({
            "dataset_sha256": data["dataset_sha256"],
            "tokenizer_sha256": data["tokenizer_sha256"],
            "cursor_end": data["cursor_end"],
            "rows_per_step": data["rows_per_step"],
            "context_length": data["context_length"],
            "probe_region": data["probe_region"],
        }),
        "step": int(checkpoint["step"]),
    }
    return components


def replay_comparison(primary, replay):
    primary_identity = primary.get("run_identity")
    replay_identity = replay.get("run_identity")
    identities_valid = (
        isinstance(primary_identity, dict)
        and isinstance(replay_identity, dict)
        and primary_identity.get("run_id")
        and replay_identity.get("run_id")
        and primary_identity["run_id"] != replay_identity["run_id"]
        and primary_identity.get("from_scratch") is True
        and replay_identity.get("from_scratch") is True
        and primary_identity.get("lineage_root")
        != replay_identity.get("lineage_root")
    )
    primary_components = scientific_checkpoint_components(primary)
    replay_components = scientific_checkpoint_components(replay)
    differences = sorted(
        name for name in primary_components
        if primary_components[name] != replay_components.get(name))
    return {
        "identities_valid": identities_valid,
        "scientific_components_equal": not differences,
        "different_components": differences,
        "primary_run_id": (
            primary_identity.get("run_id") if isinstance(primary_identity, dict)
            else None),
        "replay_run_id": (
            replay_identity.get("run_id") if isinstance(replay_identity, dict)
            else None),
    }


def seal_record(record):
    value = dict(record)
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
