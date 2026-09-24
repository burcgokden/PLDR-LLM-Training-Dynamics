#!/usr/bin/env python3
"""Project consumed full checkpoints to validated model-only retention files."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.confirmation_artifacts import (  # noqa: E402
    MODEL_ONLY_CHECKPOINT_FIELDS,
    MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION,
    build_model_only_state_manifest,
    sha256_path,
    validate_complete_checkpoint,
    validate_model_only_checkpoint,
)


REPORT_SCHEMA = "pldr-model-only-retention-record-v1"


def model_only_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    validate_complete_checkpoint(checkpoint)
    retained = {
        name: checkpoint[name]
        for name in MODEL_ONLY_CHECKPOINT_FIELDS
        if name not in {"schema_version", "state_manifest"}
    }
    retained["schema_version"] = MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION
    retained["state_manifest"] = build_model_only_state_manifest(retained)
    validate_model_only_checkpoint(retained)
    return retained


def project(path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(path).resolve()
    before_bytes = checkpoint_path.stat().st_size
    before_sha256 = sha256_path(checkpoint_path)
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") == MODEL_ONLY_CHECKPOINT_SCHEMA_VERSION:
        validate_model_only_checkpoint(payload)
        complete_state_sha256 = None
        disposition = "already_model_only"
    else:
        complete_state_sha256 = payload.get(
            "state_manifest", {}).get("complete_state_sha256")
        payload = model_only_payload(payload)
        temporary = checkpoint_path.with_name(
            checkpoint_path.name + ".model-only.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
        disposition = "projected_after_coverage"
    return {
        "path": str(checkpoint_path),
        "disposition": disposition,
        "full_checkpoint_sha256": before_sha256,
        "complete_state_sha256": complete_state_sha256,
        "model_state_sha256": payload["state_manifest"]["model_state_sha256"],
        "retained_file_sha256": sha256_path(checkpoint_path),
        "bytes_before": before_bytes,
        "bytes_after": checkpoint_path.stat().st_size,
        "optimizer_state_retained": False,
        "recoverable_from_retained_file": False,
    }


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--also", action="append", default=[])
    parser.add_argument("--report", required=True)
    arguments = parser.parse_args()
    paths = [arguments.checkpoint, *arguments.also]
    if len({str(Path(path).resolve()) for path in paths}) != len(paths):
        raise ValueError("retention paths must be unique")
    records = [project(path) for path in paths]
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "model_only_retained",
        "records": records,
        "all_optimizer_states_removed": all(
            not record["optimizer_state_retained"] for record in records),
    }
    write_json(arguments.report, report)
    print(json.dumps({
        "output": str(Path(arguments.report).resolve()),
        "projected": len(records),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
