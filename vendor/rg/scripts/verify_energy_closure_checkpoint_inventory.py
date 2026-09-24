#!/usr/bin/env python3
"""Relocate a sealed closure-checkpoint inventory and run tamper challenges."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from row_rgmap.analysis import file_sha256, seal_record, verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402
from scripts.create_energy_closure_checkpoint_inventory import (  # noqa: E402
    SCHEMA as INVENTORY_SCHEMA,
    safe_member,
)


SCHEMA = "pldr-row-rg-energy-closure-checkpoint-relocation-v1"
SOURCE_PATHS = (
    "scripts/verify_energy_closure_checkpoint_inventory.py",
    "scripts/create_energy_closure_checkpoint_inventory.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
)


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("checkpoint relocation requires a clean committed repository")
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def verify_inventory_sources(record: dict[str, Any]) -> None:
    commit = record.get("code_commit")
    sources = record.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(sources, list):
        raise ValueError("checkpoint inventory source registry is incomplete")
    for row in sources:
        digest, size = git_blob_descriptor(ROOT, commit, row["repository_path"])
        if (
            row.get("code_commit") != commit
            or row.get("sha256") != digest
            or row.get("size_bytes") != size
        ):
            raise ValueError("checkpoint inventory source registry does not replay")


def validate_checkpoint_set(root: Path, inventory: dict[str, Any]) -> list[Path]:
    root = root.resolve()
    paths = []
    for row in inventory["checkpoints"]:
        path = safe_member(root, row["path"])
        if file_sha256(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise ValueError("checkpoint descriptor does not replay")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("checkpoint inventory contains duplicate paths")
    run_root = safe_member(
        root, inventory["source_run"]["locator"]["path"], require_file=False
    )
    actual = {
        path.resolve()
        for path in (run_root / "trajectories").glob("*/*/checkpoints/*")
        if path.is_file()
    }
    if actual != set(paths):
        raise ValueError("checkpoint filesystem set differs from the sealed inventory")
    return paths


def challenge_tamper(root: Path, inventory: dict[str, Any], row: dict[str, Any]) -> bool:
    path = safe_member(root, row["path"])
    original_size = path.stat().st_size
    try:
        with path.open("ab") as stream:
            stream.write(b"tamper")
        try:
            validate_checkpoint_set(root, inventory)
        except ValueError:
            return True
        return False
    finally:
        with path.open("r+b") as stream:
            stream.truncate(original_size)


def challenge_missing(root: Path, inventory: dict[str, Any], row: dict[str, Any]) -> bool:
    path = safe_member(root, row["path"])
    holding = root / "challenge-holding-file"
    os.replace(path, holding)
    try:
        try:
            validate_checkpoint_set(root, inventory)
        except (FileNotFoundError, ValueError):
            return True
        return False
    finally:
        os.replace(holding, path)


def challenge_extra(root: Path, inventory: dict[str, Any], row: dict[str, Any]) -> bool:
    registered = safe_member(root, row["path"])
    extra = registered.parent / "unregistered-checkpoint.pt"
    extra.write_bytes(b"extra")
    try:
        try:
            validate_checkpoint_set(root, inventory)
        except ValueError:
            return True
        return False
    finally:
        extra.unlink()


def verify_relocation(
    data_root: Path,
    inventory_path: str,
    *,
    code_commit: str,
    analysis_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    data_root = data_root.resolve()
    source_path = safe_member(data_root, inventory_path)
    inventory = json.loads(source_path.read_text(encoding="utf-8"))
    verify_record_seal(inventory)
    verify_inventory_sources(inventory)
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("unknown checkpoint inventory schema")
    source_paths = validate_checkpoint_set(data_root, inventory)

    with tempfile.TemporaryDirectory(prefix="row-rgmap-checkpoint-relocation-") as directory:
        relocated_root = Path(directory).resolve()
        roots_distinct = relocated_root != data_root
        if not roots_distinct:
            raise ValueError("relocation root is not distinct")
        copies = []
        for row, source in zip(inventory["checkpoints"], source_paths):
            target = safe_member(relocated_root, row["path"], require_file=False)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            source_descriptor = {
                "anchor": "data_root",
                "path": row["path"],
                "sha256": file_sha256(source),
                "size_bytes": source.stat().st_size,
            }
            copy_descriptor = {
                "anchor": "relocated_data_root",
                "path": row["path"],
                "sha256": file_sha256(target),
                "size_bytes": target.stat().st_size,
            }
            copies.append(
                {
                    "trajectory_id": row["trajectory_id"],
                    "branch": row["branch"],
                    "step": row["step"],
                    "source": source_descriptor,
                    "relocated_copy": copy_descriptor,
                    "bytes_equal": source_descriptor["sha256"]
                    == copy_descriptor["sha256"]
                    and source_descriptor["size_bytes"]
                    == copy_descriptor["size_bytes"],
                }
            )
        relocated_paths = validate_checkpoint_set(relocated_root, inventory)
        all_bytes_equal = all(row["bytes_equal"] for row in copies)
        relocation_complete = len(relocated_paths) == len(source_paths) == len(copies)

        step_zero = next(row for row in inventory["checkpoints"] if row["step"] == 0)
        step_final = next(
            row
            for row in inventory["checkpoints"]
            if row["step"] == max(inventory["expected_steps_per_branch"])
        )
        challenges = {
            "step_zero_byte_tamper_rejected": challenge_tamper(
                relocated_root, inventory, step_zero
            ),
            "final_step_byte_tamper_rejected": challenge_tamper(
                relocated_root, inventory, step_final
            ),
            "missing_checkpoint_rejected": challenge_missing(
                relocated_root, inventory, step_zero
            ),
            "extra_checkpoint_rejected": challenge_extra(
                relocated_root, inventory, step_final
            ),
        }
        post_challenge_replay = len(
            validate_checkpoint_set(relocated_root, inventory)
        ) == len(copies)
        all_challenges_rejected = all(challenges.values()) and post_challenge_replay
        if not all_bytes_equal or not relocation_complete or not all_challenges_rejected:
            raise ValueError("checkpoint relocation validation failed")

        encoded = json.dumps(
            copies, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload = {
            "schema_version": SCHEMA,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit,
            "analysis_sources": analysis_sources,
            "input_inventory": {
                "anchor": "data_root",
                "path": inventory_path,
                "sha256": file_sha256(source_path),
                "size_bytes": source_path.stat().st_size,
                "record_sha256": inventory["record_sha256"],
            },
            "roots": {
                "source_role": "data_root",
                "copy_role": "temporary relocated data root",
                "distinct": roots_distinct,
            },
            "copies": {
                "file_count": len(copies),
                "total_size_bytes": sum(row["source"]["size_bytes"] for row in copies),
                "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
                "files": copies,
            },
            "tamper_challenges": challenges | {
                "post_challenge_replay": post_challenge_replay,
                "all_rejected": all_challenges_rejected,
            },
            "checks": {
                "roots_distinct": roots_distinct,
                "all_checkpoint_bytes_equal": all_bytes_equal,
                "complete_checkpoint_set_relocated": relocation_complete,
                "all_tamper_challenges_rejected": all_challenges_rejected,
            },
        }
    return seal_record(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--inventory-path", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    inventory_path = Path(arguments.inventory_path)
    if inventory_path.is_absolute() or ".." in inventory_path.parts:
        raise ValueError("--inventory-path must be data-root relative")
    commit, sources = committed_identity()
    record = verify_relocation(
        Path(arguments.data_root),
        inventory_path.as_posix(),
        code_commit=commit,
        analysis_sources=sources,
    )
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": record["record_sha256"],
                "checkpoint_count": record["copies"]["file_count"],
                "all_tamper_challenges_rejected": record["checks"][
                    "all_tamper_challenges_rejected"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
