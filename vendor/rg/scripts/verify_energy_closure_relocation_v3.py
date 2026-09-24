#!/usr/bin/env python3
"""Relocate a closure artifact graph and prove analysis-path portability."""

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

from row_rgmap.analysis import (  # noqa: E402
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402
from scripts.analyze_energy_closure_probe_v3 import (  # noqa: E402
    RESULT_SCHEMA,
    RUN_SCHEMA,
    resolve_data_root_locator,
)


SCHEMA = "pldr-row-rg-energy-closure-relocation-validation-v3"
SOURCE_PATHS = (
    "scripts/verify_energy_closure_relocation_v3.py",
    "scripts/analyze_energy_closure_probe_v3.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/closure_contract.py",
    "src/row_rgmap/provenance_v3.py",
)
VOLATILE_OR_PROVENANCE_FIELDS = (
    "analysis_sources",
    "code_commit",
    "completed_at_utc",
    "record_sha256",
)


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("relocation validation requires a clean committed repository")
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        worktree = ROOT / repository_path
        if (
            not worktree.is_file()
            or file_sha256(worktree) != digest
            or worktree.stat().st_size != size
        ):
            raise RuntimeError(f"relocation source differs from HEAD: {repository_path}")
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def copy_member(
    source_root: Path,
    target_root: Path,
    relative: str,
    inventory: dict[str, dict[str, Any]],
) -> None:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("relocation member is not a safe relative path")
    source = (source_root / member).resolve()
    target = (target_root / member).resolve()
    try:
        source.relative_to(source_root.resolve())
        target.relative_to(target_root.resolve())
    except ValueError as error:
        raise ValueError("relocation member escapes its root") from error
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    source_digest = file_sha256(source)
    source_size = source.stat().st_size
    target_digest = file_sha256(target)
    target_size = target.stat().st_size
    bytes_equal = target_digest == source_digest and target_size == source_size
    if not bytes_equal:
        raise ValueError("relocated artifact bytes changed")
    key = member.as_posix()
    row = {
        "path": key,
        "source": {
            "anchor": "data_root",
            "path": key,
            "sha256": source_digest,
            "size_bytes": source_size,
        },
        "relocated_copy": {
            "anchor": "relocated_data_root",
            "path": key,
            "sha256": target_digest,
            "size_bytes": target_size,
        },
        "bytes_equal": bytes_equal,
    }
    previous = inventory.get(key)
    if previous is not None and previous != row:
        raise ValueError("duplicate relocation member has inconsistent identity")
    inventory[key] = row


def descriptor_member(prefix: Path, descriptor: dict[str, Any]) -> str:
    relative = descriptor.get("path")
    if not isinstance(relative, str):
        raise ValueError("artifact descriptor has no relative path")
    return (prefix / relative).as_posix()


def validate_source_checkpoint_census(
    run_root: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Census original checkpoint directories before any relocation copy."""

    run_root = run_root.resolve()
    expected_global: set[str] = set()
    expected_checkpoint_roots: set[str] = set()
    total_size = 0
    branch_censuses = []
    branches = manifest.get("branches")
    if not isinstance(branches, dict) or not branches:
        raise ValueError("source branch registry is missing")
    for trajectory_id, rows in branches.items():
        if not isinstance(trajectory_id, str) or not isinstance(rows, list):
            raise ValueError("source branch registry is malformed")
        for row in rows:
            branch = row.get("branch")
            descriptors = row.get("output_checkpoints")
            if not isinstance(branch, str) or not isinstance(descriptors, list):
                raise ValueError("source checkpoint descriptor registry is malformed")
            checkpoint_root = (
                run_root / "trajectories" / trajectory_id / branch / "checkpoints"
            )
            if not checkpoint_root.is_dir() or checkpoint_root.is_symlink():
                raise ValueError("source checkpoint directory is missing or a symlink")
            expected_checkpoint_roots.add(
                checkpoint_root.relative_to(run_root).as_posix()
            )
            expected: set[str] = set()
            for descriptor in descriptors:
                if (
                    not isinstance(descriptor, dict)
                    or set(descriptor)
                    != {
                        "trajectory_id",
                        "branch",
                        "step",
                        "path",
                        "sha256",
                        "size_bytes",
                    }
                    or descriptor.get("trajectory_id") != trajectory_id
                    or descriptor.get("branch") != branch
                ):
                    raise ValueError("source checkpoint descriptor is malformed")
                relative_text = descriptor.get("path")
                if not isinstance(relative_text, str):
                    raise ValueError("source checkpoint path is missing")
                relative = Path(relative_text)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("source checkpoint path is not a safe relative path")
                unresolved = run_root / relative
                try:
                    unresolved.parent.resolve().relative_to(run_root)
                    unresolved.relative_to(checkpoint_root)
                except ValueError as error:
                    raise ValueError(
                        "source checkpoint descriptor escapes its checkpoint directory"
                    ) from error
                if unresolved.is_symlink() or not unresolved.is_file():
                    raise ValueError("source checkpoint is missing or a symlink")
                if (
                    file_sha256(unresolved) != descriptor.get("sha256")
                    or unresolved.stat().st_size != descriptor.get("size_bytes")
                ):
                    raise ValueError("source checkpoint descriptor fails")
                key = relative.as_posix()
                if key in expected_global:
                    raise ValueError("source checkpoint descriptor is duplicated")
                expected.add(key)
                expected_global.add(key)
                total_size += unresolved.stat().st_size
            actual: set[str] = set()
            for directory, directory_names, file_names in os.walk(
                checkpoint_root, followlinks=False
            ):
                directory_path = Path(directory)
                if directory_names:
                    raise ValueError("source checkpoint directory has undeclared subdirectories")
                for name in file_names:
                    member = directory_path / name
                    if member.is_symlink() or not member.is_file():
                        raise ValueError("source checkpoint inventory contains a non-file")
                    actual.add(member.relative_to(run_root).as_posix())
            if actual != expected:
                raise ValueError("source checkpoint directory census is not exact")
            branch_censuses.append(
                {
                    "trajectory_id": trajectory_id,
                    "branch": branch,
                    "checkpoint_count": len(expected),
                    "paths": sorted(expected),
                }
            )
    trajectory_root = run_root / "trajectories"
    actual_checkpoint_roots: set[str] = set()
    for checkpoint_root in trajectory_root.glob("*/*/checkpoints"):
        if checkpoint_root.is_symlink() or not checkpoint_root.is_dir():
            raise ValueError("source checkpoint-directory census contains a non-directory")
        actual_checkpoint_roots.add(
            checkpoint_root.relative_to(run_root).as_posix()
        )
    if actual_checkpoint_roots != expected_checkpoint_roots:
        raise ValueError("source checkpoint-directory census is not exact")

    inventory = manifest.get("checkpoint_inventory", {})
    if (
        inventory.get("descriptor_count") != len(expected_global)
        or inventory.get("total_size_bytes") != total_size
        or inventory.get("all_emitted_files_bound") is not True
    ):
        raise ValueError("source global checkpoint census differs from manifest")
    encoded = json.dumps(
        sorted(expected_global), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "checkpoint_count": len(expected_global),
        "total_size_bytes": total_size,
        "path_census_sha256": hashlib.sha256(encoded).hexdigest(),
        "branches": branch_censuses,
        "exact": True,
    }


def scientific_payload(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in VOLATILE_OR_PROVENANCE_FIELDS
    }


def inventory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    encoded = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(row["source"]["size_bytes"] for row in rows),
        "inventory_sha256": hashlib.sha256(encoded).hexdigest(),
        "files": rows,
    }


def validate_relocation(
    data_root: Path,
    run_locator: dict[str, str],
    *,
    code_commit: str,
    analysis_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    data_root = data_root.resolve()
    run_root = resolve_data_root_locator(data_root, run_locator)
    manifest_path = run_root / "run-manifest.json"
    result_path = run_root / "result.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    original_result = json.loads(result_path.read_text(encoding="utf-8"))
    verify_record_seal(manifest)
    verify_record_seal(original_result)
    if (
        manifest.get("schema_version") != RUN_SCHEMA
        or original_result.get("schema_version") != RESULT_SCHEMA
    ):
        raise ValueError("relocation input schema mismatch")
    source_checkpoint_census = validate_source_checkpoint_census(
        run_root, manifest
    )

    with tempfile.TemporaryDirectory(
        prefix="row-rgmap-closure-relocation-"
    ) as directory:
        relocated_root = Path(directory).resolve()
        roots_distinct = relocated_root != data_root
        if not roots_distinct:
            raise ValueError("relocation root is not distinct")
        copied: dict[str, dict[str, Any]] = {}

        run_prefix = Path(run_locator["path"])
        copy_member(
            data_root,
            relocated_root,
            (run_prefix / "run-manifest.json").as_posix(),
            copied,
        )
        for descriptor in manifest["inputs"].values():
            copy_member(
                data_root,
                relocated_root,
                descriptor_member(run_prefix, descriptor),
                copied,
            )
        for branch_rows in manifest["branches"].values():
            for row in branch_rows:
                for key in (
                    "resume_checkpoint",
                    "energies",
                    "metrics",
                    "metadata",
                    "producer_log",
                ):
                    copy_member(
                        data_root,
                        relocated_root,
                        descriptor_member(run_prefix, row[key]),
                        copied,
                    )
                for checkpoint in row["output_checkpoints"]:
                    copy_member(
                        data_root,
                        relocated_root,
                        descriptor_member(run_prefix, checkpoint),
                        copied,
                    )

        source_locator = manifest["source_run"]["locator"]
        source_prefix = Path(source_locator["path"])
        for name in ("config-resolved.json", "protocol-resolved.json"):
            copy_member(
                data_root,
                relocated_root,
                (source_prefix / name).as_posix(),
                copied,
            )
        for descriptor in manifest["source_run"]["checkpoints"]:
            for key in ("path", "metadata_path"):
                copy_member(
                    data_root,
                    relocated_root,
                    (source_prefix / descriptor[key]).as_posix(),
                    copied,
                )

        for key in ("run_manifest", "result"):
            descriptor = manifest["predecessor"][key]
            copy_member(
                data_root,
                relocated_root,
                descriptor["path"],
                copied,
            )

        relocated_run = resolve_data_root_locator(relocated_root, run_locator)
        relocated_result_path = relocated_root / "relocated-result.json"
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = "src:."
        command = [
            sys.executable,
            "scripts/analyze_energy_closure_probe_v3.py",
            "--run-root",
            str(relocated_run),
            "--data-root",
            str(relocated_root),
            "--output",
            str(relocated_result_path),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        analyzer_completed = completed.returncode == 0 and relocated_result_path.is_file()
        if not analyzer_completed:
            raise RuntimeError(
                "relocated analyzer failed: "
                + (completed.stderr or completed.stdout)[-2000:]
            )
        relocated_result = json.loads(
            relocated_result_path.read_text(encoding="utf-8")
        )
        verify_record_seal(relocated_result)
        equal = scientific_payload(original_result) == scientific_payload(
            relocated_result
        )
        if not equal:
            raise ValueError("relocated analysis changed scientific payload")
        rows = [copied[key] for key in sorted(copied)]
        all_copied_bytes_equal = all(row["bytes_equal"] for row in rows)
        successor_checkpoint_paths = {
            descriptor_member(run_prefix, checkpoint)
            for branch_rows in manifest["branches"].values()
            for row in branch_rows
            for checkpoint in row["output_checkpoints"]
        }
        relocated_successor_checkpoint_count = sum(
            path in copied for path in successor_checkpoint_paths
        )
        expected_successor_checkpoint_count = manifest["checkpoint_inventory"][
            "descriptor_count"
        ]
        all_successor_checkpoints_relocated = (
            relocated_successor_checkpoint_count
            == expected_successor_checkpoint_count
            == len(successor_checkpoint_paths)
        )
        if not all_copied_bytes_equal or not all_successor_checkpoints_relocated:
            raise ValueError("relocated artifact inventory is incomplete")
        payload = {
            "schema_version": SCHEMA,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit,
            "analysis_sources": analysis_sources,
            "run_locator": run_locator,
            "original_result": {
                "anchor": "data_root",
                "path": result_path.relative_to(data_root).as_posix(),
                "sha256": file_sha256(result_path),
                "size_bytes": result_path.stat().st_size,
                "record_sha256": original_result["record_sha256"],
            },
            "relocated_result": {
                "sha256": file_sha256(relocated_result_path),
                "size_bytes": relocated_result_path.stat().st_size,
                "record_sha256": relocated_result["record_sha256"],
                "analyzer_stdout_sha256": hashlib.sha256(
                    completed.stdout.encode("utf-8")
                ).hexdigest(),
            },
            "copied_artifacts": inventory_summary(rows),
            "source_checkpoint_census": source_checkpoint_census,
            "successor_checkpoint_inventory": {
                "expected_count": expected_successor_checkpoint_count,
                "relocated_count": relocated_successor_checkpoint_count,
                "all_relocated": all_successor_checkpoints_relocated,
            },
            "comparison": {
                "excluded_top_level_fields": list(
                    VOLATILE_OR_PROVENANCE_FIELDS
                ),
                "scientific_payload_equal": equal,
            },
            "checks": {
                "source_checkpoint_census_exact": source_checkpoint_census["exact"],
                "roots_distinct": roots_distinct,
                "all_copied_artifact_bytes_equal": all_copied_bytes_equal,
                "all_successor_checkpoints_relocated": (
                    all_successor_checkpoints_relocated
                ),
                "relocated_analyzer_completed": analyzer_completed,
                "scientific_payload_equal": equal,
            },
        }
    return seal_record(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--run-path", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    run_path = Path(arguments.run_path)
    if run_path.is_absolute() or ".." in run_path.parts or not run_path.parts:
        raise ValueError("--run-path must be a nonescaping data-root-relative path")
    commit, sources = committed_identity()
    record = validate_relocation(
        Path(arguments.data_root),
        {"anchor": "data_root", "path": run_path.as_posix()},
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
                "copied_file_count": record["copied_artifacts"]["file_count"],
                "scientific_payload_equal": record["comparison"][
                    "scientific_payload_equal"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
