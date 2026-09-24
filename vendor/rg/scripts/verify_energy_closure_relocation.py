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
from scripts.analyze_energy_closure_probe_v2 import (  # noqa: E402
    RESULT_SCHEMA,
    RUN_SCHEMA,
    resolve_data_root_locator,
)


SCHEMA = "pldr-row-rg-energy-closure-relocation-validation-v1"
SOURCE_PATHS = (
    "scripts/verify_energy_closure_relocation.py",
    "scripts/analyze_energy_closure_probe_v2.py",
    "src/row_rgmap/analysis.py",
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
    digest = file_sha256(source)
    if (
        file_sha256(target) != digest
        or target.stat().st_size != source.stat().st_size
    ):
        raise ValueError("relocated artifact bytes changed")
    key = member.as_posix()
    row = {
        "path": key,
        "sha256": digest,
        "size_bytes": source.stat().st_size,
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
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
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

    with tempfile.TemporaryDirectory(
        prefix="row-rgmap-closure-relocation-"
    ) as directory:
        relocated_root = Path(directory).resolve()
        if relocated_root == data_root:
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
            "scripts/analyze_energy_closure_probe_v2.py",
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
            check=True,
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
            "comparison": {
                "excluded_top_level_fields": list(
                    VOLATILE_OR_PROVENANCE_FIELDS
                ),
                "scientific_payload_equal": equal,
            },
            "checks": {
                "roots_distinct": True,
                "all_copied_artifact_bytes_equal": True,
                "relocated_analyzer_completed": True,
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
