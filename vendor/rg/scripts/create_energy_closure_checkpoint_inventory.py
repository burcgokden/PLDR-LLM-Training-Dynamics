#!/usr/bin/env python3
"""Create a sealed inventory for every checkpoint emitted by a closure run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, seal_record, verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


SCHEMA = "pldr-row-rg-energy-closure-checkpoint-inventory-v1"
RUN_SCHEMA = "pldr-row-rg-energy-closure-probe-run-v2"
PRODUCER_SCHEMA = "pldr-row-rg-energy-producer-metadata-v3"
SOURCE_PATHS = (
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
        raise RuntimeError("checkpoint inventory requires a clean committed repository")
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


def safe_member(root: Path, relative: str, *, require_file: bool = True) -> Path:
    member = Path(relative)
    if member.is_absolute() or ".." in member.parts or not member.parts:
        raise ValueError("artifact path is not a safe relative path")
    root = root.resolve()
    path = (root / member).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("artifact path escapes its root") from error
    if require_file and not path.is_file():
        raise FileNotFoundError(path)
    return path


def verify_descriptor(root: Path, descriptor: dict[str, Any]) -> Path:
    path = safe_member(root, descriptor["path"])
    if (
        descriptor.get("sha256") != file_sha256(path)
        or descriptor.get("size_bytes") != path.stat().st_size
    ):
        raise ValueError("artifact descriptor does not replay")
    return path


def anchored_descriptor(path: Path, data_root: Path) -> dict[str, Any]:
    data_root = data_root.resolve()
    path = path.resolve()
    return {
        "anchor": "data_root",
        "path": path.relative_to(data_root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def inventory_run(
    data_root: Path,
    run_path: str,
    *,
    code_commit: str,
    analysis_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    data_root = data_root.resolve()
    run_root = safe_member(data_root, run_path, require_file=False)
    if not run_root.is_dir():
        raise ValueError("closure run root is not a directory")
    manifest_path = run_root / "run-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verify_record_seal(manifest)
    if manifest.get("schema_version") != RUN_SCHEMA:
        raise ValueError("checkpoint inventory input schema mismatch")
    trajectory_ids = sorted(manifest["branches"])
    branch_ids = manifest["branch_ids"]
    if len(trajectory_ids) != 2 or len(branch_ids) != 8:
        raise ValueError("checkpoint inventory input branch registry is unexpected")

    checkpoint_rows = []
    metadata_rows = []
    for trajectory_id in trajectory_ids:
        rows = manifest["branches"][trajectory_id]
        by_branch = {row["branch"]: row for row in rows}
        if len(by_branch) != len(rows) or set(by_branch) != set(branch_ids):
            raise ValueError("closure branch registry is ambiguous")
        for branch in branch_ids:
            branch_row = by_branch[branch]
            metadata_path = verify_descriptor(run_root, branch_row["metadata"])
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if (
                metadata.get("schema_version") != PRODUCER_SCHEMA
                or metadata.get("trajectory_id") != trajectory_id
            ):
                raise ValueError("producer metadata identity mismatch")
            registered = metadata.get("checkpoints")
            if (
                not isinstance(registered, list)
                or len(registered) != 2
                or {row.get("step") for row in registered}
                != {0, int(manifest["final_step"])}
            ):
                raise ValueError("producer checkpoint registry is incomplete")
            branch_root = metadata_path.parent
            checkpoint_root = (branch_root / "checkpoints").resolve()
            declared = set()
            for item in registered:
                if not isinstance(item, dict) or set(item) != {
                    "path", "sha256", "size_bytes", "step"
                }:
                    raise ValueError("producer checkpoint descriptor is malformed")
                path = safe_member(branch_root, item["path"])
                try:
                    path.relative_to(checkpoint_root)
                except ValueError as error:
                    raise ValueError("producer checkpoint escapes its branch") from error
                if (
                    file_sha256(path) != item["sha256"]
                    or path.stat().st_size != item["size_bytes"]
                ):
                    raise ValueError("producer checkpoint descriptor fails")
                declared.add(path)
                checkpoint_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "branch": branch,
                        "step": item["step"],
                        **anchored_descriptor(path, data_root),
                    }
                )
            actual = {
                path.resolve() for path in checkpoint_root.glob("*") if path.is_file()
            }
            if actual != declared:
                raise ValueError("branch checkpoint file set is not exactly registered")
            metadata_rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "branch": branch,
                    **anchored_descriptor(metadata_path, data_root),
                }
            )

    checkpoint_rows.sort(key=lambda row: (row["trajectory_id"], row["branch"], row["step"]))
    metadata_rows.sort(key=lambda row: (row["trajectory_id"], row["branch"]))
    expected_count = len(trajectory_ids) * len(branch_ids) * 2
    all_files_bound = len(checkpoint_rows) == expected_count
    if not all_files_bound:
        raise ValueError("global checkpoint inventory is incomplete")
    payload = {
        "schema_version": SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": code_commit,
        "analysis_sources": analysis_sources,
        "source_run": {
            "locator": {"anchor": "data_root", "path": run_path},
            "manifest": anchored_descriptor(manifest_path, data_root)
            | {"record_sha256": manifest["record_sha256"]},
            "code_commit": manifest["code_commit"],
            "schema_version": manifest["schema_version"],
        },
        "trajectory_ids": trajectory_ids,
        "branch_ids": branch_ids,
        "expected_steps_per_branch": [0, int(manifest["final_step"])],
        "producer_metadata": metadata_rows,
        "checkpoints": checkpoint_rows,
        "summary": {
            "expected_checkpoint_count": expected_count,
            "checkpoint_count": len(checkpoint_rows),
            "total_size_bytes": sum(row["size_bytes"] for row in checkpoint_rows),
            "all_emitted_checkpoint_files_bound": all_files_bound,
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
    record = inventory_run(
        Path(arguments.data_root),
        run_path.as_posix(),
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
                "checkpoint_count": record["summary"]["checkpoint_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
