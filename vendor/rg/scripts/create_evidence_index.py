#!/usr/bin/env python3
"""Create a source-bound index of manuscript evidence and diagnostic records."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


SCHEMA = "pldr-row-rg-evidence-index-v2"
SPEC_SCHEMA = "pldr-row-rg-evidence-index-spec-v2"
STATUSES = {"current", "retained", "diagnostic", "unavailable-dependency"}
INVENTORY_CLASSIFICATIONS = {
    "current-evidence",
    "retained-evidence",
    "diagnostic",
    "failed-launch",
    "superseded-artifact",
    "excluded",
}
COVERAGE_MODES = {
    "representative-file-digest",
    "recursive-tree-census",
    "internal-sealed-chain-verification",
}
BASE_SOURCE_PATHS = (
    "scripts/create_evidence_index.py",
    "scripts/create_public_evidence_bundle.py",
    "scripts/analyze_staged_predictive_closure.py",
    "scripts/check_manuscript_release.py",
    "scripts/render_confirmation_tex.py",
    "scripts/render_evidence_lineage_tex.py",
    "scripts/render_observer_regime_tex.py",
    "scripts/render_staged_predictive_closure_tex.py",
    "scripts/verify_staged_predictive_closure_relocation.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
    "src/row_rgmap/staged_validation.py",
)


def committed_identity(
    specification_path: Path,
) -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        specification_source = specification_path.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise ValueError("evidence specification must be inside the repository") from error
    sources = []
    for repository_path in (*BASE_SOURCE_PATHS, specification_source):
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        worktree_path = ROOT / repository_path
        if (
            not worktree_path.is_file()
            or file_sha256(worktree_path) != digest
            or worktree_path.stat().st_size != size
        ):
            raise RuntimeError(
                f"index source differs from commit {commit}: {repository_path}"
            )
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def resolve_member(data_root: Path, value: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError("evidence-index paths must be nonempty and relative")
    path = (data_root / value).resolve()
    try:
        path.relative_to(data_root)
    except ValueError as error:
        raise ValueError("evidence-index path escapes the data root") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def record_identity(path: Path) -> str | None:
    if path.suffix != ".json":
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if "record_sha256" not in value:
        return None
    verify_record_seal(value)
    return value["record_sha256"]


def validate_support(
    support: dict[str, Any], entries: dict[str, dict[str, Any]]
) -> None:
    scalar = (
        "qualification",
        "observer_record",
        "stage_record",
        "excluded_disposition",
        "retained_record",
        "pilot_result",
        "stage_predecessor",
        "closure_record",
        "closure_relocation_record",
        "staged_closure_manifest",
        "staged_closure_record",
        "staged_closure_relocation_record",
        "closure_timing_record",
        "closure_checkpoint_inventory",
        "closure_checkpoint_inventory_relocation",
        "failed_smoke_root",
        "failed_smoke_protocol",
        "failed_smoke_events",
        "failed_smoke_attempt_0",
        "failed_smoke_attempt_1",
        "corrected_smoke_root",
        "corrected_smoke_result",
        "final_smoke_root",
        "final_smoke_result",
        "final_smoke_verification",
        "closure_source_smoke_root",
        "closure_source_smoke_result",
        "closure_source_smoke_verification",
        "formal_build_boundary_record",
    )
    vector = (
        "qualification_history",
        "planning_records",
        "closure_source_smoke_entries",
    )
    if set(support) != set(scalar).union(vector):
        raise ValueError("evidence-index support registry has unexpected keys")
    for key in scalar:
        if support[key] not in entries:
            raise ValueError(f"unknown support entry: {key}")
    for key in vector:
        values = support[key]
        if (
            not isinstance(values, list)
            or not values
            or len(set(values)) != len(values)
            or any(value not in entries for value in values)
        ):
            raise ValueError(f"invalid support entry list: {key}")


def deterministic_tree_census(path: Path) -> dict[str, Any]:
    """Content-address every regular file below a declared artifact child."""

    if not path.is_dir():
        raise ValueError("tree census requires a directory")
    rows = []
    for member in sorted(path.rglob("*")):
        if member.is_symlink():
            raise ValueError(f"tree census refuses symlink: {member}")
        if member.is_file():
            rows.append(
                {
                    "path": member.relative_to(path).as_posix(),
                    "sha256": file_sha256(member),
                    "size_bytes": member.stat().st_size,
                }
            )
    payload = json.dumps(
        rows, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {
        "file_count": len(rows),
        "total_size_bytes": sum(row["size_bytes"] for row in rows),
        "tree_sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_executed_inventory(
    executed_root: Path,
    rows: Any,
    entries: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Require exact classification of every immediate child of executed/."""

    if not executed_root.is_dir() or not isinstance(rows, list) or not rows:
        raise ValueError("executed inventory specification is missing")
    actual = {path.name: path for path in executed_root.iterdir()}
    declared: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("invalid executed inventory row")
        name = row.get("name")
        classification = row.get("classification")
        role = row.get("role")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or name in declared
            or classification not in INVENTORY_CLASSIFICATIONS
            or not isinstance(role, str)
            or not role
        ):
            raise ValueError("invalid executed inventory row")
        reason = row.get("reason")
        if classification in {"failed-launch", "superseded-artifact", "excluded"}:
            if not isinstance(reason, str) or not reason:
                raise ValueError(f"executed inventory reason is required: {name}")
        elif reason is not None and (not isinstance(reason, str) or not reason):
            raise ValueError(f"invalid executed inventory reason: {name}")
        declared[name] = row
    if set(actual) != set(declared):
        missing = sorted(set(declared) - set(actual))
        unclassified = sorted(set(actual) - set(declared))
        raise ValueError(
            "executed inventory differs from filesystem: "
            f"missing={missing}, unclassified={unclassified}"
        )

    inventory = []
    for name in sorted(declared):
        row = declared[name]
        path = actual[name]
        representatives = row.get("representative_entry_ids", [])
        if (
            not isinstance(representatives, list)
            or len(set(representatives)) != len(representatives)
            or any(identifier not in entries for identifier in representatives)
        ):
            raise ValueError(f"invalid inventory representatives: {name}")
        prefix = f"executed/{name}"
        if any(
            entries[identifier]["path"] != prefix
            and not entries[identifier]["path"].startswith(prefix + "/")
            for identifier in representatives
        ):
            raise ValueError(f"inventory representative escapes child: {name}")
        if row["classification"] == "failed-launch" and not representatives:
            raise ValueError("failed launch requires representative indexed records")
        sealed_chain = row.get("sealed_chain_verification", False)
        if not isinstance(sealed_chain, bool):
            raise ValueError(f"invalid sealed-chain coverage flag: {name}")
        coverage_modes = []
        if representatives:
            coverage_modes.append("representative-file-digest")
        if row.get("tree_census") is True:
            coverage_modes.append("recursive-tree-census")
        if sealed_chain:
            coverage_modes.append("internal-sealed-chain-verification")
        if not set(coverage_modes).issubset(COVERAGE_MODES):
            raise ValueError(f"invalid verification coverage mode: {name}")

        item = {
            "name": name,
            "kind": "directory" if path.is_dir() else "file",
            "classification": row["classification"],
            "role": row["role"],
            "representative_entry_ids": representatives,
            "verification_coverage_modes": coverage_modes,
        }
        if "reason" in row:
            item["reason"] = row["reason"]
        if row.get("tree_census") is True:
            item["tree_census"] = deterministic_tree_census(path)
        elif row.get("tree_census") not in (None, False):
            raise ValueError(f"invalid tree-census flag: {name}")
        inventory.append(item)
    return inventory


def build_index(
    data_root: Path,
    specification_path: Path,
    *,
    code_commit: str,
    analysis_sources: list[dict[str, Any]],
) -> dict[str, Any]:
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    if specification.get("schema_version") != SPEC_SCHEMA:
        raise ValueError("unknown evidence-index specification schema")
    rows = specification.get("entries")
    if not isinstance(rows, list) or not rows:
        raise ValueError("evidence-index specification is empty")

    entries: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    available_digests = {
        file_sha256(path)
        for path in (data_root / "protocol-v3").glob("*.json")
        if path.is_file()
    }
    for row in rows:
        identifier = row.get("id")
        status = row.get("status")
        role = row.get("role")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in by_id
            or status not in STATUSES
            or not isinstance(role, str)
            or not role
        ):
            raise ValueError("invalid evidence-index entry")
        path = resolve_member(data_root, row.get("path"))
        entry: dict[str, Any] = {
            "id": identifier,
            "status": status,
            "role": role,
            "path": str(path.relative_to(data_root)),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        canonical = record_identity(path)
        if canonical is not None:
            entry["record_sha256"] = canonical
        unavailable = row.get("unavailable_dependency_sha256")
        if status == "unavailable-dependency":
            if (
                not isinstance(unavailable, str)
                or len(unavailable) != 64
                or unavailable in available_digests
            ):
                raise ValueError(
                    "unavailable dependency is malformed or unexpectedly present"
                )
            entry["unavailable_dependency_sha256"] = unavailable
        elif unavailable is not None:
            raise ValueError("only unavailable-dependency rows may name a dependency")
        entries.append(entry)
        by_id[identifier] = entry

    support = specification.get("support_inputs")
    if not isinstance(support, dict):
        raise ValueError("evidence-index support registry is missing")
    validate_support(support, by_id)
    inventory = build_executed_inventory(
        data_root / "executed",
        specification.get("executed_inventory"),
        by_id,
    )
    return seal_record(
        {
            "schema_version": SCHEMA,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": code_commit,
            "analysis_sources": analysis_sources,
            "specification": {
                "repository_path": str(specification_path.relative_to(ROOT)),
                "sha256": file_sha256(specification_path),
                "size_bytes": specification_path.stat().st_size,
            },
            "data_root_contract": "paths are relative to the authorized experiment root",
            "entries": entries,
            "executed_inventory": inventory,
            "support_inputs": support,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--specification", default=str(ROOT / "configs" / "evidence-index-v11.json")
    )
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    data_root = Path(arguments.data_root).resolve()
    specification = Path(arguments.specification).resolve()
    commit, sources = committed_identity(specification)
    record = build_index(
        data_root,
        specification,
        code_commit=commit,
        analysis_sources=sources,
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "entry_count": len(record["entries"]),
                "record_sha256": record["record_sha256"],
                "status": "complete",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
