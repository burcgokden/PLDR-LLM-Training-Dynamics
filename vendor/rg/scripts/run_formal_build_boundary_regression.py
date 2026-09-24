#!/usr/bin/env python3
"""Exercise the fresh-build boundary and reject a compiling stale type fingerprint."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, seal_record  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


SCHEMA = "pldr-row-rg-formal-build-boundary-regression-v1"
SOURCE_PATHS = (
    "scripts/run_formal_build_boundary_regression.py",
    "scripts/check_formal_soundness.py",
    "scripts/render_formal_correspondence.py",
    "RowRGMap/GaugeFiber.lean",
    "formal-correspondence.json",
)


def repository_identity() -> tuple[str, list[dict[str, Any]]]:
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
        raise RuntimeError("formal boundary regression requires a clean repository")
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


def run_audit(manuscript_root: Path, lake: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "scripts/check_formal_soundness.py",
            "--manuscript-root",
            str(manuscript_root),
            "--lake",
            lake,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def output_descriptor(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    stdout = completed.stdout.encode("utf-8")
    stderr = completed.stderr.encode("utf-8")
    return {
        "returncode": completed.returncode,
        "stdout_sha256": hashlib.sha256(stdout).hexdigest(),
        "stderr_sha256": hashlib.sha256(stderr).hexdigest(),
        "stdout_size_bytes": len(stdout),
        "stderr_size_bytes": len(stderr),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript-root", required=True)
    parser.add_argument("--lake", default="lake")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    manuscript_root = Path(arguments.manuscript_root).resolve()
    commit, sources = repository_identity()

    fresh = run_audit(manuscript_root, arguments.lake)
    fresh_passed = fresh.returncode == 0
    if not fresh_passed:
        raise RuntimeError("fresh formal audit failed: " + (fresh.stderr or fresh.stdout)[-2000:])

    source_path = ROOT / "RowRGMap" / "GaugeFiber.lean"
    original = source_path.read_text(encoding="utf-8")
    old_binder = (
        "    (henergy : ∀ x, energy (consistent x) = energy x)\n"
        "    (b : ℕ) (x : X) :\n"
    )
    new_binder = (
        "    (henergy : ∀ x, energy (consistent x) = energy x)\n"
        "    (typeProbe : True) (b : ℕ) (x : X) :\n"
    )
    if original.count(old_binder) != 1:
        raise RuntimeError("formal type-mutation anchor is not unique")
    mutated = original.replace(old_binder, new_binder)


    stale = None
    restored = None
    try:
        source_path.write_text(mutated, encoding="utf-8")
        stale = run_audit(manuscript_root, arguments.lake)
    finally:
        source_path.write_text(original, encoding="utf-8")
        restored = run_audit(manuscript_root, arguments.lake)
    assert stale is not None and restored is not None
    stale_output = stale.stdout + stale.stderr
    stale_type_digest_rejected = (
        stale.returncode != 0
        and "Lean declaration-type digest mismatch" in stale_output
    )
    restoration_passed = restored.returncode == 0
    source_restored = file_sha256(source_path) == hashlib.sha256(
        original.encode("utf-8")
    ).hexdigest()
    repository_clean_after = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout == ""
    checks = {
        "fresh_build_and_audit_passed": fresh_passed,
        "compiling_type_mutation_rejected_by_digest": stale_type_digest_rejected,
        "restored_build_and_audit_passed": restoration_passed,
        "source_bytes_restored": source_restored,
        "repository_clean_after": repository_clean_after,
    }
    if not all(checks.values()):
        raise RuntimeError(f"formal build-boundary regression failed: {checks}")
    payload = {
        "schema_version": SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "analysis_sources": sources,
        "toolchain": (ROOT / "lean-toolchain").read_text(encoding="utf-8").strip(),
        "mutation": {
            "module": "RowRGMap.GaugeFiber",
            "declaration": "paired_gauge_successor_equal",
            "kind": "additional explicit True hypothesis",
            "source_mutation_compiled_before_fingerprint_check": stale_type_digest_rejected,
        },
        "fresh_audit": output_descriptor(fresh),
        "stale_type_audit": output_descriptor(stale),
        "restored_audit": output_descriptor(restored),
        "checks": checks,
    }
    record = seal_record(payload)
    output = Path(arguments.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": "complete", "record_sha256": record["record_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
