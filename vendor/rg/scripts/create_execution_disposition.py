#!/usr/bin/env python3
"""Create a separate content record for an integrity-incomplete execution."""

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

from row_rgmap.analysis import verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


SOURCE_PATHS = (
    "scripts/create_execution_disposition.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def descriptor(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        worktree_path = ROOT / repository_path
        if (
            not worktree_path.is_file()
            or file_sha256(worktree_path) != digest
            or worktree_path.stat().st_size != size
        ):
            raise RuntimeError(
                f"disposition source differs from commit {commit}: {repository_path}"
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


def scientific_payload(record: dict[str, Any]) -> dict[str, Any]:
    gate = record["failure_gate"]
    return {
        "run_id": record["run_id"],
        "disposition": record["disposition"],
        "scientific_flow_evidence": record["scientific_flow_evidence"],
        "allowed_use": record["allowed_use"],
        "failure_gate": {
            "quantity": gate["quantity"],
            "tolerance": gate["tolerance"],
        },
        "historical_event_log_rewritten": record[
            "historical_event_log_rewritten"
        ],
        "contemporaneous_failure_event_present": record[
            "contemporaneous_failure_event_present"
        ],
        "event_names": record["event_names"],
        "trajectories_with_saved_state": record[
            "trajectories_with_saved_state"
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--predecessor")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    root = Path(arguments.run_root).resolve()
    config_path = root / "config-resolved.json"
    protocol_path = root / "protocol-resolved.json"
    qualification_path = root / "qualification-resolved.json"
    root_record_path = root / "run-root.json"
    events_path = root / "events.jsonl"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    qualification = json.loads(qualification_path.read_text(encoding="utf-8"))
    verify_record_seal(qualification)
    root_record = json.loads(root_record_path.read_text(encoding="utf-8"))
    verify_record_seal(root_record)
    events = [
        json.loads(line)
        for line in events_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    trajectories = []
    for trajectory in config["trajectories"]:
        trajectory_id = trajectory["trajectory_id"]
        directory = root / "trajectories" / trajectory_id
        logs = sorted(directory.glob("producer-attempt-*.log"))
        checkpoints = sorted((directory / "checkpoints").glob("checkpoint-step*.pt"))
        if not logs and not checkpoints:
            continue
        final_step = max(
            (
                int(path.stem.removeprefix("checkpoint-step"))
                for path in checkpoints
            ),
            default=None,
        )
        trajectories.append(
            {
                "trajectory_id": trajectory_id,
                "seed": trajectory["seed"],
                "physical_device": trajectory["device"],
                "final_saved_step": final_step,
                "trained_full_registered_horizon": final_step
                == int(protocol["expected_updates"]),
                "attempt_logs": [descriptor(path, root) for path in logs],
                "checkpoints": [descriptor(path, root) for path in checkpoints],
                "final_artifacts_present": all(
                    (directory / name).is_file()
                    for name in ("energies.npz", "metrics.npz", "metadata.json")
                ),
            }
        )
    gate_keys = [
        key
        for key in protocol["identity_tolerances"]
        if key.startswith("float32_gate_shape")
    ]
    if len(gate_keys) != 1:
        raise ValueError("historical protocol has no unique float32 gate-shape key")
    gate_key = gate_keys[0]
    commit, sources = committed_identity()
    failures = []
    for trajectory in trajectories:
        for attempt in trajectory["attempt_logs"]:
            log_path = root / attempt["path"]
            failures.append(
                {
                    "trajectory_id": trajectory["trajectory_id"],
                    "time_utc": datetime.fromtimestamp(
                        log_path.stat().st_mtime, timezone.utc
                    ).isoformat(),
                    "attempt_log_sha256": attempt["sha256"],
                }
            )
    record: dict[str, Any] = {
        "schema_version": "pldr-row-rg-execution-disposition-v5",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "code_commit": commit,
        "analysis_sources": sources,
        "run_id": config["run_id"],
        "disposition": "INTEGRITY_GATE_FAILED",
        "scientific_flow_evidence": False,
        "allowed_use": "saved-checkpoint numerical sensitivity control",
        "failure_gate": {
            "protocol_key": gate_key,
            "quantity": "quotient-energy relative LayerNorm factorization residual",
            "tolerance": float(protocol["identity_tolerances"][gate_key]),
        },
        "historical_event_log_rewritten": False,
        "contemporaneous_failure_event_present": any(
            row.get("event") == "PRODUCER_FAILED" for row in events
        ),
        "absence_explanation": (
            "The runner version used for this execution raised after the producer "
            "returned nonzero but did not yet implement a PRODUCER_FAILED event."
        ),
        "inputs": {
            "run_root": descriptor(root_record_path, root),
            "config": descriptor(config_path, root),
            "protocol": descriptor(protocol_path, root),
            "qualification": descriptor(qualification_path, root),
            "event_log": descriptor(events_path, root),
        },
        "launch_qualification_record_sha256": qualification["record_sha256"],
        "prelaunch_smoke_record_present": any(
            path.is_file() and "smoke" in path.name.lower()
            for path in root.rglob("*")
        ),
        "event_names": [row.get("event") for row in events],
        "failure_times": failures,
        "trajectories_with_saved_state": trajectories,
    }
    if arguments.predecessor:
        predecessor_path = Path(arguments.predecessor).resolve()
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        verify_record_seal(predecessor)
        record["predecessor"] = {
            "record_sha256": predecessor["record_sha256"],
            "file_sha256": file_sha256(predecessor_path),
            "size_bytes": predecessor_path.stat().st_size,
            "scientific_leaves_equal": scientific_payload(predecessor)
            == scientific_payload(record),
        }
        if not record["predecessor"]["scientific_leaves_equal"]:
            raise ValueError("disposition scientific leaves differ from predecessor")
    record["record_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": record["record_sha256"],
                "saved_trajectory_count": len(trajectories),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
