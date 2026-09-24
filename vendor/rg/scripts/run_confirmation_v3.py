#!/usr/bin/env python3
"""Run four registered trajectories with immutable inputs and append-only analyses.

The production state machine is deliberately small:

``ABSENT``
    A mode-free invocation creates the root and enters ``ACTIVE``. Resume
    and analyze-only are invalid.
``ACTIVE``
    Producer failure or interruption leaves an event-chained partial root.
    Only ``--resume`` may launch missing producers. ``--analyze-only`` is
    valid only after every final producer artifact exists. An incomplete
    historical root must be resumed from its producer commit. These two
    refusals raise :class:`ActiveRootContinuationError` without mutation.
``FINALIZED_VALID``
    A root manifest exists and independent verification succeeds. Resume is
    a verified no-op. Analyze-only may append a new analysis and a new
    manifest without changing producer artifacts or earlier lineage.
``FINALIZED_DAMAGED``
    A root manifest exists but preflight verification fails. Both
    continuation modes raise :class:`FinalizedRootMutationError` before any
    producer launch; reproduction must use a new output root.

The event alphabet distinguishes producer completion, controlled failure,
and interruption. Every event has a sequence number, parent digest, and
content digest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import analyze_energy_npz as energy_io  # noqa: E402
import row_rgmap.analysis as analysis_module  # noqa: E402
import row_rgmap.confirmation_v3 as confirmation_module  # noqa: E402
import row_rgmap.diagnostics_v3 as diagnostics_module  # noqa: E402
import row_rgmap.provenance_v3 as provenance_module  # noqa: E402
import verify_registered_artifacts as registered_verifier  # noqa: E402


CONFIG_SCHEMA = "pldr-row-rg-four-trajectory-run-v3"
PROTOCOL_SCHEMA = "pldr-row-rg-confirmation-protocol-v3"
RESULT_SCHEMA = "pldr-row-rg-confirmation-result-v5"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class FinalizedRootMutationError(RuntimeError):
    """A finalized run is incomplete or differs from its immutable manifest."""


class ActiveRootContinuationError(RuntimeError):
    """An ACTIVE root cannot be continued in the requested mode or checkout."""


def _finalized_root_error(error: BaseException) -> FinalizedRootMutationError:
    detail = str(error).strip() or error.__class__.__name__
    return FinalizedRootMutationError(
        "finalized run root is damaged and immutable: "
        f"{detail}; use a new output root to reproduce the run"
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(analysis_module.canonical_json_bytes(value)).hexdigest()


def _write_new_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _git_identity() -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    if status:
        raise RuntimeError("registered execution requires a clean committed repository")
    return commit, status


def _resolve(path: str, parent: Path) -> Path:
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (parent / candidate).resolve()


def _source_digest_at_commit(path: Path, commit: str) -> str:
    """Hash repository sources from the recorded commit, not the worktree."""

    relative = provenance_module.repository_relative_source(str(path), ROOT)
    if relative is None:
        if not path.is_file():
            raise FileNotFoundError(path)
        return analysis_module.file_sha256(path)
    result = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )
    return hashlib.sha256(result.stdout).hexdigest()


def _verify_bound_producer(
    config: dict[str, Any], commit: str, *, require_worktree: bool
) -> None:
    """Verify an optional repository-relative producer identity before launch."""

    descriptor = config.get("producer_source")
    if descriptor is None:
        return
    if not isinstance(descriptor, dict) or set(descriptor) != {
        "repository_path", "code_commit", "sha256", "size_bytes"
    }:
        raise ValueError("producer_source must be a complete committed blob descriptor")
    repository_path = descriptor.get("repository_path")
    if (
        not isinstance(repository_path, str)
        or not repository_path
        or Path(repository_path).is_absolute()
        or ".." in Path(repository_path).parts
    ):
        raise ValueError("bound producer path must be repository-relative")
    if descriptor.get("code_commit") != commit:
        raise ValueError("bound producer commit differs from the run producer commit")
    expected_digest, expected_size = provenance_module.git_blob_descriptor(
        ROOT, commit, repository_path
    )
    if (
        descriptor.get("sha256") != expected_digest
        or descriptor.get("size_bytes") != expected_size
    ):
        raise ValueError("bound producer descriptor differs from the committed blob")
    if repository_path not in config["producer_command"]:
        raise ValueError("producer command does not invoke the bound repository path")
    if require_worktree:
        worktree_path = ROOT / repository_path
        if (
            not worktree_path.is_file()
            or analysis_module.file_sha256(worktree_path) != expected_digest
            or worktree_path.stat().st_size != expected_size
        ):
            raise ValueError("executed producer file differs from its committed blob")


def _termination_metadata(return_code: int) -> tuple[str, dict[str, Any]]:
    """Classify an observed process status without inferring its external cause.

    A negative subprocess return code directly records signal termination.
    A positive status 128 + n follows a shell convention and can also be
    produced by an ordinary process exit, so causal attribution remains
    explicitly undetermined in both encodings.
    """

    signal_number = None
    encoding = None
    if return_code < 0:
        signal_number = -return_code
        encoding = "subprocess-negative-signal"
    elif 128 < return_code <= 255:
        candidate = return_code - 128
        try:
            signal.Signals(candidate)
        except ValueError:
            pass
        else:
            signal_number = candidate
            encoding = "conventional-128-plus-signal"
    if signal_number is not None:
        try:
            signal_name = signal.Signals(signal_number).name
        except ValueError:
            signal_name = f"SIGNAL_{signal_number}"
        return (
            "PRODUCER_INTERRUPTED",
            {
                "classification": "EXTERNAL_TERMINATION",
                "encoding": encoding,
                "signal_number": signal_number,
                "signal_name": signal_name,
                "cause_attribution": "UNDETERMINED",
            },
        )
    return (
        "PRODUCER_FAILED",
        {
            "classification": "PROCESS_EXIT",
            "exit_status": return_code,
        },
    )


def _metadata_path(
    raw: str, metadata_path: Path, fallback_directory: Path | None = None
) -> Path:
    candidate = Path(raw)
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (metadata_path.parent / candidate).resolve()
    )
    if fallback_directory is not None:
        fallback = (fallback_directory / candidate.name).resolve()
        try:
            resolved.relative_to(fallback_directory.resolve())
        except ValueError:
            if fallback.is_file():
                return fallback
    return resolved


def _load_config(config_path: Path) -> tuple[dict, dict, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unknown v3 run configuration schema")
    if not isinstance(config.get("run_id"), str) or _SAFE_ID.fullmatch(config["run_id"]) is None:
        raise ValueError("run identifier is missing or unsafe")
    devices = config.get("devices")
    if devices != [0, 1]:
        raise ValueError("the campaign requires physical devices [0, 1]")
    trajectories = config.get("trajectories")
    if not isinstance(trajectories, list) or len(trajectories) != 4:
        raise ValueError("the registered campaign requires four trajectories")
    identifiers = [row.get("trajectory_id") for row in trajectories]
    seeds = [row.get("seed") for row in trajectories]
    offsets = [row.get("train_document_offset") for row in trajectories]
    if (
        any(not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None for value in identifiers)
        or len(set(identifiers)) != 4
        or any(not isinstance(value, int) or isinstance(value, bool) for value in seeds)
        or len(set(seeds)) != 4
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in offsets)
        or len(set(offsets)) != 4
    ):
        raise ValueError("trajectory identifiers, seeds, or document offsets are invalid")
    for wave in (0, 1):
        rows = [row for row in trajectories if row.get("wave") == wave]
        if sorted(row.get("device") for row in rows) != devices:
            raise ValueError("each wave must bind one trajectory to each physical device")
    protocol_path = _resolve(config.get("protocol_path", ""), config_path.parent)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unknown v3 protocol schema")
    if protocol.get("expected_updates") != sum(protocol["segments"].values()):
        raise ValueError("protocol segments do not sum to expected_updates")
    if protocol.get("randomness") is not None:
        seed_registry = provenance_module.validate_seed_registry(protocol)
        for trajectory in trajectories:
            domain = trajectory.get("seed_domain")
            if domain not in seed_registry or trajectory.get("seed") != seed_registry[domain]:
                raise ValueError("trajectory seed does not match its protocol domain")
        gaussian_seed = protocol.get("analysis", {}).get(
            "gaussian_fixed_law", {}
        ).get("seed")
        if gaussian_seed != seed_registry.get("analysis:gaussian-fixed-law"):
            raise ValueError("Gaussian calibration seed is outside the protocol registry")
    command = config.get("producer_command")
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("producer_command must be a nonempty argument vector")
    required = {
        "{device}", "{trajectory_id}", "{seed}", "{train_document_offset}",
        "{output}", "{metrics}", "{metadata}", "{checkpoint_dir}",
        "{protocol}", "{code_commit}",
    }
    joined = "\n".join(command)
    if any(value not in joined for value in required):
        raise ValueError("producer command lacks a registered placeholder")
    return config, protocol, protocol_path


def _event_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    previous = None
    for index, row in enumerate(rows):
        observed = row.get("event_sha256")
        unsigned = dict(row)
        unsigned.pop("event_sha256", None)
        if row.get("sequence") != index or row.get("previous_event_sha256") != previous:
            raise ValueError("event log sequence or parent digest is invalid")
        if observed != _canonical_digest(unsigned):
            raise ValueError("event log integrity digest does not replay")
        previous = observed
    return rows


def _append_event(path: Path, event: str, payload: dict[str, Any]) -> dict[str, Any]:
    rows = _event_rows(path)
    row = {
        "sequence": len(rows),
        "time_utc": _utc_now(),
        "event": event,
        "payload": payload,
        "previous_event_sha256": rows[-1]["event_sha256"] if rows else None,
    }
    row["event_sha256"] = _canonical_digest(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    return row


def _root_identity(
    config_path: Path,
    config: dict,
    protocol_path: Path,
    protocol: dict,
    commit: str,
) -> dict[str, Any]:
    return analysis_module.seal_record(
        {
            "schema_version": "pldr-row-rg-run-root-v3",
            "run_id": config["run_id"],
            "created_at_utc": _utc_now(),
            "code_commit": commit,
            "config": {
                "source_path": str(config_path),
                "source_file_sha256": analysis_module.file_sha256(config_path),
                "canonical_sha256": _canonical_digest(config),
            },
            "protocol": {
                "source_path": str(protocol_path),
                "source_file_sha256": analysis_module.file_sha256(protocol_path),
                "canonical_sha256": _canonical_digest(protocol),
            },
        }
    )


def _initialize_root(
    output_root: Path,
    config_path: Path,
    config: dict,
    protocol_path: Path,
    protocol: dict,
    commit: str,
) -> dict[str, Any]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"run root is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    identity = _root_identity(config_path, config, protocol_path, protocol, commit)
    _write_new_json(output_root / "run-root.json", identity)
    _write_new_json(output_root / "config-resolved.json", config)
    _write_new_json(output_root / "protocol-resolved.json", protocol)
    _append_event(
        output_root / "events.jsonl",
        "RUN_INITIALIZED",
        {
            "run_id": config["run_id"],
            "root_record_sha256": identity["record_sha256"],
            "code_commit": commit,
        },
    )
    return identity


def _validate_root(
    output_root: Path, config: dict, protocol: dict, commit: str
) -> dict[str, Any]:
    identity_path = output_root / "run-root.json"
    if not identity_path.is_file():
        raise ValueError("existing run has no immutable root identity")
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    analysis_module.verify_record_seal(identity)
    if (
        identity.get("run_id") != config["run_id"]
        or identity.get("config", {}).get("canonical_sha256") != _canonical_digest(config)
        or identity.get("protocol", {}).get("canonical_sha256") != _canonical_digest(protocol)
    ):
        raise ValueError("run root belongs to a different commit, configuration, or protocol")
    if json.loads((output_root / "config-resolved.json").read_text(encoding="utf-8")) != config:
        raise ValueError("resolved configuration was mutated")
    if json.loads((output_root / "protocol-resolved.json").read_text(encoding="utf-8")) != protocol:
        raise ValueError("resolved protocol was mutated")
    _event_rows(output_root / "events.jsonl")
    return identity


def _trajectory_paths(root: Path, trajectory_id: str) -> dict[str, Path]:
    directory = root / "trajectories" / trajectory_id
    return {
        "directory": directory,
        "energies": directory / "energies.npz",
        "metrics": directory / "metrics.npz",
        "metadata": directory / "metadata.json",
        "checkpoints": directory / "checkpoints",
    }


def _missing_final_artifacts(config: dict, output_root: Path) -> list[str]:
    missing = []
    for trajectory in config["trajectories"]:
        paths = _trajectory_paths(output_root, trajectory["trajectory_id"])
        for name in ("energies", "metrics", "metadata"):
            if not paths[name].is_file():
                missing.append(
                    str(paths[name].relative_to(output_root))
                )
    return missing


def _attempt_path(directory: Path) -> Path:
    existing = sorted(directory.glob("producer-attempt-*.log"))
    return directory / f"producer-attempt-{len(existing) + 1:03d}.log"


def _latest_checkpoint(directory: Path, final_step: int) -> Path | None:
    candidates = sorted(directory.glob("checkpoint-step*.pt"))
    eligible = []
    for path in candidates:
        match = re.fullmatch(r"checkpoint-step([0-9]{6})[.]pt", path.name)
        if match and int(match.group(1)) < final_step:
            eligible.append(path)
    return eligible[-1] if eligible else None


def _format_command(
    template: list[str],
    trajectory: dict[str, Any],
    paths: dict[str, Path],
    protocol_path: Path,
    commit: str,
    resume_from: Path | None,
) -> list[str]:
    values = {
        "device": str(trajectory["device"]),
        "trajectory_id": trajectory["trajectory_id"],
        "seed": str(trajectory["seed"]),
        "train_document_offset": str(trajectory["train_document_offset"]),
        "output": str(paths["energies"]),
        "metrics": str(paths["metrics"]),
        "metadata": str(paths["metadata"]),
        "checkpoint_dir": str(paths["checkpoints"]),
        "protocol": str(protocol_path),
        "code_commit": commit,
    }
    try:
        command = [value.format_map(values) for value in template]
    except KeyError as error:
        raise ValueError(f"unknown producer placeholder: {error.args[0]}") from error
    if resume_from is not None:
        command.extend(["--resume-from", str(resume_from)])
    return command


def _validate_file_descriptor(path: Path, descriptor: dict[str, Any], label: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    if descriptor.get("sha256") != analysis_module.file_sha256(path):
        raise ValueError(f"{label} digest mismatch")
    if descriptor.get("size_bytes") != path.stat().st_size:
        raise ValueError(f"{label} size mismatch")


def _load_metrics(path: Path, expected_updates: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != {"loss_steps", "losses"}:
            raise ValueError("metrics artifact has an unknown schema")
        steps = np.asarray(payload["loss_steps"])
        losses = np.asarray(payload["losses"])
    if (
        steps.dtype.kind not in "iu"
        or losses.dtype != np.dtype(np.float64)
        or steps.tolist() != list(range(1, expected_updates + 1))
        or losses.shape != (expected_updates,)
        or not np.isfinite(losses).all()
    ):
        raise ValueError("metrics artifact violates the registered contract")
    return steps, losses


def _validate_metadata(
    trajectory: dict[str, Any],
    paths: dict[str, Path],
    protocol: dict,
    commit: str,
) -> tuple[dict[str, Any], tuple[Any, ...], tuple[np.ndarray, np.ndarray]]:
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "pldr-row-rg-energy-producer-metadata-v3":
        raise ValueError("producer metadata has an unknown schema")
    if (
        metadata.get("trajectory_id") != trajectory["trajectory_id"]
        or metadata.get("physical_device") != trajectory["device"]
        or metadata.get("cuda_visible_devices") != str(trajectory["device"])
        or metadata.get("visible_gpu_count") != 1
        or metadata.get("seed") != trajectory["seed"]
        or metadata.get("code_commit") != commit
        or metadata.get("protocol", {}).get("canonical_sha256") != _canonical_digest(protocol)
        or metadata.get("data", {}).get("training", {}).get("first_document")
        != trajectory["train_document_offset"]
    ):
        raise ValueError("producer metadata identity differs from the run plan")
    sources = metadata.get("sources", {})
    for prefix in ("producer", "reference_model", "reference_attention", "recorder"):
        raw_path = sources.get(f"{prefix}_path", "")
        candidate = Path(raw_path)
        path = (
            (ROOT / candidate).resolve()
            if not candidate.is_absolute() and prefix in ("producer", "recorder")
            else _metadata_path(raw_path, paths["metadata"])
        )
        if sources.get(f"{prefix}_sha256") != _source_digest_at_commit(path, commit):
            raise ValueError(f"imported producer source mismatch: {prefix}")
    loaded = energy_io.load(paths["energies"])
    energies, steps, trajectory_ids, map_ids = loaded
    expected = int(protocol["expected_updates"])
    if (
        steps.tolist() != list(range(expected + 1))
        or trajectory_ids.tolist() != [trajectory["trajectory_id"]]
        or map_ids.tolist() != metadata.get("capture", {}).get("map_ids")
        or metadata.get("capture", {}).get("states") != expected + 1
    ):
        raise ValueError("energy artifact registry differs from producer metadata")
    _validate_file_descriptor(paths["energies"], metadata["artifacts"]["energies"], "energy artifact")
    metrics = _load_metrics(paths["metrics"], expected)
    _validate_file_descriptor(paths["metrics"], metadata["artifacts"]["metrics"], "metrics artifact")
    registered_checkpoints = [int(value) for value in protocol["plga_analysis"]["checkpoint_steps"]]
    if [row.get("step") for row in metadata.get("checkpoints", [])] != registered_checkpoints:
        raise ValueError("checkpoint registry differs from the protocol")
    for row in metadata["checkpoints"]:
        path = _metadata_path(
            row["path"], paths["metadata"], paths["checkpoints"]
        )
        try:
            path.resolve().relative_to(paths["checkpoints"].resolve())
        except ValueError as error:
            raise ValueError("checkpoint path escapes the trajectory directory") from error
        _validate_file_descriptor(path, row, f"checkpoint step {row['step']}")
    return metadata, loaded, metrics


def _producer_completed_payload(
    trajectory_id: str, paths: dict[str, Path]
) -> dict[str, Any]:
    return {
        "trajectory_id": trajectory_id,
        "metadata_sha256": analysis_module.file_sha256(paths["metadata"]),
        "energy_sha256": analysis_module.file_sha256(paths["energies"]),
        "metrics_sha256": analysis_module.file_sha256(paths["metrics"]),
    }



def _run_producers(
    config: dict,
    protocol: dict,
    protocol_path: Path,
    output_root: Path,
    commit: str,
) -> list[tuple[dict[str, Any], tuple[Any, ...], tuple[np.ndarray, np.ndarray], dict[str, Any]]]:
    validated: dict[str, tuple[Any, ...]] = {}
    timeout = int(config.get("producer_timeout_seconds", 14400))
    for wave in (0, 1):
        processes = []
        streams = []
        for trajectory in [row for row in config["trajectories"] if row["wave"] == wave]:
            paths = _trajectory_paths(output_root, trajectory["trajectory_id"])
            paths["directory"].mkdir(parents=True, exist_ok=True)
            complete = all(paths[name].is_file() for name in ("energies", "metrics", "metadata"))
            if complete:
                loaded = _validate_metadata(
                    trajectory, paths, protocol, commit
                )
                validated[trajectory["trajectory_id"]] = loaded
                expected_payload = _producer_completed_payload(
                    trajectory["trajectory_id"], paths
                )
                completion_rows = [
                    row for row in _event_rows(output_root / "events.jsonl")
                    if row.get("event") == "PRODUCER_COMPLETED"
                    and row.get("payload", {}).get("trajectory_id")
                    == trajectory["trajectory_id"]
                ]
                if not completion_rows:
                    _append_event(
                        output_root / "events.jsonl",
                        "PRODUCER_COMPLETED",
                        expected_payload,
                    )
                elif completion_rows[-1].get("payload") != expected_payload:
                    raise ValueError("completed producer event differs from final artifacts")
                continue
            if any(paths[name].exists() for name in ("energies", "metrics", "metadata")):
                raise ValueError("partial final producer artifacts cannot be reused")
            resume_from = _latest_checkpoint(paths["checkpoints"], int(protocol["expected_updates"]))
            command = _format_command(
                config["producer_command"], trajectory, paths, protocol_path, commit, resume_from
            )
            log_path = _attempt_path(paths["directory"])
            stream = log_path.open("xb")
            streams.append(stream)
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(trajectory["device"])
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            processes.append((trajectory, paths, command, log_path, process))
            _append_event(
                output_root / "events.jsonl",
                "PRODUCER_STARTED",
                {
                    "trajectory_id": trajectory["trajectory_id"],
                    "wave": wave,
                    "device": trajectory["device"],
                    "attempt_log": str(log_path.relative_to(output_root)),
                    "resume_checkpoint": (
                        str(resume_from.relative_to(output_root)) if resume_from else None
                    ),
                },
            )
        deadline = time.monotonic() + timeout
        failures = []
        try:
            for trajectory, paths, command, log_path, process in processes:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise subprocess.TimeoutExpired(command, timeout)
                return_code = process.wait(timeout=remaining)
                if return_code:
                    failures.append((trajectory["trajectory_id"], return_code))
            if failures:
                raise RuntimeError(f"producer failures: {failures}")
        except BaseException:
            for *_, process in processes:
                if process.poll() is None:
                    process.terminate()
            for *_, process in processes:
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            for (trajectory, paths, _, log_path, process), stream in zip(
                processes, streams
            ):
                stream.flush()
                os.fsync(stream.fileno())
                if process.returncode == 0:
                    _validate_metadata(trajectory, paths, protocol, commit)
                    _append_event(
                        output_root / "events.jsonl",
                        "PRODUCER_COMPLETED",
                        _producer_completed_payload(trajectory["trajectory_id"], paths),
                    )
                else:
                    return_code = process.returncode
                    event, termination = _termination_metadata(return_code)
                    _append_event(
                        output_root / "events.jsonl",
                        event,
                        {
                            "trajectory_id": trajectory["trajectory_id"],
                            "wave": wave,
                            "return_code": return_code,
                            "termination": termination,
                            "attempt_log": str(log_path.relative_to(output_root)),
                            "attempt_log_sha256": analysis_module.file_sha256(log_path),
                            "attempt_log_size_bytes": log_path.stat().st_size,
                        },
                    )
            raise
        finally:
            for stream in streams:
                stream.close()
        for trajectory, paths, command, log_path, _ in processes:
            loaded = _validate_metadata(trajectory, paths, protocol, commit)
            validated[trajectory["trajectory_id"]] = loaded
            _append_event(
                output_root / "events.jsonl",
                "PRODUCER_COMPLETED",
                _producer_completed_payload(trajectory["trajectory_id"], paths),
            )
    rows = []
    ranges = []
    for trajectory in config["trajectories"]:
        metadata, loaded, metrics = validated[trajectory["trajectory_id"]]
        training = metadata["data"]["training"]
        ranges.append((training["first_document"], training["last_document_exclusive"]))
        rows.append((metadata, loaded, metrics, trajectory))
    for index, (left, right) in enumerate(ranges):
        for other_left, other_right in ranges[index + 1 :]:
            if max(left, other_left) < min(right, other_right):
                raise ValueError("executed training document ranges overlap")
    return rows


def _numeric_summary(values: Any) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(array):
        return {
            "count": 0,
            "minimum": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _gain_summary(values: np.ndarray, block_size: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(values):
        return {
            **_numeric_summary(values),
            "zero_count": 0,
            "contracting_fraction": None,
            "marginal_fraction": None,
            "expanding_fraction": None,
            "conditional_positive_mean_log_gain_per_step": None,
        }
    positive = values[values > 0.0]
    result = _numeric_summary(values)
    result.update(
        {
            "zero_count": int(np.sum(values == 0.0)),
            "contracting_fraction": float(np.mean(values < 1.0)),
            "marginal_fraction": float(np.mean(values == 1.0)),
            "expanding_fraction": float(np.mean(values > 1.0)),
            "conditional_positive_mean_log_gain_per_step": (
                float(np.log(positive).mean() / block_size) if len(positive) else None
            ),
        }
    )
    return result


def _exact_block_census(energies: np.ndarray, block_sizes: list[int]) -> dict[str, Any]:
    array = np.asarray(energies, dtype=np.float64)
    trajectories, states, maps = array.shape
    face_mask = array == 0.0
    cumulative_faces = np.cumsum(face_mask, axis=1, dtype=np.int64)
    levels = []
    for block_size in block_sizes:
        starts = np.arange(0, states - block_size, block_size)
        stops = starts + block_size
        source = array[:, starts, :]
        target = array[:, stops, :]
        before = np.where(starts[None, :, None] > 0, cumulative_faces[:, starts - 1, :], 0)
        face_count = cumulative_faces[:, stops, :] - before
        touches = face_count > 0
        gains = np.zeros_like(source)
        face_free = ~touches
        gains[face_free] = target[face_free] / source[face_free]
        transported = np.where(touches, target, 0.0)
        predicted = gains * source + transported
        scale = np.maximum(np.maximum(np.abs(source), np.abs(target)), 1e-300)
        levels.append(
            {
                "block_size": block_size,
                "aligned_blocks_per_map": int(len(starts)),
                "total_blocks": int(trajectories * maps * len(starts)),
                "blocks_touching_exact_face": int(np.sum(touches)),
                "blocks_with_transported_source": int(np.sum(transported > 0.0)),
                "semigroup_replay_relative_residual": float(
                    np.max(np.abs(predicted - target) / scale)
                ),
                "face_free_gain": _gain_summary(gains[face_free], block_size),
                "transported_source": _numeric_summary(transported.reshape(-1)),
            }
        )
    return {
        "schema_version": "pldr-row-rg-energy-array-analysis-v3",
        "shape": {"trajectories": trajectories, "times": states, "maps": maps},
        "block_sizes": block_sizes,
        "fine_exact_face_state_count": int(np.sum(face_mask)),
        "fine_face_closure_count": int(
            np.sum((array[:, :-1, :] > 0.0) & (array[:, 1:, :] == 0.0))
        ),
        "fine_face_restart_count": int(
            np.sum((array[:, :-1, :] == 0.0) & (array[:, 1:, :] > 0.0))
        ),
        "levels": levels,
        "maximum_semigroup_replay_relative_residual": max(
            row["semigroup_replay_relative_residual"] for row in levels
        ),
    }


def _loss_analysis(rows: list[tuple[Any, ...]], protocol: dict) -> dict[str, Any]:
    window = int(protocol["analysis"]["window_length"])
    result = []
    for metadata, _, (_, losses), trajectory in rows:
        windows = losses[: len(losses) // window * window].reshape(-1, window).mean(axis=1)
        result.append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "initial_update_loss": float(losses[0]),
                "final_update_loss": float(losses[-1]),
                "minimum_loss": float(losses.min()),
                "median_loss": float(np.median(losses)),
                "window_length": window,
                "window_mean_losses": [float(value) for value in windows],
                "tokens_per_update": metadata["data"]["tokens_per_update"],
                "training_document_range": [
                    metadata["data"]["training"]["first_document"],
                    metadata["data"]["training"]["last_document_exclusive"],
                ],
            }
        )
    return {"trajectories": result}


def _plga_analysis(rows: list[tuple[Any, ...]], protocol: dict) -> dict[str, Any]:
    steps = [int(value) for value in protocol["plga_analysis"]["checkpoint_steps"]]
    checkpoint_rows = []
    for step in steps:
        diagnostics = []
        controls = []
        for metadata, *_ in rows:
            match = next(
                row["diagnostics"]
                for row in metadata["plga_diagnostics_by_checkpoint"]
                if row["step"] == step
            )
            diagnostics.extend(match["rows"])
            controls.append(match["constrained_head_controls"])
        defects = [row["constant_input_defect_norm"] for row in diagnostics]
        kappas = [row["sampled_operator_norm_lower_bound"] for row in diagnostics]
        secants = [row["realized_secant_gain"] for row in diagnostics]
        structural = {}
        for name in ("W_one", "bias", "power", "coupling_one", "coupling_bias"):
            structural[name] = {
                "absolute": _numeric_summary(
                    [row["structural_residuals"][name]["absolute"] for row in diagnostics]
                ),
                "relative": _numeric_summary(
                    [row["structural_residuals"][name]["relative"] for row in diagnostics]
                ),
            }
        checkpoint_rows.append(
            {
                "step": step,
                "map_count": len(diagnostics),
                "constant_input_defect_norm": _numeric_summary(defects),
                "sampled_operator_norm_lower_bound": _numeric_summary(kappas),
                "realized_secant_gain": _numeric_summary(secants),
                "structural_residuals": structural,
                "maximum_decomposition_relative_residual": max(
                    row["decomposition_relative_residual"] for row in diagnostics
                ),
                "constrained_control_maximum_defect": max(
                    row["maximum_constant_input_defect_norm"] for row in controls
                ),
                "direct_face_preservation_passed": max(defects)
                <= float(protocol["plga_analysis"]["direct_defect_tolerance"]),
            }
        )
    return {
        "schema_version": "pldr-plga-campaign-analysis-v3",
        "checkpoints": checkpoint_rows,
        "face_preservation_passed_at_every_checkpoint": all(
            row["direct_face_preservation_passed"] for row in checkpoint_rows
        ),
        "operator_quantity_is_grid_lower_bound": True,
        "decomposition_residual_is_implementation_invariant": True,
    }


def _analysis_sources(commit: str) -> list[dict[str, Any]]:
    paths = {
        Path(__file__).resolve(),
        Path(energy_io.__file__).resolve(),
        Path(analysis_module.__file__).resolve(),
        Path(confirmation_module.__file__).resolve(),
        Path(diagnostics_module.__file__).resolve(),
        Path(provenance_module.__file__).resolve(),
        (ROOT / "src/row_rgmap/core.py").resolve(),
        (ROOT / "src/row_rgmap/statistics.py").resolve(),
    }
    rows = []
    for path in sorted(paths):
        relative = path.relative_to(ROOT).as_posix()
        digest, size = provenance_module.git_blob_descriptor(ROOT, commit, relative)
        rows.append(
            {
                "repository_path": relative,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return rows


def _artifact_rows(output_root: Path, rows: list[tuple[Any, ...]]) -> list[dict[str, Any]]:
    result = []
    for metadata, _, _, trajectory in rows:
        paths = _trajectory_paths(output_root, trajectory["trajectory_id"])
        result.append(
            {
                "trajectory_id": trajectory["trajectory_id"],
                "device": trajectory["device"],
                "seed": trajectory["seed"],
                "energies": {
                    "path": str(paths["energies"].relative_to(output_root)),
                    "sha256": analysis_module.file_sha256(paths["energies"]),
                    "size_bytes": paths["energies"].stat().st_size,
                },
                "metrics": {
                    "path": str(paths["metrics"].relative_to(output_root)),
                    "sha256": analysis_module.file_sha256(paths["metrics"]),
                    "size_bytes": paths["metrics"].stat().st_size,
                },
                "metadata": {
                    "path": str(paths["metadata"].relative_to(output_root)),
                    "sha256": analysis_module.file_sha256(paths["metadata"]),
                    "size_bytes": paths["metadata"].stat().st_size,
                },
                "checkpoints": [
                    {
                        "step": row["step"],
                        "path": str(
                            _metadata_path(
                                row["path"], paths["metadata"], paths["checkpoints"]
                            ).relative_to(
                                output_root
                            )
                        ),
                        "sha256": row["sha256"],
                        "size_bytes": row["size_bytes"],
                    }
                    for row in metadata["checkpoints"]
                ],
            }
        )
    return result


def _next_directory(root: Path, prefix: str) -> Path:
    indices = []
    for path in root.glob(f"{prefix}-*"):
        match = re.fullmatch(
            re.escape(prefix) + r"-([0-9]{4})(?:[.]json)?", path.name
        )
        if match:
            indices.append(int(match.group(1)))
    return root / f"{prefix}-{max(indices, default=0) + 1:04d}"


def _analyze(
    output_root: Path,
    config: dict,
    protocol: dict,
    identity: dict,
    rows: list[tuple[Any, ...]],
    commit: str,
) -> dict[str, Any]:
    analysis_directory = _next_directory(output_root / "analyses", "analysis")
    analysis_directory.mkdir(parents=True, exist_ok=False)
    _append_event(
        output_root / "events.jsonl",
        "ANALYSIS_STARTED",
        {"analysis_id": analysis_directory.name, "code_commit": commit},
    )
    energies = np.concatenate([loaded[0] for _, loaded, _, _ in rows], axis=0)
    steps = rows[0][1][1]
    trajectory_ids = np.concatenate([loaded[2] for _, loaded, _, _ in rows])
    map_ids = rows[0][1][3]
    for _, loaded, _, _ in rows[1:]:
        if not np.array_equal(steps, loaded[1]) or not np.array_equal(map_ids, loaded[3]):
            raise ValueError("producer step or map registries differ")
    combined_directory = output_root / "raw"
    combined_directory.mkdir(parents=True, exist_ok=True)
    combined_path = combined_directory / "combined-energies.npz"
    if not combined_path.exists():
        np.savez_compressed(
            combined_path,
            energies=energies,
            steps=steps,
            trajectory_ids=trajectory_ids,
            map_ids=map_ids,
        )
    else:
        existing = energy_io.load(combined_path)
        if not all(
            np.array_equal(observed, expected)
            for observed, expected in zip(existing, (energies, steps, trajectory_ids, map_ids))
        ):
            raise ValueError("immutable combined energy artifact differs from producers")
    decision = confirmation_module.analyze_confirmation_v3(
        energies, trajectory_ids, map_ids, protocol
    )
    exact = _exact_block_census(
        energies, [int(value) for value in protocol["block_sizes"]]
    )
    face_ledger = diagnostics_module.face_excursion_ledger(
        energies,
        [str(value) for value in trajectory_ids.tolist()],
        [str(value) for value in map_ids.tolist()],
    )
    producer_artifacts = _artifact_rows(output_root, rows)
    record = analysis_module.seal_record(
        {
            "schema_version": RESULT_SCHEMA,
            "analysis_id": analysis_directory.name,
            "run_id": config["run_id"],
            "completed_at_utc": _utc_now(),
            "run_root_record_sha256": identity["record_sha256"],
            "protocol_sha256": _canonical_digest(protocol),
            "qualification_record_sha256": config["qualification"]["record_sha256"],
            "code_commit": commit,
            "analyzer_code_commit": commit,
            "producer_code_commit": identity["code_commit"],
            "analysis_sources": _analysis_sources(commit),
            "combined_energy_artifact": {
                "path": str(combined_path.relative_to(output_root)),
                "sha256": analysis_module.file_sha256(combined_path),
                "size_bytes": combined_path.stat().st_size,
            },
            "producer_artifacts": producer_artifacts,
            "exact_rg_analysis": exact,
            "face_excursion_ledger": face_ledger,
            "registered_decision": decision,
            "loss_analysis": _loss_analysis(rows, protocol),
            "plga_analysis": _plga_analysis(rows, protocol),
        }
    )
    result_path = analysis_directory / "result.json"
    _write_new_json(result_path, record)
    analysis_manifest = analysis_module.seal_record(
        {
            "schema_version": "pldr-row-rg-analysis-manifest-v4",
            "analysis_id": analysis_directory.name,
            "run_root_record_sha256": identity["record_sha256"],
            "protocol_sha256": _canonical_digest(protocol),
            "analyzer_code_commit": commit,
            "producer_code_commit": identity["code_commit"],
            "parent_raw_artifact_sha256": analysis_module.file_sha256(combined_path),
            "result": {
                "path": str(result_path.relative_to(output_root)),
                "sha256": analysis_module.file_sha256(result_path),
                "record_sha256": record["record_sha256"],
            },
            "analysis_sources": record["analysis_sources"],
        }
    )
    _write_new_json(analysis_directory / "manifest.json", analysis_manifest)
    _append_event(
        output_root / "events.jsonl",
        "ANALYSIS_COMPLETED",
        {
            "analysis_id": analysis_directory.name,
            "result_record_sha256": record["record_sha256"],
            "result_file_sha256": analysis_module.file_sha256(result_path),
        },
    )
    events = output_root / "events.jsonl"
    manifest_directory = output_root / "manifests"
    manifest_directory.mkdir(parents=True, exist_ok=True)
    manifest_id = _next_directory(manifest_directory, "manifest").name
    prior_manifests = sorted(manifest_directory.glob("manifest-[0-9][0-9][0-9][0-9].json"))
    previous_manifest_record_sha256 = None
    if prior_manifests:
        previous_manifest = json.loads(prior_manifests[-1].read_text(encoding="utf-8"))
        analysis_module.verify_record_seal(previous_manifest)
        previous_manifest_record_sha256 = previous_manifest["record_sha256"]
    root_manifest = analysis_module.seal_record(
        {
            "schema_version": "pldr-row-rg-root-manifest-v3.1",
            "manifest_id": manifest_id,
            "previous_manifest_record_sha256": previous_manifest_record_sha256,
            "created_at_utc": _utc_now(),
            "run_id": config["run_id"],
            "run_root_record_sha256": identity["record_sha256"],
            "resolved_inputs": {
                "configuration": {
                    "path": "config-resolved.json",
                    "sha256": analysis_module.file_sha256(
                        output_root / "config-resolved.json"
                    ),
                    "canonical_sha256": _canonical_digest(config),
                },
                "protocol": {
                    "path": "protocol-resolved.json",
                    "sha256": analysis_module.file_sha256(
                        output_root / "protocol-resolved.json"
                    ),
                    "canonical_sha256": _canonical_digest(protocol),
                },
                "qualification": {
                    "path": "qualification-resolved.json",
                    "sha256": analysis_module.file_sha256(
                        output_root / "qualification-resolved.json"
                    ),
                    "record_sha256": config["qualification"]["record_sha256"],
                },
            },
            "event_log": {
                "path": events.name,
                "sha256": analysis_module.file_sha256(events),
                "event_count": len(_event_rows(events)),
            },
            "producer_artifacts": producer_artifacts,
            "combined_energy_artifact": record["combined_energy_artifact"],
            "analysis_manifest": {
                "path": str((analysis_directory / "manifest.json").relative_to(output_root)),
                "sha256": analysis_module.file_sha256(analysis_directory / "manifest.json"),
                "record_sha256": analysis_manifest["record_sha256"],
            },
        }
    )
    _write_new_json(manifest_directory / f"{root_manifest['manifest_id']}.json", root_manifest)
    return record


def execute(
    config_path: Path,
    output_root: Path,
    *,
    resume: bool = False,
    analyze_only: bool = False,
) -> dict[str, Any]:
    if resume and analyze_only:
        raise ValueError("resume and analyze-only are mutually exclusive")
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    config, protocol, protocol_source = _load_config(config_path)
    analyzer_commit, _ = _git_identity()
    root_path = output_root / "run-root.json"
    existing_manifest = any((output_root / "manifests").glob("manifest-*.json"))

    if root_path.exists():
        if not (resume or analyze_only):
            raise FileExistsError("existing run requires an explicit continuation mode")
        try:
            identity = _validate_root(output_root, config, protocol, analyzer_commit)
            producer_commit = str(identity["code_commit"])
            _, qualification = provenance_module.validate_qualification(
                config.get("qualification", {}),
                config_path.parent,
                protocol_source,
                protocol,
                repository_root=ROOT,
                source_commit=producer_commit,
            )
            qualification_copy = output_root / "qualification-resolved.json"
            if (
                not qualification_copy.is_file()
                or json.loads(qualification_copy.read_text(encoding="utf-8"))
                != qualification
            ):
                raise ValueError("resolved qualification record was removed or mutated")
            if existing_manifest:
                missing = _missing_final_artifacts(config, output_root)
                if missing:
                    raise FileNotFoundError(
                        "final producer artifacts are missing: " + ", ".join(missing)
                    )
                registered_verifier.verify(output_root)
        except FinalizedRootMutationError:
            raise
        except (OSError, KeyError, RuntimeError, ValueError) as error:
            if existing_manifest:
                raise _finalized_root_error(error) from error
            raise
    else:
        if resume or analyze_only:
            raise ValueError("continuation mode requires an existing run root")
        producer_commit = analyzer_commit
        _, qualification = provenance_module.validate_qualification(
            config.get("qualification", {}),
            config_path.parent,
            protocol_source,
            protocol,
            repository_root=ROOT,
            source_commit=producer_commit,
        )
        _verify_bound_producer(config, producer_commit, require_worktree=True)
        identity = _initialize_root(
            output_root,
            config_path,
            config,
            protocol_source,
            protocol,
            producer_commit,
        )
        _write_new_json(output_root / "qualification-resolved.json", qualification)

    missing = _missing_final_artifacts(config, output_root)
    if analyze_only and missing:
        raise ActiveRootContinuationError(
            f"analyze-only requires complete final producer artifacts: {missing}"
        )
    if producer_commit != analyzer_commit and missing:
        raise ActiveRootContinuationError(
            "historical roots with missing producer artifacts must be resumed "
            "from a checkout of their producer commit"
        )
    _verify_bound_producer(
        config, producer_commit, require_worktree=bool(missing)
    )

    event_count_before = len(_event_rows(output_root / "events.jsonl"))
    rows = _run_producers(
        config,
        protocol,
        output_root / "protocol-resolved.json",
        output_root,
        producer_commit,
    )
    event_count_after = len(_event_rows(output_root / "events.jsonl"))
    if resume and existing_manifest and event_count_after == event_count_before:
        *_, result_path = registered_verifier.load_latest_lineage(output_root)
        return json.loads(result_path.read_text(encoding="utf-8"))

    result = _analyze(
        output_root, config, protocol, identity, rows, analyzer_commit
    )
    registered_verifier.verify(output_root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--resume", action="store_true")
    modes.add_argument("--analyze-only", action="store_true")
    arguments = parser.parse_args()
    try:
        result = execute(
            Path(arguments.config),
            Path(arguments.output_root),
            resume=arguments.resume,
            analyze_only=arguments.analyze_only,
        )
    except (ActiveRootContinuationError, FinalizedRootMutationError) as error:
        parser.exit(2, f"runner error: {error}\n")
    decision = result["registered_decision"]
    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_id": result["analysis_id"],
                "record_sha256": result["record_sha256"],
                "flow_counts": decision["flow_classification_counts"],
                "phase_counts": decision["phase_classification_counts"],
                "finite_variance_fixed_law_consistent": decision[
                    "finite_variance_fixed_law_consistent_on_registered_scales"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
