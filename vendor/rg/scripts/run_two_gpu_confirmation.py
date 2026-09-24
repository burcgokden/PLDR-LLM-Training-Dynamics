#!/usr/bin/env python3
"""Run, validate, resume, and seal the registered two-GPU confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_energy_npz import load  # noqa: E402
from row_rgmap.analysis import (  # noqa: E402
    analyze_energy_array,
    canonical_json_bytes,
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.confirmation import analyze_confirmation  # noqa: E402


CONFIG_SCHEMA = "pldr-row-rg-two-gpu-confirmation-v2"
PROTOCOL_SCHEMA = "pldr-row-rg-confirmation-protocol-v2"
RESULT_SCHEMA = "pldr-row-rg-two-gpu-confirmation-result-v3"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _object_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _analysis_source_rows() -> list[dict[str, Any]]:
    paths = (
        ROOT / "scripts" / "run_two_gpu_confirmation.py",
        ROOT / "scripts" / "analyze_energy_npz.py",
        ROOT / "src" / "row_rgmap" / "analysis.py",
        ROOT / "src" / "row_rgmap" / "confirmation.py",
        ROOT / "src" / "row_rgmap" / "core.py",
        ROOT / "src" / "row_rgmap" / "statistics.py",
    )
    return [
        {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _resolve_relative(path: str, parent: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (parent / candidate).resolve()


def _format_command(
    command: Any, *, device: int, output: Path, metadata: Path
) -> list[str]:
    if not isinstance(command, list) or not command or not all(
        isinstance(value, str) and value for value in command
    ):
        raise ValueError("producer_command must be a nonempty string argument vector")
    replacements = {
        "device": str(device),
        "output": str(output),
        "metadata": str(metadata),
    }
    try:
        return [value.format_map(replacements) for value in command]
    except KeyError as error:
        raise ValueError(f"unknown producer placeholder: {error.args[0]}") from error


def _argument_value(command: list[str], flag: str) -> str:
    indices = [index for index, value in enumerate(command) if value == flag]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError(f"producer command must contain exactly one {flag}")
    return command[indices[0] + 1]


def _load_inputs(config_path: Path) -> tuple[dict, dict, Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unknown confirmation configuration schema")
    if config.get("devices") != [0, 1]:
        raise ValueError("the registered runner requires devices [0, 1]")
    run_id = config.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise ValueError("run_id is missing or unsafe")
    expected_updates = config.get("expected_updates")
    if (
        not isinstance(expected_updates, int)
        or isinstance(expected_updates, bool)
        or expected_updates < 2
    ):
        raise ValueError("expected_updates must be an integer of at least two")
    timeout = config.get("producer_timeout_seconds", 21600)
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise ValueError("producer_timeout_seconds must be a positive integer")
    protocol_value = config.get("protocol_path")
    if not isinstance(protocol_value, str) or not protocol_value:
        raise ValueError("protocol_path is required")
    protocol_path = _resolve_relative(protocol_value, config_path.parent)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unknown confirmation protocol schema")
    if protocol.get("expected_updates") != expected_updates:
        raise ValueError("configuration and protocol update counts differ")

    blocks = protocol.get("block_sizes")
    if (
        not isinstance(blocks, list)
        or not blocks
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 1
            for value in blocks
        )
        or blocks != sorted(set(blocks))
    ):
        raise ValueError("protocol block_sizes must be sorted unique positive integers")
    rules = protocol.get("decision_rules", {})
    split = int(expected_updates * float(rules.get("estimation_fraction", 0.0)))
    batch = int(rules.get("time_batch_length", 0))
    minimum = int(rules.get("minimum_time_batches", 0))
    minimum_law = int(
        rules.get("gaussian_calibration", {}).get("minimum_blocks", 0)
    )
    if split < batch * minimum or expected_updates - split < batch * minimum:
        raise ValueError("registered estimation or holdout segment has too few batches")
    if (expected_updates - split) // max(blocks) < minimum_law:
        raise ValueError("largest registered block has too few holdout blocks")

    formatted = _format_command(
        config.get("producer_command"),
        device=0,
        output=Path("ENERGIES.npz"),
        metadata=Path("METADATA.json"),
    )
    if int(_argument_value(formatted, "--steps")) != expected_updates:
        raise ValueError("producer --steps does not equal expected_updates")
    if int(_argument_value(formatted, "--physical-device")) != 0:
        raise ValueError("producer physical-device placeholder is not bound")
    if _argument_value(formatted, "--output") != "ENERGIES.npz":
        raise ValueError("producer output placeholder is not bound")
    if _argument_value(formatted, "--metadata-output") != "METADATA.json":
        raise ValueError("producer metadata placeholder is not bound")
    return config, protocol, protocol_path


def _artifact_paths(output_root: Path, device: int) -> tuple[Path, Path, Path]:
    return (
        output_root / f"device-{device}-energies.npz",
        output_root / f"device-{device}-metadata.json",
        output_root / f"device-{device}-producer.log",
    )


def _validate_declared_file(
    path_value: Any, digest_value: Any, *, label: str, device: int
) -> None:
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"device {device} metadata lacks {label} path")
    if (
        not isinstance(digest_value, str)
        or len(digest_value) != 64
        or any(character not in "0123456789abcdef" for character in digest_value)
    ):
        raise ValueError(f"device {device} metadata has invalid {label} digest")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"device {device} declared {label} is unavailable: {path}")
    if file_sha256(path) != digest_value:
        raise ValueError(f"device {device} declared {label} digest mismatch")


def _validate_output_artifact(
    output_root: Path, descriptor: Any, *, label: str
) -> Path:
    if not isinstance(descriptor, dict):
        raise ValueError(f"{label} descriptor is missing")
    name = descriptor.get("path")
    digest = descriptor.get("sha256")
    size = descriptor.get("size_bytes")
    if (
        not isinstance(name, str)
        or not name
        or Path(name).name != name
        or Path(name).is_absolute()
    ):
        raise ValueError(f"{label} path is unsafe")
    path = output_root / name
    if not path.is_file():
        raise ValueError(f"{label} is unavailable: {path}")
    if not isinstance(size, int) or isinstance(size, bool) or path.stat().st_size != size:
        raise ValueError(f"{label} size mismatch")
    if not isinstance(digest, str) or file_sha256(path) != digest:
        raise ValueError(f"{label} digest mismatch")
    return path


def _validate_metadata(
    metadata_path: Path,
    artifact_path: Path,
    loaded: tuple,
    *,
    device: int,
    expected_updates: int,
    protocol: dict,
) -> dict:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != "pldr-row-rg-energy-producer-metadata-v2":
        raise ValueError(f"device {device} has unknown metadata schema")
    if metadata.get("physical_device") != device:
        raise ValueError(f"device {device} metadata has wrong physical device")
    if metadata.get("cuda_visible_devices") != str(device):
        raise ValueError(f"device {device} metadata has wrong CUDA binding")
    if metadata.get("visible_gpu_count") != 1:
        raise ValueError(f"device {device} producer did not see exactly one GPU")
    if metadata.get("logical_device") != "cuda:0":
        raise ValueError(f"device {device} did not use logical cuda:0")

    sources = metadata.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(f"device {device} metadata lacks source identities")
    for prefix, label in (
        ("producer", "producer source"),
        ("reference_model", "reference model source"),
        ("reference_attention", "reference attention source"),
    ):
        _validate_declared_file(
            sources.get(f"{prefix}_path"),
            sources.get(f"{prefix}_sha256"),
            label=label,
            device=device,
        )
    data = metadata.get("data")
    if not isinstance(data, dict):
        raise ValueError(f"device {device} metadata lacks dataset identity")
    _validate_declared_file(
        data.get("shard_path"),
        data.get("shard_sha256"),
        label="dataset shard",
        device=device,
    )

    energies, steps, trajectory_ids, map_ids = loaded
    capture = metadata.get("capture", {})
    if metadata.get("optimizer", {}).get("updates") != expected_updates:
        raise ValueError(f"device {device} metadata has wrong update count")
    if capture.get("states") != expected_updates + 1:
        raise ValueError(f"device {device} metadata has wrong state count")
    if steps.tolist() != list(range(expected_updates + 1)):
        raise ValueError(f"device {device} artifact has wrong step grid")
    if trajectory_ids.tolist() != [capture.get("trajectory_id")]:
        raise ValueError(f"device {device} trajectory registry mismatch")
    if map_ids.tolist() != capture.get("map_ids"):
        raise ValueError(f"device {device} map registry mismatch")
    if energies.shape[1] != expected_updates + 1:
        raise ValueError(f"device {device} artifact has wrong time dimension")
    if metadata.get("energy_artifact", {}).get("sha256") != file_sha256(
        artifact_path
    ):
        raise ValueError(f"device {device} metadata artifact digest mismatch")
    tolerance = float(
        protocol["identity_tolerances"]["float32_gate_shape_relative"]
    )
    residual = float(capture.get("gate_shape_maximum_relative_residual", np.inf))
    if residual > tolerance:
        raise ValueError(f"device {device} gate-shape identity exceeds tolerance")
    return metadata


def _summary(values: np.ndarray) -> dict:
    return {
        "count": int(len(values)),
        "minimum": float(values.min()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": float(values.max()),
    }


def _plga_decision(metadata_rows: list[dict], protocol: dict) -> dict:
    interventions = [row["plga_interventions_at_final_state"] for row in metadata_rows]
    rows = [entry for intervention in interventions for entry in intervention["rows"]]
    expected = sum(int(intervention["map_count"]) for intervention in interventions)
    if not rows or len(rows) != expected:
        raise ValueError("PLGA intervention registry is empty or inconsistent")
    defects = np.asarray(
        [row["constant_input_defect_norm"] for row in rows], dtype=np.float64
    )
    ratios = np.asarray(
        [row["defect_to_output_norm_ratio"] for row in rows], dtype=np.float64
    )
    secants = np.asarray(
        [row["realized_secant_gain"] for row in rows], dtype=np.float64
    )
    derivatives = np.asarray(
        [row["maximum_sampled_directional_gain"] for row in rows], dtype=np.float64
    )
    residuals = np.asarray(
        [row["decomposition_relative_residual"] for row in rows], dtype=np.float64
    )
    if not all(
        np.isfinite(values).all()
        for values in (defects, ratios, secants, derivatives, residuals)
    ):
        raise ValueError("PLGA intervention record contains non-finite values")
    rules = protocol["decision_rules"]["plga_transfer"]
    defect_bound = float(rules["maximum_constant_input_defect_norm"])
    decomposition_bound = float(rules["maximum_decomposition_relative_residual"])
    derivative_bound = float(rules["maximum_sampled_directional_gain"])
    face_pass = bool(defects.max() <= defect_bound)
    decomposition_pass = bool(residuals.max() <= decomposition_bound)
    derivative_pass = bool(derivatives.max() <= derivative_bound)
    return {
        "schema_version": "pldr-plga-transfer-decision-v1",
        "implemented_intervention_count": len(rows),
        "constant_input_defect_norm": _summary(defects),
        "defect_to_output_norm_ratio": _summary(ratios),
        "realized_secant_gain": _summary(secants),
        "maximum_sampled_directional_gain": _summary(derivatives),
        "decomposition_relative_residual": _summary(residuals),
        "face_preservation_bound": defect_bound,
        "decomposition_residual_bound": decomposition_bound,
        "sampled_directional_gain_bound": derivative_bound,
        "finite_horizon_face_preservation_gate_passed": face_pass,
        "exact_decomposition_gate_passed": decomposition_pass,
        "sampled_response_bound_gate_passed": derivative_pass,
        "all_registered_transfer_gates_passed": (
            face_pass and decomposition_pass and derivative_pass
        ),
        "outcome": (
            "IMPLEMENTED_PLGA_TRANSFER_GATES_PASSED"
            if face_pass and decomposition_pass and derivative_pass
            else "IMPLEMENTED_PLGA_FINITE_FACE_PRESERVATION_REJECTED"
            if not face_pass
            else "IMPLEMENTED_PLGA_TRANSFER_UNRESOLVED"
        ),
        "scope": "FINAL_STATE_IMPLEMENTED_HEADS_AND_REGISTERED_SEGMENT_GRID",
    }


def _write_state(path: Path, **fields: Any) -> None:
    previous = {}
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
    previous.update(fields)
    previous["updated_at_utc"] = _utc_now()
    _write_json_atomic(path, previous)


def execute(
    config_path: Path,
    output_root: Path,
    *,
    resume: bool = False,
    reanalyze: bool = False,
) -> dict:
    config_path = config_path.resolve()
    config, protocol, protocol_path = _load_inputs(config_path)
    config_digest = _object_digest(config)
    protocol_digest = _object_digest(protocol)
    devices = config["devices"]
    expected_updates = int(config["expected_updates"])
    timeout_seconds = int(config.get("producer_timeout_seconds", 21600))
    final_path = output_root / "confirmation-result.json"
    state_path = output_root / "run-state.json"
    if resume and reanalyze:
        raise ValueError("resume and reanalyze are mutually exclusive")
    reuse_artifacts = resume or reanalyze

    if output_root.exists() and not reuse_artifacts and any(output_root.iterdir()):
        raise FileExistsError(f"confirmation output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    if resume and final_path.is_file():
        result = json.loads(final_path.read_text(encoding="utf-8"))
        verify_record_seal(result)
        if result.get("config_sha256") != config_digest:
            raise ValueError("existing result belongs to a different configuration")
        if result.get("protocol_sha256") != protocol_digest:
            raise ValueError("existing result belongs to a different protocol")
        if not state_path.is_file():
            raise ValueError("completed result has no run state")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        result_file_digest = file_sha256(final_path)
        if (
            state.get("status") != "COMPLETE"
            or state.get("result_record_sha256") != result["record_sha256"]
            or state.get("result_file_sha256") != result_file_digest
        ):
            raise ValueError("completed result and run state are inconsistent")
        manifest_path = output_root / "run-manifest.json"
        if not manifest_path.is_file():
            raise ValueError("completed result has no run manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verify_record_seal(manifest)
        if manifest.get("analysis_sources") != result.get("analysis_sources"):
            raise ValueError("completed result and analysis sources are inconsistent")
        for index, source in enumerate(result.get("analysis_sources", [])):
            _validate_declared_file(
                source.get("path"),
                source.get("sha256"),
                label=f"analysis source {index}",
                device=-1,
            )
            path = Path(source["path"])
            if source.get("size_bytes") != path.stat().st_size:
                raise ValueError(f"analysis source {index} size mismatch")
        if (
            manifest.get("run_id") != result.get("run_id")
            or manifest.get("result", {}).get("sha256") != result_file_digest
            or manifest.get("result", {}).get("record_sha256")
            != result["record_sha256"]
            or manifest.get("combined_artifact") != result.get("combined_artifact")
            or manifest.get("producer_artifacts") != result.get("producer_artifacts")
        ):
            raise ValueError("completed result and run manifest are inconsistent")
        if manifest.get("config_resolved_sha256") != file_sha256(
            output_root / "config-resolved.json"
        ) or manifest.get("protocol_resolved_sha256") != file_sha256(
            output_root / "protocol-resolved.json"
        ):
            raise ValueError("resolved frozen inputs do not match the manifest")

        loaded_rows = []
        for producer in result["producer_artifacts"]:
            device = int(producer["device"])
            energy_path = _validate_output_artifact(
                output_root,
                producer.get("energy_artifact"),
                label=f"device {device} energy artifact",
            )
            metadata_path = _validate_output_artifact(
                output_root,
                producer.get("metadata_artifact"),
                label=f"device {device} metadata artifact",
            )
            _validate_output_artifact(
                output_root,
                producer.get("log_artifact"),
                label=f"device {device} log artifact",
            )
            loaded = load(energy_path)
            _validate_metadata(
                metadata_path,
                energy_path,
                loaded,
                device=device,
                expected_updates=expected_updates,
                protocol=protocol,
            )
            loaded_rows.append(loaded)
        combined_path = _validate_output_artifact(
            output_root,
            result.get("combined_artifact"),
            label="combined energy artifact",
        )
        combined_loaded = load(combined_path)
        expected_combined = (
            np.concatenate([row[0] for row in loaded_rows], axis=0),
            loaded_rows[0][1],
            np.concatenate([row[2] for row in loaded_rows], axis=0),
            loaded_rows[0][3],
        )
        if not all(
            np.array_equal(observed, expected)
            for observed, expected in zip(combined_loaded, expected_combined)
        ):
            raise ValueError("completed combined artifact does not replay")
        return result
    if reuse_artifacts and state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("config_sha256") != config_digest or state.get(
            "protocol_sha256"
        ) != protocol_digest:
            raise ValueError("resume state belongs to a different protocol")

    _write_json_atomic(output_root / "config-resolved.json", config)
    _write_json_atomic(output_root / "protocol-resolved.json", protocol)
    _write_state(
        state_path,
        schema_version="pldr-row-rg-run-state-v1",
        status="RUNNING",
        started_at_utc=_utc_now(),
        run_id=config["run_id"],
        config_sha256=config_digest,
        protocol_sha256=protocol_digest,
        reanalyzing_existing_artifacts=reanalyze,
    )

    processes: list[tuple[int, list[str], subprocess.Popen]] = []
    streams = []
    commands: dict[int, list[str]] = {}
    loaded_by_device = {}
    metadata_by_device = {}
    try:
        for device in devices:
            artifact, metadata_path, log_path = _artifact_paths(output_root, device)
            command = _format_command(
                config["producer_command"],
                device=device,
                output=artifact,
                metadata=metadata_path,
            )
            commands[device] = command
            artifact_set = (
                artifact,
                metadata_path,
                log_path,
            )
            if reanalyze and not all(
                path.is_file() for path in artifact_set
            ):
                raise FileNotFoundError(
                    f"--reanalyze requires complete existing artifacts for device {device}"
                )
            if reuse_artifacts and all(
                path.is_file() for path in artifact_set
            ):
                loaded = load(artifact)
                metadata = _validate_metadata(
                    metadata_path,
                    artifact,
                    loaded,
                    device=device,
                    expected_updates=expected_updates,
                    protocol=protocol,
                )
                loaded_by_device[device] = loaded
                metadata_by_device[device] = metadata
                continue
            if any(path.exists() for path in artifact_set):
                raise FileExistsError(
                    f"device {device} has an incomplete producer artifact set"
                )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = str(device)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            stream = log_path.open("ab" if reuse_artifacts else "xb")
            streams.append(stream)
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
            processes.append((device, command, process))

        deadline = time.monotonic() + timeout_seconds
        failures = []
        for device, command, process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            return_code = process.wait(timeout=remaining)
            if return_code != 0:
                failures.append({"device": device, "return_code": return_code})
        if failures:
            raise RuntimeError(f"producer failure: {failures}")
    except BaseException as error:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()
        for _, _, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=10.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        _write_state(
            state_path,
            status="FAILED",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        raise
    finally:
        for stream in streams:
            stream.close()

    try:
        for device in devices:
            if device in loaded_by_device:
                continue
            artifact, metadata_path, _ = _artifact_paths(output_root, device)
            loaded = load(artifact)
            metadata = _validate_metadata(
                metadata_path,
                artifact,
                loaded,
                device=device,
                expected_updates=expected_updates,
                protocol=protocol,
            )
            loaded_by_device[device] = loaded
            metadata_by_device[device] = metadata

        reference_steps = loaded_by_device[devices[0]][1]
        reference_maps = loaded_by_device[devices[0]][3]
        for device in devices[1:]:
            _, steps, _, map_ids = loaded_by_device[device]
            if not np.array_equal(steps, reference_steps):
                raise ValueError("device artifacts use different step grids")
            if not np.array_equal(map_ids, reference_maps):
                raise ValueError("device artifacts use different map registries")
        energies = np.concatenate(
            [loaded_by_device[device][0] for device in devices], axis=0
        )
        trajectory_ids = np.concatenate(
            [loaded_by_device[device][2] for device in devices], axis=0
        )
        if len(np.unique(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("trajectory_ids must be distinct across devices")

        blocks = tuple(int(value) for value in protocol["block_sizes"])
        exact_analysis = analyze_energy_array(energies, block_sizes=blocks)
        decision = analyze_confirmation(energies, trajectory_ids, reference_maps, protocol)
        plga = _plga_decision(
            [metadata_by_device[device] for device in devices], protocol
        )
        combined = output_root / "combined-energies.npz"
        if combined.exists():
            if not reanalyze:
                raise FileExistsError(combined)
            existing_combined = load(combined)
            expected_combined = (
                energies,
                reference_steps,
                trajectory_ids,
                reference_maps,
            )
            if not all(
                np.array_equal(observed, expected)
                for observed, expected in zip(existing_combined, expected_combined)
            ):
                raise ValueError("existing combined artifact does not replay")
        else:
            np.savez_compressed(
                combined,
                energies=energies,
                steps=reference_steps,
                trajectory_ids=trajectory_ids,
                map_ids=reference_maps,
            )

        producer_rows = []
        for device in devices:
            artifact, metadata_path, log_path = _artifact_paths(output_root, device)
            producer_rows.append(
                {
                    "device": device,
                    "command": commands[device],
                    "energy_artifact": {
                        "path": artifact.name,
                        "sha256": file_sha256(artifact),
                        "size_bytes": artifact.stat().st_size,
                    },
                    "metadata_artifact": {
                        "path": metadata_path.name,
                        "sha256": file_sha256(metadata_path),
                        "size_bytes": metadata_path.stat().st_size,
                    },
                    "log_artifact": {
                        "path": log_path.name,
                        "sha256": file_sha256(log_path),
                        "size_bytes": log_path.stat().st_size,
                    },
                    "validated_identity": {
                        "physical_device": metadata_by_device[device]["physical_device"],
                        "visible_gpu_count": metadata_by_device[device]["visible_gpu_count"],
                        "gpu": metadata_by_device[device]["gpu"],
                        "seed": metadata_by_device[device]["seed"],
                        "wall_seconds": metadata_by_device[device]["wall_seconds"],
                        "gate_shape_maximum_relative_residual": metadata_by_device[device][
                            "capture"
                        ]["gate_shape_maximum_relative_residual"],
                    },
                }
            )

        record = seal_record(
            {
                "schema_version": RESULT_SCHEMA,
                "run_id": config["run_id"],
                "completed_at_utc": _utc_now(),
                "config_sha256": config_digest,
                "protocol_sha256": protocol_digest,
                "source_config": {
                    "path": str(config_path),
                    "file_sha256": file_sha256(config_path),
                },
                "source_protocol": {
                    "path": str(protocol_path),
                    "file_sha256": file_sha256(protocol_path),
                },
                "analysis_sources": _analysis_source_rows(),
                "producer_artifacts": producer_rows,
                "combined_artifact": {
                    "path": combined.name,
                    "sha256": file_sha256(combined),
                    "size_bytes": combined.stat().st_size,
                },
                "exact_rg_analysis": exact_analysis,
                "registered_decision": decision,
                "plga_transfer_decision": plga,
            }
        )
        _write_json_atomic(final_path, record)
        manifest = seal_record(
            {
                "schema_version": "pldr-row-rg-run-manifest-v1",
                "run_id": config["run_id"],
                "created_at_utc": _utc_now(),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "config_resolved_sha256": file_sha256(
                    output_root / "config-resolved.json"
                ),
                "protocol_resolved_sha256": file_sha256(
                    output_root / "protocol-resolved.json"
                ),
                "result": {
                    "path": final_path.name,
                    "sha256": file_sha256(final_path),
                    "record_sha256": record["record_sha256"],
                },
                "combined_artifact": record["combined_artifact"],
                "analysis_sources": record["analysis_sources"],
                "producer_artifacts": producer_rows,
            }
        )
        _write_json_atomic(output_root / "run-manifest.json", manifest)
        _write_state(
            state_path,
            status="COMPLETE",
            completed_at_utc=_utc_now(),
            result_record_sha256=record["record_sha256"],
            result_file_sha256=file_sha256(final_path),
        )
        return record
    except BaseException as error:
        _write_state(
            state_path,
            status="FAILED",
            failure_type=type(error).__name__,
            failure_message=str(error),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--reanalyze", action="store_true")
    arguments = parser.parse_args()
    result = execute(
        Path(arguments.config),
        Path(arguments.output_root),
        resume=arguments.resume,
        reanalyze=arguments.reanalyze,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "outcome": result["registered_decision"]["outcome"],
                "plga_outcome": result["plga_transfer_decision"]["outcome"],
                "record_sha256": result["record_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
