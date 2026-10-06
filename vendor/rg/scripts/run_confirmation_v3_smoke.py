#!/usr/bin/env python3
"""Execute a small source-bound smoke run of the production runner.

The short model run exercises the real two-device wave, artifact, analysis,
manifest, observer, and stage-replay paths. The result is infrastructure
evidence, not evidence about long-time PLDR dynamics.
"""

from __future__ import annotations
from companion_paths import configured_path

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import (  # noqa: E402
    file_sha256,
    seal_record,
    verify_record_seal,
)
from row_rgmap.provenance_v3 import derive_seed, git_blob_descriptor  # noqa: E402
from scripts.run_confirmation_v3 import execute  # noqa: E402
from scripts.verify_registered_artifacts import verify  # noqa: E402
from scripts.verify_stage_resolved_observer import verify as verify_stage  # noqa: E402


def write_new_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def committed_identity() -> str:
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
        raise RuntimeError("smoke execution requires a clean committed repository")
    return commit


def source_descriptors(commit: str, paths: list[str]) -> list[dict]:
    rows = []
    for repository_path in paths:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        rows.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return rows


def descriptor(path: Path, archive_root: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    verify_record_seal(value)
    return {
        "path": path.relative_to(archive_root).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
        "record_sha256": value["record_sha256"],
    }


def run_command(arguments: list[str]) -> dict:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        arguments,
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in completed.stdout.splitlines() if line]
    if not lines:
        raise RuntimeError(f"subprocess produced no result: {arguments[1]}")
    return json.loads(lines[-1])


def smoke_protocol(master_seed: int, training_domains: list[str]) -> dict:
    domains = [*training_domains, "analysis:gaussian-fixed-law"]
    derived = {domain: derive_seed(master_seed, domain) for domain in domains}
    return {
        "schema_version": "pldr-row-rg-confirmation-protocol-v3",
        "protocol_id": "production-runner-stage-a-smoke-v9",
        "expected_updates": 32,
        "segments": {
            "burn_in_updates": 8,
            "design_updates": 8,
            "holdout_updates": 16,
        },
        "randomness": {
            "master_seed": master_seed,
            "domains": domains,
            "derived_seeds": derived,
            "derivation": "SHA-256 domain-separated unsigned 32-bit integer",
        },
        "block_sizes": [1, 2, 4, 8],
        "observer_effect_floor": 1e-12,
        "identity_tolerances": {
            "float64_affine_replay_relative": 1e-12,
            "float32_gate_shape_map_backward_relative": 1e-6,
            "plga_decomposition_relative": 1e-12,
            "constrained_plga_defect_norm": 1e-9,
        },
        "analysis": {
            "familywise_alpha": 0.05,
            "familywise_cell_count": 12,
            "time_batch_length": 4,
            "window_length": 8,
            "target_mde_per_update": 0.01,
            "law_block_sizes": [2, 4],
            "variance_block_sizes": [1, 2],
            "excursion_events": {
                "layer": 1,
                "absolute_log_gain_threshold": 1.0,
                "cluster_gap_updates": 2,
            },
            "stationarity": {
                "maximum_split_t": 1e6,
                "maximum_trend_t": 1e6,
                "minimum_variance_ratio": 0.0,
                "maximum_variance_ratio": 1e12,
                "maximum_design_holdout_shift_t": 1e6,
                "minimum_design_holdout_variance_ratio": 0.0,
                "maximum_design_holdout_variance_ratio": 1e12,
            },
            "gaussian_fixed_law": {
                "seed": derived["analysis:gaussian-fixed-law"],
                "bootstrap_replicates": 99,
                "minimum_blocks": 4,
                "diagnostic_band_alpha": 0.05,
            },
            "variance_scaling": {
                "minimum_variance_exponent": -100.0,
                "maximum_variance_exponent": 100.0,
                "maximum_design_holdout_exponent_shift": 100.0,
            },
            "regular_drift_prediction": {
                "minimum_design_r_squared": 0.0,
                "minimum_gamma": -100.0,
                "maximum_gamma": 100.0,
                "maximum_cumulative_relative_error": 1e6,
            },
        },
        "plga_analysis": {
            "checkpoint_steps": [0, 8, 16, 32],
            "segment_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "operator_power_iterations": 2,
            "operator_start_count": 1,
            "structural_residual_tolerance": 1e-9,
            "direct_defect_tolerance": 1e-6,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", required=True)
    parser.add_argument(
        "--reference-root",
        default=(
            configured_path('code:row/experiments')
        ),
    )
    parser.add_argument(
        "--dataset-shard",
        default=(
            configured_path('assets:refinedweb/datasets/huggingface_datasets/tiiuae___falcon-refinedweb/default/0.0.0/c735840575b629292b41da8dde11dcd523d4f91c/falcon-refinedweb-train-00000-of-05518.arrow')
        ),
    )
    parser.add_argument("--refinedweb-root", default=configured_path('assets:refinedweb'))
    arguments = parser.parse_args()
    archive_root = Path(arguments.archive_root).resolve()
    if archive_root.exists():
        raise FileExistsError(f"smoke archive already exists: {archive_root}")
    commit = committed_identity()
    archive_root.mkdir(parents=True)
    input_root = archive_root / "inputs"
    input_root.mkdir()
    output_root = archive_root / "run"
    reference_root = Path(arguments.reference_root).resolve()
    dataset_shard = Path(arguments.dataset_shard).resolve()

    protocol_path = input_root / "protocol.json"
    trajectory_ids = [f"stage-a-smoke-t{index}" for index in range(4)]
    training_domains = [f"training:{identifier}" for identifier in trajectory_ids]
    protocol = smoke_protocol(2026090409, training_domains)
    write_new_json(protocol_path, protocol)

    derived_seed_registry_replayed = all(
        protocol["randomness"]["derived_seeds"][domain]
        == derive_seed(protocol["randomness"]["master_seed"], domain)
        for domain in protocol["randomness"]["domains"]
    )
    production_model_smoke_contract_passed = bool(
        protocol["expected_updates"] == 32
        and protocol["segments"]
        == {"burn_in_updates": 8, "design_updates": 8, "holdout_updates": 16}
        and len(trajectory_ids) == 4
        and reference_root.is_dir()
        and dataset_shard.is_file()
    )
    launch_gates = {
        "production_model_smoke_contract_passed": (
            production_model_smoke_contract_passed
        ),
        "derived_seed_registry_replayed": derived_seed_registry_replayed,
    }
    qualification = seal_record(
        {
            "schema_version": "pldr-row-rg-confirmation-v3-qualification-v1",
            "protocol_id": protocol["protocol_id"],
            "protocol_file_sha256": file_sha256(protocol_path),
            "code_commit": commit,
            "gates": launch_gates,
            "all_launch_gates_passed": all(launch_gates.values()),
            "qualification_sources": source_descriptors(
                commit,
                [
                    "scripts/run_confirmation_v3_smoke.py",
                    "src/row_rgmap/provenance_v3.py",
                ],
            ),
        }
    )
    qualification_path = input_root / "qualification.json"
    write_new_json(qualification_path, qualification)
    trajectories = [
        {
            "trajectory_id": trajectory_ids[index],
            "wave": index // 2,
            "device": index % 2,
            "seed_domain": training_domains[index],
            "seed": protocol["randomness"]["derived_seeds"][
                training_domains[index]
            ],
            "train_document_offset": 150000 + 1000 * index,
        }
        for index in range(4)
    ]
    config = {
        "schema_version": "pldr-row-rg-four-trajectory-run-v3",
        "run_id": "production-runner-stage-a-smoke-v9",
        "devices": [0, 1],
        "producer_timeout_seconds": 1800,
        "protocol_path": str(protocol_path),
        "qualification": {
            "path": str(qualification_path),
            "file_sha256": file_sha256(qualification_path),
            "record_sha256": qualification["record_sha256"],
        },
        "trajectories": trajectories,
        "producer_source": source_descriptors(
            commit, ["scripts/pldr_energy_producer_v3.py"]
        )[0],
        "producer_command": [
            sys.executable,
            "scripts/pldr_energy_producer_v3.py",
            "--logical-device", "cuda:0",
            "--physical-device", "{device}",
            "--trajectory-id", "{trajectory_id}",
            "--seed", "{seed}",
            "--output", "{output}",
            "--metrics-output", "{metrics}",
            "--metadata-output", "{metadata}",
            "--checkpoint-dir", "{checkpoint_dir}",
            "--protocol", "{protocol}",
            "--code-commit", "{code_commit}",
            "--reference-experiments", str(reference_root),
            "--dataset-shard", str(dataset_shard),
            "--steps", "32",
            "--layers", "3",
            "--heads", "4",
            "--dk", "64",
            "--contexts", "4",
            "--context-length", "64",
            "--batch-size", "4",
            "--learning-rate", "0.001",
            "--train-document-offset", "{train_document_offset}",
            "--probe-document-offset", "140000",
            "--max-sequence-length", "128",
        ],
    }
    config_path = input_root / "config.json"
    write_new_json(config_path, config)
    result = execute(config_path, output_root)

    observer_path = archive_root / "observer" / "result.json"
    observer_cli = run_command(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "analyze_observer_regime.py"),
            "--run-root",
            str(output_root),
            "--output",
            str(observer_path),
        ]
    )
    stage_path = archive_root / "stage" / "result.json"
    stage_cli = run_command(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "run_stage_resolved_observer.py"),
            "--completed-run-root",
            str(output_root),
            "--devices",
            "cpu",
            "cuda:0",
            "cuda:1",
            "--precision-bits",
            "192",
            "--refinedweb-root",
            str(Path(arguments.refinedweb_root).resolve()),
            "--output",
            str(stage_path),
        ]
    )

    verified = verify(output_root)
    stage_verified = verify_stage(stage_path, output_root, reference_root)
    result_path = output_root / "analyses" / "analysis-0001" / "result.json"
    manifest_path = output_root / "manifests" / "manifest-0001.json"
    observer_completed = observer_cli.get("status") == "complete"
    stage_completed = stage_cli.get("status") == "complete"
    artifact_verifier_passed = bool(verified)
    stage_verifier_passed = bool(stage_verified)
    all_checks_passed = all(
        (
            observer_completed,
            stage_completed,
            artifact_verifier_passed,
            stage_verifier_passed,
            derived_seed_registry_replayed,
        )
    )
    verification = seal_record(
        {
            "schema_version": "pldr-row-rg-stage-a-smoke-verification-v1",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_commit": commit,
            "analysis_sources": source_descriptors(
                commit,
                [
                    "scripts/run_confirmation_v3_smoke.py",
                    "scripts/verify_registered_artifacts.py",
                    "scripts/verify_stage_resolved_observer.py",
                    "src/row_rgmap/analysis.py",
                ],
            ),
            "inputs": {
                "protocol": {
                    "path": protocol_path.relative_to(archive_root).as_posix(),
                    "sha256": file_sha256(protocol_path),
                    "size_bytes": protocol_path.stat().st_size,
                },
                "qualification": descriptor(qualification_path, archive_root),
                "runner_result": descriptor(result_path, archive_root),
                "root_manifest": descriptor(manifest_path, archive_root),
                "observer_record": descriptor(observer_path, archive_root),
                "stage_record": descriptor(stage_path, archive_root),
            },
            "checks": {
                "registered_artifact_verifier": verified,
                "registered_artifact_verifier_passed": artifact_verifier_passed,
                "observer_command_result": observer_cli,
                "observer_command_completed": observer_completed,
                "stage_command_result": stage_cli,
                "stage_command_completed": stage_completed,
                "stage_record_verifier": stage_verified,
                "stage_record_verifier_passed": stage_verifier_passed,
                "derived_seed_registry_replayed": derived_seed_registry_replayed,
                "all_checks_passed": all_checks_passed,
            },
            "interpretation": {
                "scientific_scope": "infrastructure smoke, not long-time dynamics evidence",
                "registered_evidence_root_mutation_status": "not modified",
            },
        }
    )
    verification_path = archive_root / "verification.json"
    write_new_json(verification_path, verification)

    print(
        json.dumps(
            {
                "status": "complete",
                "analysis_id": result["analysis_id"],
                "record_sha256": result["record_sha256"],
                "manifest_id": verified["selected_root_manifest_id"],
                "event_count": verified["event_count"],
                "code_commit": commit,
                "observer_record_sha256": descriptor(observer_path, archive_root)[
                    "record_sha256"
                ],
                "stage_record_sha256": stage_verified["record_sha256"],
                "verification_record_sha256": verification["record_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
