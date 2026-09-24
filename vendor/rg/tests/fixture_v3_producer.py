"""Deterministic subprocess fixture for the production v3 runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def descriptor(path: Path, relative: str) -> dict[str, object]:
    return {
        "path": relative,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-document-offset", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metrics-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--behavior")
    parser.add_argument("--resume-from")
    return parser.parse_args()


def plga_diagnostics(map_ids: list[str]) -> dict[str, object]:
    structural = {
        name: {"absolute": 0.0, "relative": 0.0}
        for name in (
            "W_one",
            "bias",
            "power",
            "coupling_one",
            "coupling_bias",
        )
    }
    return {
        "rows": [
            {
                "map_id": map_id,
                "constant_input_defect_norm": 2e-6,
                "sampled_operator_norm_lower_bound": 0.5,
                "realized_secant_gain": 0.4,
                "decomposition_relative_residual": 0.0,
                "structural_residuals": structural,
            }
            for map_id in map_ids
        ],
        "constrained_head_controls": {
            "maximum_constant_input_defect_norm": 0.0,
        },
    }


def main() -> None:
    arguments = parse()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(arguments.physical_device):
        raise RuntimeError("runner did not isolate the requested physical device")

    behavior: dict[str, str] = {}
    if arguments.behavior:
        behavior = json.loads(Path(arguments.behavior).read_text(encoding="utf-8"))
    action = behavior.get(arguments.trajectory_id, "success")
    checkpoint_directory = Path(arguments.checkpoint_dir)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if arguments.resume_from is None and action == "fail":
        print("controlled fixture failure", flush=True)
        raise SystemExit(17)
    if arguments.resume_from is None and action == "interrupt":
        (checkpoint_directory / "checkpoint-step000004.pt").write_bytes(
            b"deterministic interrupted checkpoint\n"
        )
        print("controlled fixture interruption", flush=True)
        raise SystemExit(130)

    protocol_path = Path(arguments.protocol)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    updates = int(protocol["expected_updates"])
    output = Path(arguments.output)
    metrics_output = Path(arguments.metrics_output)
    metadata_output = Path(arguments.metadata_output)
    output.parent.mkdir(parents=True, exist_ok=True)

    map_ids = ["c0.L0.H0", "c0.L1.H0", "c0.L2.H0"]
    update = np.arange(updates, dtype=np.float64)
    generator = np.random.default_rng(arguments.seed)
    noise = generator.normal(0.0, 2e-4, size=(updates, len(map_ids)))
    drift = np.asarray([-3e-4, -1e-4, 2e-4], dtype=np.float64)
    log_edges = noise + drift[None, :]
    logs = np.concatenate(
        [
            np.zeros((1, len(map_ids)), dtype=np.float64),
            np.cumsum(log_edges, axis=0),
        ],
        axis=0,
    )
    energies = np.exp(logs)[None, :, :]
    np.savez_compressed(
        output,
        energies=energies,
        steps=np.arange(updates + 1, dtype=np.int64),
        trajectory_ids=np.asarray([arguments.trajectory_id]),
        map_ids=np.asarray(map_ids),
    )
    np.savez_compressed(
        metrics_output,
        loss_steps=np.arange(1, updates + 1, dtype=np.int64),
        losses=2.0 - 0.01 * np.arange(updates, dtype=np.float64),
    )

    checkpoint_rows = []
    for step in protocol["plga_analysis"]["checkpoint_steps"]:
        path = checkpoint_directory / f"checkpoint-step{int(step):06d}.pt"
        path.write_bytes(
            f"{arguments.trajectory_id}:{int(step)}:{arguments.seed}\n".encode()
        )
        checkpoint_rows.append(
            descriptor(path, f"checkpoints/{path.name}") | {"step": int(step)}
        )

    fixture_path = Path(__file__).resolve()
    fixture_sha = sha256(fixture_path)
    source_sha = "0" * 64 if action == "bad_source" else fixture_sha
    initial = energies[0, 0]
    metadata = {
        "schema_version": "pldr-row-rg-energy-producer-metadata-v3",
        "trajectory_id": arguments.trajectory_id,
        "physical_device": arguments.physical_device,
        "logical_device": "cuda:0",
        "cuda_visible_devices": visible,
        "visible_gpu_count": 1,
        "seed": arguments.seed,
        "code_commit": arguments.code_commit,
        "wall_seconds": 0.01,
        "protocol": {
            "path": os.path.relpath(protocol_path, metadata_output.parent),
            "canonical_sha256": canonical_digest(protocol),
            "file_sha256": sha256(protocol_path),
        },
        "data": {
            "tokens_per_update": 8,
            "training": {
                "first_document": arguments.train_document_offset,
                "last_document_exclusive": arguments.train_document_offset + 1,
                "document_count": 1,
                "token_count": updates * 8,
            },
        },
        "sources": {
            "producer_path": "tests/fixture_v3_producer.py",
            "producer_sha256": source_sha,
            "reference_model_path": str(fixture_path),
            "reference_model_sha256": fixture_sha,
            "reference_attention_path": str(fixture_path),
            "reference_attention_sha256": fixture_sha,
            "recorder_path": "tests/fixture_v3_producer.py",
            "recorder_sha256": fixture_sha,
        },
        "capture": {
            "map_ids": map_ids,
            "states": updates + 1,
            "hook_identity_checks": len(checkpoint_rows),
            "gate_shape_maximum_quotient_energy_absolute_residual": 0.0,
            "initial_energy_summary": {
                "count": len(initial),
                "minimum": float(initial.min()),
                "median": float(np.median(initial)),
                "p95": float(np.quantile(initial, 0.95)),
                "maximum": float(initial.max()),
            },
        },
        "plga_diagnostics_by_checkpoint": [
            {
                "step": int(step),
                "diagnostics": plga_diagnostics(map_ids),
            }
            for step in protocol["plga_analysis"]["checkpoint_steps"]
        ],
        "checkpoints": checkpoint_rows,
        "artifacts": {
            "energies": descriptor(output, "energies.npz"),
            "metrics": descriptor(metrics_output, "metrics.npz"),
        },
    }
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
