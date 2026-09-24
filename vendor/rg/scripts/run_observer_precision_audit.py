#!/usr/bin/env python3
"""Reevaluate saved PLDR row maps on CPU and both GPUs.

The audit separates four quantities at every checkpoint map:

* the energy of the native float32 LayerNorm output, evaluated by the same
  float64 recorder rule used during training;
* a float64 analytic LayerNorm energy from the captured float32 input;
* the full-output reconstruction residual;
* an exact-input rational enclosure for every native represented zero.

The integrity-incomplete run may be supplied as a numerical sensitivity
control. It is labelled as such in the record and never enters a flow result.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.observer import (  # noqa: E402
    enclosure_class,
    layernorm_row_energy_enclosure,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: dict[str, Any]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("record_sha256", None)
    return json.dumps(
        unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def content_record(value: dict[str, Any]) -> dict[str, Any]:
    record = dict(value)
    record["record_sha256"] = hashlib.sha256(canonical_bytes(record)).hexdigest()
    return record


def option(command: list[str], name: str) -> str:
    try:
        index = command.index(name)
    except ValueError as error:
        raise ValueError(f"producer command lacks {name}") from error
    if index + 1 >= len(command):
        raise ValueError(f"producer command has no value for {name}")
    return str(command[index + 1])


def probe_tokens(shard: Path, document_offset: int, count: int) -> tuple[np.ndarray, str]:
    from datasets import Dataset

    dataset = Dataset.from_file(str(shard))
    values = bytearray()
    index = document_offset
    while len(values) < count:
        if index >= len(dataset):
            raise ValueError("dataset ended before the registered probe")
        content = dataset[index]["content"]
        if not isinstance(content, str):
            raise TypeError("registered probe document is not text")
        values.extend(content.encode("utf-8", errors="strict"))
        values.append(10)
        index += 1
    selected = bytes(values[:count])
    digest = hashlib.sha256(selected).hexdigest()
    tokens = np.frombuffer(selected, dtype=np.uint8).astype(np.int64) + 1
    return tokens, digest


def row_energy(maps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(maps, dtype=np.float64)
    equal = np.all(values == values[:, 0:1, :], axis=(1, 2))
    centered = values - values.mean(axis=1, keepdims=True, dtype=np.float64)
    energies = np.sum(centered * centered, axis=(1, 2), dtype=np.float64)
    energies[equal] = 0.0
    return energies, equal


def distinct_rows(maps: np.ndarray) -> list[int]:
    return [int(len(np.unique(value, axis=0))) for value in maps]


def summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def build_model(torch: Any, model_class: Any, specification: dict[str, int], device: Any) -> Any:
    d_model = specification["heads"] * specification["dk"]
    model = model_class(
        num_layers=specification["layers"],
        d_model=d_model,
        num_heads=specification["heads"],
        dff=int(2.7 * d_model),
        input_vocab_size=257,
        A_dff=2 * specification["dk"],
        num_reslayerA=2,
        num_denseA=2,
        max_seq_len=specification["max_sequence_length"],
        device=device,
    )
    return model.to(device=device, dtype=torch.float32).eval()


def capture(
    torch: Any,
    model: Any,
    probe: Any,
    context_length: int,
) -> list[tuple[Any, Any, Any, Any, float]]:
    mask = torch.triu(
        torch.ones(
            context_length,
            context_length,
            dtype=torch.float32,
            device=probe.device,
        ),
        diagonal=1,
    )[None, None]
    captured: dict[int, tuple[Any, Any, Any, Any, float]] = {}
    handles = []
    for layer, decoder_layer in enumerate(model.decoder.dec_layers):
        norm = decoder_layer.mha1.reslayerAs[-1].layernormA

        def hook(module: Any, inputs: Any, output: Any, *, index: int = layer) -> None:
            captured[index] = (
                inputs[0].detach().clone(),
                output.detach().clone(),
                module.weight.detach().clone(),
                module.bias.detach().clone(),
                float(module.eps),
            )

        handles.append(norm.register_forward_hook(hook))
    try:
        with torch.no_grad():
            model([probe[:, :-1], mask])
    finally:
        for handle in handles:
            handle.remove()
    if sorted(captured) != list(range(len(model.decoder.dec_layers))):
        raise RuntimeError("LayerNorm hooks did not fire exactly once")
    return [captured[index] for index in sorted(captured)]


def map_arrays(
    captured: list[tuple[Any, Any, Any, Any, float]],
    contexts: int,
    layers: int,
    heads: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    before_rows = []
    output_rows = []
    gamma_rows = []
    bias_rows = []
    epsilons = []
    for context in range(contexts):
        for layer in range(layers):
            before, output, gamma, bias, epsilon = captured[layer]
            for head in range(heads):
                before_rows.append(before[context, head].float().cpu().numpy())
                output_rows.append(output[context, head].float().cpu().numpy())
                gamma_rows.append(gamma.float().cpu().numpy())
                bias_rows.append(bias.float().cpu().numpy())
                epsilons.append(epsilon)
    return (
        np.asarray(before_rows, dtype=np.float32),
        np.asarray(output_rows, dtype=np.float32),
        np.asarray(gamma_rows, dtype=np.float32),
        np.asarray(bias_rows, dtype=np.float32),
        np.asarray(epsilons, dtype=np.float64),
    )


def evaluate_maps(
    before: np.ndarray,
    output: np.ndarray,
    gamma: np.ndarray,
    bias: np.ndarray,
    epsilons: np.ndarray,
    map_ids: list[str],
    precision_bits: int,
) -> dict[str, Any]:
    direct, equal_output = row_energy(output)
    centered_feature = before.astype(np.float64)
    centered_feature -= centered_feature.mean(axis=2, keepdims=True, dtype=np.float64)
    variance = np.mean(centered_feature * centered_feature, axis=2, keepdims=True)
    shape = centered_feature / np.sqrt(epsilons[:, None, None] + variance)
    expected = shape * gamma.astype(np.float64)[:, None, :] + bias.astype(np.float64)[:, None, :]
    factorized, _ = row_energy(expected)
    reconstruction_norm = np.linalg.norm(
        output.astype(np.float64) - expected, axis=(1, 2)
    )
    output_norm = np.linalg.norm(output.astype(np.float64), axis=(1, 2))
    reconstruction_relative = reconstruction_norm / np.maximum(
        output_norm, np.finfo(np.float64).tiny
    )
    discrepancy = np.abs(direct - factorized)
    symmetric = discrepancy / np.maximum.reduce(
        [direct, factorized, np.full_like(direct, np.finfo(np.float64).tiny)]
    )
    input_distinct = distinct_rows(before)
    output_distinct = distinct_rows(output)
    zero_rows = []
    for index in np.flatnonzero(equal_output):
        enclosure = layernorm_row_energy_enclosure(
            before[index],
            gamma[index],
            float(epsilons[index]),
            precision_bits=precision_bits,
        )
        lower, upper = enclosure
        zero_rows.append(
            {
                "map_id": map_ids[index],
                "map_index": int(index),
                "input_distinct_rows": input_distinct[index],
                "float64_factorized_energy": float(factorized[index]),
                "enclosure_lower": float(lower),
                "enclosure_upper": float(upper),
                "enclosure_classification": enclosure_class(enclosure),
            }
        )
    return {
        "direct_energy": direct.tolist(),
        "float64_factorized_energy": factorized.tolist(),
        "input_distinct_rows": input_distinct,
        "output_distinct_rows": output_distinct,
        "represented_zero_count": int(equal_output.sum()),
        "direct_energy_summary": summary(direct),
        "float64_factorized_energy_summary": summary(factorized),
        "maximum_reconstruction_relative_residual": float(reconstruction_relative.max()),
        "maximum_quotient_energy_absolute_discrepancy": float(discrepancy.max()),
        "maximum_quotient_energy_symmetric_relative_discrepancy": float(symmetric.max()),
        "represented_zero_rows": zero_rows,
    }


def checkpoint_inventory(run_root: Path, evidence_role: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config_path = run_root / "config-resolved.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    descriptors = []
    for trajectory in config["trajectories"]:
        trajectory_id = trajectory["trajectory_id"]
        checkpoint_dir = run_root / "trajectories" / trajectory_id / "checkpoints"
        expected = {}
        metadata_path = run_root / "trajectories" / trajectory_id / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            expected = {
                int(row["step"]): row["sha256"] for row in metadata["checkpoints"]
            }
        for path in sorted(checkpoint_dir.glob("checkpoint-step*.pt")):
            digest = file_sha256(path)
            step = int(path.stem.removeprefix("checkpoint-step"))
            if expected and expected.get(step) != digest:
                raise ValueError(f"checkpoint digest mismatch: {path}")
            descriptors.append(
                {
                    "run_id": config["run_id"],
                    "evidence_role": evidence_role,
                    "trajectory_id": trajectory_id,
                    "producing_device": int(trajectory["device"]),
                    "step": step,
                    "path": str(path.relative_to(run_root)),
                    "sha256": digest,
                    "size_bytes": path.stat().st_size,
                    "absolute_path": path,
                }
            )
    return config, descriptors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-run-root", required=True)
    parser.add_argument("--diagnostic-run-root")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0", "cuda:1"])
    parser.add_argument("--precision-bits", type=int, default=192)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    completed_root = Path(arguments.completed_run_root).resolve()
    run_specs = [(completed_root, "COMPLETED_DECISION_RECORD")]
    if arguments.diagnostic_run_root:
        run_specs.append(
            (Path(arguments.diagnostic_run_root).resolve(), "INTEGRITY_INCOMPLETE_NUMERICAL_CONTROL")
        )
    runs = []
    checkpoints = []
    for run_root, role in run_specs:
        config, inventory = checkpoint_inventory(run_root, role)
        runs.append((run_root, role, config))
        checkpoints.extend(inventory)
    if not checkpoints:
        raise ValueError("no saved checkpoints were found")

    base_command = runs[0][2]["producer_command"]
    specification = {
        "layers": int(option(base_command, "--layers")),
        "heads": int(option(base_command, "--heads")),
        "dk": int(option(base_command, "--dk")),
        "contexts": int(option(base_command, "--contexts")),
        "context_length": int(option(base_command, "--context-length")),
        "max_sequence_length": int(option(base_command, "--max-sequence-length")),
    }
    reference = Path(option(base_command, "--reference-experiments")).resolve()
    shard = Path(option(base_command, "--dataset-shard")).resolve()
    probe_offset = int(option(base_command, "--probe-document-offset"))
    probe_count = specification["contexts"] * (specification["context_length"] + 1)
    token_values, probe_digest = probe_tokens(shard, probe_offset, probe_count)
    metadata_probe_digest = json.loads(
        (completed_root / "trajectories" / runs[0][2]["trajectories"][0]["trajectory_id"] / "metadata.json").read_text(encoding="utf-8")
    )["data"]["probe"]["byte_token_sha256"]
    if probe_digest != metadata_probe_digest:
        raise ValueError("reconstructed probe bytes differ from producer metadata")

    sys.path.insert(0, str(reference))
    from pldr_model_v510 import PLDR_Model

    resolved_devices = []
    models = {}
    probes = {}
    for label in arguments.devices:
        device = torch.device(label)
        if device.type == "cuda":
            if not torch.cuda.is_available() or device.index is None or device.index >= torch.cuda.device_count():
                raise RuntimeError(f"requested device is unavailable: {label}")
            properties = torch.cuda.get_device_properties(device)
            description = {
                "label": label,
                "type": "cuda",
                "name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        else:
            description = {"label": label, "type": "cpu", "name": "host CPU"}
        torch.manual_seed(0)
        models[label] = build_model(torch, PLDR_Model, specification, device)
        probes[label] = torch.tensor(token_values, dtype=torch.long, device=device).reshape(
            specification["contexts"], specification["context_length"] + 1
        )
        resolved_devices.append(description)

    map_ids = [
        f"c{context}.L{layer}.H{head}"
        for context in range(specification["contexts"])
        for layer in range(specification["layers"])
        for head in range(specification["heads"])
    ]
    evaluations = []
    checkpoint_inputs = []
    for descriptor in checkpoints:
        checkpoint = torch.load(
            descriptor["absolute_path"], map_location="cpu", weights_only=False
        )
        if (
            checkpoint.get("trajectory_id") != descriptor["trajectory_id"]
            or int(checkpoint.get("step", -1)) != descriptor["step"]
        ):
            raise ValueError("checkpoint identity fields disagree with its path")
        recorded = np.asarray(checkpoint["recorder"]["energies"], dtype=np.float64)[-1]
        if recorded.shape != (len(map_ids),):
            raise ValueError("checkpoint recorder has an unexpected map count")
        checkpoint_inputs.append(
            {
                key: value
                for key, value in descriptor.items()
                if key != "absolute_path"
            }
        )
        for device_label in arguments.devices:
            model = models[device_label]
            model.load_state_dict(checkpoint["model"], strict=True)
            model.eval()
            captured = capture(
                torch,
                model,
                probes[device_label],
                specification["context_length"],
            )
            before, output, gamma, bias, epsilons = map_arrays(
                captured,
                specification["contexts"],
                specification["layers"],
                specification["heads"],
            )
            measured = evaluate_maps(
                before,
                output,
                gamma,
                bias,
                epsilons,
                map_ids,
                arguments.precision_bits,
            )
            direct = np.asarray(measured["direct_energy"], dtype=np.float64)
            measured.update(
                {
                    "run_id": descriptor["run_id"],
                    "evidence_role": descriptor["evidence_role"],
                    "trajectory_id": descriptor["trajectory_id"],
                    "step": descriptor["step"],
                    "device": device_label,
                    "producing_device": descriptor["producing_device"],
                    "recorded_energy_bitwise_match_count": int(
                        sum(a.hex() == b.hex() for a, b in zip(direct, recorded))
                    ),
                    "recorded_represented_zero_count": int(np.count_nonzero(recorded == 0.0)),
                    "layer_gain_norms": [
                        float(np.linalg.norm(captured[layer][2].double().cpu().numpy()))
                        for layer in range(specification["layers"])
                    ],
                    "layer_bias_norms": [
                        float(np.linalg.norm(captured[layer][3].double().cpu().numpy()))
                        for layer in range(specification["layers"])
                    ],
                }
            )
            evaluations.append(measured)
        del checkpoint

    grouped = {}
    for evaluation in evaluations:
        key = (evaluation["run_id"], evaluation["trajectory_id"], evaluation["step"])
        grouped.setdefault(key, {})[evaluation["device"]] = evaluation
    cross_device = []
    device_dependent_zero_count = 0
    all_device_bitwise_equal_count = 0
    for (run_id, trajectory_id, step), by_device in sorted(grouped.items()):
        energies = [
            np.asarray(by_device[label]["direct_energy"], dtype=np.float64)
            for label in arguments.devices
        ]
        zero_status = np.stack([value == 0.0 for value in energies])
        dependent = np.any(zero_status != zero_status[0:1], axis=0)
        equal = np.asarray(
            [
                len({float(values[index]).hex() for values in energies}) == 1
                for index in range(len(map_ids))
            ]
        )
        device_dependent_zero_count += int(dependent.sum())
        all_device_bitwise_equal_count += int(equal.sum())
        cross_device.append(
            {
                "run_id": run_id,
                "trajectory_id": trajectory_id,
                "step": step,
                "represented_zero_count_by_device": {
                    label: int((values == 0.0).sum())
                    for label, values in zip(arguments.devices, energies)
                },
                "zero_status_device_dependent_count": int(dependent.sum()),
                "all_device_bitwise_equal_energy_count": int(equal.sum()),
            }
        )

    completed_producing = []
    for evaluation in evaluations:
        if evaluation["evidence_role"] != "COMPLETED_DECISION_RECORD":
            continue
        if evaluation["device"] == f"cuda:{evaluation['producing_device']}":
            completed_producing.append(evaluation)
    completed_zero_rows = [
        row
        for evaluation in completed_producing
        for row in evaluation["represented_zero_rows"]
    ]
    class_counts = {
        name: sum(row["enclosure_classification"] == name for row in completed_zero_rows)
        for name in (
            "CERTIFIED_EXACT_FACE",
            "CERTIFIED_POSITIVE",
            "OBSERVER_NEAR_FACE",
        )
    }
    positive_reference = [
        row["float64_factorized_energy"]
        for row in completed_zero_rows
        if row["enclosure_classification"] == "CERTIFIED_POSITIVE"
    ]
    completed_metadata = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(completed_root.glob("trajectories/*/metadata.json"))
    ]
    old_tolerance = 2e-6
    old_gate_failures = sum(
        float(row["capture"]["gate_shape_maximum_quotient_energy_symmetric_relative_residual"])
        > old_tolerance
        for row in completed_metadata
    )

    record = content_record(
        {
            "schema_version": "pldr-row-rg-observer-precision-audit-v4",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "precision_bits": arguments.precision_bits,
            "arithmetic_policy": {
                "native_model_dtype": "float32",
                "recorder_energy_dtype": "float64",
                "factorized_route_dtype": "float64",
                "exact_input_enclosure": "binary-rational inputs with dyadic square-root bounds",
                "tf32_enabled": False,
            },
            "devices": resolved_devices,
            "architecture": specification,
            "map_ids": map_ids,
            "inputs": {
                "runs": [
                    {
                        "run_id": config["run_id"],
                        "evidence_role": role,
                        "config_sha256": file_sha256(run_root / "config-resolved.json"),
                    }
                    for run_root, role, config in runs
                ],
                "checkpoints": checkpoint_inputs,
                "dataset_shard": {
                    "path": str(shard),
                    "sha256": file_sha256(shard),
                },
                "probe_document_offset": probe_offset,
                "probe_byte_token_sha256": probe_digest,
                "model_source": {
                    "path": "pldr_model_v510.py",
                    "sha256": file_sha256(reference / "pldr_model_v510.py"),
                },
                "attention_source": {
                    "path": "power_law_attention_layer_v510.py",
                    "sha256": file_sha256(reference / "power_law_attention_layer_v510.py"),
                },
            },
            "evaluations": evaluations,
            "cross_device_checkpoint_summary": cross_device,
            "summary": {
                "checkpoint_count": len(checkpoints),
                "completed_checkpoint_count": sum(
                    row["evidence_role"] == "COMPLETED_DECISION_RECORD"
                    for row in checkpoints
                ),
                "device_map_evaluation_count": len(evaluations) * len(map_ids),
                "device_dependent_zero_status_count": device_dependent_zero_count,
                "all_device_bitwise_equal_energy_count": all_device_bitwise_equal_count,
                "completed_producing_device_represented_zero_count": len(completed_zero_rows),
                "completed_producing_device_zero_classification_counts": class_counts,
                "completed_positive_reference_energy_minimum": min(positive_reference)
                if positive_reference
                else None,
                "completed_positive_reference_energy_maximum": max(positive_reference)
                if positive_reference
                else None,
                "completed_old_quotient_relative_gate_tolerance": old_tolerance,
                "completed_paths_failing_old_quotient_relative_gate": old_gate_failures,
                "completed_paths_passing_layernorm_reconstruction_gate": sum(
                    float(row["capture"]["gate_shape_maximum_map_backward_relative_residual"])
                    <= 2e-6
                    for row in completed_metadata
                ),
                "interpretation": {
                    "integrity_incomplete_run_used_for_flow_inference": False,
                    "native_represented_zero_implies_real_face": False,
                    "certified_exact_face_requires_enclosure_upper_zero": True,
                },
            },
        }
    )
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
                **record["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
