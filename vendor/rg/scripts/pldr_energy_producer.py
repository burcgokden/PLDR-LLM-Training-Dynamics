#!/usr/bin/env python3
"""Produce consecutive PLDR row-map energies on one isolated GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-device", default="cuda:0")
    parser.add_argument("--physical-device", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--reference-experiments", required=True)
    parser.add_argument("--dataset-shard", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dk", type=int, default=64)
    parser.add_argument("--contexts", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--train-document-offset", type=int, default=0)
    parser.add_argument("--train-document-stride", type=int, default=4096)
    parser.add_argument("--probe-document-offset", type=int, default=20000)
    parser.add_argument("--max-sequence-length", type=int, default=128)
    return parser.parse_args()


def _byte_tokens(dataset, start: int, count: int) -> tuple[list[int], dict]:
    tokens: list[int] = []
    documents = 0
    index = start
    while len(tokens) < count:
        if index >= len(dataset):
            raise ValueError("dataset shard ended before the requested byte stream")
        content = dataset[index]["content"]
        if not isinstance(content, str):
            raise ValueError(f"dataset row {index} has no string content")
        encoded = content.encode("utf-8", errors="strict")
        tokens.extend(byte + 1 for byte in encoded)
        tokens.append(11)
        documents += 1
        index += 1
    selected = tokens[:count]
    array_bytes = bytes((token - 1) % 256 for token in selected)
    return selected, {
        "first_document": start,
        "last_document_exclusive": index,
        "document_count": documents,
        "token_count": count,
        "byte_token_sha256": hashlib.sha256(array_bytes).hexdigest(),
    }


def _center(value):
    return value - value.mean(dim=0, keepdim=True)


def _plga_map(torch, functional, layer, head: int, value):
    plga = layer.mha1.plgatt_layer
    weight = plga.Wlst[head].detach().double()
    bias = plga.blst[head].detach().double()
    power = plga.pwlst[head].detach().double()
    coupling = plga.alst[head].detach().double()
    coupling_bias = plga.balst[head].detach().double()
    z = weight @ value + bias
    base = z * functional.silu(z) + 1e-9
    return coupling @ torch.pow(base, power) + coupling_bias


def _plga_directional_derivative(torch, functional, layer, head, value, direction):
    plga = layer.mha1.plgatt_layer
    weight = plga.Wlst[head].detach().double()
    bias = plga.blst[head].detach().double()
    power = plga.pwlst[head].detach().double()
    coupling = plga.alst[head].detach().double()
    z = weight @ value + bias
    sigmoid = torch.sigmoid(z)
    silu = functional.silu(z)
    phi_prime = silu + z * (sigmoid + z * sigmoid * (1.0 - sigmoid))
    base = z * silu + 1e-9
    dbase = phi_prime * (weight @ direction)
    return coupling @ (power * torch.pow(base, power - 1.0) * dbase)


def _summary(numpy, values) -> dict:
    array = numpy.asarray(values, dtype=numpy.float64)
    return {
        "count": int(len(array)),
        "minimum": float(array.min()),
        "median": float(numpy.median(array)),
        "p95": float(numpy.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def _plga_interventions(torch, numpy, functional, model, row_maps: dict) -> dict:
    rows = []
    segment_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
    layers = model.decoder.dec_layers
    for map_id, matrix in row_maps.items():
        fields = map_id.replace("c", "").replace("L", "").replace("H", "").split(".")
        context, layer_index, head = (int(value) for value in fields)
        del context
        value = matrix.detach().double()
        quotient = _center(value)
        constant = value.mean(dim=0, keepdim=True).expand_as(value)
        output = _plga_map(torch, functional, layers[layer_index], head, value)
        constant_output = _plga_map(
            torch, functional, layers[layer_index], head, constant
        )
        output_quotient = _center(output)
        defect = _center(constant_output)
        response = _center(output - constant_output)
        input_norm = float(torch.linalg.vector_norm(quotient))
        output_norm = float(torch.linalg.vector_norm(output_quotient))
        defect_norm = float(torch.linalg.vector_norm(defect))
        response_norm = float(torch.linalg.vector_norm(response))
        directional_gains = []
        if input_norm > 0.0:
            for fraction in segment_grid:
                point = constant + fraction * quotient
                derivative = _plga_directional_derivative(
                    torch, functional, layers[layer_index], head, point, quotient
                )
                directional_gains.append(
                    float(torch.linalg.vector_norm(_center(derivative))) / input_norm
                )
        values = [
            input_norm, output_norm, defect_norm, response_norm, *directional_gains
        ]
        if not all(math.isfinite(item) for item in values):
            raise ValueError(f"non-finite PLGA intervention at {map_id}")
        decomposition_residual = float(
            torch.linalg.vector_norm(output_quotient - response - defect)
        ) / max(output_norm, 1e-300)
        rows.append(
            {
                "map_id": map_id,
                "layer": layer_index,
                "head": head,
                "input_quotient_norm": input_norm,
                "output_quotient_norm": output_norm,
                "constant_input_defect_norm": defect_norm,
                "realized_response_norm": response_norm,
                "defect_to_output_norm_ratio": defect_norm / max(output_norm, 1e-300),
                "realized_secant_gain": response_norm / max(input_norm, 1e-300),
                "maximum_sampled_directional_gain": (
                    max(directional_gains) if directional_gains else None
                ),
                "segment_grid": list(segment_grid),
                "decomposition_relative_residual": decomposition_residual,
            }
        )
    return {
        "schema_version": "pldr-plga-intervention-v1",
        "map_count": len(rows),
        "rows": rows,
        "constant_input_defect_norm": _summary(
            numpy, [row["constant_input_defect_norm"] for row in rows]
        ),
        "defect_to_output_norm_ratio": _summary(
            numpy, [row["defect_to_output_norm_ratio"] for row in rows]
        ),
        "realized_secant_gain": _summary(
            numpy, [row["realized_secant_gain"] for row in rows]
        ),
        "maximum_sampled_directional_gain": _summary(
            numpy, [row["maximum_sampled_directional_gain"] for row in rows]
        ),
        "maximum_decomposition_relative_residual": max(
            row["decomposition_relative_residual"] for row in rows
        ),
    }


def main() -> None:
    arguments = _parse()
    started_at = _utc_now()
    wall_start = time.perf_counter()
    output = Path(arguments.output)
    metadata_output = Path(arguments.metadata_output)
    reference = Path(arguments.reference_experiments).resolve()
    shard = Path(arguments.dataset_shard).resolve()
    if output.suffix != ".npz" or metadata_output.suffix != ".json":
        raise ValueError("producer outputs must use .npz and .json suffixes")
    if output.exists() or metadata_output.exists():
        raise FileExistsError("producer refuses to overwrite an output")
    if arguments.steps < 2:
        raise ValueError("steps must be at least two")
    if arguments.dk % 2:
        raise ValueError("dk must be even for rotary embeddings")
    if arguments.context_length + 1 > arguments.max_sequence_length:
        raise ValueError("context length exceeds the rotary embedding registry")
    if arguments.train_document_offset < 0 or arguments.probe_document_offset < 0:
        raise ValueError("document offsets must be nonnegative")
    model_source = reference / "pldr_model_v510.py"
    attention_source = reference / "power_law_attention_layer_v510.py"
    if not model_source.is_file() or not attention_source.is_file():
        raise FileNotFoundError("reference PLDR sources are missing")
    if not shard.is_file():
        raise FileNotFoundError(shard)

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from datasets import Dataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"producer requires exactly one visible GPU, saw {torch.cuda.device_count()}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(arguments.physical_device):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r}, expected {arguments.physical_device}"
        )

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(reference))
    from row_rgmap.recorder import RowEnergyRecorder, row_centered_energy
    from pldr_model_v510 import PLDR_Model

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = Dataset.from_file(str(shard))
    sequence_width = arguments.context_length + 1
    train_offset = (
        arguments.train_document_offset
        + arguments.physical_device * arguments.train_document_stride
    )
    train_count = arguments.steps * arguments.batch_size * sequence_width
    probe_count = arguments.contexts * sequence_width
    train_values, train_data = _byte_tokens(dataset, train_offset, train_count)
    probe_values, probe_data = _byte_tokens(
        dataset, arguments.probe_document_offset, probe_count
    )
    train = torch.tensor(train_values, dtype=torch.long).reshape(
        arguments.steps, arguments.batch_size, sequence_width
    )
    probe = torch.tensor(probe_values, dtype=torch.long).reshape(
        arguments.contexts, sequence_width
    )

    device = torch.device(arguments.logical_device)
    d_model = arguments.heads * arguments.dk
    model = PLDR_Model(
        num_layers=arguments.layers,
        d_model=d_model,
        num_heads=arguments.heads,
        dff=int(2.7 * d_model),
        input_vocab_size=257,
        A_dff=2 * arguments.dk,
        num_reslayerA=2,
        num_denseA=2,
        max_seq_len=arguments.max_sequence_length,
        device=device,
    ).to(device=device, dtype=torch.float32)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-5,
        weight_decay=0.1,
    )
    mask = torch.triu(
        torch.ones(
            arguments.context_length,
            arguments.context_length,
            device=device,
            dtype=torch.float32,
        ),
        diagonal=1,
    )[None, None]
    train_device = train.to(device)
    probe_device = probe.to(device)

    layernorms = [
        model.decoder.dec_layers[index].mha1.reslayerAs[-1].layernormA
        for index in range(arguments.layers)
    ]
    captured = {}

    def make_hook(layer_index):
        def hook(module, inputs, result):
            captured[layer_index] = (
                inputs[0].detach(),
                result.detach(),
                module.weight.detach(),
                float(module.eps),
            )
        return hook

    map_ids = [
        f"c{context}.L{layer}.H{head}"
        for context in range(arguments.contexts)
        for layer in range(arguments.layers)
        for head in range(arguments.heads)
    ]
    recorder = RowEnergyRecorder(map_ids)
    factorization_maximum = 0.0
    hook_identity_checks = 0
    exact_zero_energies = 0
    minimum_energy = math.inf
    maximum_energy = 0.0
    final_rows = None

    def observe(step: int):
        nonlocal factorization_maximum, hook_identity_checks
        nonlocal exact_zero_energies, minimum_energy, maximum_energy, final_rows
        model.eval()
        handles = [
            layernorm.register_forward_hook(make_hook(layer_index))
            for layer_index, layernorm in enumerate(layernorms)
        ]
        try:
            with torch.no_grad():
                _, _, _, cache = model([probe_device[:, :-1], mask])
        finally:
            for handle in handles:
                handle.remove()
        rows = {}
        for context in range(arguments.contexts):
            for layer_index in range(arguments.layers):
                for head in range(arguments.heads):
                    rows[f"c{context}.L{layer_index}.H{head}"] = (
                        cache[layer_index][2][context, head]
                    )
        if step in (0, arguments.steps):
            for layer_index in range(arguments.layers):
                if not torch.equal(captured[layer_index][1], cache[layer_index][2]):
                    raise RuntimeError("forward hook does not match cached row map")
                before, after, gamma, epsilon = captured[layer_index]
                centered = before - before.mean(dim=-1, keepdim=True)
                shape = centered / torch.sqrt(
                    epsilon + (centered * centered).mean(dim=-1, keepdim=True)
                )
                for context in range(arguments.contexts):
                    for head in range(arguments.heads):
                        normalized = shape[context, head].double()
                        centered_normalized = _center(normalized)
                        shape_energy = (centered_normalized ** 2).sum(dim=0)
                        predicted = float(((gamma.double() ** 2) * shape_energy).sum())
                        direct = row_centered_energy(
                            cache[layer_index][2][context, head]
                        )
                        relative = abs(predicted - direct) / max(direct, 1e-300)
                        factorization_maximum = max(
                            factorization_maximum, relative
                        )
                hook_identity_checks += 1
        recorder.record(step, rows)
        energies = np.asarray(
            [row_centered_energy(value) for value in rows.values()],
            dtype=np.float64,
        )
        exact_zero_energies += int(np.sum(energies == 0.0))
        minimum_energy = min(minimum_energy, float(energies.min()))
        maximum_energy = max(maximum_energy, float(energies.max()))
        if step == arguments.steps:
            final_rows = rows
        return energies

    initial_energies = observe(0)
    losses = []
    model.train()
    for update in range(1, arguments.steps + 1):
        batch = train_device[update - 1]
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits, _, _, _ = model([inputs, mask])
        loss = functional.cross_entropy(logits.reshape(-1, 257), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at update {update}")
        losses.append(float(loss.detach()))
        energies = observe(update)
        model.train()
        if update == 1 or update % 256 == 0 or update == arguments.steps:
            print(
                json.dumps(
                    {
                        "update": update,
                        "loss": losses[-1],
                        "energy_minimum": float(energies.min()),
                        "energy_median": float(np.median(energies)),
                        "energy_maximum": float(energies.max()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if factorization_maximum > 2e-6:
        raise RuntimeError(
            f"gate-shape identity residual {factorization_maximum:.3e} exceeds 2e-6"
        )
    if final_rows is None:
        raise RuntimeError("final row-map observation was not captured")
    plga = _plga_interventions(
        torch, np, functional, model, final_rows
    )
    torch.cuda.synchronize()
    recorder.write_npz(
        output,
        trajectory_id=(
            f"pldr-refinedweb-byte-seed{arguments.seed}-gpu"
            f"{arguments.physical_device}"
        ),
    )
    finished_at = _utc_now()
    wall_seconds = time.perf_counter() - wall_start
    metadata = {
        "schema_version": "pldr-row-rg-energy-producer-metadata-v2",
        "started_at_utc": started_at,
        "finished_at_utc": finished_at,
        "wall_seconds": wall_seconds,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "logical_device": arguments.logical_device,
        "physical_device": arguments.physical_device,
        "cuda_visible_devices": visible,
        "visible_gpu_count": torch.cuda.device_count(),
        "seed": arguments.seed,
        "model": {
            "implementation": "PLDR_Model from pldr_model_v510.py",
            "layers": arguments.layers,
            "heads": arguments.heads,
            "dk": arguments.dk,
            "d_model": d_model,
            "contexts": arguments.contexts,
            "context_length": arguments.context_length,
            "vocabulary": 257,
            "vocabulary_encoding": "UTF-8 byte value plus one; zero reserved",
            "parameter_count": parameter_count,
            "layernorm_epsilon": 1e-6,
            "plga_base_epsilon": 1e-9,
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": arguments.learning_rate,
            "betas": [0.9, 0.95],
            "epsilon": 1e-5,
            "weight_decay": 0.1,
            "gradient_clip": "elementwise value 1.0",
            "schedule": "constant",
            "updates": arguments.steps,
        },
        "data": {
            "dataset": "tiiuae/falcon-refinedweb local read-only cache",
            "shard_path": str(shard),
            "shard_sha256": _sha256(shard),
            "training": train_data,
            "probe": probe_data,
        },
        "sources": {
            "producer_path": str(Path(__file__).resolve()),
            "producer_sha256": _sha256(Path(__file__).resolve()),
            "reference_model_path": str(model_source),
            "reference_model_sha256": _sha256(model_source),
            "reference_attention_path": str(attention_source),
            "reference_attention_sha256": _sha256(attention_source),
        },
        "capture": {
            "trajectory_id": (
                f"pldr-refinedweb-byte-seed{arguments.seed}-gpu"
                f"{arguments.physical_device}"
            ),
            "map_ids": map_ids,
            "states": arguments.steps + 1,
            "native_energy_dtype": "float64",
            "exact_zero_energy_count": exact_zero_energies,
            "minimum_energy": minimum_energy,
            "maximum_energy": maximum_energy,
            "initial_energy_summary": _summary(np, initial_energies),
            "hook_identity_checks": hook_identity_checks,
            "gate_shape_maximum_relative_residual": factorization_maximum,
        },
        "loss": {
            "initial_update_loss": losses[0],
            "final_update_loss": losses[-1],
            "minimum": min(losses),
            "median": float(np.median(losses)),
            "maximum": max(losses),
        },
        "plga_interventions_at_final_state": plga,
        "peak_allocated_memory_mib": (
            torch.cuda.max_memory_allocated() / 2 ** 20
        ),
        "energy_artifact": {
            "path": str(output.resolve()),
            "sha256": _sha256(output),
            "size_bytes": output.stat().st_size,
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "metadata": str(metadata_output),
                "energy_artifact": str(output),
                "wall_seconds": wall_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
