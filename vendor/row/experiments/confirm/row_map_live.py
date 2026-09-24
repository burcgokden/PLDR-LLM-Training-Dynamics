#!/usr/bin/env python3
"""Live adapters for the archive-bound row-map confirmation campaign.

Geometry is measured on all retained w8 checkpoints.  Complete AdamW state
exists at the 24,000-update checkpoints, which own real-loss qualification,
automatic differentiation of the 192-dimensional gate successor, and short
paired continuations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, hessian, jacrev, jvp, vjp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from gate_shape import adamw_gate_block, normalized_shape  # noqa: E402
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    sha256_path,
    write_json_atomic,
    write_npz_atomic,
)
from pldr_model_v510 import PLDR_Model  # noqa: E402
from row_map_dynamics import (  # noqa: E402
    dimensionless_state_scale,
    graph_contrast,
    relative_matrix_residual,
)


SCHEMA = "pldr-row-map-live-v1"
REGISTRY_SCHEMA = "pldr-row-map-registry-v1"
RUN_NAMES = (
    "w8-const-lr7.5e-4",
    "w8-const-lr7.5e-4-s2",
    "w8-const-lr7.5e-4-s3",
)
CHECKPOINT_FILES = (
    "ckpt_1000.pt", "ckpt_2000.pt", "ckpt_4000.pt",
    "ckpt_8000.pt", "ckpt_16000.pt", "ckpt_final.pt",
)


def _resource_record(device, started):
    elapsed = time.monotonic() - started
    cuda = device.type == "cuda"
    return {
        "elapsed_seconds": elapsed,
        "gpu_device_seconds": elapsed if cuda else 0.0,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if cuda else 0),
        "peak_gpu_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if cuda else 0),
        "peak_host_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1024 if sys.platform != "darwin" else 1),
    }


def _load_raw_checkpoint(
    path, *, require_optimizer=False, device="cpu",
    allow_restart_successor=False,
):
    checkpoint_path = Path(path)
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False)
    required = {"step", "model", "data_offset", "data_offset_end", "config"}
    if not isinstance(payload, dict) or not required.issubset(payload):
        raise ValueError("checkpoint is not a supported PLDR w8 archive")
    if require_optimizer and "opt" not in payload:
        raise ValueError("this stage requires archived AdamW moments")
    config = payload["config"]
    expected = {
        "layers": 3, "heads": 4, "dk": 64, "adff": 170,
        "ctx": 256, "batch": 32, "optimizer": "adamw",
        "lr": 7.5e-4, "warmup": 250, "const_lr": True,
        "wd": 0.1, "clip": 1.0,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"checkpoint configuration disagrees at {key}")
    registered_step = int(payload["step"]) in {
        1000, 2000, 4000, 8000, 16000, 24000}
    continuation = payload.get("confirmation", {})
    continuation_step = (
        continuation.get("schema_version") == "pldr-row-map-continuation-v1"
        and 24000 <= int(payload["step"]) <= 24016)
    restart_successor = (
        allow_restart_successor
        and int(payload["step"]) in {1001, 2001, 4001, 8001, 16001, 24001}
    )
    if not registered_step and not continuation_step and not restart_successor:
        raise ValueError("checkpoint step is outside the registered time grid")
    payload["content_sha256"] = sha256_path(checkpoint_path)
    payload["source_path"] = str(checkpoint_path)
    return payload


def _model_from_raw(checkpoint, device, *, dtype=None):
    config = checkpoint["config"]
    embedding = checkpoint["model"].get("decoder.embedding.weight")
    if embedding is None or embedding.ndim != 2:
        raise ValueError("checkpoint embedding cannot determine vocabulary")
    width = int(config["heads"]) * int(config["dk"])
    feed_forward_width = int(math.ceil(width * 4 * 2 / 3))
    model = PLDR_Model(
        num_layers=int(config["layers"]),
        d_model=width,
        num_heads=int(config["heads"]),
        dff=feed_forward_width,
        input_vocab_size=int(embedding.shape[0]),
        A_dff=int(config["adff"]),
        num_reslayerA=8,
        num_denseA=2,
        max_seq_len=4096,
        device=device,
    )
    if sum(parameter.numel() for parameter in model.parameters()) != 20_622_914:
        raise ValueError("checkpoint model does not have the registered size")
    model.load_state_dict(checkpoint["model"])
    model.to(device=device)
    if dtype is not None:
        model.to(dtype=dtype)
    model.eval()
    return model


def _causal_mask(length, device, dtype):
    return torch.triu(
        torch.ones(length, length, device=device, dtype=dtype), diagonal=1,
    )[None, None]


def _loss(model, rows):
    inputs = rows[:, :-1]
    targets = rows[:, 1:]
    mask = _causal_mask(inputs.shape[1], inputs.device, next(model.parameters()).dtype)
    logits = model([inputs, mask])[0]
    keep = targets != 0
    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
        reduction="none").reshape_as(targets)
    return torch.sum(per_token * keep.to(per_token.dtype)) / torch.sum(keep)


def _gate_parameter_name(layer_index):
    return (
        f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."
        "7.layernormA.weight"
    )


def _gate_bias_name(layer_index):
    return _gate_parameter_name(layer_index).removesuffix("weight") + "bias"


def _token_chunks(tokens_path, indices, context_length, device):
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    chunk_count = len(tokens) // context_length
    matrix = tokens[:chunk_count * context_length].reshape(
        chunk_count, context_length)
    if any(index < 0 or index >= chunk_count for index in indices):
        raise ValueError("a registered context leaves the token archive")
    return [
        torch.from_numpy(matrix[index].astype(np.int64, copy=True))[None].to(device)
        for index in indices
    ]


def _next_training_batch(checkpoint, tokens_path, device):
    config = checkpoint["config"]
    context = int(config["ctx"])
    batch = int(config["batch"])
    start = int(checkpoint["data_offset_end"])
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    chunks = len(tokens) // context
    matrix = tokens[:chunks * context].reshape(chunks, context)
    if start < 0 or start + batch > chunks:
        raise ValueError("checkpoint next minibatch leaves the token archive")
    return torch.from_numpy(
        matrix[start:start + batch].astype(np.int64, copy=True)).to(device)


def create_registry(arguments):
    tokens = Path(arguments.tokens)
    tokenizer = Path(arguments.tokenizer)
    context = 256
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    chunk_count = len(data) // context
    construction_count = int(arguments.construction_contexts)
    validation_count = int(arguments.validation_contexts)
    if construction_count < 1 or validation_count < 1:
        raise ValueError("both context partitions must be nonempty")
    required = construction_count + validation_count
    start = int(arguments.reserved_start)
    if start < 0:
        start = chunk_count - required - 128
    if start < 0 or start + required > chunk_count:
        raise ValueError("context registry leaves the token archive")
    construction = list(range(start, start + construction_count))
    validation = list(range(
        start + construction_count, start + required))
    application_pairs = []
    for role, chunks in (
            ("construction", construction), ("validation", validation)):
        unique_pairs = [
            (chunks[left], chunks[right])
            for left in range(len(chunks))
            for right in range(left + 1, len(chunks))
        ]
        if len(unique_pairs) < 12:
            raise ValueError("each PLGA role needs twelve unique context pairs")
        for layer in range(3):
            for head in range(4):
                pair_index = layer * 4 + head
                left_chunk, right_chunk = unique_pairs[pair_index]
                application_pairs.append({
                    "id": f"plga-{role[0]}-{pair_index:02d}",
                    "ownership": role,
                    "layer": layer,
                    "head": head,
                    "left_chunk": left_chunk,
                    "right_chunk": right_chunk,
                })
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "context_length": context,
        "construction_chunks": construction,
        "validation_chunks": validation,
        "application_pairs": application_pairs,
        "graph_rule": "unit-weight-chain-on-flattened-registered-rows-v1",
        "checkpoint_steps": [1000, 2000, 4000, 8000, 16000, 24000],
        "construction_seed": 1234,
        "validation_seeds": [2222, 3333],
        "tokens_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    registry["registry_sha256"] = digest_object(registry)
    write_json_atomic(arguments.output, registry)


def _load_registry(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    if registry.get("schema_version") != REGISTRY_SCHEMA:
        raise ValueError("unknown row-map registry schema")
    unsigned = dict(registry)
    recorded = unsigned.pop("registry_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("row-map registry digest does not replay")
    construction = set(registry["construction_chunks"])
    validation = set(registry["validation_chunks"])
    if (
        len(construction) != 8
        or len(validation) != 16
        or construction & validation
    ):
        raise ValueError("context partitions must be fixed and disjoint")
    if (
        registry.get("context_length") != 256
        or registry.get("graph_rule")
        != "unit-weight-chain-on-flattened-registered-rows-v1"
        or registry.get("checkpoint_steps")
        != [1000, 2000, 4000, 8000, 16000, 24000]
        or registry.get("construction_seed") != 1234
        or registry.get("validation_seeds") != [2222, 3333]
    ):
        raise ValueError("registry constants disagree with the protocol")
    pairs = registry.get("application_pairs")
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise ValueError("registry needs exactly twenty-four PLGA pairs")
    if len({row.get("id") for row in pairs}) != 24:
        raise ValueError("PLGA pair identifiers must be unique")
    for role, allowed in (
            ("construction", construction), ("validation", validation)):
        owned = [row for row in pairs if row.get("ownership") == role]
        if len(owned) != 12 or {
            (row.get("layer"), row.get("head")) for row in owned
        } != {(layer, head) for layer in range(3) for head in range(4)}:
            raise ValueError("PLGA layer-head ownership is incomplete")
        context_pairs = []
        for row in owned:
            left = row.get("left_chunk")
            right = row.get("right_chunk")
            if left not in allowed or right not in allowed or left == right:
                raise ValueError("a PLGA pair violates context ownership")
            context_pairs.append(tuple(sorted((left, right))))
        if len(set(context_pairs)) != 12:
            raise ValueError("PLGA context pairs must be unique within a role")
    return registry


def inventory(arguments):
    root = Path(arguments.run_root)
    records = []
    for run_name in RUN_NAMES:
        run_root = root / run_name
        for filename in CHECKPOINT_FILES:
            path = run_root / filename
            if not path.is_file():
                records.append({
                    "run": run_name, "filename": filename,
                    "status": "MISSING",
                })
                continue
            checkpoint = _load_raw_checkpoint(path)
            expected_step = {
                "ckpt_1000.pt": 1000,
                "ckpt_2000.pt": 2000,
                "ckpt_4000.pt": 4000,
                "ckpt_8000.pt": 8000,
                "ckpt_16000.pt": 16000,
                "ckpt_final.pt": 24000,
            }[filename]
            expected_seed = {
                "w8-const-lr7.5e-4": 1234,
                "w8-const-lr7.5e-4-s2": 2222,
                "w8-const-lr7.5e-4-s3": 3333,
            }[run_name]
            if (
                int(checkpoint["step"]) != expected_step
                or checkpoint["config"].get("name") != run_name
                or int(checkpoint["config"].get("seed", -1)) != expected_seed
            ):
                raise ValueError(
                    "checkpoint filename, run identity, step, or seed disagrees")
            records.append({
                "run": run_name,
                "filename": filename,
                "status": "READY",
                "step": int(checkpoint["step"]),
                "sha256": checkpoint["content_sha256"],
                "bytes": path.stat().st_size,
                "has_optimizer_state": "opt" in checkpoint,
                "data_offset_end": int(checkpoint["data_offset_end"]),
                "seed": int(checkpoint["config"]["seed"]),
                "vocabulary_size": int(
                    checkpoint["model"]["decoder.embedding.weight"].shape[0]),
            })
    ready = all(row["status"] == "READY" for row in records)
    final_optimizer = all(
        row["has_optimizer_state"]
        for row in records if row.get("step") == 24000)
    intermediate_optimizer_absent = all(
        not row["has_optimizer_state"]
        for row in records if row.get("step") in {1000, 2000, 4000, 8000, 16000})
    payload = {
        "schema_version": "pldr-row-map-inventory-v1",
        "ready": ready and final_optimizer and intermediate_optimizer_absent,
        "geometry_nodes_ready": ready,
        "final_dynamics_nodes_ready": final_optimizer,
        "intermediate_optimizer_state_declared_absent": (
            intermediate_optimizer_absent),
        "records": records,
    }
    payload["inventory_sha256"] = digest_object(payload)
    write_json_atomic(arguments.output, payload)


def _capture_gate_rows(model, token_batches, layer_index):
    decoder_layer = model.decoder.dec_layers[layer_index]
    blocks = decoder_layer.mha1.reslayerAs
    physical_rows = []
    preactivations = []
    outputs = []

    def physical_hook(_module, inputs):
        physical_rows.append(inputs[0][0].detach())

    def final_hook(_module, inputs, output):
        preactivations.append(inputs[0].detach())
        outputs.append(output.detach())

    first_handle = blocks[0].register_forward_pre_hook(physical_hook)
    final_handle = blocks[-1].layernormA.register_forward_hook(final_hook)
    try:
        for tokens in token_batches:
            inputs = tokens[:, :-1]
            mask = _causal_mask(
                inputs.shape[1], inputs.device, next(model.parameters()).dtype)
            with torch.no_grad():
                model([inputs, mask])
    finally:
        first_handle.remove()
        final_handle.remove()
    if not (
        len(physical_rows) == len(preactivations) == len(outputs)
        == len(token_batches)
    ):
        raise RuntimeError("row-map hooks did not fire once per context")

    def flatten(values):
        return torch.cat([
            value.reshape(-1, value.shape[-1]) for value in values])

    return flatten(physical_rows), flatten(preactivations), flatten(outputs)


def _row_map_shape_function(model, layer_index, physical_rows, names, values):
    mapping = dict(zip(names, values))
    blocks = model.decoder.dec_layers[layer_index].mha1.reslayerAs
    prefix = f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."
    current = physical_rows
    for block_index, block in enumerate(blocks):
        local_prefix = f"{prefix}{block_index}."
        residual = current
        for dense_index, dense in enumerate(block.denseAs):
            dense_prefix = f"{local_prefix}denseAs.{dense_index}."
            local = {
                name[len(dense_prefix):]: parameter
                for name, parameter in mapping.items()
                if name.startswith(dense_prefix)
            }
            current = functional_call(dense, local, (current,), strict=False)
        preactivation = current + residual
        if block_index == len(blocks) - 1:
            centered = preactivation - preactivation.mean(dim=-1, keepdim=True)
            return centered / torch.sqrt(
                block.layernormA.eps
                + torch.mean(centered * centered, dim=-1, keepdim=True))
        norm_prefix = f"{local_prefix}layernormA."
        local = {
            name[len(norm_prefix):]: parameter
            for name, parameter in mapping.items()
            if name.startswith(norm_prefix)
        }
        current = functional_call(
            block.layernormA, local, (preactivation,), strict=False)
    raise RuntimeError("row-map shape function has no final residual unit")


def _shape_directional(
        source_model, target_model, layer_index,
        source_physical_rows, target_physical_rows):
    prefix = f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."
    excluded = {_gate_parameter_name(layer_index), _gate_bias_name(layer_index)}
    source_named = dict(source_model.named_parameters())
    target_named = dict(target_model.named_parameters())
    names = tuple(
        name for name in source_named
        if name.startswith(prefix) and name not in excluded)
    values = tuple(source_named[name] for name in names)
    directions = tuple(target_named[name] - source_named[name] for name in names)

    if source_physical_rows.shape != target_physical_rows.shape:
        raise ValueError("source and target physical-row registries disagree")

    def shape_function(local_physical_rows, *parameters):
        return _row_map_shape_function(
            source_model, layer_index, local_physical_rows, names, parameters)

    shape, directional = jvp(
        shape_function,
        (source_physical_rows, *values),
        (target_physical_rows - source_physical_rows, *directions))
    return shape.detach(), directional.detach()


def _gate_loss_components(model, rows, gate_name):
    gate = dict(model.named_parameters())[gate_name]
    rows = rows.to(gate.device)
    inputs = rows[:, :-1]
    targets = rows[:, 1:]
    mask = _causal_mask(inputs.shape[1], gate.device, gate.dtype)
    keep = targets != 0
    source_count = int(torch.sum(keep).item())
    if source_count <= 0:
        raise ValueError("a loss microbatch has no nonpadding target")

    def logits_for_gate(local_gate):
        return functional_call(
            model, {gate_name: local_gate}, ([inputs, mask],), strict=False)[0]

    def loss_for_gate(local_gate):
        logits = logits_for_gate(local_gate)
        per_token = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1),
            reduction="none").reshape_as(targets)
        return torch.sum(
            per_token * keep.to(per_token.dtype)) / source_count

    return logits_for_gate, loss_for_gate, source_count


def _gate_microbatches(rows):
    if rows.ndim != 2 or len(rows) < 1:
        raise ValueError("gate derivatives need a nonempty row batch")
    return [rows[index:index + 1] for index in range(len(rows))]


def _weighted_gate_gradient_function(
        model, token_batches, layer_index):
    gate_name = _gate_parameter_name(layer_index)
    terms = [
        _gate_loss_components(model, rows, gate_name)
        for rows in token_batches
    ]
    total = sum(item[2] for item in terms)
    gradients = [grad(item[1]) for item in terms]
    weights = [item[2] / total for item in terms]

    def gradient_function(local_gate):
        result = torch.zeros_like(local_gate)
        for weight, gradient_term in zip(weights, gradients):
            result = result + weight * gradient_term(local_gate)
        return result

    return gradient_function


def _gate_loss_derivatives(model, token_batches, layer_index):
    gate_name = _gate_parameter_name(layer_index)
    gate = dict(model.named_parameters())[gate_name]
    loss_terms = []
    gradient_terms = []
    true_terms = []
    source_counts = []
    for rows in token_batches:
        _logits, loss_for_gate, source_count = _gate_loss_components(
            model, rows, gate_name)
        loss_terms.append(loss_for_gate(gate).detach())
        gradient_terms.append(grad(loss_for_gate)(gate).detach())
        true_terms.append(hessian(loss_for_gate)(gate).detach())
        source_counts.append(source_count)
    loss_terms = torch.stack(loss_terms)
    gradient_terms = torch.stack(gradient_terms)
    true_terms = torch.stack(true_terms)
    counts = torch.as_tensor(
        source_counts, device=gate.device, dtype=gate.dtype)
    weights = counts / torch.sum(counts)
    loss_value = torch.sum(weights * loss_terms)
    gradient_value = torch.sum(weights[:, None] * gradient_terms, dim=0)
    true = torch.sum(weights[:, None, None] * true_terms, dim=0)
    true = 0.5 * (true + true.T)
    return {
        "loss_value": loss_value,
        "gradient_at_gate": gradient_value,
        "true_hessian": true,
        "loss_terms": loss_terms,
        "gradient_at_gate_terms": gradient_terms,
        "true_hessian_terms": true_terms,
        "source_count_terms": counts,
    }


def _dense_loss_sector(model, token_batches, layer_index):
    gate_name = _gate_parameter_name(layer_index)
    gate = dict(model.named_parameters())[gate_name]
    dimension = gate.numel()
    loss_terms = []
    gradient_terms = []
    fisher_terms = []
    true_terms = []
    force_terms = []
    source_counts = []
    for rows in token_batches:
        logits_for_gate, loss_for_gate, source_count = _gate_loss_components(
            model, rows, gate_name)
        loss_terms.append(loss_for_gate(gate).detach())
        gradient_terms.append(grad(loss_for_gate)(gate).detach())
        true_term = hessian(loss_for_gate)(gate).detach()
        force_term = grad(loss_for_gate)(torch.zeros_like(gate)).detach()
        logits, pullback = vjp(logits_for_gate, gate)
        probability = torch.softmax(logits.detach(), dim=-1)
        targets = rows[:, 1:].to(gate.device)
        keep = (targets != 0).to(gate.dtype)
        columns = []
        for index in range(dimension):
            direction = torch.zeros_like(gate)
            direction[index] = 1.0
            _, action = jvp(logits_for_gate, (gate,), (direction,))
            mean = torch.sum(probability * action, dim=-1, keepdim=True)
            cotangent = probability * (action - mean)
            cotangent = cotangent * keep[..., None] / source_count
            columns.append(pullback(cotangent)[0].detach())
        fisher_terms.append(torch.column_stack(columns))
        true_terms.append(true_term)
        force_terms.append(force_term)
        source_counts.append(source_count)
    loss_terms = torch.stack(loss_terms)
    gradient_terms = torch.stack(gradient_terms)
    fisher_terms = torch.stack(fisher_terms)
    true_terms = torch.stack(true_terms)
    force_terms = torch.stack(force_terms)
    counts = torch.as_tensor(
        source_counts, device=gate.device, dtype=gate.dtype)
    weights = counts / torch.sum(counts)
    fisher = torch.sum(weights[:, None, None] * fisher_terms, dim=0)
    true = torch.sum(weights[:, None, None] * true_terms, dim=0)
    fisher = 0.5 * (fisher + fisher.T)
    true = 0.5 * (true + true.T)
    return {
        "loss_value": torch.sum(weights * loss_terms),
        "gradient_at_gate": torch.sum(
            weights[:, None] * gradient_terms, dim=0),
        "fisher": fisher,
        "true_hessian": true,
        "signed_second_jet": true - fisher,
        "force_at_zero": torch.sum(weights[:, None] * force_terms, dim=0),
        "loss_terms": loss_terms,
        "gradient_at_gate_terms": gradient_terms,
        "fisher_terms": fisher_terms,
        "true_hessian_terms": true_terms,
        "force_at_zero_terms": force_terms,
        "source_count_terms": counts,
    }

def geometry(arguments):
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    checkpoint = _load_raw_checkpoint(arguments.checkpoint)
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["tokens_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    model = _model_from_raw(checkpoint, device, dtype=torch.float64)
    role_key = f"{arguments.role}_chunks"
    chunks = registry[role_key]
    if arguments.context_limit:
        chunks = chunks[:int(arguments.context_limit)]
    batches = _token_chunks(
        arguments.tokens, chunks, registry["context_length"], device)
    layer_index = int(arguments.layer)
    physical, preactivation, output = _capture_gate_rows(
        model, batches, layer_index)
    final_norm = model.decoder.dec_layers[
        layer_index].mha1.reslayerAs[-1].layernormA
    gate = final_norm.weight.detach().cpu().numpy()
    bias = final_norm.bias.detach().cpu().numpy()
    preactivation_array = preactivation.detach().cpu().numpy()
    shape = normalized_shape(
        preactivation_array, epsilon=float(final_norm.eps))
    edge = np.column_stack([
        np.arange(len(shape) - 1, dtype=np.int64),
        np.arange(1, len(shape), dtype=np.int64),
    ])
    weights = np.ones(len(edge), dtype=np.float64)
    graph = graph_contrast(gate, shape, edge, weights)
    factorization = bias + gate * shape
    factorization_residual = float(np.linalg.norm(
        output.detach().cpu().numpy() - factorization))
    shape_directional = np.full_like(shape, np.nan)
    target_shape = np.full_like(shape, np.nan)
    shape_remainder = np.full_like(shape, np.nan)
    target_sha256 = ""
    if arguments.target_checkpoint:
        target = _load_raw_checkpoint(arguments.target_checkpoint)
        if target["config"] != checkpoint["config"]:
            raise ValueError("source and target checkpoint configs disagree")
        target_model = _model_from_raw(target, device, dtype=torch.float64)
        target_physical, target_preactivation, _target_output = _capture_gate_rows(
            target_model, batches, layer_index)
        source_shape, directional = _shape_directional(
            model, target_model, layer_index, physical, target_physical)
        if not torch.allclose(
                source_shape, torch.as_tensor(shape, device=device),
                rtol=1e-9, atol=1e-11):
            raise ArithmeticError("shape JVP source does not replay")
        shape_directional = directional.cpu().numpy()
        target_shape = normalized_shape(
            target_preactivation.cpu().numpy(), epsilon=float(final_norm.eps))
        shape_remainder = target_shape - shape - shape_directional
        target_sha256 = target["content_sha256"]
    loss_sector = None
    if arguments.loss_sector:
        loss_sector = _dense_loss_sector(model, batches, layer_index)
    payload = {
        "schema_version": np.asarray(SCHEMA),
        "stage": np.asarray("G"),
        "checkpoint_sha256": np.asarray(checkpoint["content_sha256"]),
        "target_checkpoint_sha256": np.asarray(target_sha256),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "tokens_sha256": np.asarray(registry["tokens_sha256"]),
        "role": np.asarray(arguments.role),
        "run_name": np.asarray(checkpoint["config"]["name"]),
        "seed": np.asarray(checkpoint["config"]["seed"], dtype=np.int64),
        "step": np.asarray(checkpoint["step"], dtype=np.int64),
        "layer": np.asarray(layer_index, dtype=np.int64),
        "context_chunks": np.asarray(chunks, dtype=np.int64),
        "gamma": gate,
        "beta": bias,
        "layernorm_epsilon": np.asarray(float(final_norm.eps)),
        "physical_rows": physical.detach().cpu().numpy(),
        "final_preactivation": preactivation_array,
        "normalized_shape": shape,
        "shape_directional": shape_directional,
        "target_normalized_shape": target_shape,
        "shape_remainder": shape_remainder,
        "row_output": output.detach().cpu().numpy(),
        "graph_edges": edge,
        "graph_weights": weights,
        "q": graph["q"],
        "energy": np.asarray(graph["energy"]),
        "graph_diameter": np.asarray(graph["graph_diameter"]),
        "minimum_weight": np.asarray(graph["minimum_weight"]),
        "vertex_diameter": np.asarray(graph["vertex_diameter"]),
        "graph_diameter_bound": np.asarray(graph["diameter_bound"]),
        "cover_radius": np.asarray(0.0),
        "cover_lipschitz_bound": np.asarray(0.0),
        "covered_domain_bound": np.asarray(graph["diameter_bound"]),
        "energy_identity_residual": np.asarray(
            graph["energy_identity_residual"]),
        "factorization_residual": np.asarray(factorization_residual),
    }
    if loss_sector is not None:
        for name, value in loss_sector.items():
            payload[name] = value.detach().cpu().numpy()
    for name, value in _resource_record(device, started).items():
        payload[f"resource_{name}"] = np.asarray(value)
    write_npz_atomic(arguments.output, **payload)


def _optimizer_and_gate(checkpoint, model, layer_index):
    config = checkpoint["config"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), betas=(0.9, 0.95),
        eps=1e-5, weight_decay=float(config["wd"]))
    optimizer.load_state_dict(checkpoint["opt"])
    named = dict(model.named_parameters())
    gate = named[_gate_parameter_name(layer_index)]
    state = optimizer.state.get(gate)
    if not state or not {"step", "exp_avg", "exp_avg_sq"}.issubset(state):
        raise ValueError("optimizer archive omits the selected final gate state")
    return optimizer, gate, state


def optimizer_block(arguments):
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    checkpoint = _load_raw_checkpoint(
        arguments.checkpoint, require_optimizer=True)
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["tokens_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    model = _model_from_raw(checkpoint, device, dtype=torch.float64)
    layer_index = int(arguments.layer)
    optimizer, gate_parameter, state = _optimizer_and_gate(
        checkpoint, model, layer_index)
    rows = _next_training_batch(checkpoint, arguments.tokens, device)
    gate_name = _gate_parameter_name(layer_index)
    base_gate = gate_parameter.detach()
    first = state["exp_avg"].detach()
    second = state["exp_avg_sq"].detach()
    group = optimizer.param_groups[0]
    beta1, beta2 = map(float, group["betas"])
    epsilon = float(group["eps"])
    eta = float(group["lr"])
    decay = float(group["weight_decay"])
    step_before = int(state["step"].item())
    step_after = step_before + 1
    microbatches = _gate_microbatches(rows)
    gradient_function = _weighted_gate_gradient_function(
        model, microbatches, layer_index)
    qualification_sector = None
    if arguments.qualification:
        # Q uses exactly the archived run's next training minibatch. The
        # one-row microbatches are combined with their nonpadding counts, so
        # this is algebraically the same objective with bounded peak memory.
        qualification_sector = _dense_loss_sector(
            model, microbatches, layer_index)
        raw_gradient = qualification_sector["gradient_at_gate"]
        true_hessian = qualification_sector["true_hessian"]
    else:
        derivatives = _gate_loss_derivatives(
            model, microbatches, layer_index)
        raw_gradient = derivatives["gradient_at_gate"]
        true_hessian = derivatives["true_hessian"]
    clip_value = float(checkpoint["config"]["clip"])
    optimizer_arguments = {
        "learning_rate": eta,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "optimizer_step": step_after,
        "weight_decay": decay,
        "clip_value": clip_value,
    }
    analytic = adamw_gate_block(
        true_hessian.cpu().numpy(), base_gate.cpu().numpy(),
        first.cpu().numpy(), second.cpu().numpy(), raw_gradient.cpu().numpy(),
        boundary_tolerance=float(arguments.boundary_tolerance),
        **optimizer_arguments)
    dimension = base_gate.numel()
    source_state = torch.cat([base_gate, first, second])

    def complete_successor(vector):
        local_gate = vector[:dimension]
        local_first = vector[dimension:2 * dimension]
        local_second = vector[2 * dimension:]
        raw = gradient_function(local_gate)
        clipped = torch.clamp(raw, -clip_value, clip_value)
        first_next = beta1 * local_first + (1.0 - beta1) * clipped
        second_next = beta2 * local_second + (1.0 - beta2) * clipped * clipped
        first_hat = first_next / (1.0 - beta1 ** step_after)
        second_hat = second_next / (1.0 - beta2 ** step_after)
        gate_next = (
            (1.0 - eta * decay) * local_gate
            - eta * first_hat / (torch.sqrt(second_hat) + epsilon))
        return torch.cat([gate_next, first_next, second_next])

    independent = jacrev(
        complete_successor, chunk_size=1)(source_state).detach().cpu().numpy()
    analytic_operator = analytic["operator"]
    residual = relative_matrix_residual(analytic_operator, independent)
    successor = complete_successor(source_state).detach()
    source_second_hat = (
        second / (1.0 - beta2 ** step_before)).cpu().numpy()
    second_hat = (
        successor[2 * dimension:]
        / (1.0 - beta2 ** step_after)).cpu().numpy()
    source_scale = dimensionless_state_scale(
        source_second_hat, epsilon=epsilon)
    target_scale = dimensionless_state_scale(second_hat, epsilon=epsilon)
    signed_norm = float(np.linalg.norm(true_hessian.cpu().numpy(), ord=2))
    payload = dict(
        schema_version=np.asarray(SCHEMA),
        stage=np.asarray("Q" if arguments.qualification else "D-block"),
        checkpoint_sha256=np.asarray(checkpoint["content_sha256"]),
        run_name=np.asarray(checkpoint["config"]["name"]),
        seed=np.asarray(checkpoint["config"]["seed"], dtype=np.int64),
        registry_sha256=np.asarray(registry["registry_sha256"]),
        tokens_sha256=np.asarray(registry["tokens_sha256"]),
        device=np.asarray(str(device)),
        step=np.asarray(checkpoint["step"], dtype=np.int64),
        layer=np.asarray(layer_index, dtype=np.int64),
        gamma=base_gate.cpu().numpy(),
        first_moment=first.cpu().numpy(),
        second_moment=second.cpu().numpy(),
        raw_gradient=raw_gradient.cpu().numpy(),
        true_hessian=true_hessian.cpu().numpy(),
        learning_rate=np.asarray(eta),
        beta1=np.asarray(beta1),
        beta2=np.asarray(beta2),
        adam_epsilon=np.asarray(epsilon),
        weight_decay=np.asarray(decay),
        clip_value=np.asarray(clip_value),
        optimizer_step_before=np.asarray(step_before, dtype=np.int64),
        optimizer_step_after=np.asarray(step_after, dtype=np.int64),
        analytic_operator=analytic_operator,
        autodiff_operator=independent,
        relative_operator_residual=np.asarray(residual),
        clipping_boundary_distance=np.asarray(
            np.min(np.abs(np.abs(raw_gradient.cpu().numpy()) - clip_value))),
        successor_state=successor.cpu().numpy(),
        source_state_scale=source_scale,
        target_state_scale=target_scale,
        true_hessian_operator_norm=np.asarray(signed_norm),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in _resource_record(device, started).items()
        },
    )
    if qualification_sector is not None:
        for name, value in qualification_sector.items():
            key = "sector_true_hessian" if name == "true_hessian" else name
            payload[key] = value.detach().cpu().numpy()
    write_npz_atomic(arguments.output, **payload)


def _source_normalized_shape(source_model, rows, layer_index):
    module = source_model.decoder.dec_layers[
        layer_index].mha1.reslayerAs[-1].layernormA
    captured = []

    def hook(_module, inputs, _output):
        value = inputs[0]
        centered = value - value.mean(dim=-1, keepdim=True)
        captured.append((centered / torch.sqrt(
            module.eps + torch.mean(
                centered * centered, dim=-1, keepdim=True))).detach())

    handle = module.register_forward_hook(hook)
    try:
        with torch.no_grad():
            _loss(source_model, rows)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("source shape hook did not fire exactly once")
    return captured[0]


def _install_shape_clamp(model, layer_index, source_shape):
    module = model.decoder.dec_layers[
        layer_index].mha1.reslayerAs[-1].layernormA

    def hook(local_module, _inputs, _output):
        if source_shape.shape != _output.shape:
            raise ValueError("shape clamp source and target rows disagree")
        return local_module.bias + local_module.weight * source_shape

    return module.register_forward_hook(hook)


def _install_shape_observer(model, layer_index, forced_shape=None):
    module = model.decoder.dec_layers[
        layer_index].mha1.reslayerAs[-1].layernormA
    captured = []

    def hook(local_module, inputs, output):
        value = inputs[0]
        centered = value - value.mean(dim=-1, keepdim=True)
        actual = centered / torch.sqrt(
            local_module.eps
            + torch.mean(centered * centered, dim=-1, keepdim=True))
        effective = actual if forced_shape is None else forced_shape
        if effective.shape != output.shape:
            raise ValueError("observed shape and LayerNorm output disagree")
        captured.append(effective.detach())
        if forced_shape is not None:
            return local_module.bias + local_module.weight * effective

    return module.register_forward_hook(hook), captured


def _shape_metric(shape):
    flattened = shape.detach().reshape(-1, shape.shape[-1])
    if len(flattened) < 2:
        raise ValueError("shape metric needs at least two physical rows")
    difference = flattened[1:] - flattened[:-1]
    return torch.sum(difference * difference, dim=0)


def _shape_metric_flux(shape, directional, target_shape):
    source_flat = shape.detach().reshape(-1, shape.shape[-1])
    direction_flat = directional.detach().reshape(-1, directional.shape[-1])
    target_flat = target_shape.detach().reshape(-1, target_shape.shape[-1])
    source_difference = source_flat[1:] - source_flat[:-1]
    direction_difference = direction_flat[1:] - direction_flat[:-1]
    target_difference = target_flat[1:] - target_flat[:-1]
    source_q = torch.sum(source_difference * source_difference, dim=0)
    target_q = torch.sum(target_difference * target_difference, dim=0)
    directional_q = 2.0 * torch.sum(
        source_difference * direction_difference, dim=0)
    return source_q, target_q, directional_q, (
        target_q - source_q - directional_q)


def _save_raw_node(path, model, optimizer, checkpoint, step, cursor, arm):
    payload = {
        "step": int(step),
        "model": model.state_dict(),
        "opt": optimizer.state_dict(),
        "data_offset": int(checkpoint["data_offset_end"]),
        "data_offset_end": int(cursor),
        "config": dict(checkpoint["config"]),
        "confirmation": {
            "schema_version": "pldr-row-map-continuation-v1",
            "source_checkpoint_sha256": checkpoint["content_sha256"],
            "arm": arm,
        },
    }
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def continue_training(arguments):
    checkpoint = _load_raw_checkpoint(
        arguments.checkpoint, require_optimizer=True)
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["tokens_sha256"]:
        raise ValueError("token archive disagrees with the registry")
    arm = arguments.arm
    if arguments.capture_blocks and arm != "baseline_continue":
        raise ValueError("chronological blocks are captured on baseline only")
    device = torch.device(arguments.device)
    model = _model_from_raw(checkpoint, device)
    model.train()
    config = checkpoint["config"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), betas=(0.9, 0.95),
        eps=1e-5, weight_decay=float(config["wd"]))
    optimizer.load_state_dict(checkpoint["opt"])
    layer_index = int(arguments.layer)
    gate = dict(model.named_parameters())[_gate_parameter_name(layer_index)]
    gate_source = gate.detach().clone()
    gate_state_source = {
        name: value.detach().clone()
        for name, value in optimizer.state[gate].items()
        if torch.is_tensor(value)
    }
    source_model = None
    if arm == "freeze_normalized_shape":
        source_model = _model_from_raw(checkpoint, device)
        source_model.eval()
    linearization_model = None
    if arguments.shape_jvp:
        linearization_model = _model_from_raw(checkpoint, device)
        linearization_model.eval()
    tokens = np.memmap(arguments.tokens, dtype=np.uint16, mode="r")
    context = int(config["ctx"])
    chunks = len(tokens) // context
    matrix = tokens[:chunks * context].reshape(chunks, context)
    batch_size = int(config["batch"])
    source_cursor = int(checkpoint["data_offset_end"])
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    observable_rows = []
    source_path = output / "node_00000.pt"
    _save_raw_node(
        source_path, model, optimizer, checkpoint, checkpoint["step"],
        source_cursor, arm)
    records.append({
        "node": 0, "step": int(checkpoint["step"]),
        "checkpoint": source_path.name, "sha256": sha256_path(source_path),
    })
    for local_step in range(int(arguments.updates)):
        start = source_cursor + local_step * batch_size
        if start + batch_size > chunks:
            raise ValueError("continuation leaves the token archive")
        rows = torch.from_numpy(
            matrix[start:start + batch_size].astype(np.int64, copy=True)).to(device)
        jvp_source_physical = None
        if linearization_model is not None:
            linearization_model.load_state_dict(model.state_dict())
            jvp_source_physical, jvp_source_preactivation, _ = _capture_gate_rows(
                linearization_model, [rows], layer_index)
        block_derivatives = None
        block_first_before = None
        block_second_before = None
        block_step_before = None
        if arguments.capture_blocks:
            block_derivatives = _gate_loss_derivatives(
                model, _gate_microbatches(rows), layer_index)
            block_state = optimizer.state[gate]
            block_first_before = block_state["exp_avg"].detach().clone()
            block_second_before = block_state["exp_avg_sq"].detach().clone()
            block_step_before = int(block_state["step"].item())
        optimizer.zero_grad(set_to_none=True)
        forced_shape = None
        if source_model is not None:
            forced_shape = _source_normalized_shape(
                source_model, rows, layer_index)
        gate_before = gate.detach().clone()
        observer, captured_before = _install_shape_observer(
            model, layer_index, forced_shape)
        try:
            loss = _loss(model, rows)
        finally:
            observer.remove()
        if len(captured_before) != 1:
            raise RuntimeError("training shape observer did not fire once")
        q_before = _shape_metric(captured_before[0])
        energy_before = torch.sum(q_before * gate_before * gate_before)
        loss.backward()
        measured_raw_gradient = gate.grad.detach().clone()
        torch.nn.utils.clip_grad_value_(
            model.parameters(), clip_value=float(config["clip"]))
        optimizer.step()
        optimizer_gate_after = gate.detach().clone()
        group = optimizer.param_groups[0]
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        if learning_rate <= 0.0:
            raise ValueError("continuation learning rate must be positive")
        decay_multiplier = 1.0 - learning_rate * weight_decay
        adaptive_direction = (
            decay_multiplier * gate_before - optimizer_gate_after
        ) / learning_rate
        block_payload = {}
        if block_derivatives is not None:
            beta1, beta2 = map(float, group["betas"])
            adam_epsilon = float(group["eps"])
            clip_value = float(config["clip"])
            block_state_after = optimizer.state[gate]
            block_step_after = int(block_state_after["step"].item())
            if block_step_after != block_step_before + 1:
                raise ArithmeticError("AdamW clock did not advance by one")
            raw_gradient = block_derivatives["gradient_at_gate"]
            true_hessian = block_derivatives["true_hessian"]
            raw_scale = max(
                float(torch.linalg.vector_norm(raw_gradient).cpu()),
                float(torch.linalg.vector_norm(measured_raw_gradient).cpu()))
            gradient_replay_residual = (
                0.0 if raw_scale == 0.0 else float(
                    torch.linalg.vector_norm(
                        raw_gradient - measured_raw_gradient).cpu()) / raw_scale)
            analytic = adamw_gate_block(
                true_hessian.detach().cpu().numpy(),
                gate_before.cpu().numpy(),
                block_first_before.cpu().numpy(),
                block_second_before.cpu().numpy(),
                raw_gradient.detach().cpu().numpy(),
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=adam_epsilon,
                optimizer_step=block_step_after,
                weight_decay=weight_decay,
                clip_value=clip_value,
                boundary_tolerance=0.0,
            )
            source_second_hat = (
                block_second_before
                / (1.0 - beta2 ** block_step_before)).cpu().numpy()
            target_second_hat = (
                block_state_after["exp_avg_sq"].detach()
                / (1.0 - beta2 ** block_step_after)).cpu().numpy()
            block_payload = {
                "block_first_moment": block_first_before.cpu().numpy(),
                "block_second_moment": block_second_before.cpu().numpy(),
                "block_raw_gradient": raw_gradient.detach().cpu().numpy(),
                "block_true_hessian": true_hessian.detach().cpu().numpy(),
                "block_operator": analytic["operator"],
                "block_source_state_scale": dimensionless_state_scale(
                    source_second_hat, epsilon=adam_epsilon),
                "block_target_state_scale": dimensionless_state_scale(
                    target_second_hat, epsilon=adam_epsilon),
                "block_beta1": beta1,
                "block_beta2": beta2,
                "block_adam_epsilon": adam_epsilon,
                "block_clip_value": clip_value,
                "block_optimizer_step_after": block_step_after,
                "block_gradient_replay_residual": gradient_replay_residual,
                "block_clipping_boundary_distance": float(np.min(np.abs(
                    np.abs(raw_gradient.detach().cpu().numpy())
                    - clip_value))),
            }
        if arm == "freeze_final_gate":
            with torch.no_grad():
                gate.copy_(gate_source)
                for name, value in gate_state_source.items():
                    optimizer.state[gate][name].copy_(value)
        elif arm == "disable_final_gate_decay":
            with torch.no_grad():
                gate.add_(float(config["lr"]) * float(config["wd"]) * gate_before)
        elif arm not in {
                "baseline_continue", "freeze_normalized_shape",
                "replay_physical_rows"}:
            raise ValueError("unknown continuation arm")
        gate_after = gate.detach().clone()
        observer, captured_after = _install_shape_observer(
            model, layer_index, forced_shape)
        try:
            with torch.no_grad():
                _loss(model, rows)
        finally:
            observer.remove()
        if len(captured_after) != 1:
            raise RuntimeError("successor shape observer did not fire once")
        q_after = _shape_metric(captured_after[0])
        q_directional = torch.full_like(q_before, torch.nan)
        q_remainder = torch.full_like(q_before, torch.nan)
        shape_jvp_residual = math.nan
        if linearization_model is not None:
            target_physical, target_preactivation, _ = _capture_gate_rows(
                model, [rows], layer_index)
            source_shape, directional_shape = _shape_directional(
                linearization_model, model, layer_index,
                jvp_source_physical, target_physical)
            target_shape = torch.as_tensor(
                normalized_shape(
                    target_preactivation.detach().cpu().numpy(),
                    epsilon=float(model.decoder.dec_layers[
                        layer_index].mha1.reslayerAs[-1].layernormA.eps)),
                device=device, dtype=source_shape.dtype)
            _, _, q_directional, q_remainder = _shape_metric_flux(
                source_shape, directional_shape, target_shape)
            source_shape_direct = torch.as_tensor(
                normalized_shape(
                    jvp_source_preactivation.detach().cpu().numpy(),
                    epsilon=float(linearization_model.decoder.dec_layers[
                        layer_index].mha1.reslayerAs[-1].layernormA.eps)),
                device=device, dtype=source_shape.dtype)
            source_scale = max(
                float(torch.linalg.vector_norm(source_shape).cpu()),
                float(torch.linalg.vector_norm(source_shape_direct).cpu()))
            shape_jvp_residual = (
                0.0 if source_scale == 0.0 else float(
                    torch.linalg.vector_norm(
                        source_shape - source_shape_direct).cpu())
                / source_scale)
        fixed_q_energy = torch.sum(q_before * gate_after * gate_after)
        energy_after = torch.sum(q_after * gate_after * gate_after)
        observable_rows.append({
            "gamma_before": gate_before.cpu().numpy(),
            "gamma_optimizer_after": optimizer_gate_after.cpu().numpy(),
            "gamma_after": gate_after.cpu().numpy(),
            "adaptive_direction": adaptive_direction.cpu().numpy(),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "q_before": q_before.cpu().numpy(),
            "q_after": q_after.cpu().numpy(),
            "energy_before": float(energy_before.cpu()),
            "energy_after": float(energy_after.cpu()),
            "gate_work": float((fixed_q_energy - energy_before).cpu()),
            "shape_work": float((energy_after - fixed_q_energy).cpu()),
            "q_directional": q_directional.cpu().numpy(),
            "q_remainder": q_remainder.cpu().numpy(),
            "shape_jvp_source_residual": shape_jvp_residual,
            "loss": float(loss.detach().cpu()),
            **block_payload,
        })
        node = local_step + 1
        path = output / f"node_{node:05d}.pt"
        cursor = source_cursor + node * batch_size
        _save_raw_node(
            path, model, optimizer, checkpoint,
            int(checkpoint["step"]) + node, cursor, arm)
        records.append({
            "node": node,
            "step": int(checkpoint["step"]) + node,
            "checkpoint": path.name,
            "sha256": sha256_path(path),
            "loss": float(loss.detach().cpu()),
        })
    observables_path = output / "observables.npz"
    observable_payload = {
        "schema_version": np.asarray(
            "pldr-row-map-intervention-observables-v2"),
        "arm": np.asarray(arm),
        "run_name": np.asarray(checkpoint["config"]["name"]),
        "seed": np.asarray(checkpoint["config"]["seed"], dtype=np.int64),
        "layer": np.asarray(layer_index, dtype=np.int64),
        "source_checkpoint_sha256": np.asarray(
            checkpoint["content_sha256"]),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "tokens_sha256": np.asarray(registry["tokens_sha256"]),
        "capture_blocks": np.asarray(bool(arguments.capture_blocks)),
        "local_step": np.arange(
            1, len(observable_rows) + 1, dtype=np.int64),
    }
    common_fields = (
        "gamma_before", "gamma_optimizer_after", "gamma_after",
        "adaptive_direction", "learning_rate", "weight_decay",
        "q_before", "q_after", "energy_before", "energy_after",
        "gate_work", "shape_work", "q_directional", "q_remainder",
        "shape_jvp_source_residual", "loss",
    )
    for name in common_fields:
        observable_payload[name] = np.asarray([
            row[name] for row in observable_rows])
    if arguments.capture_blocks:
        block_fields = (
            "block_first_moment", "block_second_moment",
            "block_raw_gradient", "block_true_hessian", "block_operator",
            "block_source_state_scale", "block_target_state_scale",
            "block_beta1", "block_beta2", "block_adam_epsilon",
            "block_clip_value", "block_optimizer_step_after",
            "block_gradient_replay_residual",
            "block_clipping_boundary_distance",
        )
        for name in block_fields:
            observable_payload[name] = np.asarray([
                row[name] for row in observable_rows])
    write_npz_atomic(observables_path, **observable_payload)
    manifest = {
        "schema_version": "pldr-row-map-continuation-manifest-v1",
        "arm": arm,
        "layer": layer_index,
        "source_checkpoint_sha256": checkpoint["content_sha256"],
        "registry_sha256": registry["registry_sha256"],
        "tokens_sha256": registry["tokens_sha256"],
        "updates": int(arguments.updates),
        "capture_blocks": bool(arguments.capture_blocks),
        "records": records,
        "observables": {
            "path": observables_path.name,
            "sha256": sha256_path(observables_path),
        },
    }
    manifest["manifest_sha256"] = digest_object(manifest)
    write_json_atomic(output / "manifest.json", manifest)


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    command = subparsers.add_parser("inventory")
    command.add_argument("--run-root", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=inventory)

    command = subparsers.add_parser("registry")
    command.add_argument("--tokens", required=True)
    command.add_argument("--tokenizer", required=True)
    command.add_argument("--construction-contexts", type=int, default=8)
    command.add_argument("--validation-contexts", type=int, default=16)
    command.add_argument("--reserved-start", type=int, default=-1)
    command.add_argument("--output", required=True)
    command.set_defaults(function=create_registry)

    command = subparsers.add_parser("geometry")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--target-checkpoint", default="")
    command.add_argument("--tokens", required=True)
    command.add_argument("--registry", required=True)
    command.add_argument("--role", choices=("construction", "validation"), required=True)
    command.add_argument("--layer", type=int, choices=(0, 1, 2), required=True)
    command.add_argument("--context-limit", type=int, default=0)
    command.add_argument("--loss-sector", action="store_true")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--output", required=True)
    command.set_defaults(function=geometry)

    command = subparsers.add_parser("optimizer-block")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--tokens", required=True)
    command.add_argument("--registry", required=True)
    command.add_argument("--layer", type=int, choices=(0, 1, 2), required=True)
    command.add_argument("--boundary-tolerance", type=float, default=1e-10)
    command.add_argument("--qualification", action="store_true")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--output", required=True)
    command.set_defaults(function=optimizer_block)

    command = subparsers.add_parser("continue")
    command.add_argument("--checkpoint", required=True)
    command.add_argument("--tokens", required=True)
    command.add_argument("--registry", required=True)
    command.add_argument("--arm", choices=(
        "baseline_continue", "freeze_final_gate",
        "disable_final_gate_decay", "freeze_normalized_shape",
        "replay_physical_rows"), required=True)
    command.add_argument("--layer", type=int, choices=(0, 1, 2), required=True)
    command.add_argument("--updates", type=int, default=16)
    command.add_argument("--shape-jvp", action="store_true")
    command.add_argument("--capture-blocks", action="store_true")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--output", required=True)
    command.set_defaults(function=continue_training)
    return parser


def main():
    arguments = _parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
