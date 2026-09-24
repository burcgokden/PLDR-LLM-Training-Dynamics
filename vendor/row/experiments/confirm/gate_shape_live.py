#!/usr/bin/env python3
"""Checkpoint-bound producers for the gate-shape confirmation program.

The qualification path replays one CPU-realized row-map fixture on each
device.  The scientific path captures the final LayerNorm preactivation,
recomputes the exact gate-shape geometry, forms dense gate-sector loss
matrices by automatic differentiation, and builds the full 48-dimensional
AdamW derivative from operation-ordered optimizer snapshots.
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
from torch.func import functional_call, grad, hessian, jvp, vjp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirmation_artifacts import load_complete_checkpoint  # noqa: E402
from gate_shape import (  # noqa: E402
    adamw_gate_block,
    finite_difference_adamw_block,
    gate_shape_energy_increment,
    global_gate_oscillation_bound,
    normalized_shape,
    residual_scaled_spectrum,
    row_contrast_geometry,
)
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    sha256_path,
    write_json_atomic,
    write_npz_atomic,
)
from gate_shape_protocol_specs import ARCHITECTURE  # noqa: E402
from live_confirmation_producers import _model_from_checkpoint  # noqa: E402
from pldr_model_v510 import ResLayerA  # noqa: E402


PRODUCER_SCHEMA = "pldr-gate-shape-live-v2"
REGISTRY_SCHEMA = "pldr-gate-shape-registry-v1"
QUALIFICATION_SEED = 350011


def _producer_digest():
    return sha256_path(Path(__file__))


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


def _causal_mask(length, device, dtype):
    return torch.triu(
        torch.ones(length, length, device=device, dtype=dtype), diagonal=1,
    )[None, None]


def _new_row_map(*, device="cpu", dtype=torch.float64):
    blocks = torch.nn.ModuleList([
        ResLayerA(
            depth=ARCHITECTURE["head_width"],
            A_dff=ARCHITECTURE["glu_hidden_width"],
            num_denseA=ARCHITECTURE["glu_blocks_per_residual_unit"],
            device=device,
        )
        for _ in range(ARCHITECTURE["row_map_residual_units"])
    ])
    return blocks.to(device=device, dtype=dtype)


def realize_fixture(arguments):
    """Realize every random qualification tensor once on CPU."""

    torch.manual_seed(QUALIFICATION_SEED)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(QUALIFICATION_SEED + 1)
    blocks = _new_row_map(device="cpu", dtype=torch.float64)
    rows = torch.randn(
        ARCHITECTURE["context_length"], ARCHITECTURE["head_width"],
        generator=generator, dtype=torch.float64)
    projection = torch.randn(
        ARCHITECTURE["head_width"], 23,
        generator=generator, dtype=torch.float64) / math.sqrt(16.0)
    targets = torch.arange(ARCHITECTURE["context_length"], dtype=torch.int64) % 23
    arrays = {
        "schema_version": np.asarray("pldr-gate-shape-q-fixture-v1"),
        "seed": np.asarray(QUALIFICATION_SEED, dtype=np.int64),
        "rows": rows.numpy(),
        "projection": projection.numpy(),
        "targets": targets.numpy(),
    }
    for name, value in blocks.state_dict().items():
        arrays[f"state::{name}"] = value.detach().cpu().numpy()
    write_npz_atomic(arguments.output, **arrays)


def _load_fixture(path, device):
    arrays = load_npz(
        path, required=("schema_version", "seed", "rows", "projection", "targets"))
    if arrays["schema_version"].shape != () or arrays["schema_version"].item() != (
        "pldr-gate-shape-q-fixture-v1"
    ):
        raise ValueError("unknown gate-shape qualification fixture")
    blocks = _new_row_map(device=device, dtype=torch.float64)
    expected = set(blocks.state_dict())
    realized = {
        name.removeprefix("state::")
        for name in arrays if name.startswith("state::")
    }
    if realized != expected:
        raise ValueError("qualification fixture row-map state is incomplete")
    state = {
        name: torch.as_tensor(
            arrays[f"state::{name}"], device=device, dtype=torch.float64)
        for name in sorted(expected)
    }
    blocks.load_state_dict(state)
    return arrays, blocks


def _run_row_map(blocks, rows):
    value = rows
    final_preactivation = None

    def capture(_module, inputs):
        nonlocal final_preactivation
        final_preactivation = inputs[0]

    handle = blocks[-1].layernormA.register_forward_pre_hook(capture)
    try:
        for block in blocks:
            value = block([value])
    finally:
        handle.remove()
    if final_preactivation is None:
        raise RuntimeError("final LayerNorm preactivation was not captured")
    return value, final_preactivation


def qualify(arguments):
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    arrays, blocks = _load_fixture(arguments.fixture, device)
    rows = torch.as_tensor(arrays["rows"], device=device, dtype=torch.float64)
    projection = torch.as_tensor(
        arrays["projection"], device=device, dtype=torch.float64)
    targets = torch.as_tensor(arrays["targets"], device=device, dtype=torch.int64)
    output, preactivation = _run_row_map(blocks, rows)
    final_norm = blocks[-1].layernormA
    gamma = final_norm.weight.detach().clone()
    beta = final_norm.bias.detach().clone()
    shape = (
        preactivation - preactivation.mean(dim=-1, keepdim=True)
    ) / torch.sqrt(
        final_norm.eps
        + torch.mean(
            (preactivation - preactivation.mean(dim=-1, keepdim=True)) ** 2,
            dim=-1, keepdim=True))
    factorized = beta + gamma * shape
    factorization_residual = torch.linalg.vector_norm(output - factorized)
    zero_output = beta + torch.zeros_like(gamma) * shape
    zero_diameter = torch.max(torch.linalg.vector_norm(
        zero_output[:, None, :] - zero_output[None, :, :], dim=-1))
    scale = torch.tensor(-1.375, device=device, dtype=torch.float64)
    scaled_output = beta + scale * gamma * shape
    scaling_residual = torch.linalg.vector_norm(
        (scaled_output[1:] - scaled_output[:-1])
        - scale * (output[1:] - output[:-1]))

    fixed_shape = shape.detach()

    def loss_for_gate(local_gamma):
        local_output = beta + local_gamma * fixed_shape
        logits = local_output @ projection
        return F.cross_entropy(logits, targets)

    true_hessian = hessian(loss_for_gate)(gamma).detach()
    raw_gradient = grad(loss_for_gate)(gamma).detach()
    probability = torch.softmax(output.detach() @ projection, dim=-1)
    jacobian = fixed_shape[:, :, None] * projection[None, :, :]
    fisher = torch.zeros_like(true_hessian)
    for source in range(len(rows)):
        j = jacobian[source].T
        p = probability[source]
        covariance = torch.diag(p) - p[:, None] * p[None, :]
        fisher += j.T @ covariance @ j / len(rows)
    signed = true_hessian - fisher

    first = torch.linspace(0.02, 0.05, 16, device=device, dtype=torch.float64)
    second = torch.linspace(0.3, 0.6, 16, device=device, dtype=torch.float64)
    block = adamw_gate_block(
        true_hessian.cpu().numpy(), gamma.cpu().numpy(), first.cpu().numpy(),
        second.cpu().numpy(), raw_gradient.cpu().numpy(),
        learning_rate=7e-4, beta1=0.9, beta2=0.95, epsilon=1e-5,
        optimizer_step=19, weight_decay=0.1, clip_value=1.0,
        boundary_tolerance=1e-12)

    hessian_numpy = true_hessian.cpu().numpy()
    gamma_numpy = gamma.cpu().numpy()
    gradient_numpy = raw_gradient.cpu().numpy()

    def affine_gradient(position):
        return gradient_numpy + hessian_numpy @ (position - gamma_numpy)

    numerical = finite_difference_adamw_block(
        affine_gradient, gamma_numpy, first.cpu().numpy(), second.cpu().numpy(),
        step_size=2e-6, learning_rate=7e-4, beta1=0.9, beta2=0.95,
        epsilon=1e-5, optimizer_step=19, weight_decay=0.1,
        clip_value=1.0)
    resource_record = _resource_record(device, started)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(PRODUCER_SCHEMA),
        stage=np.asarray("Q"),
        producer_sha256=np.asarray(_producer_digest()),
        fixture_sha256=np.asarray(sha256_path(arguments.fixture)),
        device=np.asarray(str(device)),
        context_length=np.asarray(len(rows), dtype=np.int64),
        row_output=output.detach().cpu().numpy(),
        final_preactivation=preactivation.detach().cpu().numpy(),
        normalized_shape=shape.detach().cpu().numpy(),
        scaling_factor=np.asarray(float(scale.detach().cpu())),
        scaled_output=scaled_output.detach().cpu().numpy(),
        gamma=gamma_numpy,
        beta=beta.cpu().numpy(),
        layernorm_epsilon=np.asarray(float(final_norm.eps)),
        true_hessian=hessian_numpy,
        fisher=fisher.detach().cpu().numpy(),
        first_moment=first.cpu().numpy(),
        second_moment=second.cpu().numpy(),
        learning_rate=np.asarray(7e-4),
        beta1=np.asarray(0.9),
        beta2=np.asarray(0.95),
        adam_epsilon=np.asarray(1e-5),
        optimizer_step=np.asarray(19, dtype=np.int64),
        weight_decay=np.asarray(0.1),
        clip_value=np.asarray(1.0),
        signed_second_jet=signed.detach().cpu().numpy(),
        gate_gradient=gradient_numpy,
        adamw_operator=block["operator"],
        adamw_independent_operator=numerical,
        factorization_residual=np.asarray(float(factorization_residual.detach().cpu())),
        zero_gate_diameter=np.asarray(float(zero_diameter.detach().cpu())),
        scaling_residual=np.asarray(float(scaling_residual.detach().cpu())),
        global_gate_bound=np.asarray(global_gate_oscillation_bound(gamma_numpy)),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resource_record.items()
        },
    )


def create_registry(arguments):
    tokens = Path(arguments.tokens).resolve()
    tokenizer = Path(arguments.tokenizer).resolve()
    context = ARCHITECTURE["context_length"]
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    chunk_count = len(data) // context
    construction_count = ARCHITECTURE["construction_contexts_per_anchor"]
    validation_count = ARCHITECTURE["validation_contexts_per_anchor"]
    construction_pairs = ARCHITECTURE["construction_application_pairs"]
    validation_pairs = ARCHITECTURE["validation_application_pairs"]
    required = (
        construction_count + validation_count
        + 2 * construction_pairs + 2 * validation_pairs)
    start = int(arguments.reserved_start)
    if start < 0:
        start = chunk_count - required - 64
    if start < 0 or start + required > chunk_count:
        raise ValueError("gate-shape registry exceeds the token archive")
    cursor = start

    def contexts(prefix, count):
        nonlocal cursor
        rows = [
            {"id": f"{prefix}{index:03d}", "chunk_index": cursor + index}
            for index in range(count)
        ]
        cursor += count
        return rows

    construction = contexts("gc", construction_count)
    validation = contexts("gv", validation_count)
    application_pairs = []
    for partition, count in (
        ("construction", construction_pairs),
        ("validation", validation_pairs),
    ):
        prefix = "ga-c" if partition == "construction" else "ga-v"
        for index in range(count):
            application_pairs.append({
                "id": f"{prefix}{index:03d}",
                "left_chunk": cursor,
                "right_chunk": cursor + 1,
            })
            cursor += 2
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "context_length": context,
        "construction": construction,
        "validation": validation,
        "application_pairs": application_pairs,
        "anchors": ARCHITECTURE["anchor_source_steps"],
        "pair_rule": "adjacent-flattened-row-pairs-v1",
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    registry["registry_sha256"] = digest_object(registry)
    write_json_atomic(arguments.output, registry)


def load_registry(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    required = {
        "schema_version", "context_length", "construction", "validation",
        "application_pairs", "anchors", "pair_rule", "dataset_sha256",
        "tokenizer_sha256", "registry_sha256",
    }
    if not isinstance(registry, dict) or set(registry) != required:
        raise ValueError("gate-shape registry has an invalid schema")
    if registry["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError("unknown gate-shape registry")
    unsigned = dict(registry)
    recorded = unsigned.pop("registry_sha256")
    if recorded != digest_object(unsigned):
        raise ValueError("gate-shape registry digest does not replay")
    chunks = []
    identifiers = []
    for role in ("construction", "validation"):
        for row in registry[role]:
            if set(row) != {"id", "chunk_index"}:
                raise ValueError("gate-shape context row is malformed")
            identifiers.append(row["id"])
            chunks.append(int(row["chunk_index"]))
    for row in registry["application_pairs"]:
        if set(row) != {"id", "left_chunk", "right_chunk"}:
            raise ValueError("gate-shape application pair is malformed")
        if int(row["left_chunk"]) == int(row["right_chunk"]):
            raise ValueError("gate-shape application self-pair is forbidden")
        identifiers.append(row["id"])
        chunks.extend((int(row["left_chunk"]), int(row["right_chunk"])))
    if len(identifiers) != len(set(identifiers)) or len(chunks) != len(set(chunks)):
        raise ValueError("gate-shape registry identifiers or chunks overlap")
    return registry


def _token_batch(tokens_path, registry, row):
    context = int(registry["context_length"])
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    start = int(row["chunk_index"]) * context
    stop = start + context
    if stop > len(data):
        raise ValueError("registry token chunk lies outside the token archive")
    return torch.from_numpy(
        np.asarray(data[start:stop], dtype=np.int64).copy())[None]


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
            device = next(model.parameters()).device
            inputs = tokens[:, :-1].to(device)
            mask = _causal_mask(
                inputs.shape[1], device, next(model.parameters()).dtype)
            with torch.no_grad():
                model([inputs, mask])
    finally:
        first_handle.remove()
        final_handle.remove()
    if not (
        len(physical_rows) == len(preactivations) == len(outputs)
        == len(token_batches)
    ):
        raise RuntimeError("gate-shape hooks did not fire once per context")
    flatten = lambda values: torch.cat([  # noqa: E731
        value.reshape(-1, value.shape[-1]) for value in values])
    return flatten(physical_rows), flatten(preactivations), flatten(outputs)


def _pair_indices(row_count):
    if row_count < 2:
        raise ValueError("gate geometry needs at least two physical rows")
    usable = row_count - row_count % 2
    left = np.arange(0, usable, 2, dtype=np.int64)
    right = left + 1
    weights = np.full(len(left), 1.0 / len(left), dtype=np.float64)
    return left, right, weights


def _gate_parameter_name(model, layer_index):
    last = len(model.decoder.dec_layers[layer_index].mha1.reslayerAs) - 1
    return (
        f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."
        f"{last}.layernormA.weight")


def _dense_loss_sector(model, token_batches, layer_index):
    name = _gate_parameter_name(model, layer_index)
    named = dict(model.named_parameters())
    if name not in named:
        raise ValueError("final row-map LayerNorm gate parameter is absent")
    base_gamma = named[name].detach()
    dimension = base_gamma.numel()
    fisher_total = torch.zeros(
        dimension, dimension, device=base_gamma.device, dtype=base_gamma.dtype)
    true_total = torch.zeros_like(fisher_total)
    force_total = torch.zeros_like(base_gamma)
    fisher_terms = []
    true_terms = []
    force_terms = []
    for tokens in token_batches:
        inputs = tokens[:, :-1].to(base_gamma.device)
        targets = tokens[:, 1:].to(base_gamma.device)
        mask = _causal_mask(inputs.shape[1], base_gamma.device, base_gamma.dtype)

        def logits_for_gate(gamma_value):
            return functional_call(
                model, {name: gamma_value}, ([inputs, mask],), strict=False)[0]

        def loss_for_gate(gamma_value):
            logits = logits_for_gate(gamma_value)
            return F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

        true_term = hessian(loss_for_gate)(base_gamma).detach()
        force_term = grad(loss_for_gate)(torch.zeros_like(base_gamma)).detach()
        true_total += true_term
        force_total += force_term
        logits, pullback = vjp(logits_for_gate, base_gamma)
        probability = torch.softmax(logits.detach(), dim=-1)
        source_count = logits.shape[0] * logits.shape[1]
        columns = []
        for index in range(dimension):
            direction = torch.zeros_like(base_gamma)
            direction[index] = 1.0
            _, action = jvp(logits_for_gate, (base_gamma,), (direction,))
            mean = torch.sum(probability * action, dim=-1, keepdim=True)
            cotangent = probability * (action - mean) / source_count
            columns.append(pullback(cotangent)[0].detach())
        fisher_term = torch.column_stack(columns)
        fisher_total += fisher_term
        fisher_terms.append(fisher_term)
        true_terms.append(true_term)
        force_terms.append(force_term)
    count = len(token_batches)
    fisher = 0.5 * (fisher_total + fisher_total.T) / count
    true = 0.5 * (true_total + true_total.T) / count
    return {
        "fisher": fisher,
        "true_hessian": true,
        "signed_second_jet": true - fisher,
        "force_at_zero": force_total / count,
        "fisher_terms": torch.stack(fisher_terms),
        "true_hessian_terms": torch.stack(true_terms),
        "force_at_zero_terms": torch.stack(force_terms),
    }


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


def _shape_q_directional(
        source_model, target_model, layer_index, physical_rows,
        left_indices, right_indices, weights):
    prefix = f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."
    final_gate = _gate_parameter_name(source_model, layer_index)
    final_bias = final_gate.removesuffix("weight") + "bias"
    source_named = dict(source_model.named_parameters())
    target_named = dict(target_model.named_parameters())
    names = tuple(
        name for name in source_named
        if name.startswith(prefix) and name not in {final_gate, final_bias})
    values = tuple(source_named[name] for name in names)
    directions = tuple(target_named[name] - source_named[name] for name in names)
    left = torch.as_tensor(left_indices, device=physical_rows.device)
    right = torch.as_tensor(right_indices, device=physical_rows.device)
    weight = torch.as_tensor(weights, device=physical_rows.device,
                             dtype=physical_rows.dtype)

    def q_function(*parameters):
        shape = _row_map_shape_function(
            source_model, layer_index, physical_rows, names, parameters)
        difference = shape[left] - shape[right]
        return torch.sum(weight[:, None] * difference * difference, dim=0)

    source_q, directional = jvp(q_function, values, directions)
    return source_q.detach(), directional.detach()


def geometry(arguments):
    checkpoint = load_complete_checkpoint(arguments.checkpoint)
    registry = load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("geometry token archive disagrees with the registry")
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("checkpoint does not own the gate-shape registry")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = _model_from_checkpoint(checkpoint, device).double()
    contexts = registry[arguments.registry_role]
    if arguments.context_limit:
        contexts = contexts[:int(arguments.context_limit)]
    if not contexts:
        raise ValueError("geometry selected an empty context partition")
    token_batches = [
        _token_batch(arguments.tokens, registry, row) for row in contexts]
    layer_index = int(arguments.layer)
    physical, preactivation, output = _capture_gate_rows(
        model, token_batches, layer_index)
    final_norm = model.decoder.dec_layers[layer_index].mha1.reslayerAs[-1].layernormA
    gamma = final_norm.weight.detach().double().cpu().numpy()
    beta = final_norm.bias.detach().double().cpu().numpy()
    preactivation_numpy = preactivation.double().cpu().numpy()
    shape = normalized_shape(
        preactivation_numpy, epsilon=float(final_norm.eps))
    factorized = beta + gamma * shape
    factorization_residual = float(np.linalg.norm(
        output.double().cpu().numpy() - factorized))
    left, right, weights = _pair_indices(len(shape))
    contrast = row_contrast_geometry(
        gamma, shape[left], shape[right], weights)
    loss_sector = _dense_loss_sector(model, token_batches, layer_index)
    fisher = loss_sector["fisher"].double().cpu().numpy()
    true_hessian = loss_sector["true_hessian"].double().cpu().numpy()
    signed = loss_sector["signed_second_jet"].double().cpu().numpy()
    reconstruction_residual = float(np.linalg.norm(
        true_hessian - fisher - signed, ord=2))
    spectrum = residual_scaled_spectrum(
        true_hessian, reconstruction_residual=reconstruction_residual)
    q_directional = np.full_like(contrast["q"], np.nan)
    target_digest = ""
    if arguments.target_checkpoint:
        target_checkpoint = load_complete_checkpoint(arguments.target_checkpoint)
        if target_checkpoint["measurement_registry"] != registry:
            raise ValueError("target checkpoint changed the gate-shape registry")
        target_model = _model_from_checkpoint(target_checkpoint, device).double()
        source_q, directional = _shape_q_directional(
            model, target_model, layer_index, physical,
            left, right, weights)
        if not np.allclose(
            source_q.cpu().numpy(), contrast["q"], rtol=1e-9, atol=1e-11):
            raise ArithmeticError("source shape JVP does not replay measured q")
        q_directional = directional.cpu().numpy()
        target_digest = target_checkpoint["state_manifest"]["complete_state_sha256"]
    resource_record = _resource_record(device, started)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(PRODUCER_SCHEMA),
        stage=np.asarray("G"),
        producer_sha256=np.asarray(_producer_digest()),
        checkpoint_sha256=np.asarray(
            checkpoint["state_manifest"]["complete_state_sha256"]),
        target_checkpoint_sha256=np.asarray(target_digest),
        tokens_sha256=np.asarray(sha256_path(arguments.tokens)),
        registry_sha256=np.asarray(registry["registry_sha256"]),
        registry_role=np.asarray(arguments.registry_role),
        step=np.asarray(checkpoint["step"], dtype=np.int64),
        context_ids=np.asarray([row["id"] for row in contexts]),
        layer=np.asarray(layer_index, dtype=np.int64),
        gamma=gamma,
        beta=beta,
        layernorm_epsilon=np.asarray(float(final_norm.eps)),
        physical_rows=physical.double().cpu().numpy(),
        final_preactivation=preactivation_numpy,
        row_output=output.double().cpu().numpy(),
        normalized_shape=shape,
        pair_left=left,
        pair_right=right,
        pair_weights=weights,
        q=contrast["q"],
        energy=np.asarray(contrast["energy"]),
        q_directional=q_directional,
        fisher=fisher,
        signed_second_jet=signed,
        true_hessian=true_hessian,
        force_at_zero=loss_sector["force_at_zero"].double().cpu().numpy(),
        fisher_terms=loss_sector["fisher_terms"].double().cpu().numpy(),
        true_hessian_terms=(
            loss_sector["true_hessian_terms"].double().cpu().numpy()),
        force_at_zero_terms=(
            loss_sector["force_at_zero_terms"].double().cpu().numpy()),
        eigenvalues=spectrum["eigenvalues"],
        spectral_threshold=np.asarray(spectrum["zero_threshold"]),
        factorization_residual=np.asarray(factorization_residual),
        hessian_reconstruction_residual=np.asarray(reconstruction_residual),
        global_gate_bound=np.asarray(global_gate_oscillation_bound(gamma)),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resource_record.items()
        },
    )


def intervention_geometry(arguments):
    """Capture gate and normalized-shape energy without a loss Hessian."""

    checkpoint = load_complete_checkpoint(arguments.checkpoint)
    registry = load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("intervention token archive disagrees with the registry")
    if checkpoint["measurement_registry"] != registry:
        raise ValueError("intervention checkpoint does not own the registry")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = _model_from_checkpoint(checkpoint, device).double()
    contexts = registry["validation"]
    token_batches = [
        _token_batch(arguments.tokens, registry, row) for row in contexts]
    layer_index = int(arguments.layer)
    if not 0 <= layer_index < ARCHITECTURE["layers"]:
        raise ValueError("intervention geometry layer lies outside the model")
    _physical, preactivation, output = _capture_gate_rows(
        model, token_batches, layer_index)
    final_norm = model.decoder.dec_layers[
        layer_index].mha1.reslayerAs[-1].layernormA
    gamma = final_norm.weight.detach().double().cpu().numpy()
    beta = final_norm.bias.detach().double().cpu().numpy()
    preactivation_numpy = preactivation.double().cpu().numpy()
    shape = normalized_shape(
        preactivation_numpy, epsilon=float(final_norm.eps))
    left, right, weights = _pair_indices(len(shape))
    contrast = row_contrast_geometry(
        gamma, shape[left], shape[right], weights)
    resources = _resource_record(device, started)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(PRODUCER_SCHEMA),
        stage=np.asarray("I-geometry"),
        producer_sha256=np.asarray(_producer_digest()),
        checkpoint_sha256=np.asarray(
            checkpoint["state_manifest"]["complete_state_sha256"]),
        tokens_sha256=np.asarray(sha256_path(arguments.tokens)),
        registry_sha256=np.asarray(registry["registry_sha256"]),
        registry_role=np.asarray("validation"),
        step=np.asarray(checkpoint["step"], dtype=np.int64),
        context_ids=np.asarray([row["id"] for row in contexts]),
        layer=np.asarray(layer_index, dtype=np.int64),
        gamma=gamma,
        beta=beta,
        layernorm_epsilon=np.asarray(float(final_norm.eps)),
        final_preactivation=preactivation_numpy,
        row_output=output.double().cpu().numpy(),
        normalized_shape=shape,
        pair_left=left,
        pair_right=right,
        pair_weights=weights,
        q=contrast["q"],
        energy=np.asarray(contrast["energy"]),
        global_gate_bound=np.asarray(global_gate_oscillation_bound(gamma)),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in resources.items()
        },
    )


def capture_block(arguments):
    geometry_archive = load_npz(
        arguments.geometry,
        required=("true_hessian", "gamma", "layer", "checkpoint_sha256"))
    snapshot = torch.load(arguments.snapshot, map_location="cpu", weights_only=True)
    if snapshot.get("schema_version") != "pldr-adamw-transport-v2":
        raise ValueError("capture snapshot has an unknown schema")
    layer_index = int(np.asarray(geometry_archive["layer"]).item())
    expected_name = (
        f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."
        f"{ARCHITECTURE['row_map_residual_units'] - 1}.layernormA.weight")
    candidates = [
        row for rows in snapshot.get("blocks", {}).values()
        for row in rows if row.get("name") == expected_name
    ]
    if len(candidates) != 1:
        raise ValueError("snapshot does not contain exactly one final gate tensor")
    row = candidates[0]
    required = {
        "theta_before", "raw_gradient", "first_moment_before",
        "second_moment_before", "state_step_after", "beta1", "beta2",
        "epsilon", "weight_decay", "clip_value", "decay_mask",
    }
    if not required.issubset(row) or row["raw_gradient"] is None:
        raise ValueError("snapshot final gate row is incomplete")

    def array(name):
        return row[name].detach().double().reshape(-1).numpy()

    gamma = array("theta_before")
    if not np.allclose(
        gamma, geometry_archive["gamma"], rtol=1e-7, atol=1e-9):
        raise ValueError("snapshot gate disagrees with geometry source")
    raw_gradient = array("raw_gradient")
    first = array("first_moment_before")
    second = array("second_moment_before")
    decay_mask = array("decay_mask").astype(np.float64)
    clip_value = row["clip_value"]
    hessian_value = np.asarray(geometry_archive["true_hessian"], dtype=np.float64)
    optimizer = {
        "learning_rate": float(snapshot["learning_rate_applied"]),
        "beta1": float(row["beta1"]),
        "beta2": float(row["beta2"]),
        "epsilon": float(row["epsilon"]),
        "optimizer_step": int(row["state_step_after"]),
        "weight_decay": float(row["weight_decay"]) * decay_mask,
        "clip_value": (None if clip_value is None else float(clip_value)),
    }
    boundary_tolerance = float(arguments.boundary_tolerance)
    block = adamw_gate_block(
        hessian_value, gamma, first, second, raw_gradient,
        boundary_tolerance=boundary_tolerance, **optimizer)

    def affine_gradient(position):
        return raw_gradient + hessian_value @ (position - gamma)

    independent = finite_difference_adamw_block(
        affine_gradient, gamma, first, second,
        step_size=float(arguments.finite_difference_step), **optimizer)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(PRODUCER_SCHEMA),
        stage=np.asarray("C"),
        producer_sha256=np.asarray(_producer_digest()),
        geometry_sha256=np.asarray(sha256_path(arguments.geometry)),
        snapshot_sha256=np.asarray(sha256_path(arguments.snapshot)),
        step=np.asarray(snapshot["global_optimizer_step"], dtype=np.int64),
        layer=np.asarray(layer_index, dtype=np.int64),
        gamma=gamma,
        first_moment=first,
        second_moment=second,
        raw_gradient=raw_gradient,
        true_hessian=hessian_value,
        learning_rate=np.asarray(optimizer["learning_rate"]),
        beta1=np.asarray(optimizer["beta1"]),
        beta2=np.asarray(optimizer["beta2"]),
        adam_epsilon=np.asarray(optimizer["epsilon"]),
        optimizer_step=np.asarray(optimizer["optimizer_step"], dtype=np.int64),
        weight_decay=np.asarray(optimizer["weight_decay"]),
        clip_value=np.asarray(
            np.nan if optimizer["clip_value"] is None
            else optimizer["clip_value"]),
        finite_difference_step=np.asarray(float(arguments.finite_difference_step)),
        clipped_gradient=block["clipped_gradient"],
        clip_derivative=block["clip_derivative"],
        clipping_boundary_distance=np.asarray(
            np.inf if clip_value is None
            else np.min(np.abs(np.abs(raw_gradient) - float(clip_value)))),
        operator=block["operator"],
        independent_operator=independent,
        operator_replay_residual=np.asarray(
            np.linalg.norm(block["operator"] - independent, ord=2)),
    )


def energy_step(arguments):
    source = load_npz(
        arguments.source,
        required=("gamma", "q", "q_directional", "step", "layer"))
    target = load_npz(
        arguments.target, required=("gamma", "q", "step", "layer"))
    if int(source["layer"].item()) != int(target["layer"].item()):
        raise ValueError("energy-step geometry layers disagree")
    if int(target["step"].item()) <= int(source["step"].item()):
        raise ValueError("energy-step target must follow its source")
    if not np.isfinite(source["q_directional"]).all():
        raise ValueError("source geometry omitted its source-owned q JVP")
    increment = gate_shape_energy_increment(
        source["gamma"], target["gamma"], source["q"], target["q"])
    gamma_next = np.asarray(target["gamma"], dtype=np.float64)
    linear_shape_work = float(np.sum(
        source["q_directional"] * gamma_next * gamma_next))
    shape_remainder = np.asarray(
        target["q"] - source["q"] - source["q_directional"],
        dtype=np.float64)
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(PRODUCER_SCHEMA),
        stage=np.asarray("C-energy-step"),
        producer_sha256=np.asarray(_producer_digest()),
        source_sha256=np.asarray(sha256_path(arguments.source)),
        target_sha256=np.asarray(sha256_path(arguments.target)),
        source_step=source["step"],
        target_step=target["step"],
        layer=source["layer"],
        energy=np.asarray(increment["energy"]),
        energy_next=np.asarray(increment["energy_next"]),
        gate_work=np.asarray(increment["gate_work"]),
        shape_work=np.asarray(increment["shape_work"]),
        shape_linear_work=np.asarray(linear_shape_work),
        shape_remainder=shape_remainder,
        identity_residual=np.asarray(increment["identity_residual"]),
    )


def _parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fixture = subparsers.add_parser("fixture")
    fixture.add_argument("--output", required=True)
    fixture.set_defaults(function=realize_fixture)

    qualification = subparsers.add_parser("qualify")
    qualification.add_argument("--fixture", required=True)
    qualification.add_argument("--device", required=True)
    qualification.add_argument("--output", required=True)
    qualification.set_defaults(function=qualify)

    registry = subparsers.add_parser("registry")
    registry.add_argument("--tokens", required=True)
    registry.add_argument("--tokenizer", required=True)
    registry.add_argument("--reserved-start", type=int, default=-1)
    registry.add_argument("--output", required=True)
    registry.set_defaults(function=create_registry)

    geometry_parser = subparsers.add_parser("geometry")
    geometry_parser.add_argument("--checkpoint", required=True)
    geometry_parser.add_argument("--target-checkpoint", default="")
    geometry_parser.add_argument("--tokens", required=True)
    geometry_parser.add_argument("--registry", required=True)
    geometry_parser.add_argument("--registry-role", choices=(
        "construction", "validation"), required=True)
    geometry_parser.add_argument("--layer", type=int, required=True)
    geometry_parser.add_argument("--context-limit", type=int, default=0)
    geometry_parser.add_argument("--device", default="cuda:0")
    geometry_parser.add_argument("--output", required=True)
    geometry_parser.set_defaults(function=geometry)

    intervention = subparsers.add_parser("intervention-geometry")
    intervention.add_argument("--checkpoint", required=True)
    intervention.add_argument("--tokens", required=True)
    intervention.add_argument("--registry", required=True)
    intervention.add_argument("--layer", type=int, required=True)
    intervention.add_argument("--device", default="cuda:0")
    intervention.add_argument("--output", required=True)
    intervention.set_defaults(function=intervention_geometry)

    capture = subparsers.add_parser("capture-block")
    capture.add_argument("--geometry", required=True)
    capture.add_argument("--snapshot", required=True)
    capture.add_argument("--boundary-tolerance", type=float, default=1e-8)
    capture.add_argument("--finite-difference-step", type=float, default=2e-6)
    capture.add_argument("--output", required=True)
    capture.set_defaults(function=capture_block)

    step = subparsers.add_parser("energy-step")
    step.add_argument("--source", required=True)
    step.add_argument("--target", required=True)
    step.add_argument("--output", required=True)
    step.set_defaults(function=energy_step)
    return parser


def main():
    arguments = _parser().parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
