#!/usr/bin/env python3
"""Checkpoint-bound producers for the comprehensive collapse program.

The normal producer differentiates the finite row-map stencil with respect
to the registered row-map parameters.  It streams forward-mode columns to
host storage and evaluates the complete vocabulary-loss pullback with
JVP/Fisher/VJP and HVP actions.  No command accepts a caller-created Gram,
Hessian, normal basis, pass flag, or scientific decision.
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
from torch.func import functional_call, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from comprehensive_collapse import (  # noqa: E402
    adamw_successor_jacobian,
    environment_metadata,
    joint_row_direction_cover,
    scheduled_path_metrics,
    vector_bernstein_radius,
)
from comprehensive_protocol_specs import ARCHITECTURE  # noqa: E402
from confirmation_artifacts import (  # noqa: E402
    digest_object,
    load_checkpoint_set,
    load_complete_checkpoint,
    sha256_path,
    write_json_atomic,
)
from live_confirmation_producers import (  # noqa: E402
    _application_batches,
    _capture,
    _coordinate_for_checkpoint,
    _model_from_checkpoint,
    _non_rate_digest,
)
from streamed_parameter_normal import (  # noqa: E402
    complete_fisher_normal_action,
    flatten_tensors,
    full_loss_normal_action,
    linearized_normal_coordinates,
    normal_basis_from_dense_jacobian,
    parameter_tuple,
    selected_pair_normal_action,
    stream_forward_jacobian,
    symmetric_lanczos,
)


PRODUCER_SCHEMA = "pldr-comprehensive-live-producer-v4"
REGISTRY_SCHEMA = "pldr-comprehensive-measurement-registry-v4"


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _producer_digest():
    return sha256_path(Path(__file__))


def _write_npz(path, values):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(destination)


def _safe_relative(path, root):
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("an artifact path lies outside its declared root") from error


def create_registry(arguments):
    tokens = Path(arguments.tokens)
    tokenizer = Path(arguments.tokenizer)
    context = int(arguments.context_length)
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    chunk_count = len(data) // context
    required = (
        arguments.construction_contexts
        + arguments.validation_contexts
        + 2 * arguments.application_pairs
    )
    start = (
        int(arguments.reserved_start)
        if arguments.reserved_start >= 0 else chunk_count - required - 64)
    if start < 0 or start + required > chunk_count:
        raise ValueError("the immutable registry exceeds the token archive")
    cursor = start

    def rows(prefix, count):
        nonlocal cursor
        result = [
            {"id": f"{prefix}{index:04d}", "chunk_index": cursor + index}
            for index in range(count)
        ]
        cursor += count
        return result

    construction = rows("c", arguments.construction_contexts)
    validation = rows("v", arguments.validation_contexts)
    application = []
    for index in range(arguments.application_pairs):
        application.append({
            "id": f"a{index:04d}",
            "left_chunk": cursor + 2 * index,
            "right_chunk": cursor + 2 * index + 1,
        })
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "context_length": context,
        "construction": construction,
        "validation": validation,
        "application_pairs": application,
        "finite_stencil": {
            "step": float(arguments.stencil_step),
            "direction_algorithm": "numpy-pcg64-normalized-v1",
            "direction_seed": int(arguments.direction_seed),
            "one_direction_per_physical_row": True,
            "joint_cover_algorithm":
                "scaled-row-plus-direction-nearest-v1",
            "cover_directions_per_row":
                int(arguments.cover_directions_per_row),
            "cover_seed": int(arguments.challenge_seed) + 29,
        },
        "selected_edge_rule": {
            "kind": "target-plus-fixed-competitors",
            "competitor_ids": np.random.default_rng(
                int(arguments.challenge_seed) + 17).choice(
                    32000, size=int(arguments.edge_alternatives),
                    replace=False).astype(int).tolist(),
        },
        "challenge_seed": int(arguments.challenge_seed),
        "dataset_sha256": sha256_path(tokens),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    registry["registry_sha256"] = _digest(registry)
    write_json_atomic(arguments.output, registry)


def load_registry(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    required = {
        "schema_version", "context_length", "construction", "validation",
        "application_pairs", "finite_stencil", "selected_edge_rule",
        "challenge_seed", "dataset_sha256", "tokenizer_sha256",
        "registry_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("comprehensive measurement registry has an invalid schema")
    if value["schema_version"] != REGISTRY_SCHEMA:
        raise ValueError("unknown comprehensive measurement registry")
    unsigned = dict(value)
    digest = unsigned.pop("registry_sha256")
    if digest != _digest(unsigned):
        raise ValueError("measurement registry digest does not replay")
    identifiers = []
    chunks = []
    for role in ("construction", "validation"):
        for row in value[role]:
            if set(row) != {"id", "chunk_index"}:
                raise ValueError("a registry context row is malformed")
            identifiers.append(row["id"])
            chunks.append(int(row["chunk_index"]))
    for row in value["application_pairs"]:
        if set(row) != {"id", "left_chunk", "right_chunk"}:
            raise ValueError("an application registry row is malformed")
        identifiers.append(row["id"])
        chunks.extend((int(row["left_chunk"]), int(row["right_chunk"])))
    if len(identifiers) != len(set(identifiers)) or len(chunks) != len(set(chunks)):
        raise ValueError("registry identifiers and token chunks must be disjoint")
    return value


def _token_batch(tokens_path, registry, row):
    context = int(registry["context_length"])
    data = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    start = int(row["chunk_index"]) * context
    stop = start + context
    if stop > len(data):
        raise ValueError("registry token chunk lies outside the archive")
    return torch.from_numpy(
        np.asarray(data[start:stop], dtype=np.int64).copy())[None]


def _causal_mask(length, device, dtype):
    return torch.triu(
        torch.ones(length, length, device=device, dtype=dtype), diagonal=1,
    )[None, None]


def _capture_physical_rows(model, token_batches, layer_index):
    rows = []
    block = model.decoder.dec_layers[layer_index].mha1.reslayerAs[0]

    def hook(_module, inputs):
        value = inputs[0][0]
        rows.append(value.detach())

    handle = block.register_forward_pre_hook(hook)
    try:
        for tokens in token_batches:
            device = next(model.parameters()).device
            inputs = tokens[:, :-1].to(device)
            mask = _causal_mask(
                inputs.shape[1], device, next(model.parameters()).dtype)
            with torch.no_grad():
                model([inputs, mask])
    finally:
        handle.remove()
    if len(rows) != len(token_batches):
        raise RuntimeError("the physical-row hook did not fire exactly once per context")
    return torch.cat([value.reshape(-1, value.shape[-1]) for value in rows])


def _selected_parameter_prefix(layer_index):
    return f"decoder.dec_layers.{layer_index}.mha1.reslayerAs."


def _selected_parameter_tuple(model, layer_index):
    prefix = _selected_parameter_prefix(layer_index)
    return parameter_tuple(model, lambda name, _parameter: name.startswith(prefix))


def _row_map_apply(model, layer_index, parameter_mapping, rows):
    value = rows
    blocks = model.decoder.dec_layers[layer_index].mha1.reslayerAs
    base_prefix = _selected_parameter_prefix(layer_index)
    for index, block in enumerate(blocks):
        prefix = f"{base_prefix}{index}."
        local = {
            name[len(prefix):]: parameter
            for name, parameter in parameter_mapping.items()
            if name.startswith(prefix)
        }
        value = functional_call(block, local, ([value],), strict=False)
    return value


def _directions(count, dimension, seed, *, device, dtype):
    generator = np.random.default_rng(int(seed))
    values = generator.normal(size=(count, dimension))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return torch.as_tensor(values, device=device, dtype=dtype)


def _finite_stencil_function(
        model, layer_index, layout, physical_rows, directions, step):
    def function(parameters):
        mapping = dict(zip(layout.names, parameters))
        base = _row_map_apply(
            model, layer_index, mapping, physical_rows)
        displaced = _row_map_apply(
            model, layer_index, mapping, physical_rows + step * directions)
        return ((displaced - base) / step).reshape(-1)
    return function


def _logit_function(model, layout, tokens):
    device = next(model.parameters()).device
    inputs = tokens[:, :-1].to(device)
    mask = _causal_mask(inputs.shape[1], device, next(model.parameters()).dtype)

    def function(parameters):
        mapping = dict(zip(layout.names, parameters))
        logits = functional_call(
            model, mapping, ([inputs, mask],), strict=False)[0][0]
        return logits - logits.mean(dim=-1, keepdim=True)
    return function, tokens[0, 1:].to(device)


def _selected_edges(probability, target, competitors):
    selected = []
    if probability.ndim != 2 or target.ndim != 1:
        raise ValueError("all-position edges require source-vocabulary logits")
    if probability.shape[0] != target.numel():
        raise ValueError("target and source-position counts disagree")
    for source, raw_target in enumerate(target.tolist()):
        for index in competitors:
            index = int(index)
            if index != raw_target and 0 <= index < probability.shape[1]:
                selected.append((
                    source, min(raw_target, index), max(raw_target, index)))
    if not selected:
        raise ValueError("selected-edge construction produced an empty graph")
    products = [
        float(probability[source, left] * probability[source, right])
        for source, left, right in selected
    ]
    return selected, products


def _memory_record(device, started):
    cuda = device.type == "cuda"
    elapsed = time.monotonic() - started
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


def _resource_arrays(device, started):
    record = _memory_record(device, started)
    return {
        "resource_device": np.asarray(str(device)),
        "resource_environment_json": np.asarray(
            json.dumps(environment_metadata(), sort_keys=True)),
        **{
            f"resource_{name}": np.asarray(value)
            for name, value in record.items()
        },
    }


def qualify(arguments):
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    torch.manual_seed(3411)
    dtype = torch.float64
    parameter = torch.randn(8, device=device, dtype=dtype, requires_grad=True)
    row = torch.linspace(-1, 1, 256, device=device, dtype=dtype)

    def logits(parameters):
        value = parameters[0]
        features = torch.stack([
            torch.sin(row * value[index % 8]) for index in range(16)], dim=1)
        return features.mean(dim=0) + value.repeat(2)

    parameters = (parameter,)
    direction = (torch.linspace(0.1, 0.8, 8, device=device, dtype=dtype),)
    output, logit_action = jvp(logits, (parameters,), (direction,))
    probability = torch.softmax(output, dim=0)
    fisher_action = probability * logit_action - probability * torch.sum(
        probability * logit_action)

    def loss(values):
        value = logits(values)
        return torch.logsumexp(value, dim=0) - value[3]

    from torch.func import grad as torch_grad
    _, hessian_action = jvp(torch_grad(loss), (parameters,), (direction,))
    gradient_jacobian = np.asarray([[1.4, -0.2], [0.3, 0.9]])
    adam = adamw_successor_jacobian(
        gradient_jacobian, np.asarray([0.2, -0.3]),
        np.asarray([0.1, -0.15]), np.asarray([0.7, 0.8]),
        learning_rate=1e-3, beta1=0.9, beta2=0.99, epsilon=1e-8,
        optimizer_step=17, weight_decay=0.01)
    payload = {
        "schema_version": PRODUCER_SCHEMA,
        "stage": "Q",
        "producer_sha256": _producer_digest(),
        "device": str(device),
        "context_length": 256,
        "environment": environment_metadata(),
        "row_jvp_probe": {
            "direction": direction[0].detach().cpu().tolist(),
            "output": logit_action.detach().cpu().tolist(),
        },
        "fisher_probe": fisher_action.detach().cpu().tolist(),
        "true_hessian_probe": flatten_tensors(
            hessian_action, parameter_tuple_from_values(parameters)
        ).detach().cpu().tolist(),
        "adamw_successor_operator": adam["operator"].tolist(),
        "resource": _memory_record(device, started),
    }
    payload["record_sha256"] = _digest(payload)
    write_json_atomic(arguments.output, payload)


def parameter_tuple_from_values(values):
    from streamed_parameter_normal import ParameterLayout
    return ParameterLayout(
        names=tuple(f"p{index}" for index in range(len(values))),
        shapes=tuple(tuple(value.shape) for value in values),
        sizes=tuple(value.numel() for value in values),
    )


def normal(arguments):
    output_root = Path(arguments.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(arguments.checkpoint).resolve()
    tokens_path = Path(arguments.tokens).resolve()
    registry_path = Path(arguments.registry).resolve()
    checkpoint = load_complete_checkpoint(checkpoint_path)
    registry = load_registry(registry_path)
    if sha256_path(tokens_path) != registry["dataset_sha256"]:
        raise ValueError("normal producer token archive disagrees with the registry")
    if digest_object(checkpoint.get("measurement_registry", {})) not in {
        registry["registry_sha256"], digest_object(registry),
    } and checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint measurement registry disagrees with N registry")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    model = _model_from_checkpoint(checkpoint, device).double()
    layer_index = int(arguments.layer)
    layout, parameters = _selected_parameter_tuple(model, layer_index)
    clip_value = checkpoint["optimizer_config"].get("gradient_clip_value")
    if (
        checkpoint["optimizer_config"].get("gradient_clip_kind") != "value"
        or clip_value is None or not math.isfinite(float(clip_value))
        or float(clip_value) <= 0
    ):
        raise ValueError("N requires a positive realized value-clip bound")
    clip_value = float(clip_value)
    chart_contexts = registry["construction"]
    action_source_contexts = registry[arguments.registry_role]
    if arguments.registry_role == "validation" and arguments.pair_product_lower <= 0:
        raise ValueError("validation N requires a positive sealed pair-product lower bound")
    if arguments.context_limit:
        chart_contexts = chart_contexts[:arguments.context_limit]
    token_batches = [
        _token_batch(tokens_path, registry, row) for row in chart_contexts]
    physical_rows = _capture_physical_rows(model, token_batches, layer_index)
    directions = _directions(
        physical_rows.shape[0], physical_rows.shape[1],
        registry["finite_stencil"]["direction_seed"],
        device=physical_rows.device, dtype=physical_rows.dtype)
    stencil_step = float(registry["finite_stencil"]["step"])

    cover_batches = [
        _token_batch(tokens_path, registry, row)
        for row in action_source_contexts
    ]
    cover_physical_rows = _capture_physical_rows(
        model, cover_batches, layer_index)
    directions_per_row = int(
        registry["finite_stencil"]["cover_directions_per_row"])
    query_rows = cover_physical_rows.repeat_interleave(
        directions_per_row, dim=0)
    query_directions = _directions(
        query_rows.shape[0], query_rows.shape[1],
        registry["finite_stencil"]["cover_seed"],
        device=query_rows.device, dtype=query_rows.dtype)
    registered_rows_array = physical_rows.detach().double().cpu().numpy()
    registered_directions_array = directions.detach().double().cpu().numpy()
    center = np.mean(registered_rows_array, axis=0, keepdims=True)
    row_scale = max(float(np.median(np.linalg.norm(
        registered_rows_array - center, axis=1))), 1e-12)
    joint_cover = joint_row_direction_cover(
        registered_rows_array, registered_directions_array,
        query_rows.detach().double().cpu().numpy(),
        query_directions.detach().double().cpu().numpy(),
        row_scale=row_scale)

    stencil_function = _finite_stencil_function(
        model, layer_index, layout, physical_rows, directions, stencil_step)
    jacobian_path = output_root / "finite_stencil_parameter_jacobian.npy"
    stream_record = stream_forward_jacobian(
        stencil_function, parameters, layout, jacobian_path,
        block_size=arguments.tangent_block_size)
    jacobian = np.load(jacobian_path, mmap_mode="r")
    normal_result = normal_basis_from_dense_jacobian(
        jacobian, relative_tolerance=arguments.rank_tolerance)
    basis_path = output_root / "parameter_normal_basis.npy"
    np.save(basis_path, normal_result["basis"], allow_pickle=False)
    left_basis_path = output_root / "finite_stencil_left_basis.npy"
    np.save(
        left_basis_path, normal_result["left_basis"], allow_pickle=False)
    singular_path = output_root / "finite_stencil_singular_values.npy"
    np.save(singular_path, normal_result["singular_values"], allow_pickle=False)
    basis = normal_result["basis"]
    left_basis = normal_result["left_basis"]
    normal_dimension = int(normal_result["rank"])
    if normal_dimension == 0:
        raise ValueError("finite-stencil parameter normal has zero dimension")

    action_contexts = (
        action_source_contexts
        if arguments.action_contexts == 0
        else action_source_contexts[:arguments.action_contexts])
    logit_records = []
    actions = []
    for row in action_contexts:
        tokens = _token_batch(tokens_path, registry, row)
        logit_function, target = _logit_function(model, layout, tokens)
        with torch.no_grad():
            probability = torch.softmax(
                logit_function(parameters), dim=-1)
        edges, realized_products = _selected_edges(
            probability, target,
            registry["selected_edge_rule"]["competitor_ids"])
        if arguments.pair_product_lower > 0:
            products = [float(arguments.pair_product_lower)] * len(edges)
            if any(bound > realized * (1 + 1e-12)
                   for bound, realized in zip(products, realized_products)):
                raise ValueError("sealed pair-product lower bound fails on this context")
        else:
            products = realized_products
        logit_records.append({
            "id": row["id"], "targets": target.detach().cpu().tolist(),
            "edges": [list(edge) for edge in edges],
            "realized_probability_products": realized_products,
            "lower_probability_products": products,
        })
        actions.append((logit_function, probability, edges, products, target))

    from torch.func import grad as torch_grad
    normal_gradient_samples = []
    for logit_function, _probability, _edges, _products, target in actions:
        def source_loss(values, function=logit_function, index=target):
            logits = function(values)
            return (
                torch.logsumexp(logits, dim=-1)
                - logits.gather(1, index[:, None]).squeeze(1)
            ).mean()
        gradient = torch_grad(source_loss)(parameters)
        flat_gradient = (
            flatten_tensors(gradient, layout).detach().double().cpu().numpy())
        clipped_gradient = np.clip(
            flat_gradient, -clip_value, clip_value)
        normal_gradient_samples.append(basis.T @ clipped_gradient)
    normal_gradient_samples = np.asarray(normal_gradient_samples)

    def pair_action(coordinate):
        total = np.zeros(normal_dimension)
        for logit_function, probability, edges, products, _target in actions:
            total += selected_pair_normal_action(
                logit_function, parameters, layout, basis, probability,
                edges, products, coordinate) / probability.shape[0]
        return total / len(actions)

    def fisher_action(coordinate):
        total = np.zeros(normal_dimension)
        for logit_function, probability, _edges, _products, _target in actions:
            total += complete_fisher_normal_action(
                logit_function, parameters, layout, basis, probability,
                coordinate) / probability.shape[0]
        return total / len(actions)

    def true_loss_action(coordinate):
        total = np.zeros(normal_dimension)
        for logit_function, _probability, _edges, _products, target in actions:
            def loss(values, function=logit_function, index=target):
                logits = function(values)
                return (
                    torch.logsumexp(logits, dim=-1)
                    - logits.gather(1, index[:, None]).squeeze(1)
                ).mean()
            total += full_loss_normal_action(
                loss, parameters, layout, basis, coordinate)
        return total / len(actions)

    krylov = (
        normal_dimension if int(arguments.krylov_iterations) == 0
        else min(int(arguments.krylov_iterations), normal_dimension))
    pair_ritz = symmetric_lanczos(
        pair_action, normal_dimension, iterations=krylov,
        seed=registry["challenge_seed"])
    full_ritz = symmetric_lanczos(
        true_loss_action, normal_dimension, iterations=krylov,
        seed=registry["challenge_seed"] + 1)
    generator = np.random.default_rng(registry["challenge_seed"] + 2)
    challenge = generator.normal(
        size=(arguments.challenge_directions, normal_dimension))
    challenge /= np.linalg.norm(challenge, axis=1, keepdims=True)
    stencil_challenge = (
        (left_basis * normal_result["singular_values"][:normal_dimension])
        @ challenge.T
    ).T
    pair_challenge = np.asarray([pair_action(value) for value in challenge])
    fisher_challenge = np.asarray([fisher_action(value) for value in challenge])
    full_challenge = np.asarray([true_loss_action(value) for value in challenge])
    arrays_path = output_root / "normal_actions.npz"
    _write_npz(arrays_path, {
        "challenge_directions": challenge,
        "stencil_challenge_actions": stencil_challenge,
        "normal_gradient_samples": normal_gradient_samples,
        "pair_challenge_actions": pair_challenge,
        "fisher_challenge_actions": fisher_challenge,
        "signed_second_challenge_quadratics": np.einsum(
            "ij,ij->i", challenge, full_challenge - fisher_challenge),
        "full_loss_challenge_actions": full_challenge,
        "joint_cover_selected_indices":
            joint_cover["selected_registered_index"],
        "joint_cover_row_radii": joint_cover["row_radii"],
        "joint_cover_direction_radii": joint_cover["direction_radii"],
        "pair_ritz_values": pair_ritz["ritz_values"],
        "pair_ritz_residuals": pair_ritz["residuals"],
        "pair_ritz_intervals": pair_ritz["intervals"],
        "full_loss_ritz_values": full_ritz["ritz_values"],
        "full_loss_ritz_residuals": full_ritz["residuals"],
        "full_loss_ritz_intervals": full_ritz["intervals"],
    })
    resource_record = _memory_record(device, started)
    metadata = {
        "schema_version": PRODUCER_SCHEMA,
        "stage": "N",
        "engineering_only": bool(arguments.engineering_only),
        "registry_role": arguments.registry_role,
        "pair_product_lower": float(arguments.pair_product_lower),
        "producer_sha256": _producer_digest(),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "checkpoint_state_sha256": checkpoint["state_manifest"][
            "complete_state_sha256"],
        "tokens_path": str(tokens_path),
        "dataset_sha256": sha256_path(tokens_path),
        "registry_path": str(registry_path),
        "registry_sha256": registry["registry_sha256"],
        "layer": layer_index,
        "construction_context_ids": [row["id"] for row in chart_contexts],
        "parameter_names": list(layout.names),
        "parameter_count": layout.dimension,
        "normal_gradient_rule":
            "ambient-value-clip-then-orthogonal-normal-projection-v1",
        "normal_gradient_clip_value": clip_value,
        "normal_rank": normal_dimension,
        "stencil_output_dimension": int(jacobian.shape[0]),
        "rank_threshold": normal_result["rank_threshold"],
        "range_residual": normal_result["range_residual"],
        "orthogonality_residual": normal_result["orthogonality_residual"],
        "left_orthogonality_residual":
            normal_result["left_orthogonality_residual"],
        "stream": stream_record,
        "action_contexts": logit_records,
        "challenge_seed": registry["challenge_seed"],
        "krylov_iterations": krylov,
        "joint_cover": {
            "algorithm":
                registry["finite_stencil"]["joint_cover_algorithm"],
            "registry_role": arguments.registry_role,
            "query_context_ids": [
                row["id"] for row in action_source_contexts],
            "directions_per_row": directions_per_row,
            "challenge_count":
                int(joint_cover["row_radii"].shape[0]),
            "row_scale": joint_cover["row_scale"],
            "maximum_row_radius":
                joint_cover["maximum_row_radius"],
            "maximum_direction_radius":
                joint_cover["maximum_direction_radius"],
        },
        "environment": environment_metadata(),
        "resource": resource_record,
        "artifacts": {
            "finite_stencil_jacobian": jacobian_path.name,
            "normal_basis": basis_path.name,
            "normal_left_basis": left_basis_path.name,
            "singular_values": singular_path.name,
            "normal_actions": arrays_path.name,
        },
    }
    metadata["artifact_sha256"] = {
        name: sha256_path(output_root / relative)
        for name, relative in metadata["artifacts"].items()
    }
    metadata["record_sha256"] = _digest(metadata)
    write_json_atomic(output_root / "normal_metadata.json", metadata)


def _checkpoint_stencil_observables(
        checkpoint, tokens_path, registry, device, layer_index, *,
        measure_direct):
    """Evaluate the registered row-map stencil from one complete checkpoint."""

    if checkpoint["measurement_registry"] != registry:
        raise ValueError("a checkpoint changed the measurement registry")
    model = _model_from_checkpoint(checkpoint, device).double()
    layout, parameters = _selected_parameter_tuple(model, layer_index)
    batches = [
        _token_batch(tokens_path, registry, row)
        for row in registry["construction"]
    ]
    physical_rows = _capture_physical_rows(model, batches, layer_index)
    directions = _directions(
        physical_rows.shape[0], physical_rows.shape[1],
        registry["finite_stencil"]["direction_seed"],
        device=physical_rows.device, dtype=physical_rows.dtype)
    mapping = dict(zip(layout.names, parameters))
    step = float(registry["finite_stencil"]["step"])
    with torch.no_grad():
        base = _row_map_apply(model, layer_index, mapping, physical_rows)
        displaced = _row_map_apply(
            model, layer_index, mapping, physical_rows + step * directions)
        stencil = (displaced - base) / step
    vector = stencil.detach().reshape(-1).double().cpu().numpy()
    maximum = float(torch.linalg.vector_norm(
        stencil.reshape(stencil.shape[0], -1), dim=1).max().cpu())
    direct_value = None
    if measure_direct:
        def row_function(rows):
            return _row_map_apply(model, layer_index, mapping, rows)

        _, direct = jvp(row_function, (physical_rows,), (directions,))
        direct_value = float(torch.linalg.vector_norm(
            direct.reshape(direct.shape[0], -1),
            dim=1).max().detach().cpu())
    return vector, maximum, direct_value


def trajectory(arguments):
    """Recompute registered stencil and direct actions at every checkpoint."""

    checkpoints = load_checkpoint_set(
        arguments.checkpoint_manifest, arguments.artifact_root)
    if len(checkpoints) < 2:
        raise ValueError("T trajectory production needs at least two checkpoints")
    steps = np.asarray([int(value["step"]) for value in checkpoints])
    if np.any(np.diff(steps) <= 0):
        raise ValueError("T checkpoints must be strictly ordered")
    registry = load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("T token archive disagrees with the registry")
    stencil_values = []
    stencil_vectors = []
    direct_values = []
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for checkpoint in checkpoints:
        vector, maximum, direct = _checkpoint_stencil_observables(
            checkpoint, arguments.tokens, registry, device,
            int(arguments.layer), measure_direct=True)
        stencil_vectors.append(vector)
        stencil_values.append(maximum)
        direct_values.append(direct)
    _write_npz(arguments.output, {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "checkpoint_state_sha256": np.asarray([
            value["state_manifest"]["complete_state_sha256"]
            for value in checkpoints]),
        "tokens_sha256": np.asarray(sha256_path(arguments.tokens)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "layer": np.asarray(int(arguments.layer), dtype=np.int64),
        "steps": steps.astype(np.int64),
        "finite_stencil_vectors": np.asarray(
            stencil_vectors, dtype=np.float64),
        "finite_stencil_norm": np.asarray(stencil_values, dtype=np.float64),
        "direct_seminorm": np.asarray(direct_values, dtype=np.float64),
        **_resource_arrays(device, started),
    })


def _snapshot_rows(path, tensor_group):
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if snapshot.get("schema_version") != "pldr-adamw-transport-v2":
        raise ValueError("D snapshot has an unknown schema")
    rows = snapshot.get("blocks", {}).get(tensor_group, [])
    if not rows:
        raise ValueError(f"D snapshot has no {tensor_group} tensors")
    required = {
        "theta_before", "clipped_gradient", "first_moment_before",
        "second_moment_before", "theta_after_intervention",
        "first_moment_after", "second_moment_after", "state_step_after",
        "beta1", "beta2", "epsilon", "weight_decay",
        "clip_derivative", "decay_mask",
    }
    if any(not required.issubset(row) for row in rows):
        raise ValueError("D snapshot omits a required optimizer tensor")
    return snapshot, rows


def _flatten_snapshot(rows, name):
    tensors = [
        row[name].detach().double().reshape(-1)
        for row in rows if torch.is_tensor(row.get(name))
    ]
    if not tensors:
        raise ValueError(f"D snapshot has no tensor values for {name}")
    return torch.cat(tensors)


def _scalar_live_adamw_operator(
        gradient_edge, clipped_gradient, first_moment, second_moment,
        *, learning_rate, beta1, beta2, epsilon, optimizer_step,
        weight_decay, clip_derivative, decay_mask):
    edge = torch.tensor(float(gradient_edge), dtype=torch.float64)
    gradient = torch.tensor(float(clipped_gradient), dtype=torch.float64)
    rate = torch.tensor(float(learning_rate), dtype=torch.float64)
    beta_one = torch.tensor(float(beta1), dtype=torch.float64)
    beta_two = torch.tensor(float(beta2), dtype=torch.float64)
    eps = torch.tensor(float(epsilon), dtype=torch.float64)
    decay = torch.tensor(float(weight_decay * decay_mask), dtype=torch.float64)
    clip_cell = torch.tensor(float(clip_derivative), dtype=torch.float64)
    correction_one = 1.0 - beta_one ** int(optimizer_step)
    correction_two = 1.0 - beta_two ** int(optimizer_step)

    def successor(state):
        theta, moment, variance = state.unbind()
        local_gradient = gradient + clip_cell * edge * theta
        next_moment = beta_one * moment + (1.0 - beta_one) * local_gradient
        next_variance = (
            beta_two * variance
            + (1.0 - beta_two) * local_gradient.square())
        corrected_moment = next_moment / correction_one
        corrected_variance = next_variance / correction_two
        next_theta = (
            (1.0 - rate * decay) * theta
            - rate * corrected_moment
            / (torch.sqrt(corrected_variance) + eps))
        return torch.stack((next_theta, next_moment, next_variance))

    base = torch.tensor(
        [0.0, float(first_moment), float(second_moment)],
        dtype=torch.float64, requires_grad=True)
    return torch.autograd.functional.jacobian(
        successor, base, create_graph=False, strict=True).detach().numpy()


def dynamics(arguments):
    """Test a challenge lift against the measured linearized normal path."""

    started = time.monotonic()
    snapshot_paths = [Path(value).resolve() for value in arguments.snapshot]
    if len(snapshot_paths) < 2 or any(not path.is_file() for path in snapshot_paths):
        raise ValueError("D needs at least two existing optimizer snapshots")

    with Path(arguments.normal_metadata).open("r", encoding="utf-8") as stream:
        normal_metadata = json.load(stream)
    unsigned_metadata = dict(normal_metadata)
    recorded_metadata = unsigned_metadata.pop("record_sha256", None)
    if (
        normal_metadata.get("schema_version") != PRODUCER_SCHEMA
        or normal_metadata.get("stage") != "N"
        or normal_metadata.get("producer_sha256") != _producer_digest()
        or recorded_metadata != _digest(unsigned_metadata)
    ):
        raise ValueError("D normal metadata is not an active N record")
    factor_paths = {
        "normal_left_basis": Path(arguments.normal_left_basis).resolve(),
        "singular_values": Path(arguments.singular_values).resolve(),
        "normal_actions": Path(arguments.normal_actions).resolve(),
    }
    for name, source in factor_paths.items():
        if normal_metadata["artifact_sha256"].get(name) != sha256_path(source):
            raise ValueError(f"D normal factor {name} changed after N production")
    left_basis = np.load(factor_paths["normal_left_basis"], mmap_mode="r")
    singular = np.load(factor_paths["singular_values"], mmap_mode="r")
    with np.load(factor_paths["normal_actions"], allow_pickle=False) as normal:
        intervals = np.asarray(normal["full_loss_ritz_intervals"], dtype=float)
    if intervals.ndim != 2 or intervals.shape[1] != 2:
        raise ValueError("D normal actions have malformed full-loss intervals")
    if left_basis.shape != (
            int(normal_metadata["stencil_output_dimension"]),
            int(normal_metadata["normal_rank"])):
        raise ValueError("D left normal basis has a wrong shape")
    if singular.ndim != 1 or singular.size < int(normal_metadata["normal_rank"]):
        raise ValueError("D singular-value factor has a wrong shape")

    analytic = []
    live = []
    probe_errors = []
    optimizer_observed = []
    optimizer_after_observed = []
    snapshot_steps = []
    for snapshot_path in snapshot_paths:
        snapshot, rows = _snapshot_rows(snapshot_path, arguments.tensor_group)
        snapshot_steps.append(int(snapshot["global_optimizer_step"]))
        gradient = _flatten_snapshot(rows, "clipped_gradient")
        moment = _flatten_snapshot(rows, "first_moment_before")
        variance = _flatten_snapshot(rows, "second_moment_before")
        theta = _flatten_snapshot(rows, "theta_before")
        theta_after = _flatten_snapshot(rows, "theta_after_intervention")
        moment_after = _flatten_snapshot(rows, "first_moment_after")
        variance_after = _flatten_snapshot(rows, "second_moment_after")
        count = max(gradient.numel(), 1)
        gradient_scalar = float(torch.linalg.vector_norm(gradient) / math.sqrt(count))
        moment_scalar = float(torch.linalg.vector_norm(moment) / math.sqrt(count))
        variance_scalar = float(torch.mean(variance.clamp_min(0)))
        theta_scalar = float(torch.linalg.vector_norm(theta) / math.sqrt(count))
        optimizer_after_observed.append(float(
            torch.linalg.vector_norm(torch.stack((
                torch.linalg.vector_norm(theta_after) / math.sqrt(count),
                torch.linalg.vector_norm(moment_after) / math.sqrt(count),
                torch.mean(variance_after.clamp_min(0)),
            )))))
        optimizer_observed.append(float(math.sqrt(
            theta_scalar ** 2 + moment_scalar ** 2 + variance_scalar ** 2)))
        beta1_values = {float(row["beta1"]) for row in rows}
        beta2_values = {float(row["beta2"]) for row in rows}
        epsilon_values = {float(row["epsilon"]) for row in rows}
        state_steps = {int(row["state_step_after"]) for row in rows}
        if any(len(values) != 1 for values in (
                beta1_values, beta2_values, epsilon_values, state_steps)):
            raise ValueError("D snapshot optimizer scalars disagree within a block")
        beta1 = next(iter(beta1_values))
        beta2 = next(iter(beta2_values))
        epsilon = next(iter(epsilon_values))
        optimizer_step = next(iter(state_steps))
        weight_decay = max(float(row["weight_decay"]) for row in rows)
        decay_mask = float(torch.mean(
            _flatten_snapshot(rows, "decay_mask").double()))
        clip_derivative = float(torch.mean(
            _flatten_snapshot(rows, "clip_derivative").double()))
        edge = float(np.min(intervals[:, 0]))
        coefficients = {
            "learning_rate": float(snapshot["learning_rate_applied"]),
            "beta1": beta1,
            "beta2": beta2,
            "epsilon": epsilon,
            "optimizer_step": optimizer_step,
            "weight_decay": weight_decay,
            "clip_derivative": np.asarray([[clip_derivative]]),
            "decay_mask": np.asarray([decay_mask]),
        }
        result = adamw_successor_jacobian(
            np.asarray([[edge]]), np.asarray([gradient_scalar]),
            np.asarray([moment_scalar]), np.asarray([variance_scalar]),
            **coefficients)
        live_operator = _scalar_live_adamw_operator(
            edge, gradient_scalar, moment_scalar, variance_scalar,
            learning_rate=coefficients["learning_rate"],
            beta1=beta1, beta2=beta2, epsilon=epsilon,
            optimizer_step=optimizer_step, weight_decay=weight_decay,
            clip_derivative=clip_derivative, decay_mask=decay_mask)
        analytic.append(result["operator"])
        live.append(live_operator)
        probe_errors.append(float(np.max(np.abs(
            result["operator"] - live_operator))))
    expected_windows = [
        [int(step) for step in window]
        for window in ARCHITECTURE["optimizer_snapshot_windows"]
    ]
    expected_snapshot_steps = [
        step for window in expected_windows for step in window]
    if snapshot_steps != expected_snapshot_steps:
        raise ValueError("D snapshots disagree with the two registered windows")
    operator_block_offsets = np.asarray(
        [0, *np.cumsum([len(window) for window in expected_windows])],
        dtype=np.int64)
    node_steps_list = []
    node_block_offsets = [0]
    optimizer_node_values = []
    for block_index, window in enumerate(expected_windows):
        operator_start = int(operator_block_offsets[block_index])
        operator_stop = int(operator_block_offsets[block_index + 1])
        nodes = [window[0] - 1, *window]
        node_steps_list.extend(nodes)
        node_block_offsets.append(len(node_steps_list))
        optimizer_node_values.extend([
            optimizer_observed[operator_start],
            *optimizer_after_observed[operator_start:operator_stop],
        ])
    node_steps = np.asarray(node_steps_list, dtype=np.int64)
    node_block_offsets = np.asarray(node_block_offsets, dtype=np.int64)
    optimizer_observed = optimizer_node_values

    trajectory_path = Path(arguments.direct_trajectory).resolve()
    with np.load(trajectory_path, allow_pickle=False) as trajectory:
        if (
            trajectory["producer_code_sha256"].shape != ()
            or trajectory["producer_code_sha256"].item() != _producer_digest()
            or trajectory["registry_sha256"].item()
                != normal_metadata["registry_sha256"]
        ):
            raise ValueError("D trajectory provenance disagrees with N")
        trajectory_steps = np.asarray(trajectory["steps"], dtype=np.int64)
        stencil_vectors = np.asarray(
            trajectory["finite_stencil_vectors"], dtype=np.float64)
        stencil_norms = np.asarray(
            trajectory["finite_stencil_norm"], dtype=np.float64)
        direct_values = np.asarray(
            trajectory["direct_seminorm"], dtype=np.float64)
    if len(set(trajectory_steps.tolist())) != trajectory_steps.size:
        raise ValueError("D trajectory steps are not unique")
    step_to_index = {
        int(step): index for index, step in enumerate(trajectory_steps)}
    if any(int(step) not in step_to_index for step in node_steps):
        raise ValueError("D trajectory omits an optimizer-window boundary")
    selected = np.asarray(
        [step_to_index[int(step)] for step in node_steps], dtype=np.int64)
    selected_vectors = stencil_vectors[selected]
    if selected_vectors.shape[1] != left_basis.shape[0]:
        raise ValueError("D trajectory stencil and N left basis disagree")
    coordinate = linearized_normal_coordinates(
        selected_vectors, left_basis, singular)
    observed = np.asarray(coordinate["amplitudes"], dtype=np.float64)
    reconstruction_residuals = np.asarray(
        coordinate["stencil_reconstruction_residuals"], dtype=np.float64)
    direct = direct_values[selected]
    selected_stencil_norms = stencil_norms[selected]

    metrics_values = []
    gain_values = []
    force_values = []
    radius_values = []
    for block_index in range(len(expected_windows)):
        operator_start = int(operator_block_offsets[block_index])
        operator_stop = int(operator_block_offsets[block_index + 1])
        node_start = int(node_block_offsets[block_index])
        node_stop = int(node_block_offsets[block_index + 1])
        schedule = scheduled_path_metrics(live[operator_start:operator_stop])
        block_gains = np.asarray(schedule["gains"], dtype=np.float64)
        block_observed = observed[node_start:node_stop]
        block_forces = np.maximum(
            0.0,
            block_observed[1:] - block_gains * block_observed[:-1])
        block_forces = np.nextafter(
            block_forces, np.full_like(block_forces, np.inf))
        block_radii = np.empty(block_gains.size + 1, dtype=np.float64)
        block_radii[0] = np.nextafter(block_observed[0], np.inf)
        for index in range(block_gains.size):
            block_radii[index + 1] = np.nextafter(
                max(
                    block_observed[index + 1],
                    block_gains[index] * block_radii[index]
                    + block_forces[index]),
                np.inf)
        metrics_values.extend(schedule["metrics"])
        gain_values.extend(block_gains)
        force_values.extend(block_forces)
        radius_values.extend(block_radii)
    metrics = np.asarray(metrics_values, dtype=np.float64)
    gains = np.asarray(gain_values, dtype=np.float64)
    quadratic = np.zeros_like(gains)
    forces = np.asarray(force_values, dtype=np.float64)
    radii = np.asarray(radius_values, dtype=np.float64)
    direct_coefficient = float(singular[0])
    direct_remainders = np.maximum(
        0.0, direct - direct_coefficient * observed)
    direct_coefficients = np.full_like(observed, direct_coefficient)

    if arguments.prediction_lock:
        with Path(arguments.prediction_lock).open(
                "r", encoding="utf-8") as stream:
            lock = json.load(stream)
        unsigned = dict(lock)
        recorded = unsigned.pop("lock_sha256", None)
        if (
            lock.get("schema_version") != "pldr-prediction-lock-v4"
            or recorded != _digest(unsigned)
        ):
            raise ValueError("D prediction lock does not replay")
        candidates = []
        for locked_schedule, locked_metrics, locked_radii, locked_direct in zip(
                lock["schedule"], lock["path_metrics"], lock["radii"],
                lock["direct_conversion"]):
            if (
                locked_schedule.get("node_steps") == node_steps.tolist()
                and locked_schedule.get("operator_block_offsets")
                    == operator_block_offsets.tolist()
                and locked_schedule.get("node_block_offsets")
                    == node_block_offsets.tolist()
            ):
                candidates.append((
                    locked_schedule, locked_metrics, locked_radii,
                    locked_direct))
        if len(candidates) != 1:
            raise ValueError(
                "D prediction lock does not select one matching two-window schedule")
        locked_schedule, locked_metrics, locked_radii, locked_direct = candidates[0]
        gains = np.asarray(locked_schedule["gains"], dtype=np.float64)
        quadratic = np.asarray(
            locked_schedule["quadratic"], dtype=np.float64)
        forces = np.asarray(locked_schedule["forces"], dtype=np.float64)
        metrics = np.asarray(locked_metrics["metrics"], dtype=np.float64)
        radii = np.asarray(locked_radii["values"], dtype=np.float64)
        direct_coefficient = float(locked_direct["coefficient"])
        direct_remainder = float(locked_direct["remainder"])
        if not math.isclose(
                float(locked_direct["singular_max"]), float(singular[0]),
                rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("D locked direct conversion uses another N factor")
        direct_coefficients = np.full_like(observed, direct_coefficient)
        direct_remainders = np.full_like(observed, direct_remainder)

    _write_npz(arguments.output, {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "tensor_group": np.asarray(arguments.tensor_group),
        "snapshot_sha256": np.asarray([
            sha256_path(source) for source in snapshot_paths]),
        "normal_metadata_sha256": np.asarray(
            sha256_path(arguments.normal_metadata)),
        "normal_actions_sha256": np.asarray(
            sha256_path(arguments.normal_actions)),
        "normal_left_basis_sha256": np.asarray(
            sha256_path(arguments.normal_left_basis)),
        "singular_values_sha256": np.asarray(
            sha256_path(arguments.singular_values)),
        "trajectory_sha256": np.asarray(sha256_path(trajectory_path)),
        "operator_block_offsets": operator_block_offsets,
        "node_block_offsets": node_block_offsets,
        "node_steps": node_steps,
        "normal_coordinate_rule": np.asarray(
            "svd-linearized-least-norm-v1"),
        "normal_coordinate_reconstruction_residuals":
            reconstruction_residuals,
        "stencil_norm_observed": selected_stencil_norms,
        "operators": np.asarray(analytic),
        "live_operators": np.asarray(live),
        "metrics": metrics,
        "radii": radii,
        "gains": gains,
        "quadratic": quadratic,
        "forces": forces,
        "normal_observed": observed,
        "direct_observed": direct,
        "direct_remainders": direct_remainders,
        "direct_coefficients": direct_coefficients,
        "optimizer_state_observed": np.asarray(
            optimizer_observed, dtype=np.float64),
        "successor_probe_errors": np.asarray(probe_errors),
        **_resource_arrays(torch.device("cpu"), started),
    })

def _has_second_moment(checkpoint):
    state = checkpoint.get("opt", {}).get("state", {})
    values = state.values() if isinstance(state, dict) else ()
    return bool(values) and all(
        isinstance(value, dict) and "exp_avg_sq" in value
        for value in values)


def _normal_coordinate_factors(
        metadata_path, left_basis_path, singular_values_path, registry):
    with Path(metadata_path).open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    unsigned = dict(metadata)
    recorded = unsigned.pop("record_sha256", None)
    if (
        metadata.get("schema_version") != PRODUCER_SCHEMA
        or metadata.get("stage") != "N"
        or metadata.get("engineering_only") is not False
        or metadata.get("registry_role") != "validation"
        or metadata.get("producer_sha256") != _producer_digest()
        or metadata.get("registry_sha256") != registry["registry_sha256"]
        or recorded != _digest(unsigned)
    ):
        raise ValueError("I normal metadata is not an active validation record")
    factor_paths = {
        "normal_left_basis": Path(left_basis_path).resolve(),
        "singular_values": Path(singular_values_path).resolve(),
    }
    for name, source in factor_paths.items():
        if metadata["artifact_sha256"].get(name) != sha256_path(source):
            raise ValueError(f"I normal factor {name} changed after N production")
    left_basis = np.load(factor_paths["normal_left_basis"], mmap_mode="r")
    singular = np.load(factor_paths["singular_values"], mmap_mode="r")
    rank = int(metadata["normal_rank"])
    if left_basis.shape != (int(metadata["stencil_output_dimension"]), rank):
        raise ValueError("I left singular basis has a wrong shape")
    if singular.ndim != 1 or singular.size < rank or np.any(singular[:rank] <= 0):
        raise ValueError("I retained singular values are malformed")
    return metadata, left_basis, singular


def _intervention_statistics(series):
    series = np.asarray(series, dtype=np.float64)
    if series.ndim != 1 or series.size < 2 or np.any(series < 0):
        raise ValueError("an intervention normal series is malformed")
    log_ratio = np.diff(np.log(np.maximum(series, np.finfo(float).tiny)))
    velocity = np.diff(series)
    signs = np.sign(velocity)
    changes = np.flatnonzero(signs[1:] * signs[:-1] < 0)
    return (
        float(np.mean(log_ratio)),
        float(np.sum(np.arctan2(
            velocity, np.maximum(series[1:], np.finfo(float).tiny)))),
        float(changes[0] + 1 if changes.size else len(velocity)),
    )


def intervention(arguments):
    """Measure rate branches in the finite-stencil SVD normal coordinate."""

    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    sources = load_checkpoint_set(
        arguments.source_manifest, arguments.artifact_root)
    low = load_checkpoint_set(arguments.low_manifest, arguments.artifact_root)
    high = load_checkpoint_set(arguments.high_manifest, arguments.artifact_root)
    if len(sources) != 1 or len(low) != 16 or len(high) != 16:
        raise ValueError("I needs one source and sixteen snapshots per rate arm")
    source = sources[0]
    branches = (low, high)
    registry = load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("I token archive disagrees with the registry")
    metadata, left_basis, singular = _normal_coordinate_factors(
        arguments.normal_metadata, arguments.normal_left_basis,
        arguments.singular_values, registry)
    layer_index = int(metadata["layer"])
    source_digest = source["state_manifest"]["complete_state_sha256"]
    source_step = int(source["step"])
    branch_run_ids = []
    branch_non_rate = []
    branch_rates = []
    for branch in branches:
        if any(checkpoint["measurement_registry"] != registry
               for checkpoint in branch):
            raise ValueError("an I branch changed the measurement registry")
        steps = [int(checkpoint["step"]) for checkpoint in branch]
        if steps[0] <= source_step or any(
                right <= left for left, right in zip(steps, steps[1:])):
            raise ValueError("an I branch is not an ordered continuation")
        identities = [checkpoint.get("run_identity") for checkpoint in branch]
        run_ids = {
            value.get("run_id") for value in identities
            if isinstance(value, dict)}
        lineages = {
            value.get("lineage_root") for value in identities
            if isinstance(value, dict)}
        if (
            len(run_ids) != 1 or None in run_ids
            or len(lineages) != 1 or None in lineages
            or any(
                not isinstance(value, dict)
                or value.get("from_scratch") is not False
                or value.get("source_checkpoint_sha256") != source_digest
                for value in identities)
        ):
            raise ValueError("an I branch lacks a source-bound continuation identity")
        branch_run_ids.append(next(iter(run_ids)))
        non_rate = {_non_rate_digest(checkpoint) for checkpoint in branch}
        if len(non_rate) != 1:
            raise ValueError("non-rate configuration changed within an I branch")
        branch_non_rate.append(next(iter(non_rate)))
        rates = {
            float(checkpoint["config"].get("learning_rate_multiplier", 1.0))
            for checkpoint in branch}
        if len(rates) != 1:
            raise ValueError("the rate multiplier changed within an I branch")
        branch_rates.append(next(iter(rates)))
    if len(set(branch_run_ids)) != 2:
        raise ValueError("I rate arms must have distinct run identities")
    if len(set(branch_non_rate)) != 1:
        raise ValueError("I rate arms differ in a non-rate configuration field")
    if len(set(branch_rates)) != 2:
        raise ValueError("I rate arms must use distinct registered multipliers")
    if [checkpoint["step"] for checkpoint in low] != [
            checkpoint["step"] for checkpoint in high]:
        raise ValueError("I rate arms must own identical step nodes")

    def coordinate(checkpoint):
        vector, _maximum, _direct = _checkpoint_stencil_observables(
            checkpoint, arguments.tokens, registry, device, layer_index,
            measure_direct=False)
        result = linearized_normal_coordinates(
            vector[None, :], left_basis, singular)
        return (
            float(result["amplitudes"][0]),
            float(result["stencil_reconstruction_residuals"][0]),
        )

    source_amplitude, source_residual = coordinate(source)
    amplitudes = []
    residuals = []
    damping = []
    phase = []
    onset = []
    second = []
    for branch in branches:
        measured = [coordinate(checkpoint) for checkpoint in branch]
        series = np.asarray(
            [source_amplitude, *[value[0] for value in measured]],
            dtype=np.float64)
        series_residuals = np.asarray(
            [source_residual, *[value[1] for value in measured]],
            dtype=np.float64)
        branch_damping, branch_phase, branch_onset = _intervention_statistics(
            series)
        amplitudes.append(series)
        residuals.append(series_residuals)
        damping.append(branch_damping)
        phase.append(branch_phase)
        onset.append(branch_onset)
        second.append(all(_has_second_moment(value) for value in branch))
    _write_npz(arguments.output, {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "source_manifest_sha256": np.asarray(
            sha256_path(arguments.source_manifest)),
        "low_manifest_sha256": np.asarray(sha256_path(arguments.low_manifest)),
        "high_manifest_sha256": np.asarray(sha256_path(arguments.high_manifest)),
        "normal_metadata_sha256": np.asarray(
            sha256_path(arguments.normal_metadata)),
        "normal_left_basis_sha256": np.asarray(
            sha256_path(arguments.normal_left_basis)),
        "singular_values_sha256": np.asarray(
            sha256_path(arguments.singular_values)),
        "tokens_sha256": np.asarray(sha256_path(arguments.tokens)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "source_checkpoint_state_sha256": np.asarray(source_digest),
        "branch_checkpoint_state_sha256": np.asarray([
            [value["state_manifest"]["complete_state_sha256"]
             for value in branch] for branch in branches]),
        "source_step": np.asarray(source_step, dtype=np.int64),
        "branch_steps": np.asarray([
            [int(value["step"]) for value in branch]
            for branch in branches], dtype=np.int64),
        "branch_run_ids": np.asarray(branch_run_ids),
        "branch_non_rate_digest": np.asarray(branch_non_rate),
        "rate_multiplier": np.asarray(branch_rates, dtype=np.float64),
        "normal_coordinate_rule": np.asarray("svd-linearized-least-norm-v1"),
        "normal_amplitudes": np.asarray(amplitudes, dtype=np.float64),
        "normal_reconstruction_residuals": np.asarray(
            residuals, dtype=np.float64),
        "damping": np.asarray(damping, dtype=np.float64),
        "phase": np.asarray(phase, dtype=np.float64),
        "oscillation_onset": np.asarray(onset, dtype=np.float64),
        "second_moment_used": np.asarray(second, dtype=np.bool_),
        **_resource_arrays(device, started),
    })


def _path_logits(model, tokens, layer_index, replacement):
    module = model.decoder.dec_layers[layer_index].mha1.plgatt_layer

    def hook(_module, inputs):
        values = list(inputs[0])
        values[3] = replacement
        return (values,)

    handle = module.register_forward_pre_hook(hook)
    try:
        device = next(model.parameters()).device
        inputs = tokens[:, :-1].to(device)
        mask = _causal_mask(
            inputs.shape[1], device, next(model.parameters()).dtype)
        with torch.no_grad():
            logits = model([inputs, mask])[0]
    finally:
        handle.remove()
    value = logits[0, -1].detach().double().cpu().numpy()
    return value - np.mean(value)


def _construction_application_batches(tokens_path, registry):
    rows = registry["construction"]
    if len(rows) < 2:
        raise ValueError("application construction needs at least two contexts")
    return [
        (
            f"ac{index:04d}",
            _token_batch(tokens_path, registry, rows[2 * index]),
            _token_batch(tokens_path, registry, rows[2 * index + 1]),
        )
        for index in range(len(rows) // 2)
    ]


def _locked_application_certificate(lock, subdivisions):
    candidates = []
    center_steps = set()
    for value in lock.get("application", []):
        if not isinstance(value, dict):
            continue
        if int(value.get("subdivisions", -1)) != subdivisions:
            continue
        radii = np.asarray(value.get("cell_average_radii", []), dtype=float)
        if radii.shape != (subdivisions,) or np.any(radii <= 0):
            raise ValueError("A prediction lock has malformed cell radii")
        candidates.append(radii)
        center_steps.add(float(value.get("center_finite_difference_step", -1)))
    if not candidates or len(center_steps) != 1:
        raise ValueError("A prediction lock has no unique matching certificate")
    center_step = next(iter(center_steps))
    if not math.isfinite(center_step) or center_step <= 0:
        raise ValueError("A prediction lock has an invalid center step")
    return np.maximum.reduce(candidates), center_step


def application(arguments):
    """Evaluate construction or held-out oriented PLGA intervention paths."""

    checkpoints = load_checkpoint_set(
        arguments.checkpoint_manifest, arguments.artifact_root)
    if not checkpoints:
        raise ValueError("A needs at least one complete checkpoint")
    registry = load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("A token archive disagrees with the registry")
    if arguments.registry_role == "construction":
        pair_records = _construction_application_batches(
            arguments.tokens, registry)
    else:
        pair_records = _application_batches(
            checkpoints[0], arguments.tokens,
            ARCHITECTURE["application_pairs"])
    if not pair_records:
        raise ValueError("A selected an empty pair registry")
    subdivisions = int(arguments.subdivisions)
    if subdivisions <= 0:
        raise ValueError("A subdivisions must be positive")
    if arguments.registry_role == "validation" and not arguments.prediction_lock:
        raise ValueError("validation A requires a sealed prediction lock")
    if arguments.registry_role == "construction" and arguments.prediction_lock:
        raise ValueError("construction A cannot consume a prediction lock")
    locked_radii = None
    locked_center_step = None
    if arguments.prediction_lock:
        with Path(arguments.prediction_lock).open(
                "r", encoding="utf-8") as stream:
            lock = json.load(stream)
        unsigned = dict(lock)
        recorded = unsigned.pop("lock_sha256", None)
        if (
            lock.get("schema_version") != "pldr-prediction-lock-v4"
            or recorded != _digest(unsigned)
            or lock.get("partition_sha256") != registry["registry_sha256"]
        ):
            raise ValueError("A prediction lock does not replay against the registry")
        locked_radii, locked_center_step = _locked_application_certificate(
            lock, subdivisions)
    widths = np.full(subdivisions, 1.0 / subdivisions)
    finite_step = min(1e-4, 0.1 / subdivisions)
    if (
        locked_center_step is not None
        and not math.isclose(
            finite_step, locked_center_step, rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("A center step does not match the prediction lock")
    centers = []
    radii = []
    residual_norms = []
    node_logits = []
    references = []
    comparisons = []
    margins = []
    pair_ids = []
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    for pair_index, (identifier, left, right) in enumerate(pair_records):
        checkpoint = checkpoints[pair_index % len(checkpoints)]
        if checkpoint["measurement_registry"] != registry:
            raise ValueError("an A checkpoint changed the measurement registry")
        model = _model_from_checkpoint(checkpoint, device).double()
        layer_index = int(arguments.layer)
        left_stages, _ = _capture(model, left)
        left_value = left_stages[layer_index]["A"]
        left_rows = _capture_physical_rows(model, [left], layer_index)
        right_rows = _capture_physical_rows(model, [right], layer_index)
        if left_rows.shape != right_rows.shape:
            raise ValueError("an A source-donor row pair has incompatible shapes")
        layout, parameters = _selected_parameter_tuple(model, layer_index)
        mapping = dict(zip(layout.names, parameters))

        def row_map_at(value):
            rows = left_rows + float(value) * (right_rows - left_rows)
            with torch.no_grad():
                mapped = _row_map_apply(
                    model, layer_index, mapping, rows)
            return mapped.reshape(left_value.shape)

        source_endpoint_error = float(torch.linalg.vector_norm(
            row_map_at(0.0) - left_value).detach().cpu())
        if source_endpoint_error > 1e-10:
            raise ValueError("A source endpoint does not replay through the row map")

        def path(value):
            return _path_logits(
                model, left, layer_index, row_map_at(value))

        nodes = np.linspace(0.0, 1.0, subdivisions + 1)
        logits = np.asarray([path(value) for value in nodes])
        local_centers = []
        local_residuals = []
        for index in range(subdivisions):
            left_node = nodes[index]
            right_node = nodes[index + 1]
            midpoint = 0.5 * (left_node + right_node)
            derivative = (
                path(midpoint + finite_step)
                - path(midpoint - finite_step)
            ) / (2 * finite_step)
            secant = (logits[index + 1] - logits[index]) / widths[index]
            residual = math.nextafter(
                float(np.linalg.norm(secant - derivative)), math.inf)
            local_centers.append(derivative)
            local_residuals.append(residual)
        reference = logits[0]
        comparison = logits[-1]
        ordered = np.sort(reference)
        margin = float(ordered[-1] - ordered[-2])
        centers.append(local_centers)
        residual_norms.append(local_residuals)
        node_logits.append(logits)
        references.append(reference)
        comparisons.append(comparison)
        margins.append(margin)
        pair_ids.append(identifier)
    if locked_radii is None:
        radii = np.asarray(residual_norms)
    else:
        radii = np.broadcast_to(
            locked_radii, (len(pair_ids), subdivisions)).copy()
    _write_npz(arguments.output, {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "checkpoint_state_sha256": np.asarray([
            value["state_manifest"]["complete_state_sha256"]
            for value in checkpoints]),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "registry_role": np.asarray(arguments.registry_role),
        "pair_ids": np.asarray(pair_ids),
        "integral_centers": np.asarray(centers),
        "integral_radii": np.asarray(radii),
        "cell_average_residual_norms": np.asarray(residual_norms),
        "path_node_logits": np.asarray(node_logits),
        "center_finite_difference_step": np.asarray(finite_step),
        "interval_widths": np.broadcast_to(
            widths, (len(pair_ids), subdivisions)).copy(),
        "reference_logits": np.asarray(references),
        "comparison_logits": np.asarray(comparisons),
        "reference_margins": np.asarray(margins),
        **_resource_arrays(device, started),
    })


def construction_summary(arguments):
    """Freeze numeric construction outputs before validation artifacts open."""

    registry = load_registry(arguments.registry)
    with Path(arguments.normal_metadata).open("r", encoding="utf-8") as stream:
        normal_metadata = json.load(stream)
    if (
        normal_metadata.get("producer_sha256") != _producer_digest()
        or normal_metadata.get("registry_role") != "construction"
        or normal_metadata.get("engineering_only") is not False
        or normal_metadata.get("registry_sha256") != registry["registry_sha256"]
    ):
        raise ValueError("construction N metadata is not an active complete record")
    with np.load(arguments.normal_actions, allow_pickle=False) as normal:
        pair_intervals = np.asarray(normal["pair_ritz_intervals"], dtype=float)
        full_intervals = np.asarray(
            normal["full_loss_ritz_intervals"], dtype=float)
        pair_ritz_count = int(normal["pair_ritz_values"].size)
        full_ritz_count = int(normal["full_loss_ritz_values"].size)
        gradients = np.asarray(normal["normal_gradient_samples"], dtype=float)
        signed = np.asarray(
            normal["signed_second_challenge_quadratics"], dtype=float)
    with np.load(arguments.dynamics, allow_pickle=False) as raw:
        if raw["producer_code_sha256"].item() != _producer_digest():
            raise ValueError("construction D archive has foreign provenance")
        metrics = np.asarray(raw["metrics"], dtype=float)
        gains = np.asarray(raw["gains"], dtype=float)
        quadratic = np.asarray(raw["quadratic"], dtype=float)
        forces = np.asarray(raw["forces"], dtype=float)
        radii = np.asarray(raw["radii"], dtype=float)
        dynamics_node_steps = np.asarray(raw["node_steps"], dtype=np.int64)
        dynamics_operator_block_offsets = np.asarray(
            raw["operator_block_offsets"], dtype=np.int64)
        dynamics_node_block_offsets = np.asarray(
            raw["node_block_offsets"], dtype=np.int64)
        dynamics_direct_coefficients = np.asarray(
            raw["direct_coefficients"], dtype=float)
        dynamics_direct_remainders = np.asarray(
            raw["direct_remainders"], dtype=float)
    with np.load(arguments.intervention, allow_pickle=False) as raw:
        if raw["producer_code_sha256"].item() != _producer_digest():
            raise ValueError("construction I archive has foreign provenance")
        intervention_order = {
            "source_step": int(raw["source_step"].item()),
            "damping": np.argsort(raw["damping"]).tolist(),
            "phase": np.argsort(raw["phase"]).tolist(),
            "oscillation_onset":
                np.argsort(raw["oscillation_onset"]).tolist(),
        }
    with np.load(arguments.application, allow_pickle=False) as raw:
        if (
            raw["producer_code_sha256"].item() != _producer_digest()
            or raw["registry_role"].item() != "construction"
        ):
            raise ValueError("construction A archive has foreign provenance")
        application_centers = np.asarray(raw["integral_centers"], dtype=float)
        application_nodes = np.asarray(raw["path_node_logits"], dtype=float)
        application_widths = np.asarray(raw["interval_widths"], dtype=float)
        application_recorded_residuals = np.asarray(
            raw["cell_average_residual_norms"], dtype=float)
        application_center_step = float(
            raw["center_finite_difference_step"].item())
    if (
        int(normal_metadata["krylov_iterations"])
            != int(normal_metadata["normal_rank"])
        or pair_ritz_count != int(normal_metadata["normal_rank"])
        or full_ritz_count != int(normal_metadata["normal_rank"])
    ):
        raise ValueError(
            "construction N needs a complete normal-dimensional spectrum")
    if (
        application_nodes.shape[0] == 0
        or application_nodes.shape[1] != application_centers.shape[1] + 1
        or application_widths.shape != application_centers.shape[:2]
        or application_centers.shape[1]
            != int(arguments.application_subdivisions)
    ):
        raise ValueError("construction A archive has malformed path arrays")
    application_secants = (
        np.diff(application_nodes, axis=1)
        / application_widths[:, :, None]
    )
    application_residuals = np.linalg.norm(
        application_secants - application_centers, axis=2)
    if not np.allclose(
            application_residuals, application_recorded_residuals,
            rtol=1e-12, atol=1e-12):
        raise ValueError("construction A residuals do not replay")
    products = [
        float(value)
        for row in normal_metadata["action_contexts"]
        for value in row["realized_probability_products"]
    ]
    if not products:
        raise ValueError("construction selected-edge products are empty")
    safety = float(arguments.safety_factor)
    if not 0 < safety <= 1:
        raise ValueError("construction safety factor must lie in (0, 1]")
    force = float(np.linalg.norm(np.mean(gradients, axis=0)))
    second = float(np.max(np.abs(signed), initial=0.0))
    gradient_bound = (
        float(normal_metadata["normal_gradient_clip_value"])
        * math.sqrt(int(normal_metadata["parameter_count"])))
    centered_gradient_bound = 2.0 * gradient_bound
    gradient_variance_bound = gradient_bound ** 2
    failure_probability = float(
        ARCHITECTURE["finite_batch_failure_probability"])
    finite_batch_radius = vector_bernstein_radius(
        gradient_variance_bound, centered_gradient_bound,
        ARCHITECTURE["validation_contexts"],
        int(normal_metadata["normal_rank"]), failure_probability)
    population_force_bound = math.nextafter(force / safety, math.inf)
    normal_force_envelope = math.nextafter(
        population_force_bound + finite_batch_radius, math.inf)
    summary = {
        "schema_version": "pldr-construction-summary-v4",
        "code_sha256": digest_object({
            "producer": _producer_digest(),
            "analyzer": sha256_path(
                EXPERIMENTS / "analysis"
                / "comprehensive_confirmation_analysis.py"),
        }),
        "partition_sha256": registry["registry_sha256"],
        "schedule": {
            "node_steps": dynamics_node_steps.tolist(),
            "operator_block_offsets":
                dynamics_operator_block_offsets.tolist(),
            "node_block_offsets": dynamics_node_block_offsets.tolist(),
            "gains": gains.tolist(),
            "quadratic": quadratic.tolist(),
            "forces": forces.tolist(),
        },
        "selected_edges": {
            "pair_product_lower": math.nextafter(
                safety * min(products), 0.0),
            "rule": registry["selected_edge_rule"],
        },
        "intervals": {
            "pair_edge_lower": float(np.min(pair_intervals[:, 0])),
            "full_loss_edge_lower": float(np.min(full_intervals[:, 0])),
            "construction_balance_force_bound": population_force_bound,
            "finite_batch_failure_probability": failure_probability,
            "finite_batch_variance_bound": gradient_variance_bound,
            "finite_batch_centered_bound": centered_gradient_bound,
            "finite_batch_sample_count":
                ARCHITECTURE["validation_contexts"],
            "finite_batch_dimension": int(normal_metadata["normal_rank"]),
            "finite_batch_force_radius": finite_batch_radius,
            "normal_force_max": normal_force_envelope,
            "balance_second_abs_max": math.nextafter(
                second / safety, math.inf),
            "metric_condition_max": math.nextafter(
                max(np.linalg.cond(value) for value in metrics) / safety,
                math.inf),
            "intervention_order": intervention_order,
        },
        "path_metrics": {"metrics": metrics.tolist()},
        "radii": {"values": radii.tolist()},
        "direct_conversion": {
            "rule": "construction-locked-affine-normal-to-direct-v1",
            "coefficient": math.nextafter(
                float(np.max(dynamics_direct_coefficients)) / safety,
                math.inf),
            "remainder": math.nextafter(
                float(np.max(dynamics_direct_remainders)) / safety,
                math.inf),
            "singular_max": float(np.max(dynamics_direct_coefficients)),
        },
        "application": {
            "subdivisions": int(arguments.application_subdivisions),
            "oriented_center_rule":
                "midpoint-derivative-with-locked-cell-average-residual-v2",
            "construction_pair_count": int(application_nodes.shape[0]),
            "center_finite_difference_step": application_center_step,
            "cell_average_radii": [
                math.nextafter(float(value) / safety, math.inf)
                for value in np.max(application_residuals, axis=0)
            ],
        },
    }
    write_json_atomic(arguments.output, summary)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry = subparsers.add_parser("registry")
    registry.add_argument("--tokens", required=True)
    registry.add_argument("--tokenizer", required=True)
    registry.add_argument("--context-length", type=int, default=256)
    registry.add_argument("--construction-contexts", type=int, default=48)
    registry.add_argument("--validation-contexts", type=int, default=32)
    registry.add_argument("--application-pairs", type=int, default=32)
    registry.add_argument("--reserved-start", type=int, default=-1)
    registry.add_argument("--stencil-step", type=float, default=2 ** -10)
    registry.add_argument("--direction-seed", type=int, default=340021)
    registry.add_argument("--challenge-seed", type=int, default=340031)
    registry.add_argument("--edge-alternatives", type=int, default=8)
    registry.add_argument(
        "--cover-directions-per-row", type=int,
        default=ARCHITECTURE["cover_directions_per_row"])
    registry.add_argument("--output", required=True)
    registry.set_defaults(function=create_registry)

    qualification = subparsers.add_parser("qualify")
    qualification.add_argument("--device", default="cuda:0")
    qualification.add_argument("--output", required=True)
    qualification.set_defaults(function=qualify)

    normal_parser = subparsers.add_parser("normal")
    normal_parser.add_argument("--checkpoint", required=True)
    normal_parser.add_argument("--tokens", required=True)
    normal_parser.add_argument("--registry", required=True)
    normal_parser.add_argument("--device", default="cuda:0")
    normal_parser.add_argument("--layer", type=int, default=0)
    normal_parser.add_argument("--context-limit", type=int, default=0)
    normal_parser.add_argument(
        "--registry-role", choices=["construction", "validation"],
        default="construction")
    normal_parser.add_argument("--pair-product-lower", type=float, default=0.0)
    normal_parser.add_argument(
        "--action-contexts", type=int, default=0,
        help="zero uses every context in the selected registry role")
    normal_parser.add_argument("--tangent-block-size", type=int, default=8)
    normal_parser.add_argument("--rank-tolerance", type=float, default=1e-10)
    normal_parser.add_argument(
        "--krylov-iterations", type=int, default=0,
        help="zero runs a complete normal-dimensional factorization")
    normal_parser.add_argument("--challenge-directions", type=int, default=4)
    normal_parser.add_argument("--engineering-only", action="store_true")
    normal_parser.add_argument("--output-root", required=True)
    normal_parser.set_defaults(function=normal)

    trajectory_parser = subparsers.add_parser("trajectory")
    trajectory_parser.add_argument("--checkpoint-manifest", required=True)
    trajectory_parser.add_argument("--artifact-root", required=True)
    trajectory_parser.add_argument("--tokens", required=True)
    trajectory_parser.add_argument("--registry", required=True)
    trajectory_parser.add_argument("--device", default="cuda:0")
    trajectory_parser.add_argument("--layer", type=int, default=0)
    trajectory_parser.add_argument("--output", required=True)
    trajectory_parser.set_defaults(function=trajectory)

    dynamics_parser = subparsers.add_parser("dynamics")
    dynamics_parser.add_argument(
        "--snapshot", action="append", required=True)
    dynamics_parser.add_argument("--normal-metadata", required=True)
    dynamics_parser.add_argument("--normal-left-basis", required=True)
    dynamics_parser.add_argument("--singular-values", required=True)
    dynamics_parser.add_argument("--normal-actions", required=True)
    dynamics_parser.add_argument("--direct-trajectory", required=True)
    dynamics_parser.add_argument("--prediction-lock")
    dynamics_parser.add_argument("--tensor-group", default="phi")
    dynamics_parser.add_argument("--output", required=True)
    dynamics_parser.set_defaults(function=dynamics)

    intervention_parser = subparsers.add_parser("intervention")
    intervention_parser.add_argument("--source-manifest", required=True)
    intervention_parser.add_argument("--low-manifest", required=True)
    intervention_parser.add_argument("--high-manifest", required=True)
    intervention_parser.add_argument("--artifact-root", required=True)
    intervention_parser.add_argument("--tokens", required=True)
    intervention_parser.add_argument("--registry", required=True)
    intervention_parser.add_argument("--normal-metadata", required=True)
    intervention_parser.add_argument("--normal-left-basis", required=True)
    intervention_parser.add_argument("--singular-values", required=True)
    intervention_parser.add_argument("--device", default="cuda:0")
    intervention_parser.add_argument("--output", required=True)
    intervention_parser.set_defaults(function=intervention)

    application_parser = subparsers.add_parser("application")
    application_parser.add_argument("--checkpoint-manifest", required=True)
    application_parser.add_argument("--artifact-root", required=True)
    application_parser.add_argument("--tokens", required=True)
    application_parser.add_argument("--registry", required=True)
    application_parser.add_argument("--device", default="cuda:0")
    application_parser.add_argument("--layer", type=int, default=0)
    application_parser.add_argument("--subdivisions", type=int, default=8)
    application_parser.add_argument(
        "--registry-role", choices=["construction", "validation"],
        default="validation")
    application_parser.add_argument("--prediction-lock")
    application_parser.add_argument("--output", required=True)
    application_parser.set_defaults(function=application)

    summary_parser = subparsers.add_parser("construction-summary")
    summary_parser.add_argument("--registry", required=True)
    summary_parser.add_argument("--normal-metadata", required=True)
    summary_parser.add_argument("--normal-actions", required=True)
    summary_parser.add_argument("--dynamics", required=True)
    summary_parser.add_argument("--intervention", required=True)
    summary_parser.add_argument("--application", required=True)
    summary_parser.add_argument("--safety-factor", type=float, default=0.9)
    summary_parser.add_argument(
        "--application-subdivisions", type=int, default=8)
    summary_parser.add_argument("--output", required=True)
    summary_parser.set_defaults(function=construction_summary)

    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
