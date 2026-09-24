#!/usr/bin/env python3
"""Live, checkpoint-bound producers for the T/N/I/A confirmation stages.

No subcommand accepts caller-created scientific tensors.  Every numeric
observation is computed from a validated complete checkpoint, its immutable
measurement registry, the bound token file, and the model implementation in
this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import hessian, jacrev


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirmation_artifacts import (  # noqa: E402
    digest_object,
    load_checkpoint_set,
    load_json_object,
    nested_state_digest,
    sha256_path,
    validate_measurement_registry,
    write_json_atomic,
)
from pldr_model_v510 import PLDR_Model  # noqa: E402


PRODUCER_SCHEMA = "pldr-live-composite-producer-v3"
PARTITION_SCHEMA = "pldr-confirmation-partitions-v3"


def _producer_digest():
    return sha256_path(Path(__file__))


def _write_npz(path, values):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **values)
    temporary.replace(destination)


def _mask(length, device, dtype):
    return torch.triu(
        torch.ones(length, length, device=device, dtype=dtype), diagonal=1,
    )[None, None]


def create_measurement_registry(arguments):
    dataset = Path(arguments.dataset)
    tokenizer = Path(arguments.tokenizer)
    context = int(arguments.context_length)
    tokens = np.memmap(dataset, dtype=np.uint16, mode="r")
    chunks = len(tokens) // context
    required = (
        arguments.construction_count
        + arguments.validation_count
        + 2 * arguments.application_count
    )
    start = (
        int(arguments.reserved_start)
        if arguments.reserved_start >= 0
        else chunks - max(required + 64, arguments.reserve_chunks)
    )
    if start < 0 or start + required > chunks:
        raise ValueError("the requested immutable registry exceeds the dataset")
    cursor = start

    def rows(prefix, count):
        nonlocal cursor
        result = [
            {"id": f"{prefix}{index:04d}", "chunk_index": cursor + index}
            for index in range(count)
        ]
        cursor += count
        return result

    construction = rows("c", arguments.construction_count)
    validation = rows("v", arguments.validation_count)
    application = []
    for index in range(arguments.application_count):
        application.append({
            "id": f"a{index:04d}",
            "left_chunk": cursor + 2 * index,
            "right_chunk": cursor + 2 * index + 1,
        })
    registry = {
        "schema_version": "pldr-measurement-registry-v3",
        "context_length": context,
        "construction": construction,
        "validation": validation,
        "application_pairs": application,
        "dataset_sha256": sha256_path(dataset),
        "tokenizer_sha256": sha256_path(tokenizer),
    }
    validate_measurement_registry(registry, require_nonempty=True)
    write_json_atomic(arguments.output, registry)
    if arguments.partition_output:
        partition = {
            "schema_version": PARTITION_SCHEMA,
            "construction_ids": [row["id"] for row in construction],
            "validation_ids": [row["id"] for row in validation],
            "application_ids": [row["id"] for row in application],
            "dataset_sha256": registry["dataset_sha256"],
            "tokenizer_sha256": registry["tokenizer_sha256"],
            "measurement_registry_sha256": digest_object(registry),
        }
        write_json_atomic(arguments.partition_output, partition)


def _registry_batches(checkpoint, token_path, role, limit=None):
    registry = checkpoint["measurement_registry"]
    validate_measurement_registry(registry, require_nonempty=True)
    if sha256_path(token_path) != registry["dataset_sha256"]:
        raise ValueError("token file does not match the checkpoint registry")
    context = registry["context_length"]
    data = np.memmap(token_path, dtype=np.uint16, mode="r")
    chunks = len(data) // context
    matrix = data[:chunks * context].reshape(chunks, context)
    rows = registry[role]
    if limit is not None:
        rows = rows[:int(limit)]
    return [
        (
            row["id"],
            torch.from_numpy(
                matrix[row["chunk_index"]].astype(np.int64, copy=True)
            )[None],
        )
        for row in rows
    ]


def _application_batches(checkpoint, token_path, limit=None):
    registry = checkpoint["measurement_registry"]
    validate_measurement_registry(registry, require_nonempty=True)
    if sha256_path(token_path) != registry["dataset_sha256"]:
        raise ValueError("token file does not match the checkpoint registry")
    context = registry["context_length"]
    data = np.memmap(token_path, dtype=np.uint16, mode="r")
    chunks = len(data) // context
    matrix = data[:chunks * context].reshape(chunks, context)
    rows = registry["application_pairs"]
    if limit is not None:
        rows = rows[:int(limit)]
    return [
        (
            row["id"],
            torch.from_numpy(
                matrix[row["left_chunk"]].astype(np.int64, copy=True)
            )[None],
            torch.from_numpy(
                matrix[row["right_chunk"]].astype(np.int64, copy=True)
            )[None],
        )
        for row in rows
    ]


def _model_from_checkpoint(checkpoint, device):
    config = checkpoint["model_config"]
    model = PLDR_Model(
        num_layers=int(config["layers"]),
        d_model=int(config["width"]),
        num_heads=int(config["heads"]),
        dff=int(config["feed_forward_width"]),
        input_vocab_size=int(config["vocabulary_size"]),
        A_dff=int(config["row_hidden_width"]),
        num_reslayerA=int(config["row_residual_layers"]),
        num_denseA=int(config["row_dense_layers"]),
        max_seq_len=int(config["maximum_sequence_length"]),
        device=device,
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def _capture(model, tokens):
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = tokens[:, :-1].to(device)
    causal = _mask(inputs.shape[1], device, dtype)
    captured = [dict() for _ in model.decoder.dec_layers]
    handles = []

    for layer_index, decoder_layer in enumerate(model.decoder.dec_layers):
        def pre_hook(_module, hook_inputs, index=layer_index):
            values = hook_inputs[0]
            captured[index]["plga_inputs"] = tuple(
                value.detach() for value in values)

        def plga_hook(_module, _inputs, output, index=layer_index):
            captured[index]["plga_output"] = (
                output[0].detach(),
                tuple(value.detach() for value in output[1]),
            )

        def residual_hook(_module, hook_inputs, output, index=layer_index):
            captured[index]["residual_input"] = hook_inputs[0].detach()
            captured[index]["residual"] = output.detach()

        def layernorm_hook(_module, hook_inputs, output, index=layer_index):
            captured[index]["layernorm_input"] = hook_inputs[0].detach()
            captured[index]["layernorm"] = output.detach()

        handles.extend([
            decoder_layer.mha1.plgatt_layer.register_forward_pre_hook(
                pre_hook),
            decoder_layer.mha1.plgatt_layer.register_forward_hook(plga_hook),
            decoder_layer.layernorm1.register_forward_hook(residual_hook),
            decoder_layer.layernorm2.register_forward_hook(layernorm_hook),
        ])
    with torch.no_grad():
        logits, _, _, _ = model([inputs, causal])
    for handle in handles:
        handle.remove()

    stages = []
    for layer in captured:
        q, k, v, post_row_map, local_mask = layer["plga_inputs"]
        attention_output, weights = layer["plga_output"]
        a_lm, powers, _coupling, _bias, g_lm, scores = weights
        a_p = torch.pow(a_lm, powers[None])
        probabilities = torch.softmax(scores + local_mask * -1e9, dim=-1)
        centered = (
            layer["layernorm_input"]
            - layer["layernorm_input"].mean(dim=-1, keepdim=True)
        )
        stages.append({
            "q": q,
            "k": k,
            "v": v,
            "mask": local_mask,
            "A": post_row_map,
            "A_LM": a_lm,
            "A_P": a_p,
            "G_LM": g_lm,
            "scores": scores,
            "softmax": probabilities,
            "attention_output": attention_output,
            "residual": layer["residual"],
            "layer_norm": layer["layernorm"],
            "layernorm_centered_floor": float(
                torch.linalg.vector_norm(centered, dim=-1).min()),
        })
    return stages, logits.detach()


def _component_names(checkpoint):
    return sorted(
        name for name in checkpoint["state_manifest"]
        if name != "complete_state_sha256"
    )


def _component_row(checkpoint, names):
    manifest = checkpoint["state_manifest"]
    if sorted(name for name in manifest if name != "complete_state_sha256") != names:
        raise ValueError("checkpoint state-manifest component registry changed")
    return [
        value if isinstance(value, str) else digest_object(value)
        for value in (manifest[name] for name in names)
    ]


def _schedule_rate(checkpoint):
    config = checkpoint["config"]
    step = int(checkpoint["step"])
    base = float(config["lr"]) * float(
        config.get("learning_rate_multiplier", 1.0))
    warmup = max(1, int(config["warmup"]))
    if config.get("const_lr", False):
        factor = min(1.0, (step + 1) / warmup)
    else:
        total = int(checkpoint["schedule_total_steps"])
        floor = float(config.get("anneal_floor", 0.1))
        hold = int(config.get("hold_until", -1))
        if hold >= 0:
            if step <= warmup and warmup > 1:
                factor = step / warmup
            elif step <= hold:
                factor = 1.0
            else:
                phase = (min(step, total) - hold) / (total - hold)
                factor = (
                    (1 - floor) * 0.5
                    * (1 + math.cos(math.pi * phase)) + floor
                )
        elif step <= warmup:
            factor = step / warmup
        else:
            phase = (min(step, total) - warmup) / (total - warmup)
            factor = (
                (1 - floor) * 0.5
                * (1 + math.cos(math.pi * phase)) + floor
            )
    return base * factor


def training_observations(arguments):
    landmarks = load_checkpoint_set(
        arguments.landmark_manifest, arguments.artifact_root)
    replays = load_checkpoint_set(
        arguments.replay_manifest, arguments.artifact_root)
    if len(landmarks) != 4 or len(replays) != 1:
        raise ValueError("T requires four primary landmarks and one replay target")
    names = _component_names(landmarks[0])
    values = {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "component_names": np.asarray(names),
        "landmark_component_digests": np.asarray([
            _component_row(checkpoint, names) for checkpoint in landmarks
        ]),
        "replay_component_digests": np.asarray([
            _component_row(checkpoint, names) for checkpoint in replays
        ]),
        "landmark_complete_digests": np.asarray([
            checkpoint["state_manifest"]["complete_state_sha256"]
            for checkpoint in landmarks
        ]),
        "replay_complete_digests": np.asarray([
            checkpoint["state_manifest"]["complete_state_sha256"]
            for checkpoint in replays
        ]),
        "applied_learning_rates": np.asarray([
            checkpoint["optimizer_config"]["groups"][0]["learning_rate"]
            for checkpoint in landmarks
        ], dtype=np.float64).reshape(1, 4),
        "recomputed_learning_rates": np.asarray([
            _schedule_rate(checkpoint) for checkpoint in landmarks
        ], dtype=np.float64).reshape(1, 4),
        "landmark_steps": np.asarray([
            checkpoint["step"] for checkpoint in landmarks
        ], dtype=np.int64).reshape(1, 4),
        "seed_ids": np.asarray([
            landmarks[0]["config"]["seed"],
        ], dtype=np.int64),
        "data_cursors": np.asarray([
            checkpoint["data_offset_end"] for checkpoint in landmarks
        ], dtype=np.int64),
        "model_shape": np.asarray([
            landmarks[0]["model_config"]["width"],
            landmarks[0]["model_config"]["layers"],
            landmarks[0]["model_config"]["heads"],
            landmarks[0]["schedule_total_steps"],
        ], dtype=np.int64),
        "optimizer_steps": np.asarray([
            checkpoint["step"] for checkpoint in landmarks
        ], dtype=np.int64),
    }
    _write_npz(arguments.output, values)


def _local_functions(layer, module, head):
    q = layer["q"][0, head].double()
    k = layer["k"][0, head].double()
    v = layer["v"][0, head].double()
    mask = layer["mask"][0, 0].bool()
    width = q.shape[-1]
    weight = module.Wlst[head].detach().double()
    bias = module.blst[head].detach().double()
    power = module.pwlst[head].detach().double()
    coupling = module.alst[head].detach().double()
    coupling_bias = module.balst[head].detach().double()
    allowed = ~mask

    def probabilities(flat_a):
        a_value = flat_a.reshape(width, width)
        preactivation = weight @ a_value + bias
        a_lm = preactivation * F.silu(preactivation) + 1e-9
        a_p = torch.pow(a_lm, power)
        g_lm = coupling @ a_p + coupling_bias
        scores = q @ g_lm @ k.T / math.sqrt(width)
        return torch.softmax(scores.masked_fill(mask, -1e9), dim=-1)

    with torch.no_grad():
        frozen_probability = probabilities(layer["A"][0, head].double().reshape(-1))
        selected = frozen_probability.argmax(dim=-1)

    def observation(flat_a):
        probability = probabilities(flat_a)
        attention = probability @ v
        return torch.cat((
            2 * torch.sqrt(probability[allowed].clamp_min(1e-300)),
            attention.reshape(-1),
        ))

    def local_loss(flat_a):
        probability = probabilities(flat_a)
        rows = torch.arange(probability.shape[0], device=probability.device)
        return -torch.log(
            probability[rows, selected].clamp_min(1e-300)).sum()

    return observation, local_loss, probabilities


def _factor_from_gram(gram):
    gram = (gram + gram.T) / 2
    values, vectors = torch.linalg.eigh(gram)
    values = values.clamp_min(0)
    return torch.diag(torch.sqrt(values)) @ vectors.T


def _anchor_factors(model, batches, layer_index, hessian_rows):
    factors = []
    full_hessians = []
    hessian_fisher_grams = []
    fisher_grams = []
    probability_floors = []
    projection_errors = []
    coordinate_rows = []
    module = model.decoder.dec_layers[layer_index].mha1.plgatt_layer
    for row_index, (_identifier, tokens) in enumerate(batches):
        layers, _ = _capture(model, tokens)
        layer = layers[layer_index]
        coordinate_rows.append(
            layer["A"].double().mean(dim=1).reshape(-1).cpu())
        gram = None
        local_full = None
        for head in range(layer["A"].shape[1]):
            observation, local_loss, probabilities = _local_functions(
                layer, module, head)
            flat_a = layer["A"][0, head].double().reshape(-1)
            jacobian = jacrev(observation)(flat_a)
            head_gram = jacobian.T @ jacobian
            gram = head_gram if gram is None else gram + head_gram
            with torch.no_grad():
                probability = probabilities(flat_a)
                probability_floors.append(float(
                    probability[~layer["mask"][0, 0].bool()].min()))
                errors = []
                for query in range(probability.shape[0]):
                    allowed = ~layer["mask"][0, 0, query].bool()
                    p = probability[query, allowed]
                    softmax_hessian = torch.diag(p) - torch.outer(p, p)
                    errors.append(float(softmax_hessian.sum(dim=0).abs().max()))
                projection_errors.append(max(errors))
            if row_index < hessian_rows:
                head_full = hessian(local_loss)(flat_a)
                local_full = (
                    head_full if local_full is None
                    else local_full + head_full
                )
        factors.append(_factor_from_gram(gram).cpu().numpy())
        fisher_grams.append(gram.cpu().numpy())
        if local_full is not None:
            hessian_fisher_grams.append(gram.cpu().numpy())
            full_hessians.append(local_full.detach().cpu().numpy())
    coordinates = torch.stack(coordinate_rows)
    centered = coordinates - coordinates.mean(dim=0, keepdim=True)
    normal_coordinate = float(torch.linalg.vector_norm(centered))
    fisher = np.mean(fisher_grams, axis=0)
    full = (
        np.mean(full_hessians, axis=0)
        if full_hessians else fisher.copy()
    )
    hessian_fisher = (
        np.mean(hessian_fisher_grams, axis=0)
        if hessian_fisher_grams else fisher.copy()
    )
    return {
        "factors": np.asarray(factors),
        "fisher": fisher,
        "full": full,
        "hessian_fisher": hessian_fisher,
        "probability_floor": min(probability_floors),
        "projection_error": max(projection_errors),
        "normal_coordinate": normal_coordinate,
        "coordinates": coordinates.numpy(),
    }


def _lifted_operator(alpha, beta, decay, eigenvalue):
    return np.asarray([
        [
            1 - decay - alpha * (1 - beta) * eigenvalue,
            -alpha * beta,
        ],
        [(1 - beta) * eigenvalue, beta],
    ], dtype=np.float64)


def _lyapunov_metric(operator):
    size = operator.shape[0]
    system = np.eye(size * size) - np.kron(operator.T, operator.T)
    try:
        value = np.linalg.solve(
            system, np.eye(size).reshape(-1, order="F"),
        ).reshape((size, size), order="F")
        value = (value + value.T) / 2
        if np.linalg.eigvalsh(value)[0] <= 0:
            raise np.linalg.LinAlgError
        return value
    except np.linalg.LinAlgError:
        return np.eye(size)


def _mask_fraction(mapping):
    total = sum(value.numel() for value in mapping.values())
    active = sum(int(value.sum()) for value in mapping.values())
    return active / max(total, 1)


def normal_observations(arguments):
    sources = load_checkpoint_set(
        arguments.source_manifest, arguments.artifact_root)
    targets = load_checkpoint_set(
        arguments.target_manifest, arguments.artifact_root)
    if not sources or len(sources) != len(targets):
        raise ValueError("N needs paired nonempty source and target checkpoints")
    device = torch.device(arguments.device)
    registries = {digest_object(value["measurement_registry"]) for value in sources}
    if len(registries) != 1:
        raise ValueError("N source checkpoints do not share one registry")
    construction_factors = []
    validation_factors = []
    validation_full_hessians = []
    validation_hessian_fishers = []
    construction_states = []
    validation_states = []
    direct_initial_construction = []
    direct_observed_construction = []
    direct_initial_validation = []
    direct_observed_validation = []
    probability_floors = []
    projection_errors = []

    for source, target in zip(sources, targets):
        if source["measurement_registry"] != target["measurement_registry"]:
            raise ValueError("N successor changed the measurement registry")
        model = _model_from_checkpoint(source, device)
        construction_batches = _registry_batches(
            source, arguments.tokens, "construction",
            arguments.rows_per_split)
        validation_batches = _registry_batches(
            source, arguments.tokens, "validation",
            arguments.rows_per_split)
        construction = _anchor_factors(
            model, construction_batches, arguments.layer,
            arguments.hessian_rows)
        validation = _anchor_factors(
            model, validation_batches, arguments.layer,
            arguments.hessian_rows)
        target_model = _model_from_checkpoint(target, device)
        target_construction = _anchor_factors(
            target_model, construction_batches, arguments.layer, 0)
        target_validation = _anchor_factors(
            target_model, validation_batches, arguments.layer, 0)
        construction_factors.append(construction["factors"])
        validation_factors.append(validation["factors"])
        validation_full_hessians.append(validation["full"])
        validation_hessian_fishers.append(validation["hessian_fisher"])
        probability_floors.append([
            construction["probability_floor"],
            validation["probability_floor"],
        ])
        projection_errors.append([
            construction["projection_error"],
            validation["projection_error"],
        ])

        group = source["optimizer_config"]["groups"][0]
        beta = float(group["betas"][0])
        learning_rate = float(group["learning_rate"])
        alpha = learning_rate / (1 - beta ** max(int(source["step"]), 1))

        n0c = construction["normal_coordinate"]
        n1c = target_construction["normal_coordinate"]
        n0v = validation["normal_coordinate"]
        n1v = target_validation["normal_coordinate"]
        yc0 = np.asarray([n0c, 0.0])
        yc1 = np.asarray([n1c, (n1c - n0c) / max(alpha, 1e-300)])
        yv0 = np.asarray([n0v, 0.0])
        yv1 = np.asarray([n1v, (n1v - n0v) / max(alpha, 1e-300)])
        construction_states.append(np.stack((yc0, yc1)))
        validation_states.append(np.stack((yv0, yv1)))

        construction_rows = construction["coordinates"]
        target_construction_rows = target_construction["coordinates"]
        construction_initial = np.linalg.norm(
            construction_rows
            - construction_rows.mean(axis=0, keepdims=True), axis=1)
        construction_observed = np.linalg.norm(
            target_construction_rows - construction_rows, axis=1)
        source_rows = validation["coordinates"]
        target_rows = target_validation["coordinates"]
        initial = np.linalg.norm(
            source_rows - source_rows.mean(axis=0, keepdims=True), axis=1)
        observed = np.linalg.norm(target_rows - source_rows, axis=1)
        direct_initial_construction.append(construction_initial)
        direct_observed_construction.append(construction_observed)
        direct_initial_validation.append(initial)
        direct_observed_validation.append(observed)

    values = {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "checkpoint_complete_digests": np.asarray([
            [
                source["state_manifest"]["complete_state_sha256"],
                target["state_manifest"]["complete_state_sha256"],
            ]
            for source, target in zip(sources, targets)
        ]),
        "measurement_registry_sha256": np.asarray(next(iter(registries))),
        "construction_observation_factors": np.asarray(construction_factors),
        "validation_observation_factors": np.asarray(validation_factors),
        "validation_full_hessian_matrices": np.asarray(
            validation_full_hessians),
        "validation_hessian_fisher_matrices": np.asarray(
            validation_hessian_fishers),
        "quotient_probability_floors": np.asarray(probability_floors),
        "quotient_projection_errors": np.asarray(projection_errors),
        "normal_states_construction": np.asarray(construction_states),
        "normal_states_validation": np.asarray(validation_states),
        "direct_initial_defects_construction": np.asarray(
            direct_initial_construction),
        "direct_observed_defects_construction": np.asarray(
            direct_observed_construction),
        "direct_initial_defects_validation": np.asarray(
            direct_initial_validation),
        "direct_observed_defects_validation": np.asarray(
            direct_observed_validation),
    }
    _write_npz(arguments.output, values)


def _non_rate_digest(checkpoint):
    groups = []
    for group in checkpoint["optimizer_config"]["groups"]:
        groups.append({
            key: value for key, value in group.items()
            if key not in {"learning_rate", "initial_learning_rate"}
        })
    return digest_object({
        "model_config": checkpoint["model_config"],
        "optimizer": {
            **checkpoint["optimizer_config"],
            "groups": groups,
        },
        "measurement_registry": checkpoint["measurement_registry"],
        "data_digests": {
            "dataset": checkpoint["data_state"]["dataset_sha256"],
            "tokenizer": checkpoint["data_state"]["tokenizer_sha256"],
        },
        "operation_order": checkpoint["operation_order"],
        "intervention_state": checkpoint["intervention_state"],
    })


def _coordinate_for_checkpoint(checkpoint, token_path, device, limit):
    model = _model_from_checkpoint(checkpoint, device)
    batches = _registry_batches(
        checkpoint, token_path, "validation", limit)
    rows = []
    for _identifier, tokens in batches:
        layers, _ = _capture(model, tokens)
        rows.append(layers[0]["A"].double().reshape(-1).cpu())
    matrix = torch.stack(rows)
    return float(torch.linalg.vector_norm(
        matrix - matrix.mean(dim=0, keepdim=True)))


def intervention_observations(arguments):
    sources = load_checkpoint_set(
        arguments.source_manifest, arguments.artifact_root)
    low = load_checkpoint_set(
        arguments.low_manifest, arguments.artifact_root)
    high = load_checkpoint_set(
        arguments.high_manifest, arguments.artifact_root)
    if len(sources) != 1 or not low or len(low) != len(high):
        raise ValueError("I requires one source and equal low/high branches")
    source = sources[0]
    for checkpoint in (*low, *high):
        if checkpoint["measurement_registry"] != source["measurement_registry"]:
            raise ValueError("an intervention branch changed the registry")
    with np.load(arguments.normal_observations, allow_pickle=False) as archive:
        if archive["producer_code_sha256"].item() != _producer_digest():
            raise ValueError("I normal observations have foreign provenance")
        eigenvalues = np.asarray(
            archive["fisher_ritz_intervals"], dtype=float)
        registry_digest = archive["measurement_registry_sha256"].item()
    if registry_digest != digest_object(source["measurement_registry"]):
        raise ValueError("I normal observations use another registry")
    device = torch.device(arguments.device)
    source_amplitude = _coordinate_for_checkpoint(
        source, arguments.tokens, device, arguments.rows_per_split)
    branch_amplitudes = []
    for branch in (low, high):
        branch_amplitudes.append([
            source_amplitude,
            *[
                _coordinate_for_checkpoint(
                    checkpoint, arguments.tokens, device,
                    arguments.rows_per_split)
                for checkpoint in branch
            ],
        ])
    alpha = []
    beta = []
    decay = []
    for branch in (low, high):
        branch_alpha = []
        branch_beta = []
        branch_decay = []
        for checkpoint in branch:
            group = checkpoint["optimizer_config"]["groups"][0]
            beta_value = float(group["betas"][0])
            rate = float(group["learning_rate"])
            branch_alpha.append(
                rate / (1 - beta_value ** max(int(checkpoint["step"]), 1)))
            branch_beta.append(beta_value)
            branch_decay.append(rate * float(group["weight_decay"]))
        alpha.append(branch_alpha)
        beta.append(branch_beta)
        decay.append(branch_decay)
    values = {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "source_complete_digests": np.asarray([[
            source["state_manifest"]["complete_state_sha256"],
            source["state_manifest"]["complete_state_sha256"],
        ]]),
        "branch_complete_digests": np.asarray([[
            [checkpoint["state_manifest"]["complete_state_sha256"]
             for checkpoint in low],
            [checkpoint["state_manifest"]["complete_state_sha256"]
             for checkpoint in high],
        ]]),
        "branch_non_rate_digests": np.asarray([[
            _non_rate_digest(low[0]), _non_rate_digest(high[0]),
        ]]),
        "alpha_effective": np.asarray([alpha]),
        "beta1": np.asarray([beta]),
        "realized_decay": np.asarray([decay]),
        "normal_eigenvalues": np.full(
            (1, 2, len(low)),
            max(float(eigenvalues[0, 0]), np.finfo(float).tiny),
        ),
        "normal_amplitudes": np.asarray([[*branch_amplitudes]]),
    }
    _write_npz(arguments.output, values)


STAGE_NAMES = (
    "A_LM", "A_P", "G_LM", "scores", "softmax",
    "attention_output", "residual", "layer_norm", "logits",
)


def _pair_distances(model, left, right):
    left_layers, left_logits = _capture(model, left)
    right_layers, right_logits = _capture(model, right)
    row = []
    observed = []
    floors = []
    for left_layer, right_layer in zip(left_layers, right_layers):
        row.append(float(torch.linalg.vector_norm(
            left_layer["A"] - right_layer["A"])))
        distances = []
        for name in STAGE_NAMES[:-1]:
            distances.append(float(torch.linalg.vector_norm(
                left_layer[name] - right_layer[name])))
        distances.append(float(torch.linalg.vector_norm(
            left_logits - right_logits)))
        observed.append(distances)
        floors.append(min(
            left_layer["layernorm_centered_floor"],
            right_layer["layernorm_centered_floor"],
        ))
    return (
        np.asarray(row),
        np.asarray(observed),
        np.asarray(floors),
        left_logits[:, -1].double().cpu().numpy()[0],
        right_logits[:, -1].double().cpu().numpy()[0],
    )


def application_observations(arguments):
    checkpoints = load_checkpoint_set(
        arguments.checkpoint_manifest, arguments.artifact_root)
    if len(checkpoints) != 2:
        raise ValueError("A requires exactly the early and late checkpoints")
    device = torch.device(arguments.device)
    all_row = []
    all_observed = []
    all_floors = []
    all_construction_row = []
    all_construction_observed = []
    references = []
    comparisons = []
    application_ids = None
    registry_digests = set()

    for checkpoint in checkpoints:
        registry_digests.add(digest_object(checkpoint["measurement_registry"]))
        model = _model_from_checkpoint(checkpoint, device)
        construction = _registry_batches(
            checkpoint, arguments.tokens, "construction",
            2 * arguments.construction_pairs)
        construction_records = []
        for index in range(0, len(construction) - 1, 2):
            construction_records.append(_pair_distances(
                model, construction[index][1], construction[index + 1][1]))
        if not construction_records:
            raise ValueError("A construction registry has no source pairs")
        all_construction_row.append(np.asarray([
            record[0] for record in construction_records
        ]))
        all_construction_observed.append(np.asarray([
            record[1] for record in construction_records
        ]))
        pairs = _application_batches(
            checkpoint, arguments.tokens, arguments.application_pairs)
        identifiers = [row[0] for row in pairs]
        if application_ids is None:
            application_ids = identifiers
        elif application_ids != identifiers:
            raise ValueError("A checkpoints changed application identifiers")
        rows = []
        observations = []
        floors = []
        reference_logits = []
        comparison_logits = []
        for _identifier, left, right in pairs:
            row, observed, floor, reference, comparison = _pair_distances(
                model, left, right)
            rows.append(row)
            observations.append(observed)
            floors.append(floor)
            reference_logits.append(reference)
            comparison_logits.append(comparison)
        all_row.append(rows)
        all_observed.append(observations)
        all_floors.append(floors)
        references.append(reference_logits)
        comparisons.append(comparison_logits)
    if len(registry_digests) != 1:
        raise ValueError("A checkpoints do not share one registry")
    values = {
        "producer_code_sha256": np.asarray(_producer_digest()),
        "checkpoint_complete_digests": np.asarray([
            checkpoint["state_manifest"]["complete_state_sha256"]
            for checkpoint in checkpoints
        ]),
        "measurement_registry_sha256": np.asarray(
            next(iter(registry_digests))),
        "application_pair_ids": np.asarray(application_ids),
        "stage_names": np.asarray(STAGE_NAMES),
        "construction_row_defects": np.asarray(all_construction_row),
        "construction_stage_differences": np.asarray(
            all_construction_observed),
        "row_defects": np.asarray(all_row),
        "observed_stage_differences": np.asarray(all_observed),
        "layernorm_centered_norms": np.asarray(all_floors),
        "reference_logits": np.asarray(references),
        "comparison_logits": np.asarray(comparisons),
    }
    _write_npz(arguments.output, values)


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    registry = subparsers.add_parser("measurement-registry")
    registry.add_argument("--dataset", required=True)
    registry.add_argument("--tokenizer", required=True)
    registry.add_argument("--context-length", type=int, default=256)
    registry.add_argument("--construction-count", type=int, default=32)
    registry.add_argument("--validation-count", type=int, default=32)
    registry.add_argument("--application-count", type=int, default=32)
    registry.add_argument("--reserve-chunks", type=int, default=5120)
    registry.add_argument("--reserved-start", type=int, default=-1)
    registry.add_argument("--output", required=True)
    registry.add_argument("--partition-output")
    registry.set_defaults(function=create_measurement_registry)

    training = subparsers.add_parser("training")
    training.add_argument("--artifact-root", required=True)
    training.add_argument("--landmark-manifest", required=True)
    training.add_argument("--replay-manifest", required=True)
    training.add_argument("--output", required=True)
    training.set_defaults(function=training_observations)

    normal = subparsers.add_parser("normal")
    normal.add_argument("--artifact-root", required=True)
    normal.add_argument("--source-manifest", required=True)
    normal.add_argument("--target-manifest", required=True)
    normal.add_argument("--tokens", required=True)
    normal.add_argument("--device", default="cuda:0")
    normal.add_argument("--layer", type=int, default=0)
    normal.add_argument("--rows-per-split", type=int, default=32)
    normal.add_argument("--hessian-rows", type=int, default=2)
    normal.add_argument("--output", required=True)
    normal.set_defaults(function=normal_observations)

    intervention = subparsers.add_parser("intervention")
    intervention.add_argument("--artifact-root", required=True)
    intervention.add_argument("--source-manifest", required=True)
    intervention.add_argument("--low-manifest", required=True)
    intervention.add_argument("--high-manifest", required=True)
    intervention.add_argument("--normal-observations", required=True)
    intervention.add_argument("--tokens", required=True)
    intervention.add_argument("--device", default="cuda:0")
    intervention.add_argument("--rows-per-split", type=int, default=32)
    intervention.add_argument("--output", required=True)
    intervention.set_defaults(function=intervention_observations)

    application = subparsers.add_parser("application")
    application.add_argument("--artifact-root", required=True)
    application.add_argument("--checkpoint-manifest", required=True)
    application.add_argument("--tokens", required=True)
    application.add_argument("--device", default="cuda:0")
    application.add_argument("--construction-pairs", type=int, default=16)
    application.add_argument("--application-pairs", type=int, default=32)
    application.add_argument("--output", required=True)
    application.set_defaults(function=application_observations)

    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
