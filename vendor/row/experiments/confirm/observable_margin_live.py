#!/usr/bin/env python3
"""Native-update producer for compact fixed-probe observable ledgers."""

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
from torch.func import grad, jvp, vjp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

from confirmation_artifacts import validate_complete_checkpoint  # noqa: E402
from gate_shape import plga_secant_chain  # noqa: E402
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    sha256_path,
    write_json_atomic,
    write_npz_atomic,
)
from observable_margin import (  # noqa: E402
    bridge_sector_replay,
    certified_bridge_margin,
    deterministic_knn_mst_graph,
    effective_resistance_constant,
    executed_source_margin,
    graph_energy,
    native_roundoff_charge,
    occupied_power_coefficient,
    pack_union_graph,
    shape_metric_secant,
    three_factor_score_secant,
    unpack_union_graph,
)
from observable_margin_specs import (  # noqa: E402
    ARCHITECTURE,
    ARITHMETIC,
    CAMPAIGN_ID,
    REGISTRY,
)
from row_map_live import (  # noqa: E402
    _causal_mask,
    _gate_loss_components,
    _gate_parameter_name,
    _load_raw_checkpoint,
    _loss,
    _model_from_raw,
)


LEDGER_SCHEMA = "pldr-observable-margin-ledger-v1"
LOCK_SCHEMA = "pldr-observable-margin-construction-lock-v1"


def _load_deciding_checkpoint(
    path, *, device="cpu", allow_restart_successor=False,
):
    checkpoint = _load_raw_checkpoint(
        path, require_optimizer=True, device=device,
        allow_restart_successor=allow_restart_successor)
    canonical = {
        name: value for name, value in checkpoint.items()
        if name not in {"content_sha256", "source_path"}}
    validate_complete_checkpoint(canonical)
    if checkpoint["state_manifest"]["complete_state_sha256"] != (
        canonical["state_manifest"]["complete_state_sha256"]
    ):
        raise ValueError("checkpoint complete-state digest changed during loading")
    return checkpoint


def _load_registry(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    from confirmation_artifacts import validate_measurement_registry
    validate_measurement_registry(value, require_nonempty=True)
    required = {
        "schema_version", "context_length", "construction", "validation",
        "anchors", "context_head_block_rule", "block_row_rule",
        "graph_transfer", "plga_binding_rule", "intervention_units",
        "intervention_unit_rule", "dataset_sha256", "tokenizer_sha256",
        "registry_sha256",
    }
    if set(value) != required or value["schema_version"] != (
        "pldr-observable-probe-registry-v1"
    ):
        raise ValueError("observable probe registry is malformed")
    if (
        len(value["construction"]) != REGISTRY["construction_contexts"]
        or len(value["validation"]) != REGISTRY["heldout_contexts"]
        or value["context_head_block_rule"] != REGISTRY["context_head_block_rule"]
        or value["block_row_rule"] != REGISTRY["block_row_rule"]
        or value["graph_transfer"] != REGISTRY["graph_transfer"]
    ):
        raise ValueError("observable probe geometry changed")
    return value


def _load_lock(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unknown observable construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("observable construction lock digest does not replay")
    return value


def _token_matrix(tokens, context):
    data = np.memmap(tokens, dtype=np.uint16, mode="r")
    count = len(data) // context
    return data[:count * context].reshape(count, context)


def _rows(matrix, indices, device):
    return [
        torch.from_numpy(
            matrix[index:index + 1].astype(np.int64, copy=True)).to(device)
        for index in indices
    ]


def _capture_context(model, tokens, layer_index, head_index):
    blocks = model.decoder.dec_layers[layer_index].mha1.reslayerAs
    physical = []
    shape = []

    def physical_hook(_module, inputs):
        value = inputs[0]
        if isinstance(value, (list, tuple)):
            if len(value) != 1 or not torch.is_tensor(value[0]):
                raise TypeError("row-map block input wrapper is malformed")
            value = value[0]
        if not torch.is_tensor(value):
            raise TypeError("row-map block input is not a tensor")
        physical.append(value.detach())

    def shape_hook(module, inputs, _output):
        value = inputs[0]
        centered = value - value.mean(dim=-1, keepdim=True)
        shape.append((centered / torch.sqrt(
            module.eps + torch.mean(
                centered * centered, dim=-1, keepdim=True))).detach())

    first = blocks[0].register_forward_pre_hook(physical_hook)
    final = blocks[-1].layernormA.register_forward_hook(shape_hook)
    try:
        inputs = tokens[:, :-1]
        mask = _causal_mask(
            inputs.shape[1], inputs.device, next(model.parameters()).dtype)
        with torch.no_grad():
            model([inputs, mask])
    finally:
        first.remove()
        final.remove()
    if len(physical) != 1 or len(shape) != 1:
        raise RuntimeError("row-map probe hooks did not fire exactly once")
    if (
        physical[0].ndim != 4 or shape[0].shape != physical[0].shape
        or physical[0].shape[0] != 1
        or not 0 <= head_index < physical[0].shape[1]
    ):
        raise ValueError("registered context-head block has an unexpected shape")
    return physical[0][0, head_index], shape[0][0, head_index]


def _probe_node(model, contexts, layer_index, head_indices, graph=None):
    """Evaluate one connected ordered union of complete context-head blocks."""

    if len(contexts) != len(head_indices):
        raise ValueError("probe contexts and registered heads disagree")
    was_training = model.training
    model.eval()
    physical_blocks = []
    shape_blocks = []
    for tokens, head_index in zip(contexts, head_indices, strict=True):
        point, shape = _capture_context(
            model, tokens, layer_index, int(head_index))
        physical_blocks.append(point.double().cpu().numpy())
        shape_blocks.append(shape.double().cpu().numpy())
    if was_training:
        model.train()
    row_counts = [len(value) for value in physical_blocks]
    if any(count != ARCHITECTURE["row_width"] for count in row_counts):
        raise ValueError("a union block does not contain the complete d rows")
    block_offsets = np.concatenate((
        np.asarray([0], dtype=np.int64), np.cumsum(row_counts, dtype=np.int64)))
    physical_union = np.concatenate(physical_blocks, axis=0)
    shape_union = np.concatenate(shape_blocks, axis=0)
    if graph is None:
        graph = deterministic_knn_mst_graph(
            physical_union, neighbors=REGISTRY["neighbors"])
    gate = dict(model.named_parameters())[_gate_parameter_name(layer_index)]
    gamma = gate.detach().double().cpu().numpy()
    geometry = graph_energy(
        gamma, shape_union, graph["edges"], graph["weights"])
    return {
        "gamma": gamma,
        "physical": [physical_union],
        "shapes": [shape_union],
        "graphs": [graph],
        "block_vertex_offsets": block_offsets,
        "block_head_index": np.asarray(head_indices, dtype=np.int64),
        "q": np.asarray([geometry["q"]]),
        "energy": np.asarray([geometry["energy"]]),
        "diameter": np.asarray([geometry["vertex_diameter"]]),
    }


def _shape_increment_metric(source, target, graph):
    edges = graph["edges"]
    weights = graph["weights"]
    source_edge = source[edges[:, 0]] - source[edges[:, 1]]
    increment = target - source
    increment_edge = increment[edges[:, 0]] - increment[edges[:, 1]]
    return (
        2.0 * np.sum(weights[:, None] * source_edge * increment_edge, axis=0),
        np.sum(weights[:, None] * increment_edge * increment_edge, axis=0),
    )


def _source_actions(model, rows, layer_index, raw_gradient):
    """Stream exact zero-gate bridge actions plus compact minibatch dispersion."""

    gate_name = _gate_parameter_name(layer_index)
    gate = dict(model.named_parameters())[gate_name]
    terms = []
    for start in range(0, len(rows), 8):
        microbatch = rows[start:start + 8]
        logits_for_gate, loss_for_gate, source_count = _gate_loss_components(
            model, microbatch, gate_name)
        base = gate.detach()
        zero = torch.zeros_like(base)
        true_action = jvp(
            grad(loss_for_gate), (zero,), (base,), strict=True)[1].detach()
        logits, pullback = vjp(logits_for_gate, zero)
        _unused, logit_action = jvp(
            logits_for_gate, (zero,), (base,), strict=True)
        probability = torch.softmax(logits.detach(), dim=-1)
        targets = microbatch[:, 1:].to(base.device)
        keep = (targets != 0).to(base.dtype)
        mean = torch.sum(probability * logit_action, dim=-1, keepdim=True)
        cotangent = probability * (logit_action - mean)
        cotangent = cotangent * keep[..., None] / source_count
        fisher_action = pullback(cotangent)[0].detach()
        force = grad(loss_for_gate)(zero).detach()
        raw_local = grad(loss_for_gate)(base).detach()
        terms.append((
            float(source_count), fisher_action, true_action, force, raw_local))
    total = sum(row[0] for row in terms)
    weights = torch.as_tensor(
        [row[0] / total for row in terms], dtype=torch.float64,
        device=raw_gradient.device)

    def weighted(index):
        result = torch.zeros_like(raw_gradient, dtype=torch.float64)
        for weight, row in zip(weights, terms, strict=True):
            result = result + weight * row[index].double()
        return result

    fisher_action = weighted(1)
    true_action = weighted(2)
    force = weighted(3)
    signed_action = true_action - fisher_action
    nonlinear = raw_gradient.detach().double() - force - true_action
    components = []
    for row in terms:
        local_fisher = row[1].double()
        local_true = row[2].double()
        local_force = row[3].double()
        local_raw = row[4].double()
        components.append(torch.stack((
            local_fisher,
            local_true - local_fisher,
            local_force,
            local_raw - local_force - local_true,
        )))
    component_array = torch.stack(components)
    aggregate = (fisher_action, signed_action, force, nonlinear)
    aggregate_from_components = torch.sum(
        weights[:, None, None] * component_array, dim=0)
    aggregate_stack = torch.stack(aggregate)
    aggregation_residual = torch.linalg.vector_norm(
        aggregate_from_components - aggregate_stack)
    aggregation_scale = (
        torch.sum(
            weights[:, None]
            * torch.linalg.vector_norm(component_array, dim=2), dim=0)
        + torch.linalg.vector_norm(aggregate_stack, dim=1)
    ).sum()
    source_epsilon = torch.finfo(raw_gradient.dtype).eps
    aggregation_enclosure = (
        64.0 * source_epsilon * torch.clamp(aggregation_scale, min=1.0)
    )
    if not torch.isfinite(aggregation_residual) or (
        aggregation_residual > aggregation_enclosure
    ):
        raise ArithmeticError("microbatch bridge sources do not aggregate")
    return {
        "aggregate": aggregate,
        "weights": weights,
        "source_counts": torch.as_tensor(
            [row[0] for row in terms], dtype=torch.float64,
            device=raw_gradient.device),
        "components": component_array,
        "aggregation_residual": aggregation_residual,
        "aggregation_enclosure": aggregation_enclosure,
    }


def _bridge_directions(
    gate, raw_gradient, clipped_gradient, state_before, optimizer_step_after,
    group, source_actions,
):
    """Reconstruct the exact float64 shadow moment and seven-source bridge.

    Native input moments and the realized native clipped gradient are promoted
    before arithmetic. The native successor is never substituted into this
    bridge; its complete moment, square-root, and gate-arithmetic discrepancy
    is charged later by ``native_roundoff_charge``.
    """

    beta1, beta2 = map(float, group["betas"])
    epsilon = float(group["eps"])
    step = int(optimizer_step_after)
    if step < 1:
        raise ValueError("optimizer shadow step must be positive")
    first_before = state_before["exp_avg"].detach().double()
    second_before = state_before["exp_avg_sq"].detach().double()
    raw = raw_gradient.detach().double()
    clipped = clipped_gradient.detach().double()
    first_shadow = beta1 * first_before + (1.0 - beta1) * clipped
    second_shadow = beta2 * second_before + (1.0 - beta2) * clipped * clipped
    first_hat = first_shadow / (1.0 - beta1 ** step)
    second_hat = second_shadow / (1.0 - beta2 ** step)
    preconditioner = 1.0 / (torch.sqrt(second_hat) + epsilon)
    alpha = (1.0 - beta1) / (1.0 - beta1 ** step)
    lag_numerator = beta1 * first_before / (1.0 - beta1 ** step)
    reference = 1.0
    fisher, signed, force, nonlinear = [
        value.double() for value in source_actions["aggregate"]]
    directions = torch.stack([
        alpha * reference * fisher,
        alpha * reference * signed,
        alpha * reference * force,
        alpha * reference * nonlinear,
        alpha * reference * (clipped - raw),
        alpha * (preconditioner - reference) * clipped,
        preconditioner * lag_numerator,
    ])
    adaptive = first_hat * preconditioner
    component = source_actions["components"].double()
    microbatch_count = component.shape[0]
    microbatch_directions = torch.empty(
        (microbatch_count, 7, gate.numel()), dtype=torch.float64,
        device=gate.device)
    microbatch_directions[:, 0] = alpha * reference * component[:, 0]
    microbatch_directions[:, 1] = alpha * reference * component[:, 1]
    microbatch_directions[:, 2] = alpha * reference * component[:, 2]
    microbatch_directions[:, 3] = alpha * reference * component[:, 3]
    # Clip, preconditioner, and lag are post-aggregation executed sources.
    # Repeating their exact direction across source microbatches gives a
    # declared zero between-microbatch dispersion for those sectors.
    microbatch_directions[:, 4] = directions[4]
    microbatch_directions[:, 5] = directions[5]
    microbatch_directions[:, 6] = directions[6]
    weighted_microbatch = torch.sum(
        source_actions["weights"][:, None, None] * microbatch_directions,
        dim=0)
    direction_residual = torch.linalg.vector_norm(
        weighted_microbatch - directions)
    direction_scale = (
        torch.sum(
            source_actions["weights"][:, None]
            * torch.linalg.vector_norm(microbatch_directions, dim=2), dim=0)
        + torch.linalg.vector_norm(directions, dim=1)
    ).sum()
    source_epsilon = torch.finfo(raw_gradient.dtype).eps
    direction_enclosure = (
        64.0 * source_epsilon * torch.clamp(direction_scale, min=1.0)
    )
    if not torch.isfinite(direction_residual) or (
        direction_residual > direction_enclosure
    ):
        raise ArithmeticError("microbatch sector directions do not aggregate")
    return {
        "adaptive_direction": adaptive,
        "sector_directions": directions,
        "first_moment_before": first_before,
        "second_moment_before": second_before,
        "clipped_gradient": clipped,
        "first_moment_shadow_after": first_shadow,
        "second_moment_shadow_after": second_shadow,
        "shadow_preconditioner": preconditioner,
        "optimizer_step_after": step,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "source_microbatch_weight": source_actions["weights"],
        "source_microbatch_count": source_actions["source_counts"],
        "sector_microbatch_direction": microbatch_directions,
        "microbatch_direction_residual": direction_residual,
        "microbatch_direction_enclosure": direction_enclosure,
    }


def _plga_arrays(
    model, contexts, context_ids, head_indices, layer_index, energy_at_node,
    resistance,
):
    """Capture complete same-context collapsed-anchor PLGA endpoint pairs."""

    if not len(contexts) == len(context_ids) == len(head_indices):
        raise ValueError("PLGA context, head, and tensor identities disagree")

    def run(tokens, gcaches=None):
        inputs = tokens[:, :-1]
        mask = _causal_mask(
            inputs.shape[1], inputs.device, next(model.parameters()).dtype)
        with torch.no_grad():
            return model([inputs, mask], Gcachelst=gcaches)

    was_training = model.training
    model.eval()
    pair_ids = []
    pair_context = []
    pair_head = []
    anchor_rows = []
    generators = []
    collapsed = []
    metric_left = []
    metric_right = []
    weights = []
    biases = []
    exponents = []
    couplings = []
    coupling_biases = []
    coefficients = []
    curvature_left = []
    curvature_right = []
    secant_prediction = []
    secant_residual = []
    operator_bound = []
    generator_difference_norm = []
    stacked_generator_bound = []
    centered_reference = []
    centered_candidate = []
    reference_margin = []
    reference_winner = []
    candidate_winner = []
    centered_difference_norm = []
    downstream_ratio = []
    reference_queries = []
    candidate_queries = []
    reference_keys = []
    candidate_keys = []
    reference_scores = []
    candidate_scores = []
    score_query_terms = []
    score_generator_terms = []
    score_key_terms = []
    score_predictions = []
    score_identity_residuals = []
    query_operator_norms = []
    key_operator_norms = []
    score_difference_norms = []
    candidate_cache_residual = []
    try:
        for context_index, (tokens, context_id) in enumerate(
            zip(contexts, context_ids, strict=True)
        ):
            selected_module = model.decoder.dec_layers[
                layer_index].mha1.plgatt_layer

            def run_with_qk(local_caches=None):
                captured_qk = []

                def capture_qk(_module, inputs):
                    values = inputs[0]
                    captured_qk.append((
                        values[0].detach(), values[1].detach()))

                handle = selected_module.register_forward_pre_hook(capture_qk)
                try:
                    local_result = run(tokens, local_caches)
                finally:
                    handle.remove()
                if len(captured_qk) != 1:
                    raise RuntimeError("PLGA Q/K hook did not fire exactly once")
                local_query = captured_qk[0][0][0].double().cpu().numpy()
                local_key = captured_qk[0][1][0].double().cpu().numpy()
                return local_result, local_query, local_key

            result, original_query_all, original_key_all = run_with_qk()
            logits, attention, kvcache = result[0], result[2], result[3]
            base_caches = [
                [row[0].detach(), row[4].detach()] for row in attention]
            generator_all = kvcache[layer_index][2][0]
            module = model.decoder.dec_layers[
                layer_index].mha1.plgatt_layer
            selected_head = int(head_indices[context_index])
            if not 0 <= selected_head < generator_all.shape[0]:
                raise ValueError("registered PLGA head leaves the model")
            for head in [selected_head]:
                generator = generator_all[head].detach().double().cpu().numpy()
                anchor_row = 0
                collapsed_generator = np.repeat(
                    generator[anchor_row:anchor_row + 1],
                    generator.shape[0], axis=0)
                weight = module.Wlst[head].detach().double().cpu().numpy()
                bias = module.blst[head].detach().double().cpu().numpy()
                powers = module.pwlst[head].detach().double().cpu().numpy()
                coupling = module.alst[head].detach().double().cpu().numpy()
                coupling_bias = module.balst[head].detach().double().cpu().numpy()
                replay = plga_secant_chain(
                    collapsed_generator, generator, weight, bias, powers,
                    coupling, coupling_bias)
                if min(
                    float(np.min(replay["metric_left"])),
                    float(np.min(replay["metric_right"])),
                ) <= 0.0:
                    raise ArithmeticError("PLGA occupied power base is not positive")
                local_coefficients = np.empty_like(replay["metric_left"])
                for index in np.ndindex(local_coefficients.shape):
                    local_coefficients[index] = occupied_power_coefficient(
                        replay["metric_left"][index],
                        replay["metric_right"][index], powers[index]
                    )["coefficient"]
                reference_caches = [list(row) for row in base_caches]
                reference_metric = attention[layer_index][0].detach().clone()
                reference_curvature = attention[layer_index][4].detach().clone()
                reference_metric[0, head] = torch.as_tensor(
                    replay["metric_left"], device=reference_metric.device,
                    dtype=reference_metric.dtype)
                reference_curvature[0, head] = torch.as_tensor(
                    replay["curvature_left"], device=reference_curvature.device,
                    dtype=reference_curvature.dtype)
                reference_caches[layer_index] = [
                    reference_metric, reference_curvature]
                (
                    reference_result, reference_query_all, reference_key_all,
                ) = run_with_qk(reference_caches)
                (
                    candidate_result, candidate_query_all, candidate_key_all,
                ) = run_with_qk(base_caches)
                reference_logits = reference_result[0][0, -1]
                candidate_logits = candidate_result[0][0, -1]
                reference_array = reference_logits.detach().double().cpu().numpy()
                candidate_array = candidate_logits.detach().double().cpu().numpy()
                reference_query = reference_query_all[head]
                candidate_query = candidate_query_all[head]
                reference_key = reference_key_all[head]
                candidate_key = candidate_key_all[head]
                reference_score = reference_result[2][layer_index][5][
                    0, head].detach().double().cpu().numpy()
                candidate_score = candidate_result[2][layer_index][5][
                    0, head].detach().double().cpu().numpy()
                score_chain = three_factor_score_secant(
                    reference_query, candidate_query,
                    replay["curvature_left"], replay["curvature_right"],
                    reference_key, candidate_key)
                score_prediction = score_chain["predicted_difference"]
                score_difference = candidate_score - reference_score
                score_scale = max(
                    float(np.linalg.norm(score_difference)),
                    float(np.linalg.norm(score_prediction)), 1e-300)
                score_identity_residual = float(np.linalg.norm(
                    score_difference - score_prediction)) / score_scale
                original_array = logits[0, -1].detach().double().cpu().numpy()
                reference_center = reference_array - np.mean(reference_array)
                candidate_center = candidate_array - np.mean(candidate_array)
                winner = int(np.argmax(reference_center))
                margin = float(
                    reference_center[winner]
                    - np.max(np.delete(reference_center, winner)))
                difference_norm = float(np.linalg.norm(
                    candidate_center - reference_center))
                score_difference_norm = float(np.linalg.norm(score_difference))
                pair_ids.append(
                    f"{context_id}-l{layer_index}-h{head}-anchor0")
                pair_context.append(context_index)
                pair_head.append(head)
                anchor_rows.append(anchor_row)
                generators.append(generator)
                collapsed.append(collapsed_generator)
                metric_left.append(replay["metric_left"])
                metric_right.append(replay["metric_right"])
                weights.append(weight)
                biases.append(bias)
                exponents.append(powers)
                couplings.append(coupling)
                coupling_biases.append(coupling_bias)
                coefficients.append(local_coefficients)
                curvature_left.append(replay["curvature_left"])
                curvature_right.append(replay["curvature_right"])
                secant_prediction.append(replay["predicted_difference"])
                secant_residual.append(replay["identity_residual"])
                operator_bound.append(replay["operator_bound"])
                generator_difference_norm.append(float(np.linalg.norm(
                    generator - collapsed_generator)))
                stacked_generator_bound.append(math.sqrt(max(
                    generator.shape[0] * float(resistance)
                    * float(energy_at_node), 0.0)))
                centered_reference.append(reference_center)
                centered_candidate.append(candidate_center)
                reference_margin.append(margin)
                reference_winner.append(winner)
                candidate_winner.append(int(np.argmax(candidate_center)))
                centered_difference_norm.append(difference_norm)
                downstream_ratio.append(
                    difference_norm / max(score_difference_norm, 1e-300))
                reference_queries.append(reference_query)
                candidate_queries.append(candidate_query)
                reference_keys.append(reference_key)
                candidate_keys.append(candidate_key)
                reference_scores.append(reference_score)
                candidate_scores.append(candidate_score)
                score_query_terms.append(score_chain["query_term"])
                score_generator_terms.append(score_chain["generator_term"])
                score_key_terms.append(score_chain["key_term"])
                score_predictions.append(score_prediction)
                score_identity_residuals.append(score_identity_residual)
                query_operator_norms.append(max(
                    float(np.linalg.norm(reference_query, ord=2)),
                    float(np.linalg.norm(candidate_query, ord=2))))
                key_operator_norms.append(max(
                    float(np.linalg.norm(reference_key, ord=2)),
                    float(np.linalg.norm(candidate_key, ord=2))))
                score_difference_norms.append(score_difference_norm)
                candidate_cache_residual.append(float(np.linalg.norm(
                    candidate_array - original_array)))
    finally:
        if was_training:
            model.train()
    return {
        "plga_pair_id": np.asarray(pair_ids),
        "plga_context_index": np.asarray(pair_context, dtype=np.int64),
        "plga_head": np.asarray(pair_head, dtype=np.int64),
        "plga_anchor_row": np.asarray(anchor_rows, dtype=np.int64),
        "plga_generator": np.asarray(generators),
        "plga_collapsed_anchor_generator": np.asarray(collapsed),
        "plga_left": np.asarray(metric_left),
        "plga_right": np.asarray(metric_right),
        "plga_weight": np.asarray(weights),
        "plga_bias": np.asarray(biases),
        "plga_exponent": np.asarray(exponents),
        "plga_coupling": np.asarray(couplings),
        "plga_coupling_bias": np.asarray(coupling_biases),
        "plga_coefficient": np.asarray(coefficients),
        "plga_curvature_left": np.asarray(curvature_left),
        "plga_curvature_right": np.asarray(curvature_right),
        "plga_secant_prediction": np.asarray(secant_prediction),
        "plga_secant_identity_residual": np.asarray(secant_residual),
        "plga_secant_operator_bound": np.asarray(operator_bound),
        "plga_generator_difference_norm": np.asarray(
            generator_difference_norm),
        "plga_stacked_generator_bound": np.asarray(stacked_generator_bound),
        "plga_reference_centered_logits": np.asarray(centered_reference),
        "plga_candidate_centered_logits": np.asarray(centered_candidate),
        "plga_reference_margin": np.asarray(reference_margin),
        "plga_reference_winner": np.asarray(reference_winner, dtype=np.int64),
        "plga_candidate_winner": np.asarray(candidate_winner, dtype=np.int64),
        "plga_centered_logit_difference_norm": np.asarray(
            centered_difference_norm),
        "plga_downstream_score_ratio": np.asarray(downstream_ratio),
        "plga_reference_query": np.asarray(reference_queries),
        "plga_candidate_query": np.asarray(candidate_queries),
        "plga_reference_key": np.asarray(reference_keys),
        "plga_candidate_key": np.asarray(candidate_keys),
        "plga_reference_pre_mask_score": np.asarray(reference_scores),
        "plga_candidate_pre_mask_score": np.asarray(candidate_scores),
        "plga_score_query_term": np.asarray(score_query_terms),
        "plga_score_generator_term": np.asarray(score_generator_terms),
        "plga_score_key_term": np.asarray(score_key_terms),
        "plga_score_difference_prediction": np.asarray(score_predictions),
        "plga_score_identity_residual": np.asarray(score_identity_residuals),
        "plga_query_operator_norm": np.asarray(query_operator_norms),
        "plga_key_operator_norm": np.asarray(key_operator_norms),
        "plga_score_difference_norm": np.asarray(score_difference_norms),
        "plga_candidate_cache_replay_residual": np.asarray(
            candidate_cache_residual),
    }


def _locked_union_graph(lock):
    graph_record = lock["graph"]
    required = {
        "rule", "neighbors", "mapping", "union_graph_sha256",
        "vertex_count", "edges", "weights", "effective_resistance",
        "source_physical_sha256",
    }
    if set(graph_record) != required:
        raise ValueError("construction lock omits the exact union graph")
    if graph_record["mapping"] != REGISTRY["graph_transfer"]:
        raise ValueError("construction union-graph identity mapping changed")
    graph = unpack_union_graph(
        graph_record["vertex_count"], graph_record["edges"],
        graph_record["weights"], rule=graph_record["rule"],
        neighbors=graph_record["neighbors"],
        expected_sha256=graph_record["union_graph_sha256"],
    )
    resistance = float(graph_record["effective_resistance"])
    if not math.isfinite(resistance) or resistance < 0.0:
        raise ValueError("locked effective resistance is invalid")
    return graph, resistance


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _frozen_affine(lock, transitions, contexts):
    report = json.loads(Path(lock["source_report"]["path"]).read_text())
    archive_path = Path(report["source"]["path"]).resolve()
    binding = lock["bindings"]["source_ledger"]
    if str(archive_path) != binding["path"] or sha256_path(archive_path) != binding["sha256"]:
        raise ValueError("construction-owned affine source changed after lock")
    load_npz(archive_path, required=("affine_factor", "affine_forcing"))
    frozen = lock["frozen_envelopes"]
    factor = np.asarray(
        frozen["affine_factor_upper_by_step"], dtype=np.float64)
    forcing = np.asarray(
        frozen["affine_forcing_upper_by_step"], dtype=np.float64)
    if factor.shape != (transitions,) or forcing.shape != (transitions,):
        raise ValueError("locked affine path has another block length")
    return (
        np.repeat(factor[:, None], contexts, axis=1),
        np.repeat(forcing[:, None], contexts, axis=1),
    )


def produce(arguments):
    started = time.monotonic()
    stage = arguments.stage
    lock = _load_lock(arguments.lock) if arguments.lock else None
    checkpoint_path = str(arguments.checkpoint)
    if lock is not None:
        checkpoint_path = checkpoint_path.replace(
            "{anchor}", str(lock["selected"]["anchor"]))
    checkpoint = _load_deciding_checkpoint(checkpoint_path, device="cpu")
    registry = _load_registry(arguments.registry)
    if sha256_path(arguments.tokens) != registry["dataset_sha256"]:
        raise ValueError("token archive disagrees with observable registry")
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint does not own the fixed probe registry")
    if int(checkpoint["config"]["seed"]) != int(arguments.seed):
        raise ValueError("checkpoint seed disagrees with the stage job identity")
    identity = checkpoint.get("run_identity")
    if identity is not None and identity["run_id"] != checkpoint["config"]["name"]:
        raise ValueError("checkpoint run identity and configuration disagree")
    if stage in {"H", "M"} and lock is None:
        raise ValueError("locked H/M production requires the construction lock")
    selected = lock["selected"] if lock else None
    layer = selected["layer"] if selected else int(arguments.layer)
    updates = selected["block_length"] if selected else int(arguments.updates)
    anchor = int(checkpoint["step"])
    if selected and anchor != selected["anchor"]:
        raise ValueError("checkpoint anchor disagrees with construction lock")
    device = torch.device(arguments.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = _model_from_raw(checkpoint, device)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(checkpoint["config"]["lr"]),
        betas=(0.9, 0.95), eps=1e-5,
        weight_decay=float(checkpoint["config"]["wd"]))
    optimizer.load_state_dict(checkpoint["opt"])
    gate_parameter_name = _gate_parameter_name(layer)
    gate_parameter = dict(model.named_parameters())[gate_parameter_name]
    gate_group = next(
        value for value in optimizer.param_groups
        if any(parameter is gate_parameter for parameter in value["params"]))
    checkpoint_decay_tensor = checkpoint["branch_state"]["decay_mask"].get(
        gate_parameter_name)
    if (
        not torch.is_tensor(checkpoint_decay_tensor)
        or checkpoint_decay_tensor.dtype != torch.bool
        or checkpoint_decay_tensor.numel() != gate_parameter.numel()
    ):
        raise ValueError("checkpoint gate decay mask is missing or malformed")
    checkpoint_decay_mask = checkpoint_decay_tensor.reshape(-1).double().numpy()
    expected_checkpoint_decay = np.full(
        gate_parameter.numel(),
        float(gate_group.get("weight_decay", 0.0)) != 0.0,
        dtype=np.float64)
    if not np.array_equal(checkpoint_decay_mask, expected_checkpoint_decay):
        raise ValueError("checkpoint gate decay mask disagrees with optimizer group")
    matrix = _token_matrix(arguments.tokens, registry["context_length"])
    role_key = "construction" if arguments.role == "construction" else "validation"
    context_rows = _rows(
        matrix, [row["chunk_index"] for row in registry[role_key]], device)
    block_head_index = np.arange(
        len(context_rows), dtype=np.int64) % ARCHITECTURE["heads"]
    probe_digest = digest_object([
        {"chunk_index": row["chunk_index"], "head": int(head)}
        for row, head in zip(
            registry[role_key], block_head_index, strict=True)])
    if lock is None:
        source = _probe_node(model, context_rows, layer, block_head_index)
        union_graph = source["graphs"][0]
        union_resistance = effective_resistance_constant(
            union_graph["vertex_count"], union_graph["edges"],
            union_graph["weights"])["effective_resistance_max"]
    else:
        union_graph, union_resistance = _locked_union_graph(lock)
        source = _probe_node(
            model, context_rows, layer, block_head_index, union_graph)
    graphs = [union_graph]
    packed_graph = pack_union_graph(union_graph)
    union_graph_digest = packed_graph["union_graph_sha256"]
    resistance = np.asarray([union_resistance], dtype=np.float64)
    nodes = [source]
    linear_rows = []
    quadratic_rows = []
    directions = []
    sector_rows = []
    source_microbatch_weight_rows = []
    source_microbatch_count_rows = []
    sector_microbatch_rows = []
    first_moment_before_rows = []
    second_moment_before_rows = []
    clipped_gradient_rows = []
    first_moment_shadow_rows = []
    second_moment_shadow_rows = []
    shadow_preconditioner_rows = []
    optimizer_step_after_rows = []
    beta1_rows = []
    beta2_rows = []
    epsilon_rows = []
    learning_rates = []
    weight_decays = []
    decay_masks = []
    optimizer_decays = []
    update_digests = []
    cursor = int(checkpoint["data_offset_end"])
    batch = int(checkpoint["config"]["batch"])
    for local_step in range(updates):
        begin = cursor + local_step * batch
        if begin + batch > len(matrix):
            raise ValueError("native continuation leaves the token archive")
        registered_chunks = {
            row["chunk_index"] for row in registry["construction"] + registry["validation"]}
        if any(index in registered_chunks for index in range(begin, begin + batch)):
            raise ValueError("actual training minibatch overlaps fixed probes")
        rows = torch.from_numpy(
            matrix[begin:begin + batch].astype(np.int64, copy=True)).to(device)
        update_digests.append(hashlib.sha256(
            np.ascontiguousarray(rows.cpu().numpy()).tobytes()).hexdigest())
        gate = dict(model.named_parameters())[_gate_parameter_name(layer)]
        group = next(
            value for value in optimizer.param_groups
            if any(parameter is gate for parameter in value["params"]))
        state = optimizer.state[gate]
        state_before = {
            name: value.detach().clone() if torch.is_tensor(value) else value
            for name, value in state.items()
        }
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(model, rows)
        loss.backward()
        raw_gradient = gate.grad.detach().clone()
        actions = _source_actions(model, rows, layer, raw_gradient)
        torch.nn.utils.clip_grad_value_(
            model.parameters(), clip_value=float(checkpoint["config"]["clip"]))
        clipped_gradient = gate.grad.detach().clone()
        gate_before = gate.detach().clone()
        learning_rate = float(group["lr"])
        weight_decay = float(group["weight_decay"])
        optimizer.step()
        state_after = optimizer.state[gate]
        shadow = _bridge_directions(
            gate_before, raw_gradient, clipped_gradient, state_before,
            int(state_after["step"].item()), group, actions)
        directions.append(
            shadow["adaptive_direction"].cpu().numpy())
        sector_rows.append(shadow["sector_directions"].cpu().numpy())
        source_microbatch_weight_rows.append(
            shadow["source_microbatch_weight"].cpu().numpy())
        source_microbatch_count_rows.append(
            shadow["source_microbatch_count"].cpu().numpy())
        sector_microbatch_rows.append(
            shadow["sector_microbatch_direction"].cpu().numpy())
        first_moment_before_rows.append(
            shadow["first_moment_before"].cpu().numpy())
        second_moment_before_rows.append(
            shadow["second_moment_before"].cpu().numpy())
        clipped_gradient_rows.append(shadow["clipped_gradient"].cpu().numpy())
        first_moment_shadow_rows.append(
            shadow["first_moment_shadow_after"].cpu().numpy())
        second_moment_shadow_rows.append(
            shadow["second_moment_shadow_after"].cpu().numpy())
        shadow_preconditioner_rows.append(
            shadow["shadow_preconditioner"].cpu().numpy())
        optimizer_step_after_rows.append(shadow["optimizer_step_after"])
        beta1_rows.append(shadow["beta1"])
        beta2_rows.append(shadow["beta2"])
        epsilon_rows.append(shadow["epsilon"])
        learning_rates.append(learning_rate)
        weight_decays.append(weight_decay)
        mask = np.full(
            gate.numel(), raw_gradient is not None and weight_decay != 0.0,
            dtype=np.float64)
        if not np.array_equal(mask, checkpoint_decay_mask):
            raise ValueError("continuation gate decay mask changed after checkpoint")
        decay_masks.append(mask)
        optimizer_decays.append(weight_decay * mask)
        target = _probe_node(
            model, context_rows, layer, block_head_index, graphs[0])
        local_linear = []
        local_quadratic = []
        for old_shape, new_shape, graph in zip(
            nodes[-1]["shapes"], target["shapes"], graphs, strict=True,
        ):
            linear, quadratic = _shape_increment_metric(
                old_shape, new_shape, graph)
            local_linear.append(linear)
            local_quadratic.append(quadratic)
        linear_rows.append(local_linear)
        quadratic_rows.append(local_quadratic)
        nodes.append(target)
    gamma = np.asarray([node["gamma"] for node in nodes])
    q = np.asarray([node["q"] for node in nodes])
    energy = np.asarray([node["energy"] for node in nodes])
    diameter = np.asarray([node["diameter"] for node in nodes])
    edge_contrast = np.empty(
        (len(nodes), len(packed_graph["edges"]), gamma.shape[1]),
        dtype=np.float64)
    for node_index, node in enumerate(nodes):
        for context_index, graph in enumerate(graphs):
            begin, end = map(int, np.asarray([0, len(packed_graph["edges"])], dtype=np.int64)[
                context_index:context_index + 2])
            edge = graph["edges"]
            edge_contrast[node_index, begin:end] = (
                node["shapes"][context_index][edge[:, 0]]
                - node["shapes"][context_index][edge[:, 1]])
    union_source_physical = np.asarray(source["physical"][0])
    union_source_physical_sha256 = _array_sha256(union_source_physical)
    linear_q = np.asarray(linear_rows)
    quadratic_q = np.asarray(quadratic_rows)
    directions = np.asarray(directions)
    sector_rows = np.asarray(sector_rows)
    source_microbatch_weight_rows = np.asarray(source_microbatch_weight_rows)
    source_microbatch_count_rows = np.asarray(source_microbatch_count_rows)
    sector_microbatch_rows = np.asarray(sector_microbatch_rows)
    first_moment_before_rows = np.asarray(first_moment_before_rows)
    second_moment_before_rows = np.asarray(second_moment_before_rows)
    clipped_gradient_rows = np.asarray(clipped_gradient_rows)
    first_moment_shadow_rows = np.asarray(first_moment_shadow_rows)
    second_moment_shadow_rows = np.asarray(second_moment_shadow_rows)
    shadow_preconditioner_rows = np.asarray(shadow_preconditioner_rows)
    optimizer_step_after_rows = np.asarray(
        optimizer_step_after_rows, dtype=np.int64)
    beta1_rows = np.asarray(beta1_rows, dtype=np.float64)
    beta2_rows = np.asarray(beta2_rows, dtype=np.float64)
    epsilon_rows = np.asarray(epsilon_rows, dtype=np.float64)
    learning_rates = np.asarray(learning_rates)
    weight_decays = np.asarray(weight_decays)
    decay_masks = np.asarray(decay_masks)
    optimizer_decays = np.asarray(optimizer_decays)
    contexts = q.shape[1]
    roundoff = np.zeros((updates, contexts), dtype=np.float64)
    factor = np.zeros_like(roundoff)
    forcing = np.zeros_like(roundoff)
    if lock:
        factor, forcing = _frozen_affine(lock, updates, contexts)
    else:
        for step in range(updates):
            shadow = gamma[step] - learning_rates[step] * (
                optimizer_decays[step] * gamma[step] + directions[step])
            for context_index in range(contexts):
                shape = shape_metric_secant(
                    shadow, q[step, context_index], q[step + 1, context_index],
                    linear_q[step, context_index],
                    quadratic_q[step, context_index])
                native = native_roundoff_charge(
                    shadow, gamma[step + 1], q[step + 1, context_index])
                roundoff[step, context_index] = native["floating_point_charge"]
                exact = executed_source_margin(
                    gamma[step], q[step, context_index], directions[step],
                    learning_rate=learning_rates[step],
                    weight_decay=weight_decays[step],
                    decay_mask=decay_masks[step],
                    signed_shape_gain=shape["signed_linear_gain"],
                    shape_quadratic_charge=shape["quadratic_charge"],
                    floating_point_charge=roundoff[step, context_index])
                bridge = bridge_sector_replay(
                    gamma[step], q[step, context_index], directions[step],
                    sector_rows[step])
                certified = certified_bridge_margin(
                    learning_rates[step],
                    bridge["sector_gain_without_learning_rate"])
                numerator_before_fp = (
                    exact["margin_numerator"]
                    + exact["floating_point_charge"]
                    - exact["adaptive_gain"]
                    + certified["certified_adaptive_gain"])
                factor[step, context_index] = max(
                    0.0, 1.0 - numerator_before_fp
                    / max(energy[step, context_index], 1e-300))
                forcing[step, context_index] = roundoff[step, context_index]
    if lock:
        for step in range(updates):
            shadow = gamma[step] - learning_rates[step] * (
                optimizer_decays[step] * gamma[step] + directions[step])
            for context_index in range(contexts):
                roundoff[step, context_index] = native_roundoff_charge(
                    shadow, gamma[step + 1], q[step + 1, context_index]
                )["floating_point_charge"]
    plga = _plga_arrays(
        model, context_rows, [row["id"] for row in registry[role_key]],
        block_head_index, layer, energy[-1, 0], resistance[0])
    elapsed = time.monotonic() - started
    peak_host = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_host *= 1024
    write_npz_atomic(
        arguments.output,
        schema_version=np.asarray(LEDGER_SCHEMA),
        campaign_id=np.asarray(CAMPAIGN_ID),
        stage=np.asarray(stage),
        job_id=np.asarray(arguments.job_id),
        seed=np.asarray(int(arguments.seed), dtype=np.int64),
        role=np.asarray(arguments.role),
        layer=np.asarray(layer, dtype=np.int64),
        anchor=np.asarray(anchor, dtype=np.int64),
        block_length=np.asarray(updates, dtype=np.int64),
        native_dtype=np.asarray("float32"),
        shadow_dtype=np.asarray("float64"),
        checkpoint_sha256=np.asarray(checkpoint["content_sha256"]),
        checkpoint_complete_state_sha256=np.asarray(
            checkpoint["state_manifest"]["complete_state_sha256"]),
        checkpoint_branch_state_sha256=np.asarray(
            checkpoint["state_manifest"]["branch_state_sha256"]),
        checkpoint_measurement_registry_sha256=np.asarray(
            checkpoint["state_manifest"]["measurement_registry_sha256"]),
        gate_parameter_name=np.asarray(gate_parameter_name),
        checkpoint_gate_decay_mask=checkpoint_decay_mask,
        checkpoint_gate_decay_mask_sha256=np.asarray(
            _array_sha256(checkpoint_decay_mask)),
        probe_registry_sha256=np.asarray(registry["registry_sha256"]),
        probe_chunk_ids_sha256=np.asarray(probe_digest),
        union_graph_sha256=np.asarray(union_graph_digest),
        graph_rule=np.asarray(REGISTRY["graph_rule"]),
        graph_transfer_mapping=np.asarray(REGISTRY["graph_transfer"]),
        graph_vertex_count=np.asarray(
            packed_graph["vertex_count"], dtype=np.int64),
        graph_edges=packed_graph["edges"],
        graph_weights=packed_graph["weights"],
        block_vertex_offsets=source["block_vertex_offsets"],
        block_head_index=source["block_head_index"],
        union_source_physical=union_source_physical,
        union_source_physical_sha256=np.asarray(
            union_source_physical_sha256),
        producer_sha256=np.asarray(sha256_path(__file__)),
        resource_resolved_device=np.asarray(str(device)),
        local_step=np.arange(anchor, anchor + updates + 1, dtype=np.int64),
        node_probe_registry_sha256=np.full(
            updates + 1, registry["registry_sha256"]),
        node_probe_chunk_ids_sha256=np.full(updates + 1, probe_digest),
        update_batch_sha256=np.asarray(update_digests),
        gamma=gamma,
        normalized_edge_contrast=edge_contrast,
        q=q,
        linear_q=linear_q,
        quadratic_q=quadratic_q,
        adaptive_direction=directions,
        bridge_sector_direction=sector_rows,
        source_microbatch_weight=source_microbatch_weight_rows,
        source_microbatch_nonpadding_count=source_microbatch_count_rows,
        bridge_sector_microbatch_direction=sector_microbatch_rows,
        native_gate_first_moment_before=first_moment_before_rows,
        native_gate_second_moment_before=second_moment_before_rows,
        realized_clipped_gate_gradient=clipped_gradient_rows,
        shadow_gate_first_moment_after=first_moment_shadow_rows,
        shadow_gate_second_moment_after=second_moment_shadow_rows,
        shadow_gate_preconditioner=shadow_preconditioner_rows,
        optimizer_step_after=optimizer_step_after_rows,
        adam_beta1=beta1_rows,
        adam_beta2=beta2_rows,
        adam_epsilon=epsilon_rows,
        learning_rate=learning_rates,
        weight_decay=weight_decays,
        gate_decay_mask=decay_masks,
        optimizer_gate_weight_decay=optimizer_decays,
        roundoff_charge=roundoff,
        affine_factor=factor,
        affine_forcing=forcing,
        vertex_diameter=diameter,
        effective_resistance_max=resistance,
        producer_energy=energy,
        **plga,
        resource_elapsed_seconds=np.asarray(elapsed),
        resource_gpu_device_seconds=np.asarray(
            elapsed if device.type == "cuda" else 0.0),
        resource_peak_gpu_allocated_bytes=np.asarray(
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else 0, dtype=np.int64),
        resource_peak_gpu_reserved_bytes=np.asarray(
            int(torch.cuda.max_memory_reserved(device))
            if device.type == "cuda" else 0, dtype=np.int64),
        resource_peak_host_bytes=np.asarray(peak_host, dtype=np.int64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("B", "H", "M"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--role", choices=("construction", "heldout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--updates", type=int, default=16)
    parser.add_argument("--lock", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output", required=True)
    produce(parser.parse_args())


if __name__ == "__main__":
    main()
