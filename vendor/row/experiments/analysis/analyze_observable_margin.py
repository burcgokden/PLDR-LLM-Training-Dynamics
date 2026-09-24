#!/usr/bin/env python3
"""Independent analyzer for fixed-probe observable-margin ledgers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    sha256_path,
    write_json_atomic,
)
from gate_shape import plga_secant_chain  # noqa: E402
from observable_margin import (  # noqa: E402
    block_affine_convolution,
    bridge_sector_replay,
    certified_bridge_margin,
    deterministic_knn_mst_graph,
    effective_resistance_bound,
    effective_resistance_constant,
    executed_source_margin,
    gate_first_and_shapley,
    graph_energy,
    native_roundoff_charge,
    occupied_power_coefficient,
    shape_from_edge_contrasts,
    shape_metric_secant,
    three_factor_score_secant,
    unpack_union_graph,
)
from observable_margin_specs import (  # noqa: E402
    ARCHITECTURE,
    ARITHMETIC,
    BRIDGE_SECTORS,
    CAMPAIGN_ID,
    REGISTRY,
    TRANSFER,
)


LEDGER_SCHEMA = "pldr-observable-margin-ledger-v1"
REPORT_SCHEMA = "pldr-observable-margin-analysis-v1"
REQUIRED = {
    "schema_version", "campaign_id", "stage", "job_id", "seed", "role",
    "layer", "anchor", "block_length", "native_dtype", "shadow_dtype",
    "checkpoint_sha256", "checkpoint_complete_state_sha256",
    "checkpoint_branch_state_sha256", "checkpoint_measurement_registry_sha256",
    "gate_parameter_name", "checkpoint_gate_decay_mask",
    "checkpoint_gate_decay_mask_sha256", "probe_registry_sha256",
    "probe_chunk_ids_sha256", "union_graph_sha256", "graph_rule",
    "graph_transfer_mapping", "graph_vertex_count", "graph_edges",
    "graph_weights", "block_vertex_offsets", "block_head_index",
    "union_source_physical", "union_source_physical_sha256",
    "producer_sha256", "resource_resolved_device",
    "local_step", "node_probe_registry_sha256",
    "node_probe_chunk_ids_sha256", "update_batch_sha256", "gamma",
    "normalized_edge_contrast", "q",
    "linear_q", "quadratic_q", "adaptive_direction",
    "bridge_sector_direction", "source_microbatch_weight",
    "source_microbatch_nonpadding_count",
    "bridge_sector_microbatch_direction",
    "native_gate_first_moment_before",
    "native_gate_second_moment_before", "realized_clipped_gate_gradient",
    "shadow_gate_first_moment_after", "shadow_gate_second_moment_after",
    "shadow_gate_preconditioner", "optimizer_step_after", "adam_beta1",
    "adam_beta2", "adam_epsilon", "learning_rate", "weight_decay",
    "gate_decay_mask", "optimizer_gate_weight_decay", "roundoff_charge",
    "affine_factor", "affine_forcing",
    "vertex_diameter", "effective_resistance_max", "producer_energy",
    "plga_pair_id", "plga_context_index", "plga_head", "plga_anchor_row",
    "plga_generator", "plga_collapsed_anchor_generator", "plga_left",
    "plga_right", "plga_weight", "plga_bias", "plga_exponent",
    "plga_coupling", "plga_coupling_bias", "plga_coefficient",
    "plga_curvature_left", "plga_curvature_right",
    "plga_secant_prediction", "plga_secant_identity_residual",
    "plga_secant_operator_bound", "plga_generator_difference_norm",
    "plga_stacked_generator_bound", "plga_reference_centered_logits",
    "plga_candidate_centered_logits", "plga_reference_margin",
    "plga_reference_winner", "plga_candidate_winner",
    "plga_centered_logit_difference_norm", "plga_downstream_score_ratio",
    "plga_reference_query", "plga_candidate_query",
    "plga_reference_key", "plga_candidate_key",
    "plga_reference_pre_mask_score", "plga_candidate_pre_mask_score",
    "plga_score_query_term", "plga_score_generator_term",
    "plga_score_key_term", "plga_score_difference_prediction",
    "plga_score_identity_residual", "plga_query_operator_norm",
    "plga_key_operator_norm", "plga_score_difference_norm",
    "plga_candidate_cache_replay_residual",
}


def _scalar(raw, name, cast=float):
    if name not in raw or np.asarray(raw[name]).shape != ():
        raise ValueError(f"{name} must be one scalar")
    return cast(np.asarray(raw[name]).item())


def _relative(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = max(float(np.linalg.norm(left)), float(np.linalg.norm(right)), 1e-300)
    return float(np.linalg.norm(left - right)) / scale


def _array_sha256(value):
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _finite_archive(path):
    raw = load_npz(path, required=REQUIRED)
    for name, value in raw.items():
        if value.dtype.hasobject:
            raise ValueError(f"{name} uses object dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains a nonfinite value")
    if _scalar(raw, "schema_version", str) != LEDGER_SCHEMA:
        raise ValueError("unknown observable-margin ledger schema")
    if _scalar(raw, "campaign_id", str) != CAMPAIGN_ID:
        raise ValueError("observable ledger belongs to another campaign")
    return raw


def _load_lock(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("schema_version") != "pldr-observable-margin-construction-lock-v1":
        raise ValueError("unknown observable construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("observable construction lock digest does not replay")
    return value


def analyze(path, lock_path=None):
    path = Path(path).resolve()
    raw = _finite_archive(path)
    stage = _scalar(raw, "stage", str)
    role = _scalar(raw, "role", str)
    if stage not in {"B", "H", "M", "I"}:
        raise ValueError("observable ledger has a non-measurement stage")
    if role not in {"construction", "heldout", "control"}:
        raise ValueError("observable ledger role is invalid")
    if _scalar(raw, "native_dtype", str) != "float32":
        raise ValueError("deciding ledger did not execute the native float32 update")
    if _scalar(raw, "shadow_dtype", str) != "float64":
        raise ValueError("deciding ledger is not a float64 shadow replay")

    gamma = np.asarray(raw["gamma"], dtype=np.float64)
    edge_contrast = np.asarray(
        raw["normalized_edge_contrast"], dtype=np.float64)
    recorded_q = np.asarray(raw["q"], dtype=np.float64)
    recorded_linear_q = np.asarray(raw["linear_q"], dtype=np.float64)
    recorded_quadratic_q = np.asarray(raw["quadratic_q"], dtype=np.float64)
    direction = np.asarray(raw["adaptive_direction"], dtype=np.float64)
    sector_direction = np.asarray(raw["bridge_sector_direction"], dtype=np.float64)
    source_microbatch_weight = np.asarray(
        raw["source_microbatch_weight"], dtype=np.float64)
    source_microbatch_count = np.asarray(
        raw["source_microbatch_nonpadding_count"], dtype=np.float64)
    sector_microbatch_direction = np.asarray(
        raw["bridge_sector_microbatch_direction"], dtype=np.float64)
    first_moment_before = np.asarray(
        raw["native_gate_first_moment_before"], dtype=np.float64)
    second_moment_before = np.asarray(
        raw["native_gate_second_moment_before"], dtype=np.float64)
    clipped_gradient = np.asarray(
        raw["realized_clipped_gate_gradient"], dtype=np.float64)
    recorded_first_shadow = np.asarray(
        raw["shadow_gate_first_moment_after"], dtype=np.float64)
    recorded_second_shadow = np.asarray(
        raw["shadow_gate_second_moment_after"], dtype=np.float64)
    recorded_preconditioner = np.asarray(
        raw["shadow_gate_preconditioner"], dtype=np.float64)
    optimizer_step_after = np.asarray(raw["optimizer_step_after"], dtype=np.int64)
    beta1 = np.asarray(raw["adam_beta1"], dtype=np.float64)
    beta2 = np.asarray(raw["adam_beta2"], dtype=np.float64)
    epsilon = np.asarray(raw["adam_epsilon"], dtype=np.float64)
    learning_rate = np.asarray(raw["learning_rate"], dtype=np.float64)
    weight_decay = np.asarray(raw["weight_decay"], dtype=np.float64)
    decay_mask = np.asarray(raw["gate_decay_mask"], dtype=np.float64)
    checkpoint_decay_mask = np.asarray(
        raw["checkpoint_gate_decay_mask"], dtype=np.float64)
    optimizer_decay = np.asarray(
        raw["optimizer_gate_weight_decay"], dtype=np.float64)
    recorded_roundoff = np.asarray(raw["roundoff_charge"], dtype=np.float64)
    factors = np.asarray(raw["affine_factor"], dtype=np.float64)
    forcing = np.asarray(raw["affine_forcing"], dtype=np.float64)
    diameter = np.asarray(raw["vertex_diameter"], dtype=np.float64)
    resistance = np.asarray(raw["effective_resistance_max"], dtype=np.float64)
    producer_energy = np.asarray(raw["producer_energy"], dtype=np.float64)
    steps = np.asarray(raw["local_step"], dtype=np.int64)
    transitions = len(gamma) - 1
    if gamma.ndim != 2 or transitions < 1:
        raise ValueError("gamma must contain at least one transition")
    dimension = gamma.shape[1]
    if (
        recorded_q.ndim != 3 or recorded_q.shape[0] != len(gamma)
        or recorded_q.shape[2] != dimension
    ):
        raise ValueError("q has the wrong node/context/gate shape")
    contexts = recorded_q.shape[1]
    expected = {
        "linear_q": (transitions, contexts, dimension),
        "quadratic_q": (transitions, contexts, dimension),
        "adaptive_direction": (transitions, dimension),
        "bridge_sector_direction": (transitions, len(BRIDGE_SECTORS), dimension),
        "native_gate_first_moment_before": (transitions, dimension),
        "native_gate_second_moment_before": (transitions, dimension),
        "realized_clipped_gate_gradient": (transitions, dimension),
        "shadow_gate_first_moment_after": (transitions, dimension),
        "shadow_gate_second_moment_after": (transitions, dimension),
        "shadow_gate_preconditioner": (transitions, dimension),
        "optimizer_step_after": (transitions,),
        "adam_beta1": (transitions,),
        "adam_beta2": (transitions,),
        "adam_epsilon": (transitions,),
        "learning_rate": (transitions,),
        "weight_decay": (transitions,),
        "gate_decay_mask": (transitions, dimension),
        "optimizer_gate_weight_decay": (transitions, dimension),
        "roundoff_charge": (transitions, contexts),
        "affine_factor": (transitions, contexts),
        "affine_forcing": (transitions, contexts),
        "vertex_diameter": (transitions + 1, contexts),
        "effective_resistance_max": (contexts,),
        "producer_energy": (transitions + 1, contexts),
        "local_step": (transitions + 1,),
        "node_probe_registry_sha256": (transitions + 1,),
        "node_probe_chunk_ids_sha256": (transitions + 1,),
        "update_batch_sha256": (transitions,),
    }
    values = {
        "linear_q": recorded_linear_q,
        "quadratic_q": recorded_quadratic_q,
        "adaptive_direction": direction,
        "bridge_sector_direction": sector_direction,
        "native_gate_first_moment_before": first_moment_before,
        "native_gate_second_moment_before": second_moment_before,
        "realized_clipped_gate_gradient": clipped_gradient,
        "shadow_gate_first_moment_after": recorded_first_shadow,
        "shadow_gate_second_moment_after": recorded_second_shadow,
        "shadow_gate_preconditioner": recorded_preconditioner,
        "optimizer_step_after": optimizer_step_after,
        "adam_beta1": beta1,
        "adam_beta2": beta2,
        "adam_epsilon": epsilon,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gate_decay_mask": decay_mask,
        "optimizer_gate_weight_decay": optimizer_decay,
        "roundoff_charge": recorded_roundoff,
        "affine_factor": factors,
        "affine_forcing": forcing,
        "vertex_diameter": diameter,
        "effective_resistance_max": resistance,
        "producer_energy": producer_energy,
        "local_step": steps,
        "node_probe_registry_sha256": raw["node_probe_registry_sha256"],
        "node_probe_chunk_ids_sha256": raw["node_probe_chunk_ids_sha256"],
        "update_batch_sha256": raw["update_batch_sha256"],
    }
    for name, shape in expected.items():
        if values[name].shape != shape:
            raise ValueError(f"{name} has shape {values[name].shape}, expected {shape}")
    microbatch_count = ARCHITECTURE["batch_size"] // 8
    if (
        source_microbatch_weight.shape != (transitions, microbatch_count)
        or source_microbatch_count.shape != (transitions, microbatch_count)
        or sector_microbatch_direction.shape != (
            transitions, microbatch_count, len(BRIDGE_SECTORS), dimension)
        or np.any(source_microbatch_weight < 0.0)
        or np.any(source_microbatch_count <= 0.0)
    ):
        raise ValueError("source microbatch sector evidence is malformed")
    normalized_source_count = source_microbatch_count / np.sum(
        source_microbatch_count, axis=1, keepdims=True)
    source_weight_residual = _relative(
        source_microbatch_weight, normalized_source_count)
    replay_sector_mean = np.sum(
        source_microbatch_weight[:, :, None, None]
        * sector_microbatch_direction, axis=1)
    source_sector_mean_residual = _relative(
        replay_sector_mean, sector_direction)
    source_sector_dispersion = np.sqrt(np.sum(
        source_microbatch_weight[:, :, None]
        * np.sum(
            (sector_microbatch_direction - sector_direction[:, None]) ** 2,
            axis=3),
        axis=1))
    if not np.array_equal(steps, np.arange(steps[0], steps[0] + len(steps))):
        raise ValueError("local steps are not consecutive")

    graph_digest = _scalar(raw, "union_graph_sha256", str)
    graph_rule = _scalar(raw, "graph_rule", str)
    graph_mapping = _scalar(raw, "graph_transfer_mapping", str)
    if (
        graph_rule != REGISTRY["graph_rule"]
        or graph_mapping != REGISTRY["graph_transfer"]
    ):
        raise ValueError("ledger union-graph rule or identity mapping is not registered")
    graph = unpack_union_graph(
        _scalar(raw, "graph_vertex_count", int), raw["graph_edges"],
        raw["graph_weights"], rule=graph_rule,
        neighbors=REGISTRY["neighbors"], expected_sha256=graph_digest)
    if contexts != 1:
        raise ValueError("observable energy must use one connected union graph")
    block_offsets = np.asarray(raw["block_vertex_offsets"], dtype=np.int64)
    block_heads = np.asarray(raw["block_head_index"], dtype=np.int64)
    block_count = REGISTRY[
        "construction_contexts" if role == "construction" else "heldout_contexts"]
    expected_offsets = np.arange(
        block_count + 1, dtype=np.int64) * ARCHITECTURE["row_width"]
    expected_heads = np.arange(
        block_count, dtype=np.int64) % ARCHITECTURE["heads"]
    if (
        not np.array_equal(block_offsets, expected_offsets)
        or not np.array_equal(block_heads, expected_heads)
        or graph["vertex_count"] != int(expected_offsets[-1])
    ):
        raise ValueError("ordered union context-head block registry changed")
    if edge_contrast.shape != (
        transitions + 1, len(graph["edges"]), dimension,
    ):
        raise ValueError("packed normalized edge contrasts have the wrong shape")
    expected_weight = np.full(
        len(graph["edges"]), 1.0 / len(graph["edges"]), dtype=np.float64)
    graph_uniform_weights = bool(np.array_equal(graph["weights"], expected_weight))
    ordered = [tuple(map(int, edge)) for edge in graph["edges"]]
    graph_unique_ordered_edges = bool(
        ordered == sorted(set(ordered))
        and all(left < right for left, right in ordered))

    source_physical = np.asarray(raw["union_source_physical"], dtype=np.float64)
    source_physical_digest = _scalar(
        raw, "union_source_physical_sha256", str)
    if source_physical.shape != (
        graph["vertex_count"], ARCHITECTURE["row_width"],
    ):
        raise ValueError("construction union-graph coordinates have the wrong shape")
    source_coordinate_digest_replay = bool(
        _array_sha256(source_physical) == source_physical_digest)
    construction_algorithm_replay = True
    if role == "construction":
        rebuilt = deterministic_knn_mst_graph(
            source_physical, neighbors=REGISTRY["neighbors"])
        construction_algorithm_replay = bool(
            rebuilt["graph_sha256"] == graph["graph_sha256"]
            and np.array_equal(rebuilt["edges"], graph["edges"])
            and np.array_equal(rebuilt["weights"], graph["weights"]))

    replay_resistance_value = effective_resistance_constant(
        graph["vertex_count"], graph["edges"], graph["weights"]
    )["effective_resistance_max"]
    replay_resistance = np.asarray([replay_resistance_value], dtype=np.float64)
    resistance_replay_residual = _relative(resistance, replay_resistance)
    q = np.empty_like(recorded_q)
    linear_q = np.empty_like(recorded_linear_q)
    quadratic_q = np.empty_like(recorded_quadratic_q)
    replay_diameter = np.empty_like(diameter)
    edge_cycle_residual = 0.0
    final_union_shape_rows = None
    for node in range(transitions + 1):
        contrast = edge_contrast[node]
        reconstruction = shape_from_edge_contrasts(
            graph["vertex_count"], graph["edges"], contrast)
        edge_cycle_residual = max(
            edge_cycle_residual, reconstruction["relative_cycle_residual"])
        if node == transitions:
            final_union_shape_rows = reconstruction["anchored_rows"].copy()
        geometry = graph_energy(
            gamma[node], reconstruction["anchored_rows"],
            graph["edges"], graph["weights"])
        q[node, 0] = geometry["q"]
        replay_diameter[node, 0] = geometry["vertex_diameter"]
        if node < transitions:
            increment_edge = edge_contrast[node + 1] - contrast
            linear_q[node, 0] = 2.0 * np.sum(
                graph["weights"][:, None] * contrast * increment_edge, axis=0)
            quadratic_q[node, 0] = np.sum(
                graph["weights"][:, None] * increment_edge * increment_edge,
                axis=0)
    q_replay_residual = _relative(q, recorded_q)
    linear_q_replay_residual = _relative(linear_q, recorded_linear_q)
    quadratic_q_replay_residual = _relative(
        quadratic_q, recorded_quadratic_q)
    diameter_replay_residual = _relative(replay_diameter, diameter)
    diameter = replay_diameter
    if np.any(q < 0.0) or np.any(quadratic_q < -1e-13):
        raise ValueError("q or quadratic shape charge is negative")
    if np.any((decay_mask != 0.0) & (decay_mask != 1.0)):
        raise ValueError("gate decay mask is not binary")
    expected_gate_name = (
        f"decoder.dec_layers.{_scalar(raw, 'layer', int)}.mha1."
        "reslayerAs.7.layernormA.weight")
    if _scalar(raw, "gate_parameter_name", str) != expected_gate_name:
        raise ValueError("ledger gate parameter identity changed")
    if (
        checkpoint_decay_mask.shape != (dimension,)
        or np.any((checkpoint_decay_mask != 0.0) & (checkpoint_decay_mask != 1.0))
        or _array_sha256(checkpoint_decay_mask)
        != _scalar(raw, "checkpoint_gate_decay_mask_sha256", str)
        or not np.all(decay_mask == checkpoint_decay_mask[None, :])
    ):
        raise ValueError("checkpoint-bound gate decay mask does not replay")

    if (
        np.any(second_moment_before < 0.0)
        or np.any(recorded_second_shadow < 0.0)
        or np.any((beta1 < 0.0) | (beta1 >= 1.0))
        or np.any((beta2 < 0.0) | (beta2 >= 1.0))
        or np.any(epsilon <= 0.0)
        or not np.array_equal(optimizer_step_after, steps[1:])
    ):
        raise ValueError("shadow moment domain or optimizer clocks are invalid")
    replay_first_shadow = (
        beta1[:, None] * first_moment_before
        + (1.0 - beta1[:, None]) * clipped_gradient)
    replay_second_shadow = (
        beta2[:, None] * second_moment_before
        + (1.0 - beta2[:, None]) * clipped_gradient * clipped_gradient)
    replay_preconditioner = 1.0 / (
        np.sqrt(replay_second_shadow / (
            1.0 - beta2[:, None] ** optimizer_step_after[:, None]))
        + epsilon[:, None])
    replay_adaptive_direction = (
        replay_first_shadow / (
            1.0 - beta1[:, None] ** optimizer_step_after[:, None])
        * replay_preconditioner)
    first_shadow_residual = _relative(
        replay_first_shadow, recorded_first_shadow)
    second_shadow_residual = _relative(
        replay_second_shadow, recorded_second_shadow)
    preconditioner_shadow_residual = _relative(
        replay_preconditioner, recorded_preconditioner)
    adaptive_shadow_residual = _relative(
        replay_adaptive_direction, direction)

    registry_digest = _scalar(raw, "probe_registry_sha256", str)
    chunk_digest = _scalar(raw, "probe_chunk_ids_sha256", str)
    node_registry = np.asarray(raw["node_probe_registry_sha256"], dtype=np.str_)
    node_chunks = np.asarray(raw["node_probe_chunk_ids_sha256"], dtype=np.str_)
    update_batches = np.asarray(raw["update_batch_sha256"], dtype=np.str_)
    fixed_registry = bool(np.all(node_registry == registry_digest))
    fixed_chunks = bool(np.all(node_chunks == chunk_digest))
    update_probe_separation = bool(np.all(update_batches != chunk_digest))

    energy = np.einsum("nd,ncd,nd->nc", gamma, q, gamma)
    energy_residual = _relative(energy, producer_energy)
    lock = _load_lock(lock_path) if lock_path else None
    lock_transfer = None
    if lock is not None:
        selected = lock["selected"]
        if (
            _scalar(raw, "anchor", int) != selected["anchor"]
            or _scalar(raw, "layer", int) != selected["layer"]
            or _scalar(raw, "block_length", int) != selected["block_length"]
        ):
            raise ValueError("held-out ledger cell differs from construction lock")
        frozen = lock["frozen_envelopes"]
        locked_factor = np.asarray(
            frozen["affine_factor_upper_by_step"], dtype=np.float64)
        locked_forcing = np.asarray(
            frozen["affine_forcing_upper_by_step"], dtype=np.float64)
        locked_roundoff = np.asarray(
            frozen["roundoff_charge_upper_by_step"], dtype=np.float64)
        locked_adaptive = np.asarray(
            frozen["adaptive_direction_norm_upper_by_step"], dtype=np.float64)
        locked_sectors = np.asarray(
            frozen["bridge_sector_norm_upper_by_step"], dtype=np.float64)
        if not (
            locked_factor.shape == locked_forcing.shape == locked_roundoff.shape
            == locked_adaptive.shape == (transitions,)
            and locked_sectors.shape == (transitions, len(BRIDGE_SECTORS))
        ):
            raise ValueError("construction lock envelope arrays are malformed")
        factor_residual = _relative(
            factors, np.repeat(locked_factor[:, None], contexts, axis=1))
        forcing_residual = _relative(
            forcing, np.repeat(locked_forcing[:, None], contexts, axis=1))
        locked_resistance = float(lock["graph"]["effective_resistance"])
        resistance_excess = abs(
            replay_resistance_value - locked_resistance) / max(
                replay_resistance_value, locked_resistance, 1e-300)
        roundoff_excess = float(np.max(
            recorded_roundoff - locked_roundoff[:, None])) / max(
                float(np.max(locked_roundoff)),
                float(np.max(recorded_roundoff)), 1e-300)
        adaptive_norm = np.linalg.norm(direction, axis=1)
        adaptive_excess = float(np.max(adaptive_norm - locked_adaptive)) / max(
            float(np.max(locked_adaptive)), float(np.max(adaptive_norm)), 1e-300)
        sector_norm = np.linalg.norm(sector_direction, axis=2)
        sector_excess = float(np.max(sector_norm - locked_sectors)) / max(
            float(np.max(locked_sectors)), float(np.max(sector_norm)), 1e-300)
        energy_path = np.asarray(frozen["energy_upper_path"], dtype=np.float64)
        diameter_path = np.asarray(frozen["diameter_upper_path"], dtype=np.float64)
        if energy_path.shape != (transitions + 1,) or diameter_path.shape != (
            transitions + 1,
        ):
            raise ValueError("construction prediction paths have the wrong length")
        energy_coverage_excess = float(np.max(energy - energy_path[:, None])) / max(
            float(np.max(energy)), float(np.max(energy_path)), 1e-300)
        diameter_coverage_excess = float(
            np.max(diameter - diameter_path[:, None])) / max(
                float(np.max(diameter)), float(np.max(diameter_path)), 1e-300)
        lock_transfer = {
            "lock_sha256": lock["lock_sha256"],
            "selected_cell_unchanged": True,
            "probe_registry_unchanged": (
                registry_digest == lock["bindings"]["probe_registry_sha256"]),
            "graph_rule_unchanged": (
                graph_rule == lock["graph"]["rule"]),
            "graph_identity_mapping_unchanged": (
                graph_mapping == lock["graph"]["mapping"]),
            "union_graph_sha256_unchanged": (
                graph_digest == lock["graph"]["union_graph_sha256"]),
            "factor_relative_residual": factor_residual,
            "forcing_relative_residual": forcing_residual,
            "resistance_relative_excess": resistance_excess,
            "roundoff_relative_excess": roundoff_excess,
            "adaptive_relative_excess": adaptive_excess,
            "sector_relative_excess": sector_excess,
            "energy_coverage_relative_excess": energy_coverage_excess,
            "diameter_coverage_relative_excess": diameter_coverage_excess,
            "enlargement_allowed": False,
        }
    shape_residuals = []
    ledger_residuals = []
    roundoff_residuals = []
    bridge_residuals = []
    margins = np.zeros((transitions, contexts), dtype=np.float64)
    signed_roundoff = np.zeros_like(margins)
    attribution_residuals = []
    sector_gains = {name: [] for name in BRIDGE_SECTORS}
    certified_adaptive = np.zeros_like(margins)
    aggregate_adaptive = np.zeros_like(margins)
    aggregate_gain_residuals = []
    decay_residual = _relative(
        optimizer_decay, weight_decay[:, None] * decay_mask)
    for index in range(transitions):
        shadow_gamma = gamma[index] - learning_rate[index] * (
            optimizer_decay[index] * gamma[index] + direction[index])
        for context in range(contexts):
            bridge = bridge_sector_replay(
                gamma[index], q[index, context], direction[index],
                sector_direction[index],
            )
            bridge_residuals.append(bridge["relative_direction_residual"])
            certified = certified_bridge_margin(
                learning_rate[index],
                bridge["sector_gain_without_learning_rate"],
            )
            certified_adaptive[index, context] = certified[
                "certified_adaptive_gain"]
            aggregate_adaptive[index, context] = certified[
                "aggregate_adaptive_gain"]
            for name in BRIDGE_SECTORS:
                sector_gains[name].append(
                    learning_rate[index]
                    * bridge["sector_gain_without_learning_rate"][name])
            shape = shape_metric_secant(
                shadow_gamma, q[index, context], q[index + 1, context],
                linear_q[index, context], quadratic_q[index, context],
            )
            shape_residuals.append(shape["relative_reconstruction_residual"])
            roundoff = native_roundoff_charge(
                shadow_gamma, gamma[index + 1], q[index + 1, context])
            roundoff_residuals.append(roundoff["relative_reconstruction_residual"])
            signed_roundoff[index, context] = roundoff["signed_work"]
            charge_scale = max(
                abs(roundoff["floating_point_charge"]),
                abs(recorded_roundoff[index, context]), 1e-300,
            )
            roundoff_residuals.append(
                abs(roundoff["floating_point_charge"]
                    - recorded_roundoff[index, context]) / charge_scale)
            exact = executed_source_margin(
                gamma[index], q[index, context], direction[index],
                learning_rate=learning_rate[index],
                weight_decay=weight_decay[index],
                decay_mask=decay_mask[index],
                signed_shape_gain=shape["signed_linear_gain"],
                shape_quadratic_charge=shape["quadratic_charge"],
                floating_point_charge=recorded_roundoff[index, context],
            )
            aggregate_gain_residuals.append(_relative(
                [exact["adaptive_gain"]],
                [certified["aggregate_adaptive_gain"]],
            ))
            # The exact aggregate bridge remains an identity diagnostic.  The
            # deciding observable margin follows the registered sector rule:
            # signed Fisher and signed-jet credit, absolute charge for every
            # remaining bridge projection.
            margins[index, context] = (
                exact["margin_numerator"] - exact["adaptive_gain"]
                + certified["certified_adaptive_gain"]
            )
            ideal_next = (
                exact["fixed_shape_energy_next"] + shape["shape_work"]
                + roundoff["signed_work"]
            )
            ledger_residuals.append(_relative(
                [ideal_next], [energy[index + 1, context]]))
            attribution = gate_first_and_shapley(
                gamma[index], gamma[index + 1],
                q[index, context], q[index + 1, context])
            attribution_residuals.extend([
                abs(attribution["gate_first_residual"]),
                abs(attribution["shapley_residual"]),
            ])

    affine_enclosure_residual = 0.0
    block_rows = []
    for context in range(contexts):
        replay = block_affine_convolution(
            energy[0, context], factors[:, context], forcing[:, context])
        scale = max(float(np.max(energy[:, context])), 1e-300)
        affine_enclosure_residual = max(
            affine_enclosure_residual,
            float(np.max(energy[:, context] - replay["trajectory_bound"])) / scale,
        )
        block_rows.append({
            "context": context,
            "factor_product": replay["factor_product"],
            "forcing_convolution": replay["forcing_convolution"],
            "source_energy": float(energy[0, context]),
            "endpoint_bound": replay["endpoint_bound"],
            "endpoint_energy": float(energy[-1, context]),
            "source_diameter": float(diameter[0, context]),
            "source_diameter_bound": effective_resistance_bound(
                energy[0, context], resistance[context]),
            "endpoint_diameter_bound": effective_resistance_bound(
                replay["endpoint_bound"], resistance[context]),
            "relative_energy_change": float(
                (energy[-1, context] - energy[0, context])
                / max(energy[0, context], 1e-300)),
            "summed_margin": float(np.sum(margins[:, context])),
        })

    graph_excess = 0.0
    for node in range(transitions + 1):
        for context in range(contexts):
            bound = effective_resistance_bound(
                energy[node, context], resistance[context])
            scale = max(bound, diameter[node, context], 1e-300)
            graph_excess = max(
                graph_excess, (diameter[node, context] - bound) / scale)

    plga_pair_id = np.asarray(raw["plga_pair_id"], dtype=np.str_)
    plga_context = np.asarray(raw["plga_context_index"], dtype=np.int64)
    plga_head = np.asarray(raw["plga_head"], dtype=np.int64)
    plga_anchor = np.asarray(raw["plga_anchor_row"], dtype=np.int64)
    plga_generator = np.asarray(raw["plga_generator"], dtype=np.float64)
    plga_collapsed = np.asarray(
        raw["plga_collapsed_anchor_generator"], dtype=np.float64)
    plga_left = np.asarray(raw["plga_left"], dtype=np.float64)
    plga_right = np.asarray(raw["plga_right"], dtype=np.float64)
    plga_weight = np.asarray(raw["plga_weight"], dtype=np.float64)
    plga_bias = np.asarray(raw["plga_bias"], dtype=np.float64)
    plga_exponent = np.asarray(raw["plga_exponent"], dtype=np.float64)
    plga_coupling = np.asarray(raw["plga_coupling"], dtype=np.float64)
    plga_coupling_bias = np.asarray(
        raw["plga_coupling_bias"], dtype=np.float64)
    plga_recorded = np.asarray(raw["plga_coefficient"], dtype=np.float64)
    plga_curvature_left = np.asarray(
        raw["plga_curvature_left"], dtype=np.float64)
    plga_curvature_right = np.asarray(
        raw["plga_curvature_right"], dtype=np.float64)
    plga_prediction = np.asarray(
        raw["plga_secant_prediction"], dtype=np.float64)
    plga_recorded_residual = np.asarray(
        raw["plga_secant_identity_residual"], dtype=np.float64)
    plga_operator = np.asarray(
        raw["plga_secant_operator_bound"], dtype=np.float64)
    plga_generator_norm = np.asarray(
        raw["plga_generator_difference_norm"], dtype=np.float64)
    plga_generator_bound = np.asarray(
        raw["plga_stacked_generator_bound"], dtype=np.float64)
    plga_reference_logits = np.asarray(
        raw["plga_reference_centered_logits"], dtype=np.float64)
    plga_candidate_logits = np.asarray(
        raw["plga_candidate_centered_logits"], dtype=np.float64)
    plga_reference_margin = np.asarray(
        raw["plga_reference_margin"], dtype=np.float64)
    plga_reference_winner = np.asarray(
        raw["plga_reference_winner"], dtype=np.int64)
    plga_candidate_winner = np.asarray(
        raw["plga_candidate_winner"], dtype=np.int64)
    plga_centered_norm = np.asarray(
        raw["plga_centered_logit_difference_norm"], dtype=np.float64)
    plga_downstream_ratio = np.asarray(
        raw["plga_downstream_score_ratio"], dtype=np.float64)
    plga_reference_query = np.asarray(
        raw["plga_reference_query"], dtype=np.float64)
    plga_candidate_query = np.asarray(
        raw["plga_candidate_query"], dtype=np.float64)
    plga_reference_key = np.asarray(
        raw["plga_reference_key"], dtype=np.float64)
    plga_candidate_key = np.asarray(
        raw["plga_candidate_key"], dtype=np.float64)
    plga_reference_score = np.asarray(
        raw["plga_reference_pre_mask_score"], dtype=np.float64)
    plga_candidate_score = np.asarray(
        raw["plga_candidate_pre_mask_score"], dtype=np.float64)
    plga_score_query_term = np.asarray(
        raw["plga_score_query_term"], dtype=np.float64)
    plga_score_generator_term = np.asarray(
        raw["plga_score_generator_term"], dtype=np.float64)
    plga_score_key_term = np.asarray(
        raw["plga_score_key_term"], dtype=np.float64)
    plga_score_prediction = np.asarray(
        raw["plga_score_difference_prediction"], dtype=np.float64)
    plga_recorded_score_residual = np.asarray(
        raw["plga_score_identity_residual"], dtype=np.float64)
    plga_query_norm = np.asarray(
        raw["plga_query_operator_norm"], dtype=np.float64)
    plga_key_norm = np.asarray(
        raw["plga_key_operator_norm"], dtype=np.float64)
    plga_score_norm = np.asarray(
        raw["plga_score_difference_norm"], dtype=np.float64)
    plga_cache_residual = np.asarray(
        raw["plga_candidate_cache_replay_residual"], dtype=np.float64)
    pair_count = block_count
    matrix_dimension = ARCHITECTURE["row_width"]
    matrix_shape = (pair_count, matrix_dimension, matrix_dimension)
    pair_shape = (pair_count,)
    if not (
        plga_pair_id.shape == plga_context.shape == plga_head.shape
        == plga_anchor.shape == pair_shape
        and plga_generator.shape == plga_collapsed.shape == plga_left.shape
        == plga_right.shape == plga_weight.shape == plga_bias.shape
        == plga_exponent.shape == plga_coupling.shape
        == plga_coupling_bias.shape == plga_recorded.shape
        == plga_curvature_left.shape == plga_curvature_right.shape
        == plga_prediction.shape == matrix_shape
        and plga_recorded_residual.shape == plga_operator.shape
        == plga_generator_norm.shape == plga_generator_bound.shape
        == plga_reference_margin.shape == plga_reference_winner.shape
        == plga_candidate_winner.shape == plga_centered_norm.shape
        == plga_downstream_ratio.shape == plga_recorded_score_residual.shape
        == plga_query_norm.shape == plga_key_norm.shape
        == plga_score_norm.shape == plga_cache_residual.shape == pair_shape
        and plga_reference_query.ndim == 3
        and plga_candidate_query.shape == plga_reference_query.shape
        and plga_reference_key.shape == plga_reference_query.shape
        and plga_candidate_key.shape == plga_reference_query.shape
        and plga_reference_query.shape[0] == pair_count
        and plga_reference_query.shape[2] == matrix_dimension
        and plga_reference_score.ndim == 3
        and plga_candidate_score.shape == plga_reference_score.shape
        and plga_score_query_term.shape == plga_reference_score.shape
        and plga_score_generator_term.shape == plga_reference_score.shape
        and plga_score_key_term.shape == plga_reference_score.shape
        and plga_score_prediction.shape == plga_reference_score.shape
        and plga_reference_score.shape[0] == pair_count
        and plga_reference_logits.ndim == 2
        and plga_candidate_logits.shape == plga_reference_logits.shape
        and plga_reference_logits.shape[0] == pair_count
    ):
        raise ValueError("complete PLGA endpoint arrays disagree")
    expected_context = np.arange(block_count, dtype=np.int64)
    expected_ids = np.asarray([
        f"{'omc' if role == 'construction' else 'omh'}{block:03d}-"
        f"l{_scalar(raw, 'layer', int)}-h{int(block_heads[block])}-anchor0"
        for block in range(block_count)
    ])
    if (
        not np.array_equal(plga_context, expected_context)
        or not np.array_equal(plga_head, block_heads)
        or not np.array_equal(plga_pair_id, expected_ids)
        or not np.all(plga_anchor == 0)
    ):
        raise ValueError("PLGA context/head/layer/node/anchor registry is incomplete")

    plga_residuals = []
    plga_rows = []
    plga_positive_bases = True
    plga_energy_bridge = True
    plga_score_transfer = True
    plga_lock_transfer = True
    plga_qualified_count = 0
    locked_plga = lock.get("plga") if lock is not None else None
    if lock is not None and not isinstance(locked_plga, dict):
        raise ValueError("construction lock omits PLGA source envelopes")
    if locked_plga is not None:
        expected_ordinal_binding = [
            {"ordinal": int(index), "head": int(head), "anchor_row": 0}
            for index, head in enumerate(block_heads)
        ]
        if (
            locked_plga.get("selected_layer") != _scalar(raw, "layer", int)
            or locked_plga.get("selected_node") != (
                _scalar(raw, "anchor", int) + _scalar(raw, "block_length", int))
            or locked_plga.get("anchor_row") != 0
            or locked_plga.get("ordinal_context_head_binding")
            != expected_ordinal_binding
        ):
            raise ValueError("locked PLGA comparison identities changed")
        locked_operator = np.asarray(
            locked_plga["secant_operator_upper_by_context_head"],
            dtype=np.float64)
        locked_downstream = np.asarray(
            locked_plga["downstream_response_envelope_by_context_head"],
            dtype=np.float64)
        locked_generator = np.asarray(
            locked_plga["stacked_generator_upper_by_context_head"],
            dtype=np.float64)
        locked_query = np.asarray(
            locked_plga["query_operator_upper_by_context_head"],
            dtype=np.float64)
        locked_key = np.asarray(
            locked_plga["key_operator_upper_by_context_head"],
            dtype=np.float64)
        locked_centered = np.asarray(
            locked_plga["centered_logit_remainder_upper_by_context_head"],
            dtype=np.float64)
        expected_lock_shape = (block_count,)
        if not all(value.shape == expected_lock_shape for value in (
            locked_operator, locked_downstream, locked_generator,
            locked_query, locked_key, locked_centered,
        )):
            raise ValueError("locked PLGA context/head envelopes are malformed")
    for pair in range(pair_count):
        context = int(plga_context[pair])
        head = int(plga_head[pair])
        anchor_row = int(plga_anchor[pair])
        expected_collapsed = np.repeat(
            plga_generator[pair, anchor_row:anchor_row + 1],
            matrix_dimension, axis=0)
        collapsed_residual = _relative(
            plga_collapsed[pair], expected_collapsed)
        replay = plga_secant_chain(
            plga_collapsed[pair], plga_generator[pair], plga_weight[pair],
            plga_bias[pair], plga_exponent[pair], plga_coupling[pair],
            plga_coupling_bias[pair])
        coefficient_residual = 0.0
        for index in np.ndindex((matrix_dimension, matrix_dimension)):
            coefficient = occupied_power_coefficient(
                plga_left[pair][index], plga_right[pair][index],
                plga_exponent[pair][index])
            scale = max(
                abs(plga_recorded[pair][index]),
                abs(coefficient["coefficient"]), 1e-300)
            coefficient_residual = max(
                coefficient_residual,
                abs(plga_recorded[pair][index] - coefficient["coefficient"])
                / scale)
        secant_residual = max(
            collapsed_residual,
            _relative(plga_left[pair], replay["metric_left"]),
            _relative(plga_right[pair], replay["metric_right"]),
            _relative(plga_curvature_left[pair], replay["curvature_left"]),
            _relative(plga_curvature_right[pair], replay["curvature_right"]),
            _relative(plga_prediction[pair], replay["predicted_difference"]),
            coefficient_residual,
            abs(plga_recorded_residual[pair] - replay["identity_residual"])
            / max(abs(plga_recorded_residual[pair]),
                  abs(replay["identity_residual"]), 1e-300),
            abs(plga_operator[pair] - replay["operator_bound"])
            / max(abs(plga_operator[pair]), abs(replay["operator_bound"]),
                  1e-300),
        )
        plga_residuals.append(secant_residual)
        expected_generator_norm = float(np.linalg.norm(
            plga_generator[pair] - plga_collapsed[pair]))
        expected_generator_bound = math.sqrt(max(
            matrix_dimension * resistance[0] * energy[-1, 0], 0.0))
        block_begin = int(block_offsets[context])
        block_end = int(block_offsets[context + 1])
        expected_generator_difference = (
            final_union_shape_rows[block_begin:block_end]
            - final_union_shape_rows[block_begin + anchor_row]) * gamma[-1]
        bridge_residual = max(
            _relative(
                plga_generator[pair] - plga_collapsed[pair],
                expected_generator_difference),
            abs(expected_generator_norm - plga_generator_norm[pair])
            / max(expected_generator_norm, plga_generator_norm[pair], 1e-300),
            abs(expected_generator_bound - plga_generator_bound[pair])
            / max(expected_generator_bound, plga_generator_bound[pair], 1e-300),
        )
        plga_energy_bridge = plga_energy_bridge and bool(
            bridge_residual <= ARITHMETIC["relative_identity_tolerance"]
            and plga_generator_norm[pair]
            <= plga_generator_bound[pair] * (
                1.0 + ARITHMETIC["relative_identity_tolerance"]))
        centered = plga_candidate_logits[pair] - plga_reference_logits[pair]
        centered_norm = float(np.linalg.norm(centered))
        winner = int(np.argmax(plga_reference_logits[pair]))
        margin = float(
            plga_reference_logits[pair, winner]
            - np.max(np.delete(plga_reference_logits[pair], winner)))
        score_chain = three_factor_score_secant(
            plga_reference_query[pair], plga_candidate_query[pair],
            plga_curvature_left[pair], plga_curvature_right[pair],
            plga_reference_key[pair], plga_candidate_key[pair])
        expected_reference_score = score_chain["reference_score"]
        expected_candidate_score = score_chain["candidate_score"]
        expected_score_prediction = score_chain["predicted_difference"]
        score_difference = (
            plga_candidate_score[pair] - plga_reference_score[pair])
        score_norm = float(np.linalg.norm(score_difference))
        query_norm = max(
            float(np.linalg.norm(plga_reference_query[pair], ord=2)),
            float(np.linalg.norm(plga_candidate_query[pair], ord=2)))
        key_norm = max(
            float(np.linalg.norm(plga_reference_key[pair], ord=2)),
            float(np.linalg.norm(plga_candidate_key[pair], ord=2)))
        query_endpoint_fixed = np.array_equal(
            plga_reference_query[pair], plga_candidate_query[pair])
        key_endpoint_fixed = np.array_equal(
            plga_reference_key[pair], plga_candidate_key[pair])
        downstream = centered_norm / max(score_norm, 1e-300)
        exact_score_residual = _relative(
            score_difference, expected_score_prediction)
        score_residual = max(
            abs(np.mean(plga_reference_logits[pair])),
            abs(np.mean(plga_candidate_logits[pair])),
            _relative(plga_reference_score[pair], expected_reference_score),
            _relative(plga_candidate_score[pair], expected_candidate_score),
            _relative(
                plga_score_query_term[pair], score_chain["query_term"]),
            _relative(
                plga_score_generator_term[pair],
                score_chain["generator_term"]),
            _relative(plga_score_key_term[pair], score_chain["key_term"]),
            _relative(plga_score_prediction[pair], expected_score_prediction),
            exact_score_residual,
            abs(plga_recorded_score_residual[pair] - exact_score_residual)
            / max(
                plga_recorded_score_residual[pair],
                exact_score_residual, 1e-300),
            abs(centered_norm - plga_centered_norm[pair])
            / max(centered_norm, plga_centered_norm[pair], 1e-300),
            abs(margin - plga_reference_margin[pair])
            / max(abs(margin), abs(plga_reference_margin[pair]), 1e-300),
            abs(downstream - plga_downstream_ratio[pair])
            / max(downstream, plga_downstream_ratio[pair], 1e-300),
            abs(query_norm - plga_query_norm[pair])
            / max(query_norm, plga_query_norm[pair], 1e-300),
            abs(key_norm - plga_key_norm[pair])
            / max(key_norm, plga_key_norm[pair], 1e-300),
            abs(score_norm - plga_score_norm[pair])
            / max(score_norm, plga_score_norm[pair], 1e-300),
            float(plga_cache_residual[pair]),
        )
        plga_score_transfer = plga_score_transfer and bool(
            score_residual <= ARITHMETIC["relative_identity_tolerance"]
            and query_endpoint_fixed and key_endpoint_fixed
            and np.linalg.norm(score_chain["query_term"])
            <= ARITHMETIC["absolute_identity_tolerance"]
            and np.linalg.norm(score_chain["key_term"])
            <= ARITHMETIC["absolute_identity_tolerance"]
            and plga_reference_winner[pair] == winner
            and plga_candidate_winner[pair]
            == int(np.argmax(plga_candidate_logits[pair])))
        plga_positive_bases = plga_positive_bases and bool(
            np.min(plga_left[pair]) > 0.0 and np.min(plga_right[pair]) > 0.0
            and np.isfinite(plga_exponent[pair]).all())
        locked_bound = None
        qualified = False
        no_enlargement = True
        if locked_plga is not None:
            locked_bound = float(locked_centered[context])
            tolerance_factor = 1.0 + ARITHMETIC["relative_identity_tolerance"]
            no_enlargement = bool(
                plga_operator[pair]
                <= locked_operator[context] * tolerance_factor
                and plga_downstream_ratio[pair]
                <= locked_downstream[context] * tolerance_factor
                and plga_query_norm[pair]
                <= locked_query[context] * tolerance_factor
                and plga_key_norm[pair]
                <= locked_key[context] * tolerance_factor
                and plga_generator_bound[pair]
                <= locked_generator[context] * tolerance_factor
                and plga_centered_norm[pair]
                <= locked_bound * tolerance_factor)
            plga_lock_transfer = plga_lock_transfer and no_enlargement
            qualified = bool(
                no_enlargement and math.isfinite(locked_bound)
                and math.sqrt(2.0) * locked_bound < margin)
            plga_qualified_count += int(qualified)
        plga_rows.append({
            "pair_id": str(plga_pair_id[pair]),
            "context": context,
            "head": head,
            "anchor_row": anchor_row,
            "positive_bases": bool(
                np.min(plga_left[pair]) > 0.0
                and np.min(plga_right[pair]) > 0.0),
            "real_exponent": bool(np.isfinite(plga_exponent[pair]).all()),
            "generator_difference_norm": float(plga_generator_norm[pair]),
            "stacked_generator_bound": float(plga_generator_bound[pair]),
            "secant_operator_bound": float(plga_operator[pair]),
            "query_operator_norm": float(plga_query_norm[pair]),
            "key_operator_norm": float(plga_key_norm[pair]),
            "score_difference_norm": float(plga_score_norm[pair]),
            "query_endpoint_fixed": bool(query_endpoint_fixed),
            "key_endpoint_fixed": bool(key_endpoint_fixed),
            "score_query_term_norm": float(np.linalg.norm(
                score_chain["query_term"])),
            "score_generator_term_norm": float(np.linalg.norm(
                score_chain["generator_term"])),
            "score_key_term_norm": float(np.linalg.norm(
                score_chain["key_term"])),
            "downstream_score_ratio": float(plga_downstream_ratio[pair]),
            "score_transfer_relative_residual": float(exact_score_residual),
            "centered_logit_gap": margin,
            "remainder_bound": locked_bound,
            "heldout_transfer_without_enlargement": no_enlargement,
            "qualified": qualified,
        })
    plga_residual = max(plga_residuals)

    tolerance = ARITHMETIC["relative_identity_tolerance"]
    fp_forcing_scale = max(
        float(np.max(np.abs(forcing))),
        float(np.max(np.abs(recorded_roundoff))), 1e-300)
    fp_forcing_excess = float(np.max(recorded_roundoff - forcing)) / fp_forcing_scale
    checks = {
        "fixed_probe_registry_at_every_node": fixed_registry,
        "fixed_probe_chunks_at_every_node": fixed_chunks,
        "actual_minibatch_digest_separate": update_probe_separation,
        "graph_source_coordinate_digests_replay": source_coordinate_digest_replay,
        "construction_symmetric_8nn_mst_replay": construction_algorithm_replay,
        "graph_edges_unique_and_ordered": graph_unique_ordered_edges,
        "graph_weights_uniform_one_over_edge_count": graph_uniform_weights,
        "graph_connectedness_replayed": True,
        "single_connected_union_graph": bool(
            contexts == 1 and graph["vertex_count"] == expected_offsets[-1]),
        "complete_ordered_context_head_blocks": bool(
            np.array_equal(block_offsets, expected_offsets)
            and np.array_equal(block_heads, expected_heads)),
        "effective_resistance_replayed": resistance_replay_residual <= tolerance,
        "normalized_edge_contrast_cycle_replay": edge_cycle_residual <= tolerance,
        "q_replayed_from_normalized_edge_contrasts": q_replay_residual <= tolerance,
        "linear_shape_metric_replayed_from_rows": (
            linear_q_replay_residual <= tolerance),
        "quadratic_shape_metric_replayed_from_rows": (
            quadratic_q_replay_residual <= tolerance),
        "vertex_diameter_replayed_from_rows": diameter_replay_residual <= tolerance,
        "producer_energy_replay": energy_residual <= tolerance,
        "exact_shape_secant": max(shape_residuals) <= tolerance,
        "native_moment_inputs_promoted_to_float64": True,
        "shadow_first_moment_replayed": first_shadow_residual <= tolerance,
        "shadow_second_moment_replayed": second_shadow_residual <= tolerance,
        "shadow_preconditioner_replayed": (
            preconditioner_shadow_residual <= tolerance),
        "shadow_adaptive_direction_replayed": adaptive_shadow_residual <= tolerance,
        "native_float32_successor_shadow_replay": max(ledger_residuals) <= tolerance,
        "measured_floating_point_charge": max(roundoff_residuals) <= tolerance,
        "floating_point_charge_nonnegative": bool(np.all(recorded_roundoff >= 0.0)),
        "floating_point_charge_in_affine_forcing": fp_forcing_excess <= tolerance,
        "bridge_sector_vector_sum": max(bridge_residuals) <= tolerance,
        "source_microbatch_weights_replayed": source_weight_residual <= tolerance,
        "source_microbatch_sector_means_replayed": (
            source_sector_mean_residual <= tolerance),
        "source_sector_dispersion_reported": bool(
            np.isfinite(source_sector_dispersion).all()),
        "aggregate_adaptive_gain_diagnostic": (
            max(aggregate_gain_residuals) <= tolerance),
        "optimizer_decay_mask_verified": bool(
            decay_residual <= tolerance
            and np.all(decay_mask == checkpoint_decay_mask[None, :])),
        "checkpoint_decay_mask_digest_replayed": True,
        "complete_checkpoint_manifest_bound": bool(
            len(_scalar(raw, "checkpoint_complete_state_sha256", str)) == 64
            and len(_scalar(raw, "checkpoint_branch_state_sha256", str)) == 64
            and len(_scalar(
                raw, "checkpoint_measurement_registry_sha256", str)) == 64),
        "resolved_native_device_recorded": bool(
            _scalar(raw, "resource_resolved_device", str)),
        "deciding_signed_credit_absolute_charge_rule": True,
        "ordered_affine_enclosure": affine_enclosure_residual <= tolerance,
        "effective_resistance_enclosure": graph_excess <= tolerance,
        "gate_first_and_shapley_efficiency": max(attribution_residuals) <= 1e-10,
        "complete_union_subset_plga_registry": bool(
            len(plga_rows) == block_count),
        "positive_base_real_exponent_plga": bool(
            plga_positive_bases and plga_residual <= tolerance),
        "energy_to_stacked_generator_bridge": plga_energy_bridge,
        "exact_three_factor_score_transfer": plga_score_transfer,
        "construction_plga_envelopes_not_enlarged": plga_lock_transfer,
    }
    if lock_transfer is not None:
        checks.update({
            "construction_probe_binding_unchanged": (
                lock_transfer["probe_registry_unchanged"]),
            "construction_graph_rule_unchanged": (
                lock_transfer["graph_rule_unchanged"]),
            "construction_graph_mapping_unchanged": (
                lock_transfer["graph_identity_mapping_unchanged"]),
            "construction_union_graph_digest_unchanged": (
                lock_transfer["union_graph_sha256_unchanged"]),
            "frozen_affine_arrays_unchanged": (
                lock_transfer["factor_relative_residual"] <= tolerance
                and lock_transfer["forcing_relative_residual"] <= tolerance),
            "construction_source_envelopes_not_enlarged": max(
                lock_transfer["resistance_relative_excess"],
                lock_transfer["roundoff_relative_excess"],
                lock_transfer["adaptive_relative_excess"],
                lock_transfer["sector_relative_excess"],
            ) <= tolerance,
            "frozen_energy_and_diameter_paths_cover": max(
                lock_transfer["energy_coverage_relative_excess"],
                lock_transfer["diameter_coverage_relative_excess"],
            ) <= tolerance,
        })
    source_energy_positive = bool(np.all(np.isfinite(energy[0])) and np.all(
        energy[0] > 0.0))
    checks["finite_positive_source_energy"] = source_energy_positive
    margin_numerator_sum = np.sum(margins, axis=0)
    relative_margin_sum = np.full(contexts, -math.inf, dtype=np.float64)
    if source_energy_positive:
        relative_margin_sum = margin_numerator_sum / energy[0]
    local_relative_margin_floor = (
        TRANSFER["numerical_margin_floor_multiplier"] * transitions
        * (ARITHMETIC["relative_identity_tolerance"]
           + ARITHMETIC["absolute_identity_tolerance"] / energy[0]))
    deciding_relative_margin_floor = local_relative_margin_floor
    if lock is not None:
        deciding_relative_margin_floor = np.full(
            contexts, float(lock["numerical_nonvacuity"][
                "relative_margin_floor"]))
    margin_floor_scale = np.maximum(
        np.maximum(local_relative_margin_floor, deciding_relative_margin_floor),
        1e-300)
    margin_floor_relative_excess = float(np.max(
        local_relative_margin_floor - deciding_relative_margin_floor)
        / np.max(margin_floor_scale))
    margin_floor_not_enlarged = bool(
        margin_floor_relative_excess
        <= ARITHMETIC["relative_identity_tolerance"])
    checks["construction_numerical_margin_floor_not_enlarged"] = (
        margin_floor_not_enlarged)
    if lock_transfer is not None:
        lock_transfer["margin_floor_relative_excess"] = (
            margin_floor_relative_excess)
    algebra_pass = all(checks.values())
    positive_margin = bool(source_energy_positive and margin_floor_not_enlarged
                           and np.all(
        relative_margin_sum > deciding_relative_margin_floor))
    endpoint_contraction = bool(np.all(energy[-1] < energy[0]))
    local_diameter_target = (
        TRANSFER["construction_safety_multiplier"]
        * TRANSFER["diameter_endpoint_fraction_of_construction_source"]
        * diameter[0])
    deciding_diameter_target = local_diameter_target
    deciding_ratio_max = TRANSFER["diameter_bound_to_observed_ratio_max"]
    if lock is not None:
        deciding_diameter_target = np.full(
            contexts, float(lock["diameter_nonvacuity"][
                "endpoint_usefulness_target"]))
        deciding_ratio_max = float(lock["diameter_nonvacuity"][
            "bound_to_observed_ratio_max"])
    endpoint_bound_to_observed_ratio = np.asarray([
        row["endpoint_diameter_bound"] / max(
            float(diameter[-1, row["context"]]),
            ARITHMETIC["absolute_identity_tolerance"])
        for row in block_rows])
    diameter_useful = bool(all(
        row["endpoint_diameter_bound"]
        < deciding_diameter_target[row["context"]]
        and endpoint_bound_to_observed_ratio[row["context"]]
        <= deciding_ratio_max
        for row in block_rows))
    affine_nonvacuity = bool(diameter_useful and all(
        row["endpoint_bound"] < row["source_energy"]
        and row["endpoint_diameter_bound"] < row["source_diameter_bound"]
        for row in block_rows
    ))
    qualified = (
        algebra_pass and positive_margin and endpoint_contraction
        and affine_nonvacuity
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source": {"path": str(path), "sha256": sha256_path(path)},
        "stage": stage,
        "job_id": _scalar(raw, "job_id", str),
        "record_key": f"{stage}:{_scalar(raw, 'job_id', str)}",
        "seed": _scalar(raw, "seed", int),
        "role": role,
        "layer": _scalar(raw, "layer", int),
        "anchor": _scalar(raw, "anchor", int),
        "block_length": _scalar(raw, "block_length", int),
        "arithmetic": {
            "native_update": "float32",
            "deciding_ledger": "float64-shadow",
            "float64_continuation_is_control_only": True,
        },
        "construction_lock_transfer": lock_transfer,
        "probe_binding": {
            "registry_sha256": registry_digest,
            "chunk_ids_sha256": chunk_digest,
            "same_at_every_node": fixed_registry and fixed_chunks,
            "update_batches_are_not_observable_rows": True,
        },
        "checks": checks,
        "metrics": {
            "graph_edge_cycle_relative_residual": edge_cycle_residual,
            "graph_q_relative_residual": q_replay_residual,
            "graph_linear_q_relative_residual": linear_q_replay_residual,
            "graph_quadratic_q_relative_residual": quadratic_q_replay_residual,
            "graph_resistance_relative_residual": resistance_replay_residual,
            "graph_diameter_relative_residual": diameter_replay_residual,
            "energy_replay_relative_residual": energy_residual,
            "shape_relative_residual_max": max(shape_residuals),
            "shadow_first_moment_relative_residual": first_shadow_residual,
            "shadow_second_moment_relative_residual": second_shadow_residual,
            "shadow_preconditioner_relative_residual": (
                preconditioner_shadow_residual),
            "shadow_adaptive_direction_relative_residual": (
                adaptive_shadow_residual),
            "native_ledger_relative_residual_max": max(ledger_residuals),
            "roundoff_relative_residual_max": max(roundoff_residuals),
            "bridge_sector_relative_residual_max": max(bridge_residuals),
            "source_microbatch_weight_relative_residual": source_weight_residual,
            "source_sector_mean_relative_residual": source_sector_mean_residual,
            "source_sector_dispersion_by_step_and_sector": (
                source_sector_dispersion.tolist()),
            "aggregate_gain_relative_residual_max": max(aggregate_gain_residuals),
            "optimizer_decay_mask_relative_residual": decay_residual,
            "certified_adaptive_gain_sum": float(np.sum(certified_adaptive)),
            "aggregate_adaptive_gain_sum_diagnostic": float(
                np.sum(aggregate_adaptive)),
            "affine_enclosure_relative_excess": affine_enclosure_residual,
            "graph_enclosure_relative_excess": graph_excess,
            "plga_relative_residual_max": plga_residual,
            "plga_complete_pair_count": len(plga_rows),
            "plga_qualified_pair_count": plga_qualified_count,
            "plga_pairs": plga_rows,
            "margin_numerator_sum_by_context": margin_numerator_sum.tolist(),
            "relative_margin_sum_by_context": relative_margin_sum.tolist(),
            "minimum_relative_margin_sum": float(np.min(relative_margin_sum)),
            "deciding_relative_margin_floor": float(np.max(
                deciding_relative_margin_floor)),
            "local_numerical_relative_margin_floor": float(np.max(
                local_relative_margin_floor)),
            "numerical_margin_floor_relative_excess": (
                margin_floor_relative_excess),
            "diameter_endpoint_usefulness_target": float(np.min(
                deciding_diameter_target)),
            "diameter_endpoint_bound_to_observed_ratio": (
                endpoint_bound_to_observed_ratio.tolist()),
            "diameter_bound_to_observed_ratio_max": deciding_ratio_max,
            "roundoff_signed_sum": float(np.sum(signed_roundoff)),
            "roundoff_charge_sum": float(np.sum(recorded_roundoff)),
            "floating_point_forcing_relative_excess": fp_forcing_excess,
            "sector_gain_sums": {
                name: float(np.sum(values)) for name, values in sector_gains.items()
            },
            "block_replay": block_rows,
        },
        "algebra_status": "PASS" if algebra_pass else "FAIL",
        "positive_margin": positive_margin,
        "endpoint_contraction": endpoint_contraction,
        "affine_bound_nonvacuity": affine_nonvacuity,
        "diameter_bound_useful": diameter_useful,
        "decision": "QUALIFIED" if qualified else "NOT_QUALIFIED",
    }
    report["analysis_sha256"] = digest_object(report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--lock", default="")
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--require-positive", action="store_true")
    arguments = parser.parse_args()
    report = analyze(arguments.input, arguments.lock or None)
    write_json_atomic(arguments.output, report)
    if arguments.require_pass and report["algebra_status"] != "PASS":
        raise SystemExit("observable-margin algebraic replay failed")
    if arguments.require_positive and report["decision"] != "QUALIFIED":
        raise SystemExit("observable-margin positive endpoint did not qualify")


if __name__ == "__main__":
    main()
