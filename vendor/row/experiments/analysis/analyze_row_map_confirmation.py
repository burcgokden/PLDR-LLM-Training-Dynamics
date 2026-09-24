#!/usr/bin/env python3
"""Independent analysis for the archive-bound row-map confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
CONFIRM = EXPERIMENTS / "confirm"
sys.path.insert(0, str(CONFIRM))

from gate_shape import adamw_gate_block, normalized_shape, plga_secant_chain  # noqa: E402
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    sha256_path,
    write_json_atomic,
)
from row_map_confirmation_specs import FIXED_CONSTANTS, RUNS  # noqa: E402
from row_map_dynamics import (  # noqa: E402
    decay_loss_log_split,
    dimensionless_state_scale,
    gate_duhamel,
    graph_contrast,
    log_route_decomposition,
    relative_matrix_residual,
    scaled_chronological_diagnostics,
)


SCHEMA = "pldr-row-map-analysis-v1"
TOLERANCE = float(FIXED_CONSTANTS["relative_identity_tolerance"])
ZERO = float(FIXED_CONSTANTS["absolute_zero_tolerance"])
MATERIAL = float(FIXED_CONSTANTS["material_energy_factor"])
TRANSFER = float(FIXED_CONSTANTS["construction_to_validation_factor"])
DOMINANCE = float(FIXED_CONSTANTS["route_dominance_fraction"])


def _text(value):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected one scalar text array")
    return str(array.item())


def _number(value):
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("expected one scalar numerical array")
    result = float(array.item())
    if not math.isfinite(result):
        raise ValueError("a deciding scalar is nonfinite")
    return result


def _relative_scalar(left, right):
    left = float(left)
    right = float(right)
    scale = max(abs(left), abs(right))
    return 0.0 if scale <= ZERO else abs(left - right) / scale


def _relative_array(left, right):
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape:
        raise ValueError("array residual operands must have one shape")
    if not (np.isfinite(first).all() and np.isfinite(second).all()):
        raise ValueError("array residual operands must be finite")
    scale = max(float(np.linalg.norm(first)), float(np.linalg.norm(second)))
    return 0.0 if scale <= ZERO else float(np.linalg.norm(first - second)) / scale


def _record(kind, inputs, checks, results):
    value = {
        "schema_version": SCHEMA,
        "kind": kind,
        "inputs": [
            {"path": Path(path).name, "sha256": sha256_path(path)}
            for path in inputs
        ],
        "checks": checks,
        "results": results,
    }
    value["all_checks_pass"] = all(bool(item) for item in checks.values())
    value["decision"] = (
        "CONFIRMED" if value["all_checks_pass"] else "NOT_CONFIRMED")
    value["analysis_sha256"] = digest_object(value)
    return value


def _route_label(energy_start, energy_end, fixed_shape_energy):
    gate_work = fixed_shape_energy - energy_start
    shape_work = energy_end - fixed_shape_energy
    residual = energy_end - energy_start - gate_work - shape_work
    if min(energy_start, energy_end) <= ZERO:
        return "zero-route", gate_work, shape_work, residual
    material = math.log(energy_start / energy_end) >= math.log(MATERIAL)
    if not material:
        return "unclassified", gate_work, shape_work, residual
    gate_decrease = max(-gate_work, 0.0)
    shape_decrease = max(-shape_work, 0.0)
    total_decrease = gate_decrease + shape_decrease
    if total_decrease <= ZERO:
        label = "mixed"
    elif gate_decrease / total_decrease >= DOMINANCE:
        label = "gate-led"
    elif shape_decrease / total_decrease >= DOMINANCE:
        label = "shape-led"
    else:
        label = "mixed"
    return label, gate_work, shape_work, residual


def _coordinate_route_label(gate_log, shape_log, total_log):
    if total_log < math.log(MATERIAL):
        return "unclassified"
    gate_share = gate_log / total_log
    shape_share = shape_log / total_log
    if gate_share >= DOMINANCE:
        return "gate-led"
    if shape_share >= DOMINANCE:
        return "shape-led"
    return "mixed"


GEOMETRY_REQUIRED = (
    "schema_version", "stage", "checkpoint_sha256", "target_checkpoint_sha256",
    "run_name", "seed",
    "role", "step", "layer", "gamma", "beta", "layernorm_epsilon",
    "final_preactivation", "normalized_shape", "row_output", "graph_edges",
    "shape_directional", "target_normalized_shape", "shape_remainder",
    "graph_weights", "q", "energy", "vertex_diameter",
    "graph_diameter_bound", "covered_domain_bound",
    "energy_identity_residual", "factorization_residual",
)


def analyze_geometry(paths):
    nodes = []
    exact_checks = []
    for path in paths:
        raw = load_npz(path, required=GEOMETRY_REQUIRED)
        if _text(raw["stage"]) != "G":
            raise ValueError("geometry analyzer received a non-G archive")
        shape = normalized_shape(
            raw["final_preactivation"],
            epsilon=_number(raw["layernorm_epsilon"]))
        shape_residual = relative_matrix_residual(
            shape, raw["normalized_shape"])
        output = raw["beta"] + raw["gamma"] * shape
        factor_residual = relative_matrix_residual(output, raw["row_output"])
        graph = graph_contrast(
            raw["gamma"], shape, raw["graph_edges"], raw["graph_weights"])
        q_residual = relative_matrix_residual(
            graph["q"][None], raw["q"][None])
        energy_residual = _relative_scalar(
            graph["energy"], _number(raw["energy"]))
        diameter_residual = _relative_scalar(
            graph["diameter_bound"], _number(raw["graph_diameter_bound"]))
        cover = _number(raw["covered_domain_bound"])
        target_digest = _text(raw["target_checkpoint_sha256"])
        shape_jvp_residual = 0.0
        directional_q = None
        q_remainder = None
        target_q = None
        if target_digest:
            shape_jvp_residual = relative_matrix_residual(
                raw["target_normalized_shape"] - shape,
                raw["shape_directional"] + raw["shape_remainder"])
            edges = raw["graph_edges"]
            weights = raw["graph_weights"]
            difference = shape[edges[:, 0]] - shape[edges[:, 1]]
            directional_difference = (
                raw["shape_directional"][edges[:, 0]]
                - raw["shape_directional"][edges[:, 1]])
            target_difference = (
                raw["target_normalized_shape"][edges[:, 0]]
                - raw["target_normalized_shape"][edges[:, 1]])
            directional_q = 2.0 * np.sum(
                weights[:, None] * difference * directional_difference,
                axis=0)
            target_q = np.sum(
                weights[:, None] * target_difference * target_difference,
                axis=0)
            q_remainder = target_q - graph["q"] - directional_q
        checks = {
            "shape_reconstruction": shape_residual <= TOLERANCE,
            "factorization": factor_residual <= TOLERANCE,
            "q_reconstruction": q_residual <= TOLERANCE,
            "energy_reconstruction": energy_residual <= TOLERANCE,
            "diameter_reconstruction": diameter_residual <= TOLERANCE,
            "graph_encloses_vertices": (
                graph["vertex_diameter"] <= graph["diameter_bound"] + ZERO),
            "cover_encloses_vertices": graph["vertex_diameter"] <= cover + ZERO,
            "shape_jvp_plus_remainder": shape_jvp_residual <= TOLERANCE,
        }
        exact_checks.extend(checks.values())
        nodes.append({
            "path": str(path),
            "raw": raw,
            "run_name": _text(raw["run_name"]),
            "seed": int(_number(raw["seed"])),
            "role": _text(raw["role"]),
            "step": int(_number(raw["step"])),
            "layer": int(_number(raw["layer"])),
            "shape": shape,
            "target_digest": target_digest,
            "target_shape": raw["target_normalized_shape"],
            "directional_q": directional_q,
            "target_q": target_q,
            "q_remainder": q_remainder,
            "graph": graph,
            "checks": checks,
            "max_reconstruction_residual": max(
                shape_residual, factor_residual, q_residual,
                energy_residual, diameter_residual, shape_jvp_residual),
        })
    groups = {}
    for node in nodes:
        key = (node["run_name"], node["seed"], node["role"], node["layer"])
        groups.setdefault(key, []).append(node)
    intervals = []
    route_residuals = []
    material_coordinate_labels = []
    validation_collapse = {}
    for (run_name, seed, role, layer), values in sorted(groups.items()):
        values.sort(key=lambda item: item["step"])
        if len({value["step"] for value in values}) != len(values):
            raise ValueError("geometry group repeats a checkpoint step")
        histories = log_route_decomposition(
            np.stack([value["raw"]["gamma"] for value in values]),
            np.stack([value["raw"]["q"] for value in values]),
            zero_tolerance=ZERO)
        for source, target in zip(values[:-1], values[1:]):
            linked = (
                source["target_digest"] == _text(
                    target["raw"]["checkpoint_sha256"])
                and relative_matrix_residual(
                    source["target_shape"], target["shape"]) <= TOLERANCE)
            exact_checks.append(linked)
        positive = histories["positive_step_mask"]
        if np.any(positive):
            route_residuals.append(float(np.max(np.abs(
                histories["log_identity_residual"][positive]))))
        for index, (source, target) in enumerate(zip(values[:-1], values[1:])):
            energy_start = source["graph"]["energy"]
            energy_end = target["graph"]["energy"]
            fixed_shape = float(np.sum(
                source["raw"]["q"] * target["raw"]["gamma"] ** 2))
            label, gate_work, shape_work, additive_residual = _route_label(
                energy_start, energy_end, fixed_shape)
            ratio = (
                energy_end / energy_start if energy_start > ZERO
                else (0.0 if energy_end <= ZERO else None))
            collapse = ratio is not None and ratio <= 1.0 / MATERIAL
            if role == "validation":
                validation_collapse.setdefault(seed, False)
                validation_collapse[seed] |= collapse
            coordinate_routes = []
            for coordinate in range(histories["energy"].shape[1]):
                if positive[index, coordinate]:
                    gate_log = float(
                        histories["gate_log_decrement"][index, coordinate])
                    shape_log = float(
                        histories["shape_log_decrement"][index, coordinate])
                    total_log = float(
                        histories["total_log_decrement"][index, coordinate])
                    route_label = _coordinate_route_label(
                        gate_log, shape_log, total_log)
                    material = total_log >= math.log(MATERIAL)
                    if material:
                        material_coordinate_labels.append(route_label)
                    coordinate_routes.append({
                        "coordinate": coordinate,
                        "status": "positive",
                        "gate_log_decrement": gate_log,
                        "shape_log_decrement": shape_log,
                        "total_log_decrement": total_log,
                        "material_collapse": material,
                        "route_label": route_label,
                    })
                elif histories["energy"][index + 1, coordinate] <= ZERO:
                    coordinate_routes.append({
                        "coordinate": coordinate,
                        "status": "zero-at-target",
                        "gate_log_decrement": None,
                        "shape_log_decrement": None,
                        "total_log_decrement": None,
                        "material_collapse": True,
                        "route_label": "zero-route",
                    })
                else:
                    coordinate_routes.append({
                        "coordinate": coordinate,
                        "status": "zero-at-source-or-thresholded",
                        "gate_log_decrement": None,
                        "shape_log_decrement": None,
                        "total_log_decrement": None,
                        "material_collapse": False,
                        "route_label": "unclassified",
                    })
            intervals.append({
                "run_name": run_name,
                "seed": seed,
                "role": role,
                "layer": layer,
                "source_step": source["step"],
                "target_step": target["step"],
                "energy_start": energy_start,
                "energy_end": energy_end,
                "energy_ratio": ratio,
                "collapse_cell": collapse,
                "gate_work": gate_work,
                "shape_work": shape_work,
                "shape_directional_q_norm": (
                    float(np.linalg.norm(source["directional_q"]))
                    if source["directional_q"] is not None else None),
                "shape_q_remainder_norm": (
                    float(np.linalg.norm(source["q_remainder"]))
                    if source["q_remainder"] is not None else None),
                "route_label": label,
                "additive_identity_residual": additive_residual,
                "positive_coordinate_routes": int(np.sum(positive[index])),
                "zero_coordinates_at_target": int(np.sum(
                    histories["energy"][index + 1] <= ZERO)),
                "coordinate_routes": coordinate_routes,
            })
    max_route_residual = max(route_residuals, default=0.0)
    route_exact = max_route_residual <= TOLERANCE
    exact_checks.append(route_exact)
    complete_validation = bool(validation_collapse) and all(
        validation_collapse.values())
    checks = {
        "all_node_identities": all(exact_checks),
        "coordinate_log_routes": route_exact,
        "all_material_coordinate_routes_classified": all(
            label in {"gate-led", "shape-led", "mixed"}
            for label in material_coordinate_labels),
        "all_graph_bounds": all(
            node["checks"]["graph_encloses_vertices"] for node in nodes),
        "all_covered_domain_bounds": all(
            node["checks"]["cover_encloses_vertices"] for node in nodes),
        "each_present_validation_seed_has_collapse_cell": complete_validation,
    }
    results = {
        "node_count": len(nodes),
        "group_count": len(groups),
        "interval_count": len(intervals),
        "maximum_node_reconstruction_residual": max(
            (node["max_reconstruction_residual"] for node in nodes), default=0.0),
        "maximum_coordinate_log_residual": max_route_residual,
        "material_positive_coordinate_count": len(material_coordinate_labels),
        "material_coordinate_route_counts": {
            label: material_coordinate_labels.count(label)
            for label in ("gate-led", "shape-led", "mixed")
        },
        "validation_collapse_by_seed": validation_collapse,
        "nodes": [{
            "run_name": node["run_name"], "seed": node["seed"],
            "role": node["role"], "step": node["step"],
            "layer": node["layer"], "energy": node["graph"]["energy"],
            "vertex_diameter": node["graph"]["vertex_diameter"],
            "graph_diameter_bound": node["graph"]["diameter_bound"],
            "checks": node["checks"],
        } for node in nodes],
        "intervals": intervals,
    }
    return _record("geometry", paths, checks, results)


BLOCK_REQUIRED = (
    "stage", "checkpoint_sha256", "run_name", "seed", "device", "step", "layer",
    "gamma", "first_moment", "second_moment", "raw_gradient",
    "true_hessian", "learning_rate", "beta1", "beta2", "adam_epsilon",
    "weight_decay", "clip_value", "optimizer_step_after",
    "analytic_operator", "autodiff_operator", "source_state_scale",
    "target_state_scale", "clipping_boundary_distance",
)


QUALIFICATION_REQUIRED = (
    "loss_value", "gradient_at_gate", "fisher", "sector_true_hessian",
    "signed_second_jet", "force_at_zero", "loss_terms",
    "gradient_at_gate_terms", "fisher_terms", "true_hessian_terms",
    "force_at_zero_terms", "source_count_terms",
)


CONTINUATION_REQUIRED = (
    "schema_version", "arm", "run_name", "seed", "layer", "local_step",
    "gamma_before", "gamma_optimizer_after", "gamma_after",
    "adaptive_direction", "learning_rate", "weight_decay",
    "q_before", "q_after", "energy_before", "energy_after", "gate_work",
    "shape_work", "loss",
)


DYNAMICS_REQUIRED = CONTINUATION_REQUIRED + (
    "capture_blocks", "q_directional", "q_remainder",
    "shape_jvp_source_residual", "block_first_moment",
    "block_second_moment", "block_raw_gradient", "block_true_hessian",
    "block_operator", "block_source_state_scale",
    "block_target_state_scale", "block_beta1", "block_beta2",
    "block_adam_epsilon", "block_clip_value",
    "block_optimizer_step_after", "block_gradient_replay_residual",
    "block_clipping_boundary_distance",
)


def analyze_dynamics(paths):
    ownership = {row["seed"]: row["ownership"] for row in RUNS}
    reports = []
    trajectory_gradient_tolerance = float(
        FIXED_CONSTANTS["trajectory_gradient_replay_tolerance"])
    for path in paths:
        raw = load_npz(path, required=DYNAMICS_REQUIRED)
        if _text(raw["schema_version"]) != (
                "pldr-row-map-intervention-observables-v2"):
            raise ValueError("unknown dynamics observable schema")
        if _text(raw["arm"]) != "baseline_continue":
            raise ValueError("dynamics analysis requires baseline continuations")
        if not bool(np.asarray(raw["capture_blocks"]).item()):
            raise ValueError("dynamics archive omits chronological blocks")
        node_count = len(raw["local_step"])
        rebuilt_q = (
            raw["q_before"] + raw["q_directional"] + raw["q_remainder"])
        q_residual = relative_matrix_residual(rebuilt_q, raw["q_after"])
        rebuilt_before = np.sum(
            raw["q_before"] * raw["gamma_before"] ** 2, axis=1)
        fixed_shape = np.sum(
            raw["q_before"] * raw["gamma_after"] ** 2, axis=1)
        rebuilt_after = np.sum(
            raw["q_after"] * raw["gamma_after"] ** 2, axis=1)
        work_residual = max(
            _relative_array(rebuilt_before, raw["energy_before"]),
            _relative_array(rebuilt_after, raw["energy_after"]),
            _relative_array(
                fixed_shape - rebuilt_before, raw["gate_work"]),
            _relative_array(
                rebuilt_after - fixed_shape, raw["shape_work"]),
        )

        rates = np.asarray(raw["learning_rate"], dtype=np.float64)
        decays = np.asarray(raw["weight_decay"], dtype=np.float64)
        if not np.all(decays == decays[0]):
            raise ValueError("one continuation changed its decay coefficient")
        rebuilt_optimizer_gate = (
            (1.0 - rates[:, None] * decays[:, None])
            * raw["gamma_before"]
            - rates[:, None] * raw["adaptive_direction"])
        gate_step_residual = _relative_array(
            rebuilt_optimizer_gate, raw["gamma_optimizer_after"])
        baseline_intervention_residual = _relative_array(
            raw["gamma_optimizer_after"], raw["gamma_after"])
        chain_residual = _relative_array(
            raw["gamma_before"][1:], raw["gamma_after"][:-1])
        duhamel = gate_duhamel(
            raw["gamma_before"][0], raw["adaptive_direction"],
            rates, float(decays[0]))
        duhamel_residual = max(
            _relative_array(duhamel["history"][1:], raw["gamma_after"]),
            _relative_array(duhamel["reconstructed"], raw["gamma_after"][-1]),
        )
        log_residuals = []
        positive_log_coordinates = 0
        for index in range(node_count):
            split = decay_loss_log_split(
                raw["gamma_before"][index],
                raw["adaptive_direction"][index],
                rates[index], decays[index])
            valid = split["valid_mask"]
            positive_log_coordinates += int(np.sum(valid))
            if np.any(valid):
                log_residuals.append(float(np.max(np.abs(
                    split["identity_residual"][valid]))))
        gate_log_residual = max(log_residuals, default=0.0)

        operators = []
        block_replay_residuals = []
        scale_formula_residuals = []
        predicted_gate = []
        predicted_first = []
        predicted_second = []
        for index in range(node_count):
            block_result = adamw_gate_block(
                raw["block_true_hessian"][index],
                raw["gamma_before"][index],
                raw["block_first_moment"][index],
                raw["block_second_moment"][index],
                raw["block_raw_gradient"][index],
                learning_rate=rates[index],
                beta1=float(raw["block_beta1"][index]),
                beta2=float(raw["block_beta2"][index]),
                epsilon=float(raw["block_adam_epsilon"][index]),
                optimizer_step=int(raw["block_optimizer_step_after"][index]),
                weight_decay=decays[index],
                clip_value=float(raw["block_clip_value"][index]),
                boundary_tolerance=float(
                    FIXED_CONSTANTS["clipping_boundary_tolerance"]),
            )
            block = block_result["operator"]
            operators.append(block)
            block_replay_residuals.append(relative_matrix_residual(
                block, raw["block_operator"][index]))
            predicted_gate.append(block_result["gamma"])
            predicted_first.append(block_result["first_moment"])
            predicted_second.append(block_result["second_moment"])
            step_after = int(raw["block_optimizer_step_after"][index])
            beta2 = float(raw["block_beta2"][index])
            epsilon = float(raw["block_adam_epsilon"][index])
            source_second_hat = (
                raw["block_second_moment"][index]
                / (1.0 - beta2 ** (step_after - 1)))
            target_second_hat = (
                block_result["second_moment"]
                / (1.0 - beta2 ** step_after))
            scale_formula_residuals.extend([
                _relative_array(
                    dimensionless_state_scale(
                        source_second_hat, epsilon=epsilon),
                    raw["block_source_state_scale"][index]),
                _relative_array(
                    dimensionless_state_scale(
                        target_second_hat, epsilon=epsilon),
                    raw["block_target_state_scale"][index]),
            ])
        predicted_gate = np.asarray(predicted_gate)
        predicted_first = np.asarray(predicted_first)
        predicted_second = np.asarray(predicted_second)
        block_gate_successor_residual = _relative_array(
            predicted_gate, raw["gamma_optimizer_after"])
        block_moment_chain_residual = max(
            _relative_array(
                predicted_first[:-1], raw["block_first_moment"][1:]),
            _relative_array(
                predicted_second[:-1], raw["block_second_moment"][1:]),
        )
        scale_adjacency_residual = max(
            (_relative_array(
                raw["block_target_state_scale"][index],
                raw["block_source_state_scale"][index + 1])
             for index in range(node_count - 1)),
            default=0.0)
        state_scales = [
            raw["block_source_state_scale"][0],
            *list(raw["block_target_state_scale"]),
        ]
        chronological = scaled_chronological_diagnostics(
            operators, state_scales)

        seed = int(_number(raw["seed"]))
        if seed not in ownership:
            raise ValueError("dynamics archive has an unregistered seed")
        remainder_norm = np.linalg.norm(raw["q_remainder"], axis=1)
        reports.append({
            "path": str(path),
            "run_name": _text(raw["run_name"]),
            "seed": seed,
            "ownership": ownership[seed],
            "layer": int(_number(raw["layer"])),
            "node_count": node_count,
            "q_identity_residual": q_residual,
            "work_identity_residual": work_residual,
            "gate_step_residual": gate_step_residual,
            "gate_duhamel_residual": duhamel_residual,
            "gate_log_split_residual": gate_log_residual,
            "positive_gate_log_coordinates": positive_log_coordinates,
            "baseline_intervention_residual": baseline_intervention_residual,
            "gate_chain_residual": chain_residual,
            "maximum_shape_jvp_source_residual": float(np.max(
                raw["shape_jvp_source_residual"])),
            "maximum_block_reconstruction_residual": max(
                block_replay_residuals, default=0.0),
            "maximum_block_gradient_replay_residual": float(np.max(
                raw["block_gradient_replay_residual"])),
            "block_gate_successor_residual": block_gate_successor_residual,
            "block_moment_chain_residual": block_moment_chain_residual,
            "maximum_scale_formula_residual": max(
                scale_formula_residuals, default=0.0),
            "minimum_clipping_boundary_distance": float(np.min(
                raw["block_clipping_boundary_distance"])),
            "scale_adjacency_residual": scale_adjacency_residual,
            "chronological_spectral_radius": chronological["spectral_radius"],
            "chronological_scaled_norm": chronological["scaled_norm"],
            "endpoint_scaling_condition": chronological[
                "endpoint_scaling_condition"],
            "maximum_prefix_spectral_radius": float(np.max(
                chronological["prefix_spectral_radii"])),
            "maximum_prefix_scaled_norm": float(np.max(
                chronological["prefix_scaled_norms"])),
            "raw_chronological_norm_diagnostic": float(np.linalg.norm(
                chronological["product"], ord=2)),
            "remainder_norms": remainder_norm,
            "total_gate_work": float(np.sum(raw["gate_work"])),
            "total_shape_work": float(np.sum(raw["shape_work"])),
        })
    transfer_rows = []
    for layer in range(3):
        source = [
            row for row in reports
            if row["layer"] == layer and row["ownership"] == "construction"]
        validation = [
            row for row in reports
            if row["layer"] == layer and row["ownership"] == "validation"]
        if len(source) != 1 or not validation:
            continue
        bound = float(np.max(source[0]["remainder_norms"]))
        validation_maximum = max(
            float(np.max(row["remainder_norms"])) for row in validation)
        transfer_rows.append({
            "layer": layer,
            "construction_remainder_bound": bound,
            "validation_maximum": validation_maximum,
            "factor_2_enclosure": validation_maximum <= TRANSFER * bound + ZERO,
        })
    checks = {
        "nonempty": bool(reports),
        "all_windows_have_sixteen_updates": all(
            row["node_count"] == FIXED_CONSTANTS["continuation_updates"]
            for row in reports),
        "all_shape_jvp_sources_replay": all(
            row["maximum_shape_jvp_source_residual"] <= TOLERANCE
            for row in reports),
        "all_shape_jvp_remainders_replay": all(
            row["q_identity_residual"] <= TOLERANCE for row in reports),
        "all_gate_shape_work_replays": all(
            row["work_identity_residual"] <= TOLERANCE for row in reports),
        "all_decay_plus_loss_memory_paths_replay": all(
            max(
                row["gate_step_residual"], row["gate_duhamel_residual"],
                row["gate_log_split_residual"], row["gate_chain_residual"],
                row["baseline_intervention_residual"],
            ) <= TOLERANCE
            and row["positive_gate_log_coordinates"] > 0
            for row in reports),
        "all_chronological_blocks_reconstruct": all(
            row["maximum_block_reconstruction_residual"] <= TOLERANCE
            and row["maximum_block_gradient_replay_residual"]
            <= trajectory_gradient_tolerance
            and row["block_gate_successor_residual"] <= TOLERANCE
            and row["block_moment_chain_residual"] <= TOLERANCE
            and row["maximum_scale_formula_residual"] <= TOLERANCE
            and row["minimum_clipping_boundary_distance"]
            > FIXED_CONSTANTS["clipping_boundary_tolerance"]
            and row["scale_adjacency_residual"] <= TOLERANCE
            for row in reports),
        "all_layers_have_transfer_tests": len(transfer_rows) == 3,
        "fixed_factor_2_remainder_transfer": (
            len(transfer_rows) == 3
            and all(row["factor_2_enclosure"] for row in transfer_rows)),
    }
    return _record("dynamics", paths, checks, {
        "windows": [{
            key: value for key, value in row.items()
            if key != "remainder_norms" and key != "path"
        } for row in reports],
        "transfer": transfer_rows,
    })

LOSS_REQUIRED = (
    "checkpoint_sha256", "run_name", "seed", "role", "step", "layer",
    "loss_value", "gradient_at_gate", "fisher", "true_hessian",
    "signed_second_jet", "force_at_zero", "loss_terms",
    "gradient_at_gate_terms", "fisher_terms", "true_hessian_terms",
    "force_at_zero_terms", "source_count_terms",
)


def analyze_loss(paths):
    reports = []
    for path in paths:
        raw = load_npz(path, required=LOSS_REQUIRED)
        reconstructed = raw["fisher"] + raw["signed_second_jet"]
        split_residual = relative_matrix_residual(
            reconstructed, raw["true_hessian"])
        fisher_eigenvalues = np.linalg.eigvalsh(
            0.5 * (raw["fisher"] + raw["fisher"].T))
        fisher_scale = max(float(np.linalg.norm(raw["fisher"], ord=2)), 1.0)
        context_count = len(raw["true_hessian_terms"])
        context_arrays = (
            "loss_terms", "gradient_at_gate_terms", "fisher_terms",
            "force_at_zero_terms", "source_count_terms",
        )
        if not (
            all(len(raw[name]) == context_count for name in context_arrays)
            and context_count > 0
        ):
            raise ValueError("loss-sector context arrays disagree")
        counts = np.asarray(raw["source_count_terms"], dtype=np.float64)
        if np.any(counts <= 0.0):
            raise ValueError("loss-sector source counts must be positive")
        weights = counts / np.sum(counts)
        aggregate_residual = max(
            _relative_scalar(
                float(weights @ raw["loss_terms"]),
                _number(raw["loss_value"])),
            _relative_array(
                np.sum(
                    weights[:, None] * raw["gradient_at_gate_terms"], axis=0),
                raw["gradient_at_gate"]),
            relative_matrix_residual(
                np.sum(
                    weights[:, None, None] * raw["fisher_terms"], axis=0),
                raw["fisher"]),
            relative_matrix_residual(
                np.sum(
                    weights[:, None, None] * raw["true_hessian_terms"], axis=0),
                raw["true_hessian"]),
            _relative_array(
                np.sum(
                    weights[:, None] * raw["force_at_zero_terms"], axis=0),
                raw["force_at_zero"]),
        )
        context_fisher_minimum = min(float(np.linalg.eigvalsh(
            0.5 * (value + value.T))[0]) for value in raw["fisher_terms"])
        context_norms = np.asarray([
            np.linalg.norm(value, ord=2)
            for value in raw["true_hessian_terms"]], dtype=np.float64)
        force_norms = np.linalg.norm(raw["force_at_zero_terms"], axis=1)
        reports.append({
            "path": str(path),
            "run_name": _text(raw["run_name"]),
            "seed": int(_number(raw["seed"])),
            "role": _text(raw["role"]),
            "step": int(_number(raw["step"])),
            "layer": int(_number(raw["layer"])),
            "context_count": context_count,
            "split_relative_residual": split_residual,
            "weighted_aggregate_relative_residual": aggregate_residual,
            "total_nonpadding_sources": int(np.sum(counts)),
            "fisher_minimum_eigenvalue": float(fisher_eigenvalues[0]),
            "context_fisher_minimum_eigenvalue": context_fisher_minimum,
            "signed_jet_operator_norm": float(np.linalg.norm(
                raw["signed_second_jet"], ord=2)),
            "true_hessian_minimum_eigenvalue": float(np.linalg.eigvalsh(
                0.5 * (raw["true_hessian"] + raw["true_hessian"].T))[0]),
            "force_at_zero_norm": float(np.linalg.norm(raw["force_at_zero"])),
            "context_hessian_norm_mean": float(np.mean(context_norms)),
            "context_hessian_norm_dispersion": float(np.std(
                context_norms, ddof=1)) if context_count > 1 else 0.0,
            "context_force_norm_mean": float(np.mean(force_norms)),
            "context_force_norm_dispersion": float(np.std(
                force_norms, ddof=1)) if context_count > 1 else 0.0,
            "checks": {
                "fisher_plus_signed_jet": split_residual <= TOLERANCE,
                "nonpadding_weighted_aggregates": (
                    aggregate_residual <= TOLERANCE),
                "fisher_positive_semidefinite": (
                    fisher_eigenvalues[0] >= -TOLERANCE * fisher_scale
                    and context_fisher_minimum >= -TOLERANCE * fisher_scale),
                "signed_jet_is_measured": bool(np.isfinite(
                    raw["signed_second_jet"]).all()),
                "force_at_zero_is_measured": bool(np.isfinite(
                    raw["force_at_zero"]).all()),
            },
        })
    checks = {
        "nonempty": bool(reports),
        "all_true_hessian_splits_replay": all(
            row["checks"]["fisher_plus_signed_jet"] for row in reports),
        "all_fisher_terms_are_psd": all(
            row["checks"]["fisher_positive_semidefinite"] for row in reports),
        "all_nonpadding_weighted_aggregates_replay": all(
            row["checks"]["nonpadding_weighted_aggregates"]
            for row in reports),
        "all_context_statistics_present": all(
            row["context_count"] > 0 for row in reports),
    }
    return _record("loss", paths, checks, {"loss_sectors": reports})


def analyze_blocks(paths):
    reports = []
    for path in paths:
        raw = load_npz(path, required=BLOCK_REQUIRED)
        analytic = adamw_gate_block(
            raw["true_hessian"], raw["gamma"], raw["first_moment"],
            raw["second_moment"], raw["raw_gradient"],
            learning_rate=_number(raw["learning_rate"]),
            beta1=_number(raw["beta1"]), beta2=_number(raw["beta2"]),
            epsilon=_number(raw["adam_epsilon"]),
            optimizer_step=int(_number(raw["optimizer_step_after"])),
            weight_decay=_number(raw["weight_decay"]),
            clip_value=_number(raw["clip_value"]),
            boundary_tolerance=float(
                FIXED_CONSTANTS["clipping_boundary_tolerance"]),
        )["operator"]
        producer_residual = relative_matrix_residual(
            analytic, raw["analytic_operator"])
        autodiff_residual = relative_matrix_residual(
            analytic, raw["autodiff_operator"])
        diagnostics = scaled_chronological_diagnostics(
            [analytic], [raw["source_state_scale"], raw["target_state_scale"]])
        dimension = len(raw["gamma"])
        stage = _text(raw["stage"])
        if stage == "Q":
            missing = [name for name in QUALIFICATION_REQUIRED if name not in raw]
            if missing:
                raise ValueError(
                    "qualification block omits fields: " + ", ".join(missing))
            qualification_split_residual = relative_matrix_residual(
                raw["fisher"] + raw["signed_second_jet"],
                raw["sector_true_hessian"])
            counts = np.asarray(
                raw["source_count_terms"], dtype=np.float64)
            if counts.ndim != 1 or len(counts) < 1 or np.any(counts <= 0.0):
                raise ValueError("qualification source counts are invalid")
            weights = counts / np.sum(counts)
            weighted_loss = float(weights @ raw["loss_terms"])
            weighted_gradient = np.sum(
                weights[:, None] * raw["gradient_at_gate_terms"], axis=0)
            weighted_fisher = np.sum(
                weights[:, None, None] * raw["fisher_terms"], axis=0)
            weighted_true = np.sum(
                weights[:, None, None] * raw["true_hessian_terms"], axis=0)
            weighted_force = np.sum(
                weights[:, None] * raw["force_at_zero_terms"], axis=0)
            qualification_aggregate_residual = max(
                _relative_scalar(weighted_loss, _number(raw["loss_value"])),
                _relative_array(weighted_gradient, raw["gradient_at_gate"]),
                _relative_array(weighted_gradient, raw["raw_gradient"]),
                relative_matrix_residual(weighted_fisher, raw["fisher"]),
                relative_matrix_residual(
                    weighted_true, raw["sector_true_hessian"]),
                _relative_array(weighted_force, raw["force_at_zero"]),
            )
            qualification_true_residual = relative_matrix_residual(
                raw["sector_true_hessian"], raw["true_hessian"])
            qualification_finite = all(np.isfinite(raw[name]).all() for name in (
                "loss_value", "gradient_at_gate", "fisher",
                "signed_second_jet", "force_at_zero"))
            qualification_nonzero = (
                np.linalg.norm(raw["signed_second_jet"], ord=2) > ZERO)
        else:
            qualification_split_residual = None
            qualification_true_residual = None
            qualification_aggregate_residual = None
            qualification_finite = None
            qualification_nonzero = None
        reports.append({
            "path": str(path),
            "raw": raw,
            "stage": stage,
            "device": _text(raw["device"]),
            "run_name": _text(raw["run_name"]),
            "seed": int(_number(raw["seed"])),
            "step": int(_number(raw["step"])),
            "layer": int(_number(raw["layer"])),
            "operator_dimension": int(analytic.shape[0]),
            "producer_residual": producer_residual,
            "autodiff_residual": autodiff_residual,
            "spectral_radius": diagnostics["spectral_radius"],
            "scaled_norm": diagnostics["scaled_norm"],
            "endpoint_scaling_condition": diagnostics[
                "endpoint_scaling_condition"],
            "raw_euclidean_norm_diagnostic": float(
                np.linalg.norm(analytic, ord=2)),
            "gate_subblock_norm": float(np.linalg.norm(
                analytic[:dimension, :dimension], ord=2)),
            "first_moment_range": [
                float(np.min(raw["first_moment"])),
                float(np.max(raw["first_moment"]))],
            "second_moment_range": [
                float(np.min(raw["second_moment"])),
                float(np.max(raw["second_moment"]))],
            "clipping_boundary_distance": _number(
                raw["clipping_boundary_distance"]),
            "qualification_split_residual": qualification_split_residual,
            "qualification_true_hessian_residual": qualification_true_residual,
            "qualification_weighted_aggregate_residual": (
                qualification_aggregate_residual),
            "checks": {
                "dimension_is_192": analytic.shape == (192, 192),
                "producer_replays": producer_residual <= TOLERANCE,
                "complete_successor_autodiff": autodiff_residual <= TOLERANCE,
                "nonzero_real_moments": bool(
                    np.any(raw["first_moment"] != 0.0)
                    and np.any(raw["second_moment"] != 0.0)),
                "off_clipping_boundary": _number(
                    raw["clipping_boundary_distance"])
                    > FIXED_CONSTANTS["clipping_boundary_tolerance"],
                "qualification_arrays_finite": qualification_finite,
                "qualification_signed_jet_nonzero": qualification_nonzero,
            },
        })
    qualification = [row for row in reports if row["stage"] == "Q"]
    qualification_devices = {row["device"] for row in qualification}
    qualification_metadata_agrees = False
    qualification_device_residual = None
    if len(qualification) == 2 and len(qualification_devices) == 2:
        left, right = qualification
        qualification_metadata_agrees = all(
            left[key] == right[key]
            for key in ("run_name", "seed", "step", "layer"))
        fields = (
            "gamma", "first_moment", "second_moment", "raw_gradient",
            "true_hessian", "analytic_operator", "autodiff_operator",
            "loss_value", "gradient_at_gate", "fisher",
            "sector_true_hessian", "signed_second_jet", "force_at_zero",
            "loss_terms", "gradient_at_gate_terms", "fisher_terms",
            "true_hessian_terms", "force_at_zero_terms",
            "source_count_terms",
        )
        qualification_device_residual = max(
            _relative_array(left["raw"][name], right["raw"][name])
            for name in fields)
    qualification_complete = (
        len(qualification) == 2
        and len(qualification_devices) == 2
        and qualification_metadata_agrees)
    checks = {
        "nonempty": bool(reports),
        "all_analytic_blocks_replay": all(
            row["checks"]["producer_replays"] for row in reports),
        "all_complete_successors_match": all(
            row["checks"]["complete_successor_autodiff"] for row in reports),
        "all_dimensions_are_192": all(
            row["checks"]["dimension_is_192"] for row in reports),
        "all_nodes_have_real_moments": all(
            row["checks"]["nonzero_real_moments"] for row in reports),
        "all_nodes_are_smooth_cells": all(
            row["checks"]["off_clipping_boundary"] for row in reports),
        "qualification_has_two_distinct_devices": qualification_complete,
        "qualification_real_loss_and_sector_are_finite": (
            qualification_complete and all(
                row["checks"]["qualification_arrays_finite"]
                for row in qualification)),
        "qualification_fisher_plus_signed_jet": (
            qualification_complete and all(
                row["qualification_split_residual"] <= TOLERANCE
                and row["qualification_true_hessian_residual"] <= TOLERANCE
                and row["qualification_weighted_aggregate_residual"]
                <= TOLERANCE
                for row in qualification)),
        "qualification_signed_jet_is_nonzero": (
            qualification_complete and all(
                row["checks"]["qualification_signed_jet_nonzero"]
                for row in qualification)),
        "qualification_devices_agree": (
            qualification_complete
            and qualification_device_residual is not None
            and qualification_device_residual <= TOLERANCE),
    }
    return _record("blocks", paths, checks, {
        "qualification": {
            "record_count": len(qualification),
            "devices": sorted(qualification_devices),
            "metadata_agrees": qualification_metadata_agrees,
            "maximum_cross_device_relative_residual": (
                qualification_device_residual),
        },
        "blocks": [{
            key: value for key, value in row.items() if key != "raw"
        } for row in reports],
    })


PLGA_REQUIRED = (
    "stage", "checkpoint_sha256", "run_name", "seed", "role", "pair_id",
    "layer", "head",
    "generator_left", "generator_right", "weight", "bias", "powers",
    "coupling", "coupling_bias", "curvature_difference",
    "secant_prediction", "reference_centered_logits",
    "candidate_centered_logits", "reference_margin",
)


INTERVENTION_REQUIRED = CONTINUATION_REQUIRED


def analyze_interventions(paths):
    reports = []
    groups = {}
    for path in paths:
        raw = load_npz(path, required=INTERVENTION_REQUIRED)
        if _text(raw["schema_version"]) != (
                "pldr-row-map-intervention-observables-v2"):
            raise ValueError("unknown intervention observable schema")
        node_count = len(raw["local_step"])
        if any(len(raw[name]) != node_count for name in (
                "gamma_before", "gamma_optimizer_after", "gamma_after",
                "adaptive_direction", "learning_rate", "weight_decay",
                "q_before", "q_after", "energy_before", "energy_after",
                "gate_work", "shape_work", "loss")):
            raise ValueError("intervention observable histories disagree")
        rebuilt_before = np.sum(
            raw["q_before"] * raw["gamma_before"] ** 2, axis=1)
        fixed_shape = np.sum(
            raw["q_before"] * raw["gamma_after"] ** 2, axis=1)
        rebuilt_after = np.sum(
            raw["q_after"] * raw["gamma_after"] ** 2, axis=1)
        before_residual = relative_matrix_residual(
            rebuilt_before[None], raw["energy_before"][None])
        after_residual = relative_matrix_residual(
            rebuilt_after[None], raw["energy_after"][None])
        gate_residual = relative_matrix_residual(
            (fixed_shape - rebuilt_before)[None], raw["gate_work"][None])
        shape_residual = relative_matrix_residual(
            (rebuilt_after - fixed_shape)[None], raw["shape_work"][None])
        additive_residual = relative_matrix_residual(
            (raw["energy_after"] - raw["energy_before"])[None],
            (raw["gate_work"] + raw["shape_work"])[None])
        row = {
            "path": str(path),
            "raw": raw,
            "run_name": _text(raw["run_name"]),
            "seed": int(_number(raw["seed"])),
            "layer": int(_number(raw["layer"])),
            "arm": _text(raw["arm"]),
            "node_count": node_count,
            "identity_residual": max(
                before_residual, after_residual, gate_residual,
                shape_residual, additive_residual),
        }
        reports.append(row)
        groups.setdefault((row["run_name"], row["seed"], row["layer"]), {})[
            row["arm"]] = row
    group_reports = []
    required_arms = {
        "baseline_continue", "freeze_final_gate",
        "disable_final_gate_decay", "freeze_normalized_shape",
        "replay_physical_rows",
    }
    for (run_name, seed, layer), arms in sorted(groups.items()):
        if set(arms) != required_arms:
            raise ValueError("an intervention group does not have all five arms")
        baseline = arms["baseline_continue"]["raw"]
        replay = arms["replay_physical_rows"]["raw"]
        gate_freeze = arms["freeze_final_gate"]["raw"]
        shape_freeze = arms["freeze_normalized_shape"]["raw"]
        decay_off = arms["disable_final_gate_decay"]["raw"]
        common_gamma = np.stack([
            value["raw"]["gamma_before"][0] for value in arms.values()])
        common_q = np.stack([
            value["raw"]["q_before"][0] for value in arms.values()])
        direct_decay_difference = (
            decay_off["gamma_after"][0] - baseline["gamma_after"][0])
        expected_decay_difference = (
            float(baseline["learning_rate"][0])
            * float(baseline["weight_decay"][0])
            * baseline["gamma_before"][0])
        direct_decay_residual = relative_matrix_residual(
            direct_decay_difference[None], expected_decay_difference[None])
        first_optimizer_successor_residual = _relative_array(
            baseline["gamma_optimizer_after"][0],
            decay_off["gamma_optimizer_after"][0])
        group_reports.append({
            "run_name": run_name,
            "seed": seed,
            "layer": layer,
            "common_source_gate_residual": float(np.max(
                np.linalg.norm(common_gamma - common_gamma[0], axis=1))),
            "common_source_shape_residual": float(np.max(
                np.linalg.norm(common_q - common_q[0], axis=1))),
            "baseline_replay_gate_residual": relative_matrix_residual(
                baseline["gamma_after"], replay["gamma_after"]),
            "baseline_replay_shape_residual": relative_matrix_residual(
                baseline["q_after"], replay["q_after"]),
            "baseline_replay_loss_residual": _relative_array(
                baseline["loss"], replay["loss"]),
            "gate_freeze_maximum_work": float(np.max(np.abs(
                gate_freeze["gate_work"]))),
            "gate_freeze_shape_signal": float(np.sum(np.abs(
                gate_freeze["shape_work"]))),
            "shape_freeze_maximum_work": float(np.max(np.abs(
                shape_freeze["shape_work"]))),
            "shape_freeze_gate_signal": float(np.sum(np.abs(
                shape_freeze["gate_work"]))),
            "first_optimizer_successor_relative_residual": (
                first_optimizer_successor_residual),
            "first_step_decay_term_relative_residual": direct_decay_residual,
        })
    checks = {
        "nonempty": bool(reports),
        "all_exact_work_identities": all(
            row["identity_residual"] <= TOLERANCE for row in reports),
        "all_windows_have_sixteen_updates": all(
            row["node_count"] == FIXED_CONSTANTS["continuation_updates"]
            for row in reports),
        "all_groups_have_common_sources": all(
            row["common_source_gate_residual"] <= ZERO
            and row["common_source_shape_residual"] <= ZERO
            for row in group_reports),
        "all_replay_controls_match": all(
            row["baseline_replay_gate_residual"] <= TOLERANCE
            and row["baseline_replay_shape_residual"] <= TOLERANCE
            and row["baseline_replay_loss_residual"] <= TOLERANCE
            for row in group_reports),
        "gate_freeze_removes_gate_work": all(
            row["gate_freeze_maximum_work"] <= ZERO
            for row in group_reports),
        "gate_freeze_retains_shape_signal": all(
            row["gate_freeze_shape_signal"] > ZERO
            for row in group_reports),
        "shape_freeze_removes_shape_work": all(
            row["shape_freeze_maximum_work"] <= ZERO
            for row in group_reports),
        "shape_freeze_retains_gate_signal": all(
            row["shape_freeze_gate_signal"] > ZERO
            for row in group_reports),
        "decay_off_starts_from_common_optimizer_successor": all(
            row["first_optimizer_successor_relative_residual"] <= TOLERANCE
            for row in group_reports),
        "decay_off_removes_first_direct_term": all(
            row["first_step_decay_term_relative_residual"] <= TOLERANCE
            for row in group_reports),
    }
    return _record("interventions", paths, checks, {
        "groups": group_reports,
        "archive_count": len(reports),
    })


def analyze_plga(paths):
    reports = []
    for path in paths:
        raw = load_npz(path, required=PLGA_REQUIRED)
        replay = plga_secant_chain(
            raw["generator_left"], raw["generator_right"], raw["weight"],
            raw["bias"], raw["powers"], raw["coupling"],
            raw["coupling_bias"])
        prediction_residual = relative_matrix_residual(
            replay["predicted_difference"], raw["secant_prediction"])
        implementation_residual = relative_matrix_residual(
            replay["realized_difference"], raw["curvature_difference"])
        centered_difference = (
            raw["candidate_centered_logits"]
            - raw["reference_centered_logits"])
        difference_norm = float(np.linalg.norm(centered_difference))
        margin = _number(raw["reference_margin"])
        reports.append({
            "path": str(path),
            "run_name": _text(raw["run_name"]),
            "seed": int(_number(raw["seed"])),
            "role": _text(raw["role"]),
            "pair_id": _text(raw["pair_id"]),
            "layer": int(_number(raw["layer"])),
            "head": int(_number(raw["head"])),
            "minimum_base": float(min(
                np.min(replay["metric_left"]),
                np.min(replay["metric_right"]))),
            "minimum_exponent": float(np.min(raw["powers"])),
            "maximum_exponent": float(np.max(raw["powers"])),
            "secant_relative_residual": prediction_residual,
            "implementation_relative_residual": implementation_residual,
            "operator_bound": replay["operator_bound"],
            "centered_logit_difference_norm": difference_norm,
            "reference_margin": margin,
            "reference_winner": int(np.argmax(
                raw["reference_centered_logits"])),
            "candidate_winner": int(np.argmax(
                raw["candidate_centered_logits"])),
        })
    construction = [
        row for row in reports if row["role"] == "construction"]
    validation = [row for row in reports if row["role"] == "validation"]
    construction_bound = max(
        (row["centered_logit_difference_norm"] for row in construction),
        default=None)
    validation_transfer = bool(construction) and bool(validation) and all(
        row["centered_logit_difference_norm"]
        <= TRANSFER * construction_bound + ZERO for row in validation)
    for row in reports:
        row["registered_margin_bound"] = (
            construction_bound if row["role"] == "construction"
            else TRANSFER * construction_bound)
        row["sharp_margin_qualified"] = bool(
            math.sqrt(2.0) * row["registered_margin_bound"]
            < row["reference_margin"])
        row["margin_implication_holds"] = (
            not row["sharp_margin_qualified"]
            or row["reference_winner"] == row["candidate_winner"])
    qualified_validation = [
        row for row in validation if row["sharp_margin_qualified"]]
    checks = {
        "nonempty": bool(reports),
        "all_bases_strictly_positive": all(
            row["minimum_base"] > 0.0 for row in reports),
        "all_secants_replay": all(
            row["secant_relative_residual"] <= TOLERANCE
            and row["implementation_relative_residual"] <= TOLERANCE
            for row in reports),
        "signed_exponent_range_recorded": all(
            math.isfinite(row["minimum_exponent"])
            and math.isfinite(row["maximum_exponent"]) for row in reports),
        "complete_seed_owned_pair_registry": (
            len(construction) == 12
            and len(validation) == 24
            and {row["seed"] for row in construction} == {1234}
            and {row["seed"] for row in validation} == {2222, 3333}),
        "nonempty_heldout_sharp_margin_set": bool(qualified_validation),
        "all_qualified_margin_implications_hold": all(
            row["margin_implication_holds"] for row in reports),
        "factor_2_validation_transfer": validation_transfer,
    }
    return _record("plga", paths, checks, {
        "construction_bound": construction_bound,
        "heldout_registered_bound": (
            None if construction_bound is None
            else TRANSFER * construction_bound),
        "qualified_validation_pair_count": len(qualified_validation),
        "pairs": reports,
    })


def combine(paths):
    analyses = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as stream:
            value = json.load(stream)
        unsigned = dict(value)
        recorded = unsigned.pop("analysis_sha256", None)
        if value.get("schema_version") != SCHEMA or recorded != digest_object(unsigned):
            raise ValueError("analysis record digest does not replay")
        analyses.append(value)
    by_kind = {value["kind"]: value for value in analyses}
    required = {
        "geometry", "loss", "blocks", "dynamics", "interventions", "plga"}
    checks = {
        "all_analysis_kinds_present": required.issubset(by_kind),
        "geometry_confirmation": bool(
            by_kind.get("geometry", {}).get("all_checks_pass", False)),
        "loss_sector_confirmation": bool(
            by_kind.get("loss", {}).get("all_checks_pass", False)),
        "optimizer_dynamics_confirmation": bool(
            by_kind.get("blocks", {}).get("all_checks_pass", False)),
        "shape_flux_confirmation": bool(
            by_kind.get("dynamics", {}).get("all_checks_pass", False)),
        "paired_intervention_confirmation": bool(
            by_kind.get("interventions", {}).get("all_checks_pass", False)),
        "signed_power_transfer_confirmation": bool(
            by_kind.get("plga", {}).get("all_checks_pass", False)),
    }
    return _record("campaign", paths, checks, {
        "analysis_digests": {
            value["kind"]: value["analysis_sha256"] for value in analyses},
        "all_reconstruction_checks_pass": all(checks.values()),
    })


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in (
            "geometry", "loss", "blocks", "dynamics", "interventions", "plga",
            "campaign"):
        command = subparsers.add_parser(name)
        command.add_argument("archives", nargs="+")
        command.add_argument("--output", required=True)
        command.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    function = {
        "geometry": analyze_geometry,
        "loss": analyze_loss,
        "blocks": analyze_blocks,
        "dynamics": analyze_dynamics,
        "interventions": analyze_interventions,
        "plga": analyze_plga,
        "campaign": combine,
    }[arguments.command]
    record = function(arguments.archives)
    write_json_atomic(arguments.output, record)
    if arguments.require_pass and not record["all_checks_pass"]:
        failed = sorted(
            name for name, passed in record["checks"].items() if not passed)
        raise SystemExit("analysis checks failed: " + ", ".join(failed))


if __name__ == "__main__":
    main()
