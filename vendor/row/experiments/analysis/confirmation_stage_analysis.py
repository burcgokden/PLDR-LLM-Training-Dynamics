"""Recompute Q/T/N/I/A/R decisions from bound raw artifacts."""

from __future__ import annotations

import copy
from functools import lru_cache
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from confirmation_artifacts import (
    REQUEST_SCHEMA_VERSION,
    STAGE_RECORD_SCHEMA_VERSION,
    digest_object,
    load_checkpoint_set,
    load_json_object,
    load_npz_strict,
    scientific_record_digest,
    seal_record,
    sha256_path,
    validate_partition_manifest,
    verify_sealed_record,
)
from program_energy_protocol_specs import (
    ARTIFACT_ROLES,
    RESOURCE_CAPS,
    STAGES,
)
from validated_directional_taylor_mixed import (
    certify_directional_row_jacobian_remainder,
)
from validated_interval import as_fraction, nonnegative_sqrt_upper


STAGE_BY_ID = {stage["id"]: stage for stage in STAGES}


RAW_ARRAYS = {
    "T": {
        "source_model_digests",
        "target_model_digests",
        "replayed_target_model_digests",
        "applied_learning_rates",
        "recomputed_learning_rates",
        "landmark_steps",
        "seed_ids",
        "data_cursors",
        "model_shape",
        "optimizer_steps",
    },
    "N": {
        "probabilities_construction",
        "fisher_factors_construction",
        "fisher_factors_validation",
        "full_hessians_validation",
        "loss_second_derivative_tensors",
        "stage_maps_construction",
        "physical_rows_construction",
        "normal_states_validation",
        "linear_operators_construction",
        "linear_operators_validation",
        "forcing_validation",
        "nonlinear_second_directional_actions_construction",
        "direct_defect_tensors_validation",
        "direct_stage_maps_construction",
        "direct_additive_tensors_validation",
    },
    "I": {
        "source_state_digests",
        "branch_non_rate_digests",
        "learning_rates",
        "beta1",
        "weight_decay",
        "normal_eigenvalues",
        "normal_amplitudes",
    },
    "A": {
        "row_defects",
        "operator_norms",
        "additive_defects",
        "observed_stage_differences",
        "layernorm_centered_norms",
        "logit_margins",
        "argmax_equal",
        "row_factor_counts",
    },
}


def _measurement(value: Any, passed: bool, derivation: str) -> dict:
    if isinstance(value, np.generic):
        value = value.item()
    return {
        "value": value,
        "passed": bool(passed),
        "derivation": derivation,
    }


def _relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path must be a nonempty string")
    path = Path(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError("artifact paths must be normalized and relative")
    return path


def _resolve_request(request, artifact_root, stage):
    required = {
        "schema_version", "campaign_id", "stage", "time_semantics",
        "artifacts",
    }
    if not isinstance(request, dict) or set(request) != required:
        raise ValueError("stage request has missing or unknown fields")
    if request["schema_version"] != REQUEST_SCHEMA_VERSION:
        raise ValueError("unknown stage request schema")
    if request["stage"] != stage["id"]:
        raise ValueError("request stage disagrees with the analyzer stage")
    if request["time_semantics"] not in stage["time_semantics"]:
        raise ValueError("request time semantics are inadmissible")
    rows = request["artifacts"]
    if not isinstance(rows, list) or len(rows) != len(ARTIFACT_ROLES):
        raise ValueError("request must bind every artifact role exactly once")
    root = Path(artifact_root).resolve()
    resolved = {}
    records = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "role", "path", "sha256",
        }:
            raise ValueError("artifact binding has missing or unknown fields")
        role = row["role"]
        if role not in ARTIFACT_ROLES or role in resolved:
            raise ValueError("artifact role is unknown or duplicated")
        relative = _relative_path(row["path"])
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("artifact does not resolve inside its root")
        digest = sha256_path(path)
        if digest != row["sha256"]:
            raise ValueError("artifact digest mismatch")
        resolved[role] = path
        records.append({
            "role": role,
            "path": relative.as_posix(),
            "sha256": digest,
            "bytes": path.stat().st_size,
        })
    if set(resolved) != set(ARTIFACT_ROLES):
        raise ValueError("artifact-role registry is incomplete")

    protocol = load_json_object(resolved["stage_protocol"])
    required_protocol = {
        "schema_version", "stage", "title", "dependencies",
        "time_semantics", "fresh_execution", "purpose", "gpu_hour_cap",
        "required_checks", "required_artifact_roles", "raw_schema",
        "analyzer_sha256", "decision_source",
    }
    if not isinstance(protocol, dict) or set(protocol) != required_protocol:
        raise ValueError("stage protocol has missing or unknown fields")
    if protocol["schema_version"] != "pldr-collapse-stage-protocol-v2":
        raise ValueError("unknown stage protocol schema")
    if (
        protocol["stage"] != stage["id"]
        or protocol["required_checks"] != list(stage["required_checks"])
        or protocol["required_artifact_roles"] != list(ARTIFACT_ROLES)
        or protocol["raw_schema"] != stage["raw_schema"]
    ):
        raise ValueError("stage protocol disagrees with the frozen registry")
    if protocol["analyzer_sha256"] != sha256_path(Path(__file__)):
        raise ValueError("stage protocol does not bind this analyzer")
    partition = load_json_object(resolved["partition_manifest"])
    validate_partition_manifest(partition)
    return resolved, sorted(records, key=lambda row: row["role"]), partition


def _resource_result(path, stage):
    value = load_json_object(path)
    required = {
        "schema_version", "work_units", "gpu_device_seconds",
        "peak_gpu_bytes_each", "peak_host_bytes",
        "retained_artifact_bytes", "wall_clock_seconds",
    }
    if set(value) != required:
        raise ValueError("resource metadata has missing or unknown fields")
    if value["schema_version"] != "pldr-resource-metadata-v2":
        raise ValueError("unknown resource metadata schema")
    seconds = value["gpu_device_seconds"]
    if (
        not isinstance(seconds, list)
        or any(
            not isinstance(item, (int, float))
            or isinstance(item, bool)
            or not math.isfinite(float(item))
            or float(item) < 0
            for item in seconds
        )
    ):
        raise ValueError("GPU device seconds must be a nonnegative number list")
    numeric_names = (
        "work_units", "peak_gpu_bytes_each", "peak_host_bytes",
        "retained_artifact_bytes", "wall_clock_seconds",
    )
    numbers = {}
    for name in numeric_names:
        raw = value[name]
        if (
            not isinstance(raw, (int, float))
            or isinstance(raw, bool)
            or not math.isfinite(float(raw))
            or float(raw) < 0
        ):
            raise ValueError(f"resource value {name} is invalid")
        numbers[name] = float(raw)
    if numbers["work_units"] > 0 and (
        numbers["peak_host_bytes"] == 0
        or numbers["retained_artifact_bytes"] == 0
        or numbers["wall_clock_seconds"] == 0
    ):
        raise ValueError("nonempty work cannot report zero required resources")
    gpu_hours = sum(float(item) for item in seconds) / 3600.0
    checks = {
        "gpu_count": len(seconds) <= RESOURCE_CAPS["gpu_count"],
        "gpu_memory":
            numbers["peak_gpu_bytes_each"] <= RESOURCE_CAPS["gpu_bytes_each"],
        "host_memory":
            numbers["peak_host_bytes"] <= RESOURCE_CAPS["host_bytes_total"],
        "retained_artifacts":
            numbers["retained_artifact_bytes"]
            <= RESOURCE_CAPS["retained_artifact_bytes"],
        "stage_gpu_hours": gpu_hours <= stage["gpu_hour_cap"],
        "campaign_gpu_hours": gpu_hours <= RESOURCE_CAPS["hard_gpu_hours"],
        "wall_clock":
            numbers["wall_clock_seconds"] / 3600.0
            <= RESOURCE_CAPS["hard_wall_clock_hours"],
    }
    return {
        "reported": value,
        "recomputed_gpu_hours": gpu_hours,
        "checks": checks,
        "passed": all(checks.values()),
    }


@lru_cache(maxsize=4)
def _replay_q_certificate_payload(payload_json):
    payload = json.loads(payload_json)
    required = {
        "schema_version", "source_specification", "target_specification",
        "source_rows", "target_rows", "derivative_certificates",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        return False, (), math.inf
    source = payload["source_specification"]
    target = payload["target_specification"]
    source_rows = payload["source_rows"]
    target_rows = payload["target_rows"]
    recorded = payload["derivative_certificates"]
    if (
        payload["schema_version"] != "pldr-q-taylor-certificate-inputs-v3"
        or not isinstance(source, dict) or not isinstance(target, dict)
        or source.get("width") != 8 or target.get("width") != 8
        or not isinstance(source_rows, list) or len(source_rows) != 64
        or not isinstance(target_rows, list) or len(target_rows) != 64
        or not isinstance(recorded, list) or len(recorded) != 64
        or any(
            not isinstance(row, list) or len(row) != 8
            for row in (*source_rows, *target_rows)
        )
    ):
        return False, (), math.inf
    replayed = []
    try:
        for source_row, target_row in zip(source_rows, target_rows):
            replayed.append(certify_directional_row_jacobian_remainder(
                source, target, source_row, target_row))
        exact = replayed == recorded
        bounds = tuple(float(as_fraction(row["remainder_upper"]))
                       for row in replayed)
        rational_bounds = [
            as_fraction(row["remainder_upper"]) for row in replayed]
        stack = nonnegative_sqrt_upper(sum(
            (value * value for value in rational_bounds),
            as_fraction(0),
        ))
        return exact, bounds, float(stack)
    except (ArithmeticError, KeyError, TypeError, ValueError):
        return False, (), math.inf


def _replay_q_certificates(jvp):
    inputs = jvp.get("taylor_certificate_inputs")
    certificates = jvp.get("derivative_certificates")
    if not isinstance(inputs, dict) or not isinstance(certificates, list):
        return False, (), math.inf
    payload = {**inputs, "derivative_certificates": certificates}
    return _replay_q_certificate_payload(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _q_measurements(path):
    record = load_json_object(path)
    required = {
        "producer_code_sha256", "schema_version", "stage", "device",
        "owned_edge", "replay",
        "jvp_and_taylor", "measured_geometry", "resource_metadata",
        "conditions", "decision", "scientific_record_sha256",
        "record_sha256",
    }
    if set(record) != required:
        raise ValueError("Q raw record has missing or unknown fields")
    verify_sealed_record(record)
    if record.get("schema_version") != "pldr-q-live-record-v2":
        raise ValueError("Q raw record has an unknown schema")
    geometry = record["measured_geometry"]
    jvp = record["jvp_and_taylor"]
    edge = record["owned_edge"]
    tolerance = 256 * np.finfo(np.float64).eps
    witness = jvp["live_positive_witness"]
    replay_ok, replayed_uppers, replayed_stack_upper = (
        _replay_q_certificates(jvp))
    producer_path = (
        Path(__file__).resolve().parents[1]
        / "confirm" / "qualification_normal_stability.py")
    producer_ok = record["producer_code_sha256"] == sha256_path(producer_path)
    taylor_rows = jvp["taylor_rows"]
    per_row_enclosed = (
        replay_ok
        and len(taylor_rows) == jvp["row_count"] == len(replayed_uppers)
        and [item["row_index"] for item in taylor_rows]
        == list(range(jvp["row_count"]))
        and all(
            0 <= item["observed_remainder"]
            <= item["certified_remainder_upper"]
            == replayed_uppers[item["row_index"]]
            for item in taylor_rows
        )
    )
    mutated = copy.deepcopy(record)
    mutated["decision"] = "CORRUPTED"
    mutation_rejected = False
    try:
        verify_sealed_record(mutated)
    except ValueError:
        mutation_rejected = True
    return {
        "all_rows_registered": _measurement(
            jvp["row_count"], (
                jvp["row_count"] == 64
                and replay_ok
                and jvp["derivative_certificate_count"] == jvp["row_count"]
            ), "registered_rows_and_derivative_certificate_count"),
        "dense_matrix_free_jvp": _measurement(
            jvp["dense_action_error"],
            jvp["dense_action_error"] < tolerance,
            "independent_dense_jacobian_times_direction_minus_jvp"),
        "moving_row_decomposition": _measurement(
            jvp["moving_row_decomposition_error"],
            jvp["moving_row_decomposition_error"] < tolerance,
            "joint_jvp_minus_parameter_and_physical_row_parts"),
        "taylor_remainder_enclosed": _measurement(
            {
                "stack_observed": jvp["observed_remainder"],
                "stack_upper": jvp["certified_remainder_upper"],
                "rows_checked": len(taylor_rows),
            },
            per_row_enclosed and jvp["observed_remainder"]
            <= jvp["certified_remainder_upper"]
            == replayed_stack_upper,
            "each_observed_row_and_stack_remainder_against_whole_segment_bound"),
        "layernorm_floor_measured": _measurement(
            geometry["layernorm_centered_floor"],
            geometry["layernorm_centered_floor"] > 0,
            "minimum_centered_input_norm_over_all_registered_rows"),
        "adamw_successor_exact": _measurement(
            geometry["adamw_replay_relative_error"],
            geometry["adamw_replay_relative_error"] < tolerance,
            "operation_ordered_clipped_adamw_tensor_replay"),
        "energy_identity": _measurement(
            jvp["energy_identity_error"],
            jvp["energy_identity_error"]
            <= tolerance * max(jvp["energy"], 1.0),
            "sum_centered_stack_energy_equals_row_count_times_covariance_trace"),
        "negative_work_measured": _measurement(
            geometry["optimizer_gradient_work"],
            geometry["optimizer_gradient_work"] < 0,
            "raw_gradient_inner_product_with_realized_parameter_update"),
        "positive_witness_constructed": _measurement(
            witness["kappa"],
            witness["kappa"]["numerator"] < witness["kappa"]["denominator"],
            "resolvent_witness_of_recomputed_live_comparison"),
        "provenance_mutation_rejected": _measurement(
            record["record_sha256"], mutation_rejected and producer_ok,
            "active_producer_digest_and_mutated_record_digest_replay"),
    }


def _t_measurements(raw, source_path, target_path, artifact_root):
    landmarks_states = load_checkpoint_set(source_path, artifact_root)
    replay_states = load_checkpoint_set(target_path, artifact_root)
    if len(landmarks_states) != 8 or len(replay_states) != 2:
        raise ValueError(
            "T requires eight landmark states and two fresh replay states")
    expected_source = [
        landmarks_states[index]["state_manifest"]["model_sha256"]
        for index in (0, 4)
    ]
    expected_target = [
        landmarks_states[index]["state_manifest"]["model_sha256"]
        for index in (3, 7)
    ]
    expected_replay = [
        checkpoint["state_manifest"]["model_sha256"]
        for checkpoint in replay_states
    ]
    observed_source = np.asarray(
        raw["source_model_digests"]).reshape(-1).tolist()
    observed_target = np.asarray(
        raw["target_model_digests"]).reshape(-1).tolist()
    observed_replay = np.asarray(
        raw["replayed_target_model_digests"]).reshape(-1).tolist()
    seed_ids = np.asarray(raw["seed_ids"]).reshape(-1)
    landmarks = np.asarray(raw["landmark_steps"])
    model_shape = np.asarray(raw["model_shape"]).reshape(-1)
    clocks = np.asarray(raw["optimizer_steps"]).reshape(-1)
    cursors = np.asarray(raw["data_cursors"]).reshape(-1)
    tolerance = 64 * np.finfo(np.float64).eps
    lr_error = (
        float(np.max(np.abs(
            raw["applied_learning_rates"]
            - raw["recomputed_learning_rates"]), initial=0.0))
        if raw["applied_learning_rates"].shape
        == raw["recomputed_learning_rates"].shape
        else math.inf
    )
    steps = [checkpoint["step"] for checkpoint in landmarks_states]
    cursor_expected = [
        checkpoint["data_offset_end"] for checkpoint in landmarks_states]
    complete_digests = [
        checkpoint["state_manifest"]["complete_state_sha256"]
        for checkpoint in (*landmarks_states, *replay_states)
    ]
    replay_match = (
        observed_source == expected_source
        and observed_target == expected_target
        and observed_replay == expected_replay
        and expected_target == expected_replay
    )
    return {
        "two_distinct_seeds": _measurement(
            seed_ids.tolist(),
            seed_ids.size == 2 and len(set(seed_ids.tolist())) == 2,
            "unique_seed_identifiers_in_run_manifest"),
        "architecture_and_updates": _measurement(
            model_shape.tolist(),
            model_shape.tolist() == [64, 4, 4, 6144],
            "checkpoint_model_configuration_and_registered_horizon"),
        "four_landmarks_per_seed": _measurement(
            landmarks.tolist(),
            landmarks.shape == (2, 4)
            and np.all(np.diff(landmarks, axis=1) > 0)
            and landmarks.reshape(-1).tolist() == steps,
            "strictly_ordered_schedule_landmark_matrix"),
        "complete_checkpoint_schema": _measurement(
            complete_digests, len(set(complete_digests)) == 8,
            "full_checkpoint_validation_and_state_manifest_recomputation"),
        "optimizer_clock_consistent": _measurement(
            clocks.tolist(),
            clocks.tolist() == steps
            and all(
                all(b > a for a, b in zip(row[:-1], row[1:]))
                for row in landmarks.tolist()
            ),
            "checkpoint_and_raw_optimizer_clocks"),
        "scheduler_coefficients_recomputed": _measurement(
            lr_error, lr_error <= tolerance,
            "prepared_learning_rates_against_independent_schedule_formula"),
        "fresh_process_tensor_replay": _measurement(
            {
                "source": observed_source,
                "target": observed_target,
                "replay": observed_replay,
            },
            replay_match,
            "model_tensor_digests_from_fresh_process_complete_states"),
        "data_cursor_advances": _measurement(
            cursors.tolist(),
            cursors.tolist() == cursor_expected
            and all(
                all(b > a for a, b in zip(row[:-1], row[1:]))
                for row in cursors.reshape(2, 4).tolist()
            ),
            "saved_data_cursor_sequence"),
    }


def _minimum_singular(matrix):
    return float(np.linalg.svd(matrix, compute_uv=False)[-1])


def _lyapunov_metric(operator):
    size = operator.shape[0]
    system = np.eye(size * size) - np.kron(operator.T, operator.T)
    vector = np.eye(size).reshape(-1, order="F")
    metric = np.linalg.solve(system, vector).reshape(
        (size, size), order="F")
    return (metric + metric.T) / 2


def _metric_gain(operator, metric):
    values, vectors = np.linalg.eigh(metric)
    if values[0] <= 0:
        return math.inf
    inverse_root = (
        vectors * (1.0 / np.sqrt(values))[None, :]
    ) @ vectors.T
    form = inverse_root @ operator.T @ metric @ operator @ inverse_root
    return float(math.sqrt(max(0.0, np.linalg.eigvalsh(
        (form + form.T) / 2)[-1])))


def _n_measurements(raw, partition):
    probabilities = np.asarray(raw["probabilities_construction"], dtype=float)
    factors_c = np.asarray(raw["fisher_factors_construction"], dtype=float)
    factors_v = np.asarray(raw["fisher_factors_validation"], dtype=float)
    full_h = np.asarray(raw["full_hessians_validation"], dtype=float)
    second_tensors = np.asarray(
        raw["loss_second_derivative_tensors"], dtype=float)
    stage_maps = np.asarray(raw["stage_maps_construction"], dtype=float)
    physical = np.asarray(raw["physical_rows_construction"], dtype=float)
    kc = np.asarray(raw["linear_operators_construction"], dtype=float)
    kv = np.asarray(raw["linear_operators_validation"], dtype=float)
    states = np.asarray(raw["normal_states_validation"], dtype=float)
    forces = np.asarray(raw["forcing_validation"], dtype=float)
    nonlinear_actions = np.asarray(
        raw["nonlinear_second_directional_actions_construction"],
        dtype=float)
    direct_tensors = np.asarray(
        raw["direct_defect_tensors_validation"], dtype=float)
    direct_maps = np.asarray(
        raw["direct_stage_maps_construction"], dtype=float)
    direct_additive_tensors = np.asarray(
        raw["direct_additive_tensors_validation"], dtype=float)
    if (
        nonlinear_actions.ndim < 3 or direct_tensors.ndim < 4
        or direct_maps.ndim != 4 or direct_additive_tensors.ndim < 4
    ):
        raise ValueError("N derivative and direct-stage tensors are malformed")
    nonlinear = np.max(np.linalg.norm(
        nonlinear_actions.reshape(
            nonlinear_actions.shape[0], nonlinear_actions.shape[1], -1),
        axis=-1), axis=1) / 2.0
    direct = np.linalg.norm(direct_tensors, axis=-1)
    direct_additive = np.linalg.norm(direct_additive_tensors, axis=-1)
    direct_gains = np.asarray([
        [np.linalg.norm(matrix, ord=2) for matrix in anchor]
        for anchor in direct_maps
    ])

    if probabilities.ndim < 2 or np.any(probabilities <= 0):
        probability_floor = -math.inf
    else:
        probability_floor = float(np.min(probabilities))
    anchors = factors_c.shape[0]
    if (
        factors_c.ndim != 4 or factors_v.ndim != 4
        or factors_c.shape[0] != factors_v.shape[0]
        or full_h.shape[:2] != factors_v.shape[:2]
        or stage_maps.ndim != 5 or physical.ndim != 4
        or kc.ndim != 4 or kv.ndim != 4
        or states.ndim != 3 or forces.ndim != 3
        or states.shape[0] != anchors or states.shape[1] != forces.shape[1] + 1
        or nonlinear.shape != (anchors,)
        or second_tensors.shape[:2] != factors_v.shape[:2]
    ):
        raise ValueError("N raw arrays have inconsistent anchor dimensions")

    physical_edges = []
    factorized_edges = []
    sharp_c = []
    sharp_v = []
    residual_ratios = []
    common_gains = []
    products = []
    recurrence_ratios = []
    direct_ratios = []
    for anchor in range(anchors):
        centered = physical[anchor] - physical[anchor].mean(
            axis=1, keepdims=True)
        covariance = np.einsum(
            "srd,sre->sde", centered, centered
        ) / centered.shape[1]
        physical_edge = min(
            float(np.linalg.eigvalsh(value)[0]) for value in covariance)
        physical_edges.append(physical_edge)
        stage_edge = 1.0
        for sample in range(stage_maps.shape[1]):
            sample_edge = 1.0
            for matrix in stage_maps[anchor, sample]:
                sample_edge *= _minimum_singular(matrix)
            stage_edge = min(stage_edge, sample_edge)
        factorized_edges.append(
            probability_floor * physical_edge * stage_edge ** 2)

        construction_values = [
            float(np.linalg.eigvalsh(factor.T @ factor)[0])
            for factor in factors_c[anchor]
        ]
        validation_values = [
            float(np.linalg.eigvalsh(factor.T @ factor)[0])
            for factor in factors_v[anchor]
        ]
        sharp_c.append((min(construction_values), max(construction_values)))
        sharp_v.extend(validation_values)
        for sample, factor in enumerate(factors_v[anchor]):
            residual = full_h[anchor, sample] - factor.T @ factor
            second_norm = float(np.linalg.norm(
                second_tensors[anchor, sample].reshape(-1)))
            denominator = math.sqrt(2.0) * second_norm
            residual_ratios.append(
                float(np.linalg.norm(residual, ord=2)) / max(
                    denominator, np.finfo(float).tiny))

        nominal = np.mean(kc[anchor], axis=0)
        metric = _lyapunov_metric(nominal)
        gains = [
            _metric_gain(operator, metric)
            for operator in np.concatenate((kc[anchor], kv[anchor]), axis=0)
        ]
        common_gains.extend(gains)
        validation_gains = [
            _metric_gain(operator, metric) for operator in kv[anchor]
        ]
        products.append(float(np.prod(validation_gains)))
        for edge, operator in enumerate(kv[anchor]):
            left = float(np.linalg.norm(states[anchor, edge + 1]))
            right = (
                float(np.linalg.norm(operator @ states[anchor, edge]))
                + float(np.linalg.norm(forces[anchor, edge]))
                + float(nonlinear[anchor])
                * float(np.linalg.norm(states[anchor, edge])) ** 2
            )
            recurrence_ratios.append(
                left / max(right, np.finfo(float).tiny))

        expected = np.asarray(
            np.linalg.norm(states[anchor], axis=1), dtype=float)
        for stage in range(direct_gains.shape[1]):
            expected = (
                direct_gains[anchor, stage] * expected
                + direct_additive[anchor, :, stage]
            )
            observed = direct[anchor, :, stage]
            direct_ratios.extend(
                (observed / np.maximum(
                    expected, np.finfo(float).tiny)).tolist())

    lower_c = min(value[0] for value in sharp_c)
    upper_c = max(value[1] for value in sharp_c)
    tol = 2048 * np.finfo(float).eps
    partition_disjoint = validate_partition_manifest(partition)
    return {
        "partition_disjoint": _measurement(
            digest_object(partition), partition_disjoint,
            "set_disjointness_of_bound_row_identifiers"),
        "softmax_quotient_resolved": _measurement(
            probability_floor, probability_floor > 0,
            "minimum_probability_lower_edge_on_the_zero_sum_logit_quotient"),
        "physical_excitation_positive": _measurement(
            min(physical_edges), min(physical_edges) > 0,
            "minimum_eigenvalue_of_centered_physical_row_covariance"),
        "factorized_edge_recomputed": _measurement(
            min(factorized_edges), min(factorized_edges) > 0,
            "probability_floor_times_physical_edge_times_restricted_conorm_product"),
        "sharp_fisher_edge_enclosed": _measurement(
            {
                "construction_lower": lower_c,
                "construction_upper": upper_c,
                "validation_min": min(sharp_v),
                "validation_max": max(sharp_v),
            },
            min(sharp_v) + tol >= lower_c
            and max(sharp_v) <= upper_c + tol,
            "eigenvalues_of_raw_fisher_factor_gram_matrices"),
        "full_hessian_charge_enclosed": _measurement(
            max(residual_ratios), max(residual_ratios) <= 1 + tol,
            "spectral_norm_of_full_hessian_minus_fisher_gram_over_sqrt_two_Mz2"),
        "common_metric_contracts": _measurement(
            max(common_gains), max(common_gains) < 1,
            "one_construction_lyapunov_metric_tested_on_all_owned_operators"),
        "ordered_windows_contract": _measurement(
            max(products), max(products) < 1,
            "chronological_product_of_common_metric_gains"),
        "nonlinear_recurrence_encloses": _measurement(
            max(recurrence_ratios), max(recurrence_ratios) <= 1 + tol,
            "observed_normal_successor_against_linear_force_quadratic_recurrence"),
        "direct_validation_contracts": _measurement(
            max(direct_ratios), max(direct_ratios) <= 1 + tol,
            "row_defect_inserted_once_then_propagated_through_stage_gains"),
    }


def _first_sign_change(values):
    signs = np.sign(values)
    for index in range(1, len(signs)):
        if signs[index] != 0 and signs[index - 1] != 0 and (
            signs[index] != signs[index - 1]
        ):
            return index
    return len(values) + 1


def _i_measurements(raw):
    sources = np.asarray(raw["source_state_digests"])
    non_rate = np.asarray(raw["branch_non_rate_digests"])
    rates = np.asarray(raw["learning_rates"], dtype=float)
    beta = np.asarray(raw["beta1"], dtype=float)
    decay = np.asarray(raw["weight_decay"], dtype=float)
    eigenvalues = np.asarray(raw["normal_eigenvalues"], dtype=float)
    amplitudes = np.asarray(raw["normal_amplitudes"], dtype=float)
    if (
        sources.ndim != 2 or sources.shape[1] != 2
        or non_rate.shape != sources.shape
        or rates.ndim != 3 or rates.shape[1] != 2
        or amplitudes.shape[:2] != rates.shape[:2]
        or amplitudes.shape[2] != rates.shape[2] + 1
    ):
        raise ValueError("I raw branch arrays have inconsistent dimensions")
    beta = np.broadcast_to(beta, rates.shape)
    decay = np.broadcast_to(decay, rates.shape)
    eigenvalues = np.broadcast_to(eigenvalues, rates.shape)
    determinant_margin = 1 - beta * (1 - decay)
    origin_margin = (1 - beta) * (decay + rates * eigenvalues)
    negative_one_margin = (
        2 * (1 + beta) - decay * (1 + beta)
        - rates * (1 - beta) * eigenvalues
    )
    margins = np.minimum.reduce(
        [determinant_margin, origin_margin, negative_one_margin])
    low = margins[:, 0]
    high = margins[:, 1]
    low_terminal = np.abs(amplitudes[:, 0, -1])
    high_terminal = np.abs(amplitudes[:, 1, -1])
    low_onset = [
        _first_sign_change(value) for value in amplitudes[:, 0]
    ]
    high_onset = [
        _first_sign_change(value) for value in amplitudes[:, 1]
    ]
    return {
        "matched_source_states": _measurement(
            sources.tolist(), np.all(sources[:, 0] == sources[:, 1]),
            "source_complete_state_digest_equality_within_seed"),
        "branch_only_changes_rate": _measurement(
            non_rate.tolist(), np.all(non_rate[:, 0] == non_rate[:, 1]),
            "digest_of_all_non_rate_branch_configuration"),
        "jury_margins_recomputed": _measurement(
            float(np.min(margins)), np.isfinite(margins).all(),
            "three_exact_decay_aware_second_order_jury_margins"),
        "corridor_bracketed": _measurement(
            {
                "low_min": float(np.min(low)),
                "high_min": float(np.min(high)),
            },
            np.all(np.min(low, axis=1) > 0)
            and np.all(np.min(high, axis=1) <= 0),
            "low_branch_inside_and_high_branch_at_or_beyond_registered_edge"),
        "damping_order_observed": _measurement(
            {
                "low_terminal": low_terminal.tolist(),
                "high_terminal": high_terminal.tolist(),
            },
            np.all(low_terminal < high_terminal),
            "matched_terminal_absolute_normal_amplitudes"),
        "oscillation_onset_order_observed": _measurement(
            {"low": low_onset, "high": high_onset},
            all(high_value <= low_value for high_value, low_value in zip(
                high_onset, low_onset)),
            "first_nonzero_sign_change_in_matched_normal_amplitudes"),
    }


def _a_measurements(raw):
    row = np.asarray(raw["row_defects"], dtype=float)
    gains = np.asarray(raw["operator_norms"], dtype=float)
    additive = np.asarray(raw["additive_defects"], dtype=float)
    observed = np.asarray(raw["observed_stage_differences"], dtype=float)
    floors = np.asarray(raw["layernorm_centered_norms"], dtype=float)
    margins = np.asarray(raw["logit_margins"], dtype=float)
    argmax_equal = np.asarray(raw["argmax_equal"], dtype=bool)
    counts = np.asarray(raw["row_factor_counts"])
    if (
        row.ndim != 3 or gains.ndim != 4
        or gains.shape != additive.shape or gains.shape != observed.shape
        or gains.shape[:3] != row.shape
        or floors.shape != row.shape
        or margins.shape != row.shape[:2]
        or argmax_equal.shape != margins.shape
        or counts.shape != gains.shape
    ):
        raise ValueError("A raw stage arrays have inconsistent dimensions")
    envelope = row.copy()
    ratios = []
    for stage in range(gains.shape[-1]):
        envelope = gains[..., stage] * envelope + additive[..., stage]
        ratios.extend((
            observed[..., stage]
            / np.maximum(envelope, np.finfo(float).tiny)
        ).reshape(-1).tolist())
    registered = margins > 0
    margin_ok = np.all(argmax_equal[registered])
    return {
        "heldout_pairs_immutable": _measurement(
            row.shape[1], row.shape[1] == 32,
            "application_partition_pair_count"),
        "all_layers_registered": _measurement(
            row.shape[2], row.shape[2] == 4,
            "row_defect_layer_axis"),
        "row_factor_applied_once": _measurement(
            np.unique(counts).tolist(), np.all(counts == 1),
            "symbolic_factor_count_carried_by_each_downstream_stage_record"),
        "stage_envelopes_recomputed": _measurement(
            float(np.max(envelope)), np.isfinite(envelope).all(),
            "ordered_stage_norm_recurrence_from_raw_operator_and_additive_terms"),
        "absolute_differences_enclosed": _measurement(
            max(ratios), max(ratios) <= 1 + 2048 * np.finfo(float).eps,
            "observed_stage_difference_divided_by_recomputed_absolute_envelope"),
        "layernorm_floors_measured": _measurement(
            float(np.min(floors)), np.all(floors > 0),
            "minimum_centered_LayerNorm_input_norm_per_pair_and_layer"),
        "registered_margins_preserved": _measurement(
            int(np.sum(registered)), margin_ok,
            "argmax_identity_on_pairs_with_positive_registered_logit_margin"),
    }


def _r_measurements(path, artifact_root):
    value = load_json_object(path)
    required = {"schema_version", "files", "rebuild_pairs"}
    if set(value) != required or (
        value["schema_version"] != "pldr-r-raw-observations-v2"
    ):
        raise ValueError("R raw manifest has missing or unknown fields")
    root = Path(artifact_root).resolve()
    file_results = []
    for row in value["files"]:
        if set(row) != {"path", "sha256", "scientific"}:
            raise ValueError("R file manifest row is malformed")
        relative = _relative_path(row["path"])
        target = (root / relative).resolve()
        if root not in target.parents or not target.is_file():
            raise ValueError("R manifest path does not resolve")
        actual = sha256_path(target)
        file_results.append(actual == row["sha256"])
    pair_results = {"record": [], "table": [], "figure": []}
    for row in value["rebuild_pairs"]:
        if set(row) != {"kind", "first", "second"}:
            raise ValueError("R rebuild pair is malformed")
        if row["kind"] not in pair_results:
            raise ValueError("R rebuild kind is unknown")
        first = (root / _relative_path(row["first"])).resolve()
        second = (root / _relative_path(row["second"])).resolve()
        pair_results[row["kind"]].append(
            first.is_file() and second.is_file()
            and sha256_path(first) == sha256_path(second))
    unsigned_probe = {
        "schema_version": STAGE_RECORD_SCHEMA_VERSION,
        "stage": "R",
        "resource_metadata": {"wall_clock_seconds": 1},
        "payload": "fixed",
    }
    digest_one = scientific_record_digest(unsigned_probe)
    unsigned_probe["resource_metadata"]["wall_clock_seconds"] = 2
    digest_two = scientific_record_digest(unsigned_probe)
    return {
        "all_artifacts_resolve": _measurement(
            len(file_results), bool(file_results) and all(file_results),
            "rehash_every_bound_manifest_file"),
        "scientific_digests_replay": _measurement(
            sum(file_results), bool(file_results) and all(file_results),
            "manifest_SHA256_equality"),
        "records_recompute_byte_stably": _measurement(
            len(pair_results["record"]),
            bool(pair_results["record"]) and all(pair_results["record"]),
            "independent_record_file_byte_equality"),
        "tables_recompute_byte_stably": _measurement(
            len(pair_results["table"]),
            bool(pair_results["table"]) and all(pair_results["table"]),
            "independent_table_file_byte_equality"),
        "figures_recompute_byte_stably": _measurement(
            len(pair_results["figure"]),
            bool(pair_results["figure"]) and all(pair_results["figure"]),
            "independent_figure_file_byte_equality"),
        "resource_metadata_unsigned": _measurement(
            digest_one, digest_one == digest_two,
            "scientific_digest_before_and_after_wall_clock_mutation"),
    }


def analyze_stage(*, stage_id, request, artifact_root):
    if stage_id not in STAGE_BY_ID:
        raise ValueError("unknown confirmation stage")
    stage = STAGE_BY_ID[stage_id]
    resolved, records, partition = _resolve_request(
        request, artifact_root, stage)
    resources = _resource_result(resolved["resource_metadata"], stage)

    if stage_id == "Q":
        measurements = _q_measurements(resolved["raw_observations"])
    elif stage_id == "T":
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _t_measurements(
            raw, resolved["source_complete_state"],
            resolved["target_complete_state"], artifact_root)
    elif stage_id == "N":
        load_checkpoint_set(
            resolved["source_complete_state"], artifact_root)
        load_checkpoint_set(
            resolved["target_complete_state"], artifact_root)
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _n_measurements(raw, partition)
    elif stage_id == "I":
        load_checkpoint_set(
            resolved["source_complete_state"], artifact_root)
        load_checkpoint_set(
            resolved["target_complete_state"], artifact_root)
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _i_measurements(raw)
    elif stage_id == "A":
        load_checkpoint_set(
            resolved["source_complete_state"], artifact_root)
        load_checkpoint_set(
            resolved["target_complete_state"], artifact_root)
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _a_measurements(raw)
    else:
        measurements = _r_measurements(
            resolved["raw_observations"], artifact_root)

    if set(measurements) != set(stage["required_checks"]):
        raise RuntimeError("analyzer implementation disagrees with stage checks")
    scientific_pass = all(
        measurement["passed"] for measurement in measurements.values())
    result = {
        "schema_version": STAGE_RECORD_SCHEMA_VERSION,
        "campaign_id": request["campaign_id"],
        "stage": stage_id,
        "time_semantics": request["time_semantics"],
        "request_sha256": digest_object(request),
        "resolved_artifacts": records,
        "derived_measurements": measurements,
        "resource_metadata": resources,
        "decision": (
            "CONFIRMED" if scientific_pass and resources["passed"]
            else "NOT_CONFIRMED"),
    }
    return seal_record(result)
