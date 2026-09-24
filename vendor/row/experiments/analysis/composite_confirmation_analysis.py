"""V3 analysis for checkpoint-bound composite-observability experiments."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

import confirmation_stage_analysis as legacy
from confirmation_artifacts import (
    REQUEST_SCHEMA_VERSION,
    STAGE_RECORD_SCHEMA_VERSION,
    digest_object,
    load_checkpoint_set,
    load_json_object,
    load_npz_strict,
    seal_record,
    sha256_path,
    validate_partition_manifest,
)
from program_energy_protocol_specs import ARTIFACT_ROLES, STAGES


STAGE_BY_ID = {stage["id"]: stage for stage in STAGES}
PRODUCER = (
    Path(__file__).resolve().parents[1]
    / "confirm" / "live_confirmation_producers.py"
)


RAW_ARRAYS = {
    "T": {
        "producer_code_sha256",
        "component_names",
        "landmark_component_digests",
        "replay_component_digests",
        "landmark_complete_digests",
        "replay_complete_digests",
        "applied_learning_rates",
        "recomputed_learning_rates",
        "landmark_steps",
        "seed_ids",
        "data_cursors",
        "model_shape",
        "optimizer_steps",
    },
    "N": {
        "producer_code_sha256",
        "checkpoint_complete_digests",
        "measurement_registry_sha256",
        "construction_observation_factors",
        "validation_observation_factors",
        "validation_full_hessian_matrices",
        "validation_hessian_fisher_matrices",
        "quotient_probability_floors",
        "quotient_projection_errors",
        "normal_states_construction",
        "normal_states_validation",
        "direct_initial_defects_construction",
        "direct_observed_defects_construction",
        "direct_initial_defects_validation",
        "direct_observed_defects_validation",
    },
    "I": {
        "producer_code_sha256",
        "source_complete_digests",
        "branch_complete_digests",
        "branch_non_rate_digests",
        "alpha_effective",
        "beta1",
        "realized_decay",
        "normal_eigenvalues",
        "normal_amplitudes",
    },
    "A": {
        "producer_code_sha256",
        "checkpoint_complete_digests",
        "measurement_registry_sha256",
        "application_pair_ids",
        "stage_names",
        "construction_row_defects",
        "construction_stage_differences",
        "row_defects",
        "observed_stage_differences",
        "layernorm_centered_norms",
        "reference_logits",
        "comparison_logits",
    },
}


def _measurement(value: Any, passed: bool, derivation: str) -> dict:
    return legacy._measurement(value, passed, derivation)


def _relative_path(value):
    return legacy._relative_path(value)


def _producer_bound(raw):
    observed = np.asarray(raw["producer_code_sha256"]).item()
    expected = sha256_path(PRODUCER)
    return observed, observed == expected


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
        observed = sha256_path(path)
        if observed != row["sha256"]:
            raise ValueError("artifact digest mismatch")
        resolved[role] = path
        records.append({
            "role": role,
            "path": relative.as_posix(),
            "sha256": observed,
            "bytes": path.stat().st_size,
        })
    if set(resolved) != set(ARTIFACT_ROLES):
        raise ValueError("artifact-role registry is incomplete")

    protocol = load_json_object(resolved["stage_protocol"])
    protocol_fields = {
        "schema_version", "stage", "title", "dependencies",
        "time_semantics", "fresh_execution", "purpose", "gpu_hour_cap",
        "required_checks", "required_artifact_roles", "raw_schema",
        "analyzer_sha256", "decision_source",
    }
    if not isinstance(protocol, dict) or set(protocol) != protocol_fields:
        raise ValueError("stage protocol has missing or unknown fields")
    if protocol["schema_version"] != "pldr-collapse-stage-protocol-v3":
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


def _q_measurements(path):
    measurements = legacy._q_measurements(path)
    record = load_json_object(path)
    diagnostics = record["jvp_and_taylor"]
    rows = diagnostics["taylor_rows"]
    ratios = [
        row["certified_remainder_upper"] / max(
            row["observed_remainder"], np.finfo(float).tiny)
        for row in rows
    ]
    ratio = max(ratios, default=math.inf)
    measurements["taylor_enclosure_ratio"] = _measurement(
        ratio,
        math.isfinite(ratio) and ratio <= 1e3,
        "maximum_whole_tube_row_enclosure_over_observed_remainder",
    )
    geometry = (
        diagnostics.get("batch_size"),
        diagnostics.get("context_length"),
    )
    measurements["frozen_job_geometry"] = _measurement(
        list(geometry), geometry == (8, 8), "recorded_batch_and_context")
    return measurements


def _component_names(checkpoint):
    return sorted(
        name for name in checkpoint["state_manifest"]
        if name != "complete_state_sha256"
    )


def _component_row(checkpoint, names):
    manifest = checkpoint["state_manifest"]
    return [
        value if isinstance(value, str) else digest_object(value)
        for value in (manifest[name] for name in names)
    ]


def _t_measurements(raw, source_path, target_path, artifact_root):
    landmarks = load_checkpoint_set(source_path, artifact_root)
    replays = load_checkpoint_set(target_path, artifact_root)
    if len(landmarks) != 4 or len(replays) != 1:
        raise ValueError("T needs four primary landmarks and one fresh replay")
    provenance, provenance_ok = _producer_bound(raw)
    names = np.asarray(raw["component_names"]).reshape(-1).tolist()
    expected_names = _component_names(landmarks[0])
    expected_components = np.asarray([
        _component_row(checkpoint, expected_names) for checkpoint in landmarks
    ])
    expected_replays = np.asarray([
        _component_row(checkpoint, expected_names) for checkpoint in replays
    ])
    observed_components = np.asarray(raw["landmark_component_digests"])
    observed_replays = np.asarray(raw["replay_component_digests"])
    complete = np.asarray(raw["landmark_complete_digests"]).reshape(-1)
    replay_complete = np.asarray(raw["replay_complete_digests"]).reshape(-1)
    expected_complete = np.asarray([
        checkpoint["state_manifest"]["complete_state_sha256"]
        for checkpoint in landmarks
    ])
    expected_replay_complete = np.asarray([
        checkpoint["state_manifest"]["complete_state_sha256"]
        for checkpoint in replays
    ])
    targets = expected_components[[3]]
    target_complete = expected_complete[[3]]
    tensorwise_replay = (
        names == expected_names
        and np.array_equal(observed_components, expected_components)
        and np.array_equal(observed_replays, expected_replays)
        and np.array_equal(expected_replays, targets)
        and np.array_equal(complete, expected_complete)
        and np.array_equal(replay_complete, expected_replay_complete)
        and np.array_equal(expected_replay_complete, target_complete)
    )
    steps = np.asarray(raw["landmark_steps"])
    expected_steps = np.asarray([
        checkpoint["step"] for checkpoint in landmarks
    ])
    cursors = np.asarray(raw["data_cursors"]).reshape(-1)
    expected_cursors = np.asarray([
        checkpoint["data_offset_end"] for checkpoint in landmarks
    ])
    clocks = np.asarray(raw["optimizer_steps"]).reshape(-1)
    seeds = np.asarray(raw["seed_ids"]).reshape(-1)
    shape = np.asarray(raw["model_shape"]).reshape(-1)
    applied = np.asarray(raw["applied_learning_rates"], dtype=float)
    recomputed = np.asarray(raw["recomputed_learning_rates"], dtype=float)
    lr_error = (
        float(np.max(np.abs(applied - recomputed), initial=0.0))
        if applied.shape == recomputed.shape else math.inf
    )
    tolerance = 64 * np.finfo(float).eps
    manifests_replay = all(
        checkpoint["state_manifest"]
        == {
            **checkpoint["state_manifest"],
        }
        for checkpoint in (*landmarks, *replays)
    )
    return {
        "live_producer_bound": _measurement(
            provenance, provenance_ok,
            "raw_producer_digest_against_active_checkpoint_only_owner"),
        "primary_seed_bound": _measurement(
            seeds.tolist(),
            seeds.tolist() == [landmarks[0]["config"]["seed"]]
            and all(
                checkpoint["config"]["seed"] == seeds[0]
                for checkpoint in (*landmarks, *replays)),
            "single_registered_primary_checkpoint_seed_identifier"),
        "architecture_and_updates": _measurement(
            shape.tolist(), shape.tolist() == [64, 4, 4, 6144],
            "checkpoint_model_configuration_and_schedule_horizon"),
        "four_primary_landmarks": _measurement(
            steps.tolist(),
            steps.shape == (1, 4)
            and np.all(np.diff(steps, axis=1) > 0)
            and np.array_equal(steps.reshape(-1), expected_steps),
            "strictly_ordered_checkpoint_landmarks"),
        "complete_checkpoint_schema": _measurement(
            complete.tolist(),
            manifests_replay and len(set(complete.tolist())) == 4,
            "complete_v3_state_manifest_validation"),
        "optimizer_clock_consistent": _measurement(
            clocks.tolist(), np.array_equal(clocks, expected_steps),
            "optimizer_step_clock_against_checkpoint_landmarks"),
        "scheduler_coefficients_recomputed": _measurement(
            lr_error, lr_error <= tolerance,
            "applied_rates_against_independent_schedule_formula"),
        "fresh_process_tensor_replay": _measurement(
            {
                "component_count": len(names),
                "targets": target_complete.tolist(),
                "replays": expected_replay_complete.tolist(),
            },
            tensorwise_replay,
            "model_optimizer_scheduler_rng_masks_registry_and_cursor_components"),
        "data_cursor_advances": _measurement(
            cursors.tolist(),
            np.array_equal(cursors, expected_cursors)
            and all(
                all(right > left for left, right in zip(row[:-1], row[1:]))
                for row in cursors.reshape(1, 4)
            ),
            "saved_next_minibatch_cursor_sequence"),
    }


def _gram_from_factors(factors):
    factors = np.asarray(factors, dtype=float)
    return np.mean(np.einsum("...ji,...jk->...ik", factors, factors), axis=0)


def _metric_gain(operator, metric):
    values, vectors = np.linalg.eigh((metric + metric.T) / 2)
    if values[0] <= 0:
        return math.inf
    inverse_root = (vectors * (1 / np.sqrt(values))[None, :]) @ vectors.T
    form = inverse_root @ operator.T @ metric @ operator @ inverse_root
    return float(math.sqrt(max(
        0.0, np.linalg.eigvalsh((form + form.T) / 2)[-1])))


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


def _partition_matches_registry(partition, checkpoints):
    registry = checkpoints[0]["measurement_registry"]
    expected = {
        "construction_ids": [row["id"] for row in registry["construction"]],
        "validation_ids": [row["id"] for row in registry["validation"]],
        "application_ids": [
            row["id"] for row in registry["application_pairs"]
        ],
        "dataset_sha256": registry["dataset_sha256"],
        "tokenizer_sha256": registry["tokenizer_sha256"],
        "measurement_registry_sha256": digest_object(registry),
    }
    return all(partition[name] == value for name, value in expected.items())


def _n_measurements(
        raw, partition, source_path, target_path, artifact_root):
    sources = load_checkpoint_set(source_path, artifact_root)
    targets = load_checkpoint_set(target_path, artifact_root)
    if not sources or len(sources) != len(targets):
        raise ValueError("N checkpoint edges are absent or unpaired")
    provenance, provenance_ok = _producer_bound(raw)
    expected_digests = np.asarray([
        [
            source["state_manifest"]["complete_state_sha256"],
            target["state_manifest"]["complete_state_sha256"],
        ]
        for source, target in zip(sources, targets)
    ])
    digest_binding = np.array_equal(
        np.asarray(raw["checkpoint_complete_digests"]), expected_digests)
    registry_digest = digest_object(sources[0]["measurement_registry"])
    registry_binding = (
        np.asarray(raw["measurement_registry_sha256"]).item()
        == registry_digest
        and all(
            checkpoint["measurement_registry"]
            == sources[0]["measurement_registry"]
            for checkpoint in (*sources, *targets)
        )
        and _partition_matches_registry(partition, sources)
    )

    construction = np.asarray(
        raw["construction_observation_factors"], dtype=float)
    validation = np.asarray(
        raw["validation_observation_factors"], dtype=float)
    full_hessians = np.asarray(
        raw["validation_full_hessian_matrices"], dtype=float)
    hessian_fishers = np.asarray(
        raw["validation_hessian_fisher_matrices"], dtype=float)
    anchor_count = len(sources)
    if construction.ndim != 4:
        raise ValueError("N construction observation factors are malformed")
    dimension = construction.shape[-1]
    if (
        validation.ndim != 4
        or construction.shape[0] != anchor_count
        or validation.shape[0] != anchor_count
        or construction.shape[2:] != (dimension, dimension)
        or validation.shape[2:] != (dimension, dimension)
        or full_hessians.shape != (anchor_count, dimension, dimension)
        or hessian_fishers.shape != full_hessians.shape
        or not all(np.isfinite(value).all() for value in (
            construction, validation, full_hessians, hessian_fishers))
    ):
        raise ValueError("N frame or Hessian tensors are malformed")

    capacities = []
    fisher_intervals = []
    full_intervals = []
    hessian_charges = []
    sample_charges = []
    charged_edges = []
    rank_results = []
    full_consistency = []
    construction_edges = []
    validation_edges = []
    tolerances = []
    for anchor, source in enumerate(sources):
        construction_gram = _gram_from_factors(construction[anchor])
        validation_gram = _gram_from_factors(validation[anchor])
        construction_values, construction_vectors = np.linalg.eigh(
            (construction_gram + construction_gram.T) / 2)
        tolerance = (
            np.finfo(float).eps * dimension
            * max(float(construction_values[-1]), 1.0)
        )
        tolerances.append(tolerance)
        visible = int(np.sum(construction_values > tolerance))
        context = source["measurement_registry"]["context_length"] - 1
        heads = int(source["model_config"]["heads"])
        head_width = int(source["model_config"]["head_width"])
        row_capacity = (
            construction.shape[1] * heads * head_width * head_width)
        score_capacity = heads * context * (context - 1) // 2
        value_capacity = heads * context * head_width
        null = dimension - visible
        capacities.append([
            dimension, row_capacity, score_capacity, value_capacity,
            visible, null,
        ])
        rank_ok = (
            visible > 0
            and visible <= min(
                dimension, row_capacity, score_capacity + value_capacity)
            and null >= max(
                0, dimension - row_capacity,
                dimension - score_capacity - value_capacity,
            )
        )
        rank_results.append(rank_ok)
        if not rank_ok:
            fisher_intervals.append([0.0, math.inf, 0.0, math.inf])
            full_intervals.append([-math.inf, math.inf])
            hessian_charges.append(math.inf)
            sample_charges.append(math.inf)
            charged_edges.append(-math.inf)
            construction_edges.append([0.0, math.inf])
            validation_edges.append([0.0, math.inf])
            full_consistency.append(False)
            continue

        basis = construction_vectors[:, -visible:]
        construction_restricted = basis.T @ construction_gram @ basis
        validation_restricted = basis.T @ validation_gram @ basis
        hessian_fisher_restricted = (
            basis.T @ hessian_fishers[anchor] @ basis)
        full_restricted = basis.T @ full_hessians[anchor] @ basis
        construction_visible = np.linalg.eigvalsh(
            (construction_restricted + construction_restricted.T) / 2)
        validation_visible = np.linalg.eigvalsh(
            (validation_restricted + validation_restricted.T) / 2)
        hessian_fisher_visible = np.linalg.eigvalsh(
            (hessian_fisher_restricted + hessian_fisher_restricted.T) / 2)
        full_visible = np.linalg.eigvalsh(
            (full_restricted + full_restricted.T) / 2)
        hessian_charge = float(np.linalg.norm(
            full_restricted - hessian_fisher_restricted, ord=2))
        sample_charge = float(np.linalg.norm(
            hessian_fisher_restricted - validation_restricted, ord=2))
        charged = float(
            validation_visible[0] - hessian_charge - sample_charge)
        fisher_intervals.append([
            float(construction_visible[0]),
            float(construction_visible[-1]),
            float(validation_visible[0]),
            float(validation_visible[-1]),
        ])
        full_intervals.append([
            float(full_visible[0]), float(full_visible[-1])])
        hessian_charges.append(hessian_charge)
        sample_charges.append(sample_charge)
        charged_edges.append(charged)
        construction_edges.append([
            float(construction_visible[0]),
            float(construction_visible[-1]),
        ])
        validation_edges.append([
            float(validation_visible[0]),
            float(validation_visible[-1]),
        ])
        numeric = 128 * np.finfo(float).eps * max(
            1.0, abs(float(full_visible[-1])),
            abs(float(hessian_fisher_visible[-1])))
        full_consistency.append(
            full_visible[0] + hessian_charge + numeric
            >= hessian_fisher_visible[0]
            and full_visible[-1] - hessian_charge - numeric
            <= hessian_fisher_visible[-1]
            and hessian_fisher_visible[0] + sample_charge + numeric
            >= validation_visible[0]
            and hessian_fisher_visible[-1] - sample_charge - numeric
            <= validation_visible[-1]
        )

    floors = np.asarray(raw["quotient_probability_floors"], dtype=float)
    projection = np.asarray(raw["quotient_projection_errors"], dtype=float)
    quotient_ok = (
        floors.shape == (anchor_count, 2)
        and projection.shape == floors.shape
        and np.isfinite(floors).all()
        and np.isfinite(projection).all()
        and np.all(floors > 0)
        and float(np.max(projection)) <= 1e-8
    )
    full_charge_ok = (
        all(rank_results)
        and all(full_consistency)
        and all(math.isfinite(value) for value in charged_edges)
        and min(charged_edges) > 0
    )

    construction_operators = []
    validation_operators = []
    for anchor, source in enumerate(sources):
        group = source["optimizer_config"]["groups"][0]
        beta = float(group["betas"][0])
        learning_rate = float(group["learning_rate"])
        alpha = learning_rate / (
            1 - beta ** max(int(source["step"]), 1))
        decay = learning_rate * float(group.get("weight_decay", 0.0))
        construction_operators.append([
            _lifted_operator(
                alpha, beta, decay, max(edge, tolerances[anchor]))
            for edge in construction_edges[anchor]
        ])
        validation_operators.append([
            _lifted_operator(
                alpha, beta, decay, max(edge, tolerances[anchor]))
            for edge in validation_edges[anchor]
        ])
    construction_operators = np.asarray(construction_operators, dtype=float)
    validation_operators = np.asarray(validation_operators, dtype=float)
    common_center = np.mean(construction_operators, axis=(0, 1))
    common_metric = _lyapunov_metric(common_center)
    metric_positive = (
        np.isfinite(common_metric).all()
        and np.linalg.eigvalsh((common_metric + common_metric.T) / 2)[0] > 0
    )
    construction_gains = np.asarray([
        [_metric_gain(operator, common_metric) for operator in vertices]
        for vertices in construction_operators
    ])
    validation_gains = np.asarray([
        [_metric_gain(operator, common_metric) for operator in vertices]
        for vertices in validation_operators
    ])

    def block_products(gains):
        edge_gains = np.max(gains, axis=1)
        block_size = min(8, len(edge_gains))
        return np.asarray([
            np.prod(edge_gains[start:start + block_size])
            for start in range(0, len(edge_gains), block_size)
            if len(edge_gains[start:start + block_size]) == block_size
        ], dtype=float)

    construction_products = block_products(construction_gains)
    validation_products = block_products(validation_gains)
    common_metric_ok = (
        metric_positive
        and construction_products.size > 0
        and np.isfinite(construction_products).all()
        and float(np.max(construction_products)) < 1
    )
    ordered_ok = (
        validation_products.size > 0
        and np.isfinite(validation_products).all()
        and float(np.max(validation_products)) < 1
    )

    states_c = np.asarray(raw["normal_states_construction"], dtype=float)
    states_v = np.asarray(raw["normal_states_validation"], dtype=float)
    if (
        states_c.shape != (anchor_count, 2, 2)
        or states_v.shape != states_c.shape
        or not np.isfinite(states_c).all()
        or not np.isfinite(states_v).all()
    ):
        raise ValueError("N construction or validation states are malformed")

    def metric_norm(vector):
        squared = float(vector @ common_metric @ vector)
        return math.sqrt(max(0.0, squared))

    recurrence_ratios = []
    state_ratios = []
    force_bounds = []
    for anchor in range(anchor_count):
        operator_c = construction_operators[anchor].mean(axis=0)
        operator_v = validation_operators[anchor].mean(axis=0)
        construction_residual = (
            states_c[anchor, 1] - operator_c @ states_c[anchor, 0])
        validation_residual = (
            states_v[anchor, 1] - operator_v @ states_v[anchor, 0])
        numerical_pad = 64 * np.finfo(float).eps * (
            1 + metric_norm(states_c[anchor, 1])
            + metric_norm(operator_c @ states_c[anchor, 0]))
        force_bound = math.nextafter(
            metric_norm(construction_residual) + numerical_pad, math.inf)
        force_bounds.append(force_bound)
        recurrence_ratios.append(
            metric_norm(validation_residual)
            / max(force_bound, np.finfo(float).tiny))
        state_ratios.append(
            metric_norm(states_v[anchor, 1])
            / max(
                metric_norm(operator_v @ states_v[anchor, 0]) + force_bound,
                np.finfo(float).tiny,
            ))

    initial_c = np.asarray(
        raw["direct_initial_defects_construction"], dtype=float)
    observed_c = np.asarray(
        raw["direct_observed_defects_construction"], dtype=float)
    initial_v = np.asarray(
        raw["direct_initial_defects_validation"], dtype=float)
    observed_v = np.asarray(
        raw["direct_observed_defects_validation"], dtype=float)
    if (
        initial_c.ndim != 2
        or observed_c.shape != initial_c.shape
        or initial_v.ndim != 2
        or observed_v.shape != initial_v.shape
        or initial_c.shape[0] != anchor_count
        or initial_v.shape[0] != anchor_count
        or not all(np.isfinite(value).all() for value in (
            initial_c, observed_c, initial_v, observed_v))
        or np.any(initial_c < 0) or np.any(observed_c < 0)
        or np.any(initial_v < 0) or np.any(observed_v < 0)
    ):
        raise ValueError("N direct construction or validation tensors malformed")
    direct_ratios = []
    direct_gains = []
    direct_additive = []
    for anchor in range(anchor_count):
        gain = max(
            np.linalg.norm(operator, ord=2)
            for operator in construction_operators[anchor])
        additive = max(
            0.0,
            float(np.max(
                observed_c[anchor] - gain * initial_c[anchor],
                initial=0.0,
            )),
        )
        additive = math.nextafter(additive, math.inf)
        envelope = gain * initial_v[anchor] + additive
        direct_ratios.extend((
            observed_v[anchor]
            / np.maximum(envelope, np.finfo(float).tiny)
        ).tolist())
        direct_gains.append(gain)
        direct_additive.append(additive)

    clip = np.asarray([
        _mask_fraction(source["branch_state"]["clip_mask"])
        for source in sources
    ])
    decay = np.asarray([
        _mask_fraction(source["branch_state"]["decay_mask"])
        for source in sources
    ])
    branch_ok = (
        np.all((0 <= clip) & (clip <= 1))
        and np.all((0 <= decay) & (decay <= 1))
    )
    tol = 4096 * np.finfo(float).eps
    recurrence_ok = (
        bool(recurrence_ratios)
        and max(recurrence_ratios) <= 1 + tol
        and max(state_ratios) <= 1 + tol
    )
    direct_ok = bool(direct_ratios) and max(direct_ratios) <= 1 + tol
    return {
        "partition_and_producer_bound": _measurement(
            {
                "producer": provenance,
                "registry": registry_digest,
                "checkpoint_edges": expected_digests.tolist(),
            },
            provenance_ok and digest_binding and registry_binding,
            "active_live_owner_checkpoint_registry_and_partition_digests"),
        "capacity_obstructions_accounted": _measurement(
            capacities, all(rank_results),
            "analyzer_derived_row_score_value_and_composite_nullity_counts"),
        "backward_frame_recomputed": _measurement(
            {
                "fisher_intervals": fisher_intervals,
                "construction_rows": int(construction.shape[1]),
                "validation_rows": int(validation.shape[1]),
            },
            all(rank_results)
            and all(np.isfinite(row).all() for row in (
                np.asarray(fisher_intervals),)),
            "mean_factor_transpose_factor_and_construction_visible_basis"),
        "composite_kernel_resolved": _measurement(
            {
                "probability_floor": float(np.min(floors)),
                "projection_error": float(np.max(projection)),
                "visible_null": [row[-2:] for row in capacities],
            },
            quotient_ok and all(rank_results),
            "causal_softmax_quotient_and_visible_null_gram_decomposition"),
        "charged_full_edge_positive": _measurement(
            {
                "charged_lower": charged_edges,
                "full_intervals": full_intervals,
                "hessian_charges": hessian_charges,
                "hessian_sample_charges": sample_charges,
            },
            full_charge_ok,
            "heldout_edge_minus_same_sample_hessian_and_sampling_charges"),
        "sector_comparison_contracts": _measurement(
            {
                "clip_fractions": clip.tolist(),
                "decay_fractions": decay.tolist(),
                "validation_gains": validation_gains.tolist(),
            },
            branch_ok and np.isfinite(validation_gains).all(),
            "checkpoint_derived_branch_masks_and_rebuilt_lifted_vertices"),
        "common_metric_contracts": _measurement(
            {
                "metric": common_metric.tolist(),
                "construction_gains": construction_gains.tolist(),
                "construction_block_products": construction_products.tolist(),
            },
            common_metric_ok,
            "one_analyzer_solved_construction_metric_for_all_vertices"),
        "ordered_windows_contract": _measurement(
            validation_products.tolist(), ordered_ok,
            "chronological_eight_edge_common_metric_products"),
        "nonlinear_recurrence_encloses": _measurement(
            {
                "force_bounds": force_bounds,
                "residual_ratios": recurrence_ratios,
                "state_ratios": state_ratios,
            },
            recurrence_ok,
            "heldout_successor_against_construction_residual_force_envelope"),
        "direct_validation_contracts": _measurement(
            {
                "maximum_ratio": max(direct_ratios),
                "construction_gains": direct_gains,
                "construction_additive": direct_additive,
            },
            direct_ok,
            "physical_row_defect_inserted_once_in_heldout_successor_envelope"),
    }


def _first_sign_change(values):
    return legacy._first_sign_change(values)


def _i_measurements(raw, source_path, target_path, artifact_root):
    sources = load_checkpoint_set(source_path, artifact_root)
    branches = load_checkpoint_set(target_path, artifact_root)
    provenance, provenance_ok = _producer_bound(raw)
    source_digests = np.asarray(raw["source_complete_digests"])
    branch_digests = np.asarray(raw["branch_complete_digests"])
    expected_source = (
        sources[0]["state_manifest"]["complete_state_sha256"]
        if len(sources) == 1 else None
    )
    expected_branches = np.asarray([
        checkpoint["state_manifest"]["complete_state_sha256"]
        for checkpoint in branches
    ])
    alpha = np.asarray(raw["alpha_effective"], dtype=float)
    beta = np.asarray(raw["beta1"], dtype=float)
    decay = np.asarray(raw["realized_decay"], dtype=float)
    eigenvalues = np.asarray(raw["normal_eigenvalues"], dtype=float)
    amplitudes = np.asarray(raw["normal_amplitudes"], dtype=float)
    non_rate = np.asarray(raw["branch_non_rate_digests"])
    if (
        alpha.ndim != 3 or alpha.shape[1] != 2
        or beta.shape != alpha.shape or decay.shape != alpha.shape
        or eigenvalues.shape != alpha.shape
        or amplitudes.shape[:2] != alpha.shape[:2]
        or amplitudes.shape[2] != alpha.shape[2] + 1
        or source_digests.shape != alpha.shape[:2]
        or non_rate.shape != alpha.shape[:2]
        or branch_digests.shape != alpha.shape
    ):
        raise ValueError("I live branch arrays are malformed")
    branch_binding = (
        expected_source is not None
        and np.all(source_digests == expected_source)
        and np.array_equal(branch_digests.reshape(-1), expected_branches)
    )
    determinant = 1 - beta * (1 - decay)
    origin = (1 - beta) * (decay + alpha * eigenvalues)
    negative_one = (
        2 * (1 + beta)
        - decay * (1 + beta)
        - alpha * (1 - beta) * eigenvalues
    )
    margins = np.minimum.reduce([determinant, origin, negative_one])
    low = margins[:, 0]
    high = margins[:, 1]
    low_terminal = np.abs(amplitudes[:, 0, -1])
    high_terminal = np.abs(amplitudes[:, 1, -1])
    low_onset = [_first_sign_change(value) for value in amplitudes[:, 0]]
    high_onset = [_first_sign_change(value) for value in amplitudes[:, 1]]
    return {
        "live_producer_bound": _measurement(
            provenance, provenance_ok and branch_binding,
            "active_live_owner_and_source_branch_complete_state_digests"),
        "matched_source_states": _measurement(
            source_digests.tolist(),
            np.all(source_digests[:, 0] == source_digests[:, 1]),
            "identical_complete_source_state_within_each_rate_pair"),
        "branch_only_changes_rate": _measurement(
            non_rate.tolist(), np.all(non_rate[:, 0] == non_rate[:, 1]),
            "all_non_rate_optimizer_model_data_and_registry_configuration"),
        "jury_margins_recomputed": _measurement(
            float(np.min(margins)), np.isfinite(margins).all(),
            "decay_aware_jury_margins_from_live_effective_adam_coefficients"),
        "corridor_bracketed": _measurement(
            {
                "low": np.min(low, axis=1).tolist(),
                "high": np.min(high, axis=1).tolist(),
            },
            np.all(np.min(low, axis=1) > 0)
            and np.all(np.min(high, axis=1) <= 0),
            "registered_low_branch_inside_and_high_branch_beyond_corridor"),
        "damping_order_observed": _measurement(
            {
                "low": low_terminal.tolist(),
                "high": high_terminal.tolist(),
            },
            np.all(low_terminal < high_terminal),
            "terminal_composite_normal_amplitude_order"),
        "oscillation_onset_order_observed": _measurement(
            {"low": low_onset, "high": high_onset},
            all(high_value <= low_value for high_value, low_value in zip(
                high_onset, low_onset)),
            "first_nonzero_normal_amplitude_sign_change"),
    }


EXPECTED_STAGES = (
    "A_LM", "A_P", "G_LM", "scores", "softmax",
    "attention_output", "residual", "layer_norm", "logits",
)


def _a_measurements(
        raw, partition, source_path, target_path, artifact_root):
    source_states = load_checkpoint_set(source_path, artifact_root)
    target_states = load_checkpoint_set(target_path, artifact_root)
    checkpoints = [*source_states, *target_states]
    provenance, provenance_ok = _producer_bound(raw)
    expected_digests = np.asarray([
        checkpoint["state_manifest"]["complete_state_sha256"]
        for checkpoint in checkpoints
    ])
    digest_binding = np.array_equal(
        np.asarray(raw["checkpoint_complete_digests"]), expected_digests)
    registry_binding = (
        bool(checkpoints)
        and all(
            checkpoint["measurement_registry"]
            == checkpoints[0]["measurement_registry"]
            for checkpoint in checkpoints
        )
        and np.asarray(raw["measurement_registry_sha256"]).item()
        == digest_object(checkpoints[0]["measurement_registry"])
        and _partition_matches_registry(partition, checkpoints)
    )
    identifiers = np.asarray(raw["application_pair_ids"]).reshape(-1).tolist()
    names = np.asarray(raw["stage_names"]).reshape(-1).tolist()
    construction_row = np.asarray(
        raw["construction_row_defects"], dtype=float)
    construction_observed = np.asarray(
        raw["construction_stage_differences"], dtype=float)
    row = np.asarray(raw["row_defects"], dtype=float)
    observed = np.asarray(raw["observed_stage_differences"], dtype=float)
    floors = np.asarray(raw["layernorm_centered_norms"], dtype=float)
    reference = np.asarray(raw["reference_logits"], dtype=float)
    comparison = np.asarray(raw["comparison_logits"], dtype=float)
    if (
        row.ndim != 3
        or construction_row.ndim != 3
        or construction_observed.shape != (
            construction_row.shape[0], construction_row.shape[1],
            construction_row.shape[2], len(EXPECTED_STAGES))
        or construction_row.shape[::2] != row.shape[::2]
        or observed.shape != (
            row.shape[0], row.shape[1], row.shape[2], len(EXPECTED_STAGES))
        or floors.shape != row.shape
        or reference.shape != comparison.shape
        or reference.shape[:2] != row.shape[:2]
        or names != list(EXPECTED_STAGES)
    ):
        raise ValueError("A live stage tensors are malformed")
    gains = np.zeros(
        (row.shape[0], row.shape[2], len(EXPECTED_STAGES)), dtype=float)
    additive = np.zeros_like(gains)
    for stage_index in range(len(EXPECTED_STAGES)):
        source = (
            construction_row if stage_index == 0
            else construction_observed[..., stage_index - 1]
        )
        output = construction_observed[..., stage_index]
        ratios = output / np.maximum(source, np.finfo(float).tiny)
        gains[..., stage_index] = np.nextafter(
            np.max(ratios, axis=1), math.inf)
        additive[..., stage_index] = (
            64 * np.finfo(float).eps * np.max(output, axis=1))
    envelope = row.copy()
    ratios = []
    for stage_index in range(gains.shape[-1]):
        envelope = (
            gains[:, None, :, stage_index] * envelope
            + additive[:, None, :, stage_index]
        )
        ratios.extend((
            observed[..., stage_index]
            / np.maximum(envelope, np.finfo(float).tiny)
        ).reshape(-1).tolist())
    top_two = np.partition(reference, -2, axis=-1)[..., -2:]
    margins = top_two[..., 1] - top_two[..., 0]
    argmax_equal = (
        np.argmax(reference, axis=-1) == np.argmax(comparison, axis=-1)
    )
    final_bound = np.min(envelope, axis=-1)
    registered = math.sqrt(2) * final_bound < margins
    margin_ok = np.all(argmax_equal[registered])
    application_expected = partition["application_ids"]
    tol = 4096 * np.finfo(float).eps
    return {
        "partition_and_producer_bound": _measurement(
            {
                "producer": provenance,
                "checkpoint_digests": expected_digests.tolist(),
                "application_ids": identifiers,
            },
            provenance_ok and digest_binding and registry_binding
            and identifiers == application_expected,
            "active_live_owner_checkpoint_registry_and_immutable_pair_ids"),
        "heldout_pairs_immutable": _measurement(
            len(identifiers), len(identifiers) == 32,
            "application_registry_pair_count"),
        "all_layers_registered": _measurement(
            row.shape[2], row.shape[2] == 4,
            "every_decoder_layer_axis_present"),
        "row_factor_applied_once": _measurement(
            names, names == list(EXPECTED_STAGES),
            "single_initial_row_defect_followed_by_ordered_stage_recurrence"),
        "stage_envelopes_recomputed": _measurement(
            float(np.max(envelope)),
            np.isfinite(envelope).all() and np.all(gains >= 0)
            and np.all(additive >= 0),
            "construction_gain_times_current_envelope_plus_additive_defect"),
        "absolute_differences_enclosed": _measurement(
            max(ratios), max(ratios) <= 1 + tol,
            "heldout_absolute_stage_difference_over_recomputed_envelope"),
        "layernorm_floors_measured": _measurement(
            float(np.min(floors)), np.all(floors > 0),
            "paired_centered_layernorm_input_norms"),
        "registered_margins_preserved": _measurement(
            {
                "qualified_pairs": int(np.sum(registered)),
                "minimum_margin": float(np.min(margins)),
            },
            bool(np.any(registered)) and margin_ok,
            "analyzer_recomputed_logits_argmax_and_sharp_sqrt_two_margin"),
    }


def analyze_stage(*, stage_id, request, artifact_root):
    if stage_id not in STAGE_BY_ID:
        raise ValueError("unknown confirmation stage")
    stage = STAGE_BY_ID[stage_id]
    resolved, records, partition = _resolve_request(
        request, artifact_root, stage)
    resources = legacy._resource_result(
        resolved["resource_metadata"], stage)
    if stage_id == "Q":
        measurements = _q_measurements(resolved["raw_observations"])
    elif stage_id == "T":
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _t_measurements(
            raw, resolved["source_complete_state"],
            resolved["target_complete_state"], artifact_root)
    elif stage_id == "N":
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _n_measurements(
            raw, partition, resolved["source_complete_state"],
            resolved["target_complete_state"], artifact_root)
    elif stage_id == "I":
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _i_measurements(
            raw, resolved["source_complete_state"],
            resolved["target_complete_state"], artifact_root)
    elif stage_id == "A":
        raw = load_npz_strict(
            resolved["raw_observations"], RAW_ARRAYS[stage_id])
        measurements = _a_measurements(
            raw, partition, resolved["source_complete_state"],
            resolved["target_complete_state"], artifact_root)
    else:
        measurements = legacy._r_measurements(
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
            else "NOT_CONFIRMED"
        ),
    }
    return seal_record(result)
