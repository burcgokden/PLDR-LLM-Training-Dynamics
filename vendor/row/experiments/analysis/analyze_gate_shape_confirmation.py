#!/usr/bin/env python3
"""Fresh-process analyzer for the prospective gate-shape confirmation.

The analyzer accepts only digest-bound evidence requests.  It rebuilds the
LayerNorm factorization, contrast energies, dense spectra, full 3d AdamW
blocks, chronological products, and PLGA secants from raw arrays.  Producer
residuals and producer verdicts are never used to decide a check.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from confirmation_artifacts import (  # noqa: E402
    load_checkpoint_set,
    validate_measurement_registry,
)
from gate_shape import (  # noqa: E402
    adamw_gate_block,
    chronological_products,
    finite_difference_adamw_block,
    gate_shape_energy_increment,
    gate_shape_output,
    global_gate_oscillation_bound,
    normalized_shape,
    plga_secant_chain,
    residual_scaled_spectrum,
    row_contrast_geometry,
)
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    ledger_identity,
    load_event_ledger,
    load_npz,
    load_request,
    seal_record,
    sha256_path,
    verify_record,
    write_json_atomic,
)
from gate_shape_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    RAW_REQUIRED,
    RESOURCE_CAPS,
    STAGE_BY_ID,
    STAGES,
)


PRODUCER = CONFIRM / "gate_shape_live.py"
APPLICATION_PRODUCER = CONFIRM / "gate_shape_application.py"
PROTOCOL_ROOT = ROOT / "experiments" / "protocols" / "gate_shape_confirmation"
PRODUCER_SCHEMA = "pldr-gate-shape-live-v2"
LOCK_SCHEMA = "pldr-gate-shape-lock-v2"
FILE_SET_SCHEMA = "pldr-bound-file-set-v1"


def _json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _scalar(raw, name, cast=float):
    if name not in raw or np.asarray(raw[name]).shape != ():
        raise ValueError(f"{name} must be one scalar")
    return cast(np.asarray(raw[name]).item())


def _close(left, right, *, rtol=2e-8, atol=2e-10):
    return bool(np.allclose(left, right, rtol=rtol, atol=atol))


def _measurement(value, passed, derivation):
    return {
        "value": value,
        "passed": bool(passed),
        "derivation": derivation,
    }


def _measurements(stage, passed, values, derivation):
    return {
        name: _measurement(values.get(name), passed.get(name, False), derivation)
        for name in STAGE_BY_ID[stage]["required_checks"]
    }


def _bound_files(manifest_path, artifact_root):
    manifest = _json(manifest_path)
    if set(manifest) != {"schema_version", "files"}:
        raise ValueError("bound file-set manifest is malformed")
    if manifest["schema_version"] != FILE_SET_SCHEMA:
        raise ValueError("unknown bound file-set schema")
    root = Path(artifact_root).resolve()
    resolved = []
    seen = set()
    for row in manifest["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("bound file-set row is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("bound file-set path is not normalized")
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError("bound file-set path does not resolve")
        if path in seen or sha256_path(path) != row["sha256"]:
            raise ValueError("bound file-set alias or digest mismatch")
        seen.add(path)
        resolved.append(path)
    if not resolved:
        raise ValueError("bound file-set is empty")
    return resolved


def _load_registry(path):
    value = _json(path)
    validate_measurement_registry(value, require_nonempty=True)
    if value["schema_version"] != "pldr-gate-shape-registry-v1":
        raise ValueError("stage did not bind the gate-shape registry")
    return value


def _load_lock(path):
    value = _json(path)
    if value.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unknown gate-shape construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("gate-shape construction lock digest does not replay")
    if value.get("source_owned") is not True:
        raise ValueError("gate-shape lock is not marked source-owned")
    if value.get("validation_artifacts") != []:
        raise ValueError("construction lock contains validation evidence")
    expected_code = digest_object({
        "gate_shape_live.py": sha256_path(PRODUCER),
        "gate_shape_application.py": sha256_path(APPLICATION_PRODUCER),
    })
    if value.get("code_sha256") != expected_code:
        raise ValueError("construction lock does not bind the active producers")
    if value.get("protocol_sha256") != sha256_path(
        PROTOCOL_ROOT / "campaign_design.json"
    ):
        raise ValueError("construction lock does not bind the frozen protocol")
    return value


def _archive(path, stage=None):
    raw = load_npz(path)
    if _scalar(raw, "schema_version", str) != PRODUCER_SCHEMA:
        raise ValueError("unknown gate-shape producer archive")
    archive_stage = _scalar(raw, "stage", str)
    expected_producer = (
        APPLICATION_PRODUCER if archive_stage == "A" else PRODUCER)
    if _scalar(raw, "producer_sha256", str) != sha256_path(expected_producer):
        raise ValueError("archive was not produced by the active source")
    if stage is not None and _scalar(raw, "stage", str) != stage:
        raise ValueError(f"archive does not belong to stage {stage}")
    for name, value in raw.items():
        if value.dtype.hasobject:
            raise ValueError(f"{name} uses object dtype")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            terminal_direction = (
                name == "q_directional"
                and "target_checkpoint_sha256" in raw
                and _scalar(raw, "target_checkpoint_sha256", str) == ""
                and np.isnan(value).all())
            if name != "clip_value" and not terminal_direction:
                raise ValueError(f"{name} contains a nonfinite value")
    return raw


def _resource(raw):
    required = {
        "resource_elapsed_seconds", "resource_gpu_device_seconds",
        "resource_peak_gpu_allocated_bytes", "resource_peak_gpu_reserved_bytes",
        "resource_peak_host_bytes",
    }
    if not required.issubset(raw):
        return False, 0.0, 0.0
    values = {name: _scalar(raw, name) for name in required}
    valid = all(math.isfinite(value) and value >= 0 for value in values.values())
    return (
        valid,
        values["resource_gpu_device_seconds"],
        values["resource_elapsed_seconds"],
    )


def _checkpoint_digests(checkpoints):
    return {
        row["state_manifest"]["complete_state_sha256"]
        for row in checkpoints
    }


def _context_ids(registry, role):
    return np.asarray(
        [row["id"] for row in registry[role]], dtype=np.str_)


def _registered_transitions():
    result = {}
    for block in ARCHITECTURE["capture_update_blocks"]:
        sources = [block[0] - 1, *block[:-1]]
        for offset, (source, target) in enumerate(zip(sources, block)):
            result[(source, target)] = (block[0], offset)
    return result


def _mechanism_label(gate, shape):
    if gate > 1.25 * shape:
        return "gate-led"
    if shape > 1.25 * gate:
        return "shape-led"
    return "mixed"


def _row_diameter(rows):
    rows = np.asarray(rows, dtype=np.float64)
    return float(np.max(np.linalg.norm(
        rows[:, None, :] - rows[None, :, :], axis=-1)))


def _gate_bound_residual(rows, beta, bound):
    centered = np.asarray(rows, dtype=np.float64) - np.asarray(
        beta, dtype=np.float64)
    return 2.0 * float(np.max(np.linalg.norm(centered, axis=-1))) - float(bound)



def _q_recompute(raw, fixture):
    required = {
        "row_output", "final_preactivation", "normalized_shape",
        "gamma", "beta", "scaling_factor", "scaled_output",
        "layernorm_epsilon", "true_hessian", "fisher",
        "signed_second_jet", "gate_gradient", "first_moment",
        "second_moment", "learning_rate", "beta1", "beta2",
        "adam_epsilon", "optimizer_step", "weight_decay", "clip_value",
        "adamw_operator", "adamw_independent_operator", "context_length",
        "fixture_sha256", "device", "global_gate_bound",
    }
    if not required.issubset(raw):
        raise ValueError("qualification archive is incomplete")
    shape = normalized_shape(
        raw["final_preactivation"], epsilon=_scalar(raw, "layernorm_epsilon"))
    output = gate_shape_output(raw["beta"], raw["gamma"], shape)
    projection = np.asarray(fixture["projection"], dtype=np.float64)
    targets = np.asarray(fixture["targets"], dtype=np.int64)
    logits = output @ projection
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probability = np.exp(shifted)
    probability /= np.sum(probability, axis=1, keepdims=True)
    jacobian = shape[:, :, None] * projection[None, :, :]
    fisher = np.zeros((shape.shape[1], shape.shape[1]), dtype=np.float64)
    gradient = np.zeros(shape.shape[1], dtype=np.float64)
    for index, p in enumerate(probability):
        covariance = np.diag(p) - np.outer(p, p)
        j = jacobian[index].T
        fisher += j.T @ covariance @ j / len(shape)
        residual = p.copy()
        residual[targets[index]] -= 1.0
        gradient += j.T @ residual / len(shape)
    optimizer = {
        "learning_rate": _scalar(raw, "learning_rate"),
        "beta1": _scalar(raw, "beta1"),
        "beta2": _scalar(raw, "beta2"),
        "epsilon": _scalar(raw, "adam_epsilon"),
        "optimizer_step": _scalar(raw, "optimizer_step", int),
        "weight_decay": _scalar(raw, "weight_decay"),
        "clip_value": _scalar(raw, "clip_value"),
    }
    analytic = adamw_gate_block(
        raw["true_hessian"], raw["gamma"], raw["first_moment"],
        raw["second_moment"],
        gradient, boundary_tolerance=1e-12, **optimizer)["operator"]

    def affine_gradient(position):
        return gradient + raw["true_hessian"] @ (position - raw["gamma"])

    independent = finite_difference_adamw_block(
        affine_gradient, raw["gamma"], raw["first_moment"],
        raw["second_moment"], step_size=2e-6, **optimizer)
    factorization = float(np.linalg.norm(raw["row_output"] - output))
    zero = gate_shape_output(raw["beta"], np.zeros_like(raw["gamma"]), shape)
    zero_diameter = float(np.max(np.linalg.norm(
        zero[:, None, :] - zero[None, :, :], axis=-1)))
    shape_archive_residual = float(np.linalg.norm(
        raw["normalized_shape"] - shape))
    scale = _scalar(raw, "scaling_factor")
    scaled_output = gate_shape_output(
        raw["beta"], scale * raw["gamma"], shape)
    scaling_residual = max(
        float(np.linalg.norm(raw["scaled_output"] - scaled_output)),
        float(np.linalg.norm(
            (raw["scaled_output"][1:] - raw["scaled_output"][:-1])
            - scale * (output[1:] - output[:-1]))),
    )
    global_bound = global_gate_oscillation_bound(raw["gamma"])
    row_diameter = _row_diameter(output)
    bound_residual = abs(_scalar(raw, "global_gate_bound") - global_bound)
    return {
        "shape": shape,
        "factorization_residual": factorization,
        "zero_gate_diameter": zero_diameter,
        "shape_archive_residual": shape_archive_residual,
        "scaling_residual": scaling_residual,
        "gate_bound_residual": max(bound_residual, row_diameter - global_bound),
        "fisher": fisher,
        "gradient": gradient,
        "true_split_residual": float(np.linalg.norm(
            raw["true_hessian"] - raw["fisher"]
            - raw["signed_second_jet"], ord=2)),
        "fisher_residual": float(np.linalg.norm(raw["fisher"] - fisher, ord=2)),
        "gradient_residual": float(np.linalg.norm(raw["gate_gradient"] - gradient)),
        "affine_signed_residual": max(
            float(np.linalg.norm(raw["signed_second_jet"], ord=2)),
            float(np.linalg.norm(
                raw["true_hessian"] - raw["fisher"], ord=2))),
        "analytic": analytic,
        "independent": independent,
        "analytic_archive_residual": float(np.linalg.norm(
            raw["adamw_operator"] - analytic, ord=2)),
        "independent_archive_residual": float(np.linalg.norm(
            raw["adamw_independent_operator"] - independent, ord=2)),
        "operator_residual": float(np.linalg.norm(analytic - independent, ord=2)),
        "global_bound": global_bound,
    }


def _q_measurements(resolved):
    fixture = load_npz(
        resolved["fixture"],
        required=("schema_version", "rows", "projection", "targets"))
    if _scalar(fixture, "schema_version", str) != "pldr-gate-shape-q-fixture-v1":
        raise ValueError("unknown gate-shape qualification fixture")
    records = [
        _archive(resolved["qualification_cuda0"], "Q"),
        _archive(resolved["qualification_cuda1"], "Q"),
    ]
    recomputed = [_q_recompute(raw, fixture) for raw in records]
    fixture_digest = sha256_path(resolved["fixture"])
    same_fixture = all(
        _scalar(raw, "fixture_sha256", str) == fixture_digest
        for raw in records
    ) and {
        _scalar(raw, "device", str) for raw in records
    } == {"cuda:0", "cuda:1"}
    context_complete = all(
        _scalar(raw, "context_length", int) == ARCHITECTURE["context_length"]
        and raw["row_output"].shape == (
            ARCHITECTURE["context_length"], ARCHITECTURE["head_width"])
        for raw in records)
    factorization = max(
        max(value["factorization_residual"], value["shape_archive_residual"])
        for value in recomputed)
    zero_diameter = max(value["zero_gate_diameter"] for value in recomputed)
    scaling_residual = max(
        value["scaling_residual"] for value in recomputed)
    gate_bound_residual = max(
        value["gate_bound_residual"] for value in recomputed)
    hessian_residual = max(
        max(
            value["true_split_residual"], value["fisher_residual"],
            value["gradient_residual"], value["affine_signed_residual"],
        )
        for value in recomputed)
    adam_residual = max(
        max(value["operator_residual"], value["analytic_archive_residual"],
            value["independent_archive_residual"])
        for value in recomputed)
    scientific_names = (
        "row_output", "final_preactivation", "normalized_shape",
        "scaled_output", "gamma", "beta", "true_hessian", "fisher",
        "signed_second_jet", "gate_gradient", "adamw_operator",
    )
    cross = max(
        float(np.max(np.abs(records[0][name] - records[1][name])))
        for name in scientific_names)
    resources = [_resource(raw) for raw in records]
    passes = {
        "same_realized_fixture": same_fixture,
        "context_256_row_map_complete": context_complete,
        "gate_factorization_replays": factorization <= 2e-10,
        "zero_gate_is_constant": zero_diameter <= 2e-12,
        "gate_scaling_identity_replays": scaling_residual <= 2e-10,
        "global_gate_bound_encloses_rows": gate_bound_residual <= 2e-10,
        "dense_true_hessian_replays": hessian_residual <= 2e-8,
        "analytic_48_by_48_adamw_block_matches_autodiff": adam_residual <= 2e-6,
        "cross_device_arrays_agree_with_declared_tolerance": cross <= 2e-8,
        "measured_memory_and_seconds_recorded": all(row[0] for row in resources),
    }
    values = {
        **{name: passes[name] for name in passes},
        "gate_factorization_replays": factorization,
        "zero_gate_is_constant": zero_diameter,
        "gate_scaling_identity_replays": scaling_residual,
        "global_gate_bound_encloses_rows": gate_bound_residual,
        "dense_true_hessian_replays": hessian_residual,
        "analytic_48_by_48_adamw_block_matches_autodiff": adam_residual,
        "cross_device_arrays_agree_with_declared_tolerance": cross,
        "measured_memory_and_seconds_recorded": [row[1:] for row in resources],
    }
    return _measurements(
        "Q", passes, values,
        "Fresh NumPy reconstruction from the one CPU-realized fixture and raw arrays."), resources


def _checkpoint_sets(path, root):
    return load_checkpoint_set(path, root)


def _t_measurements(resolved, root):
    registry = _load_registry(resolved["registry"])
    if sha256_path(resolved["tokens"]) != registry["dataset_sha256"]:
        raise ValueError("T token archive disagrees with the registry")
    if sha256_path(resolved["tokenizer"]) != registry["tokenizer_sha256"]:
        raise ValueError("T tokenizer disagrees with the registry")
    construction = _checkpoint_sets(
        resolved["construction_checkpoints"], root)
    validation = _checkpoint_sets(
        resolved["validation_checkpoints"], root)
    all_checkpoints = construction + validation
    registered = set(ARCHITECTURE["anchor_source_steps"])
    for block in ARCHITECTURE["capture_update_blocks"]:
        registered.update(block)
    construction_steps = {int(value["step"]) for value in construction}
    validation_steps = {int(value["step"]) for value in validation}
    ledger_paths = (
        resolved["construction_event_ledger"],
        resolved["validation_event_ledger"],
    )
    identities = [
        ledger_identity(path, expected_updates=ARCHITECTURE["updates"])
        for path in ledger_paths
    ]
    event_rows = [load_event_ledger(path) for path in ledger_paths]
    architecture = all(
        checkpoint["measurement_registry"] == registry
        and checkpoint["model_config"]["width"] == ARCHITECTURE["width"]
        and checkpoint["model_config"]["layers"] == ARCHITECTURE["layers"]
        and checkpoint["model_config"]["heads"] == ARCHITECTURE["heads"]
        and checkpoint["model_config"]["head_width"]
        == ARCHITECTURE["head_width"]
        and checkpoint["model_config"]["row_hidden_width"]
        == ARCHITECTURE["glu_hidden_width"]
        and checkpoint["model_config"]["row_residual_layers"]
        == ARCHITECTURE["row_map_residual_units"]
        and checkpoint["model_config"]["row_dense_layers"]
        == ARCHITECTURE["glu_blocks_per_residual_unit"]
        for checkpoint in all_checkpoints)
    construction_chunks = {
        int(row["chunk_index"]) for row in registry["construction"]}
    validation_chunks = {
        int(row["chunk_index"]) for row in registry["validation"]}
    application_chunks = {
        int(row[name])
        for row in registry["application_pairs"]
        for name in ("left_chunk", "right_chunk")
    }
    partitions_disjoint = not (
        construction_chunks & validation_chunks
        or construction_chunks & application_chunks
        or validation_chunks & application_chunks)
    resources_complete = all(
        len(rows) >= ARCHITECTURE["updates"] + 2
        and rows[0].get("event") == "config"
        and any(row.get("event") == "final" for row in rows)
        and all(
            isinstance(row.get("gpu_device_seconds"), (int, float))
            and math.isfinite(float(row["gpu_device_seconds"]))
            and float(row["gpu_device_seconds"]) >= 0.0
            for row in rows
        )
        for rows in event_rows)
    passes = {
        "exactly_6144_updates": all(
            row["maximum_step"] == ARCHITECTURE["updates"]
            for row in identities),
        "all_three_anchor_blocks_complete": (
            registered.issubset(construction_steps)
            and registered.issubset(validation_steps)),
        "model_optimizer_scheduler_rng_and_data_cursor_bound": architecture,
        "one_resolved_lineage_root_per_ledger": (
            len({row["lineage_root"] for row in identities}) == 2),
        "construction_and_validation_chunks_disjoint": partitions_disjoint,
        "all_resource_events_complete": resources_complete,
    }
    return _measurements(
        "T", passes, {name: passes[name] for name in passes},
        "Complete checkpoints, registry partitions, and every ledger resource "
        "row were replayed."), []


def _tr_measurements(resolved, root):
    primary = _checkpoint_sets(resolved["primary_checkpoints"], root)
    replay = _checkpoint_sets(resolved["replay_checkpoints"], root)
    left = ledger_identity(resolved["primary_event_ledger"], expected_updates=6144)
    right = ledger_identity(resolved["replay_event_ledger"], expected_updates=6144)
    primary_by_step = {int(row["step"]): row for row in primary}
    replay_by_step = {int(row["step"]): row for row in replay}
    same_steps = set(primary_by_step) == set(replay_by_step)
    scientific = (
        "model_sha256", "optimizer_sha256", "scheduler_sha256",
        "data_state_sha256", "measurement_registry_sha256",
    )
    equal = same_steps and all(
        all(primary_by_step[step]["state_manifest"][name]
            == replay_by_step[step]["state_manifest"][name]
            for name in scientific)
        for step in primary_by_step)
    distinct = left["run_id"] != right["run_id"] and (
        left["lineage_root"] != right["lineage_root"])
    passes = {
        "distinct_run_and_lineage_identity": distinct,
        "scientific_checkpoint_components_equal": equal,
        "aliased_primary_rejected": distinct and all(
            a["state_manifest"]["complete_state_sha256"]
            != b["state_manifest"]["complete_state_sha256"]
            for a, b in zip(primary, replay)),
        "registered_update_count_and_windows_equal": same_steps,
    }
    return _measurements(
        "TR", passes, {name: passes[name] for name in passes},
        "Independent ledgers and complete checkpoint component digests were compared."), []


def _g_recompute(raw):
    epsilon = _scalar(raw, "layernorm_epsilon")
    shape = normalized_shape(raw["final_preactivation"], epsilon=epsilon)
    output = gate_shape_output(raw["beta"], raw["gamma"], shape)
    left = np.asarray(raw["pair_left"], dtype=np.int64)
    right = np.asarray(raw["pair_right"], dtype=np.int64)
    contrast = row_contrast_geometry(
        raw["gamma"], shape[left], shape[right], raw["pair_weights"])
    fisher_terms = np.asarray(raw["fisher_terms"], dtype=np.float64)
    true_terms = np.asarray(raw["true_hessian_terms"], dtype=np.float64)
    force_terms = np.asarray(raw["force_at_zero_terms"], dtype=np.float64)
    if (
        fisher_terms.ndim != 3
        or true_terms.shape != fisher_terms.shape
        or force_terms.shape != (fisher_terms.shape[0], shape.shape[1])
        or fisher_terms.shape[1:] != (shape.shape[1], shape.shape[1])
    ):
        raise ValueError("gate loss-sector term archive has the wrong shape")
    fisher = np.mean(fisher_terms, axis=0)
    fisher = 0.5 * (fisher + fisher.T)
    true_hessian = np.mean(true_terms, axis=0)
    true_hessian = 0.5 * (true_hessian + true_hessian.T)
    force = np.mean(force_terms, axis=0)
    signed = true_hessian - fisher
    split_residual = max(
        float(np.linalg.norm(
            raw["true_hessian"] - raw["fisher"]
            - raw["signed_second_jet"], ord=2)),
        float(np.linalg.norm(raw["fisher"] - fisher, ord=2)),
        float(np.linalg.norm(
            raw["true_hessian"] - true_hessian, ord=2)),
        float(np.linalg.norm(
            raw["signed_second_jet"] - signed, ord=2)),
    )
    spectrum = residual_scaled_spectrum(
        true_hessian, reconstruction_residual=split_residual)
    bound = global_gate_oscillation_bound(raw["gamma"])
    return {
        "factorization": max(
            float(np.linalg.norm(raw["row_output"] - output)),
            float(np.linalg.norm(raw["normalized_shape"] - shape)),
            abs(_scalar(raw, "global_gate_bound") - bound),
            _gate_bound_residual(output, raw["beta"], bound),
        ),
        "shape": shape,
        "contrast": contrast,
        "split_residual": split_residual,
        "force_residual": float(np.linalg.norm(
            raw["force_at_zero"] - force)),
        "spectrum": spectrum,
    }


def _g_measurements(resolved, root):
    registry = _load_registry(resolved["registry"])
    token_digest = sha256_path(resolved["tokens"])
    if token_digest != registry["dataset_sha256"]:
        raise ValueError("G token archive disagrees with the registry")
    if sha256_path(resolved["producer_source"]) != sha256_path(PRODUCER):
        raise ValueError("G request does not bind the active producer")
    checkpoints = _checkpoint_sets(resolved["checkpoints"], root)
    checkpoint_steps = {
        row["state_manifest"]["complete_state_sha256"]: int(row["step"])
        for row in checkpoints
    }
    checkpoint_roles = {}
    for row in checkpoints:
        digest = row["state_manifest"]["complete_state_sha256"]
        seed = int(row["config"]["seed"])
        if seed == ARCHITECTURE["trajectory_seeds"][0]:
            checkpoint_roles[digest] = "construction"
        elif seed == ARCHITECTURE["trajectory_seeds"][1]:
            checkpoint_roles[digest] = "validation"
        else:
            raise ValueError("G checkpoint has an unregistered trajectory seed")
    checkpoint_digests = set(checkpoint_steps)
    paths = _bound_files(resolved["gate_shape_raw_archives"], root)
    archives = [_archive(path, "G") for path in paths]
    observed = set()
    bindings = []
    for raw in archives:
        checkpoint_digest = _scalar(raw, "checkpoint_sha256", str)
        target_digest = _scalar(raw, "target_checkpoint_sha256", str)
        role = _scalar(raw, "registry_role", str)
        layer = _scalar(raw, "layer", int)
        key = (checkpoint_digest, layer)
        if key in observed:
            raise ValueError("G evidence repeats a checkpoint/layer pair")
        observed.add(key)
        bindings.append(
            checkpoint_digest in checkpoint_digests
            and (not target_digest or target_digest in checkpoint_digests)
            and checkpoint_steps.get(checkpoint_digest)
            == _scalar(raw, "step", int)
            and _scalar(raw, "tokens_sha256", str) == token_digest
            and _scalar(raw, "registry_sha256", str)
            == registry["registry_sha256"]
            and role in {"construction", "validation"}
            and checkpoint_roles.get(checkpoint_digest) == role
            and np.array_equal(raw["context_ids"], _context_ids(registry, role))
            and 0 <= layer < ARCHITECTURE["layers"])
    expected = {
        (digest, layer)
        for digest in checkpoint_digests
        for layer in range(ARCHITECTURE["layers"])
    }
    if observed != expected:
        raise ValueError("G evidence does not cover every checkpoint and layer")
    values = [_g_recompute(raw) for raw in archives]
    factorization = max(value["factorization"] for value in values)
    energy_residual = max(
        abs(value["contrast"]["energy"] - _scalar(raw, "energy"))
        + float(np.linalg.norm(value["contrast"]["q"] - raw["q"]))
        for raw, value in zip(archives, values))
    split = max(value["split_residual"] for value in values)
    force_residual = max(value["force_residual"] for value in values)
    spectra = max(
        float(np.max(np.abs(
            value["spectrum"]["eigenvalues"] - raw["eigenvalues"])))
        for raw, value in zip(archives, values))
    threshold = all(
        _close(
            value["spectrum"]["zero_threshold"],
            raw["spectral_threshold"])
        for raw, value in zip(archives, values))
    dimension = ARCHITECTURE["gate_sector_dimension_per_layer"]
    small = all(
        raw["true_hessian"].shape == (dimension, dimension)
        and raw["fisher"].shape == (dimension, dimension)
        and not any(
            token in name.lower()
            for name in raw
            for token in ("svd", "lanczos", "parameter_jacobian"))
        for raw in archives)
    passes = {
        "row_outputs_recomputed_from_final_layernorm_preactivations": (
            bool(bindings) and all(bindings) and factorization <= 2e-8),
        "q_and_energy_recomputed_from_fixed_pairs": energy_residual <= 2e-8,
        "fisher_plus_signed_jet_reconstructs_true_hessian": split <= 2e-8,
        "all_dense_spectra_recomputed": spectra <= 2e-8,
        "spectral_sign_uses_residual_scaled_threshold": threshold,
        "gate_force_at_zero_recomputed": force_residual <= 2e-8,
        "no_parameter_jacobian_svd_or_lanczos_basis": small,
    }
    measured = {name: passes[name] for name in passes}
    measured.update({
        "row_outputs_recomputed_from_final_layernorm_preactivations": (
            factorization),
        "q_and_energy_recomputed_from_fixed_pairs": energy_residual,
        "fisher_plus_signed_jet_reconstructs_true_hessian": split,
        "all_dense_spectra_recomputed": spectra,
        "gate_force_at_zero_recomputed": force_residual,
    })
    return _measurements(
        "G", passes, measured,
        "The analyzer rebuilt shape, row output, pair geometry, per-context "
        "loss terms, zero-gate force, and every dense spectrum."), [
            _resource(raw) for raw in archives]


def _c_capture(raw):
    clip_value = _scalar(raw, "clip_value")
    if math.isnan(clip_value):
        clip_value = None
    weight_decay = np.asarray(raw["weight_decay"], dtype=np.float64)
    if weight_decay.shape == ():
        weight_decay = float(weight_decay)
    optimizer = {
        "learning_rate": _scalar(raw, "learning_rate"),
        "beta1": _scalar(raw, "beta1"),
        "beta2": _scalar(raw, "beta2"),
        "epsilon": _scalar(raw, "adam_epsilon"),
        "optimizer_step": _scalar(raw, "optimizer_step", int),
        "weight_decay": weight_decay,
        "clip_value": clip_value,
    }
    analytic = adamw_gate_block(
        raw["true_hessian"], raw["gamma"], raw["first_moment"],
        raw["second_moment"], raw["raw_gradient"],
        boundary_tolerance=1e-8, **optimizer)["operator"]

    def affine_gradient(position):
        return raw["raw_gradient"] + raw["true_hessian"] @ (
            position - raw["gamma"])

    numerical = finite_difference_adamw_block(
        affine_gradient, raw["gamma"], raw["first_moment"],
        raw["second_moment"], step_size=_scalar(raw, "finite_difference_step"),
        **optimizer)
    return analytic, numerical


def _c_measurements(resolved, root):
    registry = _load_registry(resolved["registry"])
    token_digest = sha256_path(resolved["tokens"])
    if token_digest != registry["dataset_sha256"]:
        raise ValueError("C token archive disagrees with the registry")
    if sha256_path(resolved["producer_source"]) != sha256_path(PRODUCER):
        raise ValueError("C request does not bind the active producer")
    lock = _load_lock(resolved["construction_lock"])
    source_checkpoints = _checkpoint_sets(
        resolved["source_checkpoints"], root)
    endpoint_checkpoints = _checkpoint_sets(
        resolved["endpoint_checkpoints"], root)
    checkpoints = source_checkpoints + endpoint_checkpoints
    checkpoint_steps = {
        row["state_manifest"]["complete_state_sha256"]: int(row["step"])
        for row in checkpoints
    }
    checkpoint_roles = {}
    for row in checkpoints:
        digest = row["state_manifest"]["complete_state_sha256"]
        seed = int(row["config"]["seed"])
        if seed == ARCHITECTURE["trajectory_seeds"][0]:
            checkpoint_roles[digest] = "construction"
        elif seed == ARCHITECTURE["trajectory_seeds"][1]:
            checkpoint_roles[digest] = "validation"
        else:
            raise ValueError("C checkpoint has an unregistered trajectory seed")
    snapshot_paths = _bound_files(resolved["optimizer_snapshots"], root)
    snapshot_digests = {sha256_path(path) for path in snapshot_paths}
    paths = _bound_files(resolved["capture_raw_archives"], root)
    raw_by_digest = {}
    geometries = {}
    captures = []
    energy_steps = []
    stage_digest_sets = {
        "construction": set(),
        "validation": set(),
    }
    for path in paths:
        raw = _archive(path)
        digest = sha256_path(path)
        raw_by_digest[digest] = raw
        stage = _scalar(raw, "stage", str)
        if stage == "G":
            role = _scalar(raw, "registry_role", str)
            checkpoint_digest = _scalar(raw, "checkpoint_sha256", str)
            if (
                role not in stage_digest_sets
                or checkpoint_digest not in checkpoint_steps
                or checkpoint_roles[checkpoint_digest] != role
                or checkpoint_steps[checkpoint_digest]
                != _scalar(raw, "step", int)
                or _scalar(raw, "tokens_sha256", str) != token_digest
                or _scalar(raw, "registry_sha256", str)
                != registry["registry_sha256"]
                or not np.array_equal(
                    raw["context_ids"], _context_ids(registry, role))
            ):
                raise ValueError("C geometry binding does not replay")
            geometries[digest] = raw
            stage_digest_sets[role].add(digest)
        elif stage == "C":
            captures.append((digest, raw))
        elif stage == "C-energy-step":
            energy_steps.append((digest, raw))
        else:
            raise ValueError("C file set contains an unknown archive stage")
    if not geometries or not captures or not energy_steps:
        raise ValueError("C requires geometry, operator, and energy evidence")
    expected_geometry = {
        (digest, layer)
        for digest in checkpoint_steps
        for layer in range(ARCHITECTURE["layers"])
    }
    observed_geometry = {
        (_scalar(raw, "checkpoint_sha256", str),
         _scalar(raw, "layer", int))
        for raw in geometries.values()
    }
    if observed_geometry != expected_geometry:
        raise ValueError("C geometry does not cover every checkpoint and layer")

    capture_by_key = {}
    operator_residuals = []
    used_snapshots = set()
    for digest, raw in captures:
        geometry_digest = _scalar(raw, "geometry_sha256", str)
        geometry = geometries.get(geometry_digest)
        snapshot_digest = _scalar(raw, "snapshot_sha256", str)
        if geometry is None or snapshot_digest not in snapshot_digests:
            raise ValueError("C capture does not bind geometry and snapshot")
        role = _scalar(geometry, "registry_role", str)
        layer = _scalar(raw, "layer", int)
        step = _scalar(raw, "step", int)
        key = (role, layer, step)
        if (
            key in capture_by_key
            or _scalar(geometry, "layer", int) != layer
            or _scalar(geometry, "step", int) != step - 1
        ):
            raise ValueError("C capture is duplicated or misaligned")
        capture_by_key[key] = (digest, raw)
        stage_digest_sets[role].add(digest)
        used_snapshots.add(snapshot_digest)
        analytic, numerical = _c_capture(raw)
        operator_residuals.append(max(
            float(np.linalg.norm(analytic - numerical, ord=2)),
            float(np.linalg.norm(analytic - raw["operator"], ord=2)),
            float(np.linalg.norm(
                numerical - raw["independent_operator"], ord=2)),
        ))
    expected_capture = {
        (role, layer, step)
        for role in ("construction", "validation")
        for layer in range(ARCHITECTURE["layers"])
        for block in ARCHITECTURE["capture_update_blocks"]
        for step in block
    }
    if (
        set(capture_by_key) != expected_capture
        or used_snapshots != snapshot_digests
    ):
        raise ValueError("C capture coverage or snapshot binding is incomplete")

    lock_blocks = {
        (int(row["layer"]), int(row["block_start"])): row
        for row in lock["capture_blocks"]
    }
    block_metrics = {}
    lock_block_replay = []
    for role in ("construction", "validation"):
        for layer in range(ARCHITECTURE["layers"]):
            for block in ARCHITECTURE["capture_update_blocks"]:
                operators = [
                    capture_by_key[(role, layer, step)][1]["operator"]
                    for step in block
                ]
                products = chronological_products(operators)
                final_gain = float(np.linalg.norm(products[-1], ord=2))
                constituent = [
                    float(np.linalg.norm(value, ord=2))
                    for value in operators
                ]
                block_metrics[(role, layer, block[0])] = {
                    "final_gain": final_gain,
                    "constituent_gains": constituent,
                }
                if role == "construction":
                    row = lock_blocks.get((layer, block[0]))
                    lock_block_replay.append(
                        row is not None
                        and row["steps"] == list(block)
                        and np.allclose(
                            row["constituent_gains"], constituent,
                            rtol=2e-8, atol=2e-10)
                        and math.isclose(
                            float(row["block_gain"]),
                            final_gain * float(lock["safety_factor"]),
                            rel_tol=2e-8, abs_tol=2e-10))
    if len(lock_blocks) != (
        ARCHITECTURE["layers"]
        * len(ARCHITECTURE["capture_update_blocks"])
    ):
        raise ValueError("construction lock has an incomplete block map")

    transitions = _registered_transitions()
    energy_by_key = {}
    energy_residuals = []
    mechanism = {}
    for digest, raw in energy_steps:
        source_digest = _scalar(raw, "source_sha256", str)
        target_digest = _scalar(raw, "target_sha256", str)
        source = geometries.get(source_digest)
        target = geometries.get(target_digest)
        if source is None or target is None:
            raise ValueError("C energy step does not resolve both geometries")
        source_role = _scalar(source, "registry_role", str)
        target_role = _scalar(target, "registry_role", str)
        layer = _scalar(raw, "layer", int)
        source_step = _scalar(raw, "source_step", int)
        target_step = _scalar(raw, "target_step", int)
        window = transitions.get((source_step, target_step))
        key = (source_role, layer, source_step, target_step)
        if (
            key in energy_by_key
            or source_role != target_role
            or window is None
            or _scalar(source, "layer", int) != layer
            or _scalar(target, "layer", int) != layer
            or _scalar(source, "step", int) != source_step
            or _scalar(target, "step", int) != target_step
            or _scalar(source, "target_checkpoint_sha256", str)
            != _scalar(target, "checkpoint_sha256", str)
        ):
            raise ValueError("C energy transition is duplicated or misbound")
        increment = gate_shape_energy_increment(
            source["gamma"], target["gamma"], source["q"], target["q"])
        directional = np.asarray(
            source["q_directional"], dtype=np.float64)
        gamma_next = np.asarray(target["gamma"], dtype=np.float64)
        linear = float(np.sum(
            directional * gamma_next * gamma_next))
        remainder = np.asarray(
            target["q"] - source["q"] - directional, dtype=np.float64)
        residual = max(
            abs(increment["identity_residual"]),
            abs(increment["energy"] - _scalar(raw, "energy")),
            abs(increment["energy_next"] - _scalar(raw, "energy_next")),
            abs(increment["gate_work"] - _scalar(raw, "gate_work")),
            abs(increment["shape_work"] - _scalar(raw, "shape_work")),
            abs(linear - _scalar(raw, "shape_linear_work")),
            float(np.max(np.abs(remainder - raw["shape_remainder"]))),
        )
        energy_residuals.append(residual)
        block_start, offset = window
        energy_by_key[key] = {
            "increment": increment,
            "linear": linear,
            "remainder": remainder,
            "target_gamma": gamma_next,
            "block_start": block_start,
            "transition_index": offset,
        }
        accumulator = mechanism.setdefault(
            (source_role, layer, block_start), [0.0, 0.0])
        accumulator[0] += abs(increment["gate_work"])
        accumulator[1] += abs(increment["shape_work"])
        stage_digest_sets[source_role].add(digest)
    expected_energy = {
        (role, layer, source_step, target_step)
        for role in ("construction", "validation")
        for layer in range(ARCHITECTURE["layers"])
        for source_step, target_step in transitions
    }
    if set(energy_by_key) != expected_energy:
        raise ValueError("C energy evidence is incomplete")

    charge_rows = {
        (int(row["layer"]), int(row["source_step"]),
         int(row["target_step"])): row
        for row in lock["energy_charges"]
    }
    force_rows = {
        (int(row["layer"]), int(row["step"])): row
        for row in lock["forcing_charges"]
    }
    envelope_pass = []
    charge_replay = []
    for (role, layer, source_step, target_step), value in energy_by_key.items():
        charge = charge_rows.get((layer, source_step, target_step))
        if charge is None:
            raise ValueError("C lock omits a transition charge")
        increment = value["increment"]
        remainder_radius = float(np.max(np.abs(value["remainder"])))
        if role == "construction":
            charge_replay.append(
                math.isclose(
                    float(charge["gate_charge"]),
                    abs(increment["gate_work"]) * lock["safety_factor"],
                    rel_tol=2e-8, abs_tol=2e-10)
                and math.isclose(
                    float(charge["shape_linear_charge"]),
                    abs(value["linear"]) * lock["safety_factor"],
                    rel_tol=2e-8, abs_tol=2e-10)
                and math.isclose(
                    float(charge["remainder_radius"]),
                    remainder_radius * lock["safety_factor"],
                    rel_tol=2e-8, abs_tol=2e-10))
        else:
            upper = (
                increment["energy"]
                + float(charge["gate_charge"])
                + float(charge["shape_linear_charge"])
                + float(charge["remainder_radius"])
                * float(np.dot(
                    value["target_gamma"], value["target_gamma"])))
            tolerance = 2e-8 * (1.0 + abs(upper))
            envelope_pass.append(
                abs(increment["gate_work"])
                <= float(charge["gate_charge"]) + tolerance
                and abs(value["linear"])
                <= float(charge["shape_linear_charge"]) + tolerance
                and remainder_radius
                <= float(charge["remainder_radius"]) + tolerance
                and increment["energy_next"] <= upper + tolerance)
    force_pass = []
    force_replay = []
    for (role, layer, step), (_digest, raw) in capture_by_key.items():
        charge = force_rows.get((layer, step))
        if charge is None:
            raise ValueError("C lock omits a forcing charge")
        norm = float(np.linalg.norm(raw["raw_gradient"]))
        if role == "construction":
            force_replay.append(math.isclose(
                float(charge["norm"]),
                norm * lock["safety_factor"],
                rel_tol=2e-8, abs_tol=2e-10))
        else:
            force_pass.append(norm <= float(charge["norm"]) + 2e-8)

    label_rows = {
        (int(row["layer"]), int(row["block_start"])): row["label"]
        for row in lock["mechanism_labels"]
    }
    label_replay = []
    for layer in range(ARCHITECTURE["layers"]):
        for block in ARCHITECTURE["capture_update_blocks"]:
            construction = mechanism[("construction", layer, block[0])]
            validation = mechanism[("validation", layer, block[0])]
            locked = label_rows.get((layer, block[0]))
            label_replay.append(
                locked == _mechanism_label(*construction)
                and locked == _mechanism_label(*validation))

    source_digests = set(lock["source_archive_sha256"])
    code_digest = digest_object({
        "gate_shape_live.py": sha256_path(PRODUCER),
        "gate_shape_application.py": sha256_path(APPLICATION_PRODUCER),
    })
    lock_precedes = (
        lock["registry_sha256"] == registry["registry_sha256"]
        and lock["code_sha256"] == code_digest
        and lock["intervention_arms"] == ARCHITECTURE["intervention_arms"]
        and stage_digest_sets["construction"].issubset(source_digests)
        and not (stage_digest_sets["validation"] & source_digests)
        and all(lock_block_replay)
        and all(charge_replay)
        and all(force_replay))
    operator_residual = max(operator_residuals)
    energy_residual = max(energy_residuals)
    capture_with_expansion = any(
        float(row["block_gain"]) < 1.0
        and max(float(value) for value in row["constituent_gains"]) > 1.0
        for row in lock["capture_blocks"])
    product_values = [
        {
            "role": role,
            "layer": layer,
            "block_start": block_start,
            **value,
        }
        for (role, layer, block_start), value in sorted(block_metrics.items())
    ]
    passes = {
        "analytic_gate_blocks_match_automatic_differentiation": (
            operator_residual <= 2e-6),
        "clipping_boundary_distance_recorded": all(
            _scalar(raw, "clipping_boundary_distance") > 1e-8
            for _digest, raw in captures),
        "direct_ordered_products_recomputed": (
            len(product_values)
            == 2 * ARCHITECTURE["layers"]
            * len(ARCHITECTURE["capture_update_blocks"])
            and all(lock_block_replay)),
        "construction_block_and_remainder_lock_precedes_validation": (
            lock_precedes),
        "target_energy_not_read_while_constructing_envelope": (
            lock.get("target_energy_read") is False
            and lock.get("validation_artifacts") == []),
        "gate_plus_shape_work_identity_replays_at_every_node": (
            energy_residual <= 2e-8),
        "validation_nodes_lie_in_source_owned_envelope": (
            bool(envelope_pass) and all(envelope_pass)
            and bool(force_pass) and all(force_pass)),
        "capturing_block_contains_transient_expansion": (
            capture_with_expansion),
        "construction_mechanism_label_replicates": (
            bool(label_replay) and all(label_replay)),
    }
    measured = {name: passes[name] for name in passes}
    measured.update({
        "analytic_gate_blocks_match_automatic_differentiation": (
            operator_residual),
        "direct_ordered_products_recomputed": product_values,
        "gate_plus_shape_work_identity_replays_at_every_node": (
            energy_residual),
        "validation_nodes_lie_in_source_owned_envelope": {
            "energy_nodes": len(envelope_pass),
            "force_nodes": len(force_pass),
        },
    })
    return _measurements(
        "C", passes, measured,
        "The analyzer rebuilt every 48-dimensional update, each direct "
        "eight-update product, exact work identity, and source-owned "
        "transition envelope."), []


def _i_measurements(resolved, root):
    registry = _load_registry(resolved["registry"])
    token_digest = sha256_path(resolved["tokens"])
    if token_digest != registry["dataset_sha256"]:
        raise ValueError("I token archive disagrees with the registry")
    if sha256_path(resolved["producer_source"]) != sha256_path(PRODUCER):
        raise ValueError("I request does not bind the active producer")
    lock = _load_lock(resolved["construction_lock"])
    source_rows = _checkpoint_sets(resolved["source_checkpoint"], root)
    arm_rows = _checkpoint_sets(resolved["arm_checkpoints"], root)
    if len(source_rows) != 1:
        raise ValueError("I requires exactly one complete source checkpoint")
    source = source_rows[0]
    source_step = int(source["step"])
    source_digest = source["state_manifest"]["complete_state_sha256"]
    source_identity = source.get("run_identity")
    if not isinstance(source_identity, dict):
        raise ValueError("I source checkpoint lacks a run identity")

    expected_run_ids = {
        arm: f"gate-shape-{arm}-s{source_step}"
        for arm in ARCHITECTURE["intervention_arms"]
    }
    by_arm = {arm: [] for arm in ARCHITECTURE["intervention_arms"]}
    one_source = True
    for row in arm_rows:
        identity = row.get("run_identity")
        if not isinstance(identity, dict):
            raise ValueError("I arm checkpoint lacks a run identity")
        matches = [
            arm for arm, run_id in expected_run_ids.items()
            if identity["run_id"] == run_id
        ]
        if len(matches) != 1:
            raise ValueError("I arm checkpoint has an unregistered run id")
        arm = matches[0]
        by_arm[arm].append(row)
        one_source = one_source and (
            identity["source_checkpoint_sha256"] == source_digest
            and identity["parent_run_id"] == source_identity["run_id"]
            and identity["lineage_root"] == source_identity["lineage_root"]
            and identity["from_scratch"] is False)
    expected_steps = list(range(
        source_step + 1,
        source_step + 1 + ARCHITECTURE["intervention_updates"]))
    arms_complete = all(
        len(rows) == ARCHITECTURE["intervention_updates"]
        and sorted(int(row["step"]) for row in rows) == expected_steps
        for rows in by_arm.values())

    expected_controls = {
        "baseline": (1.0, -1, -1),
        "rate_0.75": (0.75, -1, -1),
        "rate_1.25": (1.25, -1, -1),
        "freeze_final_gate": (1.0, source_step, -1),
        "freeze_upstream_shape": (1.0, -1, source_step),
    }
    allowed_config = {
        "name", "run_id", "outdir", "learning_rate_multiplier",
        "freeze_final_gate_at", "freeze_upstream_shape_at",
    }
    config_signatures = []
    intervention_signatures = []
    registered = lock["intervention_arms"] == ARCHITECTURE["intervention_arms"]
    for arm, rows in by_arm.items():
        multiplier, final_freeze, upstream_freeze = expected_controls[arm]
        for row in rows:
            config = row["config"]
            state = row["intervention_state"]
            registered = registered and (
                math.isclose(
                    float(config["learning_rate_multiplier"]), multiplier,
                    rel_tol=0.0, abs_tol=1e-15)
                and int(state["freeze_final_gate_at"]) == final_freeze
                and int(state["freeze_upstream_shape_at"]) == upstream_freeze)
            config_signatures.append((
                arm,
                {
                    key: value for key, value in config.items()
                    if key not in allowed_config
                },
            ))
            intervention_signatures.append((
                arm,
                {
                    key: value for key, value in state.items()
                    if key not in {
                        "freeze_final_gate_at",
                        "freeze_upstream_shape_at",
                    }
                },
            ))
    if config_signatures:
        baseline_config = config_signatures[0][1]
        baseline_intervention = intervention_signatures[0][1]
        registered = registered and all(
            value == baseline_config for _arm, value in config_signatures)
        registered = registered and all(
            value == baseline_intervention
            for _arm, value in intervention_signatures)

    ledger_paths = _bound_files(resolved["arm_event_ledgers"], root)
    identities = [ledger_identity(path) for path in ledger_paths]
    identity_by_run = {row["run_id"]: row for row in identities}
    ledger_complete = (
        len(ledger_paths) == len(ARCHITECTURE["intervention_arms"])
        and len(identity_by_run) == len(ledger_paths)
        and all(
            run_id in identity_by_run
            and identity_by_run[run_id]["lineage_root"]
            == source_identity["lineage_root"]
            and identity_by_run[run_id]["maximum_step"]
            == source_step + ARCHITECTURE["intervention_updates"]
            for run_id in expected_run_ids.values()))
    ledger_resources = all(
        all(
            isinstance(row.get("gpu_device_seconds"), (int, float))
            and math.isfinite(float(row["gpu_device_seconds"]))
            and float(row["gpu_device_seconds"]) >= 0.0
            for row in load_event_ledger(path)
        )
        for path in ledger_paths)

    all_checkpoints = [source, *arm_rows]
    checkpoint_steps = {
        row["state_manifest"]["complete_state_sha256"]: int(row["step"])
        for row in all_checkpoints
    }
    geometry_paths = _bound_files(
        resolved["intervention_geometry_raw_archives"], root)
    geometries = {}
    binding_pass = []
    for path in geometry_paths:
        raw = _archive(path, "I-geometry")
        checkpoint_digest = _scalar(raw, "checkpoint_sha256", str)
        layer = _scalar(raw, "layer", int)
        key = (checkpoint_digest, layer)
        if key in geometries:
            raise ValueError("I geometry repeats a checkpoint/layer pair")
        geometries[key] = raw
        binding_pass.append(
            checkpoint_digest in checkpoint_steps
            and checkpoint_steps.get(checkpoint_digest)
            == _scalar(raw, "step", int)
            and _scalar(raw, "tokens_sha256", str) == token_digest
            and _scalar(raw, "registry_sha256", str)
            == registry["registry_sha256"]
            and _scalar(raw, "registry_role", str) == "validation"
            and np.array_equal(
                raw["context_ids"], _context_ids(registry, "validation"))
            and 0 <= layer < ARCHITECTURE["layers"])
    expected_geometry = {
        (digest, layer)
        for digest in checkpoint_steps
        for layer in range(ARCHITECTURE["layers"])
    }
    if set(geometries) != expected_geometry:
        raise ValueError(
            "I geometry does not cover the source and every arm checkpoint")

    geometry_residuals = []
    for raw in geometries.values():
        shape = normalized_shape(
            raw["final_preactivation"],
            epsilon=_scalar(raw, "layernorm_epsilon"))
        output = gate_shape_output(raw["beta"], raw["gamma"], shape)
        left = np.asarray(raw["pair_left"], dtype=np.int64)
        right = np.asarray(raw["pair_right"], dtype=np.int64)
        contrast = row_contrast_geometry(
            raw["gamma"], shape[left], shape[right], raw["pair_weights"])
        bound = global_gate_oscillation_bound(raw["gamma"])
        geometry_residuals.append(max(
            float(np.linalg.norm(raw["normalized_shape"] - shape)),
            float(np.linalg.norm(raw["row_output"] - output)),
            float(np.linalg.norm(raw["q"] - contrast["q"])),
            abs(_scalar(raw, "energy") - contrast["energy"]),
            abs(_scalar(raw, "global_gate_bound") - bound),
            _gate_bound_residual(output, raw["beta"], bound),
        ))

    energy_residuals = []
    freeze_gate_residuals = []
    decomposition_rows = []
    for arm, rows in by_arm.items():
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        for layer in range(ARCHITECTURE["layers"]):
            source_geometry = geometries[(source_digest, layer)]
            previous = source_geometry
            for row in ordered:
                digest = row["state_manifest"]["complete_state_sha256"]
                target = geometries[(digest, layer)]
                increment = gate_shape_energy_increment(
                    previous["gamma"], target["gamma"],
                    previous["q"], target["q"])
                energy_residuals.append(abs(increment["identity_residual"]))
                decomposition_rows.append({
                    "arm": arm,
                    "layer": layer,
                    "target_step": int(row["step"]),
                    "gate_work": increment["gate_work"],
                    "shape_work": increment["shape_work"],
                })
                if arm == "freeze_final_gate":
                    freeze_gate_residuals.append(max(
                        float(np.linalg.norm(
                            target["gamma"] - source_geometry["gamma"])),
                        abs(increment["gate_work"]),
                    ))
                previous = target
    geometry_residual = max(geometry_residuals)
    energy_residual = max(energy_residuals)
    freeze_residual = max(freeze_gate_residuals)
    passes = {
        "branches_share_one_complete_source_state": one_source,
        "all_five_registered_arms_complete": arms_complete,
        "only_registered_intervention_differs": registered,
        "real_event_ledgers_have_one_lineage_root": (
            ledger_complete and ledger_resources),
        "all_gate_shape_energies_recomputed": (
            bool(binding_pass) and all(binding_pass)
            and geometry_residual <= 2e-8
            and energy_residual <= 2e-8),
        "final_gate_freeze_has_zero_gate_work": freeze_residual <= 2e-10,
    }
    measured = {name: passes[name] for name in passes}
    measured.update({
        "all_gate_shape_energies_recomputed": {
            "geometry_residual": geometry_residual,
            "identity_residual": energy_residual,
            "decompositions": decomposition_rows,
        },
        "final_gate_freeze_has_zero_gate_work": freeze_residual,
    })
    return _measurements(
        "I", passes, measured,
        "The analyzer bound all five continuations to one source, replayed "
        "their complete ledgers and registered controls, and recomputed every "
        "gate-shape energy increment."), [
            _resource(raw) for raw in geometries.values()]


def _a_measurements(resolved, root):
    registry = _load_registry(resolved["registry"])
    token_digest = sha256_path(resolved["tokens"])
    if token_digest != registry["dataset_sha256"]:
        raise ValueError("A token archive disagrees with the registry")
    if (
        sha256_path(resolved["producer_source"])
        != sha256_path(APPLICATION_PRODUCER)
    ):
        raise ValueError("A request does not bind the active producer")
    lock = _load_lock(resolved["construction_lock"])
    checkpoints = _checkpoint_sets(resolved["checkpoints"], root)
    checkpoint_roles = {}
    for row in checkpoints:
        digest = row["state_manifest"]["complete_state_sha256"]
        seed = int(row["config"]["seed"])
        if seed == ARCHITECTURE["trajectory_seeds"][0]:
            checkpoint_roles[digest] = "construction"
        elif seed == ARCHITECTURE["trajectory_seeds"][1]:
            checkpoint_roles[digest] = "validation"
        else:
            raise ValueError("A checkpoint has an unregistered trajectory seed")
    paths = _bound_files(resolved["application_raw_archives"], root)
    archives = [(sha256_path(path), _archive(path, "A")) for path in paths]
    registered_pairs = {row["id"] for row in registry["application_pairs"]}
    observed = set()
    bindings = []
    construction_digests = set()
    validation_digests = set()
    for digest, raw in archives:
        role = _scalar(raw, "registry_role", str)
        layer = _scalar(raw, "layer", int)
        head = _scalar(raw, "head", int)
        pair_id = _scalar(raw, "pair_id", str)
        prefix = "ga-c" if role == "construction" else "ga-v"
        expected_pair = (
            f"{prefix}{layer * ARCHITECTURE['heads'] + head:03d}")
        key = (role, layer, head)
        if (
            role not in {"construction", "validation"}
            or key in observed
            or pair_id != expected_pair
            or pair_id not in registered_pairs
        ):
            raise ValueError("A archive violates the registered pair map")
        observed.add(key)
        checkpoint_digest = _scalar(raw, "checkpoint_sha256", str)
        bindings.append(
            checkpoint_roles.get(checkpoint_digest) == role
            and _scalar(raw, "tokens_sha256", str) == token_digest
            and _scalar(raw, "registry_sha256", str)
            == registry["registry_sha256"])
        if role == "construction":
            construction_digests.add(digest)
        else:
            validation_digests.add(digest)
    expected = {
        (role, layer, head)
        for role in ("construction", "validation")
        for layer in range(ARCHITECTURE["layers"])
        for head in range(ARCHITECTURE["heads"])
    }
    if observed != expected:
        raise ValueError("A archives do not cover every registered layer/head pair")

    lock_bounds = {
        (int(row["layer"]), int(row["head"])): row
        for row in lock["application_bounds"]
    }
    expected_bound_keys = {
        (layer, head)
        for layer in range(ARCHITECTURE["layers"])
        for head in range(ARCHITECTURE["heads"])
    }
    if set(lock_bounds) != expected_bound_keys:
        raise ValueError("A lock has an incomplete application-bound map")
    residuals = []
    construction_replay = []
    validation_bounds = []
    qualifications = []
    qualified_decisions = []
    margin_rows = []
    for _digest, raw in archives:
        result = plga_secant_chain(
            raw["generator_left"], raw["generator_right"], raw["weight"],
            raw["bias"], raw["powers"], raw["coupling"],
            raw["coupling_bias"])
        reference = np.asarray(raw["reference_logits"], dtype=np.float64)
        candidate = np.asarray(raw["candidate_logits"], dtype=np.float64)
        reference_centered = reference - np.mean(reference)
        candidate_centered = candidate - np.mean(candidate)
        centered_difference = candidate_centered - reference_centered
        reference_winner = int(np.argmax(reference_centered))
        candidate_winner = int(np.argmax(candidate_centered))
        margin = float(
            reference_centered[reference_winner]
            - np.max(np.delete(reference_centered, reference_winner)))
        curvature = np.asarray(
            raw["curvature_difference"], dtype=np.float64)
        residuals.append(max(
            result["identity_residual"],
            float(np.linalg.norm(
                result["activation_secant"]
                - raw["activation_secant"])),
            float(np.linalg.norm(
                result["power_secant"] - raw["power_secant"])),
            float(np.linalg.norm(
                result["predicted_difference"]
                - raw["secant_prediction"])),
            float(np.linalg.norm(
                result["realized_difference"] - curvature)),
            float(np.linalg.norm(
                reference_centered
                - raw["reference_centered_logits"])),
            float(np.linalg.norm(
                candidate_centered
                - raw["candidate_centered_logits"])),
            float(np.linalg.norm(
                centered_difference
                - raw["centered_logit_difference"])),
            abs(
                _scalar(raw, "reference_winner", int)
                - reference_winner),
            abs(
                _scalar(raw, "candidate_winner", int)
                - candidate_winner),
            abs(_scalar(raw, "reference_margin") - margin),
            abs(
                _scalar(raw, "centered_logit_difference_norm")
                - float(np.linalg.norm(centered_difference))),
        ))
        layer = _scalar(raw, "layer", int)
        head = _scalar(raw, "head", int)
        role = _scalar(raw, "registry_role", str)
        bound = lock_bounds[(layer, head)]
        generator_norm = float(np.linalg.norm(
            raw["generator_right"] - raw["generator_left"]))
        curvature_norm = float(np.linalg.norm(curvature))
        centered_norm = float(np.linalg.norm(centered_difference))
        deductive_upper = float(bound["operator_bound"]) * generator_norm
        centered_upper = float(bound["downstream_gain"]) * curvature_norm
        tolerance = 2e-8 * (
            1.0 + deductive_upper + centered_upper)
        if role == "construction":
            if curvature_norm <= 1e-14:
                downstream_gain = 0.0
            else:
                downstream_gain = centered_norm / curvature_norm
            construction_replay.append(
                bound["construction_pair_id"]
                == _scalar(raw, "pair_id", str)
                and math.isclose(
                    float(bound["operator_bound"]),
                    float(result["operator_bound"])
                    * float(lock["safety_factor"]),
                    rel_tol=2e-8, abs_tol=2e-10)
                and math.isclose(
                    float(bound["downstream_gain"]),
                    downstream_gain * float(lock["safety_factor"]),
                    rel_tol=2e-8, abs_tol=2e-10))
        else:
            enclosed = (
                curvature_norm <= deductive_upper + tolerance
                and centered_norm <= centered_upper + tolerance)
            validation_bounds.append(enclosed)
            qualified = (
                math.sqrt(2.0) * centered_upper < margin)
            qualifications.append(qualified)
            if qualified:
                qualified_decisions.append(
                    candidate_winner == reference_winner)
            margin_rows.append({
                "pair_id": _scalar(raw, "pair_id", str),
                "qualified": qualified,
                "reference_margin": margin,
                "centered_upper_bound": centered_upper,
                "decision_preserved": (
                    candidate_winner == reference_winner),
            })
    residual = max(residuals)
    source_digests = set(lock["source_archive_sha256"])
    provenance = (
        construction_digests.issubset(source_digests)
        and not (validation_digests & source_digests))
    passes = {
        "checkpoint_bindings_and_logit_differences_replay": (
            bool(bindings) and all(bindings) and residual <= 2e-8),
        "iswiglu_and_positive_power_secants_replay": residual <= 2e-8,
        "deductive_and_centered_logit_differences_enclosed": (
            provenance and all(construction_replay)
            and bool(validation_bounds) and all(validation_bounds)),
        "margin_claim_only_for_independently_qualified_pairs": (
            all(qualified_decisions)),
        "empty_qualified_set_supports_no_decision_claim": (
            any(qualifications)
            or lock.get("empty_margin_decision") is None),
    }
    measured = {name: passes[name] for name in passes}
    measured.update({
        "iswiglu_and_positive_power_secants_replay": residual,
        "deductive_and_centered_logit_differences_enclosed": {
            "validation_pairs": len(validation_bounds),
            "all_enclosed": all(validation_bounds),
        },
        "margin_claim_only_for_independently_qualified_pairs": margin_rows,
    })
    return _measurements(
        "A", passes, measured,
        "The analyzer rebuilt exact occupied-interval secants and centered "
        "logit differences, then applied only construction-owned deductive "
        "and downstream bounds to validation pairs."), [
            _resource(raw) for _digest, raw in archives]


def _r_measurements(resolved):
    rebuild = _json(resolved["rebuild_manifest"])
    mutation = _json(resolved["mutation_report"])
    resource = _json(resolved["resource_record"])
    cases = mutation.get("cases", {})
    gpu_hours = float(resource.get("gpu_device_seconds", 0.0)) / 3600.0
    wall_hours = float(resource.get("wall_clock_seconds", 0.0)) / 3600.0
    passes = {
        "all_relative_paths_resolve_under_campaign_root": rebuild.get("relative_paths_only") is True,
        "self_pairs_and_absolute_paths_rejected": cases.get("path_and_alias_mutations") is True,
        "every_deciding_tensor_mutation_rejected": cases.get("deciding_tensor_mutations") is True,
        "npz_payloads_have_npz_names": rebuild.get("npz_names_valid") is True,
        "records_rebuild_byte_stably": rebuild.get("byte_stable") is True,
        "device_host_disk_and_wall_time_recomputed": resource.get("recomputed") is True,
        "hard_resource_caps_hold": (
            gpu_hours <= RESOURCE_CAPS["hard_gpu_hours"]
            and wall_hours <= RESOURCE_CAPS["hard_wall_clock_hours"]),
    }
    return _measurements(
        "R", passes, {name: passes[name] for name in passes},
        "Fresh-process rebuild, mutation outcomes, and measured resource totals were checked."), [
            (True, gpu_hours * 3600.0, wall_hours * 3600.0)]


def analyze_stage(request_path):
    request, resolved, _protocol = load_request(request_path)
    stage = request["stage"]
    if stage not in STAGE_BY_ID:
        raise ValueError("request names an unknown gate-shape stage")
    if set(resolved) != set(RAW_REQUIRED[stage]):
        raise ValueError("request evidence roles disagree with the closed stage schema")
    root = Path(request["artifact_root"])
    if stage == "Q":
        measurements, resources = _q_measurements(resolved)
    elif stage == "T":
        measurements, resources = _t_measurements(resolved, root)
    elif stage == "TR":
        measurements, resources = _tr_measurements(resolved, root)
    elif stage == "G":
        measurements, resources = _g_measurements(resolved, root)
    elif stage == "C":
        measurements, resources = _c_measurements(resolved, root)
    elif stage == "I":
        measurements, resources = _i_measurements(resolved, root)
    elif stage == "A":
        measurements, resources = _a_measurements(resolved, root)
    else:
        measurements, resources = _r_measurements(resolved)
    expected = set(STAGE_BY_ID[stage]["required_checks"])
    if set(measurements) != expected:
        raise RuntimeError("analyzer checks disagree with the protocol")
    passed = all(row["passed"] for row in measurements.values())
    gpu_seconds = sum(row[1] for row in resources if row and row[0])
    wall_seconds = max([row[2] for row in resources if row and row[0]] or [0.0])
    return seal_record({
        "campaign_id": request["campaign_id"],
        "stage": stage,
        "request_sha256": request["request_sha256"],
        "measurements": measurements,
        "recomputed_gpu_hours": gpu_seconds / 3600.0,
        "recomputed_wall_clock_hours": wall_seconds / 3600.0,
        "decision": "CONFIRMED" if passed else "NOT_CONFIRMED",
    })


def analyze_campaign(request_paths):
    records = [analyze_stage(path) for path in request_paths]
    if not records:
        raise ValueError("campaign analysis requires at least one request")
    campaigns = {row["campaign_id"] for row in records}
    if len(campaigns) != 1:
        raise ValueError("stage requests belong to different campaigns")
    by_stage = {row["stage"]: row for row in records}
    if len(by_stage) != len(records):
        raise ValueError("campaign analysis contains a duplicate stage")
    confirmed = set()
    summary = {}
    for stage in STAGES:
        record = by_stage.get(stage["id"])
        dependencies = all(value in confirmed for value in stage["dependencies"])
        if record is None:
            verdict = "PENDING"
        elif dependencies and record["decision"] == "CONFIRMED":
            verdict = "CONFIRMED"
            confirmed.add(stage["id"])
        else:
            verdict = "NOT_CONFIRMED"
        summary[stage["id"]] = {
            "dependencies_satisfied": dependencies,
            "verdict": verdict,
            "record_sha256": record.get("record_sha256") if record else None,
        }
    resource_record = by_stage.get("R")
    if resource_record is None:
        gpu_hours = sum(row["recomputed_gpu_hours"] for row in records)
        wall_hours = sum(
            row["recomputed_wall_clock_hours"] for row in records)
    else:
        gpu_hours = resource_record["recomputed_gpu_hours"]
        wall_hours = resource_record["recomputed_wall_clock_hours"]

    return seal_record({
        "campaign_id": next(iter(campaigns)),
        "stage": "campaign",
        "summary": summary,
        "decision": "CONFIRMED" if len(confirmed) == len(STAGES) else "INCOMPLETE",
        "recomputed_gpu_hours": gpu_hours,
        "recomputed_wall_clock_hours": wall_hours,
    })


def _parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    stage = sub.add_parser("stage")
    stage.add_argument("--request", required=True)
    stage.add_argument("--output", required=True)
    campaign = sub.add_parser("campaign")
    campaign.add_argument("--request", action="append", required=True)
    campaign.add_argument("--output", required=True)
    verify = sub.add_parser("verify-record")
    verify.add_argument("--record", required=True)
    return parser


def main():
    arguments = _parser().parse_args()
    if arguments.command == "stage":
        write_json_atomic(arguments.output, analyze_stage(arguments.request))
    elif arguments.command == "campaign":
        write_json_atomic(arguments.output, analyze_campaign(arguments.request))
    else:
        if not verify_record(_json(arguments.record)):
            raise SystemExit(1)
        print("gate-shape record: verified")


if __name__ == "__main__":
    main()
