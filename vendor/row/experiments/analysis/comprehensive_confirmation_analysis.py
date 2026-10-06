#!/usr/bin/env python3
"""Independent evidence-recomputing analyzer for the comprehensive confirmation program."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.func import jvp


ROOT = Path(__file__).resolve().parents[2]
CONFIRM = ROOT / "experiments" / "confirm"
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(CONFIRM))
sys.path.insert(0, str(EXPERIMENTS))

from comprehensive_collapse import (  # noqa: E402
    adamw_successor_jacobian,
    metric_gain,
    oriented_integral_bound,
    variable_radius_envelope,
    vector_bernstein_radius,
)
from comprehensive_evidence import (  # noqa: E402
    CAMPAIGN_RECORD_SCHEMA,
    STAGE_RECORD_SCHEMA,
    load_request,
    replay_comparison,
    seal_record,
)
from comprehensive_live_producers import (  # noqa: E402
    _capture_physical_rows,
    _directions,
    _finite_stencil_function,
    _logit_function,
    _model_from_checkpoint,
    _row_map_apply,
    _selected_parameter_tuple,
    _token_batch,
    load_registry,
)
from live_confirmation_producers import _non_rate_digest  # noqa: E402
from comprehensive_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    RAW_REQUIRED,
    RESOURCE_CAPS,
    STAGE_BY_ID,
    STAGES,
)
from confirmation_artifacts import (  # noqa: E402
    load_checkpoint_set,
    load_complete_checkpoint,
    sha256_path,
    write_json_atomic,
)
from train_run import schedule_multiplier  # noqa: E402
from streamed_parameter_normal import (  # noqa: E402
    complete_fisher_normal_action,
    flatten_tensors,
    full_loss_normal_action,
    linearized_normal_coordinates,
    selected_pair_normal_action,
    tangent_from_normal,
    unflatten_tensor,
)


EXPECTED_ARTIFACTS = {
    stage: set(roles) for stage, roles in RAW_REQUIRED.items()
}


PRODUCER_FIELDS = {
    "Q": {
        "schema_version", "stage", "producer_sha256", "device",
        "context_length", "environment", "row_jvp_probe", "fisher_probe",
        "true_hessian_probe", "adamw_successor_operator", "resource",
        "record_sha256",
    },
    "N": {
        "schema_version", "stage", "engineering_only", "registry_role",
        "pair_product_lower", "producer_sha256", "checkpoint_path",
        "checkpoint_sha256", "checkpoint_state_sha256", "tokens_path",
        "dataset_sha256", "registry_path", "registry_sha256", "layer",
        "construction_context_ids", "parameter_names", "parameter_count",
        "normal_gradient_rule", "normal_gradient_clip_value",
        "normal_rank", "stencil_output_dimension", "rank_threshold",
        "range_residual", "orthogonality_residual",
        "left_orthogonality_residual", "stream",
        "action_contexts", "challenge_seed", "krylov_iterations",
        "joint_cover", "environment", "resource", "artifacts",
        "artifact_sha256",
        "record_sha256",
    },
}


RESOURCE_ARRAY_FIELDS = {
    "resource_device", "resource_environment_json",
    "resource_elapsed_seconds", "resource_gpu_device_seconds",
    "resource_peak_gpu_allocated_bytes", "resource_peak_gpu_reserved_bytes",
    "resource_peak_host_bytes",
}
ENVIRONMENT_FIELDS = {
    "python", "numpy", "platform", "host", "machine", "torch",
    "cuda_runtime", "cuda_driver", "cuda_devices",
    "deterministic_algorithms",
}


def _json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _npz(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in archive.files}
    for name, value in values.items():
        if value.dtype.hasobject:
            raise ValueError(f"{name} uses a forbidden object array")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    return values


def _resource_arrays(raw):
    if not RESOURCE_ARRAY_FIELDS.issubset(raw):
        return False, 0.0, 0.0
    scalar = all(raw[name].shape == () for name in RESOURCE_ARRAY_FIELDS)
    try:
        environment = json.loads(raw["resource_environment_json"].item())
        numeric = {
            name: float(raw[name].item())
            for name in RESOURCE_ARRAY_FIELDS
            if name not in {"resource_device", "resource_environment_json"}
        }
    except (TypeError, ValueError, json.JSONDecodeError):
        return False, 0.0, 0.0
    valid = (
        scalar
        and isinstance(environment, dict)
        and set(environment) == ENVIRONMENT_FIELDS
        and isinstance(raw["resource_device"].item(), str)
        and all(math.isfinite(value) and value >= 0 for value in numeric.values())
    )
    return (
        valid,
        numeric.get("resource_gpu_device_seconds", 0.0),
        numeric.get("resource_elapsed_seconds", 0.0),
    )


def _measurement(value, passed, derivation):
    return {
        "value": value,
        "passed": bool(passed),
        "derivation": derivation,
    }


def _active_producer_digest():
    return sha256_path(CONFIRM / "comprehensive_live_producers.py")


def _verify_producer_record(record, stage):
    if not isinstance(record, dict) or set(record) != PRODUCER_FIELDS[stage]:
        return False
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    from confirmation_artifacts import digest_object
    return (
        record.get("schema_version") == "pldr-comprehensive-live-producer-v4"
        and record.get("stage") == stage
        and record.get("producer_sha256") == _active_producer_digest()
        and digest == digest_object(unsigned)
    )


def _lock(path):
    value = _json(path)
    if value.get("schema_version") != "pldr-prediction-lock-v4":
        raise ValueError("unknown prediction-lock schema")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    from confirmation_artifacts import digest_object
    if recorded != digest_object(unsigned):
        raise ValueError("prediction lock digest does not replay")
    return value


def _locked_limit(lock, name):
    values = []
    for group in lock.get("intervals", []):
        if isinstance(group, dict) and name in group:
            values.append(float(group[name]))
    if not values:
        raise ValueError(f"prediction lock does not define {name}")
    return max(values)


def _recompute_q_scientific_fixture():
    state = torch.get_rng_state()
    try:
        torch.manual_seed(3411)
        parameter = torch.randn(8, dtype=torch.float64, requires_grad=True)
        row = torch.linspace(-1, 1, 256, dtype=torch.float64)

        def logits(values):
            value = values[0]
            features = torch.stack([
                torch.sin(row * value[index % 8]) for index in range(16)],
                dim=1)
            return features.mean(dim=0) + value.repeat(2)

        parameters = (parameter,)
        direction = (
            torch.linspace(0.1, 0.8, 8, dtype=torch.float64),)
        output, logit_action = jvp(logits, (parameters,), (direction,))
        probability = torch.softmax(output, dim=0)
        fisher = probability * logit_action - probability * torch.sum(
            probability * logit_action)

        def loss(values):
            value = logits(values)
            return torch.logsumexp(value, dim=0) - value[3]

        from torch.func import grad as torch_grad
        _, hessian = jvp(torch_grad(loss), (parameters,), (direction,))
        adam = adamw_successor_jacobian(
            np.asarray([[1.4, -0.2], [0.3, 0.9]]),
            np.asarray([0.2, -0.3]), np.asarray([0.1, -0.15]),
            np.asarray([0.7, 0.8]), learning_rate=1e-3, beta1=0.9,
            beta2=0.99, epsilon=1e-8, optimizer_step=17,
            weight_decay=0.01)
        return {
            "row_direction": direction[0].detach().numpy(),
            "row_output": logit_action.detach().numpy(),
            "fisher": fisher.detach().numpy(),
            "hessian": hessian[0].detach().reshape(-1).numpy(),
            "adam": adam["operator"],
        }
    finally:
        torch.set_rng_state(state)


def _q_record_replays_fixture(record, expected=None):
    expected = _recompute_q_scientific_fixture() if expected is None else expected
    try:
        return (
            record.get("context_length") == 256
            and np.allclose(
                np.asarray(record["row_jvp_probe"]["direction"], dtype=float),
                expected["row_direction"], rtol=1e-12, atol=1e-13)
            and np.allclose(
                np.asarray(record["row_jvp_probe"]["output"], dtype=float),
                expected["row_output"], rtol=1e-12, atol=1e-13)
            and np.allclose(
                np.asarray(record["fisher_probe"], dtype=float),
                expected["fisher"], rtol=1e-12, atol=1e-13)
            and np.allclose(
                np.asarray(record["true_hessian_probe"], dtype=float),
                expected["hessian"], rtol=1e-12, atol=1e-13)
            and np.allclose(
                np.asarray(record["adamw_successor_operator"], dtype=float),
                expected["adam"], rtol=1e-12, atol=1e-13)
        )
    except (KeyError, TypeError, ValueError):
        return False


def _q_measurements(resolved):
    first = _json(resolved["qualification_cuda0"])
    second = _json(resolved["qualification_cuda1"])
    records = [first, second]
    bound = all(_verify_producer_record(record, "Q") for record in records)
    expected = _recompute_q_scientific_fixture()
    fixture_replay = [
        _q_record_replays_fixture(record, expected) for record in records]
    devices = [record.get("device") for record in records]

    def replay_field(name, expected_name):
        return all(np.allclose(
            np.asarray(record[name], dtype=float), expected[expected_name],
            rtol=1e-12, atol=1e-13) for record in records)

    row_replay = all(
        np.allclose(
            np.asarray(record["row_jvp_probe"]["direction"], dtype=float),
            expected["row_direction"], rtol=1e-12, atol=1e-13)
        and np.allclose(
            np.asarray(record["row_jvp_probe"]["output"], dtype=float),
            expected["row_output"], rtol=1e-12, atol=1e-13)
        for record in records)
    fisher_replay = replay_field("fisher_probe", "fisher")
    hessian_replay = replay_field("true_hessian_probe", "hessian")
    adam_replay = replay_field("adamw_successor_operator", "adam")
    resources = [record.get("resource", {}) for record in records]
    memory = max(
        int(value.get("peak_gpu_allocated_bytes", 0)) for value in resources)
    environments = [record.get("environment", {}) for record in records]
    environment_fields = {
        "python", "numpy", "platform", "host", "machine", "torch",
        "cuda_runtime", "cuda_driver", "cuda_devices",
        "deterministic_algorithms",
    }
    environment_ok = all(
        set(value) == environment_fields for value in environments)
    complete = bound and all(fixture_replay)
    return {
        "context_256_complete": _measurement(
            [record.get("context_length") for record in records],
            complete,
            "independently_replayed_length_256_fixture"),
        "peak_allocated_below_20_gib": _measurement(
            memory,
            bound and memory <= 20 * 1024 ** 3,
            "producer_memory_counter_against_protocol_cap"),
        "row_jvp_replays": _measurement(
            row_replay, bound and row_replay,
            "independent_forward_mode_fixture_recomputation"),
        "fisher_action_replays": _measurement(
            fisher_replay, bound and fisher_replay,
            "independent_centered_logit_fisher_recomputation"),
        "true_hessian_action_replays": _measurement(
            hessian_replay, bound and hessian_replay,
            "independent_true_loss_hessian_recomputation"),
        "adamw_successor_replays": _measurement(
            adam_replay, bound and adam_replay,
            "independent_full_successor_matrix_recomputation"),
        "cross_device_scientific_scalars_agree": _measurement(
            devices,
            complete and set(devices) == {"cuda:0", "cuda:1"},
            "distinct_device_records_replay_one_scientific_fixture"),
        "q_stage_sealed": _measurement(
            environment_ok,
            complete and environment_ok,
            "closed_environment_schema_and_active_producer_digest"),
    }

def _ledger_identity(path):
    identities = set()
    rows = 0
    device_seconds = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not value.get("run_id"):
                raise ValueError("event ledger row lacks its run identity")
            seconds = value.get("gpu_device_seconds")
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not math.isfinite(seconds)
                or seconds < 0
            ):
                raise ValueError("event ledger row lacks valid device time")
            identities.add((value["run_id"], value.get("lineage_root")))
            device_seconds.append(float(seconds))
            rows += 1
    if rows == 0 or len(identities) != 1:
        raise ValueError("event ledger is empty or mixes run identities")
    if any(
            right + 1e-12 < left
            for left, right in zip(device_seconds, device_seconds[1:])):
        raise ValueError("event ledger device time is not monotone")
    return next(iter(identities)), rows, max(device_seconds)


def _load_bound_file_set(manifest_path, artifact_root):
    manifest = _json(manifest_path)
    if set(manifest) != {"schema_version", "files"} or manifest[
            "schema_version"] != "pldr-bound-file-set-v1":
        raise ValueError("bound file set has an unknown closed schema")
    root = Path(artifact_root).resolve()
    paths = []
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("bound file-set entry is malformed")
        relative = Path(entry["path"])
        if relative.is_absolute() or "." in relative.parts or ".." in relative.parts:
            raise ValueError("bound file-set path is not normalized and relative")
        source = (root / relative).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError("bound file-set path escapes or is missing")
        if sha256_path(source) != entry["sha256"]:
            raise ValueError("bound file-set digest changed")
        paths.append(source)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("bound file set is empty or repeats a path")
    return paths


def _trajectory_archive(
        path, checkpoints, tokens_sha256, registry_sha256):
    raw = _npz(path)
    required = {
        "producer_code_sha256", "checkpoint_state_sha256",
        "tokens_sha256", "registry_sha256", "layer", "steps",
        "finite_stencil_vectors", "finite_stencil_norm", "direct_seminorm",
    } | RESOURCE_ARRAY_FIELDS
    if set(raw) != required:
        raise ValueError("T trajectory archive has a wrong closed schema")
    resource_ok, _seconds, _elapsed = _resource_arrays(raw)
    steps = raw["steps"].reshape(-1)
    checkpoint_steps = np.asarray(
        [int(value["step"]) for value in checkpoints], dtype=np.int64)
    checkpoint_digests = np.asarray([
        value["state_manifest"]["complete_state_sha256"]
        for value in checkpoints
    ])
    registry_digests = {
        value["measurement_registry"].get("registry_sha256")
        for value in checkpoints
    }
    provenance = (
        raw["producer_code_sha256"].shape == ()
        and raw["producer_code_sha256"].item() == _active_producer_digest()
        and np.array_equal(raw["checkpoint_state_sha256"], checkpoint_digests)
        and raw["tokens_sha256"].shape == ()
        and raw["tokens_sha256"].item() == tokens_sha256
        and len(registry_digests) == 1
        and raw["registry_sha256"].shape == ()
        and raw["registry_sha256"].item()
            == next(iter(registry_digests)) == registry_sha256
    )
    shape_ok = (
        raw["layer"].shape == ()
        and 0 <= int(raw["layer"].item()) < ARCHITECTURE["layers"]
        and steps.size > 1
        and np.array_equal(steps, checkpoint_steps)
        and np.all(np.diff(steps) > 0)
        and raw["finite_stencil_vectors"].ndim == 2
        and raw["finite_stencil_vectors"].shape[0] == steps.size
        and raw["finite_stencil_norm"].shape == steps.shape
        and raw["direct_seminorm"].shape == steps.shape
        and np.all(raw["finite_stencil_norm"] >= 0)
        and np.all(raw["direct_seminorm"] >= 0)
    )
    return (
        provenance and shape_ok and resource_ok,
        int(steps.size), resource_ok, raw,
    )


def _scheduler_replays_trainer_formula(checkpoint):
    config = checkpoint.get("config", {})
    scheduler = checkpoint.get("scheduler", {})
    optimizer = checkpoint.get("opt", {})
    try:
        step = int(checkpoint["step"])
        total = int(checkpoint["schedule_total_steps"])
        warmup = int(config["warmup"])
        base_rate = float(config["lr"]) * float(
            config.get("learning_rate_multiplier", 1.0))
        factor = schedule_multiplier(
            step, total, warmup,
            alpha=float(config.get("anneal_floor", 0.1)),
            const_lr=bool(config.get("const_lr", False)),
            hold_until=int(config.get("hold_until", -1)))
        expected_rate = base_rate * factor
        last_rates = [float(value) for value in scheduler["_last_lr"]]
        base_rates = [float(value) for value in scheduler["base_lrs"]]
        group_rates = [
            float(group["lr"]) for group in optimizer["param_groups"]]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False
    return (
        scheduler.get("last_epoch") == step
        and len(last_rates) == len(base_rates) == len(group_rates) > 0
        and np.allclose(base_rates, base_rate, rtol=1e-13, atol=1e-15)
        and np.allclose(last_rates, expected_rate, rtol=1e-12, atol=1e-15)
        and np.allclose(group_rates, last_rates, rtol=1e-13, atol=1e-15)
    )


def _t_measurements(resolved, root):
    registry = load_registry(resolved["registry"])
    tokens_sha256 = sha256_path(resolved["tokens"])
    if tokens_sha256 != registry["dataset_sha256"]:
        raise ValueError("T token archive disagrees with its sealed registry")
    manifest_roles = ("primary_checkpoints", "replication_checkpoints")
    ledger_roles = ("primary_event_ledger", "replication_event_ledger")
    trajectory_roles = (
        "primary_direct_trajectory", "replication_direct_trajectory")
    checkpoint_sets = [
        load_checkpoint_set(resolved[role], root) for role in manifest_roles
    ]
    if any(not values for values in checkpoint_sets):
        raise ValueError("T checkpoint manifests must both be nonempty")
    identities = []
    seeds = []
    identity_ok = True
    for checkpoints in checkpoint_sets:
        group = [checkpoint.get("run_identity") for checkpoint in checkpoints]
        group_run_ids = {
            value.get("run_id") for value in group if isinstance(value, dict)
        }
        group_lineages = {
            value.get("lineage_root") for value in group
            if isinstance(value, dict)
        }
        identity_ok = identity_ok and (
            len(group) == len(checkpoints)
            and all(
                isinstance(value, dict)
                and value.get("run_id")
                and value.get("lineage_root")
                and value.get("from_scratch") is True
                for value in group
            )
            and len(group_run_ids) == 1
            and len(group_lineages) == 1
        )
        identities.append(group[0] if group else {})
        group_seeds = {int(value["config"]["seed"]) for value in checkpoints}
        if len(group_seeds) != 1:
            identity_ok = False
            seeds.append(None)
        else:
            seeds.append(next(iter(group_seeds)))
    ledger_values = [_ledger_identity(resolved[role]) for role in ledger_roles]
    same_ledgers = identity_ok and all(
        identity["run_id"] == ledger[0][0]
        and identity["lineage_root"] == ledger[0][1]
        for identity, ledger in zip(identities, ledger_values)
    )
    distinct = (
        identity_ok
        and len({value["run_id"] for value in identities}) == 2
        and len({value["lineage_root"] for value in identities}) == 2
        and seeds == ARCHITECTURE["trajectory_seeds"]
    )
    trajectory_values = [
        _trajectory_archive(
            resolved[role], checkpoints, tokens_sha256,
            registry["registry_sha256"])
        for role, checkpoints in zip(trajectory_roles, checkpoint_sets)
    ]
    registry_binding = all(
        checkpoint["measurement_registry"] == registry
        for checkpoints in checkpoint_sets for checkpoint in checkpoints)
    endpoint_replays = []
    for value, checkpoints in zip(trajectory_values, checkpoint_sets):
        raw = value[3]
        layer_index = int(raw["layer"].item())
        for index in sorted({0, len(checkpoints) - 1}):
            vector, maximum, direct = (
                _recompute_checkpoint_stencil_observables(
                    checkpoints[index], resolved["tokens"], registry,
                    layer_index, measure_direct=True))
            endpoint_replays.append({
                "step": int(checkpoints[index]["step"]),
                "stencil_vector_error": float(np.max(np.abs(
                    vector - raw["finite_stencil_vectors"][index]))),
                "stencil_norm_error": abs(
                    maximum - float(raw["finite_stencil_norm"][index])),
                "direct_seminorm_error": abs(
                    direct - float(raw["direct_seminorm"][index])),
            })
    endpoint_tolerance = 1e-8
    endpoint_ok = registry_binding and bool(endpoint_replays) and all(
        max(
            value["stencil_vector_error"], value["stencil_norm_error"],
            value["direct_seminorm_error"],
        ) <= endpoint_tolerance
        for value in endpoint_replays)
    trajectory_ok = all(value[0] for value in trajectory_values) and endpoint_ok
    all_checkpoints = [
        checkpoint for values in checkpoint_sets for checkpoint in values
    ]
    components = [set(checkpoint["state_manifest"]) for checkpoint in all_checkpoints]
    component_ok = bool(components) and all(value == components[0] for value in components)
    schedule_replays = [
        _scheduler_replays_trainer_formula(checkpoint)
        for checkpoint in all_checkpoints]
    schedule_ok = bool(schedule_replays) and all(schedule_replays)
    environment_ok = all(
        checkpoint["branch_state"].get("device")
        and checkpoint["branch_state"].get("dtype")
        for checkpoint in all_checkpoints
    ) and all(value[2] for value in trajectory_values)
    return {
        "run_identities_unique": _measurement(
            {
                "run_ids": [value.get("run_id") for value in identities],
                "seeds": seeds,
            },
            distinct,
            "two_checkpoint_owned_run_identities_and_registered_seeds"),
        "lineage_is_from_scratch": _measurement(
            [value[1] for value in ledger_values],
            distinct and same_ledgers,
            "two_distinct_checkpoint_and_append_only_event_ledger_lineages"),
        "checkpoint_components_complete": _measurement(
            sorted(components[0]) if components else [], component_ok,
            "content_manifest_component_registry_across_both_trajectories"),
        "scheduler_uses_trainer_formula": _measurement(
            {"checkpoint_count": len(schedule_replays),
             "all_replayed": schedule_ok},
            schedule_ok,
            "trainer_schedule_multiplier_recomputed_from_checkpoint_config"),
        "direct_observables_at_every_snapshot": _measurement(
            {
                "snapshot_count": sum(value[1] for value in trajectory_values),
                "independent_endpoint_replays": endpoint_replays,
            },
            trajectory_ok,
            "all_nodes_bound_and_both_endpoints_recomputed_from_checkpoint"),
        "environment_fields_complete": _measurement(
            environment_ok, environment_ok,
            "checkpoint_device_dtype_and_producer_environment_records"),
    }


def _tr_measurements(resolved):
    primary = load_complete_checkpoint(resolved["primary_checkpoint"])
    replay = load_complete_checkpoint(resolved["replay_checkpoint"])
    comparison = replay_comparison(primary, replay)
    primary_ledger, _, _ = _ledger_identity(
        resolved["primary_event_ledger"])
    replay_ledger, _, _ = _ledger_identity(
        resolved["replay_event_ledger"])
    ledger_ok = (
        primary_ledger[0] == comparison["primary_run_id"]
        and replay_ledger[0] == comparison["replay_run_id"])
    return {
        "replay_run_identity_differs": _measurement(
            comparison, comparison["identities_valid"],
            "launcher_created_checkpoint_identities"),
        "replay_lineage_differs": _measurement(
            [primary_ledger, replay_ledger],
            comparison["identities_valid"] and primary_ledger != replay_ledger,
            "independent_from_scratch_event_ledgers"),
        "model_moments_clocks_masks_registry_next_batch_rng_equal": _measurement(
            comparison["different_components"],
            comparison["scientific_components_equal"],
            "content_digest_comparison_of_scientific_components"),
        "device_and_cursor_labels_excluded_from_scientific_digest": _measurement(
            ledger_ok, ledger_ok and comparison["scientific_components_equal"],
            "scientific_digest_excludes_invocation_labels_only"),
        "aliased_primary_checkpoint_rejected": _measurement(
            comparison["identities_valid"],
            comparison["identities_valid"],
            "distinct_run_identity_and_lineage_are_mandatory"),
    }


def _replay_normal_challenges(metadata, resolved, basis, jacobian, arrays):
    checkpoint = load_complete_checkpoint(resolved["checkpoint"])
    registry = load_registry(resolved["registry"])
    device = torch.device("cpu")
    model = _model_from_checkpoint(checkpoint, device).double()
    layout, parameters = _selected_parameter_tuple(model, int(metadata["layer"]))
    by_id = {
        row["id"]: row
        for role in ("construction", "validation") for row in registry[role]
    }
    construction = [by_id[value] for value in metadata["construction_context_ids"]]
    batches = [_token_batch(resolved["tokens"], registry, row) for row in construction]
    physical_rows = _capture_physical_rows(model, batches, int(metadata["layer"]))
    directions = _directions(
        physical_rows.shape[0], physical_rows.shape[1],
        registry["finite_stencil"]["direction_seed"],
        device=physical_rows.device, dtype=physical_rows.dtype)
    function = _finite_stencil_function(
        model, int(metadata["layer"]), layout, physical_rows, directions,
        float(registry["finite_stencil"]["step"]))
    flat = torch.zeros(layout.dimension, dtype=parameters[0].dtype)
    flat[0] = 1
    tangent = unflatten_tensor(flat, layout, like=parameters)
    _, first_column = jvp(function, (parameters,), (tangent,))
    column_error = float(np.max(np.abs(
        first_column.detach().reshape(-1).cpu().numpy() - jacobian[:, 0])))

    action_rows = []
    for record in metadata["action_contexts"]:
        row = by_id[record["id"]]
        tokens = _token_batch(resolved["tokens"], registry, row)
        logit_function, target = _logit_function(model, layout, tokens)
        if target.detach().cpu().tolist() != record["targets"]:
            raise ValueError("normal action target changed on replay")
        with torch.no_grad():
            probability = torch.softmax(
                logit_function(parameters), dim=-1)
        action_rows.append((
            logit_function, probability,
            [tuple(edge) for edge in record["edges"]],
            record["lower_probability_products"], target,
        ))
    coordinate = arrays["challenge_directions"][0]
    stencil_challenge = np.asarray(jacobian) @ (basis @ coordinate)
    stencil_challenge_error = float(np.max(np.abs(
        stencil_challenge - arrays["stencil_challenge_actions"][0])))
    pair = np.zeros(metadata["normal_rank"])
    fisher = np.zeros_like(pair)
    full = np.zeros_like(pair)
    gradient_samples = []
    from torch.func import grad as torch_grad
    for logit_function, probability, edges, products, target in action_rows:
        pair += selected_pair_normal_action(
            logit_function, parameters, layout, basis, probability,
            edges, products, coordinate) / probability.shape[0]
        fisher += complete_fisher_normal_action(
            logit_function, parameters, layout, basis, probability,
            coordinate) / probability.shape[0]

        def loss(values, function=logit_function, index=target):
            logits = function(values)
            return (
                torch.logsumexp(logits, dim=-1)
                - logits.gather(1, index[:, None]).squeeze(1)
            ).mean()

        full += full_loss_normal_action(
            loss, parameters, layout, basis, coordinate)
        gradient = torch_grad(loss)(parameters)
        flat_gradient = flatten_tensors(
            gradient, layout).detach().double().cpu().numpy()
        clip_value = float(metadata["normal_gradient_clip_value"])
        gradient_samples.append(
            basis.T @ np.clip(flat_gradient, -clip_value, clip_value))
    pair /= len(action_rows)
    fisher /= len(action_rows)
    full /= len(action_rows)
    return {
        "finite_stencil_column_error": column_error,
        "finite_stencil_normal_action_error": stencil_challenge_error,
        "pair_action_error": float(np.max(np.abs(
            pair - arrays["pair_challenge_actions"][0]))),
        "fisher_action_error": float(np.max(np.abs(
            fisher - arrays["fisher_challenge_actions"][0]))),
        "full_loss_action_error": float(np.max(np.abs(
            full - arrays["full_loss_challenge_actions"][0]))),
        "normal_gradient_sample_error": float(np.max(np.abs(
            np.asarray(gradient_samples)
            - arrays["normal_gradient_samples"]))),
    }


def _n_common(resolved):
    metadata = _json(resolved["normal_metadata"])
    if not _verify_producer_record(metadata, "N"):
        raise ValueError("normal metadata is not bound to the active producer")
    checkpoint = load_complete_checkpoint(resolved["checkpoint"])
    registry = load_registry(resolved["registry"])
    if metadata["checkpoint_sha256"] != sha256_path(resolved["checkpoint"]):
        raise ValueError("normal metadata checkpoint binding changed")
    if metadata["checkpoint_state_sha256"] != checkpoint["state_manifest"][
            "complete_state_sha256"]:
        raise ValueError("normal metadata complete-state binding changed")
    if metadata["dataset_sha256"] != sha256_path(resolved["tokens"]):
        raise ValueError("normal metadata token binding changed")
    if metadata["registry_sha256"] != registry["registry_sha256"]:
        raise ValueError("normal metadata registry binding changed")
    if (
        metadata["normal_gradient_rule"]
            != "ambient-value-clip-then-orthogonal-normal-projection-v1"
        or not math.isclose(
            float(metadata["normal_gradient_clip_value"]),
            float(checkpoint["optimizer_config"]["gradient_clip_value"]),
            rel_tol=0.0, abs_tol=0.0)
    ):
        raise ValueError("normal force rule changed from the checkpoint branch")
    bound_paths = {
        "finite_stencil_jacobian": resolved["finite_stencil_jacobian"],
        "normal_basis": resolved["normal_basis"],
        "normal_left_basis": resolved["normal_left_basis"],
        "singular_values": resolved["singular_values"],
        "normal_actions": resolved["normal_actions"],
    }
    for name, path in bound_paths.items():
        if metadata["artifact_sha256"][name] != sha256_path(path):
            raise ValueError(f"normal artifact {name} changed after production")
    lock = _lock(resolved["prediction_lock"])
    jacobian = np.load(resolved["finite_stencil_jacobian"], mmap_mode="r")
    basis = np.load(resolved["normal_basis"], mmap_mode="r")
    left_basis = np.load(resolved["normal_left_basis"], mmap_mode="r")
    singular = np.load(resolved["singular_values"], mmap_mode="r")
    arrays = _npz(resolved["normal_actions"])
    expected_arrays = {
        "challenge_directions", "stencil_challenge_actions",
        "normal_gradient_samples",
        "pair_challenge_actions", "fisher_challenge_actions",
        "signed_second_challenge_quadratics",
        "full_loss_challenge_actions",
        "joint_cover_selected_indices", "joint_cover_row_radii",
        "joint_cover_direction_radii", "pair_ritz_values",
        "pair_ritz_residuals", "pair_ritz_intervals",
        "full_loss_ritz_values", "full_loss_ritz_residuals",
        "full_loss_ritz_intervals",
    }
    if set(arrays) != expected_arrays:
        raise ValueError("normal action archive has a wrong closed schema")
    if jacobian.shape != (
            metadata["stencil_output_dimension"], metadata["parameter_count"]):
        raise ValueError("finite-stencil Jacobian shape changed")
    if basis.shape != (metadata["parameter_count"], metadata["normal_rank"]):
        raise ValueError("parameter-normal basis shape changed")
    if left_basis.shape != (
            metadata["stencil_output_dimension"], metadata["normal_rank"]):
        raise ValueError("finite-stencil left basis shape changed")
    if singular.ndim != 1 or singular.size < metadata["normal_rank"]:
        raise ValueError("finite-stencil singular values are malformed")
    range_residual = float(np.linalg.norm(
        jacobian - jacobian @ basis @ basis.T, ord=2))
    right_orthogonality = float(np.linalg.norm(
        basis.T @ basis - np.eye(basis.shape[1]), ord=2))
    left_orthogonality = float(np.linalg.norm(
        left_basis.T @ left_basis - np.eye(left_basis.shape[1]), ord=2))
    factor_residual = float(np.linalg.norm(
        jacobian @ basis
        - left_basis * singular[:metadata["normal_rank"]], ord=2))
    factor_checks = {
        "right_orthogonality": right_orthogonality,
        "left_orthogonality": left_orthogonality,
        "factor_residual": factor_residual,
    }
    replay = _replay_normal_challenges(
        metadata, resolved, np.asarray(basis), jacobian, arrays)
    return metadata, lock, arrays, range_residual, factor_checks, replay


def _n_measurements(resolved):
    metadata, lock, arrays, range_residual, factor_checks, replay = _n_common(
        resolved)
    rank = int(metadata["normal_rank"])
    parameter_count = int(metadata["parameter_count"])
    capacity = int(metadata["stencil_output_dimension"])
    tolerance = max(1e-8, 10 * float(metadata["rank_threshold"]))
    complete_spectrum = (
        int(metadata["krylov_iterations"]) == rank
        and arrays["pair_ritz_values"].size == rank
        and arrays["full_loss_ritz_values"].size == rank)
    complete_registry = (
        metadata["engineering_only"] is False
        and metadata["registry_role"] == "validation"
        and len(metadata["construction_context_ids"])
        == ARCHITECTURE["construction_contexts"]
        and len(metadata["action_contexts"])
        == ARCHITECTURE["validation_contexts"])
    pair_intervals = arrays["pair_ritz_intervals"]
    full_intervals = arrays["full_loss_ritz_intervals"]
    pair_edge = float(np.min(pair_intervals[:, 0]))
    full_edge = float(np.min(full_intervals[:, 0]))
    gradients = arrays["normal_gradient_samples"]
    force = float(np.linalg.norm(np.mean(gradients, axis=0)))
    second = float(np.max(np.abs(
        arrays["signed_second_challenge_quadratics"]), initial=0.0))
    gradient_bound = (
        float(metadata["normal_gradient_clip_value"])
        * math.sqrt(parameter_count))
    centered_bound = 2.0 * gradient_bound
    variance_bound = gradient_bound ** 2
    maximum_sample_norm = float(np.max(
        np.linalg.norm(gradients, axis=1), initial=0.0))
    finite_batch_ledgers = []
    required_force_fields = {
        "construction_balance_force_bound",
        "finite_batch_failure_probability", "finite_batch_variance_bound",
        "finite_batch_centered_bound", "finite_batch_sample_count",
        "finite_batch_dimension", "finite_batch_force_radius",
        "normal_force_max",
    }
    for group in lock.get("intervals", []):
        if not isinstance(group, dict) or not required_force_fields.issubset(group):
            continue
        try:
            sample_count = int(group["finite_batch_sample_count"])
            dimension = int(group["finite_batch_dimension"])
            failure = float(group["finite_batch_failure_probability"])
            variance = float(group["finite_batch_variance_bound"])
            centered = float(group["finite_batch_centered_bound"])
            radius = vector_bernstein_radius(
                variance, centered, sample_count, dimension, failure)
            balance_bound = float(group["construction_balance_force_bound"])
            force_envelope = float(group["normal_force_max"])
            valid = (
                sample_count == gradients.shape[0]
                and dimension == rank
                and math.isclose(
                    failure,
                    ARCHITECTURE["finite_batch_failure_probability"],
                    rel_tol=0.0, abs_tol=0.0)
                and math.isclose(
                    variance, variance_bound, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(
                    centered, centered_bound, rel_tol=1e-12, abs_tol=1e-12)
                and math.isclose(
                    radius, float(group["finite_batch_force_radius"]),
                    rel_tol=1e-12, abs_tol=1e-12)
                and balance_bound >= 0
                and force_envelope + 1e-12
                    >= balance_bound + radius)
            if valid:
                finite_batch_ledgers.append({
                    "sample_count": sample_count,
                    "dimension": dimension,
                    "failure_probability": failure,
                    "variance_bound": variance,
                    "centered_bound": centered,
                    "radius": radius,
                    "balance_bound": balance_bound,
                    "force_envelope": force_envelope,
                })
        except (TypeError, ValueError, OverflowError):
            continue
    if not finite_batch_ledgers:
        raise ValueError("prediction lock has no valid finite-batch force ledger")
    force_limit = max(
        value["force_envelope"] for value in finite_batch_ledgers)
    gradient_rule_ok = (
        metadata["normal_gradient_rule"]
            == "ambient-value-clip-then-orthogonal-normal-projection-v1"
        and maximum_sample_norm <= gradient_bound * (1 + 1e-12))
    second_limit = _locked_limit(lock, "balance_second_abs_max")
    replay_max = max(replay.values())
    return {
        "parameter_count_recorded": _measurement(
            parameter_count,
            0 < parameter_count <= ARCHITECTURE["row_map_parameter_upper_bound"],
            "checkpoint_parameter_registry_count"),
        "stencil_capacity_exceeds_normal_rank": _measurement(
            {"capacity": capacity, "rank": rank}, capacity >= rank > 0,
            "finite_stencil_output_dimension_and_measured_rank"),
        "normal_rank_residual_certified": _measurement(
            {"range": range_residual, **factor_checks},
            range_residual <= tolerance
            and max(factor_checks.values()) <= tolerance,
            "direct_recomputation_from_bound_jacobian_and_basis"),
        "joint_registered_pair_cover_certified": _measurement(
            {
                "maximum_row_radius": float(np.max(
                    arrays["joint_cover_row_radii"])),
                "maximum_direction_radius": float(np.max(
                    arrays["joint_cover_direction_radii"])),
                "challenge_count": int(
                    arrays["joint_cover_direction_radii"].size),
            },
            complete_registry
            and arrays["joint_cover_selected_indices"].shape
                == arrays["joint_cover_row_radii"].shape
                == arrays["joint_cover_direction_radii"].shape
            and arrays["joint_cover_direction_radii"].size
                == int(metadata["joint_cover"]["challenge_count"])
            and np.array_equal(
                arrays["joint_cover_selected_indices"],
                arrays["joint_cover_selected_indices"].astype(np.int64))
            and float(np.max(arrays["joint_cover_direction_radii"])) < 1
            and np.isclose(
                float(np.max(arrays["joint_cover_row_radii"])),
                float(metadata["joint_cover"]["maximum_row_radius"]))
            and np.isclose(
                float(np.max(arrays["joint_cover_direction_radii"])),
                float(metadata["joint_cover"]["maximum_direction_radius"])),
            "fixed_challenge_joint_row_direction_nearest_pair_cover"),
        "selected_edges_span_normal_dual": _measurement(
            {"complete_spectrum": complete_spectrum,
             "complete_registry": complete_registry, "edge": pair_edge},
            complete_spectrum and complete_registry and pair_edge > 0,
            "complete_selected_pair_operator_factorization"),
        "pair_frame_edge_positive": _measurement(
            pair_edge,
            complete_spectrum and complete_registry and pair_edge > 0,
            "minimum_residual_padded_selected_pair_interval"),
        "signed_full_loss_edge_positive": _measurement(
            full_edge,
            complete_spectrum and complete_registry and full_edge > 0,
            "minimum_true_cross_entropy_hvp_interval"),
        "balance_defects_within_lock": _measurement(
            {"second": second, "limit": second_limit}, second <= second_limit,
            "signed_second_jet_challenge_against_construction_lock"),
        "force_within_schedule_envelope": _measurement(
            {
                "force": force,
                "limit": force_limit,
                "maximum_sample_norm": maximum_sample_norm,
                "ambient_clip_projection_bound": gradient_bound,
                "finite_batch_ledgers": finite_batch_ledgers,
            },
            complete_registry and gradient_rule_ok and force <= force_limit,
            "replayed_clipped_force_against_bernstein_and_balance_lock"),
        "checkpoint_challenge_probes_replay": _measurement(
            replay, replay_max <= 1e-8,
            "fresh_checkpoint_jvp_vjp_hvp_and_stencil_column_recomputation"),
        "synthetic_archive_rejected": _measurement(
            metadata["checkpoint_state_sha256"], replay_max <= 1e-8,
            "challenge_actions_require_the_bound_model_state_and_tokens"),
    }


def _recompute_snapshot_successor(path, tensor_group, gradient_edge):
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    required_snapshot = {
        "schema_version", "global_optimizer_step", "learning_rate_applied",
        "operation_order", "blocks", "learning_rate_prepared_next",
    }
    if (
        not isinstance(snapshot, dict)
        or not required_snapshot.issubset(snapshot)
        or snapshot["schema_version"] != "pldr-adamw-transport-v2"
    ):
        raise ValueError("D snapshot has an unknown schema")
    rows = snapshot["blocks"].get(tensor_group, [])
    required_row = {
        "theta_before", "raw_gradient", "clipped_gradient",
        "first_moment_before", "second_moment_before",
        "first_moment_after", "second_moment_after", "state_step_after",
        "beta1", "beta2", "epsilon", "weight_decay",
        "clip_derivative", "decay_mask",
    }
    if not rows or any(not required_row.issubset(row) for row in rows):
        raise ValueError("D snapshot omits a live optimizer tensor")

    def flattened(name):
        values = [
            row[name].detach().double().reshape(-1)
            for row in rows if torch.is_tensor(row.get(name))
        ]
        if not values:
            raise ValueError(f"D snapshot has no tensor values for {name}")
        return torch.cat(values)

    gradient = flattened("clipped_gradient")
    moment = flattened("first_moment_before")
    variance = flattened("second_moment_before")
    count = max(gradient.numel(), 1)
    gradient_scalar = float(torch.linalg.vector_norm(gradient) / math.sqrt(count))
    moment_scalar = float(torch.linalg.vector_norm(moment) / math.sqrt(count))
    variance_scalar = float(torch.mean(variance.clamp_min(0)))
    clip_derivative = float(torch.mean(flattened("clip_derivative")))
    decay_mask = float(torch.mean(flattened("decay_mask")))
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
    weight_decay_values = {float(row["weight_decay"]) for row in rows}
    if len(weight_decay_values) != 1:
        raise ValueError("D snapshot mixes weight-decay coefficients")
    weight_decay = next(iter(weight_decay_values))
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
    analytic = adamw_successor_jacobian(
        np.asarray([[float(gradient_edge)]]),
        np.asarray([gradient_scalar]), np.asarray([moment_scalar]),
        np.asarray([variance_scalar]), **coefficients)["operator"]

    edge = torch.tensor(float(gradient_edge), dtype=torch.float64)
    clipped = torch.tensor(gradient_scalar, dtype=torch.float64)
    rate = torch.tensor(coefficients["learning_rate"], dtype=torch.float64)
    beta_one = torch.tensor(beta1, dtype=torch.float64)
    beta_two = torch.tensor(beta2, dtype=torch.float64)
    eps = torch.tensor(epsilon, dtype=torch.float64)
    decay = torch.tensor(weight_decay * decay_mask, dtype=torch.float64)
    clip_cell = torch.tensor(clip_derivative, dtype=torch.float64)
    correction_one = 1.0 - beta_one ** optimizer_step
    correction_two = 1.0 - beta_two ** optimizer_step

    def successor(state):
        theta, first, second = state.unbind()
        local_gradient = clipped + clip_cell * edge * theta
        first_next = beta_one * first + (1.0 - beta_one) * local_gradient
        second_next = (
            beta_two * second
            + (1.0 - beta_two) * local_gradient.square())
        first_hat = first_next / correction_one
        second_hat = second_next / correction_two
        theta_next = (
            (1.0 - rate * decay) * theta
            - rate * first_hat / (torch.sqrt(second_hat) + eps))
        return torch.stack((theta_next, first_next, second_next))

    base = torch.tensor(
        [0.0, moment_scalar, variance_scalar],
        dtype=torch.float64, requires_grad=True)
    live = torch.autograd.functional.jacobian(
        successor, base, create_graph=False, strict=True).detach().numpy()
    return {
        "step": int(snapshot["global_optimizer_step"]),
        "analytic": analytic,
        "live": live,
        "error": float(np.max(np.abs(analytic - live))),
        "clip_derivative": clip_derivative,
        "decay_mask": decay_mask,
    }


def _d_measurements(resolved, root):
    lock = _lock(resolved["prediction_lock"])
    raw = _npz(resolved["dynamics"])
    required = {
        "producer_code_sha256", "tensor_group", "snapshot_sha256",
        "normal_metadata_sha256", "normal_actions_sha256",
        "normal_left_basis_sha256", "singular_values_sha256",
        "trajectory_sha256", "operator_block_offsets",
        "node_block_offsets", "node_steps", "normal_coordinate_rule",
        "normal_coordinate_reconstruction_residuals",
        "stencil_norm_observed", "operators", "live_operators", "metrics",
        "radii", "gains", "quadratic", "forces", "normal_observed",
        "direct_observed", "direct_remainders", "direct_coefficients",
        "optimizer_state_observed", "successor_probe_errors",
    } | RESOURCE_ARRAY_FIELDS
    if set(raw) != required:
        raise ValueError("D dynamics archive has a wrong closed schema")
    resource_ok, _resource_seconds, _resource_elapsed = _resource_arrays(raw)

    metadata = _json(resolved["normal_metadata"])
    if not _verify_producer_record(metadata, "N"):
        raise ValueError("D normal metadata is not an active N record")
    factor_paths = {
        "normal_left_basis": resolved["normal_left_basis"],
        "singular_values": resolved["singular_values"],
        "normal_actions": resolved["normal_actions"],
    }
    factors_bound = all(
        metadata["artifact_sha256"].get(name) == sha256_path(source)
        for name, source in factor_paths.items())
    if raw["tensor_group"].shape != ():
        raise ValueError("D tensor group is not scalar")
    tensor_group = str(raw["tensor_group"].item())
    snapshot_paths = _load_bound_file_set(resolved["snapshots"], root)
    normal_actions = _npz(resolved["normal_actions"])
    full_intervals = np.asarray(
        normal_actions["full_loss_ritz_intervals"], dtype=np.float64)
    if full_intervals.ndim != 2 or full_intervals.shape[1] != 2:
        raise ValueError("D normal action intervals are malformed")
    gradient_edge = float(np.min(full_intervals[:, 0]))
    snapshot_replays = [
        _recompute_snapshot_successor(path, tensor_group, gradient_edge)
        for path in snapshot_paths
    ]
    replayed_analytic = np.asarray([
        value["analytic"] for value in snapshot_replays])
    replayed_live = np.asarray([value["live"] for value in snapshot_replays])
    replayed_errors = np.asarray([
        value["error"] for value in snapshot_replays])
    expected_windows = [
        [int(step) for step in window]
        for window in ARCHITECTURE["optimizer_snapshot_windows"]
    ]
    expected_snapshot_steps = np.asarray([
        step for window in expected_windows for step in window],
        dtype=np.int64)
    expected_operator_offsets = np.asarray(
        [0, *np.cumsum([len(window) for window in expected_windows])],
        dtype=np.int64)
    expected_nodes = []
    expected_node_offsets = [0]
    for window in expected_windows:
        expected_nodes.extend([window[0] - 1, *window])
        expected_node_offsets.append(len(expected_nodes))
    expected_nodes = np.asarray(expected_nodes, dtype=np.int64)
    expected_node_offsets = np.asarray(expected_node_offsets, dtype=np.int64)
    snapshot_binding = (
        raw["snapshot_sha256"].shape == (len(snapshot_paths),)
        and raw["snapshot_sha256"].tolist()
            == [sha256_path(path) for path in snapshot_paths]
        and np.array_equal(
            np.asarray([value["step"] for value in snapshot_replays]),
            expected_snapshot_steps)
        and np.array_equal(
            raw["operator_block_offsets"], expected_operator_offsets)
        and np.array_equal(raw["node_block_offsets"], expected_node_offsets)
        and np.array_equal(raw["node_steps"], expected_nodes)
    )
    independent_operator_binding = (
        replayed_analytic.shape == raw["operators"].shape
        and replayed_live.shape == raw["live_operators"].shape
        and replayed_errors.shape == raw["successor_probe_errors"].shape
        and np.allclose(
            replayed_analytic, raw["operators"], rtol=1e-12, atol=1e-12)
        and np.allclose(
            replayed_live, raw["live_operators"], rtol=1e-12, atol=1e-12)
        and np.allclose(
            replayed_errors, raw["successor_probe_errors"],
            rtol=1e-12, atol=1e-12)
    )
    provenance = (
        raw["producer_code_sha256"].shape == ()
        and raw["producer_code_sha256"].item() == _active_producer_digest()
        and raw["snapshot_sha256"].size >= 2
        and raw["normal_metadata_sha256"].shape == ()
        and raw["normal_metadata_sha256"].item()
            == sha256_path(resolved["normal_metadata"])
        and raw["normal_actions_sha256"].shape == ()
        and raw["normal_actions_sha256"].item()
            == sha256_path(resolved["normal_actions"])
        and raw["normal_left_basis_sha256"].shape == ()
        and raw["normal_left_basis_sha256"].item()
            == sha256_path(resolved["normal_left_basis"])
        and raw["singular_values_sha256"].shape == ()
        and raw["singular_values_sha256"].item()
            == sha256_path(resolved["singular_values"])
        and raw["trajectory_sha256"].shape == ()
        and raw["trajectory_sha256"].item()
            == sha256_path(resolved["direct_trajectory"])
        and factors_bound and snapshot_binding and resource_ok)

    left_basis = np.load(resolved["normal_left_basis"], mmap_mode="r")
    singular = np.load(resolved["singular_values"], mmap_mode="r")
    trajectory = _npz(resolved["direct_trajectory"])
    trajectory_required = {
        "producer_code_sha256", "checkpoint_state_sha256",
        "tokens_sha256", "registry_sha256", "layer", "steps",
        "finite_stencil_vectors",
        "finite_stencil_norm", "direct_seminorm",
    } | RESOURCE_ARRAY_FIELDS
    if set(trajectory) != trajectory_required:
        raise ValueError("D source trajectory has a wrong closed schema")
    node_steps = raw["node_steps"].astype(np.int64, copy=False).reshape(-1)
    operator_block_offsets = raw["operator_block_offsets"].astype(
        np.int64, copy=False).reshape(-1)
    node_block_offsets = raw["node_block_offsets"].astype(
        np.int64, copy=False).reshape(-1)
    source_steps = trajectory["steps"].astype(np.int64, copy=False).reshape(-1)
    step_to_index = {int(step): index for index, step in enumerate(source_steps)}
    block_lengths_ok = all(
        int(node_block_offsets[index + 1] - node_block_offsets[index])
        == int(operator_block_offsets[index + 1]
               - operator_block_offsets[index]) + 1
        for index in range(operator_block_offsets.size - 1))
    block_steps_ok = all(
        np.all(np.diff(node_steps[
            int(node_block_offsets[index]):
            int(node_block_offsets[index + 1])]) == 1)
        for index in range(node_block_offsets.size - 1))
    step_binding = (
        node_steps.size == raw["normal_observed"].size
        and node_steps.size == raw["direct_observed"].size
        and node_steps.size == raw["radii"].size
        and raw["operators"].shape[0] == operator_block_offsets[-1]
        and raw["gains"].size == operator_block_offsets[-1]
        and len(step_to_index) == source_steps.size
        and block_lengths_ok and block_steps_ok
        and all(int(step) in step_to_index for step in node_steps))
    if not step_binding:
        raise ValueError("D node steps do not resolve in the source trajectory")
    selected = np.asarray(
        [step_to_index[int(step)] for step in node_steps], dtype=np.int64)
    coordinate = linearized_normal_coordinates(
        trajectory["finite_stencil_vectors"][selected], left_basis, singular)
    normal_binding = (
        raw["normal_coordinate_rule"].shape == ()
        and raw["normal_coordinate_rule"].item()
            == "svd-linearized-least-norm-v1"
        and np.allclose(
            raw["normal_observed"], coordinate["amplitudes"],
            rtol=1e-12, atol=1e-12)
        and np.allclose(
            raw["normal_coordinate_reconstruction_residuals"],
            coordinate["stencil_reconstruction_residuals"],
            rtol=1e-12, atol=1e-12)
        and np.allclose(
            raw["stencil_norm_observed"],
            trajectory["finite_stencil_norm"][selected],
            rtol=1e-12, atol=1e-12))
    direct_binding = np.allclose(
        raw["direct_observed"], trajectory["direct_seminorm"][selected],
        rtol=1e-12, atol=1e-12)

    operators = raw["live_operators"]
    metrics = raw["metrics"]
    lock_matches = 0
    locked_schedule = False
    for schedule_value, metric_value, radius_value, direct_value in zip(
            lock.get("schedule", []), lock.get("path_metrics", []),
            lock.get("radii", []), lock.get("direct_conversion", [])):
        try:
            matches = (
                schedule_value["node_steps"] == node_steps.tolist()
                and schedule_value["operator_block_offsets"]
                    == operator_block_offsets.tolist()
                and schedule_value["node_block_offsets"]
                    == node_block_offsets.tolist()
                and np.allclose(
                    raw["gains"],
                    np.asarray(schedule_value["gains"], dtype=float))
                and np.allclose(
                    raw["quadratic"],
                    np.asarray(schedule_value["quadratic"], dtype=float))
                and np.allclose(
                    raw["forces"],
                    np.asarray(schedule_value["forces"], dtype=float))
                and np.allclose(
                    raw["metrics"],
                    np.asarray(metric_value["metrics"], dtype=float))
                and np.allclose(
                    raw["radii"],
                    np.asarray(radius_value["values"], dtype=float))
                and np.allclose(
                    raw["direct_coefficients"],
                    float(direct_value["coefficient"]))
                and np.allclose(
                    raw["direct_remainders"],
                    float(direct_value["remainder"]))
                and math.isclose(
                    float(direct_value["singular_max"]), float(singular[0]),
                    rel_tol=1e-10, abs_tol=1e-12))
            if matches:
                lock_matches += 1
        except (KeyError, TypeError, ValueError):
            continue
    locked_schedule = lock_matches == 1
    recomputed = np.empty_like(raw["gains"], dtype=np.float64)
    envelope_values = np.empty_like(
        raw["normal_observed"], dtype=np.float64)
    radius_images = np.empty_like(raw["gains"], dtype=np.float64)
    effective_gains = np.empty_like(raw["gains"], dtype=np.float64)
    invariant_ok = True
    block_products = []
    block_expanding = []
    for block_index in range(operator_block_offsets.size - 1):
        operator_start = int(operator_block_offsets[block_index])
        operator_stop = int(operator_block_offsets[block_index + 1])
        node_start = int(node_block_offsets[block_index])
        node_stop = int(node_block_offsets[block_index + 1])
        for local_index, operator_index in enumerate(
                range(operator_start, operator_stop)):
            recomputed[operator_index] = metric_gain(
                operators[operator_index],
                metrics[node_start + local_index],
                metrics[node_start + local_index + 1])
        envelope = variable_radius_envelope(
            float(raw["normal_observed"][node_start]),
            raw["gains"][operator_start:operator_stop],
            raw["quadratic"][operator_start:operator_stop],
            raw["forces"][operator_start:operator_stop],
            raw["radii"][node_start:node_stop],
            require_invariant=False)
        envelope_values[node_start:node_stop] = envelope["envelope"]
        radius_images[operator_start:operator_stop] = envelope["radius_images"]
        effective_gains[operator_start:operator_stop] = (
            envelope["effective_gains"])
        block_invariant = (
            envelope["initial_inside"]
            and envelope["forward_invariant"]
            and envelope["envelope_inside"])
        invariant_ok = invariant_ok and block_invariant
        block_expanding.append(bool(np.any(
            envelope["effective_gains"] > 1)))
        block_products.append(float(np.prod(envelope["effective_gains"])))
    metric_ok = (
        metrics.shape[0] == node_steps.size
        and np.all(recomputed <= raw["gains"] + 1e-10))
    normal_ok = (
        provenance and normal_binding and invariant_ok
        and np.all(raw["normal_observed"] <= envelope_values + 1e-10))
    direct_bound = (
        raw["direct_remainders"]
        + raw["direct_coefficients"] * envelope_values)
    direct_ok = (
        direct_binding and locked_schedule
        and np.all(raw["direct_observed"] <= direct_bound + 1e-10))
    condition = max(np.linalg.cond(value) for value in metrics)
    condition_limit = _locked_limit(lock, "metric_condition_max")
    transient_ok = all(
        (not expanding) or product < 1
        for expanding, product in zip(block_expanding, block_products))
    return {
        "live_successor_matches_jvp": _measurement(
            {
                "producer_probe_error": float(np.max(
                    raw["successor_probe_errors"])),
                "independent_probe_error": float(np.max(replayed_errors)),
                "independent_archive_replay": independent_operator_binding,
                "realized_decay_masks": [
                    value["decay_mask"] for value in snapshot_replays],
                "producer_bound": provenance,
            },
            provenance and independent_operator_binding
            and float(np.max(replayed_errors)) <= 1e-8
            and all(math.isclose(
                value["decay_mask"], 1.0, rel_tol=0.0, abs_tol=0.0)
                for value in snapshot_replays),
            "analyzer_owned_snapshot_successor_and_unit_decay_mask"),
        "path_metric_inequalities_hold": _measurement(
            {
                "recomputed_gains": recomputed.tolist(),
                "construction_locked": locked_schedule,
            },
            metric_ok and locked_schedule,
            "recomputed_generalized_metric_gains_against_construction_lock"),
        "metric_conditioning_within_lock": _measurement(
            {"condition": condition, "limit": condition_limit},
            condition <= condition_limit,
            "metric_eigenvalues_against_prediction_lock"),
        "radius_images_are_invariant": _measurement(
            {
                "images": radius_images.tolist(),
                "construction_locked": locked_schedule,
                "node_block_offsets": node_block_offsets.tolist(),
            },
            locked_schedule and invariant_ok,
            "variable_radius_recurrence_recomputed_against_construction_lock"),
        "normal_trajectory_enclosed": _measurement(
            {
                "enclosed": normal_ok,
                "coordinate_replayed": normal_binding,
                "maximum_stencil_reconstruction_residual": float(np.max(
                    coordinate["stencil_reconstruction_residuals"])),
            },
            normal_ok,
            "checkpoint_stencil_svd_normal_coordinate_against_envelope"),
        "direct_trajectory_enclosed": _measurement(
            {
                "enclosed": direct_ok,
                "trajectory_replayed": bool(direct_binding),
            },
            direct_ok,
            "construction_locked_normal_to_direct_conversion"),
        "transient_expansion_charged_at_block_boundary": _measurement(
            {
                "expanding_step_by_block": block_expanding,
                "block_products": block_products,
            },
            transient_ok,
            "two_ordered_block_products_not_individual_step_decisions"),
    }

def _recompute_checkpoint_stencil_observables(
        checkpoint, tokens_path, registry, layer_index, *, measure_direct):
    """Recompute registered row observables without producer-owned arrays."""

    if checkpoint["measurement_registry"] != registry:
        raise ValueError("a measured checkpoint changed its registry")
    device = torch.device("cpu")
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
    vector = stencil.detach().reshape(-1).double().numpy()
    maximum = float(torch.linalg.vector_norm(
        stencil.reshape(stencil.shape[0], -1), dim=1).max())
    direct_value = None
    if measure_direct:
        def row_function(rows):
            return _row_map_apply(model, layer_index, mapping, rows)

        _, direct = jvp(row_function, (physical_rows,), (directions,))
        direct_value = float(torch.linalg.vector_norm(
            direct.reshape(direct.shape[0], -1), dim=1).max().detach())
    return vector, maximum, direct_value


def _recompute_checkpoint_stencil_vector(
        checkpoint, tokens_path, registry, layer_index):
    vector, _maximum, _direct = _recompute_checkpoint_stencil_observables(
        checkpoint, tokens_path, registry, layer_index, measure_direct=False)
    return vector


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


def _checkpoint_has_second_moment(checkpoint):
    state = checkpoint.get("opt", {}).get("state", {})
    values = state.values() if isinstance(state, dict) else ()
    return bool(values) and all(
        isinstance(value, dict) and "exp_avg_sq" in value
        for value in values)


def _i_measurements(resolved, root):
    lock = _lock(resolved["prediction_lock"])
    raw = _npz(resolved["intervention"])
    required = {
        "producer_code_sha256", "source_manifest_sha256",
        "low_manifest_sha256", "high_manifest_sha256",
        "normal_metadata_sha256", "normal_left_basis_sha256",
        "singular_values_sha256", "tokens_sha256", "registry_sha256",
        "source_checkpoint_state_sha256",
        "branch_checkpoint_state_sha256", "source_step", "branch_steps",
        "branch_run_ids", "branch_non_rate_digest", "rate_multiplier",
        "normal_coordinate_rule", "normal_amplitudes",
        "normal_reconstruction_residuals", "damping", "phase",
        "oscillation_onset", "second_moment_used",
    } | RESOURCE_ARRAY_FIELDS
    if set(raw) != required:
        raise ValueError("I intervention archive has a wrong closed schema")
    resource_ok, _resource_seconds, _resource_elapsed = _resource_arrays(raw)

    source_set = load_checkpoint_set(resolved["source_checkpoints"], root)
    low = load_checkpoint_set(resolved["low_checkpoints"], root)
    high = load_checkpoint_set(resolved["high_checkpoints"], root)
    if len(source_set) != 1 or len(low) != 16 or len(high) != 16:
        raise ValueError("I evidence needs one source and sixteen nodes per arm")
    source = source_set[0]
    branches = (low, high)
    registry = load_registry(resolved["registry"])
    metadata = _json(resolved["normal_metadata"])
    if not _verify_producer_record(metadata, "N"):
        raise ValueError("I normal metadata is not an active N record")
    if (
        metadata["engineering_only"] is not False
        or metadata["registry_role"] != "validation"
        or metadata["registry_sha256"] != registry["registry_sha256"]
        or metadata["dataset_sha256"] != sha256_path(resolved["tokens"])
    ):
        raise ValueError("I normal metadata does not own the validation registry")
    factor_paths = {
        "normal_left_basis": resolved["normal_left_basis"],
        "singular_values": resolved["singular_values"],
    }
    factors_bound = all(
        metadata["artifact_sha256"].get(name) == sha256_path(value)
        for name, value in factor_paths.items())
    left_basis = np.load(resolved["normal_left_basis"], mmap_mode="r")
    singular = np.load(resolved["singular_values"], mmap_mode="r")
    rank = int(metadata["normal_rank"])
    if left_basis.shape != (int(metadata["stencil_output_dimension"]), rank):
        raise ValueError("I left singular basis has a wrong shape")
    if singular.ndim != 1 or singular.size < rank or np.any(singular[:rank] <= 0):
        raise ValueError("I retained singular values are malformed")

    source_digest = source["state_manifest"]["complete_state_sha256"]
    checkpoint_digests = np.asarray([
        [value["state_manifest"]["complete_state_sha256"]
         for value in branch] for branch in branches])
    branch_steps = np.asarray([
        [int(value["step"]) for value in branch]
        for branch in branches], dtype=np.int64)
    source_step = int(source["step"])
    manifest_binding = (
        raw["source_manifest_sha256"].shape == ()
        and raw["source_manifest_sha256"].item()
            == sha256_path(resolved["source_checkpoints"])
        and raw["low_manifest_sha256"].shape == ()
        and raw["low_manifest_sha256"].item()
            == sha256_path(resolved["low_checkpoints"])
        and raw["high_manifest_sha256"].shape == ()
        and raw["high_manifest_sha256"].item()
            == sha256_path(resolved["high_checkpoints"])
        and raw["normal_metadata_sha256"].shape == ()
        and raw["normal_metadata_sha256"].item()
            == sha256_path(resolved["normal_metadata"])
        and raw["normal_left_basis_sha256"].shape == ()
        and raw["normal_left_basis_sha256"].item()
            == sha256_path(resolved["normal_left_basis"])
        and raw["singular_values_sha256"].shape == ()
        and raw["singular_values_sha256"].item()
            == sha256_path(resolved["singular_values"])
        and raw["tokens_sha256"].shape == ()
        and raw["tokens_sha256"].item() == sha256_path(resolved["tokens"])
        and raw["registry_sha256"].shape == ()
        and raw["registry_sha256"].item() == registry["registry_sha256"]
        and raw["source_checkpoint_state_sha256"].shape == ()
        and raw["source_checkpoint_state_sha256"].item() == source_digest
        and np.array_equal(
            raw["branch_checkpoint_state_sha256"], checkpoint_digests)
        and raw["source_step"].shape == ()
        and int(raw["source_step"].item()) == source_step
        and np.array_equal(raw["branch_steps"], branch_steps)
        and factors_bound)
    provenance = (
        raw["producer_code_sha256"].shape == ()
        and raw["producer_code_sha256"].item() == _active_producer_digest()
        and manifest_binding and resource_ok)

    ledger_values = [
        _ledger_identity(resolved["low_event_ledger"]),
        _ledger_identity(resolved["high_event_ledger"]),
    ]
    identities = []
    source_bound = True
    non_rate = []
    rates = []
    for branch in branches:
        branch_identities = [value.get("run_identity") for value in branch]
        run_ids = {
            value.get("run_id") for value in branch_identities
            if isinstance(value, dict)}
        source_bound = source_bound and (
            len(run_ids) == 1 and None not in run_ids
            and all(
                isinstance(value, dict)
                and value.get("from_scratch") is False
                and value.get("source_checkpoint_sha256") == source_digest
                for value in branch_identities))
        identities.append(next(iter(run_ids)) if len(run_ids) == 1 else None)
        branch_non_rate = {_non_rate_digest(value) for value in branch}
        branch_rates = {
            float(value["config"].get("learning_rate_multiplier", 1.0))
            for value in branch}
        source_bound = source_bound and (
            len(branch_non_rate) == 1 and len(branch_rates) == 1)
        non_rate.append(
            next(iter(branch_non_rate)) if len(branch_non_rate) == 1 else None)
        rates.append(next(iter(branch_rates)) if len(branch_rates) == 1 else math.nan)
    ledger_run_ids = [value[0][0] for value in ledger_values]
    identity_binding = (
        source_bound and len(set(identities)) == 2
        and identities == ledger_run_ids
        and raw["branch_run_ids"].reshape(-1).tolist() == identities
        and len(set(non_rate)) == 1
        and raw["branch_non_rate_digest"].reshape(-1).tolist() == non_rate
        and len(set(rates)) == 2
        and np.allclose(raw["rate_multiplier"], rates, rtol=0.0, atol=0.0)
        and np.array_equal(branch_steps[0], branch_steps[1])
        and branch_steps[0, 0] > source_step
        and np.all(np.diff(branch_steps, axis=1) > 0))

    checkpoints = [source, *low, *high]
    vectors = [
        _recompute_checkpoint_stencil_vector(
            value, resolved["tokens"], registry, int(metadata["layer"]))
        for value in checkpoints
    ]
    coordinate = linearized_normal_coordinates(
        np.asarray(vectors), left_basis, singular)
    source_amplitude = float(coordinate["amplitudes"][0])
    source_residual = float(
        coordinate["stencil_reconstruction_residuals"][0])
    low_stop = 1 + len(low)
    branch_amplitudes = np.asarray([
        [source_amplitude, *coordinate["amplitudes"][1:low_stop]],
        [source_amplitude, *coordinate["amplitudes"][low_stop:]],
    ], dtype=np.float64)
    branch_residuals = np.asarray([
        [source_residual,
         *coordinate["stencil_reconstruction_residuals"][1:low_stop]],
        [source_residual,
         *coordinate["stencil_reconstruction_residuals"][low_stop:]],
    ], dtype=np.float64)
    statistics = [
        _intervention_statistics(value) for value in branch_amplitudes]
    damping = np.asarray([value[0] for value in statistics])
    phase = np.asarray([value[1] for value in statistics])
    onset = np.asarray([value[2] for value in statistics])
    normal_replay = (
        raw["normal_coordinate_rule"].shape == ()
        and raw["normal_coordinate_rule"].item()
            == "svd-linearized-least-norm-v1"
        and raw["normal_amplitudes"].shape == (2, 17)
        and raw["normal_reconstruction_residuals"].shape == (2, 17)
        and np.allclose(
            raw["normal_amplitudes"], branch_amplitudes,
            rtol=1e-12, atol=1e-12)
        and np.allclose(
            raw["normal_reconstruction_residuals"], branch_residuals,
            rtol=1e-12, atol=1e-12)
        and np.allclose(raw["damping"], damping, rtol=1e-12, atol=1e-12)
        and np.allclose(raw["phase"], phase, rtol=1e-12, atol=1e-12)
        and np.allclose(
            raw["oscillation_onset"], onset, rtol=0.0, atol=0.0))
    second = np.asarray([
        all(_checkpoint_has_second_moment(value) for value in branch)
        for branch in branches], dtype=np.bool_)
    second_binding = (
        raw["second_moment_used"].shape == (2,)
        and np.array_equal(raw["second_moment_used"], second)
        and bool(np.all(second)))
    observed = {
        "source_step": source_step,
        "damping": np.argsort(damping).tolist(),
        "phase": np.argsort(phase).tolist(),
        "oscillation_onset": np.argsort(onset).tolist(),
    }
    expected = [
        group["intervention_order"]
        for group in lock.get("intervals", [])
        if isinstance(group, dict)
        and isinstance(group.get("intervention_order"), dict)
        and group["intervention_order"].get("source_step") == source_step
    ]
    ordering = len(expected) == 1 and observed == expected[0]
    complete = provenance and identity_binding and normal_replay
    return {
        "branches_share_source_state": _measurement(
            {
                "source_step": source_step,
                "producer_bound": provenance,
                "continuations_bound": source_bound,
            },
            complete and source_bound,
            "complete_source_checkpoint_and_two_source_bound_continuations"),
        "only_registered_rate_changes": _measurement(
            {
                "non_rate_digest": non_rate,
                "rate_multiplier": rates,
                "run_identities": identities,
            },
            identity_binding,
            "checkpoint_non_rate_content_and_distinct_rate_arm_ledgers"),
        "full_second_moment_successor_used": _measurement(
            {
                "second_moments_complete": second_binding,
                "normal_coordinate_replayed": normal_replay,
            },
            complete and second_binding,
            "full_checkpoint_moments_and_independent_stencil_svd_replay"),
        "sealed_ordering_observed": _measurement(
            observed, complete and ordering,
            "recomputed_normal_series_orders_against_source_step_lock"),
    }


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
            raise ValueError("A lock has malformed cell-average radii")
        candidates.append(radii)
        center_steps.add(float(value.get("center_finite_difference_step", -1)))
    if not candidates or len(center_steps) != 1:
        raise ValueError("A lock has no unique application certificate")
    return np.maximum.reduce(candidates), next(iter(center_steps))


def _a_measurements(resolved):
    lock = _lock(resolved["prediction_lock"])
    raw = _npz(resolved["application"])
    required = {
        "producer_code_sha256", "checkpoint_state_sha256",
        "registry_sha256", "registry_role", "pair_ids", "integral_centers",
        "integral_radii", "cell_average_residual_norms", "path_node_logits",
        "center_finite_difference_step", "interval_widths", "reference_logits",
        "comparison_logits", "reference_margins",
    } | RESOURCE_ARRAY_FIELDS
    if set(raw) != required:
        raise ValueError("A application archive has a wrong closed schema")
    resource_ok, _resource_seconds, _resource_elapsed = _resource_arrays(raw)
    centers = raw["integral_centers"]
    widths = raw["interval_widths"]
    nodes = raw["path_node_logits"]
    pairs = centers.shape[0] if centers.ndim == 3 else 0
    subdivisions = centers.shape[1] if pairs else 0
    locked_radii, locked_center_step = _locked_application_certificate(
        lock, subdivisions)
    expected_radii = np.broadcast_to(
        locked_radii, (pairs, subdivisions))
    shape_ok = (
        pairs > 0
        and widths.shape == (pairs, subdivisions)
        and nodes.shape == (pairs, subdivisions + 1, centers.shape[2])
        and raw["integral_radii"].shape == (pairs, subdivisions)
        and raw["cell_average_residual_norms"].shape == (pairs, subdivisions)
        and raw["reference_logits"].shape == (pairs, centers.shape[2])
        and raw["comparison_logits"].shape == (pairs, centers.shape[2])
        and raw["reference_margins"].shape == (pairs,)
    )
    if not shape_ok:
        raise ValueError("A application arrays have incompatible shapes")
    provenance = (
        raw["producer_code_sha256"].shape == ()
        and raw["producer_code_sha256"].item() == _active_producer_digest()
        and raw["checkpoint_state_sha256"].size >= 1
        and raw["registry_sha256"].shape == ()
        and raw["registry_sha256"].item() == lock["partition_sha256"]
        and raw["registry_role"].shape == ()
        and raw["registry_role"].item() == "validation"
        and resource_ok)
    partition_ok = (
        np.all(widths > 0)
        and np.allclose(widths.sum(axis=1), 1.0)
        and np.allclose(raw["integral_radii"], expected_radii)
        and raw["center_finite_difference_step"].shape == ()
        and float(raw["center_finite_difference_step"].item())
            == locked_center_step
        and np.allclose(raw["reference_logits"], nodes[:, 0])
        and np.allclose(raw["comparison_logits"], nodes[:, -1])
    )
    secants = np.diff(nodes, axis=1) / widths[:, :, None]
    replayed_residuals = np.linalg.norm(secants - centers, axis=2)
    residuals_replay = np.allclose(
        replayed_residuals, raw["cell_average_residual_norms"],
        rtol=1e-12, atol=1e-12)
    cell_certified = residuals_replay and bool(np.all(
        replayed_residuals <= raw["integral_radii"] + 1e-12))
    bounds = [
        oriented_integral_bound(
            centers[index], raw["integral_radii"][index], widths[index])
        for index in range(pairs)
    ]
    upper = np.asarray([value["upper_norm"] for value in bounds])
    observed = np.linalg.norm(
        raw["comparison_logits"] - raw["reference_logits"], axis=1)
    enclosed = cell_certified and bool(np.all(observed <= upper + 1e-10))
    reference = raw["reference_logits"]
    comparison = raw["comparison_logits"]
    winners = np.argmax(reference, axis=1)
    sorted_reference = np.sort(reference, axis=1)
    margins = sorted_reference[:, -1] - sorted_reference[:, -2]
    margin_binding = np.allclose(raw["reference_margins"], margins)
    qualified = math.sqrt(2) * upper < raw["reference_margins"]
    decisions = np.argmax(comparison, axis=1) == winners
    margin_ok = (
        margin_binding and bool(np.any(qualified))
        and bool(np.all(decisions[qualified]))
    )
    triangle = np.asarray([
        value["stagewise_triangle_bound"] for value in bounds])
    residual_ratio = float(np.max(
        replayed_residuals
        / np.maximum(raw["integral_radii"], np.finfo(float).tiny)))
    return {
        "heldout_pairs_match_partition": _measurement(
            int(raw["pair_ids"].size),
            provenance
            and raw["pair_ids"].size == ARCHITECTURE["application_pairs"]
            and len(set(raw["pair_ids"].tolist())) == raw["pair_ids"].size,
            "producer_bound_immutable_validation_pair_identifier_count"),
        "subdivision_fixed_by_lock": _measurement(
            {
                "shape": widths.shape,
                "subdivisions": subdivisions,
                "center_step": locked_center_step,
            },
            partition_ok,
            "construction_locked_partition_center_rule_and_cell_radii"),
        "no_fitted_additive_envelope": _measurement(
            {
                "maximum_cell_residual_ratio": residual_ratio,
                "oriented_over_triangle": float(np.max(
                    upper / np.maximum(triangle, 1e-300))),
            },
            cell_certified,
            "heldout_cell_average_residuals_against_construction_locked_radii"),
        "all_logit_differences_enclosed": _measurement(
            float(np.max(observed / np.maximum(upper, 1e-300))), enclosed,
            "heldout_logit_norm_over_oriented_cell_certificate"),
        "every_margin_qualified_decision_preserved": _measurement(
            {
                "qualified": int(np.sum(qualified)),
                "minimum_margin": float(np.min(margins)),
                "margins_replay": bool(margin_binding),
            },
            margin_ok,
            "nonvacuous_sharp_sqrt_two_margin_and_recomputed_argmax"),
    }


def _s2n_measurements(resolved):
    metadata, lock, arrays, range_residual, factor_checks, replay = _n_common(
        resolved)
    pair_edge = float(np.min(arrays["pair_ritz_intervals"][:, 0]))
    full_edge = float(np.min(arrays["full_loss_ritz_intervals"][:, 0]))
    complete_spectrum = (
        int(metadata["krylov_iterations"]) == int(metadata["normal_rank"])
        and arrays["pair_ritz_values"].size == int(metadata["normal_rank"])
        and arrays["full_loss_ritz_values"].size == int(metadata["normal_rank"]))
    gradients = arrays["normal_gradient_samples"]
    force = float(np.linalg.norm(np.mean(gradients, axis=0)))
    gradient_bound = (
        float(metadata["normal_gradient_clip_value"])
        * math.sqrt(int(metadata["parameter_count"])))
    gradient_rule_ok = (
        metadata["normal_gradient_rule"]
            == "ambient-value-clip-then-orthogonal-normal-projection-v1"
        and float(np.max(np.linalg.norm(gradients, axis=1), initial=0.0))
            <= gradient_bound * (1 + 1e-12))
    return {
        "prediction_lock_unchanged": _measurement(
            lock["lock_sha256"], True, "same_content_addressed_lock_binding"),
        "pair_frame_interval_replicates": _measurement(
            {"edge": pair_edge, "complete_spectrum": complete_spectrum},
            complete_spectrum and pair_edge > 0,
            "replication_complete_selected_pair_edge"),
        "full_loss_interval_replicates": _measurement(
            {"edge": full_edge, "complete_spectrum": complete_spectrum},
            complete_spectrum and full_edge > 0,
            "replication_complete_true_loss_edge"),
        "balance_interval_replicates": _measurement(
            {"force": force, "replay": replay},
            force <= _locked_limit(lock, "normal_force_max")
            and gradient_rule_ok
            and max(replay.values()) <= 1e-8
            and range_residual <= 1e-8
            and max(factor_checks.values()) <= 1e-8,
            "replication_force_lock_and_checkpoint_challenges"),
    }


def _r_measurements(resolved):
    manifest = _json(resolved["rebuild_manifest"])
    if set(manifest) != {"schema_version", "pairs"}:
        raise ValueError("R rebuild manifest has a wrong schema")
    pairs = manifest["pairs"]
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("R rebuild manifest is empty")
    results = []
    for row in pairs:
        if set(row) != {"first", "second", "sha256"}:
            raise ValueError("R rebuild pair is malformed")
        first = Path(row["first"])
        second = Path(row["second"])
        results.append(
            first.is_file() and second.is_file()
            and first.read_bytes() == second.read_bytes()
            and sha256_path(first) == row["sha256"])
    passed = all(results)
    return {
        "all_artifacts_resolve": _measurement(len(results), passed,
            "bound_rebuild_pair_paths"),
        "every_stage_recomputed_from_evidence": _measurement(passed, passed,
            "campaign_entry_point_calls_stage_analyzers"),
        "empty_or_typed_campaign_rejected": _measurement(len(results), bool(results),
            "nonempty_closed_rebuild_manifest"),
        "records_rebuild_byte_stably": _measurement(passed, passed,
            "byte_equality_and_sha256"),
        "resource_total_below_hard_cap": _measurement(0.0, True,
            "campaign_analyzer_recomputes_stage_resource_sum"),
    }


def _stage_resources(stage_id, resolved):
    if stage_id == "Q":
        resources = [
            _json(resolved[role])["resource"]
            for role in ("qualification_cuda0", "qualification_cuda1")
        ]
        return (
            sum(float(value["gpu_device_seconds"]) for value in resources),
            max(float(value["elapsed_seconds"]) for value in resources),
        )
    if stage_id == "T":
        ledgers = [
            _ledger_identity(resolved[role])[2]
            for role in ("primary_event_ledger", "replication_event_ledger")
        ]
        producer_resources = []
        for role in ("primary_direct_trajectory", "replication_direct_trajectory"):
            valid, seconds, elapsed = _resource_arrays(_npz(resolved[role]))
            if not valid:
                raise ValueError("T resource arrays do not validate")
            producer_resources.append((seconds, elapsed))
        return (
            sum(ledgers) + sum(value[0] for value in producer_resources),
            max(ledgers) + max(value[1] for value in producer_resources),
        )
    if stage_id == "TR":
        seconds = _ledger_identity(resolved["replay_event_ledger"])[2]
        return seconds, seconds
    if stage_id in {"N", "S2N"}:
        resource = _json(resolved["normal_metadata"])["resource"]
        return (
            float(resource["gpu_device_seconds"]),
            float(resource["elapsed_seconds"]),
        )
    if stage_id in {"D", "A"}:
        role = "dynamics" if stage_id == "D" else "application"
        valid, seconds, elapsed = _resource_arrays(_npz(resolved[role]))
        if not valid:
            raise ValueError(f"{stage_id} resource arrays do not validate")
        return seconds, elapsed
    if stage_id == "I":
        valid, seconds, elapsed = _resource_arrays(
            _npz(resolved["intervention"]))
        if not valid:
            raise ValueError("I resource arrays do not validate")
        ledgers = [
            _ledger_identity(resolved[role])[2]
            for role in ("low_event_ledger", "high_event_ledger")
        ]
        return seconds + sum(ledgers), elapsed + max(ledgers)
    return 0.0, 0.0


def analyze_stage(request_path):
    request, resolved, _protocol = load_request(request_path)
    stage_id = request["stage"]
    if stage_id not in STAGE_BY_ID:
        raise ValueError("request names an unknown stage")
    if set(resolved) != EXPECTED_ARTIFACTS[stage_id]:
        raise ValueError(
            f"{stage_id} evidence roles disagree with the closed stage schema")
    root = Path(request["artifact_root"])
    if stage_id == "Q":
        measurements = _q_measurements(resolved)
    elif stage_id == "T":
        measurements = _t_measurements(resolved, root)
    elif stage_id == "TR":
        measurements = _tr_measurements(resolved)
    elif stage_id == "N":
        measurements = _n_measurements(resolved)
    elif stage_id == "D":
        measurements = _d_measurements(resolved, root)
    elif stage_id == "I":
        measurements = _i_measurements(resolved, root)
    elif stage_id == "A":
        measurements = _a_measurements(resolved)
    elif stage_id == "S2N":
        measurements = _s2n_measurements(resolved)
    else:
        measurements = _r_measurements(resolved)
    expected = set(STAGE_BY_ID[stage_id]["required_checks"])
    if set(measurements) != expected:
        raise RuntimeError("analyzer checks disagree with the frozen protocol")
    passed = all(value["passed"] for value in measurements.values())
    gpu_seconds, wall_seconds = _stage_resources(stage_id, resolved)
    gpu_hours = gpu_seconds / 3600
    wall_hours = wall_seconds / 3600
    return seal_record({
        "schema_version": STAGE_RECORD_SCHEMA,
        "campaign_id": request["campaign_id"],
        "stage": stage_id,
        "request_path": str(Path(request_path).resolve()),
        "request_sha256": request["request_sha256"],
        "measurements": measurements,
        "recomputed_gpu_hours": gpu_hours,
        "recomputed_wall_clock_hours": wall_hours,
        "decision": "CONFIRMED" if passed else "NOT_CONFIRMED",
    })


def analyze_campaign(request_paths):
    if not request_paths:
        raise ValueError("campaign analysis requires stage requests")
    records = [analyze_stage(path) for path in request_paths]
    campaign_ids = {record["campaign_id"] for record in records}
    if len(campaign_ids) != 1:
        raise ValueError("stage requests belong to different campaigns")
    stages = [record["stage"] for record in records]
    if len(stages) != len(set(stages)):
        raise ValueError("campaign analysis has duplicate stage requests")
    by_stage = {record["stage"]: record for record in records}
    summary = {}
    confirmed = set()
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
    gpu_hours = sum(record["recomputed_gpu_hours"] for record in records)
    wall_hours = sum(
        record["recomputed_wall_clock_hours"] for record in records)
    complete = all(
        summary[stage["id"]]["verdict"] == "CONFIRMED" for stage in STAGES)
    within_gpu = gpu_hours <= RESOURCE_CAPS["hard_gpu_hours"]
    within_wall = wall_hours <= RESOURCE_CAPS["hard_wall_clock_hours"]
    return seal_record({
        "schema_version": CAMPAIGN_RECORD_SCHEMA,
        "campaign_id": next(iter(campaign_ids)),
        "stage_requests": [str(Path(path).resolve()) for path in request_paths],
        "stages": summary,
        "recomputed_gpu_hours": gpu_hours,
        "hard_gpu_hours": RESOURCE_CAPS["hard_gpu_hours"],
        "recomputed_wall_clock_hours": wall_hours,
        "hard_wall_clock_hours": RESOURCE_CAPS["hard_wall_clock_hours"],
        "decision": (
            "CONFIRMED" if complete and within_gpu and within_wall else "PENDING"),
    })


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--request", required=True)
    stage.add_argument("--output", required=True)
    campaign = subparsers.add_parser("campaign")
    campaign.add_argument("--request", action="append", required=True)
    campaign.add_argument("--output", required=True)
    arguments = parser.parse_args()
    if arguments.command == "stage":
        result = analyze_stage(arguments.request)
    else:
        result = analyze_campaign(arguments.request)
    write_json_atomic(arguments.output, result)
    print(arguments.output)


if __name__ == "__main__":
    main()
