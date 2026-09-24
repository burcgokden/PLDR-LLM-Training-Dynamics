#!/usr/bin/env python3
"""Live producers for finite-increment observable balance and gate control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
from typing import Any


# This must be present before importing torch or initializing CUDA.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.func import functional_call, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT))

from confirm.confirmation_artifacts import sha256_path  # noqa: E402
from confirm.finite_increment_live import (  # noqa: E402
    load_checkpoint,
    optimizer_from_checkpoint,
    registered_tokens,
)
from confirm.observable_balance import balance_metrics, cell_evaluable  # noqa: E402
from confirm.observable_balance_specs import (  # noqa: E402
    ANALYSIS_SCHEMA,
    ARCHITECTURE,
    CAMPAIGN_ID,
    CONSTRUCTION_SCHEMA,
    DECISION_POLICY,
    DIRECTION_CLASSES,
    HORIZON_GRID,
    LOCK_SCHEMA,
    NUMERICAL_POLICY,
    PREDICTION_SCHEMA,
    QUALIFICATION_SCHEMA,
    RESOURCE_BUDGET,
    STRATUM_POLICY,
    STRATUM_SCHEMA,
    TRAJECTORIES,
    VALIDATION_SCHEMA,
    amplitudes,
)
from confirm.observable_cocycle import digest_object  # noqa: E402
from confirm.observable_cocycle_live import (  # noqa: E402
    Direction,
    TensorTree,
    _apply_direction,
    _make_directions,
    _named_state,
    _ordered_batches,
    _physical_observation,
    _prepare_device,
    _propagate,
)
from confirm.row_map_live import (  # noqa: E402
    _causal_mask,
    _gate_parameter_name,
    _loss,
    _model_from_raw,
)


SCIENTIFIC_FAILURE_MARKERS = (
    "nonfinite",
    "variance cone",
    "zero-variance face",
    "clipping boundary",
)


def _write_npz_compressed(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    if destination.suffix != ".npz":
        raise ValueError("observable-balance records must use .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(destination)


def _resource_record(device: torch.device, started: float) -> dict[str, np.ndarray]:
    elapsed = time.monotonic() - started
    if device.type != "cuda":
        raise RuntimeError("GPU-designated observable-balance science cannot use CPU")
    return {
        "resource_elapsed_seconds": np.asarray(elapsed),
        "resource_reserved_device_seconds": np.asarray(elapsed),
        "resource_peak_gpu_allocated_bytes": np.asarray(
            int(torch.cuda.max_memory_allocated(device)), dtype=np.int64
        ),
        "resource_peak_gpu_reserved_bytes": np.asarray(
            int(torch.cuda.max_memory_reserved(device)), dtype=np.int64
        ),
        "resource_peak_host_bytes": np.asarray(
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * (1024 if sys.platform != "darwin" else 1),
            dtype=np.int64,
        ),
    }


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]).copy() for name in archive.files}


def _load_measurement_registry(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    if not value.get("construction") or not value.get("validation"):
        raise ValueError("measurement registry is empty")
    return value


def _load_inputs(
    arguments: argparse.Namespace,
    horizon: int,
) -> tuple[
    torch.device,
    Path,
    dict[str, Any],
    torch.Tensor,
    list[torch.Tensor],
    np.ndarray,
]:
    device = _prepare_device(arguments.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("a GPU-designated science node cannot fall back to CPU")
    torch.backends.cuda.matmul.allow_tf32 = bool(NUMERICAL_POLICY["allow_tf32"])
    torch.set_float32_matmul_precision("highest")
    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = _load_measurement_registry(arguments.measurement_registry)
    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and immutable measurement registry disagree")
    registered_rows, _contexts = registered_tokens(
        registry, arguments.tokens, device
    )
    batches, indices = _ordered_batches(
        checkpoint, arguments.tokens, arguments.order, device, horizon
    )
    return device, checkpoint_path, checkpoint, registered_rows, batches, indices


def _common_record(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
    device: torch.device,
    layer: int,
) -> dict[str, np.ndarray]:
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)
    return {
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "trajectory": np.asarray(str(checkpoint["config"]["name"])),
        "seed": np.asarray(int(checkpoint["config"]["seed"]), dtype=np.int64),
        "source_step": np.asarray(int(checkpoint["step"]), dtype=np.int64),
        "layer": np.asarray(int(layer), dtype=np.int64),
        "device": np.asarray(str(device)),
        "terminal_status": np.asarray("complete"),
        "torch_version": np.asarray(str(torch.__version__)),
        "cuda_runtime_version": np.asarray(str(torch.version.cuda)),
        "cudnn_version": np.asarray(int(torch.backends.cudnn.version() or 0)),
        "gpu_name": np.asarray(str(properties.name)),
        "gpu_compute_capability": np.asarray(
            f"{int(capability[0])}.{int(capability[1])}"
        ),
        "cublas_workspace_config": np.asarray(
            os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        ),
        "deterministic_algorithms": np.asarray(
            torch.are_deterministic_algorithms_enabled()
        ),
        "allow_tf32": np.asarray(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": np.asarray(
            torch.get_float32_matmul_precision()
        ),
        "measurement_registry_object_sha256": np.asarray(
            digest_object(checkpoint["measurement_registry"])
        ),
    }


def _mask_digest(parameters: TensorTree, clip_value: float) -> tuple[str, int, float]:
    digest = hashlib.sha256()
    clipped_count = 0
    minimum_distance = math.inf
    for parameter in parameters:
        if parameter.grad is None:
            raise ArithmeticError("a live parameter has no raw gradient")
        gradient = parameter.grad.detach()
        inside = gradient.abs() < clip_value
        packed = np.packbits(
            inside.to(dtype=torch.uint8).cpu().numpy().reshape(-1),
            bitorder="little",
        )
        digest.update(str(tuple(gradient.shape)).encode("ascii"))
        digest.update(packed.tobytes())
        clipped_count += int(torch.count_nonzero(~inside).item())
        minimum_distance = min(
            minimum_distance,
            float(torch.min(torch.abs(gradient.abs() - clip_value)).item()),
        )
    return digest.hexdigest(), clipped_count, minimum_distance


def _native_step_with_route(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    rows: torch.Tensor,
    clip_value: float,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, rows)
    loss.backward()
    parameters = tuple(model.parameters())
    route_digest, clipped_count, minimum_distance = _mask_digest(
        parameters, clip_value
    )
    torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_value)
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "clipping_mask_sha256": route_digest,
        "clipped_coordinate_count": clipped_count,
        "minimum_clipping_boundary_distance": minimum_distance,
    }


def _parameter_branch(
    checkpoint: dict[str, Any],
    device: torch.device,
    batches: list[torch.Tensor],
    registered_rows: torch.Tensor,
    layer: int,
    horizons: tuple[int, ...],
    direction: Direction,
    signed_amplitude: float,
) -> tuple[dict[int, np.ndarray], dict[int, TensorTree], list[dict[str, Any]]]:
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    _apply_direction(model, optimizer, direction, signed_amplitude)
    observations: dict[int, np.ndarray] = {}
    snapshots: dict[int, TensorTree] = {}
    routes = []
    clip_value = float(checkpoint["config"]["clip"])
    for local_step, rows in enumerate(batches, start=1):
        routes.append(_native_step_with_route(model, optimizer, rows, clip_value))
        if local_step in horizons:
            with torch.no_grad():
                observations[local_step] = (
                    _physical_observation(model, registered_rows, layer)
                    .to(dtype=torch.float32)
                    .cpu()
                    .numpy()
                )
                snapshots[local_step] = tuple(
                    value.detach().to(device="cpu", dtype=torch.float32).clone()
                    for value in model.parameters()
                )
    del optimizer, model
    torch.cuda.empty_cache()
    return observations, snapshots, routes


def _observe_parameter_deltas(
    checkpoint: dict[str, Any],
    device: torch.device,
    batches: list[torch.Tensor],
    registered_rows: torch.Tensor,
    layer: int,
    deltas: dict[int, TensorTree],
) -> tuple[dict[int, np.ndarray], list[dict[str, Any]]]:
    """Apply the endpoint observation derivative to finite state secants."""

    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    names, parameters, _first, _second = _named_state(model, optimizer)
    actions: dict[int, np.ndarray] = {}
    routes = []
    clip_value = float(checkpoint["config"]["clip"])
    for local_step, rows in enumerate(batches, start=1):
        routes.append(_native_step_with_route(model, optimizer, rows, clip_value))
        if local_step not in deltas:
            continue
        tangent = tuple(
            value.to(device=device, dtype=parameter.dtype)
            for value, parameter in zip(deltas[local_step], parameters, strict=True)
        )

        def observe(*local: torch.Tensor) -> torch.Tensor:
            mapping = dict(zip(names, local, strict=True))
            return _physical_observation(
                model, registered_rows, layer, mapping
            )

        _base, action = jvp(observe, parameters, tangent)
        actions[local_step] = (
            action.detach().to(dtype=torch.float32).cpu().numpy()
        )
        del tangent, action, _base
    del optimizer, model
    torch.cuda.empty_cache()
    return actions, routes


def _balance_for_branch_pair(
    *,
    checkpoint: dict[str, Any],
    device: torch.device,
    batches: list[torch.Tensor],
    registered_rows: torch.Tensor,
    layer: int,
    horizons: tuple[int, ...],
    direction: Direction,
    amplitude: float,
    homogeneous: dict[int, np.ndarray],
    replay_relative_residual: float,
) -> dict[int, dict[str, Any]]:
    plus_observation, plus_state, plus_routes = _parameter_branch(
        checkpoint,
        device,
        batches,
        registered_rows,
        layer,
        horizons,
        direction,
        amplitude,
    )
    minus_observation, minus_state, minus_routes = _parameter_branch(
        checkpoint,
        device,
        batches,
        registered_rows,
        layer,
        horizons,
        direction,
        -amplitude,
    )
    deltas = {
        horizon: tuple(
            (plus - minus).mul_(0.5)
            for plus, minus in zip(
                plus_state[horizon], minus_state[horizon], strict=True
            )
        )
        for horizon in horizons
    }
    del plus_state, minus_state
    delta_actions, base_routes = _observe_parameter_deltas(
        checkpoint,
        device,
        batches,
        registered_rows,
        layer,
        deltas,
    )
    del deltas
    result = {}
    for horizon in horizons:
        observed = 0.5 * (
            plus_observation[horizon] - minus_observation[horizon]
        )
        linearized_secant = delta_actions[horizon]
        local_homogeneous = homogeneous[horizon]
        state_correction = linearized_secant - local_homogeneous
        observation_correction = observed - linearized_secant
        metrics = balance_metrics(
            local_homogeneous,
            state_correction,
            observation_correction,
            observed,
        )
        evaluable = cell_evaluable(
            effect_norm=float(metrics["response_norm"]),
            reconstruction_relative_residual=float(
                metrics["reconstruction_relative_residual"]
            ),
            replay_relative_residual=replay_relative_residual,
        )
        route_index = int(horizon) - 1
        result[horizon] = {
            "terms": np.stack(
                [local_homogeneous, state_correction, observation_correction]
            ).astype(np.float32),
            "observed": observed.astype(np.float32),
            "metrics": metrics,
            "evaluable": evaluable,
            "plus_route": plus_routes[route_index],
            "minus_route": minus_routes[route_index],
            "base_route": base_routes[route_index],
            "route_changed": bool(
                len(
                    {
                        plus_routes[route_index]["clipping_mask_sha256"],
                        minus_routes[route_index]["clipping_mask_sha256"],
                        base_routes[route_index]["clipping_mask_sha256"],
                    }
                )
                > 1
            ),
        }
    return result


def _propagation_replay_maximum(diagnostics: list[dict[str, Any]]) -> float:
    return max(
        float(row[name])
        for row in diagnostics
        for name in (
            "parameter_replay_relative_residual",
            "first_moment_replay_relative_residual",
            "second_moment_replay_relative_residual",
        )
    )


def qualification_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    (
        device,
        checkpoint_path,
        checkpoint,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, 1)
    layer = int(arguments.layer)
    observations, _source, actions, directions, metadata, diagnostics = _propagate(
        checkpoint, device, batches, registered_rows, layer, (1,)
    )
    replay = _propagation_replay_maximum(diagnostics)
    name = "balanced_full_state"
    amplitude = amplitudes(name)[1]
    rows = _balance_for_branch_pair(
        checkpoint=checkpoint,
        device=device,
        batches=batches,
        registered_rows=registered_rows,
        layer=layer,
        horizons=(1,),
        direction=directions[name],
        amplitude=amplitude,
        homogeneous={1: amplitude * actions[1][name]},
        replay_relative_residual=replay,
    )
    local = rows[1]
    resources = _resource_record(device, started)
    qualified = bool(
        local["evaluable"]
        and int(resources["resource_peak_gpu_reserved_bytes"])
        <= int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
    )
    _write_npz_compressed(
        arguments.output,
        schema_version=np.asarray(QUALIFICATION_SCHEMA),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        qualified=np.asarray(qualified),
        training_chunk_indices=indices,
        direction=np.asarray(name),
        amplitude=np.asarray(amplitude),
        direction_algorithm_sha256=np.asarray(metadata["algorithm_sha256"]),
        base_endpoint=observations[1],
        balance_terms=local["terms"],
        observed_response=local["observed"],
        cross_gram=local["metrics"]["cross_gram"],
        reconstruction_relative_residual=np.asarray(
            local["metrics"]["reconstruction_relative_residual"]
        ),
        maximum_adamw_replay_relative_residual=np.asarray(replay),
        **resources,
    )
    if arguments.require_pass and not qualified:
        raise SystemExit("observable-balance development qualification failed")


def construction_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    horizons = tuple(int(value) for value in HORIZON_GRID)
    (
        device,
        checkpoint_path,
        checkpoint,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, max(horizons))
    layer = int(arguments.layer)
    observations, source_actions, actions, directions, metadata, diagnostics = (
        _propagate(checkpoint, device, batches, registered_rows, layer, horizons)
    )
    replay = _propagation_replay_maximum(diagnostics)
    direction_names = list(DIRECTION_CLASSES)
    amplitude_grid = np.asarray(
        [amplitudes(name) for name in direction_names], dtype=np.float64
    )
    shape = (
        len(direction_names),
        len(horizons),
        2,
    )
    term_vectors = np.empty(shape + (3,) + observations[horizons[0]].shape, np.float32)
    observed_vectors = np.empty(shape + observations[horizons[0]].shape, np.float32)
    term_norms = np.empty(shape + (3,), np.float64)
    response_norms = np.empty(shape, np.float64)
    reconstruction = np.empty(shape, np.float64)
    cancellation_index = np.empty(shape, np.float64)
    response_ratio = np.empty(shape, np.float64)
    homogeneous_inner = np.empty(shape, np.float64)
    cross_gram = np.empty(shape + (3, 3), np.float64)
    evaluable = np.empty(shape, bool)
    cancellation_dominant = np.empty(shape, bool)
    dominant_correction = np.empty(shape, dtype="U11")
    route_changed = np.empty(shape, bool)
    plus_route = np.empty(shape, dtype="U64")
    minus_route = np.empty(shape, dtype="U64")
    base_route = np.empty(shape, dtype="U64")
    for direction_index, direction_name in enumerate(direction_names):
        for amplitude_index, amplitude in enumerate(amplitude_grid[direction_index]):
            rows = _balance_for_branch_pair(
                checkpoint=checkpoint,
                device=device,
                batches=batches,
                registered_rows=registered_rows,
                layer=layer,
                horizons=horizons,
                direction=directions[direction_name],
                amplitude=float(amplitude),
                homogeneous={
                    horizon: float(amplitude) * actions[horizon][direction_name]
                    for horizon in horizons
                },
                replay_relative_residual=replay,
            )
            for horizon_index, horizon in enumerate(horizons):
                local = rows[horizon]
                metrics = local["metrics"]
                index = (direction_index, horizon_index, amplitude_index)
                term_vectors[index] = local["terms"]
                observed_vectors[index] = local["observed"]
                term_norms[index] = metrics["term_norms"]
                response_norms[index] = metrics["response_norm"]
                reconstruction[index] = metrics["reconstruction_relative_residual"]
                cancellation_index[index] = metrics["cancellation_index"]
                response_ratio[index] = metrics["response_to_homogeneous_ratio"]
                homogeneous_inner[index] = metrics[
                    "homogeneous_correction_inner_product"
                ]
                cross_gram[index] = metrics["cross_gram"]
                evaluable[index] = local["evaluable"]
                cancellation_dominant[index] = metrics["cancellation_dominant"]
                dominant_correction[index] = metrics["dominant_correction"]
                route_changed[index] = local["route_changed"]
                plus_route[index] = local["plus_route"]["clipping_mask_sha256"]
                minus_route[index] = local["minus_route"]["clipping_mask_sha256"]
                base_route[index] = local["base_route"]["clipping_mask_sha256"]
    _write_npz_compressed(
        arguments.output,
        schema_version=np.asarray(CONSTRUCTION_SCHEMA),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        training_chunk_indices=indices,
        horizons=np.asarray(horizons, dtype=np.int64),
        direction_names=np.asarray(direction_names),
        amplitudes=amplitude_grid,
        direction_algorithm_sha256=np.asarray(metadata["algorithm_sha256"]),
        source_actions=np.stack([source_actions[name] for name in direction_names]),
        base_endpoints=np.stack([observations[horizon] for horizon in horizons]),
        balance_terms=term_vectors,
        observed_responses=observed_vectors,
        term_norms=term_norms,
        response_norms=response_norms,
        reconstruction_relative_residuals=reconstruction,
        cancellation_indices=cancellation_index,
        response_to_homogeneous_ratios=response_ratio,
        homogeneous_correction_inner_products=homogeneous_inner,
        cross_grams=cross_gram,
        evaluable=evaluable,
        cancellation_dominant=cancellation_dominant,
        dominant_correction=dominant_correction,
        clipping_route_changed=route_changed,
        plus_clipping_route_sha256=plus_route,
        minus_clipping_route_sha256=minus_route,
        base_clipping_route_sha256=base_route,
        maximum_adamw_replay_relative_residual=np.asarray(replay),
        **_resource_record(device, started),
    )


def _load_lock(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    if value.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("unknown observable-balance construction lock")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("observable-balance lock digest does not replay")
    return value


def prediction_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    lock = _load_lock(arguments.lock)
    layer = int(arguments.layer)
    horizon = int(lock["layers"][str(layer)]["selected_horizon"])
    (
        device,
        checkpoint_path,
        checkpoint,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, horizon)
    observations, source_actions, actions, _directions, metadata, diagnostics = (
        _propagate(checkpoint, device, batches, registered_rows, layer, (horizon,))
    )
    replay = _propagation_replay_maximum(diagnostics)
    names = list(DIRECTION_CLASSES)
    amplitude_values = np.asarray(
        [amplitudes(name)[1] for name in names], dtype=np.float64
    )
    endpoint_actions = np.stack([actions[horizon][name] for name in names])
    homogeneous = (
        amplitude_values[:, None, None, None, None] * endpoint_actions
    ).astype(np.float32)
    _write_npz_compressed(
        arguments.output,
        schema_version=np.asarray(PREDICTION_SCHEMA),
        stage=np.asarray("prediction"),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        lock_sha256=np.asarray(str(lock["lock_sha256"])),
        horizon=np.asarray(horizon, dtype=np.int64),
        training_chunk_indices=indices,
        direction_names=np.asarray(names),
        amplitudes=amplitude_values,
        direction_algorithm_sha256=np.asarray(metadata["algorithm_sha256"]),
        source_actions=np.stack([source_actions[name] for name in names]),
        base_endpoint=observations[horizon],
        endpoint_actions=endpoint_actions,
        homogeneous_terms=homogeneous,
        maximum_adamw_replay_relative_residual=np.asarray(replay),
        **_resource_record(device, started),
    )


def validation_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    prediction_path = Path(arguments.prediction).resolve()
    prediction = _load_npz(prediction_path)
    if str(prediction["schema_version"].item()) != PREDICTION_SCHEMA:
        raise ValueError("validation dependency is not a prediction record")
    if str(prediction["terminal_status"].item()) != "complete":
        _terminal_record(
            arguments,
            VALIDATION_SCHEMA,
            ArithmeticError("nonfinite upstream prediction terminal"),
            started,
        )
        return
    if str(prediction["stage"].item()) != "prediction":
        raise ValueError("validation dependency has the wrong semantic role")
    horizon = int(prediction["horizon"].item())
    layer = int(arguments.layer)
    (
        device,
        checkpoint_path,
        checkpoint,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, horizon)
    if (
        str(prediction["checkpoint_sha256"].item()) != sha256_path(checkpoint_path)
        or int(prediction["layer"].item()) != layer
        or not np.array_equal(prediction["training_chunk_indices"], indices)
    ):
        raise ValueError("prediction and held-out validation source disagree")
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    _names, _parameters, _first, _second, directions, metadata = _make_directions(
        checkpoint,
        model,
        optimizer,
        registered_rows,
        batches[0],
        layer,
    )
    del optimizer, model
    if metadata["algorithm_sha256"] != str(
        prediction["direction_algorithm_sha256"].item()
    ):
        raise ValueError("held-out direction construction drifted")
    names = [str(value) for value in prediction["direction_names"].tolist()]
    amplitude_values = np.asarray(prediction["amplitudes"], dtype=np.float64)
    homogeneous = np.asarray(prediction["homogeneous_terms"], dtype=np.float32)
    terms = []
    observed = []
    term_norms = []
    response_norms = []
    reconstruction = []
    cancellation_indices = []
    response_ratios = []
    homogeneous_inner = []
    cross_grams = []
    evaluable = []
    dominant = []
    dominant_correction = []
    route_changed = []
    plus_routes = []
    minus_routes = []
    base_routes = []
    replay = float(prediction["maximum_adamw_replay_relative_residual"].item())
    for index, (name, amplitude) in enumerate(
        zip(names, amplitude_values, strict=True)
    ):
        local = _balance_for_branch_pair(
            checkpoint=checkpoint,
            device=device,
            batches=batches,
            registered_rows=registered_rows,
            layer=layer,
            horizons=(horizon,),
            direction=directions[name],
            amplitude=float(amplitude),
            homogeneous={horizon: homogeneous[index]},
            replay_relative_residual=replay,
        )[horizon]
        metrics = local["metrics"]
        terms.append(local["terms"])
        observed.append(local["observed"])
        term_norms.append(metrics["term_norms"])
        response_norms.append(metrics["response_norm"])
        reconstruction.append(metrics["reconstruction_relative_residual"])
        cancellation_indices.append(metrics["cancellation_index"])
        response_ratios.append(metrics["response_to_homogeneous_ratio"])
        homogeneous_inner.append(metrics["homogeneous_correction_inner_product"])
        cross_grams.append(metrics["cross_gram"])
        evaluable.append(local["evaluable"])
        dominant.append(metrics["cancellation_dominant"])
        dominant_correction.append(metrics["dominant_correction"])
        route_changed.append(local["route_changed"])
        plus_routes.append(local["plus_route"]["clipping_mask_sha256"])
        minus_routes.append(local["minus_route"]["clipping_mask_sha256"])
        base_routes.append(local["base_route"]["clipping_mask_sha256"])
    _write_npz_compressed(
        arguments.output,
        schema_version=np.asarray(VALIDATION_SCHEMA),
        stage=np.asarray("validation"),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        prediction_sha256=np.asarray(sha256_path(prediction_path)),
        lock_sha256=prediction["lock_sha256"],
        horizon=np.asarray(horizon, dtype=np.int64),
        training_chunk_indices=indices,
        direction_names=np.asarray(names),
        amplitudes=amplitude_values,
        direction_algorithm_sha256=np.asarray(metadata["algorithm_sha256"]),
        balance_terms=np.stack(terms),
        observed_responses=np.stack(observed),
        term_norms=np.asarray(term_norms),
        response_norms=np.asarray(response_norms),
        reconstruction_relative_residuals=np.asarray(reconstruction),
        cancellation_indices=np.asarray(cancellation_indices),
        response_to_homogeneous_ratios=np.asarray(response_ratios),
        homogeneous_correction_inner_products=np.asarray(homogeneous_inner),
        cross_grams=np.asarray(cross_grams),
        evaluable=np.asarray(evaluable),
        cancellation_dominant=np.asarray(dominant),
        dominant_correction=np.asarray(dominant_correction),
        clipping_route_changed=np.asarray(route_changed),
        plus_clipping_route_sha256=np.asarray(plus_routes),
        minus_clipping_route_sha256=np.asarray(minus_routes),
        base_clipping_route_sha256=np.asarray(base_routes),
        **_resource_record(device, started),
    )


def _gate_source(
    checkpoint: dict[str, Any],
    device: torch.device,
    layer: int,
) -> tuple[torch.nn.Module, torch.optim.AdamW, torch.Tensor]:
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    gate = dict(model.named_parameters())[_gate_parameter_name(layer)]
    with torch.no_grad():
        gate.zero_()
        optimizer.state[gate]["exp_avg"].zero_()
    return model, optimizer, gate


def _other_gradient_digest(
    model: torch.nn.Module, selected: torch.Tensor
) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if parameter is selected:
            continue
        if parameter.grad is None:
            raise ArithmeticError("a controlled-arm parameter has no gradient")
        digest.update(name.encode("utf-8"))
        digest.update(parameter.grad.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _gate_step(
    model: torch.nn.Module,
    optimizer: torch.optim.AdamW,
    gate: torch.Tensor,
    rows: torch.Tensor,
    clip_value: float,
    *,
    suppress_gate_gradient: bool,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    loss = _loss(model, rows)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
    clipped_gradient = gate.grad.detach().clone()
    other_digest = _other_gradient_digest(model, gate)
    state = optimizer.state[gate]
    first_before = state["exp_avg"].detach().clone()
    second_before = state["exp_avg_sq"].detach().clone()
    step_before = int(state["step"].item())
    group = optimizer.param_groups[0]
    beta1, beta2 = map(float, group["betas"])
    rate = float(group["lr"])
    epsilon = float(group["eps"])
    decay = float(group["weight_decay"])
    effective_gradient = torch.zeros_like(clipped_gradient) if suppress_gate_gradient else clipped_gradient
    expected_first = first_before.clone().lerp_(effective_gradient, 1.0 - beta1)
    expected_second = second_before.clone().mul_(beta2).addcmul_(
        effective_gradient, effective_gradient, value=1.0 - beta2
    )
    step_after = step_before + 1
    bias1 = 1.0 - beta1**step_after
    bias2_sqrt = math.sqrt(1.0 - beta2**step_after)
    denominator = expected_second.sqrt().div_(bias2_sqrt).add_(epsilon)
    expected_gate = gate.detach().clone().mul_(1.0 - rate * decay)
    expected_gate.addcdiv_(
        expected_first,
        denominator,
        value=-(rate / bias1),
    )
    if suppress_gate_gradient:
        gate.grad.zero_()
    optimizer.step()
    return {
        "loss": float(loss.detach().item()),
        "clipped_gradient": clipped_gradient.cpu().numpy(),
        "other_gradient_sha256": other_digest,
        "predicted_first": expected_first.cpu().numpy(),
        "predicted_second": expected_second.cpu().numpy(),
        "predicted_gate": expected_gate.cpu().numpy(),
        "actual_first": state["exp_avg"].detach().cpu().numpy(),
        "actual_second": state["exp_avg_sq"].detach().cpu().numpy(),
        "actual_gate": gate.detach().cpu().numpy(),
        "first_bitwise": torch.equal(state["exp_avg"], expected_first),
        "second_bitwise": torch.equal(state["exp_avg_sq"], expected_second),
        "gate_bitwise": torch.equal(gate.detach(), expected_gate),
    }


def stratum_record(arguments: argparse.Namespace) -> None:
    started = time.monotonic()
    (
        device,
        checkpoint_path,
        checkpoint,
        registered_rows,
        batches,
        indices,
    ) = _load_inputs(arguments, 1)
    layer = int(arguments.layer)
    clip_value = float(checkpoint["config"]["clip"])

    natural_model, natural_optimizer, natural_gate = _gate_source(
        checkpoint, device, layer
    )
    with torch.no_grad():
        source = (
            _physical_observation(natural_model, registered_rows, layer)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
    natural = _gate_step(
        natural_model,
        natural_optimizer,
        natural_gate,
        batches[0],
        clip_value,
        suppress_gate_gradient=False,
    )
    with torch.no_grad():
        natural_endpoint = (
            _physical_observation(natural_model, registered_rows, layer)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
    del natural_optimizer, natural_model
    torch.cuda.empty_cache()

    control_model, control_optimizer, control_gate = _gate_source(
        checkpoint, device, layer
    )
    with torch.no_grad():
        control_source = (
            _physical_observation(control_model, registered_rows, layer)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
    control = _gate_step(
        control_model,
        control_optimizer,
        control_gate,
        batches[0],
        clip_value,
        suppress_gate_gradient=True,
    )
    with torch.no_grad():
        control_endpoint = (
            _physical_observation(control_model, registered_rows, layer)
            .to(dtype=torch.float32)
            .cpu()
            .numpy()
        )
    source_equal = np.array_equal(source, control_source)
    formula_bitwise = bool(
        natural["first_bitwise"]
        and natural["second_bitwise"]
        and natural["gate_bitwise"]
    )
    control_face_bitwise = bool(
        np.count_nonzero(control["actual_gate"]) == 0
        and np.count_nonzero(control["actual_first"]) == 0
        and np.count_nonzero(control_endpoint) == 0
    )
    other_gradients_equal = bool(
        natural["other_gradient_sha256"] == control["other_gradient_sha256"]
    )
    theory_code_pass = bool(
        source_equal
        and np.count_nonzero(source) == 0
        and formula_bitwise
        and control_face_bitwise
        and other_gradients_equal
    )
    _write_npz_compressed(
        arguments.output,
        schema_version=np.asarray(STRATUM_SCHEMA),
        **_common_record(checkpoint_path, checkpoint, device, layer),
        training_chunk_indices=indices,
        source_row_quotient=source,
        natural_endpoint_row_quotient=natural_endpoint,
        controlled_endpoint_row_quotient=control_endpoint,
        natural_clipped_gate_gradient=natural["clipped_gradient"],
        natural_predicted_first_moment=natural["predicted_first"],
        natural_actual_first_moment=natural["actual_first"],
        natural_predicted_second_moment=natural["predicted_second"],
        natural_actual_second_moment=natural["actual_second"],
        natural_predicted_gate=natural["predicted_gate"],
        natural_actual_gate=natural["actual_gate"],
        controlled_actual_first_moment=control["actual_first"],
        controlled_actual_gate=control["actual_gate"],
        natural_first_moment_bitwise=np.asarray(natural["first_bitwise"]),
        natural_second_moment_bitwise=np.asarray(natural["second_bitwise"]),
        natural_gate_bitwise=np.asarray(natural["gate_bitwise"]),
        controlled_face_bitwise=np.asarray(control_face_bitwise),
        other_gradients_bitwise=np.asarray(other_gradients_equal),
        theory_code_pass=np.asarray(theory_code_pass),
        natural_force_norm=np.asarray(np.linalg.norm(natural_endpoint.reshape(-1))),
        controlled_force_norm=np.asarray(np.linalg.norm(control_endpoint.reshape(-1))),
        natural_other_gradient_sha256=np.asarray(natural["other_gradient_sha256"]),
        controlled_other_gradient_sha256=np.asarray(control["other_gradient_sha256"]),
        natural_loss=np.asarray(natural["loss"]),
        controlled_loss=np.asarray(control["loss"]),
        **_resource_record(device, started),
    )
    del control_optimizer, control_model
    torch.cuda.empty_cache()


def _terminal_record(
    arguments: argparse.Namespace,
    schema: str,
    error: BaseException,
    started: float,
) -> None:
    device = torch.device(arguments.device)
    match = re.search(
        r"(?:qualification|construction|prediction|validation|stratum-control)-"
        r"([ABC])-L([0-2])-S([0-9]+)\.npz$",
        str(arguments.output),
    )
    if match is None:
        raise ValueError("terminal record output lacks a registered unit identity")
    label, parsed_layer, source_step = match.groups()
    trajectories = {str(row["label"]): str(row["name"]) for row in TRAJECTORIES}
    if int(parsed_layer) != int(arguments.layer):
        raise ValueError("terminal output layer and command layer disagree")
    stage = {
        QUALIFICATION_SCHEMA: "qualification",
        CONSTRUCTION_SCHEMA: "construction",
        PREDICTION_SCHEMA: "prediction",
        VALIDATION_SCHEMA: "validation",
        STRATUM_SCHEMA: "stratum-control",
    }[schema]
    extras: dict[str, np.ndarray] = {"stage": np.asarray(stage)}
    if hasattr(arguments, "lock"):
        lock = _load_lock(arguments.lock)
        extras["lock_sha256"] = np.asarray(str(lock["lock_sha256"]))
        extras["horizon"] = np.asarray(
            int(lock["layers"][parsed_layer]["selected_horizon"]), dtype=np.int64
        )
    if hasattr(arguments, "prediction"):
        prediction_path = Path(arguments.prediction).resolve()
        prediction = _load_npz(prediction_path)
        extras["prediction_sha256"] = np.asarray(sha256_path(prediction_path))
        if "lock_sha256" in prediction:
            extras["lock_sha256"] = prediction["lock_sha256"]
        if "horizon" in prediction:
            extras["horizon"] = prediction["horizon"]
    payload = {
        "schema_version": np.asarray(schema),
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "terminal_status": np.asarray("adverse-scientific-terminal"),
        "failure_class": np.asarray(type(error).__name__),
        "failure_message": np.asarray(str(error)),
        "layer": np.asarray(int(arguments.layer), dtype=np.int64),
        "source_step": np.asarray(int(source_step), dtype=np.int64),
        "trajectory": np.asarray(trajectories[label]),
        "device": np.asarray(str(device)),
        **extras,
        **_resource_record(device, started),
    }
    _write_npz_compressed(arguments.output, **payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(command: argparse.ArgumentParser) -> None:
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--tokens", required=True)
        command.add_argument("--order", required=True)
        command.add_argument("--measurement-registry", required=True)
        command.add_argument("--device", required=True, choices=("cuda:0", "cuda:1"))
        command.add_argument("--layer", required=True, type=int, choices=(0, 1, 2))
        command.add_argument("--output", required=True)

    qualification = subparsers.add_parser("qualify")
    common(qualification)
    qualification.add_argument("--require-pass", action="store_true")
    qualification.set_defaults(function=qualification_record, schema=QUALIFICATION_SCHEMA)

    construction = subparsers.add_parser("construct")
    common(construction)
    construction.set_defaults(function=construction_record, schema=CONSTRUCTION_SCHEMA)

    prediction = subparsers.add_parser("predict")
    common(prediction)
    prediction.add_argument("--lock", required=True)
    prediction.set_defaults(function=prediction_record, schema=PREDICTION_SCHEMA)

    validation = subparsers.add_parser("validate")
    common(validation)
    validation.add_argument("--prediction", required=True)
    validation.set_defaults(function=validation_record, schema=VALIDATION_SCHEMA)

    stratum = subparsers.add_parser("stratum-control")
    common(stratum)
    stratum.set_defaults(function=stratum_record, schema=STRATUM_SCHEMA)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    started = time.monotonic()
    try:
        arguments.function(arguments)
    except (ArithmeticError, FloatingPointError) as error:
        if not any(marker in str(error).lower() for marker in SCIENTIFIC_FAILURE_MARKERS):
            raise
        _terminal_record(arguments, arguments.schema, error, started)


if __name__ == "__main__":
    main()
