#!/usr/bin/env python3
"""Paired one-step direct-work producer for the PLDR physical row map."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Callable

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(ROOT))

from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    tensor_mapping_digest,
    validate_measurement_registry,
)
from confirm.direct_work import (  # noqa: E402
    center_rows,
    gate_shape_secants,
    intervention_contrast,
    physical_energy,
    work_charge,
)
from confirm.direct_work_specs import (  # noqa: E402
    CAMPAIGN_ID,
    NUMERICAL_POLICY,
    RECORD_SCHEMA,
    RELEASE_ID,
)
from confirm.finite_increment_live import (  # noqa: E402
    capture_maps,
    load_checkpoint,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
    write_npz_atomic,
)
from confirm.row_map_live import (  # noqa: E402
    _gate_parameter_name,
    _loss,
    _model_from_raw,
)


def _load_registry(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_measurement_registry(value, require_nonempty=True)
    return value


def _frozen_update_batch(
    path: str | Path,
    checkpoint: dict[str, Any],
    tokens_path: str | Path,
    device: torch.device,
) -> tuple[torch.Tensor, list[int], str]:
    batch_path = Path(path).resolve(strict=True)
    value = json.loads(batch_path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "context_length",
        "batch_size",
        "dataset_sha256",
        "chunk_indices",
        "content_sha256",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("frozen update-batch registry is malformed")
    unsigned = dict(value)
    recorded = unsigned.pop("content_sha256")
    if (
        value["schema_version"] != "pldr-reserve-update-batch-v1"
        or recorded != digest_object(unsigned)
        or value["context_length"] != int(checkpoint["config"]["ctx"])
        or value["batch_size"] != int(checkpoint["config"]["batch"])
        or value["dataset_sha256"] != checkpoint["data_state"]["dataset_sha256"]
    ):
        raise ValueError("frozen update-batch registry does not replay")
    indices = value["chunk_indices"]
    if (
        not isinstance(indices, list)
        or len(indices) != value["batch_size"]
        or len(indices) != len(set(indices))
        or any(not isinstance(index, int) or isinstance(index, bool) or index < 0
               for index in indices)
    ):
        raise ValueError("frozen update-batch chunk indices are invalid")
    tokens_file = Path(tokens_path).resolve(strict=True)
    if sha256_path(tokens_file) != value["dataset_sha256"]:
        raise ValueError("frozen update batch and token archive disagree")
    data = np.memmap(tokens_file, dtype=np.uint16, mode="r")
    context = int(value["context_length"])
    total_chunks = len(data) // context
    if any(index >= total_chunks for index in indices):
        raise ValueError("frozen update batch leaves the token archive")
    chunks = data[: total_chunks * context].reshape(total_chunks, context)
    rows = torch.from_numpy(
        chunks[np.asarray(indices)].astype(np.int64, copy=True)
    ).to(device)
    return rows, [int(index) for index in indices], sha256_path(batch_path)


def _json(value: Any) -> np.ndarray:
    return np.asarray(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _array_digest(value: torch.Tensor) -> str:
    array = value.detach().contiguous().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _gradient_digest(
    model: torch.nn.Module,
    *,
    selected: Callable[[str], bool] | None = None,
) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if selected is not None and not selected(name):
            continue
        digest.update(name.encode("utf-8"))
        if parameter.grad is None:
            digest.update(b"NONE")
        else:
            digest.update(_array_digest(parameter.grad).encode("ascii"))
    return digest.hexdigest()


def _in_upstream_closure(name: str, layer: int) -> bool:
    if name.startswith(("decoder.embedding.", "decoder.layernorm1.")):
        return True
    prefix = "decoder.dec_layers."
    if not name.startswith(prefix):
        return False
    remainder = name[len(prefix):]
    index_text, separator, local = remainder.partition(".")
    if not separator or not index_text.isdigit():
        return False
    index = int(index_text)
    if index < layer:
        return True
    if index > layer:
        return False
    return local.startswith((
        "mha1.wq.",
        "mha1.layernorm1.",
        "mha1.reslayerAs.",
    ))


def _environment(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or device.index is None:
        raise RuntimeError("direct-work scientific nodes require indexed CUDA")
    command = [
        "nvidia-smi",
        f"--id={device.index}",
        "--query-gpu=index,name,uuid,driver_version,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    raw = result.stdout.strip()
    parts = [part.strip() for part in raw.split(",")]
    if len(parts) != 5 or not all(parts):
        raise RuntimeError("NVIDIA driver query did not return all required fields")
    index, name, uuid, driver, compute_capability = parts
    if int(index) != device.index:
        raise RuntimeError("NVIDIA driver query returned the wrong device")
    return {
        "nvidia_query_command": command,
        "nvidia_query_raw_output": raw,
        "nvidia_driver_version": driver,
        "gpu_name": name,
        "gpu_uuid": uuid,
        "compute_capability": compute_capability,
        "cuda_runtime": str(torch.version.cuda),
        "cudnn_version": str(torch.backends.cudnn.version()),
        "torch_version": str(torch.__version__),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _seed_branch(seed: int, step: int, layer: int) -> int:
    branch_seed = int(seed) * 100_000 + int(step) * 10 + int(layer)
    torch.manual_seed(branch_seed)
    torch.cuda.manual_seed_all(branch_seed)
    return branch_seed


def _loss_with_row_adjoint(
    model: torch.nn.Module,
    rows: torch.Tensor,
    layer: int,
    *,
    remove_centered_adjoint: bool,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    incoming: list[torch.Tensor] = []
    outgoing: list[torch.Tensor] = []
    layernorm = model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA

    def forward_hook(
        _module: torch.nn.Module,
        _inputs: tuple[Any, ...],
        output: torch.Tensor,
    ) -> None:
        def gradient_hook(gradient: torch.Tensor) -> torch.Tensor:
            incoming.append(gradient.detach().clone())
            projected = gradient.mean(dim=-2, keepdim=True).expand_as(gradient)
            result = projected if remove_centered_adjoint else gradient
            outgoing.append(result.detach().clone())
            return result

        output.register_hook(gradient_hook)

    handle = layernorm.register_forward_hook(forward_hook)
    try:
        loss = _loss(model, rows)
        loss.backward()
    finally:
        handle.remove()
    if len(incoming) != 1 or len(outgoing) != 1:
        raise RuntimeError("selected row-map adjoint hook did not fire exactly once")
    return float(loss.detach().cpu()), incoming[0], outgoing[0]


def _update_norms(
    names: tuple[str, ...], directions: tuple[torch.Tensor, ...]
) -> dict[str, float]:
    return {
        name: float(torch.linalg.vector_norm(value.detach().double()).cpu())
        for name, value in zip(names, directions, strict=True)
    }


def _run_arm(
    checkpoint: dict[str, Any],
    device: torch.device,
    update_rows: torch.Tensor,
    registry_rows: torch.Tensor,
    layer: int,
    *,
    seed: int,
    step: int,
    remove_centered_adjoint: bool,
) -> dict[str, Any]:
    branch_seed = _seed_branch(seed, step, layer)
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    model.train()
    source_shape, source_rows = capture_maps(model, registry_rows)
    gate_name = _gate_parameter_name(layer)
    source_gate = dict(model.named_parameters())[gate_name].detach().double().cpu().numpy()

    optimizer.zero_grad(set_to_none=True)
    loss, incoming, outgoing = _loss_with_row_adjoint(
        model,
        update_rows,
        layer,
        remove_centered_adjoint=remove_centered_adjoint,
    )
    full_gradient_sha256 = _gradient_digest(model)
    outside_gradient_sha256 = _gradient_digest(
        model, selected=lambda name: not _in_upstream_closure(name, layer)
    )
    centered_incoming = incoming - incoming.mean(dim=-2, keepdim=True)
    constant_incoming = incoming.mean(dim=-2, keepdim=True).expand_as(incoming)
    centered_outgoing = outgoing - outgoing.mean(dim=-2, keepdim=True)

    clip_value = float(checkpoint["config"]["clip"])
    torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=clip_value)
    clipped_gradient_sha256 = _gradient_digest(model)
    names, values, directions, successors = optimizer_displacements(model, optimizer)
    source_parameters = dict(zip(names, values, strict=True))
    endpoint_parameters = dict(zip(names, successors, strict=True))
    direction_mapping = dict(zip(names, directions, strict=True))
    source_parameter_sha256 = tensor_mapping_digest(source_parameters)
    predicted_parameter_sha256 = tensor_mapping_digest(endpoint_parameters)
    update_sha256 = tensor_mapping_digest(direction_mapping)
    endpoint_shape, endpoint_rows = capture_maps(
        model, registry_rows, endpoint_parameters
    )
    endpoint_gate = endpoint_parameters[gate_name].detach().double().cpu().numpy()

    optimizer.step()
    actual_parameters = dict(model.named_parameters())
    actual_parameter_sha256 = tensor_mapping_digest(actual_parameters)
    maximum_parameter_residual = max(
        float(torch.max(torch.abs(
            actual_parameters[name].detach() - endpoint_parameters[name]
        )).cpu())
        for name in names
    )
    actual_shape, actual_rows = capture_maps(model, registry_rows)
    maximum_observation_residual = max(
        float(np.max(np.abs(actual_shape - endpoint_shape))),
        float(np.max(np.abs(actual_rows - endpoint_rows))),
    )
    result = {
        "branch_seed": branch_seed,
        "loss": loss,
        "source_shape": source_shape[:, layer],
        "source_rows": source_rows[:, layer],
        "endpoint_shape": actual_shape[:, layer],
        "endpoint_rows": actual_rows[:, layer],
        "source_gate": source_gate,
        "endpoint_gate": endpoint_gate,
        "incoming_adjoint": incoming.detach().double().cpu().numpy(),
        "outgoing_adjoint": outgoing.detach().double().cpu().numpy(),
        "incoming_centered_adjoint_norm": float(
            torch.linalg.vector_norm(centered_incoming.double()).cpu()
        ),
        "incoming_row_constant_adjoint_norm": float(
            torch.linalg.vector_norm(constant_incoming.double()).cpu()
        ),
        "outgoing_centered_adjoint_norm": float(
            torch.linalg.vector_norm(centered_outgoing.double()).cpu()
        ),
        "full_gradient_sha256": full_gradient_sha256,
        "outside_gradient_sha256": outside_gradient_sha256,
        "clipped_gradient_sha256": clipped_gradient_sha256,
        "source_parameter_sha256": source_parameter_sha256,
        "predicted_parameter_sha256": predicted_parameter_sha256,
        "actual_parameter_sha256": actual_parameter_sha256,
        "update_sha256": update_sha256,
        "update_norms": _update_norms(names, directions),
        "maximum_parameter_replay_residual": maximum_parameter_residual,
        "maximum_observation_replay_residual": maximum_observation_residual,
    }
    del optimizer, model, endpoint_parameters, source_parameters, direction_mapping
    del values, directions, successors, incoming, outgoing
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _gate_shape_payload(prefix: str, arm: dict[str, Any], source_z: np.ndarray) -> dict[str, Any]:
    endpoint_z = center_rows(arm["endpoint_rows"])
    ledger = work_charge(source_z, endpoint_z)
    secants = gate_shape_secants(
        arm["source_shape"],
        arm["endpoint_shape"],
        arm["source_gate"],
        arm["endpoint_gate"],
    )
    return {
        f"{prefix}_endpoint_z": endpoint_z,
        f"{prefix}_increment": ledger["increment"],
        f"{prefix}_endpoint_energy": ledger["endpoint_energy"],
        f"{prefix}_energy_change": ledger["energy_change"],
        f"{prefix}_signed_work": ledger["signed_work"],
        f"{prefix}_finite_step_charge": ledger["finite_step_charge"],
        f"{prefix}_work_charge_residual": ledger["identity_residual"],
        f"{prefix}_reopening_charge": ledger["reopening_charge"],
        f"{prefix}_shape_secant": secants["shape_secant"],
        f"{prefix}_gate_secant": secants["gate_secant"],
        f"{prefix}_interaction_secant": secants["interaction_secant"],
        f"{prefix}_component_gram": secants["component_gram"],
        f"{prefix}_component_source_work": secants["component_source_work"],
        f"{prefix}_gate_shape_residual": (
            ledger["increment"]
            - secants["shape_secant"]
            - secants["gate_secant"]
            - secants["interaction_secant"]
        ),
        f"{prefix}_source_factorization_residual": (
            source_z - secants["source_factorized"]
        ),
        f"{prefix}_endpoint_factorization_residual": (
            endpoint_z - secants["endpoint_factorized"]
        ),
    }


def produce(
    arguments: argparse.Namespace,
    *,
    record_schema: str = RECORD_SCHEMA,
    campaign_id: str = CAMPAIGN_ID,
    release_id: str = RELEASE_ID,
) -> None:
    started = time.monotonic()
    device = torch.device(arguments.device)
    if device.type != "cuda" or device.index is None:
        raise ValueError("direct-work producer cannot fall back to CPU")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.use_deterministic_algorithms(
        bool(NUMERICAL_POLICY["deterministic_algorithms"])
    )
    torch.backends.cuda.matmul.allow_tf32 = bool(NUMERICAL_POLICY["allow_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(NUMERICAL_POLICY["allow_tf32"])
    environment = _environment(device)

    checkpoint_path = Path(arguments.checkpoint).resolve()
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = _load_registry(arguments.registry)
    checkpoint_registry = checkpoint.get("measurement_registry")
    registry_matches_checkpoint = checkpoint_registry == registry
    if not registry_matches_checkpoint and not arguments.allow_disjoint_registry:
        raise ValueError("checkpoint and registered physical observer disagree")
    if not registry_matches_checkpoint:
        shared = (
            "context_length",
            "dataset_sha256",
            "tokenizer_sha256",
            "total_chunks",
            "trainer_probe_reserve_chunks",
            "trainer_training_limit",
            "heads",
            "layers",
            "rows_per_map",
            "pairs_per_map",
            "pair_rule",
            "cross_map_pairs_forbidden",
        )
        if not isinstance(checkpoint_registry, dict) or any(
            checkpoint_registry.get(name) != registry.get(name)
            for name in shared
        ):
            raise ValueError("disjoint registry changes physical-observer geometry")
    if arguments.update_batch_registry is None:
        update_rows, update_chunks = ordered_training_batch(
            checkpoint, arguments.tokens, arguments.data_order, device
        )
        update_batch_sha256 = checkpoint["data_state"]["data_order_sha256"]
        update_batch_mode = "chronological-frozen-order"
    else:
        update_rows, update_chunks, update_batch_sha256 = _frozen_update_batch(
            arguments.update_batch_registry, checkpoint, arguments.tokens, device
        )
        update_batch_mode = "registered-reserve-counterfactual-continuation"
    registry_rows, contexts = registered_tokens(
        registry, arguments.tokens, device
    )

    natural = _run_arm(
        checkpoint,
        device,
        update_rows,
        registry_rows,
        arguments.layer,
        seed=arguments.seed,
        step=arguments.step,
        remove_centered_adjoint=False,
    )
    control = _run_arm(
        checkpoint,
        device,
        update_rows,
        registry_rows,
        arguments.layer,
        seed=arguments.seed,
        step=arguments.step,
        remove_centered_adjoint=True,
    )
    source_equal = bool(
        np.array_equal(natural["source_rows"], control["source_rows"])
        and np.array_equal(natural["source_shape"], control["source_shape"])
        and np.array_equal(natural["source_gate"], control["source_gate"])
    )
    incoming_equal = bool(np.array_equal(
        natural["incoming_adjoint"], control["incoming_adjoint"]
    ))
    outside_gradients_equal = bool(
        natural["outside_gradient_sha256"]
        == control["outside_gradient_sha256"]
    )
    if not source_equal or not incoming_equal or not outside_gradients_equal:
        raise ArithmeticError(
            "paired intervention invariants did not close: "
            f"source={source_equal}, incoming={incoming_equal}, "
            f"outside={outside_gradients_equal}"
        )

    source_z = center_rows(natural["source_rows"])
    source_energy = physical_energy(source_z)
    payload: dict[str, Any] = {
        "schema_version": np.asarray(record_schema),
        "campaign_id": np.asarray(campaign_id),
        "release_id": np.asarray(release_id),
        "role": np.asarray(arguments.role),
        "trajectory": np.asarray(arguments.trajectory),
        "seed": np.asarray(arguments.seed, dtype=np.int64),
        "source_step": np.asarray(arguments.step, dtype=np.int64),
        "layer": np.asarray(arguments.layer, dtype=np.int64),
        "device": np.asarray(str(device)),
        "checkpoint_sha256": np.asarray(sha256_path(checkpoint_path)),
        "registry_sha256": np.asarray(registry["registry_sha256"]),
        "data_order_sha256": np.asarray(
            checkpoint["data_state"]["data_order_sha256"]
        ),
        "measurement_registry_matches_checkpoint": np.asarray(
            registry_matches_checkpoint
        ),
        "update_batch_mode": np.asarray(update_batch_mode),
        "update_batch_sha256": np.asarray(update_batch_sha256),
        "training_chunk_indices": np.asarray(update_chunks, dtype=np.int64),
        "context_ids_json": _json([row["id"] for row in contexts]),
        "source_z": source_z,
        "source_energy": source_energy,
        "natural_incoming_adjoint": natural["incoming_adjoint"],
        "natural_outgoing_adjoint": natural["outgoing_adjoint"],
        "control_incoming_adjoint": control["incoming_adjoint"],
        "control_outgoing_adjoint": control["outgoing_adjoint"],
        "paired_source_bitwise": np.asarray(source_equal),
        "incoming_adjoint_bitwise": np.asarray(incoming_equal),
        "outside_gradients_bitwise": np.asarray(outside_gradients_equal),
        "natural_loss": np.asarray(natural["loss"]),
        "control_loss": np.asarray(control["loss"]),
        "natural_incoming_centered_adjoint_norm": np.asarray(
            natural["incoming_centered_adjoint_norm"]
        ),
        "natural_incoming_row_constant_adjoint_norm": np.asarray(
            natural["incoming_row_constant_adjoint_norm"]
        ),
        "natural_outgoing_centered_adjoint_norm": np.asarray(
            natural["outgoing_centered_adjoint_norm"]
        ),
        "control_outgoing_centered_adjoint_norm": np.asarray(
            control["outgoing_centered_adjoint_norm"]
        ),
        "natural_gradient_sha256": np.asarray(natural["full_gradient_sha256"]),
        "control_gradient_sha256": np.asarray(control["full_gradient_sha256"]),
        "natural_clipped_gradient_sha256": np.asarray(
            natural["clipped_gradient_sha256"]
        ),
        "control_clipped_gradient_sha256": np.asarray(
            control["clipped_gradient_sha256"]
        ),
        "natural_update_sha256": np.asarray(natural["update_sha256"]),
        "control_update_sha256": np.asarray(control["update_sha256"]),
        "natural_update_norms_json": _json(natural["update_norms"]),
        "control_update_norms_json": _json(control["update_norms"]),
        "natural_parameter_replay_max_abs": np.asarray(
            natural["maximum_parameter_replay_residual"]
        ),
        "control_parameter_replay_max_abs": np.asarray(
            control["maximum_parameter_replay_residual"]
        ),
        "natural_observation_replay_max_abs": np.asarray(
            natural["maximum_observation_replay_residual"]
        ),
        "control_observation_replay_max_abs": np.asarray(
            control["maximum_observation_replay_residual"]
        ),
        "environment_json": _json(environment),
    }
    payload.update(_gate_shape_payload("natural", natural, source_z))
    payload.update(_gate_shape_payload("control", control, source_z))
    payload["intervention_contrast"] = intervention_contrast(
        payload["natural_endpoint_energy"],
        payload["control_endpoint_energy"],
    )

    duplicate_max_abs = 0.0
    if arguments.qualification:
        duplicate = _run_arm(
            checkpoint,
            device,
            update_rows,
            registry_rows,
            arguments.layer,
            seed=arguments.seed,
            step=arguments.step,
            remove_centered_adjoint=False,
        )
        duplicate_energy = physical_energy(center_rows(duplicate["endpoint_rows"]))
        duplicate_max_abs = float(np.max(np.abs(
            duplicate_energy - payload["natural_endpoint_energy"]
        )))
        payload["qualification_duplicate_endpoint_energy"] = duplicate_energy
        payload["qualification_duplicate_max_abs"] = np.asarray(duplicate_max_abs)

    elapsed = time.monotonic() - started
    payload.update({
        "producer_process_elapsed_seconds": np.asarray(elapsed),
        "peak_gpu_allocated_bytes": np.asarray(
            int(torch.cuda.max_memory_allocated(device)), dtype=np.int64
        ),
        "peak_gpu_reserved_bytes": np.asarray(
            int(torch.cuda.max_memory_reserved(device)), dtype=np.int64
        ),
        "peak_host_rss_bytes": np.asarray(
            int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
            dtype=np.int64,
        ),
    })
    write_npz_atomic(arguments.output, **payload)
    print(json.dumps({
        "output": str(Path(arguments.output).resolve()),
        "trajectory": arguments.trajectory,
        "step": arguments.step,
        "layer": arguments.layer,
        "device": str(device),
        "producer_process_elapsed_seconds": elapsed,
        "qualification_duplicate_max_abs": duplicate_max_abs,
    }, sort_keys=True))


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--data-order", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument(
        "--role", choices=("development", "construction", "heldout"),
        required=True,
    )
    parser.add_argument("--device", required=True)
    parser.add_argument("--qualification", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--allow-disjoint-registry",
        action="store_true",
        help="allow a digest-bound registry with checkpoint-matched geometry",
    )
    parser.add_argument(
        "--update-batch-registry",
        help="use a digest-bound reserve batch when the frozen order is exhausted",
    )
    return parser


def main() -> None:
    arguments = argument_parser().parse_args()
    produce(arguments)


if __name__ == "__main__":
    main()
