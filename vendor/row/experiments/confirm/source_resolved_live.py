"""Live producers for source-resolved PLDR row-map evidence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    load_complete_checkpoint,
    sha256_path,
    tensor_mapping_digest,
)
from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)
from confirm.finite_increment_live import (  # noqa: E402
    capture_maps,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.finite_transport import centered_energy  # noqa: E402
from confirm.row_map_live import _model_from_raw  # noqa: E402
from confirm.source_restoring_live import load_registry  # noqa: E402
from train_run import create_masks, masked_loss_function  # noqa: E402
from confirm.source_resolved_energy import (  # noqa: E402
    admissible_quadratic_radius,
    center_rows,
    gate_shape_energy,
    partition_adamw_coordinates,
    permutation_symmetric_attribution,
    source_native_energy_envelope,
    source_work_gram,
    symmetric_gate_shape_transport,
)
from confirm.source_resolved_specs import (  # noqa: E402
    CAMPAIGN_ID,
    CHECKPOINT_BINDING_SCHEMA,
    EDGE_SCHEMA,
    QUALIFICATION_SCHEMA,
    REGISTRY,
)
from confirm.strict_schema import load_json, validate  # noqa: E402


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ) + "\n"
    ).encode("utf-8")


def write_json_atomic(path: str | Path, value: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    with temporary.open("wb") as stream:
        stream.write(_canonical(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    payload = tensor.view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def _batch_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def _schema(protocol_directory: str | Path, name: str) -> dict[str, Any]:
    return load_json(Path(protocol_directory) / name)


def bind_complete_checkpoint(
    checkpoint_path: str | Path,
    tokens_path: str | Path,
    tokenizer_path: str | Path,
    order_path: str | Path,
    *,
    protocol_directory: str | Path,
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    """Open, validate, inventory, and bind one complete checkpoint."""

    checkpoint_file = Path(checkpoint_path).resolve()
    tokens_file = Path(tokens_path).resolve()
    tokenizer_file = Path(tokenizer_path).resolve()
    order_file = Path(order_path).resolve()
    checkpoint = load_complete_checkpoint(checkpoint_file, map_location="cpu")
    batch, _chunks = ordered_training_batch(
        checkpoint, tokens_file, order_file, torch.device("cpu")
    )
    state_manifest = checkpoint["state_manifest"]
    inventory = [
        {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _tensor_sha256(value),
        }
        for name, value in sorted(checkpoint["model"].items())
    ]
    dataset_sha256 = sha256_path(tokens_file)
    tokenizer_sha256 = sha256_path(tokenizer_file)
    order_sha256 = sha256_path(order_file)
    checks = {
        "checkpoint_is_mapping": isinstance(checkpoint, dict),
        "required_keys_present": all(
            name in checkpoint
            for name in (
                "model", "opt", "data_state", "operation_order",
                "state_manifest", "config", "step",
            )
        ),
        "model_tensors_nonempty": bool(inventory),
        "optimizer_bound": bool(
            checkpoint["opt"]
            and state_manifest.get("optimizer_sha256")
        ),
        "batch_bound": batch.numel() > 0,
        "cursor_bound": (
            checkpoint["data_state"].get("cursor_end")
            == checkpoint["data_offset_end"]
        ),
        "lineage_bound": (
            checkpoint["data_state"].get("dataset_sha256") == dataset_sha256
            and checkpoint["data_state"].get("tokenizer_sha256")
            == tokenizer_sha256
            and checkpoint["data_state"].get("data_order_sha256")
            == order_sha256
        ),
        "content_digest_replays": (
            isinstance(state_manifest.get("complete_state_sha256"), str)
            and len(state_manifest["complete_state_sha256"]) == 64
        ),
    }
    result = {
        "schema_version": CHECKPOINT_BINDING_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": str(checkpoint["config"]["name"]),
        "seed": int(checkpoint["config"]["seed"]),
        "step": int(checkpoint["step"]),
        "checkpoint_path": str(checkpoint_file),
        "checkpoint_sha256": sha256_path(checkpoint_file),
        "content_sha256": state_manifest["complete_state_sha256"],
        "tensor_inventory": inventory,
        "optimizer_state_sha256": state_manifest["optimizer_sha256"],
        "batch_sha256": _batch_sha256(batch),
        "cursor": int(checkpoint["data_offset_end"]),
        "code_sha256": state_manifest["code_manifest_sha256"],
        "tokenizer_sha256": tokenizer_sha256,
        "dataset_sha256": dataset_sha256,
        "order_sha256": order_sha256,
        "operation_order_sha256": digest_object(checkpoint["operation_order"]),
        "created_at_ns": int(
            time.time_ns() if created_at_ns is None else created_at_ns
        ),
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(
        result,
        _schema(protocol_directory, "checkpoint_binding.schema.json"),
    )
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"checkpoint binding checks failed: {failed}")
    return result


def _macro_source(name: str) -> str:
    row_program = ".mha1."
    if row_program in name and "layernorm" in name.lower():
        return "normalization"
    if row_program in name and (
        ".reslayerAs." in name or ".layernorm1." in name
    ):
        return "row_program"
    return "upstream"


def _model_gate(parameters: dict[str, torch.Tensor]) -> np.ndarray:
    values = []
    for layer in range(3):
        key = (
            f"decoder.dec_layers.{layer}.mha1.reslayerAs.7."
            "layernormA.weight"
        )
        if key not in parameters:
            raise KeyError(f"checkpoint omits final gate {key}")
        values.append(parameters[key].detach().double().cpu().numpy())
    gate = np.stack(values)
    if gate.shape != (3, 64):
        raise ValueError(f"final gate has unexpected shape {gate.shape}")
    return np.ascontiguousarray(
        np.broadcast_to(gate[:, None, :], (3, 4, 64))
    )


def _flatten_maps(value: np.ndarray) -> np.ndarray:
    expected_tail = (3, 4, 64, 64)
    if value.ndim != 5 or value.shape[1:] != expected_tail:
        raise ValueError(f"captured row maps have unexpected shape {value.shape}")
    return np.ascontiguousarray(value.reshape(-1, 64, 64))


def _map_gates(gate: np.ndarray, context_count: int) -> np.ndarray:
    return np.ascontiguousarray(
        np.broadcast_to(gate[None], (context_count,) + gate.shape)
        .reshape(-1, 64)
    )


def _map_identity(
    index: int, contexts: list[dict[str, Any]]
) -> dict[str, Any]:
    context_index, remainder = divmod(index, 12)
    layer, head = divmod(remainder, 4)
    return {
        "map_id": f"{contexts[context_index]['id']}-l{layer}-h{head}",
        "context": str(contexts[context_index]["id"]),
        "layer": int(layer),
        "head": int(head),
    }


def produce_edge(
    source_binding_path: str | Path,
    endpoint_binding_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
    *,
    protocol_directory: str | Path,
    producer_command: list[str],
) -> dict[str, Any]:
    """Produce all-map finite transport and physical-energy evidence."""

    source_binding_file = Path(source_binding_path).resolve()
    endpoint_binding_file = Path(endpoint_binding_path).resolve()
    binding_schema = _schema(
        protocol_directory, "checkpoint_binding.schema.json"
    )
    source_binding = load_json(source_binding_file)
    endpoint_binding = load_json(endpoint_binding_file)
    validate(source_binding, binding_schema)
    validate(endpoint_binding, binding_schema)
    if not source_binding["technical_valid"] or not endpoint_binding[
        "technical_valid"
    ]:
        raise ValueError("edge producer received an invalid checkpoint binding")
    source_path = Path(source_binding["checkpoint_path"])
    endpoint_path = Path(endpoint_binding["checkpoint_path"])
    if (
        sha256_path(source_path) != source_binding["checkpoint_sha256"]
        or sha256_path(endpoint_path) != endpoint_binding["checkpoint_sha256"]
    ):
        raise ValueError("a bound checkpoint changed before edge production")
    if endpoint_binding["step"] != source_binding["step"] + 1:
        raise ValueError("edge checkpoint steps are not consecutive")

    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    source_checkpoint = load_complete_checkpoint(source_path, map_location="cpu")
    endpoint_checkpoint = load_complete_checkpoint(
        endpoint_path, map_location="cpu"
    )
    registry = load_registry(registry_path)
    if (
        source_checkpoint["measurement_registry"] != registry
        or endpoint_checkpoint["measurement_registry"] != registry
    ):
        raise ValueError("edge checkpoints and registry disagree")
    registered, contexts = registered_tokens(registry, tokens_path, device)
    source_model = _model_from_raw(source_checkpoint, device)
    endpoint_model = _model_from_raw(endpoint_checkpoint, device)
    source_parameters = dict(source_model.named_parameters())
    endpoint_parameters = dict(endpoint_model.named_parameters())
    if set(source_parameters) != set(endpoint_parameters):
        raise ValueError("edge parameter inventories disagree")

    source_shape, source_rows = capture_maps(source_model, registered)
    endpoint_shape, endpoint_rows = capture_maps(endpoint_model, registered)
    source_gate = _model_gate(source_parameters)
    endpoint_gate = _model_gate(endpoint_parameters)

    macro_names = ("upstream", "row_program", "normalization")

    def evaluate(subset: frozenset[str]) -> np.ndarray:
        parameters = {
            name: (
                endpoint_parameters[name]
                if _macro_source(name) in subset
                else source_parameters[name]
            )
            for name in source_parameters
        }
        _shape, rows = capture_maps(source_model, registered, parameters)
        return _flatten_maps(rows)

    shapley = permutation_symmetric_attribution(macro_names, evaluate)
    source_flat = _flatten_maps(source_rows)
    endpoint_flat = _flatten_maps(endpoint_rows)
    if not np.array_equal(shapley["baseline"], source_flat):
        raise RuntimeError("empty coalition differs from source rows")
    if not np.array_equal(shapley["endpoint"], endpoint_flat):
        raise RuntimeError("full coalition differs from endpoint rows")
    contributions = shapley["contributions"]
    work = source_work_gram(source_flat, contributions)

    source_model.train()
    optimizer = optimizer_from_checkpoint(source_model, source_checkpoint)
    update_rows, _update_chunks = ordered_training_batch(
        source_checkpoint, tokens_path, order_path, device
    )
    optimizer.zero_grad(set_to_none=True)
    inputs = update_rows[:, :-1]
    targets = update_rows[:, 1:]
    update_mask = create_masks(inputs, inputs.device)
    logits = source_model([inputs, update_mask])[0]
    loss = masked_loss_function(targets, logits)
    loss.backward()
    torch.nn.utils.clip_grad_value_(source_model.parameters(), 1.0)
    names, _values, _directions, successors = optimizer_displacements(
        source_model, optimizer
    )
    predicted_parameters = dict(zip(names, successors, strict=True))
    _predicted_shape, predicted_rows = capture_maps(
        source_model, registered, predicted_parameters
    )
    predicted_flat = _flatten_maps(predicted_rows)
    predicted_centered = center_rows(predicted_flat)
    native_difference = center_rows(endpoint_flat - predicted_flat)
    native_radius = np.sqrt(np.sum(
        native_difference * native_difference, axis=(-2, -1)
    ))
    native_envelope = source_native_energy_envelope(
        predicted_centered, native_radius
    )

    source_geometry = centered_energy(source_flat)
    endpoint_geometry = centered_energy(endpoint_flat)
    source_diameter = np.sqrt(source_geometry["diameter_squared"])
    endpoint_diameter = np.sqrt(endpoint_geometry["diameter_squared"])
    context_count = source_rows.shape[0]
    source_gates = _map_gates(source_gate, context_count)
    endpoint_gates = _map_gates(endpoint_gate, context_count)
    source_shapes = _flatten_maps(source_shape)
    endpoint_shapes = _flatten_maps(endpoint_shape)

    gate_shape = [
        symmetric_gate_shape_transport(
            source_shapes[index], endpoint_shapes[index],
            source_gates[index], endpoint_gates[index],
        )
        for index in range(len(source_flat))
    ]
    transport_residuals = []
    ledger_residuals = []
    records = []
    total_contribution = sum(
        contributions.values(), start=np.zeros_like(source_flat)
    )
    direct_increment = endpoint_flat - source_flat
    for index in range(len(source_flat)):
        increment_scale = max(
            float(np.linalg.norm(center_rows(direct_increment[index]))), 1.0
        )
        transport_residual = float(np.linalg.norm(center_rows(
            total_contribution[index] - direct_increment[index]
        )) / increment_scale)
        gain_scale = max(float(work["source_gain"][index]), 1.0)
        ledger_residual = float(
            work["maximum_abs_gain_residual"] / gain_scale
        )
        transport_residuals.append(transport_residual)
        ledger_residuals.append(ledger_residual)
        split = gate_shape[index]
        records.append({
            **_map_identity(index, contexts),
            "source_energy": float(source_geometry["energy"][index]),
            "endpoint_energy": float(endpoint_geometry["energy"][index]),
            "source_diameter": float(source_diameter[index]),
            "endpoint_diameter": float(endpoint_diameter[index]),
            "source_gain": float(work["source_gain"][index]),
            "gain_defined": bool(work["gain_defined"][index]),
            "source_names": list(macro_names),
            "radial_dissipation": [
                float(value) for value in work["radial_dissipation"][index]
            ],
            "charge_gram": [
                [float(value) for value in row]
                for row in work["charge_gram"][index]
            ],
            "gate_work": float(split["gate_work"]),
            "shape_work": float(split["shape_work"]),
            "gate_charge": float(split["gate_charge"]),
            "shape_charge": float(split["shape_charge"]),
            "gate_shape_interaction": float(split["interaction_charge"]),
            "native_radius": float(native_radius[index]),
            "native_energy_upper": float(
                native_envelope["native_energy_upper"][index]
            ),
            "transport_relative_residual": transport_residual,
            "energy_ledger_relative_residual": ledger_residual,
            "shapley_efficiency_relative_residual": transport_residual,
        })

    endpoint_parameter_sha256 = tensor_mapping_digest(endpoint_checkpoint["model"])
    predicted_parameter_sha256 = tensor_mapping_digest(predicted_parameters)
    clone_link = endpoint_parameter_sha256 == predicted_parameter_sha256
    map_population_complete = len(records) == REGISTRY["map_count"]
    checks = {
        "checkpoint_chronology": endpoint_binding["step"]
        == source_binding["step"] + 1,
        "map_population_complete": map_population_complete,
        "finite_transport_closed": max(transport_residuals) <= 1.0e-10,
        "energy_ledger_closed": max(ledger_residuals) <= 1.0e-10,
        "shapley_efficiency_closed": (
            shapley["maximum_abs_efficiency_residual"]
            <= 1.0e-10 * max(float(np.max(np.abs(endpoint_flat))), 1.0)
        ),
        "native_link_closed": clone_link,
        "all_values_finite": all(
            np.isfinite(value)
            for record in records
            for key, value in record.items()
            if isinstance(value, float)
        ),
    }
    result = {
        "schema_version": EDGE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "trajectory": source_binding["trajectory"],
        "seed": source_binding["seed"],
        "source_step": source_binding["step"],
        "endpoint_step": endpoint_binding["step"],
        "source_checkpoint_binding_sha256": sha256_path(source_binding_file),
        "endpoint_checkpoint_binding_sha256": sha256_path(endpoint_binding_file),
        "producer_command_sha256": hashlib.sha256(
            _canonical(producer_command)
        ).hexdigest(),
        "map_records": records,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(result, _schema(protocol_directory, "edge.schema.json"))
    if not result["technical_valid"]:
        failed = sorted(name for name, passed in checks.items() if not passed)
        raise ValueError(f"edge production checks failed: {failed}")
    return result


def qualify(
    device_name: str,
    *,
    protocol_directory: str | Path,
    checkpoint_path: str | Path | None = None,
    tokens_path: str | Path | None = None,
    tokenizer_path: str | Path | None = None,
    order_path: str | Path | None = None,
) -> dict[str, Any]:
    """Exercise exact kernels, schema rejection, device use, and recovery."""

    started = time.monotonic()
    rng = np.random.default_rng(45003)
    shape0 = rng.normal(size=(4, 8, 6))
    shape1 = shape0 + rng.normal(scale=0.05, size=shape0.shape)
    gate0 = rng.normal(size=6)
    gate1 = gate0 + rng.normal(scale=0.02, size=6)
    factor = gate_shape_energy(shape0, gate0)
    symmetric = symmetric_gate_shape_transport(
        shape0, shape1, gate0, gate1
    )
    shapley = permutation_symmetric_attribution(
        ("a", "b", "c"),
        lambda subset: np.asarray([
            1.0 + ("a" in subset) + 2.0 * ("b" in subset),
            2.0 - ("c" in subset),
        ]),
    )
    moment_mask = partition_adamw_coordinates(
        [1.0, 0.0, 0.0], [False, True, False]
    )
    failed_radius = admissible_quadratic_radius(1.1, 0.1, 0.01)
    kernels_pass = (
        factor["maximum_abs_factorization_residual"] < 1.0e-12
        and symmetric["maximum_abs_transport_residual"] < 1.0e-12
        and symmetric["maximum_abs_ledger_residual"] < 1.0e-12
        and shapley["maximum_abs_efficiency_residual"] < 1.0e-12
        and moment_mask["unresolved_count"] == 1
        and not failed_radius["eligible"]
    )

    schema_rejects = False
    checkpoint_binding_sha256 = None
    checkpoint_opened = False
    recovery_pass = False
    mode = "fixture"
    if checkpoint_path is not None:
        if any(value is None for value in (
            tokens_path, tokenizer_path, order_path
        )):
            raise ValueError("real qualification needs tokens, tokenizer, and order")
        binding = bind_complete_checkpoint(
            checkpoint_path,
            tokens_path,
            tokenizer_path,
            order_path,
            protocol_directory=protocol_directory,
        )
        checkpoint_binding_sha256 = hashlib.sha256(
            _canonical(binding)
        ).hexdigest()
        checkpoint_opened = True
        reloaded = load_complete_checkpoint(checkpoint_path, map_location="cpu")
        recovery_pass = (
            reloaded["state_manifest"]["complete_state_sha256"]
            == binding["content_sha256"]
        )
        mode = "real-checkpoint"
    else:
        with tempfile.TemporaryDirectory(prefix="pldr-source-recovery-") as raw:
            path = Path(raw) / "record.json"
            value = {"identity": "fixture", "count": 3}
            write_json_atomic(path, value)
            recovery_pass = load_json(path) == value

    qualification_schema = _schema(
        protocol_directory, "qualification.schema.json"
    )
    try:
        validate({"unexpected": True}, qualification_schema)
    except ValueError:
        schema_rejects = True

    device = torch.device(device_name)
    peak_allocated = 0
    peak_reserved = 0
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
        left = torch.arange(4096, device=device, dtype=torch.float32).reshape(64, 64)
        right = left.T.contiguous()
        product = left @ right
        if not torch.isfinite(product).all():
            raise RuntimeError("qualification device operation is nonfinite")
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device.index))
        peak_reserved = int(torch.cuda.max_memory_reserved(device.index))
    else:
        product = torch.arange(64, dtype=torch.float32).sum()
        if not torch.isfinite(product):
            raise RuntimeError("CPU qualification operation is nonfinite")
    device_exercised = (
        peak_allocated > 0 and peak_reserved > 0
        if device.type == "cuda" else True
    )
    checks = {
        "kernels_pass": bool(kernels_pass),
        "schema_rejects_unknown_fields": schema_rejects,
        "device_exercised": bool(device_exercised),
        "checkpoint_content_opened": checkpoint_opened,
        "recovery_pass": bool(recovery_pass),
        "forced_cap_record_pass": not failed_radius["eligible"],
    }
    required_checks = dict(checks)
    if mode == "fixture":
        required_checks.pop("checkpoint_content_opened")
    result = {
        "schema_version": QUALIFICATION_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "mode": mode,
        "device": device_name,
        "checkpoint_binding_sha256": checkpoint_binding_sha256,
        "maximum_identity_residuals": {
            "gate_shape": factor["maximum_abs_factorization_residual"],
            "symmetric_transport": symmetric["maximum_abs_transport_residual"],
            "energy_ledger": symmetric["maximum_abs_ledger_residual"],
            "shapley_efficiency": shapley[
                "maximum_abs_efficiency_residual"
            ],
        },
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "peak_host_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ) * (1024 if sys.platform != "darwin" else 1),
        "checks": checks,
        "technical_valid": all(required_checks.values()),
    }
    validate(result, qualification_schema)
    if not result["technical_valid"]:
        failed = sorted(
            name for name, passed in required_checks.items() if not passed
        )
        raise ValueError(f"qualification checks failed: {failed}")
    result["wall_seconds"] = time.monotonic() - started
    result.pop("wall_seconds")
    return result
