"""Live source-state capture for the PLDR row-map collapse certificate.

The source checkpoint, exact next minibatch, clipped AdamW direction, and
registered tokens determine every value returned here.  No checkpoint at the
next training step and no row-map value read from such a checkpoint is used.
"""

from __future__ import annotations

import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.func import functional_call, jvp


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    sha256_path,
    tensor_mapping_digest,
)
from confirm.finite_increment_live import (  # noqa: E402
    load_checkpoint,
    optimizer_from_checkpoint,
    ordered_training_batch,
    registered_tokens,
)
from confirm.source_restoring_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    MASK_CONTRACT,
    OPTIMIZER,
    REGISTRY_SCHEMA,
    TRAJECTORIES,
    training_campaign_spec,
)
from confirm.row_map_live import _model_from_raw  # noqa: E402
from train_run import create_masks, masked_loss_function  # noqa: E402
from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)


SOURCE_SCHEMA = "pldr-source-restoring-source-v1"


def load_registry(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "campaign_id", "construction", "validation",
        "context_length", "layers", "heads", "rows_per_map",
        "pairs_per_map", "pair_rule", "cross_map_pairs_forbidden",
        "dataset_sha256", "tokenizer_sha256", "registry_sha256",
        "total_chunks", "trainer_probe_reserve_chunks",
        "trainer_training_limit",
    }
    if set(value) != required:
        raise ValueError("source-restoring registry has missing or unknown fields")
    if (
        value["schema_version"] != REGISTRY_SCHEMA
        or value["campaign_id"] != CAMPAIGN_ID
        or value["context_length"] != 256
        or value["layers"] != [0, 1, 2]
        or value["heads"] != [0, 1, 2, 3]
        or value["rows_per_map"] != 64
        or value["pairs_per_map"] != 2016
        or value["cross_map_pairs_forbidden"] is not True
        or len(value["construction"]) != 8
        or len(value["validation"]) != 16
    ):
        raise ValueError("source-restoring registry geometry is invalid")
    unsigned = dict(value)
    recorded = unsigned.pop("registry_sha256")
    if digest_object(unsigned) != recorded:
        raise ValueError("source-restoring registry digest does not replay")
    contexts = list(value["construction"]) + list(value["validation"])
    expected_ids = [
        *(f"c{index:02d}" for index in range(8)),
        *(f"v{index:02d}" for index in range(8, 24)),
    ]
    if (
        any(set(row) != {"id", "chunk_index"} for row in contexts)
        or [row["id"] for row in contexts] != expected_ids
        or len({int(row["chunk_index"]) for row in contexts}) != 24
        or any(
            int(row["chunk_index"]) < value["trainer_training_limit"]
            or int(row["chunk_index"]) >= value["total_chunks"]
            for row in contexts
        )
        or value["trainer_training_limit"]
        + value["trainer_probe_reserve_chunks"]
        != value["total_chunks"]
        or value["pair_rule"]
        != "within-context-layer-head-all-unordered-pairs-v1"
    ):
        raise ValueError("source-restoring registry population is invalid")
    return value


def _model_mask(inputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Reproduce the trainer mask, including the padding-token channel."""

    if dtype != torch.float32:
        raise ValueError("the registered trainer mask requires native float32")
    return create_masks(inputs, inputs.device)


def _source_loss(model: torch.nn.Module, rows: torch.Tensor) -> torch.Tensor:
    inputs = rows[:, :-1]
    targets = rows[:, 1:]
    logits = model([
        inputs, _model_mask(inputs, next(model.parameters()).dtype)
    ])[0]
    return masked_loss_function(targets, logits)


def validate_source_checkpoint(
    checkpoint: dict[str, Any],
    registry: dict[str, Any],
    tokens_path: str | Path,
) -> dict[str, Any]:
    """Verify the registry, native-training campaign, and trajectory."""

    if checkpoint.get("measurement_registry") != registry:
        raise ValueError("checkpoint and source-restoring registry disagree")
    token_digest = sha256_path(Path(tokens_path).resolve())
    data_state = checkpoint.get("data_state", {})
    if (
        registry["dataset_sha256"] != token_digest
        or data_state.get("dataset_sha256") != token_digest
        or data_state.get("tokenizer_sha256") != registry["tokenizer_sha256"]
    ):
        raise ValueError("checkpoint, registry, and token archive disagree")
    binding = data_state.get("campaign_binding")
    required = {
        "campaign_id",
        "campaign_spec_sha256",
        "campaign_spec_file_sha256",
        "campaign_spec_path",
    }
    expected_spec = training_campaign_spec(registry["registry_sha256"])
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("checkpoint campaign binding is missing or malformed")
    campaign_path = Path(binding["campaign_spec_path"])
    if campaign_path.is_symlink() or not campaign_path.is_file():
        raise ValueError("checkpoint campaign specification is unavailable")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if (
        binding["campaign_id"] != CAMPAIGN_ID
        or binding["campaign_spec_sha256"] != expected_spec["spec_sha256"]
        or binding["campaign_spec_file_sha256"] != sha256_path(campaign_path)
        or campaign != expected_spec
    ):
        raise ValueError("checkpoint campaign identity or digest disagrees")
    config = checkpoint.get("config", {})
    expected_config = {
        "layers": ARCHITECTURE["layers"],
        "heads": ARCHITECTURE["heads"],
        "dk": ARCHITECTURE["head_width"],
        "adff": 170,
        "batch": OPTIMIZER["batch_size"],
        "ctx": OPTIMIZER["context_length"],
        "lr": OPTIMIZER["learning_rate"],
        "warmup": OPTIMIZER["warmup_updates"],
        "optimizer": "adamw",
        "wd": OPTIMIZER["weight_decay"],
        "clip": OPTIMIZER["gradient_clip_value"],
        "const_lr": OPTIMIZER["constant_rate_after_warmup"],
        "probe_region": "global",
        "accum": 1,
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("checkpoint training configuration changed")
    trajectory = next(
        (
            row
            for row in TRAJECTORIES
            if row["seed"] == config.get("seed")
            and row["name"] == config.get("name")
        ),
        None,
    )
    if trajectory is None:
        raise ValueError("checkpoint trajectory is outside the frozen campaign")
    return dict(trajectory)


def _layer_rows(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    layer: int,
    parameters: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Return physical rows in context, head, row, feature order."""

    captured: list[torch.Tensor] = []
    norm = model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA

    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        captured.append(output)

    handle = norm.register_forward_hook(hook)
    inputs = tokens[:, :-1]
    mask = _model_mask(inputs, next(model.parameters()).dtype)
    try:
        if parameters is None:
            model([inputs, mask])
        else:
            functional_call(model, parameters, ([inputs, mask],), strict=False)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("the final physical row-map hook did not fire once")
    rows = captured[0]
    expected = (
        len(tokens), ARCHITECTURE["heads"], ARCHITECTURE["rows_per_map"],
        ARCHITECTURE["head_width"],
    )
    if tuple(rows.shape) != expected:
        raise ValueError(
            "registered physical rows have shape "
            f"{tuple(rows.shape)}, expected {expected}"
        )
    return rows


def capture_physical_maps(
    model: torch.nn.Module, tokens: torch.Tensor
) -> np.ndarray:
    """Capture every registered physical map with the trainer mask."""

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            layers = [
                _layer_rows(model, tokens, layer)
                for layer in range(ARCHITECTURE["layers"])
            ]
    finally:
        model.train(was_training)
    physical = torch.stack(layers, dim=1)
    expected = (
        len(tokens),
        ARCHITECTURE["layers"],
        ARCHITECTURE["heads"],
        ARCHITECTURE["rows_per_map"],
        ARCHITECTURE["head_width"],
    )
    if tuple(physical.shape) != expected:
        raise ValueError("registered physical map geometry changed")
    value = physical.detach().double().cpu().numpy()
    return np.ascontiguousarray(
        value.reshape(-1, value.shape[-2], value.shape[-1])
    )


def parameter_block(name: str, layer: int) -> str:
    """Assign every parameter to one of three disjoint source-work blocks."""

    row_prefix = f"decoder.dec_layers.{layer}.mha1."
    final_prefix = row_prefix + "reslayerAs.7.layernormA."
    if name.startswith(final_prefix):
        return "gate"
    if name.startswith(row_prefix + "layernorm1.") or name.startswith(
        row_prefix + "reslayerAs."
    ):
        return "shape"
    return "input"


def partition_directions(
    names: tuple[str, ...],
    directions: tuple[torch.Tensor, ...],
    layer: int,
) -> dict[str, tuple[torch.Tensor, ...]]:
    blocks: dict[str, list[torch.Tensor]] = {
        "gate": [], "shape": [], "input": [],
    }
    for name, direction in zip(names, directions, strict=True):
        owner = parameter_block(name, layer)
        for block in blocks:
            blocks[block].append(
                direction if block == owner else torch.zeros_like(direction)
            )
    result = {name: tuple(value) for name, value in blocks.items()}
    for index, direction in enumerate(directions):
        reconstructed = sum(
            (result[block][index] for block in ("gate", "shape", "input")),
            start=torch.zeros_like(direction),
        )
        if not torch.equal(reconstructed, direction):
            raise AssertionError("parameter-direction partition is not exhaustive")
    return result


def partitioned_row_jvp(
    model: torch.nn.Module,
    tokens: torch.Tensor,
    layer: int,
    names: tuple[str, ...],
    values: tuple[torch.Tensor, ...],
    directions: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], float]:
    """Evaluate source rows and the three exact linear JVP contributions."""

    blocks = partition_directions(names, directions, layer)

    def row_function(*local: torch.Tensor) -> torch.Tensor:
        return _layer_rows(
            model, tokens, layer, dict(zip(names, local, strict=True))
        )

    source: torch.Tensor | None = None
    velocities: dict[str, torch.Tensor] = {}
    for block in ("gate", "shape", "input"):
        local_source, local_velocity = jvp(row_function, values, blocks[block])
        if source is None:
            source = local_source
        elif not torch.equal(source, local_source):
            raise RuntimeError("source primal changed between JVP blocks")
        velocities[block] = local_velocity
    assert source is not None
    _whole_source, whole_velocity = jvp(row_function, values, directions)
    residual = whole_velocity - sum(
        velocities.values(), start=torch.zeros_like(whole_velocity)
    )
    scale = max(float(torch.max(torch.abs(whole_velocity)).detach().cpu()), 1.0)
    maximum_residual = float(torch.max(torch.abs(residual)).detach().cpu())
    if maximum_residual > 4096.0 * torch.finfo(whole_velocity.dtype).eps * scale:
        raise RuntimeError("partitioned JVPs do not reconstruct the complete direction")
    return source, velocities, maximum_residual


def capture_source_state(
    checkpoint_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    order_path: str | Path,
    device_name: str,
) -> dict[str, Any]:
    """Capture all 288 maps and their source-state work decomposition."""

    started = time.monotonic()
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    checkpoint = load_checkpoint(
        checkpoint_path, require_optimizer=True, device="cpu"
    )
    registry = load_registry(registry_path)
    trajectory = validate_source_checkpoint(checkpoint, registry, tokens_path)
    model = _model_from_raw(checkpoint, device)
    optimizer = optimizer_from_checkpoint(model, checkpoint)
    update_rows, update_chunks = ordered_training_batch(
        checkpoint, tokens_path, order_path, device
    )
    registered_rows, contexts = registered_tokens(registry, tokens_path, device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = _source_loss(model, update_rows)
    loss.backward()
    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
    names, values, directions, _endpoint = optimizer_displacements(
        model, optimizer
    )
    model.eval()

    source_layers = []
    velocity_layers = {"gate": [], "shape": [], "input": []}
    residuals = []
    for layer in range(ARCHITECTURE["layers"]):
        source, velocities, residual = partitioned_row_jvp(
            model, registered_rows, layer, names, values, directions
        )
        source_layers.append(source.detach().cpu())
        for block in velocity_layers:
            velocity_layers[block].append(velocities[block].detach().cpu())
        residuals.append(residual)

    def pack(layers: list[torch.Tensor]) -> np.ndarray:
        value = torch.stack(layers, dim=1)
        return np.ascontiguousarray(value.reshape(
            -1, ARCHITECTURE["rows_per_map"], ARCHITECTURE["head_width"]
        ).numpy())

    cuda = device.type == "cuda"
    return {
        "source_rows": pack(source_layers),
        "gate_velocity_rows": pack(velocity_layers["gate"]),
        "shape_velocity_rows": pack(velocity_layers["shape"]),
        "input_velocity_rows": pack(velocity_layers["input"]),
        "checkpoint": checkpoint,
        "registry": registry,
        "trajectory": trajectory,
        "contexts": contexts,
        "update_chunks": update_chunks,
        "source_loss": float(loss.detach().cpu()),
        "source_parameter_sha256": tensor_mapping_digest(
            dict(zip(names, values, strict=True))
        ),
        "maximum_jvp_additivity_residual": max(residuals),
        "resources": {
            "wall_seconds": time.monotonic() - started,
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device.index))
                if cuda else 0
            ),
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(device.index))
                if cuda else 0
            ),
            "peak_host_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ) * (1024 if sys.platform != "darwin" else 1),
        },
        "parents": {
            "checkpoint_path": str(Path(checkpoint_path).resolve()),
            "checkpoint_sha256": sha256_path(Path(checkpoint_path).resolve()),
            "registry_path": str(Path(registry_path).resolve()),
            "registry_sha256": sha256_path(Path(registry_path).resolve()),
            "tokens_path": str(Path(tokens_path).resolve()),
            "tokens_sha256": sha256_path(Path(tokens_path).resolve()),
            "order_path": str(Path(order_path).resolve()),
            "order_sha256": sha256_path(Path(order_path).resolve()),
            "campaign_spec_path": checkpoint["data_state"]["campaign_binding"][
                "campaign_spec_path"
            ],
            "campaign_spec_sha256": checkpoint["data_state"]["campaign_binding"][
                "campaign_spec_file_sha256"
            ],
        },
    }
