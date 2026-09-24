"""Low-storage native row-map diameter time-course capture."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from confirm.finite_increment import per_map_diameters


def registered_token_rows(
    registry: dict[str, Any],
    token_chunks: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    contexts = list(registry["construction"]) + list(registry["validation"])
    indices = np.asarray([int(row["chunk_index"]) for row in contexts])
    if (
        indices.ndim != 1
        or len(indices) != 24
        or np.any(indices < 0)
        or np.any(indices >= len(token_chunks))
    ):
        raise ValueError("finite-increment time-course registry is invalid")
    return torch.from_numpy(
        token_chunks[indices].astype(np.int64, copy=True)
    ).to(device)


def physical_row_maps(
    model: torch.nn.Module, token_rows: torch.Tensor
) -> np.ndarray:
    outputs: list[list[torch.Tensor]] = [[] for _ in range(3)]
    handles = []
    for layer in range(3):
        layernorm = model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA

        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: torch.Tensor,
            *,
            layer_index: int = layer,
        ) -> None:
            outputs[layer_index].append(output.detach().cpu())

        handles.append(layernorm.register_forward_hook(hook))
    was_training = model.training
    model.eval()
    try:
        inputs = token_rows[:, :-1]
        mask = torch.triu(
            torch.ones(
                inputs.shape[1], inputs.shape[1],
                device=inputs.device,
                dtype=next(model.parameters()).dtype,
            ),
            diagonal=1,
        )[None, None]
        with torch.no_grad():
            model([inputs, mask])
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    if any(len(layer) != 1 for layer in outputs):
        raise RuntimeError("finite-increment row-map hook did not fire once")
    physical = torch.stack([outputs[layer][0] for layer in range(3)], dim=1)
    expected = (len(token_rows), 3, 4, 64, 64)
    if tuple(physical.shape) != expected:
        raise ValueError(
            f"time-course row maps have {tuple(physical.shape)}, expected {expected}"
        )
    return physical.double().numpy()


def record(
    model: torch.nn.Module,
    token_rows: torch.Tensor,
    *,
    step: int,
    registry_sha256: str,
) -> dict[str, Any]:
    physical = physical_row_maps(model, token_rows)
    maps = physical.reshape(-1, physical.shape[-2], physical.shape[-1])
    diameter = per_map_diameters(maps)
    return {
        "schema_version": "pldr-finite-increment-timecourse-point-v1",
        "step": int(step),
        "registry_sha256": registry_sha256,
        "map_order": "context-major-layer-major-head-major-v1",
        "diameter_squared": [
            float(value) for value in diameter["diameter_squared"]
        ],
        "maximizing_pairs": diameter["maximizing_pairs"],
        "aggregate_diameter_squared": diameter["aggregate_diameter_squared"],
        "cross_map_pairs_formed": diameter["cross_map_pairs_formed"],
    }
