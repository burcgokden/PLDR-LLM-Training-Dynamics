"""Low-storage native observer for source-resolved row-map campaigns."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from confirm.finite_increment import per_map_diameters
from confirm.source_resolved_energy import center_rows
from confirm.source_resolved_specs import CAMPAIGN_ID, TIMEPOINT_SCHEMA


POINT_SCHEMA = TIMEPOINT_SCHEMA


def _json_list(value: Any) -> list[Any]:
    """Convert one numeric array to a JSON-native list."""

    result = np.asarray(value).tolist()
    if not isinstance(result, list):
        raise TypeError("source-resolved array did not produce a JSON list")
    return result


def _causal_mask(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.triu(
        torch.ones(length, length, device=device, dtype=dtype), diagonal=1
    )[None, None]


def _capture_rows(
    model: torch.nn.Module, token_rows: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    """Capture pre-gate normalized shapes and physical rows in native order."""

    shapes: list[list[torch.Tensor]] = [[] for _ in range(3)]
    outputs: list[list[torch.Tensor]] = [[] for _ in range(3)]
    handles = []
    for layer in range(3):
        layernorm = (
            model.decoder.dec_layers[layer].mha1.reslayerAs[-1].layernormA
        )

        def hook(
            module: torch.nn.Module,
            inputs: tuple[Any, ...],
            output: torch.Tensor,
            *,
            layer_index: int = layer,
        ) -> None:
            value = inputs[0]
            centered = value - value.mean(dim=-1, keepdim=True)
            shape = centered / torch.sqrt(
                module.eps
                + torch.mean(centered * centered, dim=-1, keepdim=True)
            )
            shapes[layer_index].append(shape.detach().double().cpu())
            outputs[layer_index].append(output.detach().double().cpu())

        handles.append(layernorm.register_forward_hook(hook))
    training = model.training
    model.eval()
    try:
        inputs = token_rows[:, :-1]
        mask = _causal_mask(
            inputs.shape[1], inputs.device, next(model.parameters()).dtype
        )
        with torch.no_grad():
            model([inputs, mask])
    finally:
        for handle in handles:
            handle.remove()
        model.train(training)
    if any(len(values) != 1 for values in shapes + outputs):
        raise RuntimeError("row-map observer hooks did not fire exactly once")
    shape = torch.stack([values[0] for values in shapes], dim=1).numpy()
    physical = torch.stack([values[0] for values in outputs], dim=1).numpy()
    expected = (len(token_rows), 3, 4, 64, 64)
    if shape.shape != expected or physical.shape != expected:
        raise ValueError("row-map observer received an unexpected architecture")
    return shape, physical


def _iswiglu(value: np.ndarray) -> np.ndarray:
    sigmoid = np.empty_like(value)
    positive = value >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    sigmoid[~positive] = exponential / (1.0 + exponential)
    return value * value * sigmoid


def _plga_quotient(
    model: torch.nn.Module, physical: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure fixed-parameter PLGA quotient response and row-constant force."""

    ratios = []
    responses = []
    forces = []
    for layer in range(3):
        module = model.decoder.dec_layers[layer].mha1.plgatt_layer
        rows = np.asarray(physical[:, layer], dtype=np.float64)
        weight = module.Wlst.detach().double().cpu().numpy()[None]
        bias = module.blst.detach().double().cpu().numpy()[None]
        exponent = module.pwlst.detach().double().cpu().numpy()[None]
        coupling = module.alst.detach().double().cpu().numpy()[None]
        coupling_bias = module.balst.detach().double().cpu().numpy()[None]

        def apply(value: np.ndarray) -> np.ndarray:
            preactivation = weight @ value + bias
            base = _iswiglu(preactivation) + 1.0e-9
            powered = np.exp(exponent * np.log(base))
            result = coupling @ powered + coupling_bias
            if not np.isfinite(result).all():
                raise FloatingPointError("native PLGA quotient is nonfinite")
            return result

        constant = np.broadcast_to(
            rows.mean(axis=-2, keepdims=True), rows.shape
        )
        direct = apply(rows)
        baseline = apply(constant)
        numerator = np.linalg.norm(
            center_rows(direct - baseline), axis=(-2, -1)
        )
        denominator = np.linalg.norm(center_rows(rows), axis=(-2, -1))
        ratio = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0.0,
        )
        ratios.append(ratio)
        responses.append(numerator)
        forces.append(np.linalg.norm(center_rows(baseline), axis=(-2, -1)))
    return (
        np.stack(ratios, axis=1).reshape(-1),
        np.stack(responses, axis=1).reshape(-1),
        np.stack(forces, axis=1).reshape(-1),
    )


def record_timepoint(
    model: torch.nn.Module,
    token_rows: torch.Tensor,
    *,
    step: int,
    registry_sha256: str,
    campaign_id: str = CAMPAIGN_ID,
    point_schema: str = POINT_SCHEMA,
    expected_map_count: int = 288,
    include_absolute_plga: bool = False,
) -> dict[str, Any]:
    """Return one map summary population without retaining row tensors."""

    shape, physical = _capture_rows(model, token_rows)
    maps = physical.reshape(-1, 64, 64)
    shape_maps = shape.reshape(-1, 64, 64)
    centered = center_rows(maps)
    centered_shape = center_rows(shape_maps)
    layer_gate = np.stack([
        model.decoder.dec_layers[layer].mha1.reslayerAs[-1]
        .layernormA.weight.detach().double().cpu().numpy()
        for layer in range(3)
    ])
    map_gate = np.broadcast_to(
        layer_gate[None, :, None, :], (len(shape), 3, 4, 64)
    ).reshape(-1, 64)
    factorized = centered_shape * map_gate[:, None, :]
    factorization_scale = max(float(np.linalg.norm(centered)), 1.0)
    factorization_relative_residual = float(
        np.linalg.norm(centered - factorized) / factorization_scale
    )
    energy = np.sum(centered * centered, axis=(-2, -1))
    coordinate_energy = np.sum(centered * centered, axis=-2)
    coordinate_shape_energy = np.sum(
        centered_shape * centered_shape, axis=-2
    )
    diameter = per_map_diameters(maps)
    quotient, response, force = _plga_quotient(model, physical)
    arrays = (
        energy,
        coordinate_energy,
        coordinate_shape_energy,
        quotient,
        response,
        force,
        layer_gate,
        np.asarray([factorization_relative_residual]),
    )
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("source-resolved timepoint is nonfinite")
    if len(energy) != expected_map_count or diameter["cross_map_pairs_formed"]:
        raise ValueError("source-resolved timepoint population changed")
    result = {
        "schema_version": str(point_schema),
        "campaign_id": str(campaign_id),
        "step": int(step),
        "registry_sha256": str(registry_sha256),
        "map_order": "context-major-layer-major-head-major-v1",
        "energy": energy.tolist(),
        "diameter_squared": _json_list(diameter["diameter_squared"]),
        "maximizing_pairs": _json_list(diameter["maximizing_pairs"]),
        "gate_shape_coordinate_energy": coordinate_energy.tolist(),
        "normalized_shape_coordinate_energy": coordinate_shape_energy.tolist(),
        "layer_final_gate": layer_gate.tolist(),
        "coordinate_factorization_relative_residual": (
            factorization_relative_residual
        ),
        "plga_quotient_ratio": quotient.tolist(),
        "plga_force": force.tolist(),
        "cross_map_pairs_formed": False,
    }
    if include_absolute_plga:
        result["plga_absolute_response"] = response.tolist()
    return result
