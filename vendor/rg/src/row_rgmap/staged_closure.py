"""Exact fingerprints for staged predictive-state closure experiments.

The functions in this module deliberately separate three objects:

* a row-energy observation;
* optimizer state, either in raw coordinates or in a canonical representative
  of the measured layer-sign gauge; and
* the complete transition state, including model coordinates and random state.

The canonical representative is only for the registered ``(Z/2Z)^L`` action.
It is not a claim that this is the complete symmetry group of PLDR training.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np


GaugePredicate = Callable[[str], bool]
GaugeLayer = Callable[[str], int]


def _digest_field(digest: Any, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


def nested_sha256(value: Any) -> str:
    """Hash a nested Torch/NumPy/Python value without serialization metadata."""

    import torch

    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            _digest_field(digest, b"tensor")
            _digest_field(digest, str(array.dtype).encode("ascii"))
            _digest_field(
                digest,
                json.dumps(list(array.shape), separators=(",", ":")).encode(
                    "ascii"
                ),
            )
            _digest_field(digest, array.tobytes(order="C"))
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            _digest_field(digest, b"ndarray")
            _digest_field(digest, str(array.dtype).encode("ascii"))
            _digest_field(
                digest,
                json.dumps(list(array.shape), separators=(",", ":")).encode(
                    "ascii"
                ),
            )
            _digest_field(digest, array.tobytes(order="C"))
        elif isinstance(item, dict):
            _digest_field(digest, b"dict")
            keys = sorted(item, key=lambda key: (type(key).__name__, repr(key)))
            for key in keys:
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            _digest_field(digest, type(item).__name__.encode("ascii"))
            for member in item:
                visit(member)
        elif isinstance(item, bytes):
            _digest_field(digest, b"bytes")
            _digest_field(digest, item)
        elif item is None or isinstance(item, (bool, int, float, str)):
            _digest_field(digest, type(item).__name__.encode("ascii"))
            _digest_field(
                digest,
                json.dumps(item, allow_nan=False, separators=(",", ":")).encode(
                    "utf-8"
                ),
            )
        else:
            raise TypeError(f"unsupported fingerprint type: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def optimizer_states_by_name(
    checkpoint: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Bind serialized optimizer states to model parameter names."""

    names = list(checkpoint["model"])
    optimizer = checkpoint["optimizer"]
    parameter_ids = [
        parameter_id
        for group in optimizer["param_groups"]
        for parameter_id in group["params"]
    ]
    states = optimizer["state"]
    if (
        len(parameter_ids) != len(names)
        or len(set(parameter_ids)) != len(names)
        or set(parameter_ids) != set(states)
    ):
        raise ValueError("optimizer/model parameter registry is not bijective")
    rows = []
    for name, parameter_id in zip(names, parameter_ids):
        state = states[parameter_id]
        for key in ("exp_avg", "exp_avg_sq"):
            if key not in state or tuple(state[key].shape) != tuple(
                checkpoint["model"][name].shape
            ):
                raise ValueError(f"optimizer {key} does not match {name}")
        rows.append((name, state))
    return rows


def gauge_canonical_signs(
    checkpoint: dict[str, Any],
    is_gauge_parameter: GaugePredicate,
    gauge_layer: GaugeLayer,
) -> dict[int, int]:
    """Choose one exact representative of each registered layer-sign orbit."""

    import torch

    names_by_layer: dict[int, list[str]] = {}
    for name in checkpoint["model"]:
        if is_gauge_parameter(name):
            names_by_layer.setdefault(gauge_layer(name), []).append(name)
    if not names_by_layer:
        raise ValueError("registered gauge support is empty")
    signs = {}
    for layer, names in sorted(names_by_layer.items()):
        pivot = None
        for name in sorted(names):
            values = checkpoint["model"][name].detach().reshape(-1)
            nonzero = torch.nonzero(values != 0, as_tuple=False).reshape(-1)
            if nonzero.numel():
                pivot = values[int(nonzero[0])]
                break
        if pivot is None:
            raise ValueError(f"layer {layer} has no nonzero gauge pivot")
        signs[layer] = -1 if bool(pivot < 0) else 1
    return signs


def _canonical_tensor(
    value: Any,
    name: str,
    signs: dict[int, int],
    is_gauge_parameter: GaugePredicate,
    gauge_layer: GaugeLayer,
) -> Any:
    if not is_gauge_parameter(name):
        return value
    sign = signs[gauge_layer(name)]
    return value if sign == 1 else -value


def optimizer_state_for_fingerprint(
    checkpoint: dict[str, Any],
    *,
    canonicalize_first_moment: bool,
    is_gauge_parameter: GaugePredicate,
    gauge_layer: GaugeLayer,
) -> dict[str, Any]:
    """Return the complete AdamW state with IDs replaced by parameter names."""

    signs = (
        gauge_canonical_signs(checkpoint, is_gauge_parameter, gauge_layer)
        if canonicalize_first_moment
        else {}
    )
    state_by_name = {}
    for name, state in optimizer_states_by_name(checkpoint):
        row = dict(state)
        if canonicalize_first_moment:
            row["exp_avg"] = _canonical_tensor(
                state["exp_avg"],
                name,
                signs,
                is_gauge_parameter,
                gauge_layer,
            )
        state_by_name[name] = row

    names = list(checkpoint["model"])
    parameter_ids = [
        parameter_id
        for group in checkpoint["optimizer"]["param_groups"]
        for parameter_id in group["params"]
    ]
    names_by_id = dict(zip(parameter_ids, names))
    groups = []
    for group in checkpoint["optimizer"]["param_groups"]:
        rewritten = {key: value for key, value in group.items() if key != "params"}
        rewritten["parameter_names"] = [names_by_id[value] for value in group["params"]]
        groups.append(rewritten)
    return {"state_by_name": state_by_name, "param_groups": groups}


def optimizer_state_sha256(
    checkpoint: dict[str, Any],
    *,
    canonicalize_first_moment: bool,
    is_gauge_parameter: GaugePredicate,
    gauge_layer: GaugeLayer,
) -> str:
    return nested_sha256(
        optimizer_state_for_fingerprint(
            checkpoint,
            canonicalize_first_moment=canonicalize_first_moment,
            is_gauge_parameter=is_gauge_parameter,
            gauge_layer=gauge_layer,
        )
    )


def canonical_transition_state_sha256(
    checkpoint: dict[str, Any],
    *,
    phase: dict[str, Any],
    is_gauge_parameter: GaugePredicate,
    gauge_layer: GaugeLayer,
) -> str:
    """Hash the transition state modulo the registered layer-sign action."""

    signs = gauge_canonical_signs(checkpoint, is_gauge_parameter, gauge_layer)
    canonical_model = {
        name: _canonical_tensor(
            value, name, signs, is_gauge_parameter, gauge_layer
        )
        for name, value in checkpoint["model"].items()
    }
    canonical_optimizer = optimizer_state_for_fingerprint(
        checkpoint,
        canonicalize_first_moment=True,
        is_gauge_parameter=is_gauge_parameter,
        gauge_layer=gauge_layer,
    )
    required = (
        "step",
        "trajectory_id",
        "seed",
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "cuda_random_state",
    )
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"checkpoint transition state is incomplete: {missing}")
    return nested_sha256(
        {
            "model": canonical_model,
            "optimizer": canonical_optimizer,
            "phase": phase,
            **{key: checkpoint[key] for key in required},
        }
    )


def energy_vector_sha256(energy: np.ndarray) -> str:
    array = np.asarray(energy)
    if array.dtype != np.dtype(np.float64) or array.ndim != 1:
        raise ValueError("candidate energy must be a one-dimensional float64 vector")
    if not np.all(np.isfinite(array)) or np.any(array < 0):
        raise ValueError("candidate energy vector is invalid")
    return nested_sha256(array)


def candidate_signatures(
    checkpoint: dict[str, Any],
    energy: np.ndarray,
    row_map_sha256: str,
    *,
    phase: dict[str, Any],
    is_gauge_parameter: GaugePredicate,
    gauge_layer: GaugeLayer,
) -> dict[str, dict[str, Any]]:
    """Construct the three prespecified reduced-state candidates."""

    if not isinstance(row_map_sha256, str) or len(row_map_sha256) != 64:
        raise ValueError("row-map fingerprint is malformed")
    energy_digest = energy_vector_sha256(energy)
    raw_optimizer = optimizer_state_sha256(
        checkpoint,
        canonicalize_first_moment=False,
        is_gauge_parameter=is_gauge_parameter,
        gauge_layer=gauge_layer,
    )
    canonical_optimizer = optimizer_state_sha256(
        checkpoint,
        canonicalize_first_moment=True,
        is_gauge_parameter=is_gauge_parameter,
        gauge_layer=gauge_layer,
    )
    return {
        "energy-vector": {"energy_vector_sha256": energy_digest},
        "energy-raw-adamw-phase": {
            "energy_vector_sha256": energy_digest,
            "optimizer_state_sha256": raw_optimizer,
            "phase": phase,
        },
        "row-map-gauge-adamw-phase": {
            "row_map_sha256": row_map_sha256,
            "gauge_canonical_optimizer_state_sha256": canonical_optimizer,
            "phase": phase,
        },
    }
