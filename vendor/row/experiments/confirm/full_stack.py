"""Complete row-Jacobian stack construction and registry checks.

The primary coordinate is every entry of every registered row-map Jacobian.
The registry is compact, but its ordering is explicit: within a block the
output coordinate is major and the input coordinate is minor.  Scalar JVP or
VJP summaries can be derived from this vector, but cannot be used to create a
primary registry.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


SCHEMA_VERSION = "pldr-full-row-stack-registry-v1"


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _identifier(value, label):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    if any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
           for character in value):
        raise ValueError(f"{label} contains an invalid character")
    return value


def build_registry(layer_cells):
    """Build the deterministic complete-stack registry.

    ``layer_cells`` maps each criterion-layer index to an iterable of
    ``(cell_id, width)`` pairs.  Every block represents all ``width**2``
    Jacobian entries.  Layers must be contiguous from zero, which prevents a
    shallow architecture from inheriting an invalid fixed layer list.
    """

    if not isinstance(layer_cells, dict) or not layer_cells:
        raise ValueError("layer_cells must be a nonempty mapping")
    layers = sorted(layer_cells)
    if layers != list(range(len(layers))):
        raise ValueError("criterion layers must be every layer from zero to depth-1")
    blocks = []
    offset = 0
    seen = set()
    for layer in layers:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer indices must be integers")
        cells = list(layer_cells[layer])
        if not cells:
            raise ValueError(f"layer {layer} has no registered cover cells")
        for local_index, pair in enumerate(cells):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError("a stack cell must be (cell_id, width)")
            cell_id = _identifier(pair[0], "cell_id")
            width = pair[1]
            if not isinstance(width, int) or isinstance(width, bool) or width < 1:
                raise ValueError("stack widths must be positive integers")
            key = (layer, cell_id)
            if key in seen:
                raise ValueError("stack layer/cell keys must be unique")
            seen.add(key)
            size = width * width
            blocks.append({
                "block_index": len(blocks),
                "layer": layer,
                "cell_id": cell_id,
                "layer_cell_index": local_index,
                "width": width,
                "offset": offset,
                "size": size,
                "coordinate_order": "output_major_input_minor",
            })
            offset += size
    body = {
        "schema_version": SCHEMA_VERSION,
        "criterion_layers": layers,
        "layer_count": len(layers),
        "block_count": len(blocks),
        "dimension": offset,
        "blocks": blocks,
        "primary_coordinate": "complete_full_matrix_row_jacobian_stack",
        "scalar_projection_role": "diagnostic_only",
    }
    body["registry_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    validate_registry(body)
    return body


def validate_registry(registry):
    if not isinstance(registry, dict):
        raise TypeError("stack registry must be a mapping")
    required = {
        "schema_version", "criterion_layers", "layer_count", "block_count",
        "dimension", "blocks", "primary_coordinate",
        "scalar_projection_role", "registry_sha256",
    }
    if set(registry) != required:
        raise ValueError("stack registry has missing or unknown fields")
    if registry["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unknown stack-registry schema")
    if registry["primary_coordinate"] != (
        "complete_full_matrix_row_jacobian_stack"
    ) or registry["scalar_projection_role"] != "diagnostic_only":
        raise ValueError("primary stack was replaced by a lower-dimensional chart")
    unsigned = dict(registry)
    digest = unsigned.pop("registry_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("stack registry digest is malformed")
    if hashlib.sha256(_canonical(unsigned)).hexdigest() != digest:
        raise ValueError("stack registry digest does not replay")
    layers = registry["criterion_layers"]
    if layers != list(range(registry["layer_count"])):
        raise ValueError("stack registry does not cover every architecture layer")
    blocks = registry["blocks"]
    if len(blocks) != registry["block_count"] or not blocks:
        raise ValueError("stack block count is inconsistent")
    offset = 0
    seen = set()
    for index, block in enumerate(blocks):
        expected_fields = {
            "block_index", "layer", "cell_id", "layer_cell_index", "width",
            "offset", "size", "coordinate_order",
        }
        if not isinstance(block, dict) or set(block) != expected_fields:
            raise ValueError("stack block has missing or unknown fields")
        if block["block_index"] != index or block["offset"] != offset:
            raise ValueError("stack block offsets are not contiguous")
        if block["layer"] not in layers:
            raise ValueError("stack block uses an invalid architecture layer")
        _identifier(block["cell_id"], "cell_id")
        key = (block["layer"], block["cell_id"])
        if key in seen:
            raise ValueError("stack registry repeats a layer/cell block")
        seen.add(key)
        width = block["width"]
        if (
            not isinstance(width, int) or isinstance(width, bool) or width < 1
            or block["size"] != width * width
            or block["coordinate_order"] != "output_major_input_minor"
        ):
            raise ValueError("stack block dimensions or ordering are invalid")
        offset += block["size"]
    if offset != registry["dimension"]:
        raise ValueError("stack registry dimension is inconsistent")
    return True


def coordinate(registry, flat_index):
    """Return ``(layer, cell_id, output, input)`` for one stack coordinate."""

    validate_registry(registry)
    if (
        not isinstance(flat_index, int) or isinstance(flat_index, bool)
        or not 0 <= flat_index < registry["dimension"]
    ):
        raise IndexError("stack coordinate is outside the registry")
    for block in registry["blocks"]:
        if flat_index < block["offset"] + block["size"]:
            local = flat_index - block["offset"]
            return (
                block["layer"], block["cell_id"],
                local // block["width"], local % block["width"],
            )
    raise ArithmeticError("stack coordinate lookup failed")


def stack_jacobians(registry, jacobians):
    """Flatten one complete matrix per registered block in registry order."""

    validate_registry(registry)
    if not isinstance(jacobians, dict):
        raise TypeError("jacobians must map (layer, cell_id) to a matrix")
    expected = {(row["layer"], row["cell_id"]) for row in registry["blocks"]}
    if set(jacobians) != expected:
        missing = sorted(expected - set(jacobians))
        extra = sorted(set(jacobians) - expected)
        raise ValueError(
            f"full stack key mismatch; missing={missing}, extra={extra}")
    values = []
    for block in registry["blocks"]:
        key = (block["layer"], block["cell_id"])
        matrix = np.asarray(jacobians[key], dtype=float)
        expected_shape = (block["width"], block["width"])
        if matrix.shape != expected_shape or not np.isfinite(matrix).all():
            raise ValueError(f"Jacobian {key} is nonfinite or misdimensioned")
        values.append(matrix.reshape(-1))
    result = np.concatenate(values)
    if result.shape != (registry["dimension"],):
        raise ArithmeticError("full stack dimension did not close")
    return result


def unstack_jacobians(registry, stack):
    validate_registry(registry)
    stack = np.asarray(stack, dtype=float)
    if stack.shape != (registry["dimension"],) or not np.isfinite(stack).all():
        raise ValueError("stack vector is nonfinite or misdimensioned")
    result = {}
    for block in registry["blocks"]:
        start = block["offset"]
        stop = start + block["size"]
        result[(block["layer"], block["cell_id"])] = stack[start:stop].reshape(
            block["width"], block["width"])
    return result


def stack_to_layer_operator_bounds(registry, stack):
    """Compute diagnostic full-matrix operator norms for every stack block."""

    matrices = unstack_jacobians(registry, stack)
    result = {layer: [] for layer in registry["criterion_layers"]}
    for (layer, cell_id), matrix in matrices.items():
        result[layer].append({
            "cell_id": cell_id,
            "operator_norm": float(np.linalg.norm(matrix, ord=2)),
        })
    if any(not rows for rows in result.values()):
        raise ArithmeticError("full stack omitted an architecture layer")
    return result


def validate_block_slices(dimension, block_slices):
    """Validate a complete direct-sum partition of a vector coordinate."""

    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    rows = list(block_slices)
    if not rows:
        raise ValueError("block partition is empty")
    cursor = 0
    names = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"name", "start", "stop"}:
            raise ValueError("block slice has missing or unknown fields")
        name = _identifier(row["name"], "block name")
        if name in names:
            raise ValueError("block slice names are duplicated")
        names.add(name)
        if row["start"] != cursor or not isinstance(row["stop"], int):
            raise ValueError("block slices are not contiguous")
        if row["stop"] <= row["start"]:
            raise ValueError("block slice is empty")
        cursor = row["stop"]
    if cursor != dimension:
        raise ValueError("block slices do not cover the full coordinate")
    return True


def outward_upper(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("an outward upper bound must be finite and nonnegative")
    return value if value == 0.0 else math.nextafter(value, math.inf)
