"""Bitwise native reproduction with a separate conversion-radius ledger."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind not in "biufc":
        raise TypeError("determinism leaves must be numeric arrays")
    return result


def _leaves(value: Any, prefix: str = "") -> dict[str, np.ndarray]:
    if isinstance(value, Mapping):
        result: dict[str, np.ndarray] = {}
        for key in sorted(value, key=str):
            child = f"{prefix}.{key}" if prefix else str(key)
            result.update(_leaves(value[key], child))
        return result
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes, bytearray, np.ndarray))
    ):
        result = {}
        for index, item in enumerate(value):
            child = f"{prefix}[{index}]"
            result.update(_leaves(item, child))
        return result
    return {prefix or "root": _numpy(value)}


def compare_trees(expected: Any, observed: Any) -> dict[str, Any]:
    """Compare every numeric leaf by dtype, shape, and exact bytes."""

    left = _leaves(expected)
    right = _leaves(observed)
    names = sorted(set(left) | set(right))
    rows = []
    maximum_abs = 0.0
    bitwise = True
    for name in names:
        if name not in left or name not in right:
            rows.append({
                "name": name,
                "present_expected": name in left,
                "present_observed": name in right,
                "bitwise_equal": False,
            })
            bitwise = False
            continue
        expected_array = left[name]
        observed_array = right[name]
        same_dtype = expected_array.dtype == observed_array.dtype
        same_shape = expected_array.shape == observed_array.shape
        same_bytes = bool(
            same_dtype
            and same_shape
            and expected_array.tobytes(order="C")
            == observed_array.tobytes(order="C")
        )
        if same_shape:
            difference = (
                expected_array.astype(np.complex128)
                - observed_array.astype(np.complex128)
            )
            leaf_maximum = float(np.max(np.abs(difference), initial=0.0))
            maximum_abs = max(maximum_abs, leaf_maximum)
        else:
            leaf_maximum = float("inf")
        rows.append({
            "name": name,
            "present_expected": True,
            "present_observed": True,
            "dtype_expected": str(expected_array.dtype),
            "dtype_observed": str(observed_array.dtype),
            "shape_expected": list(expected_array.shape),
            "shape_observed": list(observed_array.shape),
            "maximum_abs_difference": leaf_maximum,
            "bitwise_equal": same_bytes,
        })
        bitwise = bitwise and same_bytes
    return {
        "bitwise_equal": bool(bitwise),
        "leaf_count": len(rows),
        "maximum_abs_difference": maximum_abs,
        "leaves": rows,
    }


def centered_row_radius(expected_rows: Any, observed_rows: Any) -> np.ndarray:
    """Return the Frobenius radius of each centered row-map difference."""

    expected = np.asarray(expected_rows, dtype=np.float64)
    observed = np.asarray(observed_rows, dtype=np.float64)
    if (
        expected.shape != observed.shape
        or expected.ndim < 2
        or not np.isfinite(expected).all()
        or not np.isfinite(observed).all()
    ):
        raise ValueError("row-map endpoints must be finite and shape matched")
    difference = observed - expected
    centered = difference - np.mean(difference, axis=-2, keepdims=True)
    return np.sqrt(np.sum(centered * centered, axis=(-2, -1)))


def native_and_conversion_report(
    native_expected: Any,
    native_observed: Any,
    native_rows_expected: Any,
    native_rows_observed: Any,
    *,
    conversion_dtype: str = "float32",
) -> dict[str, Any]:
    """Keep native bit equality and dtype conversion separate."""

    native = compare_trees(native_expected, native_observed)
    native_radius = centered_row_radius(
        native_rows_expected, native_rows_observed
    )
    observed = np.asarray(native_rows_observed)
    converted = observed.astype(conversion_dtype).astype(observed.dtype)
    conversion_radius = centered_row_radius(observed, converted)
    return {
        "native": native,
        "native_centered_row_radius": native_radius,
        "native_radius_zero_from_bitwise_equality": bool(
            native["bitwise_equal"] and np.all(native_radius == 0.0)
        ),
        "conversion": {
            "source_dtype": str(observed.dtype),
            "intermediate_dtype": str(np.dtype(conversion_dtype)),
            "centered_row_radius": conversion_radius,
        },
        "charges_combined": False,
    }


def checkpoint_link(
    source_path: str | Path, successor_path: str | Path
) -> dict[str, str]:
    source = Path(source_path).resolve()
    successor = Path(successor_path).resolve()
    if not source.is_file() or not successor.is_file():
        raise FileNotFoundError("checkpoint link needs two existing files")
    return {
        "source_path": str(source),
        "source_sha256": sha256_path(source),
        "native_successor_path": str(successor),
        "native_successor_sha256": sha256_path(successor),
    }

