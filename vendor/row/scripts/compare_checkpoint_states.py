#!/usr/bin/env python3
"""Compare trusted PLDR training checkpoints recursively and emit strict JSON.

This utility is intended for provenance checks between locally produced,
trusted checkpoints.  It deliberately uses exact equality: a prefix replay
either reproduces the selected state or it does not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SCOPES = ("model", "opt", "scheduler", "rng_states")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compare(left: Any, right: Any, path: str, mismatches: list[str]) -> None:
    if type(left) is not type(right):
        mismatches.append(
            f"{path}: type {type(left).__name__} != {type(right).__name__}"
        )
        return
    if isinstance(left, torch.Tensor):
        if left.dtype != right.dtype or left.shape != right.shape:
            mismatches.append(
                f"{path}: tensor {tuple(left.shape)}/{left.dtype} != "
                f"{tuple(right.shape)}/{right.dtype}"
            )
        elif not torch.equal(left, right):
            mismatches.append(f"{path}: tensor values differ")
        return
    if isinstance(left, np.ndarray):
        if left.dtype != right.dtype or left.shape != right.shape:
            mismatches.append(
                f"{path}: array {left.shape}/{left.dtype} != "
                f"{right.shape}/{right.dtype}"
            )
        elif not np.array_equal(left, right, equal_nan=True):
            mismatches.append(f"{path}: array values differ")
        return
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            mismatches.append(
                f"{path}: keys differ; left_only={sorted(left_keys - right_keys)!r}; "
                f"right_only={sorted(right_keys - left_keys)!r}"
            )
        for key in sorted(left_keys & right_keys, key=str):
            _compare(left[key], right[key], f"{path}.{key}", mismatches)
        return
    if isinstance(left, (list, tuple)):
        if len(left) != len(right):
            mismatches.append(f"{path}: length {len(left)} != {len(right)}")
            return
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            _compare(left_item, right_item, f"{path}[{index}]", mismatches)
        return
    if left != right:
        mismatches.append(f"{path}: values differ")


def compare_checkpoints(left_path: Path, right_path: Path, scopes: tuple[str, ...]) -> dict[str, Any]:
    left = torch.load(left_path, map_location="cpu", weights_only=False)
    right = torch.load(right_path, map_location="cpu", weights_only=False)
    if not isinstance(left, dict) or not isinstance(right, dict):
        raise ValueError("both checkpoints must contain dictionaries")
    missing = [scope for scope in scopes if scope not in left or scope not in right]
    if missing:
        raise ValueError(f"missing requested scopes: {missing}")

    mismatches: list[str] = []
    for scope in scopes:
        _compare(left[scope], right[scope], scope, mismatches)
    return {
        "schema_version": "pldr-checkpoint-exact-comparison-v1",
        "left": {"path": str(left_path.resolve()), "sha256": _sha256(left_path)},
        "right": {"path": str(right_path.resolve()), "sha256": _sha256(right_path)},
        "left_step": left.get("step"),
        "right_step": right.get("step"),
        "scopes": list(scopes),
        "exact_equal": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--scope", action="append", dest="scopes")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    scopes = tuple(args.scopes) if args.scopes else DEFAULT_SCOPES
    result = compare_checkpoints(args.left, args.right, scopes)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if not result["exact_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
