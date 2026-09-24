#!/usr/bin/env python3
"""Freeze the registered population and data orders for confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT.parent
    / "experiment-data"
    / "shared"
    / "block-normal-registration-foundation"
)
FOUNDATION_SCHEMA = "pldr-block-normal-registration-foundation-v1"
ORDER_NAMES = (
    "construction-seed7444-chunk-order.npy",
    "heldout-seed8444-chunk-order.npy",
    "heldout-seed9444-chunk-order.npy",
)


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(target)


def _immutable_files(output: Path) -> list[Path]:
    return sorted(
        path for path in output.rglob("*")
        if path.is_file() and path.name != "MANIFEST.sha256"
    )


def _write_manifest(output: Path) -> None:
    lines = [
        f"{sha256_path(path)}  {path.relative_to(output).as_posix()}\n"
        for path in _immutable_files(output)
    ]
    _write_bytes(output / "MANIFEST.sha256", "".join(lines).encode("ascii"))


def validate(output: Path) -> None:
    output = output.resolve()
    manifest = output / "MANIFEST.sha256"
    if not manifest.is_file():
        raise FileNotFoundError("registration-foundation manifest is missing")
    listed = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        path = output / relative
        if not path.is_file() or sha256_path(path) != digest:
            raise ValueError(f"registration-foundation file changed: {relative}")
        listed.add(relative)
    actual = {
        path.relative_to(output).as_posix()
        for path in _immutable_files(output)
    }
    if listed != actual:
        raise ValueError("registration-foundation membership changed")
    foundation = json.loads(
        (output / "FOUNDATION.json").read_text(encoding="utf-8")
    )
    registry_path = output / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        foundation.get("schema_version") != FOUNDATION_SCHEMA
        or foundation.get("registry_sha256") != sha256_path(registry_path)
        or registry.get("map_count") != 288
        or registry.get("within_map_pair_count") != 580_608
        or len(registry.get("construction", [])) != 8
        or len(registry.get("validation", [])) != 16
    ):
        raise ValueError("registration-foundation metadata is invalid")
    expected_orders = []
    for name in ORDER_NAMES:
        path = output / "orders" / name
        if not path.is_file():
            raise FileNotFoundError(f"registration order is missing: {name}")
        expected_orders.append({
            "name": name,
            "sha256": sha256_path(path),
            "bytes": path.stat().st_size,
        })
    if foundation.get("orders") != expected_orders:
        raise ValueError("registration-foundation order metadata changed")
    print(
        "block-normal registration foundation: verified "
        f"({len(actual)} immutable files)"
    )


def build(registry_path: Path, order_dir: Path, output: Path) -> None:
    registry_path = registry_path.resolve()
    order_dir = order_dir.resolve()
    output = output.resolve()
    if not registry_path.is_file() or not order_dir.is_dir():
        raise FileNotFoundError("registry or order directory is missing")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry_payload = canonical(registry)
    _write_bytes(output / "registry.json", registry_payload)
    order_rows = []
    for name in ORDER_NAMES:
        source = order_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"registered order is missing: {name}")
        target = output / "orders" / name
        _copy_file(source, target)
        order_rows.append({
            "name": name,
            "sha256": sha256_path(target),
            "bytes": target.stat().st_size,
        })
    foundation = {
        "schema_version": FOUNDATION_SCHEMA,
        "registry_sha256": sha256_path(output / "registry.json"),
        "dataset_sha256": registry["dataset_sha256"],
        "tokenizer_sha256": registry["tokenizer_sha256"],
        "map_count": registry["map_count"],
        "within_map_pair_count": registry["within_map_pair_count"],
        "orders": order_rows,
    }
    _write_bytes(output / "FOUNDATION.json", canonical(foundation))
    _write_bytes(
        output / "README.md",
        (
            "Digest-bound registered row-map population and three frozen "
            "data orders for the block-normal confirmation campaign.\n"
        ).encode(),
    )
    _write_manifest(output)
    validate(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--orders", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    if arguments.check:
        if arguments.registry is not None or arguments.orders is not None:
            raise SystemExit("--check does not accept source inputs")
        validate(arguments.output)
        return
    if arguments.registry is None or arguments.orders is None:
        raise SystemExit("--registry and --orders are required when freezing")
    build(arguments.registry, arguments.orders, arguments.output)


if __name__ == "__main__":
    main()
