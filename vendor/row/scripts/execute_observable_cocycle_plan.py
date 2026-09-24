#!/usr/bin/env python3
"""Execute nodes from the frozen observable full-state cocycle launch graph.

Archival execution interface: no validated interruption-safe cumulative campaign
budget guarantee. See the combined docs/RESOURCE_EXECUTION.md and
provenance/campaign-accounting.json for the inspected accounting boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(_canonical(value) + b"\n")
    temporary.replace(path)


def _safe_relative(value: str) -> Path:
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError(f"unsafe relative path: {value}")
    return Path(*candidate.parts)


def _verify_static(bundle: Path, binding: dict[str, Any]) -> None:
    if _sha256(Path(binding["tokens_path"])) != binding["tokens_sha256"]:
        raise ValueError("token archive digest drifted")
    if _sha256(bundle / "protocol" / "registry.json") != binding["registry_sha256"]:
        raise ValueError("registry digest drifted")
    for order in binding["orders"].values():
        if _sha256(bundle / _safe_relative(order["path"])) != order["sha256"]:
            raise ValueError("data order digest drifted")
    for trajectory in binding["checkpoints"].values():
        for checkpoint in trajectory.values():
            if _sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
                raise ValueError("checkpoint digest drifted")
    source_binding = _load_json(bundle / "protocol" / "source_binding.json")
    source_root = bundle / "source"
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual != set(source_binding["source_files"]):
        raise ValueError("source snapshot membership drifted")
    for relative, digest in source_binding["source_files"].items():
        if _sha256(source_root / _safe_relative(relative)) != digest:
            raise ValueError(f"source snapshot digest drifted: {relative}")
    for line in (bundle / "MANIFEST.sha256").read_text().splitlines():
        digest, relative = line.split("  ", 1)
        if _sha256(bundle / _safe_relative(relative)) != digest:
            raise ValueError(f"static manifest drifted: {relative}")


def _resolve_command(
    command: list[str],
    *,
    bundle: Path,
    binding: dict[str, Any],
    device_override: str | None,
) -> list[str]:
    resolved = []
    for raw in command:
        part = str(raw).replace("{python}", sys.executable)
        part = part.replace("{bundle}", str(bundle))
        part = part.replace("{tokens}", str(Path(binding["tokens_path"])))
        if part.startswith("{checkpoint:") and part.endswith("}"):
            label, step = part[12:-1].split(":", 1)
            part = binding["checkpoints"][label][step]["path"]
        resolved.append(part)
    if device_override and "--device" in resolved:
        resolved[resolved.index("--device") + 1] = device_override
    return resolved


def _completed(bundle: Path, node: dict[str, Any], command: list[str]) -> bool:
    path = bundle / "runtime" / node["id"] / "completed.json"
    if not path.is_file():
        return False
    record = _load_json(path)
    outputs = [bundle / _safe_relative(item) for item in node["expected_outputs"]]
    return bool(
        all(output.is_file() for output in outputs)
        and record.get("command_sha256")
        == hashlib.sha256(_canonical(command)).hexdigest()
        and record.get("outputs")
        == {
            output.relative_to(bundle).as_posix(): _sha256(output)
            for output in outputs
        }
    )


def _attempt_directory(bundle: Path, node_id: str) -> Path:
    root = bundle / "runtime" / node_id
    root.mkdir(parents=True, exist_ok=True)
    index = 1
    while (root / f"attempt-{index}").exists():
        index += 1
    result = root / f"attempt-{index}"
    result.mkdir()
    return result


def _npz_resource(outputs: list[Path]) -> dict[str, float | int]:
    candidate = next((path for path in reversed(outputs) if path.suffix == ".npz"), None)
    if candidate is None:
        return {
            "reserved_device_seconds": 0.0,
            "peak_gpu_reserved_bytes": 0,
            "peak_gpu_allocated_bytes": 0,
            "peak_host_bytes": 0,
        }
    with np.load(candidate, allow_pickle=False) as record:
        return {
            "reserved_device_seconds": float(record["resource_reserved_device_seconds"]),
            "peak_gpu_reserved_bytes": int(record["resource_peak_gpu_reserved_bytes"]),
            "peak_gpu_allocated_bytes": int(record["resource_peak_gpu_allocated_bytes"]),
            "peak_host_bytes": int(record["resource_peak_host_bytes"]),
        }


def _used_gpu_seconds(bundle: Path) -> float:
    total = 0.0
    for path in (bundle / "runtime").glob("*/completed.json"):
        total += float(_load_json(path).get("resource", {}).get("reserved_device_seconds", 0.0))
    return total


def execute_node(
    bundle: Path,
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    binding: dict[str, Any],
    *,
    hard_gpu_seconds: float,
    device_override: str | None,
) -> None:
    command = _resolve_command(
        node["command"],
        bundle=bundle,
        binding=binding,
        device_override=device_override,
    )
    if _completed(bundle, node, command):
        print(f"{node['id']}: already completed and digest-verified", flush=True)
        return
    for dependency in node["depends_on"]:
        dependency_command = _resolve_command(
            nodes[dependency]["command"],
            bundle=bundle,
            binding=binding,
            device_override=None,
        )
        if not _completed(bundle, nodes[dependency], dependency_command):
            raise RuntimeError(f"{node['id']} has incomplete dependency {dependency}")
    if _used_gpu_seconds(bundle) >= hard_gpu_seconds:
        raise RuntimeError("aggregate GPU-hour hard stop reached")
    outputs = [bundle / _safe_relative(item) for item in node["expected_outputs"]]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            f"{node['id']} has an unsealed output; preserve it as an incident before rerun"
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    attempt = _attempt_directory(bundle, node["id"])
    log_path = attempt / "stdout.log"
    started_ns = time.time_ns()
    started = time.monotonic()
    environment = dict(os.environ)
    source = bundle / "source"
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [str(source / "experiments"), str(source)]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    print(f"{node['id']}: starting", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=bundle,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        exit_code = process.wait()
    elapsed = time.monotonic() - started
    attempt_record = {
        "schema_version": "pldr-observable-cocycle-attempt-v1",
        "node": node["id"],
        "started_at_ns": started_ns,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "command": command,
        "command_sha256": hashlib.sha256(_canonical(command)).hexdigest(),
        "stdout_sha256": _sha256(log_path),
    }
    _write_json_atomic(attempt / "attempt.json", attempt_record)
    if exit_code:
        raise RuntimeError(f"{node['id']} failed with exit code {exit_code}")
    if any(not path.is_file() for path in outputs):
        raise RuntimeError(f"{node['id']} omitted a declared output")
    resource = _npz_resource(outputs)
    if int(resource["peak_gpu_reserved_bytes"]) > 20 * 1024**3:
        raise RuntimeError(f"{node['id']} exceeded the process memory cap")
    if _used_gpu_seconds(bundle) + float(resource["reserved_device_seconds"]) > hard_gpu_seconds:
        raise RuntimeError(f"{node['id']} exceeded the aggregate GPU-hour hard stop")
    completed = {
        "schema_version": "pldr-observable-cocycle-completion-v1",
        "node": node["id"],
        "command_sha256": attempt_record["command_sha256"],
        "outputs": {
            path.relative_to(bundle).as_posix(): _sha256(path) for path in outputs
        },
        "resource": resource,
    }
    _write_json_atomic(bundle / "runtime" / node["id"] / "completed.json", completed)
    print(f"{node['id']}: completed in {elapsed:.2f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--stage")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device")
    arguments = parser.parse_args()
    bundle = arguments.bundle.resolve()
    binding = _load_json(bundle / "protocol" / "input_binding.json")
    _verify_static(bundle, binding)
    plan = _load_json(bundle / "protocol" / "launch_plan.json")
    if plan.get("scientific_outcomes_control_execution") is not False:
        raise ValueError("launch graph permits outcome-dependent execution")
    nodes = {node["id"]: node for node in plan["nodes"]}
    if len(nodes) != len(plan["nodes"]):
        raise ValueError("launch graph contains duplicate node ids")
    selected = list(arguments.node)
    if arguments.stage:
        selected.extend(
            node["id"] for node in plan["nodes"] if node["stage"] == arguments.stage
        )
    if arguments.all:
        selected.extend(node["id"] for node in plan["nodes"])
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("select --node, --stage, or --all")
    hard_hours = float(
        json.loads((bundle / "protocol" / "campaign-design.json").read_text())[
            "resource_budget"
        ]["hard_aggregate_reserved_device_hours"]
    )
    for node_id in selected:
        if node_id not in nodes:
            raise KeyError(f"unknown node: {node_id}")
        execute_node(
            bundle,
            nodes[node_id],
            nodes,
            binding,
            hard_gpu_seconds=3600.0 * hard_hours,
            device_override=arguments.device,
        )


if __name__ == "__main__":
    main()
