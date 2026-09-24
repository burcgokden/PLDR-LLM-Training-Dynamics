#!/usr/bin/env python3
"""Execute nodes from the frozen observable-balance launch graph.

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
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest_object(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


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
    if candidate.is_absolute() or any(
        part in {"", ".", ".."} for part in candidate.parts
    ):
        raise ValueError(f"unsafe relative path: {value}")
    return Path(*candidate.parts)


def _verify_manifest(bundle: Path) -> None:
    manifest = bundle / "MANIFEST.sha256"
    registered: set[str] = set()
    for line in manifest.read_text(encoding="ascii").splitlines():
        digest, relative = line.split("  ", 1)
        if relative in registered:
            raise ValueError(f"duplicate static manifest member: {relative}")
        registered.add(relative)
        if _sha256(bundle / _safe_relative(relative)) != digest:
            raise ValueError(f"static manifest drifted: {relative}")
    mutable = {"records", "reports", "runtime", "incidents"}
    actual = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file()
        and path.name not in {"MANIFEST.sha256", "RUNTIME.sha256"}
        and path.relative_to(bundle).parts[0] not in mutable
    }
    if actual != registered:
        raise ValueError("static manifest membership drifted")


def _verify_static(bundle: Path, binding: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _verify_manifest(bundle)
    protocol = bundle / "protocol"
    if _sha256(Path(binding["tokens_path"])) != binding["tokens_sha256"]:
        raise ValueError("token archive digest drifted")
    if _sha256(protocol / "measurement-registry.json") != binding[
        "measurement_registry_sha256"
    ]:
        raise ValueError("measurement registry digest drifted")
    if _sha256(protocol / "campaign-registry.json") != binding[
        "campaign_registry_sha256"
    ]:
        raise ValueError("campaign registry digest drifted")
    if _sha256(protocol / "launch-authorization.json") != binding[
        "launch_authorization_sha256"
    ]:
        raise ValueError("launch authorization digest drifted")
    for order in binding["orders"].values():
        if _sha256(bundle / _safe_relative(order["path"])) != order["sha256"]:
            raise ValueError("data-order digest drifted")
    for trajectory in binding["checkpoints"].values():
        for checkpoint in trajectory.values():
            if _sha256(Path(checkpoint["path"])) != checkpoint["sha256"]:
                raise ValueError("checkpoint digest drifted")

    source_binding_path = protocol / "source-binding.json"
    source_binding = _load_json(source_binding_path)
    source_root = bundle / "source"
    actual_sources = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_file()
    }
    if actual_sources != set(source_binding["source_files"]):
        raise ValueError("source snapshot membership drifted")
    for relative, digest in source_binding["source_files"].items():
        if _sha256(source_root / _safe_relative(relative)) != digest:
            raise ValueError(f"source snapshot digest drifted: {relative}")

    registry = _load_json(protocol / "campaign-registry.json")
    unsigned = dict(registry)
    recorded_digest = unsigned.pop("registry_sha256", None)
    if recorded_digest != _digest_object(unsigned):
        raise ValueError("campaign registry self-digest drifted")
    if registry["source_binding_sha256"] != _sha256(source_binding_path):
        raise ValueError("campaign registry source binding drifted")
    plan_path = protocol / "launch-plan.json"
    plan = _load_json(plan_path)
    design = _load_json(protocol / "campaign-design.json")
    design_unsigned = dict(design)
    design_digest = design_unsigned.pop("design_sha256", None)
    if design_digest != _digest_object(design_unsigned):
        raise ValueError("campaign design self-digest drifted")
    if (
        registry["campaign_id"] != binding["campaign_id"]
        or plan["campaign_id"] != binding["campaign_id"]
        or registry["launch_plan_sha256"] != _sha256(plan_path)
        or registry["design_sha256"] != design_digest
        or plan["design_sha256"] != design_digest
    ):
        raise ValueError("campaign identities or design digests disagree")
    return plan, design


def _resolve_command(
    command: list[str], *, bundle: Path, binding: dict[str, Any]
) -> list[str]:
    resolved: list[str] = []
    for raw in command:
        part = str(raw).replace("{python}", sys.executable)
        part = part.replace("{bundle}", str(bundle))
        part = part.replace("{tokens}", str(Path(binding["tokens_path"])))
        if part.startswith("{checkpoint:") and part.endswith("}"):
            label, step = part[12:-1].split(":", 1)
            part = binding["checkpoints"][label][step]["path"]
        resolved.append(part)
    return resolved


def _validate_plan(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if (
        plan.get("scientific_outcomes_control_execution") is not False
        or plan.get("planned_devices_are_immutable") is not True
    ):
        raise ValueError("launch graph execution policy drifted")
    raw_nodes = plan.get("nodes")
    if not isinstance(raw_nodes, list) or len(raw_nodes) != 57:
        raise ValueError("launch graph node count drifted")
    nodes = {str(node["id"]): node for node in raw_nodes}
    if len(nodes) != len(raw_nodes):
        raise ValueError("launch graph contains duplicate node ids")
    outputs: set[str] = set()
    seen: set[str] = set()
    for node in raw_nodes:
        node_id = str(node["id"])
        if any(str(dependency) not in seen for dependency in node["depends_on"]):
            raise ValueError(f"launch graph is not topologically ordered at {node_id}")
        device = str(node["device"])
        if device not in {"cpu", "cuda:0", "cuda:1"}:
            raise ValueError(f"unsupported planned device: {device}")
        command = [str(value) for value in node["command"]]
        if device.startswith("cuda:"):
            if "--device" not in command:
                raise ValueError(f"GPU node lacks a device argument: {node_id}")
            if command[command.index("--device") + 1] != device:
                raise ValueError(f"command device disagrees with plan: {node_id}")
        elif "--device" in command:
            raise ValueError(f"CPU node unexpectedly specifies a GPU: {node_id}")
        for relative in node["expected_outputs"]:
            _safe_relative(str(relative))
            if relative in outputs:
                raise ValueError(f"duplicate planned output: {relative}")
            outputs.add(relative)
        seen.add(node_id)
    return nodes


def _completed(bundle: Path, node: dict[str, Any], command: list[str]) -> bool:
    path = bundle / "runtime" / str(node["id"]) / "completed.json"
    if not path.is_file():
        return False
    record = _load_json(path)
    outputs = [bundle / _safe_relative(str(item)) for item in node["expected_outputs"]]
    return bool(
        all(output.is_file() for output in outputs)
        and record.get("command_sha256") == _digest_object(command)
        and record.get("role") == node["role"]
        and record.get("planned_device") == node["device"]
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


def _attempt_gpu_seconds(bundle: Path) -> float:
    total = 0.0
    for path in (bundle / "runtime").glob("*/attempt-*/attempt.json"):
        record = _load_json(path)
        if str(record.get("planned_device", "")).startswith("cuda:"):
            total += float(record["elapsed_seconds"])
    return total


def _persistent_output_bytes(bundle: Path) -> int:
    return sum(
        path.stat().st_size
        for root in (bundle / "records", bundle / "reports")
        for path in root.rglob("*")
        if path.is_file()
    )


def execute_node(
    bundle: Path,
    node: dict[str, Any],
    nodes: dict[str, dict[str, Any]],
    binding: dict[str, Any],
    numerical_policy: dict[str, Any],
    resource_budget: dict[str, Any],
) -> None:
    if (bundle / "RUNTIME.sha256").exists():
        raise RuntimeError("runtime is sealed; no further execution is permitted")
    command = _resolve_command(node["command"], bundle=bundle, binding=binding)
    if _completed(bundle, node, command):
        print(f"{node['id']}: already completed and digest-verified", flush=True)
        return
    for dependency in node["depends_on"]:
        dependency_command = _resolve_command(
            nodes[dependency]["command"], bundle=bundle, binding=binding
        )
        if not _completed(bundle, nodes[dependency], dependency_command):
            raise RuntimeError(f"{node['id']} has incomplete dependency {dependency}")

    hard_gpu_seconds = 3600.0 * float(
        resource_budget["hard_aggregate_reserved_device_hours"]
    )
    if str(node["device"]).startswith("cuda:") and _attempt_gpu_seconds(bundle) >= hard_gpu_seconds:
        raise RuntimeError("aggregate GPU-hour hard stop reached")
    outputs = [bundle / _safe_relative(str(item)) for item in node["expected_outputs"]]
    if any(path.exists() for path in outputs):
        raise FileExistsError(
            f"{node['id']} has an unsealed output; register it as an incident before rerun"
        )
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    attempt = _attempt_directory(bundle, str(node["id"]))
    log_path = attempt / "stdout.log"
    source = bundle / "source"
    environment = dict(os.environ)
    environment.update(
        {
            "CUBLAS_WORKSPACE_CONFIG": str(
                numerical_policy["cublas_workspace_config"]
            ),
            "PYTHONPATH": os.pathsep.join(
                [str(source / "experiments"), str(source)]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    environment_record = {
        "schema_version": "pldr-observable-balance-environment-v1",
        "node": node["id"],
        "role": node["role"],
        "planned_device": node["device"],
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cublas_workspace_config": environment["CUBLAS_WORKSPACE_CONFIG"],
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": numerical_policy["deterministic_algorithms"],
        "allow_tf32": numerical_policy["allow_tf32"],
    }
    _write_json_atomic(attempt / "environment.json", environment_record)

    started_ns = time.time_ns()
    started = time.monotonic()
    print(f"{node['id']}: starting on {node['device']}", flush=True)
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
        "schema_version": "pldr-observable-balance-attempt-v1",
        "node": node["id"],
        "role": node["role"],
        "stage": node["stage"],
        "planned_device": node["device"],
        "started_at_ns": started_ns,
        "elapsed_seconds": elapsed,
        "exit_code": exit_code,
        "command": command,
        "command_sha256": _digest_object(command),
        "environment_sha256": _sha256(attempt / "environment.json"),
        "stdout_sha256": _sha256(log_path),
    }
    _write_json_atomic(attempt / "attempt.json", attempt_record)
    if exit_code:
        raise RuntimeError(f"{node['id']} failed with exit code {exit_code}")
    if any(not path.is_file() for path in outputs):
        raise RuntimeError(f"{node['id']} omitted a declared output")

    resource = _npz_resource(outputs)
    memory_cap = int(resource_budget["process_reserved_memory_cap_bytes"])
    if int(resource["peak_gpu_reserved_bytes"]) > memory_cap:
        raise RuntimeError(f"{node['id']} exceeded the process memory cap")
    if _attempt_gpu_seconds(bundle) > hard_gpu_seconds:
        raise RuntimeError(f"{node['id']} exceeded the aggregate GPU-hour hard stop")
    if _persistent_output_bytes(bundle) > int(
        resource_budget["persistent_output_cap_bytes"]
    ):
        raise RuntimeError(f"{node['id']} exceeded the persistent-output cap")

    completed = {
        "schema_version": "pldr-observable-balance-completion-v1",
        "node": node["id"],
        "role": node["role"],
        "stage": node["stage"],
        "planned_device": node["device"],
        "command_sha256": attempt_record["command_sha256"],
        "attempt": attempt.relative_to(bundle).as_posix(),
        "attempt_sha256": _sha256(attempt / "attempt.json"),
        "outputs": {
            path.relative_to(bundle).as_posix(): _sha256(path) for path in outputs
        },
        "resource": resource,
    }
    _write_json_atomic(bundle / "runtime" / str(node["id"]) / "completed.json", completed)
    print(f"{node['id']}: completed in {elapsed:.2f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--node", action="append", default=[])
    parser.add_argument("--stage")
    parser.add_argument("--all", action="store_true")
    arguments = parser.parse_args()
    bundle = arguments.bundle.resolve()
    binding = _load_json(bundle / "protocol" / "input-binding.json")
    plan, design = _verify_static(bundle, binding)
    nodes = _validate_plan(plan)

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
    for node_id in selected:
        if node_id not in nodes:
            raise KeyError(f"unknown node: {node_id}")
        execute_node(
            bundle,
            nodes[node_id],
            nodes,
            binding,
            design["numerical_policy"],
            design["resource_budget"],
        )


if __name__ == "__main__":
    main()
