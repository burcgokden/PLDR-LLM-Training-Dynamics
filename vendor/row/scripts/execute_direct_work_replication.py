#!/usr/bin/env python3
"""Execute the frozen temporal-context direct-work replication.

Archival execution interface: no validated interruption-safe cumulative campaign
budget guarantee. See the combined docs/RESOURCE_EXECUTION.md and
provenance/campaign-accounting.json for the inspected accounting boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


PHASES = ("qualification", "construction", "lock", "heldout", "analysis")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_design(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for name in (
        "binding_root",
        "campaign_root",
        "source_sha256",
        "input_sha256",
        "bound_inputs",
        "registry_sha256",
        "terminal_update_batch_sha256",
        "source_file_count",
    ):
        result.pop(name, None)
    return result


def frozen_modules(campaign_root: Path) -> dict[str, Any]:
    experiments = campaign_root / "source/experiments"
    sys.path.insert(0, str(experiments))
    from confirm.direct_work_replication_specs import (  # noqa: PLC0415
        CAMPAIGN_ID,
        RESOURCE_BUDGET,
        RESOURCE_SCHEMA,
        validate_design,
    )
    from confirm.resource_executor import (  # noqa: PLC0415
        ResourceCaps,
        run_capped,
        write_json_atomic,
    )

    return {
        "campaign_id": CAMPAIGN_ID,
        "budget": RESOURCE_BUDGET,
        "resource_schema": RESOURCE_SCHEMA,
        "validate_design": validate_design,
        "ResourceCaps": ResourceCaps,
        "run_capped": run_capped,
        "write_json_atomic": write_json_atomic,
        "experiments": experiments,
    }


def verify_frozen(
    binding_root: Path, campaign_root: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not campaign_root.is_relative_to(binding_root):
        raise ValueError("replication root escapes binding root")
    design = load_json(campaign_root / "protocol/design.json")
    plan = load_json(campaign_root / "protocol/launch-plan.json")
    modules = frozen_modules(campaign_root)
    modules["validate_design"](normalized_design(design))
    if (
        design.get("campaign_id") != modules["campaign_id"]
        or plan.get("campaign_id") != modules["campaign_id"]
        or Path(design.get("binding_root", "")).resolve() != binding_root
        or Path(design.get("campaign_root", "")).resolve() != campaign_root
        or Path(plan.get("binding_root", "")).resolve() != binding_root
        or Path(plan.get("campaign_root", "")).resolve() != campaign_root
        or plan.get("node_count") != len(plan.get("nodes", []))
        or plan.get("node_count") != 31
    ):
        raise ValueError("replication design or launch identity changed")

    source_digests = design.get("source_sha256")
    if not isinstance(source_digests, dict) or not source_digests:
        raise ValueError("replication source digest map is absent")
    for relative, expected in sorted(source_digests.items()):
        path = campaign_root / "source" / relative
        if path.is_symlink() or not path.is_file() or sha256_path(path) != expected:
            raise ValueError(f"frozen replication source changed: {relative}")

    input_digests = design.get("input_sha256")
    bound_inputs = design.get("bound_inputs")
    if (
        not isinstance(input_digests, dict)
        or not isinstance(bound_inputs, dict)
        or set(input_digests) != set(bound_inputs)
    ):
        raise ValueError("replication input bindings are incomplete")
    for name, expected in sorted(input_digests.items()):
        path = Path(bound_inputs[name]).resolve(strict=True)
        if not path.is_relative_to(binding_root) or path.is_symlink():
            raise ValueError(f"replication input binding is invalid: {name}")
        if sha256_path(path) != expected:
            raise ValueError(f"replication input changed: {name}")

    identifiers: set[str] = set()
    for node in plan["nodes"]:
        identifier = str(node["node_id"])
        if identifier in identifiers:
            raise ValueError(f"duplicate replication node: {identifier}")
        identifiers.add(identifier)
        if node.get("phase") not in PHASES:
            raise ValueError(f"unknown replication phase: {identifier}")
        for raw in node.get("expected_outputs", []):
            output = Path(raw).resolve()
            if not output.is_relative_to(campaign_root):
                raise ValueError(f"replication output escapes bundle: {identifier}")
        command = [str(value) for value in node.get("command", [])]
        if not command or not Path(command[1]).resolve().is_relative_to(
            campaign_root / "source"
        ):
            raise ValueError(f"replication command is not frozen: {identifier}")
    return design, plan, modules


def resource_path(campaign_root: Path, node: dict[str, Any]) -> Path:
    from confirm.resource_executor import resource_record_path
    return resource_record_path(campaign_root, str(node["node_id"]))


def completed(
    campaign_root: Path, node: dict[str, Any], resource_schema: str
) -> bool:
    from confirm.resource_executor import record_is_admissible
    record_path = resource_path(campaign_root, node)
    if not record_path.is_file() or any(
        not Path(path).is_file() for path in node["expected_outputs"]
    ):
        return False
    record = load_json(record_path)
    return bool(
        record.get("schema_version") == resource_schema
        and record.get("node_id") == node["node_id"]
        and record_is_admissible(record)
        and (
            "check_command" not in node
            or record.get("analysis_check_status") == "passed"
        )
    )


def producer_peaks(output: Path) -> tuple[int, int]:
    with np.load(output, allow_pickle=False) as record:
        return (
            int(np.asarray(record["peak_gpu_allocated_bytes"]).item()),
            int(np.asarray(record["peak_gpu_reserved_bytes"]).item()),
        )


def rewrite_resource(
    record: dict[str, Any],
    node: dict[str, Any],
    record_path: Path,
    modules: dict[str, Any],
) -> dict[str, Any]:
    elapsed = float(record.pop("wall_seconds"))
    device = str(node["device"])
    record["attempt_elapsed_wall_seconds"] = elapsed
    record["accelerator_reservation_wall_seconds"] = (
        elapsed if device.startswith("cuda:") else 0.0
    )
    record["node_id"] = node["node_id"]
    record["role"] = node["role"]
    record["phase"] = node["phase"]
    if device.startswith("cuda:"):
        monitor_allocated = int(record.pop("peak_gpu_allocated_bytes"))
        monitor_reserved = int(record.pop("peak_gpu_reserved_bytes"))
        record["monitor_peak_gpu_allocated_bytes"] = monitor_allocated
        record["monitor_peak_gpu_reserved_bytes"] = monitor_reserved
        output = Path(node["expected_outputs"][0])
        if output.is_file():
            producer_allocated, producer_reserved = producer_peaks(output)
        else:
            producer_allocated, producer_reserved = (0, 0)
        record["producer_peak_gpu_allocated_bytes"] = producer_allocated
        record["producer_peak_gpu_reserved_bytes"] = producer_reserved
        observation = record.get("gpu_probe_observation")
        if monitor_reserved > 0 and producer_reserved > 0 and observation:
            status = "measured"
            authority = "process-query-and-producer"
            incident = None
        elif monitor_reserved == 0 and producer_reserved > 0:
            status = "unavailable"
            authority = "producer"
            query_status = (
                observation.get("process_query_status")
                if isinstance(observation, dict) else "probe-observation-absent"
            )
            incident = {
                "classification": f"{query_status}-with-nonzero-producer",
                "disposition": "producer-counter-authoritative",
            }
        else:
            status = "inconsistent"
            authority = "none"
            incident = {
                "classification": "contradictory-gpu-memory-observations",
                "disposition": "technical-failure",
            }
        record["gpu_monitor_status"] = status
        record["gpu_memory_authority"] = authority
        record["gpu_monitor_incident"] = incident
        cap = int(modules["budget"]["process_reserved_memory_cap_bytes"])
        if producer_reserved >= cap:
            record["cap_status"] = "reached_producer_gpu_reserved_bytes"
            record["technical_valid"] = False
        if status == "inconsistent":
            record["technical_valid"] = False
    else:
        record["gpu_monitor_status"] = "not_applicable"
        record["gpu_memory_authority"] = "not_applicable"
        record["gpu_monitor_incident"] = None
    modules["write_json_atomic"](record_path, record)
    return record


def environment(modules: dict[str, Any]) -> dict[str, str]:
    return {
        "PYTHONPATH": str(modules["experiments"]),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/pldr-direct-work-replication-pycache",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    }


def run_check(
    node: dict[str, Any],
    record_path: Path,
    modules: dict[str, Any],
) -> dict[str, Any]:
    record = load_json(record_path)
    if "check_command" not in node:
        return record
    result = subprocess.run(
        [str(value) for value in node["check_command"]],
        cwd=str(record_path.parent),
        env={**__import__("os").environ, **environment(modules)},
        capture_output=True,
        text=True,
        check=False,
    )
    record["analysis_check_command"] = node["check_command"]
    record["analysis_check_exit_code"] = int(result.returncode)
    record["analysis_check_stdout"] = result.stdout
    record["analysis_check_stderr"] = result.stderr
    record["analysis_check_status"] = (
        "passed" if result.returncode == 0 else "failed"
    )
    if result.returncode != 0:
        record["technical_valid"] = False
    modules["write_json_atomic"](record_path, record)
    return record


def run_node(
    campaign_root: Path,
    node: dict[str, Any],
    modules: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if resume and completed(
        campaign_root, node, modules["resource_schema"]
    ):
        print(f"replication executor: retained {node['node_id']}")
        return
    outputs = [Path(path) for path in node["expected_outputs"]]
    from confirm.resource_executor import record_is_admissible
    record_path = resource_path(campaign_root, node)
    if any(path.exists() for path in outputs) or record_path.exists():
        raise FileExistsError(
            f"refusing to overwrite incomplete node: {node['node_id']}"
        )
    budget = modules["budget"]
    caps = modules["ResourceCaps"](
        wall_seconds=float(budget["node_wall_seconds"]),
        host_rss_bytes=48 * 1024**3,
        gpu_reserved_bytes=(
            int(budget["process_reserved_memory_cap_bytes"])
            if str(node["device"]).startswith("cuda:") else 0
        ),
        output_bytes=int(budget["persistent_output_cap_bytes"]),
        poll_seconds=0.25,
    )
    record = modules["run_capped"](
        [str(value) for value in node["command"]],
        device=str(node["device"]),
        output_root=campaign_root / "runtime" / str(node["node_id"]),
        record_path=record_path,
        caps=caps,
        environment=environment(modules),
        monitor_root=campaign_root,
        record_schema_version=modules["resource_schema"],
    )
    record = rewrite_resource(record, node, record_path, modules)
    record = run_check(node, record_path, modules)
    if (
        not record_is_admissible(record)
        or any(not path.is_file() for path in outputs)
    ):
        raise RuntimeError(f"replication node failed: {node['node_id']}")
    print(f"replication executor: completed {node['node_id']}")


def run_queue(
    campaign_root: Path,
    nodes: list[dict[str, Any]],
    modules: dict[str, Any],
    *,
    resume: bool,
) -> None:
    for node in nodes:
        run_node(campaign_root, node, modules, resume=resume)


def aggregate_hours(campaign_root: Path, resource_schema: str) -> float:
    seconds = 0.0
    from confirm.resource_executor import resource_records
    for path in resource_records(campaign_root):
        record = load_json(path)
        if record.get("schema_version") != resource_schema:
            raise ValueError(f"unknown replication resource record: {path}")
        seconds += float(record["accelerator_reservation_wall_seconds"])
    return seconds / 3600.0


def require_dependencies(
    campaign_root: Path,
    plan: dict[str, Any],
    phase: str,
    resource_schema: str,
) -> None:
    required = {
        "construction": {"qualification"},
        "lock": {"qualification", "construction"},
        "heldout": {"qualification", "construction", "lock"},
        "analysis": {"qualification", "construction", "lock", "heldout"},
    }.get(phase, set())
    missing = [
        node["node_id"]
        for node in plan["nodes"]
        if node["phase"] in required
        and not completed(campaign_root, node, resource_schema)
    ]
    if missing:
        raise ValueError(
            "replication prerequisites are incomplete: " + ", ".join(missing)
        )


def qualification_check(campaign_root: Path) -> None:
    from analysis.analyze_direct_work_replication import (  # noqa: PLC0415
        qualification,
    )

    result = qualification(campaign_root)
    if not all(math.isfinite(float(result[name])) for name in (
        "effect_floor",
        "within_device_duplicate_energy_max_abs",
        "cross_device_energy_max_abs",
        "cross_device_observer_max_abs",
    )):
        raise ArithmeticError("replication qualification is nonfinite")
    print(json.dumps({"qualification": result}, sort_keys=True))


def run_phase(
    campaign_root: Path,
    plan: dict[str, Any],
    modules: dict[str, Any],
    phase: str,
    *,
    resume: bool,
) -> None:
    require_dependencies(
        campaign_root, plan, phase, modules["resource_schema"]
    )
    if phase == "construction":
        qualification_check(campaign_root)
    before = aggregate_hours(campaign_root, modules["resource_schema"])
    hard = float(modules["budget"]["hard_aggregate_reserved_device_hours"])
    if before >= hard:
        raise RuntimeError("replication hard device-hour stop already reached")
    nodes = [node for node in plan["nodes"] if node["phase"] == phase]
    queues: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        queues.setdefault(str(node["device"]), []).append(node)
    if len(queues) == 1:
        run_queue(campaign_root, nodes, modules, resume=resume)
    else:
        if set(queues) != {"cuda:0", "cuda:1"}:
            raise ValueError("parallel replication queues must use both GPUs")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    run_queue,
                    campaign_root,
                    queue,
                    modules,
                    resume=resume,
                )
                for _device, queue in sorted(queues.items())
            ]
            for future in futures:
                future.result()
    if phase == "qualification":
        qualification_check(campaign_root)
    after = aggregate_hours(campaign_root, modules["resource_schema"])
    if after > hard:
        raise RuntimeError("replication hard device-hour stop exceeded")
    working = float(
        modules["budget"]["working_aggregate_reserved_device_hours"]
    )
    print(
        "replication executor: "
        f"phase={phase} reserved_device_hours={after:.6f} "
        f"working_budget_exceeded={after > working}"
    )


def execute(
    binding_root: Path,
    campaign_root: Path,
    phase: str,
    *,
    resume: bool,
) -> None:
    binding_root = binding_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=True)
    _design, plan, modules = verify_frozen(binding_root, campaign_root)
    selected = PHASES if phase == "all" else (phase,)
    for current in selected:
        run_phase(campaign_root, plan, modules, current, resume=resume)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--phase", choices=(*PHASES, "all"), required=True)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()
    execute(
        arguments.binding_root,
        arguments.campaign_root,
        arguments.phase,
        resume=arguments.resume,
    )


if __name__ == "__main__":
    main()
