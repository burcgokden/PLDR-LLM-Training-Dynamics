#!/usr/bin/env python3
"""Execute one frozen phase of the direct-work graph on fixed devices.

Archival execution interface: no validated interruption-safe cumulative campaign
budget guarantee. See the combined docs/RESOURCE_EXECUTION.md and
provenance/campaign-accounting.json for the inspected accounting boundary.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.direct_work_specs import (  # noqa: E402
    RESOURCE_BUDGET,
    RESOURCE_SCHEMA,
    validate_design,
)
from confirm.resource_executor import (  # noqa: E402
    ResourceCaps,
    record_is_admissible,
    resource_record_path,
    resource_records,
    run_capped,
    write_json_atomic,
)


PHASES = ("qualification", "construction", "lock", "heldout", "analysis")


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _normalized_design(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    for name in ("source_sha256", "input_sha256", "binding_root"):
        result.pop(name, None)
    return result


def _resource_record(
    campaign_root: Path, node_id: str
) -> Path:
    return resource_record_path(campaign_root, node_id)


def _completed(campaign_root: Path, node: dict[str, Any]) -> bool:
    output = Path(node["output"])
    record_path = _resource_record(campaign_root, node["node_id"])
    if not output.is_file() or not record_path.is_file():
        return False
    record = _load(record_path)
    return bool(
        record.get("schema_version") == RESOURCE_SCHEMA
        and record_is_admissible(record)
        and record.get("node_id") == node["node_id"]
    )


def _rewrite_resource_clock(
    record: dict[str, Any], node: dict[str, Any], record_path: Path
) -> None:
    elapsed = float(record.pop("wall_seconds"))
    record["attempt_elapsed_wall_seconds"] = elapsed
    record["accelerator_reservation_wall_seconds"] = (
        elapsed if str(node["device"]).startswith("cuda:") else 0.0
    )
    record["node_id"] = node["node_id"]
    record["role"] = node["role"]
    write_json_atomic(record_path, record)


def _run_node(
    campaign_root: Path,
    node: dict[str, Any],
    *,
    resume: bool,
) -> None:
    if resume and _completed(campaign_root, node):
        print(f"direct-work executor: retained {node['node_id']}")
        return
    output = Path(node["output"])
    if output.exists():
        raise FileExistsError(
            f"refusing to overwrite an incomplete node output: {output}"
        )
    record_path = _resource_record(campaign_root, node["node_id"])
    if record_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an attempt record: {record_path}"
        )
    caps = ResourceCaps(
        wall_seconds=float(RESOURCE_BUDGET["node_wall_seconds"]),
        host_rss_bytes=48 * 1024**3,
        gpu_reserved_bytes=(
            int(RESOURCE_BUDGET["process_reserved_memory_cap_bytes"])
            if str(node["device"]).startswith("cuda:") else 0
        ),
        output_bytes=int(RESOURCE_BUDGET["persistent_output_cap_bytes"]),
        poll_seconds=0.25,
    )
    record = run_capped(
        node["command"],
        device=node["device"],
        output_root=campaign_root / "runtime" / node["node_id"],
        record_path=record_path,
        caps=caps,
        environment={
            "PYTHONPATH": str(EXPERIMENTS),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": "/dev/null",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
        monitor_root=campaign_root,
        record_schema_version=RESOURCE_SCHEMA,
    )
    _rewrite_resource_clock(record, node, record_path)
    if not record_is_admissible(record) or not output.is_file():
        raise RuntimeError(f"direct-work node failed: {node['node_id']}")
    print(f"direct-work executor: completed {node['node_id']}")


def _run_device_queue(
    campaign_root: Path,
    nodes: list[dict[str, Any]],
    *,
    resume: bool,
) -> None:
    for node in nodes:
        _run_node(campaign_root, node, resume=resume)


def _aggregate_hours(campaign_root: Path) -> float:
    total = 0.0
    for path in resource_records(campaign_root):
        record = _load(path)
        if record.get("schema_version") != RESOURCE_SCHEMA:
            raise ValueError(f"unknown direct-work resource schema: {path}")
        total += float(record["accelerator_reservation_wall_seconds"])
    return total / 3600.0


def execute(
    binding_root: Path,
    campaign_root: Path,
    phase: str,
    *,
    resume: bool,
) -> None:
    binding_root = binding_root.resolve(strict=True)
    campaign_root = campaign_root.resolve(strict=True)
    if not campaign_root.is_relative_to(binding_root):
        raise ValueError("campaign root escapes binding root")
    design = _load(campaign_root / "protocol/design.json")
    validate_design(_normalized_design(design))
    plan = _load(campaign_root / "protocol/launch-plan.json")
    if (
        Path(plan["binding_root"]).resolve() != binding_root
        or Path(plan["campaign_root"]).resolve() != campaign_root
        or plan["node_count"] != len(plan["nodes"])
    ):
        raise ValueError("direct-work launch plan binding or count changed")
    nodes = [node for node in plan["nodes"] if node["phase"] == phase]
    if not nodes:
        raise ValueError(f"direct-work plan has no {phase} nodes")
    prerequisites = {
        "construction": campaign_root / "records/qualification-A-S1024-L0.npz",
        "lock": campaign_root / "records/A-S16384-L2.npz",
        "heldout": campaign_root / "records/construction-lock.json",
        "analysis": campaign_root / "records/C-S16384-L2.npz",
    }
    required = prerequisites.get(phase)
    if required is not None and not required.is_file():
        raise ValueError(f"direct-work phase prerequisite is missing: {required}")
    before = _aggregate_hours(campaign_root)
    if before >= float(RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"]):
        raise RuntimeError("direct-work hard reserved-device-hour stop already reached")
    queues: dict[str, list[dict[str, Any]]] = {}
    for node in nodes:
        queues.setdefault(node["device"], []).append(node)
    if len(queues) == 1:
        _run_device_queue(campaign_root, nodes, resume=resume)
    else:
        if set(queues) - {"cuda:0", "cuda:1"}:
            raise ValueError("parallel scientific queues must use the two fixed GPUs")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    _run_device_queue,
                    campaign_root,
                    queue,
                    resume=resume,
                )
                for _device, queue in sorted(queues.items())
            ]
            for future in futures:
                future.result()
    after = _aggregate_hours(campaign_root)
    if after > float(RESOURCE_BUDGET["hard_aggregate_reserved_device_hours"]):
        raise RuntimeError("direct-work hard reserved-device-hour stop exceeded")
    print(
        "direct-work executor: "
        f"phase={phase} reserved_device_hours={after:.6f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding-root", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
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
