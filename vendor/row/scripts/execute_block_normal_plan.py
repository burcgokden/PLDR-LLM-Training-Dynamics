#!/usr/bin/env python3
"""Execute a concrete block-normal launch graph with a deadline and observed resource admission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from confirm.block_normal_specs import CAMPAIGN_ID  # noqa: E402
from confirm.native_determinism import sha256_path  # noqa: E402
from confirm.resource_executor import (  # noqa: E402
    ResourceCaps,
    record_is_admissible,
    resource_record_path,
    resource_records,
    canonical_digest,
    directory_bytes,
    run_capped,
    write_json_atomic,
)


def _load_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != "pldr-block-normal-launch-plan-v1"
        or plan.get("campaign_id") != CAMPAIGN_ID
        or not isinstance(plan.get("nodes"), list)
    ):
        raise ValueError("unknown block-normal launch plan")
    bundle = Path(plan["bundle_root"]).resolve()
    if bundle != path.resolve().parents[1]:
        raise ValueError("launch plan is not located in its bound bundle")
    for name in ("dataset", "tokenizer", "registry"):
        item = plan[name]
        artifact = Path(item["path"]).resolve()
        if not artifact.is_file() or sha256_path(artifact) != item["sha256"]:
            raise ValueError(f"launch input changed: {name}")
    for order in plan["orders"]:
        artifact = Path(order["path"]).resolve()
        if not artifact.is_file() or sha256_path(artifact) != order["sha256"]:
            raise ValueError("a frozen data order changed")
    ids = [node["id"] for node in plan["nodes"]]
    if len(ids) != len(set(ids)):
        raise ValueError("launch plan contains duplicate node identifiers")
    id_set = set(ids)
    for node in plan["nodes"]:
        if (
            not set(node["depends_on"]).issubset(id_set)
            or not isinstance(node["command"], list)
            or not node["command"]
            or any(not isinstance(value, str) or not value
                   for value in node["command"])
        ):
            raise ValueError("launch node dependency or command is invalid")
        ResourceCaps(**node["caps"]).validate()
    return plan


def _input_ready(path: str) -> bool:
    value = Path(path)
    if value.is_file():
        return True
    if value.is_dir():
        return any(item.is_file() for item in value.rglob("*"))
    return False


def _completed_node_valid(bundle: Path, node: dict[str, Any]) -> bool:
    """Per-process compatibility check only; campaign admission uses its journal."""
    record_path = resource_record_path(bundle, node["id"])
    if not record_path.is_file():
        return False
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        record.get("schema_version")
        == "pldr-block-normal-resource-record-v1"
        and record_is_admissible(record)
        and record.get("command_sha256")
        == canonical_digest(node["command"])
        and all(Path(path).is_file() for path in node["expected_outputs"])
    )


def _gpu_count(device: str) -> int:
    indices = set(re.findall(r"cuda:(\d+)", device))
    return len(indices)


def execute(plan, *, stages, node_ids, dry_run, continue_on_failure):
    """Apply persistent full-plan admission; stage selection never resets cost."""
    from confirm.campaign_plan_executor import execute_plan
    return execute_plan(plan, stages=stages, node_ids=node_ids, dry_run=dry_run,
                        continue_on_failure=continue_on_failure,
                        resource_schema="pldr-block-normal-resource-record-v1", execution_schema="pldr-block-normal-execution-report-v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stages", nargs="*")
    parser.add_argument("--nodes", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    plan = _load_plan(arguments.plan)
    report = execute(
        plan,
        stages=None if arguments.stages is None else set(arguments.stages),
        node_ids=None if arguments.nodes is None else set(arguments.nodes),
        dry_run=arguments.dry_run,
        continue_on_failure=arguments.continue_on_failure,
    )
    output = arguments.output
    if output is None:
        output = (
            Path(plan["bundle_root"]) / "reports" / "execution-report.json"
        )
    write_json_atomic(output, report)
    print(json.dumps({
        "dry_run": report["dry_run"],
        "campaign_decision": report["campaign_decision"],
        "node_statuses": {
            row["id"]: row["status"] for row in report["nodes"]
        },
        "output": str(output.resolve()),
    }, sort_keys=True))
    if report["campaign_decision"] != "admissible" or any(row["status"] not in {"passed", "already_completed", "ready"} for row in report["nodes"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
