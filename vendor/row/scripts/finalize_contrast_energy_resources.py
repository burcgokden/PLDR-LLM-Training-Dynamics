#!/usr/bin/env python3
"""Seal measured Stage B resource caps before wider campaign execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


CAMPAIGN_ID = "pldr-contrast-energy-cocycle-confirmation-v1"
LOG_SCHEMA = "pldr-contrast-energy-node-log-v1"
LOCK_SCHEMA = "pldr-contrast-energy-resource-lock-v1"


def digest_object(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def build_lock(
    log_dir: str | Path,
    node_ids: list[str],
    *,
    provisional_peak_mib: int,
    enlargement: float,
) -> tuple[dict[str, Any], bool, list[str]]:
    directory = Path(log_dir).resolve()
    if len(node_ids) != len(set(node_ids)) or not node_ids:
        raise ValueError("Stage B resource node ids must be unique and nonempty")
    if (
        isinstance(provisional_peak_mib, bool)
        or int(provisional_peak_mib) < 1
        or not math.isfinite(float(enlargement))
        or float(enlargement) < 1.0
    ):
        raise ValueError("resource cap and enlargement must be positive")
    failures: list[str] = []
    peaks: dict[str, int] = {}
    records = []
    for node_id in node_ids:
        path = directory / f"{node_id}.json"
        if not path.is_file():
            failures.append(f"missing node log {node_id}")
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        mapping = value if isinstance(value, dict) else {}
        resources = mapping.get("resources", {})
        resources = resources if isinstance(resources, dict) else {}
        device = mapping.get("device")
        peak = resources.get("peak_gpu_memory_mib")
        wall = resources.get("wall_seconds")
        host = resources.get("peak_host_rss_bytes")
        unsigned = dict(mapping)
        recorded_digest = unsigned.pop("record_sha256", None)
        if (
            mapping.get("schema_version") != LOG_SCHEMA
            or mapping.get("node_id") != node_id
            or mapping.get("stage") != "B"
            or mapping.get("exit_code") != 0
            or recorded_digest != digest_object(unsigned)
            or not isinstance(device, str)
            or not device.startswith("cuda:")
            or isinstance(peak, bool)
            or not isinstance(peak, int)
            or peak <= 0
            or isinstance(wall, bool)
            or not isinstance(wall, (int, float))
            or not math.isfinite(float(wall))
            or float(wall) < 0.0
            or isinstance(host, bool)
            or not isinstance(host, int)
            or host < 0
        ):
            failures.append(f"invalid Stage B resource log {node_id}")
            continue
        peaks[device] = max(peaks.get(device, 0), peak)
        records.append({
            "node_id": node_id,
            "log_path": str(path),
            "device": device,
            "peak_gpu_memory_mib": peak,
            "wall_seconds": float(wall),
            "peak_host_rss_bytes": int(host),
            "record_sha256": recorded_digest,
        })
    if set(peaks) != {"cuda:0", "cuda:1"}:
        failures.append("Stage B did not measure both registered devices")
    caps = {
        device: int(math.ceil(peak * float(enlargement)))
        for device, peak in sorted(peaks.items())
    }
    for device, cap in caps.items():
        if cap > int(provisional_peak_mib):
            failures.append(
                f"{device} enlarged peak {cap} MiB exceeds provisional cap")
    passed = not failures
    value: dict[str, Any] = {
        "schema_version": LOCK_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "status": "measured_caps_sealed" if passed else "resource_gate_failed",
        "passed": passed,
        "provisional_peak_mib_each": int(provisional_peak_mib),
        "enlargement": float(enlargement),
        "measured_peak_mib": dict(sorted(peaks.items())),
        "sealed_peak_cap_mib": caps,
        "records": records,
        "failures": failures,
    }
    value["resource_lock_sha256"] = digest_object(value)
    return value, passed, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--node-id", action="append", default=[])
    parser.add_argument("--provisional-peak-mib", type=int, required=True)
    parser.add_argument("--enlargement", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    if (
        arguments.provisional_peak_mib < 1
        or not math.isfinite(arguments.enlargement)
        or arguments.enlargement < 1.0
    ):
        raise ValueError("resource cap and enlargement must be positive")
    value, passed, failures = build_lock(
        arguments.log_dir,
        arguments.node_id,
        provisional_peak_mib=arguments.provisional_peak_mib,
        enlargement=arguments.enlargement,
    )
    target = Path(arguments.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ) + "\n", encoding="utf-8")
    temporary.replace(target)
    if arguments.require_pass and not passed:
        raise SystemExit("Stage B resource gate failed: " + "; ".join(failures))
    print(json.dumps({
        "output": str(target), "passed": passed,
        "sealed_peak_cap_mib": value["sealed_peak_cap_mib"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
