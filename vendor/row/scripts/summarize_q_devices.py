#!/usr/bin/env python3
"""Validate and summarize the two target-device Q records."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(CONFIRM))

from confirmation_artifacts import (  # noqa: E402
    seal_record,
    verify_sealed_record,
    write_json_atomic,
)


DEFAULT_DIRECTORY = (
    ROOT.parent / "experiment-data" / "manuscript-revisions" / "rev33"
    / "confirmation-program" / "campaigns" / "qualification"
)


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    verify_sealed_record(value)
    return value


def _device_row(record, expected_device):
    if (
        record.get("schema_version") != "pldr-q-live-record-v2"
        or record.get("stage") != "Q"
        or record.get("device") != expected_device
    ):
        raise ValueError(f"{expected_device} Q record has the wrong identity")
    diagnostics = record["jvp_and_taylor"]
    rows = diagnostics["taylor_rows"]
    ratios = [
        row["certified_remainder_upper"]
        / max(row["observed_remainder"], float.fromhex("0x1p-1022"))
        for row in rows
    ]
    ratio = max(ratios, default=math.inf)
    conditions = record["conditions"]
    qualified = (
        record["decision"] == "QUALIFIED"
        and all(conditions.values())
        and diagnostics.get("batch_size") == 8
        and diagnostics.get("context_length") == 8
        and diagnostics.get("row_count") == 64
        and math.isfinite(ratio)
        and ratio <= 1e3
    )
    return {
        "device": expected_device,
        "scientific_record_sha256": record["scientific_record_sha256"],
        "source_complete_state_sha256": (
            record["owned_edge"]["source_state_sha256"]
        ),
        "target_complete_state_sha256": (
            record["owned_edge"]["target_state_sha256"]
        ),
        "row_count": diagnostics["row_count"],
        "batch_size": diagnostics["batch_size"],
        "context_length": diagnostics["context_length"],
        "maximum_enclosure_ratio": ratio,
        "layernorm_centered_floor": record["measured_geometry"][
            "layernorm_centered_floor"
        ],
        "wall_clock_seconds": record["resource_metadata"][
            "wall_clock_seconds"
        ],
        "qualified": qualified,
    }


def expected(directory):
    directory = Path(directory)
    records = [
        _device_row(_load(directory / "q_cuda0.json"), "cuda:0"),
        _device_row(_load(directory / "q_cuda1.json"), "cuda:1"),
    ]
    geometry_matches = all(
        (row["batch_size"], row["context_length"], row["row_count"])
        == (8, 8, 64)
        for row in records
    )
    return seal_record({
        "schema_version": "pldr-target-device-q-summary-v3",
        "devices": records,
        "geometry_matches": geometry_matches,
        "both_devices_qualified": (
            geometry_matches and all(row["qualified"] for row in records)
        ),
        "resource_metadata": {
            "gpu_device_seconds": [
                row["wall_clock_seconds"] for row in records
            ],
            "total_gpu_hours": sum(
                row["wall_clock_seconds"] for row in records
            ) / 3600.0,
        },
    })


def _payload(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _manifest(directory, report_payload):
    directory = Path(directory)
    payloads = {
        "q_cuda0.json": (directory / "q_cuda0.json").read_bytes(),
        "q_cuda1.json": (directory / "q_cuda1.json").read_bytes(),
        "q_device_summary.json": report_payload,
    }
    return "".join(
        f"{hashlib.sha256(payloads[name]).hexdigest()}  {name}\n"
        for name in sorted(payloads)
    ).encode("ascii")


def summarize(directory, check=False):
    directory = Path(directory).resolve()
    report = expected(directory)
    report_payload = _payload(report)
    manifest_payload = _manifest(directory, report_payload)
    report_path = directory / "q_device_summary.json"
    manifest_path = directory / "QUALIFICATION_MANIFEST.sha256"
    if check:
        if (
            not report_path.is_file()
            or report_path.read_bytes() != report_payload
            or not manifest_path.is_file()
            or manifest_path.read_bytes() != manifest_payload
        ):
            raise SystemExit("target-device Q summary or manifest is stale")
        if not report["both_devices_qualified"]:
            raise SystemExit("target-device Q is not qualified")
        print("target-device Q: both records and manifest verified")
        return
    write_json_atomic(report_path, report)
    manifest_path.write_bytes(manifest_payload)
    if not report["both_devices_qualified"]:
        raise SystemExit("target-device Q is not qualified")
    print("target-device Q: both devices qualified")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", default=str(DEFAULT_DIRECTORY))
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    summarize(arguments.directory, arguments.check)


if __name__ == "__main__":
    main()
