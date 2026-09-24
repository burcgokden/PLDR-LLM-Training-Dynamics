"""Digest-sealed records for the observable-margin campaign."""

from __future__ import annotations

import json
import math
from pathlib import Path
import re

from gate_shape_evidence import (
    digest_object,
    sha256_path,
    write_json_atomic,
)
from observable_margin_specs import CAMPAIGN_ID, RESOURCE_CAPS, STAGE_BY_ID


RECORD_SCHEMA = "pldr-observable-margin-stage-record-v1"
LEDGER_SCHEMA = "pldr-observable-margin-attempt-ledger-v1"
_PORTABLE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATUSES = {"PASS", "FAIL", "BLOCKED"}
_OUTPUT_KINDS = {
    "compact_npz", "sealed_json", "jsonl_ledger", "complete_checkpoint",
    "checksum_manifest", "text_log",
}


def artifact_binding(role, path, *, kind=None):
    target = Path(path).resolve()
    if not target.is_file():
        raise FileNotFoundError(target)
    binding = {
        "role": str(role),
        "path": str(target),
        "sha256": sha256_path(target),
        "bytes": target.stat().st_size,
    }
    if kind is not None:
        if kind not in _OUTPUT_KINDS:
            raise ValueError("unknown observable-margin artifact kind")
        binding["kind"] = kind
    return binding


def _validate_binding(binding, *, output):
    expected = {"role", "path", "sha256", "bytes"}
    if output:
        expected.add("kind")
    if not isinstance(binding, dict) or set(binding) != expected:
        raise ValueError("artifact binding has missing or unknown fields")
    if not _PORTABLE.fullmatch(str(binding["role"])):
        raise ValueError("artifact role is not portable")
    if not Path(binding["path"]).is_absolute():
        raise ValueError("record artifact paths must be resolved")
    if not _SHA256.fullmatch(str(binding["sha256"])):
        raise ValueError("artifact digest is malformed")
    if not isinstance(binding["bytes"], int) or binding["bytes"] < 0:
        raise ValueError("artifact byte count is invalid")
    if output and binding["kind"] not in _OUTPUT_KINDS:
        raise ValueError("output artifact kind is invalid")
    forbidden = {"raw-node", "raw_node", "model-node", "model_node"}
    normalized = f"{binding['role']} {binding['path']}".lower()
    if output and any(marker in normalized for marker in forbidden):
        raise ValueError("raw parameter-node artifacts are forbidden")


def _validate_resources(resources, stage, *, enforce_caps):
    required = {
        "requested_device", "resolved_device", "elapsed_seconds",
        "gpu_device_seconds", "peak_gpu_allocated_bytes",
        "peak_gpu_reserved_bytes", "peak_host_rss_bytes", "output_bytes",
        "stage_cumulative_gpu_hours", "cumulative_gpu_hours",
        "cumulative_wall_clock_hours", "cumulative_output_bytes", "meter_sha256",
    }
    if not isinstance(resources, dict) or set(resources) != required:
        raise ValueError("resource record has missing or unknown fields")
    if resources["requested_device"] not in [
        "cpu", *RESOURCE_CAPS["registered_devices"],
    ]:
        raise ValueError("unregistered requested device")
    if resources["resolved_device"] not in [
        "cpu", *RESOURCE_CAPS["registered_devices"],
    ]:
        raise ValueError("unregistered resolved device")
    numeric = required - {"requested_device", "resolved_device", "meter_sha256"}
    byte_fields = {
        "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes",
        "peak_host_rss_bytes", "output_bytes", "cumulative_output_bytes",
    }
    for name in numeric:
        value = resources[name]
        if (
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"resource quantity {name} is invalid")
        if name in byte_fields and not isinstance(value, int):
            raise ValueError(f"resource byte quantity {name} must be an integer")
    if not _SHA256.fullmatch(str(resources["meter_sha256"])):
        raise ValueError("resource meter digest is malformed")
    unsigned_meter = dict(resources)
    recorded_meter = unsigned_meter.pop("meter_sha256")
    if recorded_meter != digest_object(unsigned_meter):
        raise ValueError("resource meter digest does not replay")
    if enforce_caps and (
        resources["stage_cumulative_gpu_hours"]
        > STAGE_BY_ID[stage]["gpu_hour_cap"]
        or max(
            resources["peak_gpu_allocated_bytes"],
            resources["peak_gpu_reserved_bytes"],
        ) > RESOURCE_CAPS["peak_gpu_allocated_bytes_per_job"]
        or resources["peak_host_rss_bytes"]
        > RESOURCE_CAPS["peak_host_rss_bytes_per_job"]
        or resources["output_bytes"]
        > RESOURCE_CAPS["compact_output_bytes_per_job"]
        or resources["cumulative_gpu_hours"]
        > RESOURCE_CAPS["hard_gpu_hours"]
        or resources["cumulative_wall_clock_hours"]
        > RESOURCE_CAPS["hard_wall_clock_hours"]
        or resources["cumulative_output_bytes"]
        > RESOURCE_CAPS["aggregate_output_bytes"]
    ):
        raise ValueError("resource cap is exceeded")


def seal_record(record):
    value = dict(record)
    value.setdefault("schema_version", RECORD_SCHEMA)
    value.setdefault("campaign_id", CAMPAIGN_ID)
    value.pop("record_sha256", None)
    validate_record(value, sealed=False)
    value["record_sha256"] = digest_object(value)
    return value


def validate_record(record, *, sealed=True):
    required = {
        "schema_version", "campaign_id", "stage", "job_id", "record_key",
        "attempt", "status", "role", "command", "native_arithmetic",
        "deciding_arithmetic", "inputs", "outputs", "checks", "resources",
        "failure",
    }
    if sealed:
        required.add("record_sha256")
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("stage record has missing or unknown fields")
    if record["schema_version"] != RECORD_SCHEMA:
        raise ValueError("unknown observable-margin record schema")
    if record["campaign_id"] != CAMPAIGN_ID:
        raise ValueError("record belongs to another campaign")
    if record["stage"] not in STAGE_BY_ID:
        raise ValueError("record stage is not registered")
    if not _PORTABLE.fullmatch(str(record["job_id"])):
        raise ValueError("record job id is not portable")
    if record["record_key"] != f"{record['stage']}:{record['job_id']}":
        raise ValueError("record key must include stage and job id")
    if not isinstance(record["attempt"], int) or record["attempt"] < 1:
        raise ValueError("record attempt must be a positive integer")
    if record["status"] not in _STATUSES:
        raise ValueError("record status is invalid")
    if record["role"] not in {"construction", "heldout", "shared", "control"}:
        raise ValueError("record ownership role is invalid")
    if (
        not isinstance(record["command"], list) or not record["command"]
        or not all(isinstance(value, str) and value for value in record["command"])
    ):
        raise ValueError("record command must be a nonempty argv vector")
    if record["native_arithmetic"] not in {"float32", "not-applicable"}:
        raise ValueError("native arithmetic label is invalid")
    if record["deciding_arithmetic"] not in {"float64-shadow", "not-applicable"}:
        raise ValueError("deciding arithmetic label is invalid")
    if not isinstance(record["inputs"], list) or not record["inputs"]:
        raise ValueError("record inputs cannot be empty")
    if not isinstance(record["outputs"], list):
        raise ValueError("record outputs must be an array")
    for binding in record["inputs"]:
        _validate_binding(binding, output=False)
    for binding in record["outputs"]:
        _validate_binding(binding, output=True)
    if record["resources"]["output_bytes"] != sum(
        binding["bytes"] for binding in record["outputs"]
    ):
        raise ValueError("resource output bytes do not match bound artifacts")
    if not isinstance(record["checks"], dict) or not record["checks"]:
        raise ValueError("record checks cannot be empty")
    registered_checks = set(STAGE_BY_ID[record["stage"]]["required_checks"])
    if not set(record["checks"]).issubset(registered_checks):
        raise ValueError("record asserts an unregistered stage check")
    if not all(isinstance(value, bool) for value in record["checks"].values()):
        raise ValueError("record checks must be Boolean")
    failure = record["failure"]
    if record["status"] == "PASS":
        if failure is not None or not all(record["checks"].values()):
            raise ValueError("a passing record cannot carry a failure")
    else:
        if not isinstance(failure, dict) or set(failure) != {"code", "message"}:
            raise ValueError("failed and blocked records need a structured failure")
        if not all(isinstance(failure[name], str) and failure[name] for name in failure):
            raise ValueError("structured failure fields cannot be empty")
    _validate_resources(
        record["resources"], record["stage"],
        enforce_caps=record["status"] == "PASS",
    )
    if sealed:
        unsigned = dict(record)
        digest = unsigned.pop("record_sha256")
        if digest != digest_object(unsigned):
            raise ValueError("record digest does not replay")
    return True


def validate_attempt_ledger(records):
    seen = set()
    seen_digests = set()
    attempts = {}
    stage_gpu_seconds = {stage: 0.0 for stage in STAGE_BY_ID}
    campaign_gpu_seconds = 0.0
    campaign_wall_seconds = 0.0
    campaign_output_bytes = 0
    for record in records:
        validate_record(record)
        identity = (record["record_key"], record["attempt"])
        if identity in seen:
            raise ValueError("stage/job/attempt identity is duplicated")
        if record["record_sha256"] in seen_digests:
            raise ValueError("attempt record digest is duplicated")
        seen.add(identity)
        seen_digests.add(record["record_sha256"])
        previous = attempts.get(record["record_key"], 0)
        if record["attempt"] != previous + 1:
            raise ValueError(
                f"attempt sequence for {record['record_key']} is out of order")
        attempts[record["record_key"]] = record["attempt"]
        resources = record["resources"]
        gpu_seconds = float(resources["gpu_device_seconds"])
        elapsed_seconds = float(resources["elapsed_seconds"])
        output_bytes = int(resources["output_bytes"])
        stage_gpu_seconds[record["stage"]] += gpu_seconds
        campaign_gpu_seconds += gpu_seconds
        campaign_wall_seconds += elapsed_seconds
        campaign_output_bytes += output_bytes
        expected = {
            "stage_cumulative_gpu_hours": (
                stage_gpu_seconds[record["stage"]] / 3600.0),
            "cumulative_gpu_hours": campaign_gpu_seconds / 3600.0,
            "cumulative_wall_clock_hours": campaign_wall_seconds / 3600.0,
        }
        for name, value in expected.items():
            if not math.isclose(
                float(resources[name]), value, rel_tol=1e-12, abs_tol=1e-12,
            ):
                raise ValueError(
                    f"resource ledger cumulative {name} does not replay")
        if int(resources["cumulative_output_bytes"]) != campaign_output_bytes:
            raise ValueError("resource ledger cumulative output does not replay")
    return True


def load_attempt_ledger(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or set(value) != {
        "schema_version", "campaign_id", "records", "ledger_sha256",
    }:
        raise ValueError("attempt ledger has missing or unknown fields")
    if value["schema_version"] != LEDGER_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise ValueError("unknown attempt ledger")
    unsigned = dict(value)
    digest = unsigned.pop("ledger_sha256")
    if digest != digest_object(unsigned):
        raise ValueError("attempt ledger digest does not replay")
    validate_attempt_ledger(value["records"])
    return value


def write_attempt_ledger(path, records):
    records = list(records)
    validate_attempt_ledger(records)
    value = {
        "schema_version": LEDGER_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "records": records,
    }
    value["ledger_sha256"] = digest_object(value)
    write_json_atomic(path, value)
    return value
