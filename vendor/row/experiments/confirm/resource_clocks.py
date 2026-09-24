"""Schema-gated compatibility reader for resource clock records."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CURRENT_SCHEMA = "pldr-direct-work-resource-record-v2"
LEGACY_SCHEMAS = frozenset({
    "pldr-observable-balance-attempt-v1",
    "pldr-observable-cocycle-attempt-v1",
})


@dataclass(frozen=True)
class ResourceClocks:
    attempt_elapsed_wall_seconds: float
    accelerator_reservation_wall_seconds: float
    producer_process_elapsed_seconds: float | None
    legacy_schema: bool


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"resource clock is invalid: {name}")
    return number


def normalize_resource_clocks(
    record: dict[str, Any],
    *,
    producer_process_elapsed_seconds: float | None = None,
) -> ResourceClocks:
    """Normalize clocks while accepting legacy names only for legacy schemas."""
    schema = record.get("schema_version")
    if schema == CURRENT_SCHEMA:
        forbidden = {"elapsed_seconds", "wall_seconds"} & set(record)
        if forbidden:
            raise ValueError("current resource schema contains a legacy clock name")
        attempt = _finite_nonnegative(
            record["attempt_elapsed_wall_seconds"],
            "attempt_elapsed_wall_seconds",
        )
        accelerator = _finite_nonnegative(
            record["accelerator_reservation_wall_seconds"],
            "accelerator_reservation_wall_seconds",
        )
        if accelerator > attempt:
            raise ValueError("accelerator reservation exceeds attempt wall time")
        legacy = False
    elif schema in LEGACY_SCHEMAS:
        if (
            "elapsed_seconds" not in record
            or "attempt_elapsed_wall_seconds" in record
            or "accelerator_reservation_wall_seconds" in record
        ):
            raise ValueError("legacy resource clock fields disagree with schema")
        attempt = _finite_nonnegative(record["elapsed_seconds"], "elapsed_seconds")
        device = str(record.get("planned_device", record.get("device", "")))
        accelerator = attempt if device.startswith("cuda:") else 0.0
        legacy = True
    else:
        raise ValueError(f"unknown resource clock schema: {schema!r}")
    producer = (
        None
        if producer_process_elapsed_seconds is None
        else _finite_nonnegative(
            producer_process_elapsed_seconds,
            "producer_process_elapsed_seconds",
        )
    )
    return ResourceClocks(
        attempt_elapsed_wall_seconds=attempt,
        accelerator_reservation_wall_seconds=accelerator,
        producer_process_elapsed_seconds=producer,
        legacy_schema=legacy,
    )
