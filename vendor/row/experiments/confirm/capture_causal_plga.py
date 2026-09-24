#!/usr/bin/env python3
"""Capture composite PLGA tensors under the causal campaign contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm import capture_chronological_plga as producer  # noqa: E402
from confirm.causal_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    REGISTRY,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    validate_measurement_registry,
)


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _load_registry(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    validate_measurement_registry(value, require_nonempty=True)
    if value.get("schema_version") != "pldr-causal-probe-registry-v1":
        raise ValueError("PLGA capture needs the causal registry")
    return value


def _load_lock(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("causal construction lock is invalid")
    return value


def main() -> None:
    producer.ARCHITECTURE = ARCHITECTURE
    producer.CAMPAIGN_ID = CAMPAIGN_ID
    producer.LOCK_SCHEMA = LOCK_SCHEMA
    producer.REGISTRY = REGISTRY
    producer._load_registry = _load_registry
    producer._load_lock = _load_lock
    producer.main()


if __name__ == "__main__":
    main()
