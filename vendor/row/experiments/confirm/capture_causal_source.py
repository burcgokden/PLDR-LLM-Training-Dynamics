#!/usr/bin/env python3
"""Live source/JVP capture on the causal campaign's 500-update grid.

The implementation reuses the tested chronological producer while replacing
only its immutable campaign specification and registry/lock validators.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm import capture_chronological_confirmation as producer  # noqa: E402
from confirm.causal_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    NUMERICAL_POLICIES,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    validate_measurement_registry,
)


REGISTRY_SCHEMA = "pldr-causal-probe-registry-v1"


def _load_registry(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    validate_measurement_registry(value, require_nonempty=True)
    contexts = value.get("construction", []) + value.get("validation", [])
    if (
        value.get("schema_version") != REGISTRY_SCHEMA
        or len(contexts) != REGISTRY["context_count"]
        or len({int(row["chunk_index"]) for row in contexts}) != len(contexts)
    ):
        raise ValueError("causal capture registry is invalid")
    return value


def _load_lock(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
    ):
        raise ValueError("causal construction lock is invalid")
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if recorded != digest_object(unsigned):
        raise ValueError("causal construction lock digest does not replay")
    return value


def main() -> None:
    producer.ARCHITECTURE = ARCHITECTURE
    producer.CAMPAIGN_ID = CAMPAIGN_ID
    producer.LOCK_SCHEMA = LOCK_SCHEMA
    producer.NUMERICAL_POLICIES = NUMERICAL_POLICIES
    producer.OPTIMIZER = OPTIMIZER
    producer.REGISTRY = REGISTRY
    producer.TRAJECTORIES = TRAJECTORIES
    producer._load_registry = _load_registry
    producer._load_lock = _load_lock
    producer.main()


if __name__ == "__main__":
    main()
