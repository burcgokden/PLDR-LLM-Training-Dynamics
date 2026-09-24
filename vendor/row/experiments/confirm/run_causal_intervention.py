#!/usr/bin/env python3
"""Run registered complete-source-removal arms for the causal campaign."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(EXPERIMENTS))

from confirm import run_chronological_intervention as producer  # noqa: E402
from confirm.causal_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    LOCK_SCHEMA,
    OPTIMIZER,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    validate_measurement_registry,
)


def _argument(name: str) -> str:
    try:
        return sys.argv[sys.argv.index(name) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"causal intervention requires {name}") from error


def _load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON input must be an object")
    return value


def _load_registry(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    validate_measurement_registry(value, require_nonempty=True)
    if value.get("schema_version") != "pldr-causal-probe-registry-v1":
        raise ValueError("interventions need the causal registry")
    return value


def _load_lock(path: str | Path) -> dict[str, Any]:
    value = _load_json(path)
    unsigned = dict(value)
    recorded = unsigned.pop("lock_sha256", None)
    if (
        value.get("schema_version") != LOCK_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
        or value.get("constant_enlargement_after_lock") is not False
        or float(value.get("intervention_minimum_effect", 0.0)) <= 0.0
        or recorded != digest_object(unsigned)
    ):
        raise ValueError("causal construction lock is invalid")
    return value


def main() -> None:
    unit_id = _argument("--unit-id")
    match = re.fullmatch(r"cri-s\d+-(\d{2})", unit_id)
    if match is None:
        raise ValueError("causal intervention unit id is malformed")
    ordinal = int(match.group(1))
    anchors = tuple(REGISTRY["intervention_anchor_steps"])
    horizons = tuple(REGISTRY["intervention_horizons"])
    if not 0 <= ordinal < len(anchors) * len(horizons):
        raise ValueError("causal intervention ordinal is outside the registry")
    anchor = anchors[ordinal // len(horizons)]
    horizon = horizons[ordinal % len(horizons)]
    local_registry = dict(REGISTRY)
    local_registry.update({
        "intervention_anchor_step": anchor,
        "intervention_pair": (0, 1),
        "intervention_steps": horizon,
    })
    producer.CAMPAIGN_ID = CAMPAIGN_ID
    producer.LOCK_SCHEMA = LOCK_SCHEMA
    producer.OPTIMIZER = OPTIMIZER
    producer.REGISTRY = local_registry
    producer.TRAJECTORIES = TRAJECTORIES
    producer._load_registry = _load_registry
    producer._load_lock = _load_lock
    producer.main()


if __name__ == "__main__":
    main()
