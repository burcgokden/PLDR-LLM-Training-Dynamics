"""Append-only sealing of source predictions and later native observations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from confirm.block_normal import backward_metrics, block_prediction
from confirm.block_normal_specs import (
    CAMPAIGN_ID,
    NATIVE_EVIDENCE_SCHEMA,
    REGISTRY,
    SOURCE_EVIDENCE_SCHEMA,
    TRAJECTORIES,
)
from confirm.native_determinism import sha256_path


SOURCE_SPEC_FIELDS = {
    "stage",
    "trajectory",
    "seed",
    "block_start",
    "block_end",
    "operators",
    "quadratic_coefficients",
    "forces",
    "radii",
    "initial_normal_upper",
    "comparison_bounds",
}
SOURCE_FIELDS = {
    "schema_version",
    "campaign_id",
    "stage",
    "trajectory",
    "seed",
    "block_start",
    "block_end",
    "source_checkpoint_path",
    "source_checkpoint_sha256",
    "created_at_ns",
    "operators",
    "quadratic_coefficients",
    "forces",
    "radii",
    "initial_normal_upper",
    "comparison_bounds",
}
COMPARISON_BOUND_FIELDS = {
    "comparison_id",
    "prediction",
    "map_id",
    "context",
    "layer",
    "head",
    "bound_lower",
    "bound_upper",
}
OBSERVATION_FIELDS = {
    "comparison_id",
    "observed_lower",
    "observed_upper",
}
NATIVE_FIELDS = {
    "schema_version",
    "campaign_id",
    "source_prediction_sha256",
    "source_checkpoint_sha256",
    "native_checkpoint_path",
    "native_checkpoint_sha256",
    "opened_at_ns",
    "observations",
}


def canonical_json(value: Any) -> bytes:
    """Encode campaign JSON deterministically and reject nonfinite numbers."""

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def load_json_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence input must contain one JSON object")
    return value


def write_new_json(path: str | Path, value: dict[str, Any]) -> None:
    """Write one canonical artifact atomically and refuse replacement."""

    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to replace sealed artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary evidence path already exists: {temporary}")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical_json(value))
            stream.flush()
            os.fsync(stream.fileno())
        if target.exists():
            raise FileExistsError(
                f"refusing to replace sealed artifact: {target}"
            )
        os.link(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite_scalar(value: Any, name: str, *, nonnegative: bool) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        qualifier = "nonnegative " if nonnegative else ""
        raise ValueError(f"{name} must be a finite {qualifier}number")
    return result


def _validate_role(stage: str, trajectory: str, seed: int) -> None:
    rows = {row["name"]: row for row in TRAJECTORIES}
    if stage not in {"Q1", "Q2", "Q3", "Q4", "Q5"}:
        raise ValueError("source evidence stage must be Q1 through Q5")
    if trajectory not in rows or seed != rows[trajectory]["seed"]:
        raise ValueError("trajectory and seed do not match the frozen design")
    heldout = bool(rows[trajectory]["heldout"])
    if (stage == "Q5") != heldout:
        raise ValueError("only Q5 may use held-out trajectories")


def _validate_bounds(value: Any) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ValueError("source evidence needs comparison bounds")
    rows = []
    identifiers = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != COMPARISON_BOUND_FIELDS:
            raise ValueError(f"comparison bound {index} has an invalid schema")
        row = dict(raw)
        for name in ("comparison_id", "prediction", "map_id"):
            if not isinstance(row[name], str) or not row[name]:
                raise ValueError(f"comparison bound {index} has an invalid {name}")
        if row["comparison_id"] in identifiers:
            raise ValueError("comparison identifiers must be unique")
        identifiers.add(row["comparison_id"])
        if row["context"] is not None and (
            not isinstance(row["context"], str) or not row["context"]
        ):
            raise ValueError("comparison context must be a string or null")
        for name, limit in (
            ("layer", len(REGISTRY["layers"])),
            ("head", len(REGISTRY["heads"])),
        ):
            if row[name] is not None:
                item = _integer(row[name], f"comparison {name}")
                if not 0 <= item < limit:
                    raise ValueError(f"comparison {name} is outside the registry")
        lower = _finite_scalar(
            row["bound_lower"], "bound_lower", nonnegative=True
        )
        upper = _finite_scalar(
            row["bound_upper"], "bound_upper", nonnegative=True
        )
        if lower > upper:
            raise ValueError("comparison bound interval is reversed")
        row["bound_lower"] = lower
        row["bound_upper"] = upper
        rows.append(row)
    return rows


def seal_source(
    specification: Mapping[str, Any],
    checkpoint_path: str | Path,
    *,
    created_at_ns: int | None = None,
) -> dict[str, Any]:
    """Validate and seal one source-only block prediction."""

    if set(specification) != SOURCE_SPEC_FIELDS:
        raise ValueError("source specification has an invalid schema")
    stage = specification["stage"]
    trajectory = specification["trajectory"]
    if not isinstance(stage, str) or not isinstance(trajectory, str):
        raise ValueError("source stage and trajectory must be strings")
    seed = _integer(specification["seed"], "seed")
    _validate_role(stage, trajectory, seed)
    block_start = _integer(specification["block_start"], "block_start")
    block_end = _integer(specification["block_end"], "block_end")
    if block_start < 0 or block_end <= block_start:
        raise ValueError("source block endpoints are invalid")

    operators = np.asarray(specification["operators"], dtype=np.float64)
    if (
        operators.ndim != 3
        or operators.shape[0] < 1
        or operators.shape[1] != operators.shape[2]
        or not np.isfinite(operators).all()
    ):
        raise ValueError("source operators must be finite square matrices")
    dimension = operators.shape[1]
    metrics = backward_metrics(list(operators), np.eye(dimension))
    quadratic = np.asarray(
        specification["quadratic_coefficients"], dtype=np.float64
    )
    forces = np.asarray(specification["forces"], dtype=np.float64)
    radii = np.asarray(specification["radii"], dtype=np.float64)
    initial = _finite_scalar(
        specification["initial_normal_upper"],
        "initial_normal_upper",
        nonnegative=True,
    )
    prediction = block_prediction(
        metrics["gains"], quadratic, forces, radii, initial
    )
    if prediction["gains"].shape != (operators.shape[0],):
        raise AssertionError("source metric construction changed block count")
    bounds = _validate_bounds(specification["comparison_bounds"])

    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("source checkpoint does not exist")
    created = time.time_ns() if created_at_ns is None else _integer(
        created_at_ns, "created_at_ns"
    )
    if created <= 0:
        raise ValueError("source creation timestamp must be positive")
    return {
        "schema_version": SOURCE_EVIDENCE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "stage": stage,
        "trajectory": trajectory,
        "seed": seed,
        "block_start": block_start,
        "block_end": block_end,
        "source_checkpoint_path": str(checkpoint),
        "source_checkpoint_sha256": sha256_path(checkpoint),
        "created_at_ns": created,
        "operators": operators.tolist(),
        "quadratic_coefficients": quadratic.tolist(),
        "forces": forces.tolist(),
        "radii": radii.tolist(),
        "initial_normal_upper": initial,
        "comparison_bounds": bounds,
    }


def _validate_source(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != SOURCE_FIELDS
        or value.get("schema_version") != SOURCE_EVIDENCE_SCHEMA
        or value.get("campaign_id") != CAMPAIGN_ID
    ):
        raise ValueError("native evidence requires a sealed source artifact")
    return value


def _validate_observations(
    value: Any, expected_identifiers: set[str]
) -> list[dict[str, Any]]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or not value
    ):
        raise ValueError("native evidence needs observations")
    rows = []
    identifiers = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != OBSERVATION_FIELDS:
            raise ValueError(f"native observation {index} has an invalid schema")
        row = dict(raw)
        identifier = row["comparison_id"]
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("native comparison identifier is invalid")
        if identifier in identifiers:
            raise ValueError("native comparison identifiers must be unique")
        identifiers.add(identifier)
        lower = _finite_scalar(
            row["observed_lower"], "observed_lower", nonnegative=True
        )
        upper = _finite_scalar(
            row["observed_upper"], "observed_upper", nonnegative=True
        )
        if lower > upper:
            raise ValueError("native observation interval is reversed")
        row["observed_lower"] = lower
        row["observed_upper"] = upper
        rows.append(row)
    if identifiers != expected_identifiers:
        raise ValueError("native and source comparison membership disagrees")
    return rows


def seal_native(
    source_path: str | Path,
    native_checkpoint_path: str | Path,
    observations: Sequence[Mapping[str, Any]],
    *,
    opened_at_ns: int | None = None,
) -> dict[str, Any]:
    """Open a native successor only after its source file has been sealed."""

    source_file = Path(source_path).resolve()
    source = _validate_source(load_json_object(source_file))
    source_digest = sha256_path(source_file)
    expected = {
        row["comparison_id"] for row in source["comparison_bounds"]
    }
    normalized = _validate_observations(observations, expected)

    native_checkpoint = Path(native_checkpoint_path).resolve()
    source_checkpoint = Path(source["source_checkpoint_path"]).resolve()
    if native_checkpoint == source_checkpoint:
        raise ValueError("native successor must differ from its source checkpoint")
    if (
        not source_checkpoint.is_file()
        or sha256_path(source_checkpoint)
        != source["source_checkpoint_sha256"]
    ):
        raise ValueError("sealed source checkpoint changed before native opening")
    if not native_checkpoint.is_file():
        raise FileNotFoundError("native successor checkpoint does not exist")
    opened = time.time_ns() if opened_at_ns is None else _integer(
        opened_at_ns, "opened_at_ns"
    )
    if opened <= source["created_at_ns"]:
        raise ValueError("native successor must open after source sealing")
    return {
        "schema_version": NATIVE_EVIDENCE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_prediction_sha256": source_digest,
        "source_checkpoint_sha256": source["source_checkpoint_sha256"],
        "native_checkpoint_path": str(native_checkpoint),
        "native_checkpoint_sha256": sha256_path(native_checkpoint),
        "opened_at_ns": opened,
        "observations": normalized,
    }
