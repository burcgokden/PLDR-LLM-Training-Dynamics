"""Complete-checkpoint recovery and retention closure."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from pathlib import Path
from typing import Any

from confirm.native_determinism import compare_trees, sha256_path


def retention_record(
    complete_checkpoint: str | Path,
    model_only_checkpoint: str | Path,
    *,
    open_descendants: list[str],
    recovery_report: Mapping[str, Any],
    maximum_complete_checkpoints: int,
    complete_checkpoint_count: int,
) -> dict[str, Any]:
    """Recompute cleanup authorization without deleting a checkpoint."""

    complete = Path(complete_checkpoint).resolve()
    model_only = Path(model_only_checkpoint).resolve()
    if complete == model_only:
        raise ValueError("model-only output must not overwrite the checkpoint")
    if not complete.is_file() or not model_only.is_file():
        raise FileNotFoundError("retention closure needs both checkpoint files")
    if maximum_complete_checkpoints < 1 or complete_checkpoint_count < 1:
        raise ValueError("checkpoint retention counts must be positive")
    recovery_ok = bool(
        recovery_report.get("complete_reload_succeeded")
        and recovery_report.get("model_state_equal")
        and recovery_report.get("optimizer_state_equal")
        and recovery_report.get("scheduler_state_equal")
        and recovery_report.get("rng_state_equal")
    )
    storage = complete.stat().st_size + model_only.stat().st_size
    cleanup_authorized = bool(
        recovery_ok
        and not open_descendants
        and complete_checkpoint_count > maximum_complete_checkpoints
    )
    return {
        "schema_version": "pldr-block-normal-retention-closure-v1",
        "complete_checkpoint_path": str(complete),
        "complete_checkpoint_sha256": sha256_path(complete),
        "model_only_checkpoint_path": str(model_only),
        "model_only_checkpoint_sha256": sha256_path(model_only),
        "paths_distinct": True,
        "open_descendants": list(open_descendants),
        "recovery": dict(recovery_report),
        "recovery_complete": recovery_ok,
        "measured_storage_bytes": storage,
        "maximum_complete_checkpoints": maximum_complete_checkpoints,
        "complete_checkpoint_count": complete_checkpoint_count,
        "cleanup_authorized": cleanup_authorized,
        "deletion_performed": False,
    }


def in_memory_recovery_check(
    source: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> dict[str, Any]:
    """Check all state families after a complete checkpoint reload fixture."""

    required = ("model", "optimizer", "scheduler", "rng")
    if any(name not in source or name not in restored for name in required):
        raise ValueError("complete recovery fixture is missing a state family")
    comparisons = {
        name: compare_trees(source[name], restored[name]) for name in required
    }
    return {
        "complete_reload_succeeded": True,
        "model_state_equal": comparisons["model"]["bitwise_equal"],
        "optimizer_state_equal": comparisons["optimizer"]["bitwise_equal"],
        "scheduler_state_equal": comparisons["scheduler"]["bitwise_equal"],
        "rng_state_equal": comparisons["rng"]["bitwise_equal"],
        "comparisons": comparisons,
    }


def model_only_payload(complete_state: Mapping[str, Any]) -> dict[str, Any]:
    """Create a detached model-only payload without mutating complete state."""

    if "model" not in complete_state:
        raise ValueError("complete checkpoint has no model state")
    return {
        "schema_version": "pldr-model-only-copy-v1",
        "model": copy.deepcopy(complete_state["model"]),
        "source_complete_checkpoint_retained": True,
    }

