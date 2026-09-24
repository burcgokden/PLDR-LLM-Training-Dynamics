from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
CONFIRM = ROOT / "experiments" / "confirm"
ANALYSIS = ROOT / "experiments" / "analysis"
SCRIPTS = ROOT / "scripts"
for path in (CONFIRM, ANALYSIS, SCRIPTS):
    sys.path.insert(0, str(path))

from analyze_rg_universality import (  # noqa: E402
    analyze,
    load_trajectory_artifact,
)
from rg_universality_protocol_specs import (  # noqa: E402
    ARCHITECTURE_RUNS,
    BLOCK_SIZES,
    RESOURCE_CAPS,
    STAGES,
)
from run_rg_universality_campaign import plan  # noqa: E402


def _write_artifact(path, seed, width=32, depth=2):
    generator = np.random.default_rng(seed)
    matrix = np.diag([0.72, 0.48])
    factor = np.linalg.cholesky(np.eye(2) - matrix @ matrix.T)
    replicas = 96
    steps = 64
    paths = np.empty((replicas, steps + 1, 2))
    paths[:, 0] = generator.normal(size=(replicas, 2))
    for step in range(steps):
        paths[:, step + 1] = (
            paths[:, step] @ matrix.T
            + generator.normal(size=(replicas, 2)) @ factor.T
        )
    labels = np.broadcast_to(
        (np.arange(replicas) % 2)[:, None], paths.shape[:2]).copy()
    anchor_labels = np.arange(replicas) % 4
    np.savez_compressed(
        path,
        normal_paths=paths,
        hidden_labels=labels,
        anchor_labels=anchor_labels,
        model_width=np.array(width),
        model_depth=np.array(depth),
        seed=np.array(seed),
        source_state_sha256=np.array([
            f"{seed * 10 + index:064x}" for index in range(4)]),
    )


def test_rg_program_is_separate_and_fits_two_gpu_inventory():
    assert [stage["id"] for stage in STAGES] == [
        "R0", "R1", "R2", "R3", "U1", "U2", "U3"]
    assert BLOCK_SIZES == (1, 2, 4, 8, 16, 32)
    assert RESOURCE_CAPS["gpu_count"] == 2
    assert RESOURCE_CAPS[
        "campaign_is_separate_from_normal_stability_confirmation"]
    assert len(ARCHITECTURE_RUNS) == 3
    assert plan()["program"] == "separate_transition_kernel_rg_and_universality"




def test_trajectory_loader_rejects_unknown_fields(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, unknown=np.array([1]))
    with pytest.raises(ValueError, match="keys disagree"):
        load_trajectory_artifact(path)


def test_kernel_flow_analyzer_derives_record_from_raw_paths(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_artifact(first, 501)
    _write_artifact(second, 502)
    result = analyze([first, second], "kernel-flow")
    assert result["schema_version"] == "pldr-rg-universality-analysis-v1"
    assert len(result["systems"]) == 2
    assert len(result["record_sha256"]) == 64
    assert set(result["required_conditions"]) == {
        "at_least_two_systems",
        "every_system_kernel_and_flow_qualified",
    }
