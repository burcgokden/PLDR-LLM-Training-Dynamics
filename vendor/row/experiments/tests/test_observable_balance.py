from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from confirm.observable_balance import (
    balance_metrics,
    campaign_classification,
    cell_evaluable,
    select_layer_horizon,
)
from confirm.observable_balance_specs import (
    DECISION_POLICY,
    DIRECTION_CLASSES,
    HORIZON_GRID,
    SOURCE_STEPS,
    campaign_design,
)

from scripts.execute_observable_balance_plan import _validate_plan
from scripts.gen_observable_balance_protocols import (
    generated_files,
    record_inventory,
)
from scripts.seal_observable_balance_provenance import _verify_incidents
from scripts.stage_observable_balance_confirmation import (
    bound_path,
    default_paths,
    launch_plan,
    source_import_closure,
)

def test_design_predeclares_campaign_level_support_and_refutation() -> None:
    design = campaign_design()
    policy = design["decision_policy"]
    assert policy["campaign_support_fraction"] == 2.0 / 3.0
    assert "refutation" in " ".join(policy).lower()
    assert design["execution_policy"]["planned_devices_are_immutable"]
    assert design["execution_policy"]["runtime_and_incident_membership_is_sealed"]


def test_exact_balance_retains_cross_term_cancellation() -> None:
    homogeneous = np.asarray([100.0, 0.0])
    state = np.asarray([-99.5, 2.0])
    observation = np.asarray([-0.4, -2.0])
    observed = homogeneous + state + observation
    result = balance_metrics(homogeneous, state, observation, observed)
    assert result["reconstruction_relative_residual"] == 0.0
    assert result["cross_gram"].shape == (3, 3)
    assert result["homogeneous_correction_inner_product"] < 0.0
    assert result["cancellation_index"] > 10.0
    assert result["cancellation_dominant"]


def _construction_rows(valid_horizons: set[int]) -> list[dict[str, object]]:
    rows = []
    for source in SOURCE_STEPS:
        for direction in DIRECTION_CLASSES:
            for horizon in HORIZON_GRID:
                for amplitude_index in (0, 1):
                    rows.append(
                        {
                            "source_step": source,
                            "direction": direction,
                            "horizon": horizon,
                            "amplitude_index": amplitude_index,
                            "evaluable": horizon in valid_horizons,
                            "reconstruction_relative_residual": (
                                1.0e-8 if amplitude_index == 0 else 5.0e-9
                            ),
                        }
                    )
    return rows


def test_lock_selects_longest_jointly_evaluable_horizon() -> None:
    selected = select_layer_horizon(_construction_rows({1, 2, 4}))
    assert selected["selected_horizon"] == 4
    assert selected["jointly_evaluable"]
    fallback = select_layer_horizon(_construction_rows(set()))
    assert fallback["selected_horizon"] == 1
    assert fallback["fallback_used"]


def test_cell_gate_is_fail_capable() -> None:
    assert cell_evaluable(
        effect_norm=1.0,
        reconstruction_relative_residual=1.0e-6,
        replay_relative_residual=1.0e-8,
    )
    assert not cell_evaluable(
        effect_norm=0.5 * DECISION_POLICY["absolute_effect_floor"],
        reconstruction_relative_residual=0.0,
        replay_relative_residual=0.0,
    )


def test_campaign_decision_does_not_pool_heldout_trajectories() -> None:
    rows = []
    for label, dominant_count in (("B", 18), ("C", 17)):
        for index in range(27):
            rows.append(
                {
                    "trajectory_label": label,
                    "evaluable": True,
                    "cancellation_dominant": index < dominant_count,
                }
            )
    result = campaign_classification(rows)
    assert result["outcome"] == "refuted"
    assert result["trajectories"]["B"]["support"]
    assert not result["trajectories"]["C"]["support"]


def test_inventory_and_launch_graph_freeze_roles_devices_and_dependencies() -> None:
    inventory = record_inventory()
    assert inventory["expected_counts"] == {
        "qualification": 1,
        "construction": 9,
        "prediction": 18,
        "validation": 18,
        "stratum_control": 9,
        "analysis": 1,
        "total_plan_nodes": 57,
        "heldout_cells": 54,
        "construction_cells": 216,
    }
    plan = launch_plan()
    nodes = _validate_plan(plan)
    assert len(nodes) == 57
    positions = {node["id"]: index for index, node in enumerate(plan["nodes"])}
    assert all(
        positions[dependency] < positions[node["id"]]
        for node in plan["nodes"]
        for dependency in node["depends_on"]
    )
    assert len(nodes["construction-lock"]["depends_on"]) == 9
    for node in plan["nodes"]:
        command = node["command"]
        if node["device"].startswith("cuda:"):
            assert command[command.index("--device") + 1] == node["device"]
            assert "{bundle}/protocol/measurement-registry.json" in command
        else:
            assert "--device" not in command


def test_generated_protocol_checksums_and_attempt_total_replay() -> None:
    files = generated_files()
    for line in files[Path("CHECKSUMS.sha256")].decode("ascii").splitlines():
        digest, relative = line.split("  ", 1)
        assert hashlib.sha256(files[Path(relative)]).hexdigest() == digest
    ledger = json.loads(files[Path("archive-attempt-ledger.json")])
    assert ledger["prior_total_hours"] == pytest.approx(11.7753)
    design = json.loads(files[Path("campaign-design.json")])
    digest = design.pop("design_sha256")
    compact = json.dumps(
        design, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert hashlib.sha256(compact).hexdigest() == digest


def test_source_closure_contains_all_scientific_and_provenance_entrypoints() -> None:
    relative = {
        path.relative_to(ROOT).as_posix() for path in source_import_closure()
    }
    assert {
        "experiments/confirm/observable_balance.py",
        "experiments/confirm/observable_balance_live.py",
        "experiments/confirm/observable_balance_specs.py",
        "experiments/analysis/analyze_observable_balance_confirmation.py",
        "experiments/analysis/summarize_observable_balance_manuscript.py",
        "scripts/execute_observable_balance_plan.py",
        "scripts/seal_observable_balance_provenance.py",
        "scripts/stage_observable_balance_confirmation.py",
    } <= relative


def test_executor_has_no_device_override_or_hardcoded_memory_cap() -> None:
    source = (ROOT / "scripts/execute_observable_balance_plan.py").read_text()
    assert "device_override" not in source
    assert 'add_argument("--device")' not in source
    assert "20 * 1024**3" not in source
    assert 'resource_budget["process_reserved_memory_cap_bytes"]' in source


def test_live_producer_sets_cublas_before_torch_import_and_records_environment() -> None:
    source = (ROOT / "experiments/confirm/observable_balance_live.py").read_text()
    assert source.index('os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG"') < source.index(
        "import torch"
    )
    for field in (
        "torch_version",
        "cuda_runtime_version",
        "cudnn_version",
        "gpu_name",
        "deterministic_algorithms",
        "float32_matmul_precision",
    ):
        assert f'"{field}"' in source


def test_incident_index_is_exact_membership_not_a_count(tmp_path: Path) -> None:
    incident_root = tmp_path / "incidents"
    incident_root.mkdir()
    index = {
        "schema_version": "pldr-observable-balance-incident-index-v1",
        "campaign_id": "campaign",
        "incidents": [],
    }
    (incident_root / "index.json").write_text(json.dumps(index))
    assert _verify_incidents(tmp_path, "campaign") == 0
    (incident_root / "orphan.txt").write_text("retained failure")
    with pytest.raises(ValueError, match="membership"):
        _verify_incidents(tmp_path, "campaign")


def test_external_paths_derive_only_from_explicit_binding_root(
    tmp_path: Path,
) -> None:
    binding_root = tmp_path.resolve()
    paths = default_paths(binding_root)
    assert all(path.is_relative_to(binding_root) for path in paths.values())
    assert paths["output"].name == "finite-increment-observable-balance"
    assert paths["input_bundle"].parent.name == "rev47"
    assert paths["tokens"].parent.name == "refinedweb-538m-prefix-locked"


def test_external_path_escape_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes binding root"):
        bound_path(tmp_path.parent / "outside", tmp_path, "probe")
