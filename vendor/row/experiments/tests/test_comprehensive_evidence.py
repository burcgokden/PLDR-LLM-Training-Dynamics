from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "experiments" / "analysis"
CONFIRM = ROOT / "experiments" / "confirm"
sys.path.insert(0, str(ANALYSIS))
sys.path.insert(0, str(CONFIRM))

from comprehensive_confirmation_analysis import (  # noqa: E402
    _q_record_replays_fixture,
    _verify_producer_record,
    analyze_campaign,
)
from comprehensive_evidence import (  # noqa: E402
    load_request,
    replay_comparison,
    seal_request,
)
from comprehensive_live_producers import qualify  # noqa: E402
from run_comprehensive_confirmation import (  # noqa: E402
    lock_predictions,
    seal_construction_request,
)
from confirmation_artifacts import (  # noqa: E402
    digest_object,
    validate_measurement_registry,
    write_json_atomic,
)


def _scientific_checkpoint(run_id, lineage):
    return {
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "opt": {"state": {0: {"exp_avg": torch.tensor([0.1, 0.2])}}},
        "scheduler": {"last_epoch": 7},
        "rng_states": {"torch_cpu": torch.tensor([1, 2], dtype=torch.uint8)},
        "optimizer_config": {"name": "adamw"},
        "model_config": {"width": 2},
        "operation_order": ["forward", "backward", "step"],
        "branch_state": {
            "clip_mask": {"weight": torch.tensor([False, True])},
            "decay_mask": {"weight": torch.tensor([True, True])},
            "device": "cuda:0",
            "dtype": "torch.float64",
            "scheduler_phase": "decay",
        },
        "measurement_registry": {"partition": "sealed"},
        "intervention_state": {"arm": "nominal"},
        "data_state": {
            "dataset_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "cursor_start": 10,
            "cursor_end": 14,
            "rows_per_step": 4,
            "context_length": 256,
            "probe_region": "global",
            "tokens_path": "/invocation/only/tokens.bin",
        },
        "step": 7,
        "run_identity": {
            "run_id": run_id,
            "lineage_root": lineage,
            "from_scratch": True,
        },
    }


def test_replay_uses_distinct_lineages_and_complete_scientific_state():
    primary = _scientific_checkpoint("primary", "lineage-primary")
    replay = copy.deepcopy(primary)
    replay["run_identity"] = {
        "run_id": "replay",
        "lineage_root": "lineage-replay",
        "from_scratch": True,
    }
    replay["branch_state"]["device"] = "cuda:1"
    replay["data_state"]["cursor_start"] = 999
    replay["data_state"]["tokens_path"] = "/another/invocation/path.bin"
    result = replay_comparison(primary, replay)
    assert result["identities_valid"]
    assert result["scientific_components_equal"]

    replay["model"]["weight"][0] = 3.0
    changed = replay_comparison(primary, replay)
    assert not changed["scientific_components_equal"]
    assert changed["different_components"] == ["model"]


def test_aliased_replay_identity_is_rejected():
    primary = _scientific_checkpoint("primary", "lineage-primary")
    alias = copy.deepcopy(primary)
    result = replay_comparison(primary, alias)
    assert not result["identities_valid"]
    assert result["scientific_components_equal"]


def test_request_digest_and_bound_artifact_are_both_enforced(tmp_path):
    protocol = tmp_path / "protocol.json"
    artifact = tmp_path / "raw.bin"
    protocol.write_text("{}\n", encoding="utf-8")
    artifact.write_bytes(b"measured")
    request = seal_request(
        "Q", "campaign", tmp_path, ["raw=raw.bin"], protocol)
    request_path = tmp_path / "request.json"
    write_json_atomic(request_path, request)
    assert load_request(request_path)[0]["stage"] == "Q"

    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="digest changed"):
        load_request(request_path)

    summary = {
        "schema_version": "pldr-construction-summary-v4",
        "code_sha256": "c" * 64,
        "partition_sha256": "d" * 64,
        "schedule": {"node_steps": [1, 2], "gains": [0.5],
                     "quadratic": [0.0], "forces": [0.0]},
        "selected_edges": {"pair_product_lower": 0.01},
        "intervals": {"normal_force_max": 0.1},
        "path_metrics": {"metrics": [[[1.0]], [[1.0]]]},
        "radii": {"values": [1.0, 1.0]},
        "direct_conversion": {
            "rule": "construction-locked-affine-normal-to-direct-v1",
            "coefficient": 2.0, "remainder": 0.1, "singular_max": 1.8,
        },
        "application": {"subdivisions": 2},
    }
    summary_path = tmp_path / "construction_summary.json"
    write_json_atomic(summary_path, summary)
    construction_request = tmp_path / "construction_request.json"
    seal_construction_request(SimpleNamespace(
        campaign_id="campaign", artifact_root=str(tmp_path),
        summary=summary_path.name, output=str(construction_request)))
    loaded, resolved, _protocol = load_request(construction_request)
    assert loaded["stage"] == "CONSTRUCTION"
    assert set(resolved) == {"construction_summary"}

    lock_path = tmp_path / "prediction_lock.json"
    lock_predictions(SimpleNamespace(
        campaign_id="campaign", artifact_root=str(tmp_path),
        request=[construction_request.name], output=str(lock_path)))
    import json
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["direct_conversion"] == [summary["direct_conversion"]]


def test_v4_registry_is_closed_and_content_addressed():
    registry = {
        "schema_version": "pldr-comprehensive-measurement-registry-v4",
        "context_length": 256,
        "construction": [{"id": "c0", "chunk_index": 1}],
        "validation": [{"id": "v0", "chunk_index": 2}],
        "application_pairs": [
            {"id": "a0", "left_chunk": 3, "right_chunk": 4}],
        "finite_stencil": {
            "step": 2 ** -10,
            "direction_algorithm": "numpy-pcg64-normalized-v1",
            "direction_seed": 340021,
            "one_direction_per_physical_row": True,
            "joint_cover_algorithm":
                "scaled-row-plus-direction-nearest-v1",
            "cover_directions_per_row": 4,
            "cover_seed": 340060,
        },
        "selected_edge_rule": {
            "kind": "target-plus-fixed-competitors",
            "competitor_ids": [3, 7, 11],
        },
        "challenge_seed": 340031,
        "dataset_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
    }
    registry["registry_sha256"] = digest_object(registry)
    assert validate_measurement_registry(registry, require_nonempty=True)
    registry["challenge_seed"] += 1
    with pytest.raises(ValueError, match="digest"):
        validate_measurement_registry(registry)


def test_qualification_record_rejects_resealed_unknown_fields(tmp_path):
    output = tmp_path / "qualification.json"
    qualify(SimpleNamespace(device="cpu", output=str(output)))
    import json
    record = json.loads(output.read_text(encoding="utf-8"))
    assert _verify_producer_record(record, "Q")
    assert _q_record_replays_fixture(record)

    forged = copy.deepcopy(record)
    forged["fisher_probe"][0] += 0.1
    unsigned = dict(forged)
    unsigned.pop("record_sha256")
    forged["record_sha256"] = digest_object(unsigned)
    assert _verify_producer_record(forged, "Q")
    assert not _q_record_replays_fixture(forged)

    record["caller_pass"] = True
    unsigned = dict(record)
    unsigned.pop("record_sha256")
    record["record_sha256"] = digest_object(unsigned)
    assert not _verify_producer_record(record, "Q")


def test_campaign_rejects_empty_or_typed_stage_records():
    with pytest.raises(ValueError, match="requires stage requests"):
        analyze_campaign([])
    with pytest.raises((ValueError, OSError, TypeError)):
        analyze_campaign([{"stage": "Q", "decision": "CONFIRMED"}])
