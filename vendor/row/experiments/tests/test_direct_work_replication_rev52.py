import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "scripts"))

from analysis.analyze_direct_work_precision import (  # noqa: E402
    magnitude_work_bound,
)
from analysis.analyze_direct_work_replication import arm_metrics  # noqa: E402
from confirm import direct_work_live  # noqa: E402
from confirm import resource_executor  # noqa: E402
from confirm.direct_work_replication_specs import (  # noqa: E402
    CAMPAIGN_ID,
    RECORD_SCHEMA,
    campaign_design,
    validate_design,
)
import execute_direct_work_replication as executor  # noqa: E402


MAP_SHAPE = (24, 4, 64, 64)


def test_replication_design_uses_native_four_source_observer():
    design = campaign_design()
    validate_design(design)
    assert design["campaign_id"] == CAMPAIGN_ID
    assert design["release_id"] == "rev52"
    assert design["registered_maps_per_unit"] == 96
    assert design["observer_policy"]["native_causal_eligibility"]
    assert design["observer_policy"]["sources"][-1] == "implementation-defect"
    assert design["temporal_policy"][
        "terminal_node_is_not_a_historical_training_successor"
    ]


def test_frozen_reserve_batch_is_digest_bound_and_unique(tmp_path):
    tokens = tmp_path / "tokens.bin"
    np.arange(10, dtype=np.uint16).tofile(tokens)
    dataset_digest = direct_work_live.sha256_path(tokens)
    checkpoint = {
        "config": {"ctx": 2, "batch": 2},
        "data_state": {"dataset_sha256": dataset_digest},
    }
    unsigned = {
        "schema_version": "pldr-reserve-update-batch-v1",
        "context_length": 2,
        "batch_size": 2,
        "dataset_sha256": dataset_digest,
        "chunk_indices": [1, 3],
    }
    batch = {**unsigned, "content_sha256": direct_work_live.digest_object(unsigned)}
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    rows, indices, digest = direct_work_live._frozen_update_batch(
        batch_path, checkpoint, tokens, torch.device("cpu")
    )
    assert indices == [1, 3]
    assert digest == direct_work_live.sha256_path(batch_path)
    torch.testing.assert_close(rows, torch.tensor([[2, 3], [6, 7]]))

    malformed = dict(batch)
    malformed["chunk_indices"] = [1, 1]
    malformed_path = tmp_path / "malformed.json"
    malformed_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="does not replay"):
        direct_work_live._frozen_update_batch(
            malformed_path, checkpoint, tokens, torch.device("cpu")
        )


def test_nvidia_smi_observation_retains_query_and_parse(monkeypatch):
    outputs = iter((
        "0, GPU-A\n1, GPU-B\n",
        "GPU-B, 11, 123\nGPU-A, 19, 77\n",
    ))

    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 0, next(outputs), "")

    monkeypatch.setattr(resource_executor.shutil, "which", lambda _name: "/nvidia-smi")
    monkeypatch.setattr(resource_executor.subprocess, "run", fake_run)
    allocated, reserved, observation = (
        resource_executor._nvidia_smi_process_bytes("cuda:1", {11, 12})
    )
    assert allocated == reserved == 123 * 1024**2
    assert observation["target_device_indices"] == [1]
    assert observation["target_gpu_uuids"] == ["GPU-B"]
    assert observation["monitored_process_ids"] == [11, 12]
    assert {11, 12}.issubset(observation["namespace_pid_candidates"])
    assert observation["process_query_status"] == "matched-monitored-process"
    assert observation["compute_apps_matching_process_row_count"] == 1
    assert observation["parsed_process_memory_mib"] == 123
    assert "GPU-B, 11, 123" in observation["compute_apps_query_stdout"]


def _producer_output(path: Path, allocated: int, reserved: int) -> None:
    np.savez(
        path,
        peak_gpu_allocated_bytes=np.asarray(allocated, dtype=np.int64),
        peak_gpu_reserved_bytes=np.asarray(reserved, dtype=np.int64),
    )


def _resource_record(monitor_reserved: int) -> dict:
    return {
        "schema_version": "test-resource-v1",
        "device": "cuda:0",
        "wall_seconds": 2.0,
        "peak_gpu_allocated_bytes": monitor_reserved,
        "peak_gpu_reserved_bytes": monitor_reserved,
        "gpu_probe_observation": {
            "namespace_visible": True,
            "process_query_status": "no-row-matching-monitored-process",
        },
        "technical_valid": True,
        "cap_status": "within_caps",
    }


def test_zero_monitor_with_nonzero_producer_is_explicitly_unavailable(tmp_path):
    output = tmp_path / "record.npz"
    _producer_output(output, 512, 1024)
    node = {
        "node_id": "science",
        "device": "cuda:0",
        "role": "construction",
        "phase": "construction",
        "expected_outputs": [str(output)],
    }
    target = tmp_path / "resource.json"
    modules = {
        "budget": {"process_reserved_memory_cap_bytes": 4096},
        "write_json_atomic": resource_executor.write_json_atomic,
    }
    record = executor.rewrite_resource(
        _resource_record(0), node, target, modules
    )
    assert record["technical_valid"]
    assert record["gpu_monitor_status"] == "unavailable"
    assert record["gpu_memory_authority"] == "producer"
    assert record["producer_peak_gpu_reserved_bytes"] == 1024
    assert record["gpu_monitor_incident"]["classification"] == (
        "no-row-matching-monitored-process-with-nonzero-producer"
    )


def test_contradictory_monitor_and_producer_fail_closed(tmp_path):
    output = tmp_path / "record.npz"
    _producer_output(output, 0, 0)
    node = {
        "node_id": "science",
        "device": "cuda:0",
        "role": "construction",
        "phase": "construction",
        "expected_outputs": [str(output)],
    }
    modules = {
        "budget": {"process_reserved_memory_cap_bytes": 4096},
        "write_json_atomic": resource_executor.write_json_atomic,
    }
    record = executor.rewrite_resource(
        _resource_record(1024), node, tmp_path / "resource.json", modules
    )
    assert not record["technical_valid"]
    assert record["gpu_monitor_status"] == "inconsistent"


def synthetic_arm_record() -> tuple[dict[str, np.ndarray], np.ndarray]:
    generator = np.random.default_rng(52)
    source = generator.normal(scale=1.0e-4, size=MAP_SHAPE)
    shape = generator.normal(scale=1.0e-6, size=MAP_SHAPE)
    zero = np.zeros(MAP_SHAPE, dtype=np.float64)
    endpoint = source + shape
    energy_change = 2.0 * np.sum(source * shape, axis=(-2, -1))
    energy_change += np.sum(shape * shape, axis=(-2, -1))
    record = {
        "natural_endpoint_z": endpoint,
        "natural_increment": shape,
        "natural_shape_secant": shape,
        "natural_gate_secant": zero,
        "natural_interaction_secant": zero,
        "natural_source_factorization_residual": zero,
        "natural_endpoint_factorization_residual": zero,
        "natural_gate_shape_residual": zero.copy(),
        "natural_work_charge_residual": np.zeros((24, 4)),
        "natural_energy_change": energy_change,
    }
    return record, source


def test_four_source_mutation_is_local_and_fail_closed():
    record, source = synthetic_arm_record()
    result = arm_metrics(record, "natural", source)
    assert np.all(result["technical_mask"])
    record["natural_gate_shape_residual"][0, 0, 0, 0] = 1.0e-4
    mutated = arm_metrics(record, "natural", source)
    assert not mutated["defect_pass"][0]
    assert np.count_nonzero(~mutated["technical_mask"]) == 1


def test_work_bound_has_no_unit_scale_floor():
    source = np.full(MAP_SHAPE, 1.0e-20)
    endpoint = np.full(MAP_SHAPE, 2.0e-20)
    bound = magnitude_work_bound(source, endpoint)
    assert np.all(bound > 0.0)
    assert float(np.max(bound)) < 1.0e-30
    assert RECORD_SCHEMA == "pldr-direct-work-replication-record-v1"
