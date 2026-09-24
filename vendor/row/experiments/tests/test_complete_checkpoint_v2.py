from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


CONFIRM = Path(__file__).resolve().parents[1] / "confirm"
sys.path.insert(0, str(CONFIRM))

from confirmation_artifacts import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    TRAIN_UPDATE_ORDER,
    build_checkpoint_set_manifest,
    build_state_manifest,
    capture_rng_states,
    empty_measurement_registry,
    load_checkpoint_set,
    load_complete_checkpoint,
    scientific_record_digest,
    validate_complete_checkpoint,
    write_json_atomic,
)


def checkpoint(step=3):
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model": {"weight": torch.tensor([1.0, 2.0])},
        "opt": {
            "state": {
                0: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.tensor([0.1, 0.2]),
                    "exp_avg_sq": torch.tensor([0.01, 0.04]),
                },
            },
            "param_groups": [{
                "lr": 0.01, "betas": (0.9, 0.95), "params": [0],
            }],
        },
        "scheduler": {
            "last_epoch": step,
            "_last_lr": [0.01],
        },
        "rng_states": capture_rng_states(),
        "data_state": {
            "dataset_sha256": "a" * 64,
            "tokenizer_sha256": "b" * 64,
            "cursor_end": 12,
        },
        "optimizer_config": {
            "name": "adamw", "groups": [{"learning_rate": 0.01}],
        },
        "model_config": {"width": 2, "layers": 1},
        "operation_order": list(TRAIN_UPDATE_ORDER),
        "intervention_state": {"arm": "nominal"},
        "code_manifest": {"trainer": "c" * 64},
        "branch_state": {
            "schema_version": "pldr-realized-branch-state-v3",
            "clip_mask": {
                "weight": torch.zeros(2, dtype=torch.bool),
            },
            "decay_mask": {
                "weight": torch.ones(2, dtype=torch.bool),
            },
            "device": "cpu",
            "dtype": "torch.float32",
            "scheduler_phase": "constant",
        },
        "measurement_registry": empty_measurement_registry(
            dataset_sha256="a" * 64,
            tokenizer_sha256="b" * 64,
            context_length=8,
        ),
        "state_manifest": {},
        "config": {"seed": 7},
        "data_offset": 0,
        "data_offset_end": 12,
        "segment_global_offset": 0,
        "segment_local_step": step,
        "schedule_total_steps": 8,
    }
    payload["state_manifest"] = build_state_manifest(payload)
    return payload


def test_complete_checkpoint_and_set_replay(tmp_path):
    first = checkpoint(3)
    second = checkpoint(4)
    second["data_state"]["cursor_end"] = 16
    second["data_offset_end"] = 16
    second["model"]["weight"] = torch.tensor([0.9, 1.9])
    second["state_manifest"] = build_state_manifest(second)
    first_path = tmp_path / "first.pt"
    second_path = tmp_path / "second.pt"
    torch.save(first, first_path)
    torch.save(second, second_path)

    assert load_complete_checkpoint(first_path)["step"] == 3
    manifest = build_checkpoint_set_manifest(
        tmp_path, ["first.pt", "second.pt"])
    write_json_atomic(tmp_path / "states.json", manifest)
    loaded = load_checkpoint_set(tmp_path / "states.json", tmp_path)
    assert [value["step"] for value in loaded] == [3, 4]


def test_checkpoint_tampering_and_missing_state_are_rejected():
    value = checkpoint()
    value["model"]["weight"][0] = 9.0
    with pytest.raises(ValueError, match="manifest"):
        validate_complete_checkpoint(value)
    value = checkpoint()
    del value["scheduler"]
    with pytest.raises(ValueError, match="missing"):
        validate_complete_checkpoint(value)
    value = checkpoint()
    value["unregistered_scientific_state"] = torch.tensor([1.0])
    with pytest.raises(ValueError, match="unknown"):
        validate_complete_checkpoint(value)



def test_wall_clock_is_outside_scientific_digest():
    record = {
        "schema_version": "test",
        "payload": [1, 2, 3],
        "resource_metadata": {"wall_clock_seconds": 1},
    }
    first = scientific_record_digest(record)
    record["resource_metadata"]["wall_clock_seconds"] = 999
    assert scientific_record_digest(record) == first
