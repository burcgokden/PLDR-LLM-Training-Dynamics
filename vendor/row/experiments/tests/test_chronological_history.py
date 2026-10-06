from __future__ import annotations
import copy

from pathlib import Path
import sys

import pytest
import torch
from torch import nn


EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))
from confirm.confirmation_artifacts import (  # noqa: E402
    digest_object,
    validate_measurement_registry,
)

from confirm.chronological_history import (  # noqa: E402
    build_sidecar,
    empty_history,
    final_gate_parameters,
    record_clipped_gate_gradients,
    restore_history,
    save_sidecar,
    sidecar_path,
)


class _RowUnit(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.layernormA = nn.LayerNorm(width)


class _Attention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.reslayerAs = nn.ModuleList([_RowUnit(width) for _ in range(8)])


class _Layer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.mha1 = _Attention(width)


class _Decoder(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.dec_layers = nn.ModuleList([_Layer(width), _Layer(width)])


class _Model(nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.decoder = _Decoder(width)


def _advance(
    optimizer: torch.optim.AdamW,
    gates: dict[str, nn.Parameter],
    history: dict[str, list[dict[str, object]]],
    step: int,
    value: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    for parameter in gates.values():
        parameter.grad = torch.full_like(parameter, value)
    record_clipped_gate_gradients(
        history,
        gates,
        optimizer,
        global_step=step,
        history_length=3,
    )
    optimizer.step()


def test_history_sidecar_is_chronological_and_checkpoint_bound(tmp_path) -> None:
    model = _Model()
    gates = final_gate_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, betas=(0.5, 0.75), eps=1e-6,
        weight_decay=0.1,
    )
    history = empty_history(gates)
    for step, value in enumerate((0.2, -0.4, 0.8), start=1):
        _advance(optimizer, gates, history, step, value)

    checkpoint = tmp_path / "ckpt_3.pt"
    torch.save({"step": 3}, checkpoint)
    payload = build_sidecar(
        checkpoint, 3, history, gates, optimizer, history_length=3)
    first = payload["gates"][next(iter(gates))]
    assert first["ready_for_next_update"] is True
    assert first["steps_newest_first"].tolist() == [3, 2, 1]
    assert torch.allclose(
        first["previous_gradients_for_next_newest_first"],
        torch.tensor([[0.8, 0.8, 0.8], [-0.4, -0.4, -0.4]]),
    )
    assert torch.allclose(
        first["first_moment_tail_for_next"],
        torch.full((3,), 0.1),
    )

    written = save_sidecar(
        checkpoint, 3, history, gates, optimizer, history_length=3)
    assert written == sidecar_path(checkpoint)
    restored = restore_history(
        checkpoint, gates, checkpoint_step=3, history_length=3)
    assert [row["global_step"] for row in restored[next(iter(gates))]] == [1, 2, 3]

    torch.save({"step": 3, "tampered": True}, checkpoint)
    with pytest.raises(ValueError, match="digest"):
        restore_history(
            checkpoint, gates, checkpoint_step=3, history_length=3)


def test_gate_registry_rejects_incomplete_architecture() -> None:
    model = _Model()
    del model.decoder.dec_layers[1].mha1.reslayerAs[7].layernormA
    with pytest.raises(ValueError, match="incomplete"):
        final_gate_parameters(model)


def _chronological_registry() -> dict[str, object]:
    construction = [
        {"id": f"ccc{index:03d}", "chunk_index": 1000 + index}
        for index in range(8)
    ]
    validation = [
        {"id": f"cch{index:03d}", "chunk_index": 1008 + index}
        for index in range(16)
    ]
    units = [
        {
            "id": f"cci-s{seed}-{ordinal:02d}",
            "seed": seed,
            "ordinal": ordinal,
            "future_update_start": ordinal * 4,
            "update_count": 4,
            "rng_substream": 3_800_000 + seed_index * 12 + ordinal,
        }
        for seed_index, seed in enumerate((9444, 10444, 11444, 12444))
        for ordinal in range(12)
    ]
    value: dict[str, object] = {
        "schema_version": "pldr-chronological-probe-registry-v1",
        "context_length": 256,
        "construction": construction,
        "validation": validation,
        "anchors": [2000, 4000, 8000, 16000, 24000],
        "context_head_block_rule":
            "head-equals-context-index-modulo-four-v1",
        "block_row_rule":
            "complete-ordered-generator-rows-0-through-63-v1",
        "history_length": 48,
        "intervention_units": units,
        "intervention_unit_rule":
            "twelve-fixed-four-update-sites-per-heldout-seed-v1",
        "dataset_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
    }
    value["registry_sha256"] = digest_object(value)
    return value


def test_chronological_registry_digest_and_units_are_strict() -> None:
    registry = _chronological_registry()
    assert validate_measurement_registry(registry, require_nonempty=True)

    digest_mutation = copy.deepcopy(registry)
    digest_mutation["anchors"] = [2000, 4000, 8000]
    with pytest.raises(ValueError, match="digest"):
        validate_measurement_registry(digest_mutation, require_nonempty=True)

    unit_mutation = copy.deepcopy(registry)
    unit_mutation["intervention_units"][0]["future_update_start"] = 1
    unsigned = dict(unit_mutation)
    unsigned.pop("registry_sha256")
    unit_mutation["registry_sha256"] = digest_object(unsigned)
    with pytest.raises(ValueError, match="unit"):
        validate_measurement_registry(unit_mutation, require_nonempty=True)
