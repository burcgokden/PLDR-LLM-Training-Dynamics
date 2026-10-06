from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn


EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_chronological_confirmation import analyze_record  # noqa: E402
from confirm.capture_chronological_confirmation import (  # noqa: E402
    optimizer_displacements,
)
from run_chronological_qualification import (  # noqa: E402
    build_qualification_ledger,
)


def test_pre_step_adamw_displacement_matches_native_optimizer() -> None:
    torch.manual_seed(3811)
    model = nn.Sequential(nn.Linear(5, 4), nn.SiLU(), nn.Linear(4, 3))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=7.5e-4, betas=(0.9, 0.95),
        eps=1e-5, weight_decay=0.1,
    )
    for _step in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = sum((parameter * torch.randn_like(parameter)).sum()
                   for parameter in model.parameters())
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
        names, values, directions, successors = optimizer_displacements(
            model, optimizer)
        predicted = dict(zip(names, successors, strict=True))
        for value, direction, successor in zip(
            values, directions, successors, strict=True
        ):
            assert torch.equal(direction, successor - value.detach())
        optimizer.step()
        for name, parameter in model.named_parameters():
            assert torch.equal(parameter, predicted[name]), name


def test_analyzer_accepts_compact_per_row_numerical_charges(tmp_path) -> None:
    original = tmp_path / "original.npz"
    compact = tmp_path / "compact.npz"
    build_qualification_ledger(original)
    with np.load(original, allow_pickle=False) as record:
        payload = {name: record[name] for name in record.files}
    transitions = payload["normalized_rows"].shape[0] - 1
    vertices = payload["normalized_rows"].shape[1]
    payload["numerical_charge"] = np.full(
        (transitions, vertices), 5e-14)
    np.savez(compact, **payload)
    report = analyze_record(compact, pair_chunk_size=7)
    assert report["decision"] == "QUALIFIED"
    assert report["vertex_count"] == vertices


def test_optimizer_displacement_rejects_nonstandard_branch() -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, amsgrad=True)
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    with pytest.raises(ValueError, match="standard AdamW"):
        optimizer_displacements(model, optimizer)
