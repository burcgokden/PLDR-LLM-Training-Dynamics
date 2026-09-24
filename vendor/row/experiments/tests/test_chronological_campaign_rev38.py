"""Campaign-level regression and mutation tests for chronological confirmation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest


EXPERIMENTS = Path(__file__).resolve().parents[1]
ROOT = EXPERIMENTS.parent
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(EXPERIMENTS / "analysis"))

from analyze_chronological_intervention import (  # noqa: E402
    analyze_campaign,
    analyze_record,
)
from analyze_chronological_plga import (  # noqa: E402
    analyze_record as analyze_plga_record,
)
from confirm.chronological_collapse import (  # noqa: E402
    rectangular_score_transfer,
    source_removal_response,
)
from confirm.chronological_confirmation_specs import (  # noqa: E402
    CAMPAIGN_ID,
    REGISTRY,
    TRAJECTORIES,
)
from confirm.confirmation_artifacts import digest_object  # noqa: E402
from confirm.gate_shape import plga_secant_chain  # noqa: E402
from confirm.run_chronological_confirmation import (  # noqa: E402
    create_lock,
    create_plan,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _lock_payload() -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "pldr-chronological-construction-lock-v1",
        "campaign_id": CAMPAIGN_ID,
        "primary_block_length": 16,
        "history_length": 48,
        "segment_grid": [0.0, 0.5, 1.0],
        "shape_safety_factor": 2.0,
        "numerical_relative_coefficient": 0.0,
        "numerical_absolute_charge": 1e-13,
        "intervention_minimum_effect": 0.01,
        "plga_power_secant_bound": 2.0,
        "plga_transfer_coefficient_bound": 3.0,
        "centered_logit_response_coefficient": 4.0,
        "centered_logit_remainder_bound": 1e-6,
        "constant_enlargement_after_lock": False,
        "development_reports": [],
        "construction_reports": [],
        "plga_construction_report": {
            "path": "/registered/construction-plga.json",
            "sha256": "f" * 64,
        },
    }
    value["lock_sha256"] = digest_object(value)
    return value


def _write_lock(path: Path) -> dict[str, object]:
    value = _lock_payload()
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return value


def _intervention_payload(
    seed: int,
    offset: int,
    ordinal: int,
    lock: dict[str, object],
) -> dict[str, np.ndarray]:
    width = 64
    updates = REGISTRY["intervention_steps"]
    eta = 0.1
    contrast = np.zeros(width)
    contrast[0] = 1.0
    source_gate = np.ones(width)
    baseline_gate = np.ones(width)
    source_direction = np.zeros(width)
    source_direction[0] = 0.5
    prediction = float(source_removal_response(
        baseline_gate,
        contrast[None, :],
        source_direction,
        learning_rate=eta,
    )["predicted_energy_response"][0])
    energy = np.asarray([
        [1.0, 1.0, 0.99, 0.98, 0.97],
        [1.0, 1.0 + prediction, 1.08, 1.07, 1.06],
        [1.0, 0.98, 0.97, 0.96, 0.95],
        [1.0, 0.99, 0.98, 0.97, 0.96],
    ])
    batches = np.asarray([
        [_sha(f"{seed}-{ordinal}-{step}".encode()) for step in range(updates)]
        for _arm in range(4)
    ])
    anchor = REGISTRY["intervention_anchor_step"]
    future_start = ordinal * updates
    steps = np.repeat(np.arange(
        anchor + future_start + 1,
        anchor + future_start + updates + 1,
        dtype=np.int64,
    )[None, :], 4, axis=0)
    heldout_index = [
        int(row["seed"]) for row in TRAJECTORIES if row["role"] == "heldout"
    ].index(seed)
    source_model = "a" * 64
    source_optimizer = "b" * 64
    return {
        "schema_version": np.asarray(
            "pldr-chronological-intervention-ledger-v1"),
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "unit_id": np.asarray(f"cci-s{seed}-{ordinal:02d}"),
        "ordinal": np.asarray(ordinal, dtype=np.int64),
        "rng_substream": np.asarray(
            3_800_000 + heldout_index * 12 + ordinal, dtype=np.int64),
        "seed": np.asarray(seed, dtype=np.int64),
        "data_offset_chunks": np.asarray(offset, dtype=np.int64),
        "anchor": np.asarray(anchor, dtype=np.int64),
        "future_update_start": np.asarray(future_start, dtype=np.int64),
        "update_count": np.asarray(updates, dtype=np.int64),
        "layer": np.asarray(ordinal % 3, dtype=np.int64),
        "context_index": np.asarray(ordinal % 16, dtype=np.int64),
        "head": np.asarray((ordinal % 16) % 4, dtype=np.int64),
        "row_left": np.asarray(0, dtype=np.int64),
        "row_right": np.asarray(63, dtype=np.int64),
        "arm_names": np.asarray([
            "baseline",
            "adaptive-source-removal",
            "decay-removal",
            "shape-direction-removal",
        ]),
        "source_gate": source_gate,
        "source_contrast": contrast,
        "baseline_gate_successor": baseline_gate,
        "adaptive_source_direction": source_direction,
        "learning_rate": np.asarray(eta),
        "predicted_source_removal_response": np.asarray(prediction),
        "pair_energy_history": energy,
        "update_batch_sha256": batches,
        "optimizer_step_after": steps,
        "initial_model_sha256": np.asarray([source_model] * 4),
        "initial_optimizer_sha256": np.asarray([source_optimizer] * 4),
        "source_model_sha256": np.asarray(source_model),
        "source_optimizer_sha256": np.asarray(source_optimizer),
        "final_model_sha256": np.asarray(["c" * 64] * 4),
        "final_optimizer_sha256": np.asarray(["d" * 64] * 4),
        "checkpoint_sha256": np.asarray("e" * 64),
        "registry_sha256": np.asarray("f" * 64),
        "lock_sha256": np.asarray(lock["lock_sha256"]),
        "intervention_minimum_effect": np.asarray(
            lock["intervention_minimum_effect"]),
    }


def test_intervention_replay_rejects_arm_stream_mutation(tmp_path: Path) -> None:
    lock = _lock_payload()
    payload = _intervention_payload(9444, 24576, 0, lock)
    good = tmp_path / "good.npz"
    np.savez_compressed(good, **payload)
    report = analyze_record(good, lock)
    assert report["decision"] == "SITE_QUALIFIED"

    mutated = dict(payload)
    changed_batches = payload["update_batch_sha256"].copy()
    changed_batches[1, 0] = "0" * 64
    mutated["update_batch_sha256"] = changed_batches
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, **mutated)
    report = analyze_record(bad, lock)
    assert report["decision"] == "SITE_NOT_QUALIFIED"
    assert not report["checks"]["identical_update_minibatches"]


def test_complete_intervention_campaign_applies_ten_of_twelve(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "lock.json"
    lock = _write_lock(lock_path)
    paths = []
    for trajectory in TRAJECTORIES:
        if trajectory["role"] != "heldout":
            continue
        for ordinal in range(REGISTRY["intervention_sites_per_seed"]):
            path = tmp_path / f"{trajectory['seed']}-{ordinal}.npz"
            np.savez_compressed(path, **_intervention_payload(
                int(trajectory["seed"]),
                int(trajectory["data_offset_chunks"]),
                ordinal,
                lock,
            ))
            paths.append(str(path))
    report = analyze_campaign(paths, lock_path)
    assert report["decision"] == "QUALIFIED"
    assert report["qualified_sites_by_seed"] == {
        9444: 12, 10444: 12, 11444: 12, 12444: 12,
    }
    incomplete = analyze_campaign(paths[:-1], lock_path)
    assert incomplete["decision"] == "NOT_QUALIFIED"
    assert not incomplete["checks"]["complete_48_unit_registry"]


def _registry(tokens: Path, tokenizer: Path) -> dict[str, object]:
    construction = [
        {"id": f"ccc{index:03d}", "chunk_index": 1_000_000 + index}
        for index in range(8)
    ]
    validation = [
        {"id": f"cch{index:03d}", "chunk_index": 1_000_008 + index}
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
        "anchors": list(REGISTRY["anchor_steps"]),
        "context_head_block_rule":
            "head-equals-context-index-modulo-four-v1",
        "block_row_rule":
            "complete-ordered-generator-rows-0-through-63-v1",
        "history_length": 48,
        "intervention_units": units,
        "intervention_unit_rule":
            "twelve-fixed-four-update-sites-per-heldout-seed-v1",
        "dataset_sha256": _sha(tokens.read_bytes()),
        "tokenizer_sha256": _sha(tokenizer.read_bytes()),
    }
    value["registry_sha256"] = digest_object(value)
    return value


def test_execution_plan_contains_every_registered_stage(tmp_path: Path) -> None:
    tokens = tmp_path / "tokens.bin"
    tokenizer = tmp_path / "tokenizer.model"
    tokens.write_bytes(b"tokens")
    tokenizer.write_bytes(b"tokenizer")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(
        json.dumps(_registry(tokens, tokenizer), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = create_plan(
        tokens,
        tokenizer,
        registry_path,
        tmp_path / "runs",
        tmp_path / "evidence",
        tmp_path / "plan.json",
        devices=("cuda:0", "cuda:1"),
    )
    assert len(payload["jobs"]) == 261
    counts = {
        stage: sum(job["stage"] == stage for job in payload["jobs"])
        for stage in ("D", "C", "H", "I", "P")
    }
    assert counts == {"D": 28, "C": 61, "H": 64, "I": 48, "P": 60}
    heldout_pair = next(
        job for job in payload["jobs"]
        if job["stage"] == "H" and job["id"].startswith("capture-"))
    assert "--lock" in heldout_pair["command"]
    assert "{construction_primary_block}" in heldout_pair["command"]


def _write_pair_reports(
    root: Path, role: str, seed: int, offset: int,
) -> list[str]:
    anchors = (
        REGISTRY["development_anchor_steps"]
        if role == "development" else REGISTRY["anchor_steps"]
    )
    paths = []
    for layer in range(3):
        for anchor in anchors:
            for block in REGISTRY["construction_block_lengths"]:
                value = {
                    "schema_version": "pldr-chronological-analysis-v1",
                    "role": role,
                    "seed": seed,
                    "data_offset_chunks": offset,
                    "step_start": anchor,
                    "step_stop": anchor + block,
                    "layer": layer,
                    "decision": "QUALIFIED",
                }
                path = root / f"{role}-{layer}-{anchor}-{block}.json"
                path.write_text(json.dumps(value) + "\n", encoding="utf-8")
                paths.append(str(path))
    return paths


def test_construction_lock_binds_complete_pair_and_plga_reports(
    tmp_path: Path,
) -> None:
    development = _write_pair_reports(
        tmp_path, "development", 7444, 0)
    construction = _write_pair_reports(
        tmp_path, "construction", 8444, 8192)
    plga = tmp_path / "plga.json"
    plga.write_text(json.dumps({
        "schema_version": "pldr-chronological-plga-analysis-v1",
        "role": "construction",
        "decision": "CONSTRUCTION_SUMMARY",
        "record_count": 15,
        "checks": {
            "complete_construction_grid": True,
            "all_identities_valid": True,
        },
        "maxima": {
            "power_secant": 2.0,
            "transfer_coefficient": 3.0,
            "centered_response_ratio": 4.0,
        },
    }) + "\n", encoding="utf-8")
    output = tmp_path / "construction-lock.json"
    lock = create_lock(
        development,
        construction,
        output,
        plga_construction_path=plga,
        shape_safety_factor=2.0,
        numerical_relative_coefficient=1e-5,
        numerical_absolute_charge=1e-13,
        intervention_minimum_effect=1e-3,
        plga_power_secant_bound=2.0,
        plga_transfer_coefficient_bound=3.0,
        centered_logit_response_coefficient=4.0,
        centered_logit_remainder_bound=1e-6,
    )
    assert lock["primary_block_length"] == 16
    assert lock["plga_construction_report"]["sha256"] == _sha(
        plga.read_bytes())
    with pytest.raises(ValueError, match="do not cover"):
        create_lock(
            development,
            construction,
            tmp_path / "bad-lock.json",
            plga_construction_path=plga,
            shape_safety_factor=2.0,
            numerical_relative_coefficient=1e-5,
            numerical_absolute_charge=1e-13,
            intervention_minimum_effect=1e-3,
            plga_power_secant_bound=1.99,
            plga_transfer_coefficient_bound=3.0,
            centered_logit_response_coefficient=4.0,
            centered_logit_remainder_bound=1e-6,
        )


def _plga_payload() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3817)
    cells = 8
    width = 64
    generators = []
    collapsed = []
    weights = []
    biases = []
    exponents = []
    couplings = []
    coupling_biases = []
    curvature_reference = []
    curvature_candidate = []
    query_reference = []
    query_candidate = []
    key_reference = []
    key_candidate = []
    score_reference = []
    score_candidate = []
    logits_reference = []
    logits_candidate = []
    for index in range(cells):
        generator = np.eye(width)
        generator += index * 1e-3
        anchor = np.repeat(generator[0:1], width, axis=0)
        weight = 0.5 * np.eye(width)
        bias = np.ones((width, width))
        exponent = np.full((width, width), 0.5)
        coupling = 0.5 * np.eye(width)
        coupling_bias = np.zeros((width, width))
        replay = plga_secant_chain(
            anchor,
            generator,
            weight,
            bias,
            exponent,
            coupling,
            coupling_bias,
            positive_floor=1e-9,
        )
        q0 = rng.normal(scale=0.1, size=(3, width))
        q1 = q0 + rng.normal(scale=1e-2, size=q0.shape)
        k0 = rng.normal(scale=0.1, size=(4, width))
        k1 = k0 + rng.normal(scale=1e-2, size=k0.shape)
        score = rectangular_score_transfer(
            q0,
            q1,
            replay["curvature_left"],
            replay["curvature_right"],
            k0,
            k1,
        )
        reference_logits = np.asarray([4.0, 1.0, 0.0, -1.0])
        candidate_logits = reference_logits + np.asarray([
            -1e-5, 1e-5, 0.0, 0.0])
        generators.append(generator)
        collapsed.append(anchor)
        weights.append(weight)
        biases.append(bias)
        exponents.append(exponent)
        couplings.append(coupling)
        coupling_biases.append(coupling_bias)
        curvature_reference.append(replay["curvature_left"])
        curvature_candidate.append(replay["curvature_right"])
        query_reference.append(q0)
        query_candidate.append(q1)
        key_reference.append(k0)
        key_candidate.append(k1)
        score_reference.append(score["reference_score"])
        score_candidate.append(score["candidate_score"])
        logits_reference.append(reference_logits)
        logits_candidate.append(candidate_logits)
    return {
        "schema_version": np.asarray(
            "pldr-chronological-plga-ledger-v1"),
        "campaign_id": np.asarray(CAMPAIGN_ID),
        "role": np.asarray("construction"),
        "seed": np.asarray(8444, dtype=np.int64),
        "data_offset_chunks": np.asarray(8192, dtype=np.int64),
        "anchor": np.asarray(2000, dtype=np.int64),
        "layer": np.asarray(0, dtype=np.int64),
        "checkpoint_sha256": np.asarray("a" * 64),
        "registry_sha256": np.asarray("b" * 64),
        "lock_sha256": np.asarray(""),
        "positive_floor": np.asarray(1e-9),
        "plga_pair_id": np.asarray([
            f"ccc{index:03d}-l0-h{index % 4}-anchor0"
            for index in range(cells)
        ]),
        "plga_context_index": np.arange(cells, dtype=np.int64),
        "plga_head": np.arange(cells, dtype=np.int64) % 4,
        "plga_anchor_row": np.zeros(cells, dtype=np.int64),
        "plga_generator": np.asarray(generators),
        "plga_collapsed_anchor_generator": np.asarray(collapsed),
        "plga_weight": np.asarray(weights),
        "plga_bias": np.asarray(biases),
        "plga_exponent": np.asarray(exponents),
        "plga_coupling": np.asarray(couplings),
        "plga_coupling_bias": np.asarray(coupling_biases),
        "plga_curvature_left": np.asarray(curvature_reference),
        "plga_curvature_right": np.asarray(curvature_candidate),
        "plga_reference_query": np.asarray(query_reference),
        "plga_candidate_query": np.asarray(query_candidate),
        "plga_reference_key": np.asarray(key_reference),
        "plga_candidate_key": np.asarray(key_candidate),
        "plga_reference_pre_mask_score": np.asarray(score_reference),
        "plga_candidate_pre_mask_score": np.asarray(score_candidate),
        "plga_reference_centered_logits": np.asarray(logits_reference),
        "plga_candidate_centered_logits": np.asarray(logits_candidate),
        "plga_candidate_cache_replay_residual": np.zeros(cells),
    }


def test_plga_analyzer_replays_raw_tensors_and_rejects_cell_mutation(
    tmp_path: Path,
) -> None:
    payload = _plga_payload()
    path = tmp_path / "plga.npz"
    np.savez_compressed(path, **payload)
    report = analyze_plga_record(path, None)
    assert report["decision"] == "CONSTRUCTION_SUMMARY"
    assert report["cell_count"] == 8

    changed = dict(payload)
    pair_ids = payload["plga_pair_id"].copy()
    pair_ids[0] = "unregistered"
    changed["plga_pair_id"] = pair_ids
    bad = tmp_path / "plga-bad.npz"
    np.savez_compressed(bad, **changed)
    with pytest.raises(ValueError, match="dimensions"):
        analyze_plga_record(bad, None)
