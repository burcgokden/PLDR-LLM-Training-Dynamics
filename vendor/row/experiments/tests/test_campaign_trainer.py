"""Campaign-specific trainer clock, restart, and snapshot regressions."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest


torch = pytest.importorskip("torch")
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "confirm"))
sys.path.insert(0, str(EXP / "analysis"))
import measure_row_map as E0  # noqa: E402
import measure_optimizer_transport as E1  # noqa: E402
import run_confirmation_campaign as CAMPAIGN  # noqa: E402
from comprehensive_confirmation_analysis import (  # noqa: E402
    _recompute_snapshot_successor,
    _scheduler_replays_trainer_formula,
)


def _command(tokens, outdir, name, steps):
    return [
        sys.executable,
        "train_run.py",
        "--name", name,
        "--lr", "1e-3",
        "--warmup", "1",
        "--steps", str(steps),
        "--schedule_total_steps", "4",
        "--batch", "1",
        "--ctx", "16",
        "--layers", "1",
        "--heads", "2",
        "--dk", "4",
        "--adff", "8",
        "--seed", "19",
        "--device", "cpu",
        "--tokens", str(tokens),
        "--outdir", str(outdir),
        "--probe_every", "999",
        "--sharp_every", "999",
        "--sharp_pre_every", "999",
        "--sharp_block_every", "0",
        "--ckpt_steps", "",
        "--val_batches", "1",
        "--skip_generation",
    ]


def _run(command):
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        command,
        cwd=EXP,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-4000:]


def _records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_completion_fixture(path, *, checkpoint_step, rows):
    path.mkdir()
    torch.save({"step": checkpoint_step}, path / "ckpt_final.pt")
    with (path / "log.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_training_completion_seal_requires_consistent_terminal_step(tmp_path):
    runner = CAMPAIGN.CampaignRunner.__new__(CAMPAIGN.CampaignRunner)

    explicit = tmp_path / "explicit"
    _write_completion_fixture(
        explicit, checkpoint_step=4,
        rows=[{"step": 4, "loss": 1.0}, {"event": "final", "step": 4}],
    )
    assert runner._training_complete(explicit, 4)

    legacy = tmp_path / "legacy"
    _write_completion_fixture(
        legacy, checkpoint_step=4,
        rows=[{"step": 4, "loss": 1.0}, {"event": "final"}],
    )
    assert runner._training_complete(legacy, 4)

    wrong_final = tmp_path / "wrong-final"
    _write_completion_fixture(
        wrong_final, checkpoint_step=4,
        rows=[{"step": 4, "loss": 1.0}, {"event": "final", "step": 3}],
    )
    assert not runner._training_complete(wrong_final, 4)

    wrong_legacy = tmp_path / "wrong-legacy"
    _write_completion_fixture(
        wrong_legacy, checkpoint_step=4,
        rows=[{"step": 3, "loss": 1.0}, {"event": "final"}],
    )
    assert not runner._training_complete(wrong_legacy, 4)


def test_global_clock_restart_matches_monolithic_and_names_e1_groups(tmp_path):
    tokens = tmp_path / "tokens.npy"
    rng = np.random.default_rng(17)
    rng.integers(
        1, 32000, size=5140 * 16, dtype=np.uint16,
    ).tofile(tokens)
    order = tmp_path / "chunk-order.npy"
    np.save(order, np.arange(20, dtype=np.int64), allow_pickle=False)
    runs = tmp_path / "runs"

    first = _command(tokens, runs, "split-a", 2)
    first += [
        "--data_order", str(order),
        "--ckpt_steps", "1",
        "--optimizer_ckpt_steps", "2",
        "--confirmation_snapshot_steps", "2",
        "--confirmation_snapshot_blocks", "query,key,value,deductive",
    ]
    _run(first)
    source = runs / "split-a" / "ckpt_2.pt"
    source_payload = torch.load(source, map_location="cpu", weights_only=True)
    assert source_payload["step"] == 2
    assert source_payload["segment_local_step"] == 2
    assert source_payload["data_offset_end"] == 2
    assert "opt" in source_payload
    assert _scheduler_replays_trainer_formula(source_payload)

    second = _command(tokens, runs, "split-b", 2)
    second += ["--data_order", str(order), "--init_from", str(source)]
    _run(second)
    monolithic = _command(tokens, runs, "monolithic", 4)
    monolithic += ["--data_order", str(order), "--terminal_validation"]
    _run(monolithic)
    scaled = _command(tokens, runs, "scaled", 2)
    scaled += [
        "--data_order", str(order),
        "--init_from", str(source),
        "--learning_rate_multiplier", "0.5",
    ]
    _run(scaled)

    split_payload = torch.load(
        runs / "split-b" / "ckpt_final.pt",
        map_location="cpu",
        weights_only=True,
    )
    monolithic_payload = torch.load(
        runs / "monolithic" / "ckpt_final.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert split_payload["step"] == 4
    assert split_payload["segment_global_offset"] == 2
    assert split_payload["segment_local_step"] == 2
    assert split_payload["data_offset_end"] == 4
    assert _scheduler_replays_trainer_formula(split_payload)
    assert _scheduler_replays_trainer_formula(monolithic_payload)
    for name, value in monolithic_payload["model"].items():
        assert torch.equal(split_payload["model"][name], value), name

    first_rows = _records(runs / "split-a" / "log.jsonl")
    second_rows = _records(runs / "split-b" / "log.jsonl")
    mono_rows = _records(runs / "monolithic" / "log.jsonl")
    scaled_rows = _records(runs / "scaled" / "log.jsonl")
    split_steps = [row for row in first_rows + second_rows if "loss" in row]
    mono_steps = [row for row in mono_rows if "loss" in row]
    scaled_steps = [row for row in scaled_rows if "loss" in row]
    assert [row["step"] for row in split_steps] == [1, 2, 3, 4]
    assert [row["lr"] for row in split_steps] == pytest.approx(
        [row["lr"] for row in mono_steps], rel=0.0, abs=0.0,
    )
    assert [row["step"] for row in scaled_steps] == [3, 4]
    assert [row["lr"] for row in scaled_steps] == pytest.approx(
        [0.5 * row["lr"] for row in mono_steps[2:]], rel=0.0, abs=0.0,
    )
    assert second_rows[0]["global_step_offset_resolved"] == 2
    assert second_rows[0]["schedule_total_steps_resolved"] == 4
    terminal_validation = [
        row for row in mono_rows
        if row.get("event") == "terminal_validation"
    ]
    assert len(terminal_validation) == 1
    assert terminal_validation[0]["step"] == 4
    assert terminal_validation[0]["val_loss"] > 0.0
    assert second_rows[-1] == {
        "event": "final",
        "step": 4,
        "generation_skipped": True,
        "run_id": "split-b-seed19-start2",
        "lineage_root": "split-a-seed19-start0",
        "gpu_device_seconds": 0.0,
    }

    snapshot = torch.load(
        runs / "split-a" / "confirmation_transport_000000002.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert set(snapshot["blocks"]) == {
        "query", "key", "value", "deductive",
    }
    names = {
        group: [row["name"] for row in rows]
        for group, rows in snapshot["blocks"].items()
    }
    assert all(".wq." in name for name in names["query"])
    assert all(".wk." in name for name in names["key"])
    assert all(".wv." in name for name in names["value"])
    assert all(
        ".reslayerAs." in name or ".plgatt_layer." in name
        for name in names["deductive"]
    )
    assert all(
        row["state_step_after"] == 2
        for rows in snapshot["blocks"].values()
        for row in rows
    )
    assert all(
        torch.is_tensor(row["clip_derivative"])
        and torch.is_tensor(row["decay_mask"])
        for rows in snapshot["blocks"].values()
        for row in rows
    )
    successor_replay = _recompute_snapshot_successor(
        runs / "split-a" / "confirmation_transport_000000002.pt",
        "query", 0.7)
    assert successor_replay["error"] <= 1e-10
    assert successor_replay["decay_mask"] == 1.0

    e0 = E0.measure_checkpoint(
        run_directory=runs / "split-a",
        checkpoint=runs / "split-a" / "ckpt_1.pt",
        successor_checkpoint=source,
        token_path=tokens,
        device="cpu",
        batch_size=2,
        probe_offset=0,
        rows_per_layer=2,
        heldout_rows_per_layer=1,
        coordinate_padding=0.1,
        subdivisions_per_axis=1,
        maximum_boxes=1,
        resource_action_limit=1000,
        resource_storage_bytes_limit=1000000,
        output_rows=tmp_path / "e0_rows.npz",
    )
    measured = {row["name"]: row["value"] for row in e0["measurements"]}
    assert e0["schema_version"] == "pldr-row-map-direct-cover-v5"
    assert e0["layers"][0]["domain"]["kind"] == "convex_interval_hull"
    assert e0["layers"][0]["n_construction_rows"] == 2
    assert e0["layers"][0]["n_validation_rows"] == 1
    assert e0["layers"][0]["validation_domain_membership_indicator"] in {
        0.0, 1.0,
    }
    assert measured["direct_row_map_upper"] >= measured["grid_jacobian_upper"]
    assert measured["cover_identity_error"] == 0.0
    assert measured["heldout_jacobian_ratio"] <= 1.0
    assert measured["mixed_derivative_enclosure_ratio"] <= 1.0
    assert measured["third_derivative_enclosure_ratio"] <= 1.0

    for tensor_group in ("query", "deductive"):
        e1 = E1.measure_live_snapshot(
            snapshot_path=(
                runs / "split-a"
                / "confirmation_transport_000000002.pt"
            ),
            model_checkpoint=source,
            run_directory=runs / "split-a",
            row_artifact=tmp_path / "e0_rows.npz",
            tensor_group=tensor_group,
            device="cpu",
            certificate_output=tmp_path / f"e1_{tensor_group}_certificate.json",
        )
        e1_measured = {
            row["name"]: row["value"] for row in e1["measurements"]
        }
        assert e1["tensor_group"] == tensor_group
        assert e1_measured["adamw_ledger_residual"] <= 1e-10
        assert e1_measured["lifted_recurrence_residual"] <= 1e-10
        assert e1_measured["defect_recomposition_residual"] <= 1e-10
        assert e1_measured["coordinate_motion_enclosure_ratio"] <= 1.0
        assert e1_measured["taylor_remainder_enclosure_ratio"] <= 1.0
        assert e1_measured["intervention_boundary_residual"] <= 1e-10
        assert e1["maximum_raw_transport_replay_error"] >= 0.0

    retained = tmp_path / "retained-ckpt-2.pt"
    shutil.copy2(source, retained)
    retention_report = tmp_path / "retention-report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(EXP.parent / "scripts" / "retain_model_only_checkpoint.py"),
            "--checkpoint", str(retained),
            "--report", str(retention_report),
        ],
        cwd=EXP.parent,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    retained_payload = torch.load(
        retained, map_location="cpu", weights_only=True)
    assert retained_payload["schema_version"] == (
        "pldr-model-only-training-checkpoint-v1")
    assert "opt" not in retained_payload
    record = json.loads(retention_report.read_text())
    assert record["all_optimizer_states_removed"] is True
    assert record["records"][0]["bytes_after"] < (
        record["records"][0]["bytes_before"])

    rejected = _command(tokens, runs, "model-only-no-reset", 1)
    rejected += [
        "--data_order", str(order),
        "--init_from", str(retained),
    ]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    result = subprocess.run(
        rejected,
        cwd=EXP,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode != 0
    assert "model-only checkpoint initialization requires --reset_opt" in (
        result.stderr + result.stdout)

    allowed = _command(tokens, runs, "model-only-with-reset", 1)
    allowed += [
        "--data_order", str(order),
        "--init_from", str(retained),
        "--reset_opt",
    ]
    _run(allowed)
