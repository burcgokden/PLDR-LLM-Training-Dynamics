#!/usr/bin/env python3
"""Run algebra fixtures and one native update for campaign qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.comprehensive_campaign import (  # noqa: E402
    campaign_spec,
    validate_checkpoint_campaign_binding,
)
from confirm.comprehensive_gate_shape import (  # noqa: E402
    diameter_slack_summary,
    exact_pair_decomposition,
    finite_registry_geometry,
    two_arm_shape_response,
)
from confirm.confirmation_artifacts import (  # noqa: E402
    load_json_object,
    sha256_path,
    validate_complete_checkpoint,
    validate_measurement_registry,
)
from confirm.gate_shape_evidence import write_json_atomic  # noqa: E402
from confirm.row_map_live import _gate_parameter_name, _model_from_raw  # noqa: E402


REPORT_SCHEMA = "pldr-comprehensive-qualification-report-v1"


def _algebra_checks() -> dict[str, bool]:
    generator = np.random.Generator(np.random.PCG64(390039))
    rows0 = generator.normal(size=(19, 7))
    rows1 = rows0 + 0.03 * generator.normal(size=rows0.shape)
    gate0 = generator.normal(size=7)
    multiplier = generator.uniform(0.96, 0.999, size=7)
    innovation = 0.02 * generator.normal(size=7)
    gate1 = multiplier * gate0 - innovation
    geometry = finite_registry_geometry(gate0, rows0, pair_chunk_size=11)
    slack = diameter_slack_summary(
        gate0, rows0, gate1, rows1, pair_chunk_size=11)
    left, right = slack["target_maximizer"]
    decomposition = exact_pair_decomposition(
        gate0,
        (rows0[left] - rows0[right])[None, :],
        gate1,
        (rows1[left] - rows1[right])[None, :],
        decayed_gate=multiplier * gate0,
        innovation=innovation,
    )
    response = two_arm_shape_response(
        (rows0[0] - rows0[1])[None, :],
        gate1,
        (rows1[0] - rows1[1])[None, :],
        gate1 + 0.01,
        (rows1[2] - rows1[3])[None, :],
    )
    tolerance = float(campaign_spec()["decisions"]["identity_relative_tolerance"])
    return {
        "carrier_sandwich": bool(geometry["sandwich_valid"]),
        "successor_maximizer_slack": bool(
            slack["slack_identity_relative_residual"] <= tolerance),
        "gate_then_shape_endpoint": bool(
            np.max(np.abs(decomposition["decomposition_residual"]), initial=0.0)
            <= tolerance),
        "adamw_gate_balance": bool(
            np.max(np.abs(decomposition["gate_balance_residual"]), initial=0.0)
            <= tolerance),
        "two_arm_endpoint": bool(
            np.max(np.abs(response["identity_residual"]), initial=0.0)
            <= tolerance),
    }


def _resource_summary(path: Path) -> dict[str, Any]:
    summaries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if value.get("event") == "resource_summary":
                summaries.append(value)
    if len(summaries) != 1:
        raise ValueError("qualification training log needs one resource summary")
    return summaries[0]


def qualify(arguments: argparse.Namespace) -> dict[str, Any]:
    spec = campaign_spec()
    registry_path = Path(arguments.registry).resolve()
    campaign_path = Path(arguments.campaign_spec).resolve()
    order_path = Path(arguments.data_order).resolve()
    tokens = Path(arguments.tokens).resolve()
    tokenizer = Path(arguments.tokenizer).resolve()
    registry = load_json_object(registry_path)
    validate_measurement_registry(registry, require_nonempty=True)
    with campaign_path.open("r", encoding="utf-8") as stream:
        if json.load(stream) != spec:
            raise ValueError("qualification campaign specification is stale")
    if (
        registry["dataset_sha256"] != spec["data"]["dataset_sha256"]
        or registry["tokenizer_sha256"] != spec["data"]["tokenizer_sha256"]
        or sha256_path(tokens) != spec["data"]["dataset_sha256"]
        or sha256_path(tokenizer) != spec["data"]["tokenizer_sha256"]
    ):
        raise ValueError("qualification inputs disagree with the campaign")

    with tempfile.TemporaryDirectory(prefix="pldr-cgs-q0-", dir="/tmp") as temp:
        temporary = Path(temp)
        command = [
            sys.executable,
            str(EXPERIMENTS / "train_run.py"),
            "--name", "cgs-q0-live",
            "--run-id", "cgs-q0-live",
            "--lineage-root", "cgs-q0-live",
            "--lr", "0.00075",
            "--warmup", "250",
            "--const_lr",
            "--steps", "1",
            "--batch", "32",
            "--ctx", "256",
            "--layers", "3",
            "--heads", "4",
            "--dk", "64",
            "--adff", "170",
            "--seed", "7444",
            "--device", arguments.device,
            "--tokens", str(tokens),
            "--tok_model", str(tokenizer),
            "--outdir", str(temporary),
            "--data_offset", "0",
            "--data_order", str(order_path),
            "--probe_region", "global",
            "--confirmation_registry", str(registry_path),
            "--campaign_spec", str(campaign_path),
            "--optimizer", "adamw",
            "--wd", "0.1",
            "--clip", "1.0",
            "--ckpt_steps", "1",
            "--optimizer_ckpt_steps", "1",
            "--probe_every", "1000000",
            "--sharp_every", "1000000",
            "--sharp_pre_every", "1000000",
            "--skip_generation",
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            stdout_tail = "\n".join(completed.stdout.splitlines()[-20:])
            stderr_tail = "\n".join(completed.stderr.splitlines()[-20:])
            raise RuntimeError(
                "qualification training failed with exit status "
                f"{completed.returncode}; stdout tail:\n{stdout_tail}\n"
                f"stderr tail:\n{stderr_tail}"
            )
        run = temporary / "cgs-q0-live"
        checkpoint_path = run / "ckpt_1.pt"
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False)
        validate_complete_checkpoint(checkpoint)
        validate_checkpoint_campaign_binding(checkpoint)
        model = _model_from_raw(checkpoint, torch.device("cpu"))
        named = dict(model.named_parameters())
        gate_shapes = [
            tuple(named[_gate_parameter_name(layer)].shape)
            for layer in spec["trajectories"]["layer_indices"]
        ]
        optimizer_steps = []
        for state in checkpoint["opt"]["state"].values():
            if "step" in state:
                value = state["step"]
                optimizer_steps.append(
                    int(value.item()) if torch.is_tensor(value) else int(value))
        resource = _resource_summary(run / "log.jsonl")
        checkpoint_sha256 = sha256_path(checkpoint_path)

    checks = {
        **_algebra_checks(),
        "complete_checkpoint": int(checkpoint["step"]) == 1,
        "registered_architecture": (
            sum(parameter.numel() for parameter in model.parameters())
            == int(spec["model"]["parameter_count"])
            and gate_shapes == [(64,), (64,), (64,)]),
        "native_float32": (
            checkpoint["branch_state"]["dtype"] == "torch.float32"
            and all(
                not tensor.is_floating_point() or tensor.dtype == torch.float32
                for tensor in checkpoint["model"].values()
            )),
        "optimizer_advanced_once": bool(
            optimizer_steps and set(optimizer_steps) == {1}),
        "registry_binding": checkpoint["measurement_registry"] == registry,
        "order_binding": (
            checkpoint["data_state"]["data_order_sha256"]
            == sha256_path(order_path)),
        "data_cursor": checkpoint["data_state"]["cursor_end"] == 32,
    }
    return {
        "schema_version": REPORT_SCHEMA,
        "campaign_id": spec["campaign_id"],
        "campaign_spec_sha256": spec["spec_sha256"],
        "checks": checks,
        "decision": "QUALIFIED" if all(checks.values()) else "NOT_QUALIFIED",
        "checkpoint_sha256": checkpoint_sha256,
        "registry_sha256": registry["registry_sha256"],
        "data_order_sha256": sha256_path(order_path),
        "resource": resource,
        "native_stdout_tail": completed.stdout.splitlines()[-4:],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--campaign-spec", required=True)
    parser.add_argument("--data-order", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-pass", action="store_true")
    arguments = parser.parse_args()
    report = qualify(arguments)
    write_json_atomic(arguments.output, report)
    print(json.dumps({
        "decision": report["decision"],
        "output": str(Path(arguments.output).resolve()),
    }, sort_keys=True))
    if arguments.require_pass and report["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
