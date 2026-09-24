#!/usr/bin/env python3
"""Generate the deterministic chronological-collapse campaign protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from confirm.chronological_confirmation_specs import (  # noqa: E402
    ARCHITECTURE,
    CAMPAIGN_ID,
    DECISION_RULES,
    NUMERICAL_POLICIES,
    OPTIMIZER,
    REGISTRY,
    RESOURCE_BUDGET,
    SCHEMA_VERSION,
    STAGES,
    TRAJECTORIES,
)


OUTPUT = ROOT / "experiments" / "protocols" / "chronological_confirmation"
LEDGER_SCHEMA = "pldr-chronological-pair-ledger-v1"
PLGA_LEDGER_SCHEMA = "pldr-chronological-plga-ledger-v1"
INTERVENTION_LEDGER_SCHEMA = "pldr-chronological-intervention-ledger-v1"


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_steps(trajectory: dict[str, object]) -> list[int]:
    anchors = (
        REGISTRY["development_anchor_steps"]
        if trajectory["role"] == "development"
        else REGISTRY["anchor_steps"]
    )
    return [*anchors, int(trajectory["updates"])]


def _training_command(trajectory: dict[str, object]) -> list[str]:
    checkpoints = ",".join(map(str, _checkpoint_steps(trajectory)))
    return [
        "python3",
        "experiments/train_run.py",
        "--name",
        str(trajectory["name"]),
        "--run-id",
        str(trajectory["name"]),
        "--lr",
        str(OPTIMIZER["learning_rate"]),
        "--warmup",
        str(OPTIMIZER["warmup_updates"]),
        "--const_lr",
        "--steps",
        str(trajectory["updates"]),
        "--batch",
        str(OPTIMIZER["batch_size"]),
        "--ctx",
        str(OPTIMIZER["context_length"]),
        "--layers",
        str(ARCHITECTURE["layers"]),
        "--heads",
        str(ARCHITECTURE["heads"]),
        "--dk",
        str(ARCHITECTURE["head_width"]),
        "--adff",
        "170",
        "--seed",
        str(trajectory["seed"]),
        "--device",
        "{device}",
        "--tokens",
        "{tokens}",
        "--tok_model",
        "{tokenizer}",
        "--outdir",
        "{run_root}",
        "--data_offset",
        str(trajectory["data_offset_chunks"]),
        "--confirmation_registry",
        "{registry}",
        "--probe_region",
        "global",
        "--optimizer",
        "adamw",
        "--wd",
        str(OPTIMIZER["weight_decay"]),
        "--clip",
        str(OPTIMIZER["gradient_clip_value"]),
        "--chronological_history_length",
        str(REGISTRY["history_length"]),
        "--ckpt_steps",
        checkpoints,
        "--optimizer_ckpt_steps",
        checkpoints,
        "--skip_generation",
        "--terminal_validation",
    ]


def _trajectory_plan(trajectory: dict[str, object]) -> dict[str, object]:
    anchors = (
        REGISTRY["development_anchor_steps"]
        if trajectory["role"] == "development"
        else REGISTRY["anchor_steps"]
    )
    block_lengths = (
        REGISTRY["construction_block_lengths"]
        if trajectory["role"] in {"development", "construction"}
        else ("construction_locked_primary",)
    )
    return {
        **trajectory,
        "training_command": _training_command(trajectory),
        "checkpoint_steps": _checkpoint_steps(trajectory),
        "capture_cells": [
            {
                "layer": layer,
                "anchor": anchor,
                "block_length": block,
            }
            for layer in range(ARCHITECTURE["layers"])
            for anchor in anchors
            for block in block_lengths
        ],
        "plga_cells": [
            {"layer": layer, "anchor": anchor}
            for layer in range(ARCHITECTURE["layers"])
            for anchor in REGISTRY["anchor_steps"]
        ] if trajectory["role"] in {"construction", "heldout"} else [],
        "intervention_units": [
            f"cci-s{trajectory['seed']}-{ordinal:02d}"
            for ordinal in range(REGISTRY["intervention_sites_per_seed"])
        ] if trajectory["role"] == "heldout" else [],
        "history_sidecar_required": True,
        "distinct_data_order_required": True,
    }


def expected_files() -> dict[Path, bytes]:
    stage_order = [stage["id"] for stage in STAGES]
    files: dict[Path, bytes] = {}
    files[Path("campaign_design.json")] = _json({
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "status": "prospective-registered",
        "architecture": ARCHITECTURE,
        "optimizer": OPTIMIZER,
        "trajectories": list(TRAJECTORIES),
        "registry": REGISTRY,
        "numerical_policies": NUMERICAL_POLICIES,
        "decision_rules": DECISION_RULES,
        "resource_budget": RESOURCE_BUDGET,
        "stage_order": stage_order,
        "stages": list(STAGES),
        "ownership": {
            "development": "mechanism discovery only",
            "construction": (
                "locks block selection and every numerical, shape, "
                "intervention, and PLGA constant"
            ),
            "heldout": (
                "evaluates the immutable construction lock on every "
                "registered layer-anchor cell"
            ),
        },
    })
    files[Path("execution_plan.json")] = _json({
        "schema_version": "pldr-chronological-execution-plan-v1",
        "campaign_id": CAMPAIGN_ID,
        "placeholders": [
            "device", "tokens", "tokenizer", "registry", "run_root",
            "evidence_root", "construction_lock",
        ],
        "trajectories": [_trajectory_plan(row) for row in TRAJECTORIES],
        "qualification_command": [
            "python3", "scripts/run_chronological_qualification.py",
            "--output-dir", "{campaign_root}/qualification", "--require-pass",
        ],
        "pair_analysis_command_prelock": [
            "python3",
            "experiments/analysis/analyze_chronological_confirmation.py",
            "{ledger}", "--output", "{report}", "--require-pass",
        ],
        "pair_analysis_command_heldout": [
            "python3",
            "experiments/analysis/analyze_chronological_confirmation.py",
            "{ledger}", "--lock", "{construction_lock}",
            "--output", "{report}", "--require-pass",
        ],
        "plga_analysis_commands": {
            "construction": [
                "python3",
                "experiments/analysis/analyze_chronological_plga.py",
                "{construction_plga_ledgers}",
                "--output", "{construction_plga_report}", "--require-pass",
            ],
            "heldout": [
                "python3",
                "experiments/analysis/analyze_chronological_plga.py",
                "{heldout_plga_ledgers}",
                "--lock", "{construction_lock}",
                "--output", "{heldout_plga_report}", "--require-pass",
            ],
        },
        "intervention_analysis_command": [
            "python3",
            "experiments/analysis/analyze_chronological_intervention.py",
            "{intervention_ledgers}",
            "--lock", "{construction_lock}",
            "--output", "{intervention_report}", "--require-pass",
        ],
        "final_gate": "make release-check-current",
    })
    files[Path("ledger_schema.json")] = _json({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Chronological all-pairs compact ledger",
        "schema_version": LEDGER_SCHEMA,
        "container": "npz-without-pickle",
        "required_scalars": [
            "schema_version", "campaign_id", "role", "seed",
            "data_offset_chunks", "step_start", "layer", "beta1", "beta2",
            "epsilon", "checkpoint_sha256", "registry_sha256", "lock_sha256",
            "shape_safety_factor", "numerical_relative_coefficient",
            "numerical_absolute_charge",
        ],
        "required_arrays": [
            "normalized_rows", "gate", "clipped_gradient_history",
            "first_moment_tail", "second_moment_before",
            "normalized_row_jvp", "normalized_row_remainder_radius",
            "numerical_charge", "learning_rate", "weight_decay", "decay_mask",
            "optimizer_step_after", "update_batch_sha256", "segment_grid",
            "gate_update_max_abs_residual",
        ],
        "chronology": (
            "gradient histories are newest first and include the current "
            "post-clipping gradient"
        ),
        "endpoint_policy": (
            "successor rows and gates are outcomes used only for coverage, "
            "never inputs to the lower certificate"
        ),
        "pair_order": "lexicographic unordered pairs",
        "numerical_charge_policy": (
            "one nonnegative value per transition and pair, per transition "
            "and registered row, or per transition; row charges are summed "
            "at pair endpoints by the independent analyzer"
        ),
        "raw_parameter_nodes_forbidden": True,
    })
    files[Path("plga_ledger_schema.json")] = _json({
        "schema_version": PLGA_LEDGER_SCHEMA,
        "container": "npz-without-pickle",
        "construction_cells": 15,
        "heldout_cells": 60,
        "construction_contexts_per_cell": 8,
        "heldout_contexts_per_cell": 16,
        "required_raw_families": [
            "generator-and-collapsed-anchor", "PLGA-parameters",
            "occupied-curvature-endpoints", "rectangular-query-key-score",
            "centered-logits", "checkpoint-registry-lock-digests",
        ],
        "heldout_lock_required": True,
        "construction_lock_forbidden": True,
    })
    files[Path("intervention_ledger_schema.json")] = _json({
        "schema_version": INTERVENTION_LEDGER_SCHEMA,
        "container": "npz-without-pickle",
        "units": 48,
        "units_per_seed": 12,
        "updates_per_arm": 4,
        "arms": [
            "baseline", "adaptive-source-removal", "decay-removal",
            "shape-direction-removal",
        ],
        "source_anchor": REGISTRY["intervention_anchor_step"],
        "layer_rule": REGISTRY["intervention_layer_rule"],
        "context_rule": REGISTRY["intervention_context_rule"],
        "head_rule": REGISTRY["intervention_head_rule"],
        "row_pair": REGISTRY["intervention_pair"],
        "required_provenance": [
            "initial-model-digests", "initial-optimizer-digests",
            "per-arm-minibatch-digests", "optimizer-step-counters",
            "checkpoint-registry-lock-digests",
        ],
    })
    for stage in STAGES:
        files[Path(f"{stage['id'].lower()}_protocol.json")] = _json({
            "schema_version": SCHEMA_VERSION,
            "campaign_id": CAMPAIGN_ID,
            "stage": stage,
            "stop_on_failure": True,
            "producers": [
                "experiments/train_run.py",
                "experiments/confirm/capture_chronological_confirmation.py",
                "experiments/confirm/capture_chronological_plga.py",
                "experiments/confirm/run_chronological_intervention.py",
            ],
            "analyzers": [
                "experiments/analysis/analyze_chronological_confirmation.py",
                "experiments/analysis/analyze_chronological_plga.py",
                "experiments/analysis/analyze_chronological_intervention.py",
            ],
            "compact_evidence_only": True,
        })
    files[Path("README.md")] = (
        "# Chronological row-map collapse confirmation\n\n"
        "This generated bundle registers the prospective six-trajectory "
        "campaign for the actual-state chronological AdamW certificate. "
        "Development establishes a positive branch, construction freezes "
        "all constants, and four distinct-seed, distinct-offset trajectories "
        "evaluate the lock. Every registered row pair is analyzed in chunks.\n\n"
        "Regenerate with `python3 scripts/gen_chronological_protocols.py`. "
        "Stage into experiment-data with "
        "`python3 scripts/stage_chronological_confirmation.py`.\n"
    ).encode("utf-8")
    manifest = "".join(
        f"{_sha256(files[path])}  {path.as_posix()}\n"
        for path in sorted(files, key=lambda item: item.as_posix())
    )
    files[Path("CHECKSUMS.sha256")] = manifest.encode("ascii")
    return files


def generate(*, check: bool) -> None:
    expected = expected_files()
    if check:
        actual = {
            path.relative_to(OUTPUT): path.read_bytes()
            for path in OUTPUT.rglob("*")
            if path.is_file()
        } if OUTPUT.is_dir() else {}
        if actual != expected:
            raise SystemExit("chronological confirmation protocols are stale")
        print(f"chronological protocols: verified ({len(expected)} files)")
        return
    OUTPUT.mkdir(parents=True, exist_ok=True)
    unknown = {
        path.relative_to(OUTPUT)
        for path in OUTPUT.rglob("*") if path.is_file()
    } - set(expected)
    if unknown:
        raise SystemExit(f"refusing to remove unknown protocol files: {sorted(unknown)}")
    for relative, payload in expected.items():
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    print(f"chronological protocols: wrote {len(expected)} files to {OUTPUT}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    generate(check=arguments.check)


if __name__ == "__main__":
    main()
