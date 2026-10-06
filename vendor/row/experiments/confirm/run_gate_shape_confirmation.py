#!/usr/bin/env python3
"""Plan, launch, bind, and source-lock the gate-shape confirmation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from confirmation_artifacts import (  # noqa: E402
    load_complete_checkpoint,
    validate_measurement_registry,
)
from gate_shape import (  # noqa: E402
    gate_shape_energy_increment,
    plga_secant_chain,
)
from gate_shape_evidence import (  # noqa: E402
    digest_object,
    load_npz,
    seal_request,
    sha256_path,
    write_json_atomic,
)
from gate_shape_protocol_specs import (  # noqa: E402
    ARCHITECTURE,
    PREDICTIONS,
    RAW_REQUIRED,
    RESOURCE_CAPS,
    STAGES,
)


PROTOCOL_ROOT = ROOT / "experiments/protocols/gate_shape_confirmation"


def _quote(values):
    return " ".join(shlex.quote(str(value)) for value in values)


def _json(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _bound_files(path, root):
    value = _json(path)
    if (set(value) != {"schema_version", "files"}
            or value["schema_version"] != "pldr-bound-file-set-v1"):
        raise ValueError("source lock needs one bound file set")
    root = Path(root).resolve()
    result = []
    seen = set()
    for row in value["files"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("bound file-set row is malformed")
        relative = Path(row["path"])
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise ValueError("bound file-set path is not normalized")
        resolved = (root / relative).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise ValueError("bound file-set path does not resolve")
        if resolved in seen or sha256_path(resolved) != row["sha256"]:
            raise ValueError("bound file-set alias or digest mismatch")
        seen.add(resolved)
        result.append(resolved)
    if not result:
        raise ValueError("source lock cannot use an empty file set")
    return result


def plan():
    return {
        "schema_version": "pldr-gate-shape-execution-plan-v1",
        "status": "DRAFT_PENDING_TWO_DEVICE_QUALIFICATION",
        "stage_order": [stage["id"] for stage in STAGES],
        "stages": list(STAGES),
        "architecture": ARCHITECTURE,
        "resource_caps": RESOURCE_CAPS,
        "predictions": list(PREDICTIONS),
        "launch_gate": {
            "qualification_cuda0": "SEALED_Q_REQUIRED",
            "qualification_cuda1": "SEALED_Q_REQUIRED",
            "measured_unit_costs": "REQUIRED_BEFORE_PROTOCOL_FREEZE",
            "construction_lock": "REQUIRED_BEFORE_VALIDATION_ANALYSIS",
        },
    }


def _training_command(name, seed, device, tokens, tokenizer, registry, outdir):
    snapshots = sorted({
        step for block in ARCHITECTURE["capture_update_blocks"] for step in block})
    checkpoints = sorted({
        ARCHITECTURE["updates"], *ARCHITECTURE["anchor_source_steps"],
        *snapshots,
    })
    return _quote([
        "python3", "experiments/train_run.py",
        "--name", name,
        "--run-id", name,
        "--lineage-root", name,
        "--lr", "0.0003",
        "--warmup", "512",
        "--steps", ARCHITECTURE["updates"],
        "--schedule_total_steps", ARCHITECTURE["updates"],
        "--batch", ARCHITECTURE["batch_size"],
        "--ctx", ARCHITECTURE["context_length"],
        "--layers", ARCHITECTURE["layers"],
        "--heads", ARCHITECTURE["heads"],
        "--dk", ARCHITECTURE["head_width"],
        "--adff", ARCHITECTURE["glu_hidden_width"],
        "--seed", seed,
        "--device", device,
        "--tokens", tokens,
        "--tok_model", tokenizer,
        "--confirmation_registry", registry,
        "--outdir", outdir,
        "--ckpt_steps", ",".join(map(str, checkpoints)),
        "--optimizer_ckpt_steps", ",".join(map(str, checkpoints)),
        "--confirmation_snapshot_steps", ",".join(map(str, snapshots)),
        "--confirmation_snapshot_blocks", "phi",
        "--sharp_full", "0",
        "--sharp_block_every", "0",
        "--probe_region", "global",
        "--skip_generation",
        "--terminal_validation",
    ])


def training_commands(arguments):
    devices = [value for value in arguments.devices.split(",") if value]
    if len(devices) != 2 or len(set(devices)) != 2:
        raise ValueError("training needs two distinct registered devices")
    root = Path(arguments.output_root).resolve()
    rows = []
    specifications = (
        ("construction", 3401, devices[0]),
        ("validation", 3402, devices[1]),
        ("same_device_replay", 3401, devices[0]),
    )
    for role, seed, device in specifications:
        name = f"gate-shape-gate-shape-{role}-s{seed}"
        outdir = root / role
        rows.append({
            "role": role,
            "run_id": name,
            "lineage_root": name,
            "device": device,
            "command": _training_command(
                name, seed, device, Path(arguments.tokens).resolve(),
                Path(arguments.tokenizer).resolve(),
                Path(arguments.registry).resolve(), outdir),
        })
    write_json_atomic(arguments.output, {
        "schema_version": "pldr-gate-shape-training-commands-v1",
        "commands": rows,
        "scientific_semantics": (
            "Two registered trajectories plus one distinct-identity, same-device, "
            "from-scratch replay; no parameter-normal SVD or Lanczos basis."),
    })


def _continuation_command(source, source_path, arm, device, tokens, tokenizer,
                          registry, output_root):
    config = source["config"]
    model = source["model_config"]
    start = int(source["step"])
    nodes = list(range(start + 1, start + 1 + ARCHITECTURE["intervention_updates"]))
    multiplier = {"rate_0.75": 0.75, "rate_1.25": 1.25}.get(arm, 1.0)
    name = f"gate-shape-{arm}-s{start}"
    values = [
        "python3", "experiments/train_run.py",
        "--name", name,
        "--run-id", name,
        "--lineage-root", source["run_identity"]["lineage_root"],
        "--lr", config["lr"],
        "--learning_rate_multiplier", multiplier,
        "--warmup", config["warmup"],
        "--steps", len(nodes),
        "--schedule_total_steps", source["schedule_total_steps"],
        "--batch", config["batch"],
        "--ctx", config["ctx"],
        "--layers", model["layers"],
        "--heads", model["heads"],
        "--dk", model["head_width"],
        "--adff", model["row_hidden_width"],
        "--seed", config["seed"],
        "--device", device,
        "--tokens", tokens,
        "--tok_model", tokenizer,
        "--outdir", Path(output_root).resolve() / arm,
        "--init_from", source_path,
        "--confirmation_registry", registry,
        "--ckpt_steps", ",".join(map(str, nodes)),
        "--optimizer_ckpt_steps", ",".join(map(str, nodes)),
        "--confirmation_snapshot_steps", ",".join(map(str, nodes)),
        "--confirmation_snapshot_blocks", "phi",
        "--sharp_full", "0",
        "--sharp_block_every", "0",
        "--probe_region", config.get("probe_region", "global"),
        "--optimizer", config.get("optimizer", "adamw"),
        "--wd", config.get("wd", 0.1),
        "--clip", config.get("clip", 1.0),
        "--loss_mode", config.get("loss_mode", "block"),
        "--accum", config.get("accum", 1),
        "--anneal_floor", config.get("anneal_floor", 0.1),
        "--hold_until", config.get("hold_until", -1),
        "--skip_generation",
        "--terminal_validation",
    ]
    if arm == "freeze_final_gate":
        values.extend(["--freeze_final_gate_at", start])
    if arm == "freeze_upstream_shape":
        values.extend(["--freeze_upstream_shape_at", start])
    if config.get("const_lr", False):
        values.append("--const_lr")
    return name, _quote(values), nodes


def intervention_commands(arguments):
    source_path = Path(arguments.source_checkpoint).resolve()
    source = load_complete_checkpoint(source_path)
    registry_path = Path(arguments.registry).resolve()
    registry = _json(registry_path)
    validate_measurement_registry(registry, require_nonempty=True)
    if source["measurement_registry"] != registry:
        raise ValueError("intervention source does not own the registry")
    if (
        sha256_path(arguments.tokens) != registry["dataset_sha256"]
        or sha256_path(arguments.tokenizer) != registry["tokenizer_sha256"]
    ):
        raise ValueError("intervention data disagree with the registry")
    if source["optimizer_config"]["name"] != "adamw":
        raise ValueError("gate-shape interventions require live AdamW state")
    rows = []
    for arm in ARCHITECTURE["intervention_arms"]:
        name, command, nodes = _continuation_command(
            source, source_path, arm, arguments.device,
            Path(arguments.tokens).resolve(), Path(arguments.tokenizer).resolve(),
            registry_path, arguments.output_root)
        rows.append({
            "arm": arm,
            "run_id": name,
            "source_complete_state_sha256": source["state_manifest"][
                "complete_state_sha256"],
            "checkpoint_steps": nodes,
            "command": command,
        })
    write_json_atomic(arguments.output, {
        "schema_version": "pldr-gate-shape-intervention-commands-v1",
        "source_checkpoint": str(source_path),
        "source_checkpoint_sha256": sha256_path(source_path),
        "commands": rows,
        "only_registered_differences": [
            "learning_rate_multiplier", "freeze_final_gate_at",
            "freeze_upstream_shape_at", "run_id",
        ],
    })


def seal_stage(arguments):
    roles = [value.partition("=")[0] for value in arguments.artifact]
    expected = set(RAW_REQUIRED[arguments.stage])
    if len(roles) != len(set(roles)) or set(roles) != expected:
        raise ValueError(
            f"{arguments.stage} artifact roles must be: "
            + ", ".join(sorted(expected)))
    protocol = PROTOCOL_ROOT / f"{arguments.stage.lower()}_protocol.json"
    request = seal_request(
        arguments.stage, arguments.campaign_id, arguments.artifact_root,
        arguments.artifact, protocol)
    write_json_atomic(arguments.output, request)


def lock_construction(arguments):
    registry = _json(arguments.registry)
    validate_measurement_registry(registry, require_nonempty=True)
    files = _bound_files(arguments.capture_set, arguments.artifact_root)
    source_archives = {}
    live_digest = sha256_path(HERE / "gate_shape_live.py")
    application_digest = sha256_path(HERE / "gate_shape_application.py")
    geometries = {}
    capture_rows = []
    energy_rows = []
    application_rows = []
    for path in files:
        raw = load_npz(path, required=("stage", "producer_sha256"))
        stage = str(raw["stage"].item())
        digest = sha256_path(path)
        expected_producer = (
            application_digest if stage == "A" else live_digest)
        if str(raw["producer_sha256"].item()) != expected_producer:
            raise ValueError("construction evidence uses an inactive producer")
        source_archives[digest] = raw
        if stage == "G":
            if (
                str(raw["registry_role"].item()) != "construction"
                or str(raw["registry_sha256"].item())
                != registry["registry_sha256"]
            ):
                raise ValueError(
                    "construction geometry changed partition or registry")
            geometries[digest] = raw
        elif stage == "C":
            capture_rows.append(raw)
        elif stage == "C-energy-step":
            energy_rows.append(raw)
        elif stage == "A":
            if (
                str(raw["registry_role"].item()) != "construction"
                or str(raw["registry_sha256"].item())
                != registry["registry_sha256"]
            ):
                raise ValueError(
                    "construction application changed partition or registry")
            application_rows.append(raw)
        else:
            raise ValueError("construction lock file set has an unknown stage")
    if (
        not geometries or not capture_rows
        or not energy_rows or not application_rows
    ):
        raise ValueError(
            "construction lock needs geometry, capture, energy, and "
            "application evidence")
    safety = float(arguments.safety_factor)
    if not math.isfinite(safety) or safety < 1.0:
        raise ValueError("construction-lock safety factor must be at least one")
    tolerance = 2e-8
    dimension = ARCHITECTURE["lifted_gate_dimension_per_layer"]
    layer_count = ARCHITECTURE["layers"]

    capture_by_key = {}
    for raw in capture_rows:
        layer = int(raw["layer"].item())
        step = int(raw["step"].item())
        key = (layer, step)
        geometry_digest = str(raw["geometry_sha256"].item())
        geometry = geometries.get(geometry_digest)
        if (
            key in capture_by_key
            or geometry is None
            or int(geometry["layer"].item()) != layer
            or int(geometry["step"].item()) != step - 1
        ):
            raise ValueError(
                "construction capture does not bind one source geometry")
        operator = np.asarray(raw["operator"], dtype=np.float64)
        if operator.shape != (dimension, dimension):
            raise ValueError("construction gate block has the wrong dimension")
        capture_by_key[key] = raw
    expected_capture_keys = {
        (layer, step)
        for layer in range(layer_count)
        for block in ARCHITECTURE["capture_update_blocks"]
        for step in block
    }
    if set(capture_by_key) != expected_capture_keys:
        raise ValueError(
            "construction capture does not cover every registered block")

    capture_blocks = []
    forcing_charges = []
    for layer in range(layer_count):
        for block in ARCHITECTURE["capture_update_blocks"]:
            product = np.eye(dimension)
            constituent = []
            for step in block:
                raw = capture_by_key[(layer, step)]
                operator = np.asarray(raw["operator"], dtype=np.float64)
                product = operator @ product
                constituent.append(float(np.linalg.norm(operator, ord=2)))
                forcing_charges.append({
                    "layer": layer,
                    "block_start": block[0],
                    "step": step,
                    "norm": (
                        float(np.linalg.norm(raw["raw_gradient"])) * safety),
                })
            capture_blocks.append({
                "layer": layer,
                "block_start": block[0],
                "steps": list(block),
                "constituent_gains": constituent,
                "block_gain": (
                    float(np.linalg.norm(product, ord=2)) * safety),
            })

    transition_to_block = {}
    for block in ARCHITECTURE["capture_update_blocks"]:
        sources = [block[0] - 1, *block[:-1]]
        for offset, (source_step, target_step) in enumerate(
            zip(sources, block)
        ):
            transition_to_block[(source_step, target_step)] = (
                block[0], offset)
    energy_by_key = {}
    mechanism = {}
    for raw in energy_rows:
        source_digest = str(raw["source_sha256"].item())
        target_digest = str(raw["target_sha256"].item())
        source = geometries.get(source_digest)
        target = geometries.get(target_digest)
        if source is None or target is None:
            raise ValueError(
                "construction energy step lacks raw geometry endpoints")
        layer = int(raw["layer"].item())
        source_step = int(raw["source_step"].item())
        target_step = int(raw["target_step"].item())
        window = transition_to_block.get((source_step, target_step))
        key = (layer, source_step, target_step)
        if (
            key in energy_by_key
            or window is None
            or int(source["layer"].item()) != layer
            or int(target["layer"].item()) != layer
            or int(source["step"].item()) != source_step
            or int(target["step"].item()) != target_step
        ):
            raise ValueError(
                "construction energy step is outside its registered block")
        directional = np.asarray(
            source["q_directional"], dtype=np.float64)
        if not np.isfinite(directional).all():
            raise ValueError(
                "construction source geometry omitted its q derivative")
        increment = gate_shape_energy_increment(
            source["gamma"], target["gamma"], source["q"], target["q"])
        gamma_next = np.asarray(target["gamma"], dtype=np.float64)
        linear = float(np.sum(
            directional * gamma_next * gamma_next))
        remainder = np.asarray(
            target["q"] - source["q"] - directional, dtype=np.float64)
        recorded = (
            float(raw["energy"].item()),
            float(raw["energy_next"].item()),
            float(raw["gate_work"].item()),
            float(raw["shape_work"].item()),
            float(raw["shape_linear_work"].item()),
        )
        recomputed = (
            increment["energy"], increment["energy_next"],
            increment["gate_work"], increment["shape_work"], linear,
        )
        if (
            not np.allclose(
                recorded, recomputed, rtol=2e-8, atol=2e-10)
            or not np.allclose(
                raw["shape_remainder"], remainder,
                rtol=2e-8, atol=2e-10)
        ):
            raise ValueError("construction energy archive does not replay")
        block_start, offset = window
        energy_by_key[key] = {
            "layer": layer,
            "block_start": block_start,
            "transition_index": offset,
            "source_step": source_step,
            "target_step": target_step,
            "gate_charge": abs(increment["gate_work"]) * safety,
            "shape_linear_charge": abs(linear) * safety,
            "remainder_radius": (
                float(np.max(np.abs(remainder))) * safety),
        }
        accumulator = mechanism.setdefault(
            (layer, block_start), [0.0, 0.0])
        accumulator[0] += abs(increment["gate_work"])
        accumulator[1] += abs(increment["shape_work"])
    expected_energy_keys = {
        (layer, source_step, target_step)
        for layer in range(layer_count)
        for source_step, target_step in transition_to_block
    }
    if set(energy_by_key) != expected_energy_keys:
        raise ValueError("construction energy evidence is incomplete")

    mechanism_labels = []
    for (layer, block_start), (gate, shape) in sorted(
        mechanism.items()
    ):
        if gate > 1.25 * shape:
            label = "gate-led"
        elif shape > 1.25 * gate:
            label = "shape-led"
        else:
            label = "mixed"
        mechanism_labels.append({
            "layer": layer,
            "block_start": block_start,
            "label": label,
        })

    application_by_key = {}
    registered_pairs = {
        row["id"] for row in registry["application_pairs"]
        if row["id"].startswith("ga-c")
    }
    for raw in application_rows:
        layer = int(raw["layer"].item())
        head = int(raw["head"].item())
        pair_id = str(raw["pair_id"].item())
        key = (layer, head)
        expected_pair = (
            f"ga-c{layer * ARCHITECTURE['heads'] + head:03d}")
        if (
            key in application_by_key
            or pair_id != expected_pair
            or pair_id not in registered_pairs
        ):
            raise ValueError(
                "construction application pair map is incomplete")
        replay = plga_secant_chain(
            raw["generator_left"], raw["generator_right"], raw["weight"],
            raw["bias"], raw["powers"], raw["coupling"],
            raw["coupling_bias"])
        curvature = np.asarray(
            raw["curvature_difference"], dtype=np.float64)
        reference = np.asarray(raw["reference_logits"], dtype=np.float64)
        candidate = np.asarray(raw["candidate_logits"], dtype=np.float64)
        reference_centered = reference - np.mean(reference)
        candidate_centered = candidate - np.mean(candidate)
        centered_difference = candidate_centered - reference_centered
        if (
            np.linalg.norm(
                replay["realized_difference"] - curvature) > tolerance
            or np.linalg.norm(
                replay["predicted_difference"]
                - raw["secant_prediction"]) > tolerance
            or np.linalg.norm(
                centered_difference
                - raw["centered_logit_difference"]) > tolerance
        ):
            raise ValueError("construction PLGA archive does not replay")
        curvature_norm = float(np.linalg.norm(curvature))
        centered_norm = float(np.linalg.norm(centered_difference))
        if curvature_norm <= 1e-14:
            if centered_norm > tolerance:
                raise ValueError(
                    "zero curvature difference has nonzero logit transfer")
            downstream_gain = 0.0
        else:
            downstream_gain = centered_norm / curvature_norm
        application_by_key[key] = {
            "layer": layer,
            "head": head,
            "construction_pair_id": pair_id,
            "operator_bound": (
                float(replay["operator_bound"]) * safety),
            "downstream_gain": downstream_gain * safety,
        }
    expected_application_keys = {
        (layer, head)
        for layer in range(layer_count)
        for head in range(ARCHITECTURE["heads"])
    }
    if set(application_by_key) != expected_application_keys:
        raise ValueError(
            "construction application evidence lacks a layer/head pair")
    application_bounds = [
        application_by_key[key] for key in sorted(application_by_key)
    ]
    lock = {
        "schema_version": "pldr-gate-shape-lock-v2",
        "campaign_id": arguments.campaign_id,
        "code_sha256": digest_object({
            "gate_shape_live.py": live_digest,
            "gate_shape_application.py": application_digest,
        }),
        "protocol_sha256": sha256_path(PROTOCOL_ROOT / "campaign_design.json"),
        "registry_sha256": registry["registry_sha256"],
        "source_requests": [sha256_path(arguments.capture_set)],
        "source_archive_sha256": sorted(source_archives),
        "capture_blocks": capture_blocks,
        "spectral_threshold_rule": (
            "max(reconstruction_residual,100eps*max(opnorm,1))"),
        "energy_charges": [
            energy_by_key[key] for key in sorted(energy_by_key)
        ],
        "forcing_charges": forcing_charges,
        "mechanism_labels": mechanism_labels,
        "intervention_arms": list(ARCHITECTURE["intervention_arms"]),
        "application_bounds": application_bounds,
        "source_owned": True,
        "validation_artifacts": [],
        "target_energy_read": False,
        "empty_margin_decision": None,
        "safety_factor": safety,
    }
    lock["lock_sha256"] = digest_object(lock)
    write_json_atomic(arguments.output, lock)


def _parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    plan_parser = sub.add_parser("plan")
    plan_parser.add_argument("--output", required=True)
    training = sub.add_parser("training-commands")
    training.add_argument("--tokens", required=True)
    training.add_argument("--tokenizer", required=True)
    training.add_argument("--registry", required=True)
    training.add_argument("--output-root", required=True)
    training.add_argument("--devices", default="cuda:0,cuda:1")
    training.add_argument("--output", required=True)
    intervention = sub.add_parser("intervention-commands")
    intervention.add_argument("--source-checkpoint", required=True)
    intervention.add_argument("--tokens", required=True)
    intervention.add_argument("--tokenizer", required=True)
    intervention.add_argument("--registry", required=True)
    intervention.add_argument("--device", default="cuda:0")
    intervention.add_argument("--output-root", required=True)
    intervention.add_argument("--output", required=True)
    seal = sub.add_parser("seal-stage")
    seal.add_argument("--stage", choices=tuple(RAW_REQUIRED), required=True)
    seal.add_argument("--campaign-id", required=True)
    seal.add_argument("--artifact-root", required=True)
    seal.add_argument("--artifact", action="append", required=True)
    seal.add_argument("--output", required=True)
    lock = sub.add_parser("lock-construction")
    lock.add_argument("--campaign-id", required=True)
    lock.add_argument("--artifact-root", required=True)
    lock.add_argument("--capture-set", required=True)
    lock.add_argument("--registry", required=True)
    lock.add_argument("--safety-factor", type=float, default=1.10)
    lock.add_argument("--output", required=True)
    return parser


def main():
    arguments = _parser().parse_args()
    if arguments.command == "plan":
        write_json_atomic(arguments.output, plan())
    elif arguments.command == "training-commands":
        training_commands(arguments)
    elif arguments.command == "intervention-commands":
        intervention_commands(arguments)
    elif arguments.command == "seal-stage":
        seal_stage(arguments)
    else:
        lock_construction(arguments)


if __name__ == "__main__":
    main()
