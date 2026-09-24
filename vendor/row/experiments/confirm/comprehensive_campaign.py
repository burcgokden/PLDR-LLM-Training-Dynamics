"""Frozen specification for the comprehensive gate-shape campaign."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .comprehensive_gate_shape import permuted_chunk_order


SCHEMA_VERSION = "pldr-comprehensive-gate-shape-campaign-v1"
DATASET_SHA256 = (
    "7127603cbd401986b3b9bd114b5d35723d04ddff2e9624da0d013932d04ee7d7"
)
TOKENIZER_SHA256 = (
    "51f4369714712232bfc746188f347e550790d1d75e5e35bfa4399b784d2a666f"
)
PROBE_RESERVE_CHUNKS = 5120


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def digest_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def campaign_spec() -> dict[str, Any]:
    """Return a fresh copy of the immutable prospective specification."""

    value = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": "pldr-comprehensive-gate-shape-confirmation-v1",
        "status": "prospective_unexecuted",
        "scientific_target": (
            "finite-registry PLDR row-map collapse by exact gate-shape "
            "carrier, diameter-slack, and endpoint-work identities"
        ),
        "data": {
            "dataset": "tiiuae/falcon-refinedweb",
            "split": "train",
            "stream_start_document": 0,
            "token_count": 220_000_000,
            "document_count": 351_465,
            "token_dtype": "uint16",
            "packing": "nonoverlapping_contiguous_256_token_chunks",
            "dataset_sha256": DATASET_SHA256,
            "tokenizer_kind": "SentencePiece unigram",
            "tokenizer_vocabulary": 32_000,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "add_bos": False,
            "add_eos": True,
            "probe_reserve_chunks": PROBE_RESERVE_CHUNKS,
            "order_rule": (
                "PCG64 permutation of all nonprobe chunk indices keyed by "
                "trajectory seed; the order file digest is checkpoint-bound"
            ),
        },
        "model": {
            "family": "PLDR-LLM",
            "layers": 3,
            "heads": 4,
            "head_width": 64,
            "model_width": 256,
            "row_hidden_width": 170,
            "row_residual_layers": 8,
            "parameter_count": 20_622_914,
            "arithmetic": "float32 training; float64 decisive analysis",
        },
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 7.5e-4,
            "warmup_steps": 250,
            "schedule": "constant_after_warmup",
            "beta1": 0.9,
            "beta2": 0.95,
            "epsilon": 1.0e-5,
            "weight_decay": 0.1,
            "gradient_clip": {"kind": "value", "level": 1.0},
            "batch_size": 32,
            "context_length": 256,
            "gradient_accumulation": 1,
            "update_order": [
                "averaged_raw_gradient",
                "value_clip",
                "first_moment",
                "second_moment",
                "bias_correction",
                "decoupled_decay_and_loss_update",
                "post_step_intervention",
            ],
        },
        "registry": {
            "contexts": 24,
            "rows_per_context": 64,
            "row_count": 1536,
            "pair_count": 1_178_880,
            "pair_chunk_size": 32_768,
            "basis": "frozen registered token identities",
            "scope": "finite registry only",
        },
        "trajectories": {
            "development": [7444],
            "construction": [8444],
            "heldout": [9444, 10444, 11444, 12444],
            "anchors": [2000, 4000, 8000, 16000, 24000],
            "development_block_updates": 16,
            "confirmation_block_updates": 4,
            "layer_indices": [0, 1, 2],
        },
        "primary_estimands": [
            "endpoint_diameter_ratio",
            "minimum_exact_diameter_slack",
            "coordinate_gate_shape_carrier",
            "gate_duhamel_forcing",
            "exact_gate_then_shape_work",
            "two_arm_live_response_with_shape_correction",
            "plga_occupied_segment_secant_ratio",
        ],
        "decisions": {
            "identity_relative_tolerance": 9.094947017729282e-13,
            "native_float32_replay_tolerance": 1.52587890625e-5,
            "normalized_effect_floor": 6.103515625e-5,
            "development": (
                "all exact identities pass and at least one registered "
                "contracting block is found per layer"
            ),
            "construction": (
                "freeze endpoint-ratio, route, intervention, and PLGA "
                "thresholds using construction only"
            ),
            "heldout": (
                "apply the frozen lock unchanged to four seed-keyed "
                "trajectories and report every registered cell"
            ),
            "intervention": (
                "decide all four post-anchor updates jointly; require the "
                "cluster interval for the live response to clear the "
                "qualification-fixed effect floor and report the exact "
                "shape correction"
            ),
            "plga": (
                "compare each heldout occupied-segment secant magnitude to "
                "the construction maximum; query and key remain fixed"
            ),
        },
        "resource_contract": {
            "devices": "2 x NVIDIA GeForce RTX 4090",
            "device_memory_bytes_each": 24_564 * 1024 * 1024,
            "measured_training_rate_updates_per_second": 6.9,
            "projected_gpu_hours": 6.74,
            "gpu_hour_cap": 8.0,
            "compressed_output_cap_bytes": 4 * 1024**3,
            "capture_context_chunk_size": 8,
            "retention": (
                "compressed endpoint ledgers and scalar reports; no dense "
                "pair tensor and no nested-JVP tensor"
            ),
        },
        "diagnostics_only": [
            "observed endpoint JVP remainder",
            "native-float32 versus float64 replay discrepancy",
            "per-pair expansion count and maximizer changes",
        ],
    }
    value["spec_sha256"] = digest_object(value)
    return copy.deepcopy(value)


def validate_campaign_spec(value: dict[str, Any]) -> bool:
    expected = campaign_spec()
    if value != expected:
        raise ValueError("campaign specification differs from the frozen v1 plan")
    return True


def validate_checkpoint_campaign_binding(checkpoint: dict[str, Any]) -> bool:
    """Validate the immutable campaign identity stored in a checkpoint."""

    spec = campaign_spec()
    binding = checkpoint.get("data_state", {}).get("campaign_binding")
    required = {
        "campaign_id", "campaign_spec_sha256",
        "campaign_spec_file_sha256", "campaign_spec_path",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("checkpoint campaign binding is missing or malformed")
    source = Path(binding["campaign_spec_path"])
    if source.is_symlink() or not source.is_file():
        raise ValueError("checkpoint campaign specification is unavailable")
    file_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if (
        binding["campaign_id"] != spec["campaign_id"]
        or binding["campaign_spec_sha256"] != spec["spec_sha256"]
        or binding["campaign_spec_file_sha256"] != file_digest
    ):
        raise ValueError("checkpoint campaign identity or digest disagrees")
    with source.open("r", encoding="utf-8") as stream:
        if json.load(stream) != spec:
            raise ValueError("checkpoint campaign specification is stale")
    config = checkpoint.get("config", {})
    expected_config = {
        "layers": spec["model"]["layers"],
        "heads": spec["model"]["heads"],
        "dk": spec["model"]["head_width"],
        "adff": spec["model"]["row_hidden_width"],
        "batch": spec["optimizer"]["batch_size"],
        "ctx": spec["optimizer"]["context_length"],
        "accum": spec["optimizer"]["gradient_accumulation"],
        "lr": spec["optimizer"]["learning_rate"],
        "warmup": spec["optimizer"]["warmup_steps"],
        "optimizer": "adamw",
        "wd": spec["optimizer"]["weight_decay"],
        "clip": spec["optimizer"]["gradient_clip"]["level"],
        "const_lr": True,
        "probe_region": "global",
    }
    if any(config.get(name) != value for name, value in expected_config.items()):
        raise ValueError("checkpoint model, optimizer, or data configuration changed")
    model = checkpoint.get("model_config", {})
    expected_model = {
        "layers": spec["model"]["layers"],
        "heads": spec["model"]["heads"],
        "head_width": spec["model"]["head_width"],
        "width": spec["model"]["model_width"],
        "feed_forward_width": 683,
        "row_hidden_width": spec["model"]["row_hidden_width"],
        "row_residual_layers": spec["model"]["row_residual_layers"],
        "dtype": "torch.float32",
    }
    if any(model.get(name) != value for name, value in expected_model.items()):
        raise ValueError("checkpoint realized architecture changed")
    optimizer = checkpoint.get("optimizer_config", {})
    groups = optimizer.get("groups", [])
    if (
        optimizer.get("name") != "adamw"
        or optimizer.get("gradient_clip_kind") != "value"
        or optimizer.get("gradient_clip_value")
        != spec["optimizer"]["gradient_clip"]["level"]
        or not groups
        or any(
            group.get("initial_learning_rate")
            != spec["optimizer"]["learning_rate"]
            or group.get("betas") != [
                spec["optimizer"]["beta1"], spec["optimizer"]["beta2"]]
            or group.get("epsilon") != spec["optimizer"]["epsilon"]
            or group.get("weight_decay") != spec["optimizer"]["weight_decay"]
            or group.get("amsgrad") is not False
            or group.get("maximize") is not False
            for group in groups
        )
    ):
        raise ValueError("checkpoint realized AdamW configuration changed")
    data_state = checkpoint.get("data_state", {})
    if (
        data_state.get("context_length") != spec["optimizer"]["context_length"]
        or data_state.get("rows_per_step") != spec["optimizer"]["batch_size"]
        or data_state.get("probe_region") != "global"
        or not isinstance(data_state.get("data_order_sha256"), str)
        or len(data_state["data_order_sha256"]) != 64
    ):
        raise ValueError("checkpoint realized data-order configuration changed")
    return True

def write_chunk_orders(
    output_dir: str | Path,
    *,
    total_chunks: int,
) -> list[dict[str, Any]]:
    """Write one probe-safe order per trajectory and return its manifest."""

    spec = campaign_spec()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    reserved_start = int(total_chunks) - PROBE_RESERVE_CHUNKS
    if reserved_start <= 0:
        raise ValueError("token archive is smaller than the probe reservation")
    reserved = np.arange(reserved_start, int(total_chunks), dtype=np.int64)
    roles = spec["trajectories"]
    manifest = []
    for role in ("development", "construction", "heldout"):
        for seed in roles[role]:
            generated = permuted_chunk_order(
                int(total_chunks), int(seed), reserved_chunks=reserved)
            path = directory / f"{role}-seed{seed}-chunk-order.npy"
            np.save(path, generated["order"], allow_pickle=False)
            file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append({
                "role": role,
                "seed": int(seed),
                "path": path.name,
                "array_sha256": generated["sha256"],
                "file_sha256": file_digest,
                "chunk_count": generated["chunk_count"],
                "reserved_start": reserved_start,
            })
    return manifest


__all__ = [
    "DATASET_SHA256",
    "PROBE_RESERVE_CHUNKS",
    "SCHEMA_VERSION",
    "TOKENIZER_SHA256",
    "campaign_spec",
    "digest_object",
    "validate_checkpoint_campaign_binding",
    "validate_campaign_spec",
    "write_chunk_orders",
]
