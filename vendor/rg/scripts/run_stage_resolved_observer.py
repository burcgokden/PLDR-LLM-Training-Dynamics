#!/usr/bin/env python3
"""Replay every saved checkpoint through every row-map normalization stage.

This experiment distinguishes a represented float32 zero from a real-valued
face reached by the final LayerNorm. It records all six layer-local stages,
reruns the whole network in float64, and reruns the row-map pipeline in
float64 from the exact float32 S0 tensor.
"""

from __future__ import annotations
from companion_paths import legacy_path

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, seal_record, verify_record_seal  # noqa: E402
from row_rgmap.observer import (  # noqa: E402
    layernorm_row_energy_enclosure,
    represented_input_class,
)
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402
from scripts.run_observer_precision_audit import (  # noqa: E402
    checkpoint_inventory,
    option,
    probe_tokens,
)


STAGES = ("S0", "S1", "S2in", "S2out", "S3in", "S3out")
LAYERNORM_INPUT_STAGE = {
    "S1": "S0",
    "S2out": "S2in",
    "S3out": "S3in",
}
PREDECESSOR_SCHEMAS = {
    "pldr-row-rg-stage-resolved-observer-v3",
    "pldr-row-rg-stage-resolved-observer-v4",
}
PREDECESSOR_TOP_LEVEL_KEYS = {
    "precision_bits",
    "arithmetic_policy",
    "devices",
    "architecture",
    "map_ids",
    "inputs",
    "evaluations",
    "cross_device_checkpoint_summary",
    "summary",
}
PREDECESSOR_SUMMARY_KEYS = {
    "all_producing_device_float64_whole_network_final_rounded_centered_sum_maximum",
    "all_producing_device_float64_whole_network_final_rounded_centered_sum_minimum",
    "all_producing_device_float64_whole_network_final_zero_count",
    "checkpoint_count",
    "corresponding_float64_from_float32_S0_energy_maximum",
    "corresponding_float64_from_float32_S0_energy_minimum",
    "corresponding_float64_from_float32_S0_one_ulp_count",
    "corresponding_float64_from_float32_S0_positive_beyond_one_ulp_count",
    "corresponding_float64_from_float32_S0_positive_energy_minimum",
    "corresponding_float64_from_float32_S0_rounded_minimum",
    "corresponding_float64_from_float32_S0_status",
    "corresponding_float64_from_float32_S0_zero_count",
    "corresponding_float64_whole_network_energy_maximum",
    "corresponding_float64_whole_network_energy_minimum",
    "corresponding_float64_whole_network_one_ulp_count",
    "corresponding_float64_whole_network_positive_beyond_one_ulp_count",
    "corresponding_float64_whole_network_positive_energy_minimum",
    "corresponding_float64_whole_network_rounded_minimum",
    "corresponding_float64_whole_network_status",
    "corresponding_float64_whole_network_zero_count",
    "device_count",
    "device_map_evaluation_count",
    "interpretation",
    "map_count_per_checkpoint",
    "producing_device_exact_input_classification_counts",
    "producing_device_final_represented_zero_count",
    "producing_device_first_coalescence_diagnostics",
    "producing_device_first_coalescence_input_classification_counts",
    "producing_device_first_float32_coalescence_stage_counts",
    "device_dependent_zero_status_count",
}
PREDECESSOR_ARITHMETIC_KEYS = {
    "native_network",
    "whole_network_control",
    "pipeline_control",
    "rounded_centered_sum",
    "represented_energy",
    "near_collapse",
    "exact_input_enclosure",
    "tf32_enabled",
}
PREDECESSOR_ARCHITECTURE_KEYS = {
    "layers",
    "heads",
    "dk",
    "contexts",
    "context_length",
    "max_sequence_length",
}
PREDECESSOR_INPUT_KEYS = {
    "run_id",
    "resolved_config",
    "checkpoints",
    "dataset",
    "probe_document_offset",
    "probe_byte_token_sha256",
    "reference_sources",
}
PREDECESSOR_EVALUATION_KEYS = {
    "run_id",
    "trajectory_id",
    "step",
    "device",
    "producing_device",
    "recorded_energy_bitwise_match_count",
    "maps",
}
PREDECESSOR_CROSS_DEVICE_KEYS = {
    "trajectory_id",
    "step",
    "represented_zero_count_by_device",
    "device_dependent_zero_status_count",
}



SOURCE_PATHS = (
    "scripts/run_stage_resolved_observer.py",
    "scripts/run_observer_precision_audit.py",
    "src/row_rgmap/observer.py",
    "src/row_rgmap/analysis.py",
    "src/row_rgmap/provenance_v3.py",
)


def scientific_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Return stage leaves after normalizing the corrected rounded field names."""

    summary = dict(record["summary"])
    renames = {
        "all_producing_device_float64_whole_network_final_energy_minimum": (
            "all_producing_device_float64_whole_network_final_rounded_centered_sum_minimum"
        ),
        "all_producing_device_float64_whole_network_final_energy_maximum": (
            "all_producing_device_float64_whole_network_final_rounded_centered_sum_maximum"
        ),
    }
    for old, new in renames.items():
        if old in summary:
            summary[new] = summary.pop(old)
    return {
        key: record[key]
        for key in (
            "precision_bits",
            "arithmetic_policy",
            "devices",
            "architecture",
            "map_ids",
            "inputs",
            "evaluations",
            "cross_device_checkpoint_summary",
        )
    } | {"summary": summary}


def committed_identity() -> tuple[str, list[dict[str, Any]]]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("stage replay requires a clean committed worktree")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources = []
    for repository_path in SOURCE_PATHS:
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        sources.append(
            {
                "repository_path": repository_path,
                "code_commit": commit,
                "sha256": digest,
                "size_bytes": size,
            }
        )
    return commit, sources


def build_model(
    torch: Any,
    model_class: Any,
    specification: dict[str, int],
    device: Any,
    dtype: Any,
) -> Any:
    d_model = specification["heads"] * specification["dk"]
    model = model_class(
        num_layers=specification["layers"],
        d_model=d_model,
        num_heads=specification["heads"],
        dff=int(2.7 * d_model),
        input_vocab_size=257,
        A_dff=2 * specification["dk"],
        num_reslayerA=2,
        num_denseA=2,
        max_seq_len=specification["max_sequence_length"],
        device=device,
    )
    return model.to(device=device, dtype=dtype).eval()


def capture_stages(
    torch: Any,
    model: Any,
    probe: Any,
    context_length: int,
) -> dict[tuple[int, str], Any]:
    mask = torch.triu(
        torch.ones(
            context_length,
            context_length,
            dtype=torch.float32,
            device=probe.device,
        ),
        diagonal=1,
    )[None, None]
    captured: dict[tuple[int, str], Any] = {}
    handles = []

    def register(module: Any, layer: int, before: str, after: str) -> None:
        def hook(_module: Any, inputs: Any, output: Any) -> None:
            captured[(layer, before)] = inputs[0].detach().clone()
            captured[(layer, after)] = output.detach().clone()

        handles.append(module.register_forward_hook(hook))

    for layer, decoder_layer in enumerate(model.decoder.dec_layers):
        row_map = decoder_layer.mha1
        register(row_map.layernorm1, layer, "S0", "S1")
        register(row_map.reslayerAs[0].layernormA, layer, "S2in", "S2out")
        register(row_map.reslayerAs[1].layernormA, layer, "S3in", "S3out")
    try:
        with torch.no_grad():
            _, _, _, cache = model([probe[:, :-1], mask])
    finally:
        for handle in handles:
            handle.remove()
    expected = {
        (layer, stage)
        for layer in range(len(model.decoder.dec_layers))
        for stage in STAGES
    }
    if set(captured) != expected:
        raise RuntimeError("LayerNorm stage hooks did not fire exactly once")
    for layer in range(len(model.decoder.dec_layers)):
        if not torch.equal(captured[(layer, "S3out")], cache[layer][2]):
            raise RuntimeError("captured S3out differs from the model row-map cache")
    return captured


def layernorm64(torch: Any, inputs: Any, module: Any) -> Any:
    centered = inputs - inputs.mean(dim=-1, keepdim=True)
    variance = (centered * centered).mean(dim=-1, keepdim=True)
    return (
        centered
        / torch.sqrt(variance + float(module.eps))
        * module.weight.double()
        + module.bias.double()
    )


def glu64(torch: Any, inputs: Any, module: Any) -> Any:
    first = torch.nn.functional.silu(
        inputs @ module.gluw1.weight.double().T + module.gluw1.bias.double()
    )
    second = (
        inputs @ module.gluw2.weight.double().T + module.gluw2.bias.double()
    )
    return (
        (first * second) @ module.gluw3.weight.double().T
        + module.gluw3.bias.double()
    )


def pipeline64_from_float32_s0(
    torch: Any,
    float32_stages: dict[tuple[int, str], Any],
    model32: Any,
) -> dict[tuple[int, str], Any]:
    result: dict[tuple[int, str], Any] = {}
    for layer, decoder_layer in enumerate(model32.decoder.dec_layers):
        row_map = decoder_layer.mha1
        value = float32_stages[(layer, "S0")].double()
        result[(layer, "S0")] = value
        value = layernorm64(torch, value, row_map.layernorm1)
        result[(layer, "S1")] = value
        for residual_index, residual in enumerate(row_map.reslayerAs):
            before = value
            for dense in residual.denseAs:
                value = glu64(torch, value, dense)
            value = value + before
            input_stage = f"S{residual_index + 2}in"
            output_stage = f"S{residual_index + 2}out"
            result[(layer, input_stage)] = value
            value = layernorm64(torch, value, residual.layernormA)
            result[(layer, output_stage)] = value
    return result


def map_tensor(
    stages: dict[tuple[int, str], Any],
    context: int,
    layer: int,
    head: int,
    stage: str,
) -> Any:
    return stages[(layer, stage)][context, head]


def distinct_rows(tensor: Any) -> int:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return len({array[index].tobytes() for index in range(array.shape[0])})


def row_energy(tensor: Any) -> float:
    array = tensor.detach().cpu().numpy().astype(np.float64)
    centered = array - array.mean(axis=0, keepdims=True, dtype=np.float64)
    return float(np.sum(centered * centered, dtype=np.float64))


def _reduce_dyadic(numerator: int, exponent: int) -> tuple[int, int]:
    if numerator == 0:
        return 0, 0
    while numerator % 2 == 0:
        numerator //= 2
        exponent += 1
    return numerator, exponent


def _add_dyadics(
    left: tuple[int, int], right: tuple[int, int]
) -> tuple[int, int]:
    left_numerator, left_exponent = left
    right_numerator, right_exponent = right
    if left_numerator == 0:
        return right
    if right_numerator == 0:
        return left
    exponent = min(left_exponent, right_exponent)
    numerator = (
        (left_numerator << (left_exponent - exponent))
        + (right_numerator << (right_exponent - exponent))
    )
    return _reduce_dyadic(numerator, exponent)


def exact_dyadic_row_energy(array: np.ndarray) -> tuple[int, int]:
    """Return exact represented row energy as numerator times two to a power."""

    values = np.asarray(array)
    if values.dtype != np.dtype(np.float64) or values.ndim != 2:
        raise ValueError("exact dyadic row energy requires a float64 matrix")
    if not np.isfinite(values).all():
        raise ValueError("exact dyadic row energy requires finite entries")
    row_count = values.shape[0]
    if row_count < 1 or row_count & (row_count - 1):
        raise ValueError("exact dyadic row energy requires a power-of-two row count")
    row_exponent = row_count.bit_length() - 1
    total = (0, 0)
    for column in values.T:
        ratios = [float(value).as_integer_ratio() for value in column]
        denominator_exponents = [
            denominator.bit_length() - 1 for _, denominator in ratios
        ]
        common_exponent = max(denominator_exponents)
        integers = [
            numerator << (common_exponent - denominator_exponent)
            for (numerator, _), denominator_exponent in zip(
                ratios, denominator_exponents
            )
        ]
        numerator = (
            row_count * sum(value * value for value in integers)
            - sum(integers) ** 2
        )
        if numerator < 0:
            raise ArithmeticError("pairwise row energy became negative")
        total = _add_dyadics(
            total,
            _reduce_dyadic(
                numerator, -2 * common_exponent - row_exponent
            ),
        )
    return total


def maximum_coordinate_ulp_span(array: np.ndarray) -> int:
    values = np.ascontiguousarray(array, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("ulp span requires a finite float64 matrix")
    bits = values.view(np.uint64)
    sign = np.uint64(1 << 63)
    keys = np.where((bits & sign) != 0, ~bits, bits | sign)
    return max(
        int(keys[:, column].max()) - int(keys[:, column].min())
        for column in range(keys.shape[1])
    )


def exact_represented_energy_record(tensor: Any) -> dict[str, Any]:
    array = np.ascontiguousarray(
        tensor.detach().cpu().numpy(), dtype=np.float64
    )
    numerator, exponent = exact_dyadic_row_energy(array)
    exact_value = float(
        Fraction(numerator, 1 << -exponent)
        if exponent < 0
        else Fraction(numerator << exponent, 1)
    )
    ulp_span = maximum_coordinate_ulp_span(array)
    if numerator == 0:
        classification = "REPRESENTED_ZERO"
    elif ulp_span <= 1:
        classification = "ONE_ULP_NEAR_COLLAPSE"
    else:
        classification = "POSITIVE_BEYOND_ONE_ULP"
    return {
        "exact_dyadic_energy": {
            "numerator": str(numerator),
            "power_of_two_exponent": exponent,
            "float": exact_value,
        },
        "maximum_coordinate_ulp_span": ulp_span,
        "classification": classification,
    }


def stage_record(
    stages: dict[tuple[int, str], Any],
    context: int,
    layer: int,
    head: int,
) -> dict[str, dict[str, float | int]]:
    return {
        stage: {
            "distinct_rows": distinct_rows(
                map_tensor(stages, context, layer, head, stage)
            ),
            "row_centered_energy": row_energy(
                map_tensor(stages, context, layer, head, stage)
            ),
        }
        for stage in STAGES
    }


def rational_interval_record(interval: tuple[Any, Any]) -> dict[str, Any]:
    lower, upper = interval
    return {
        "lower_float": float(lower),
        "upper_float": float(upper),
        "lower_positive": lower > 0,
        "upper_zero": upper == 0,
    }


def validate_predecessor_record(record: Any) -> dict[str, Any]:
    """Validate the complete normalized predecessor before importing Torch."""

    try:
        verify_record_seal(record)
        if not isinstance(record, dict) or record.get("schema_version") not in PREDECESSOR_SCHEMAS:
            raise ValueError
        normalized = scientific_payload(record)
        if set(normalized) != PREDECESSOR_TOP_LEVEL_KEYS:
            raise ValueError
        if set(normalized["summary"]) != PREDECESSOR_SUMMARY_KEYS:
            raise ValueError
        if set(normalized["arithmetic_policy"]) != PREDECESSOR_ARITHMETIC_KEYS:
            raise ValueError
        if set(normalized["architecture"]) != PREDECESSOR_ARCHITECTURE_KEYS:
            raise ValueError
        if set(normalized["inputs"]) != PREDECESSOR_INPUT_KEYS:
            raise ValueError
        if not isinstance(normalized["precision_bits"], int) or normalized["precision_bits"] < 2:
            raise ValueError
        if any(
            not isinstance(normalized["architecture"][key], int)
            or normalized["architecture"][key] < 1
            for key in PREDECESSOR_ARCHITECTURE_KEYS
        ):
            raise ValueError

        map_ids = normalized["map_ids"]
        if (
            not isinstance(map_ids, list)
            or not map_ids
            or len(set(map_ids)) != len(map_ids)
            or any(not isinstance(value, str) or not value for value in map_ids)
        ):
            raise ValueError

        devices = normalized["devices"]
        if not isinstance(devices, list) or not devices:
            raise ValueError
        device_labels = []
        for device in devices:
            if not isinstance(device, dict) or not {"label", "type", "name"}.issubset(device):
                raise ValueError
            if any(
                not isinstance(device[key], str) or not device[key]
                for key in ("label", "type", "name")
            ):
                raise ValueError
            device_labels.append(device["label"])
        if len(set(device_labels)) != len(device_labels):
            raise ValueError

        inputs = normalized["inputs"]
        checkpoints = inputs["checkpoints"]
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError
        checkpoint_keys = {
            "evidence_role",
            "path",
            "producing_device",
            "run_id",
            "sha256",
            "size_bytes",
            "step",
            "trajectory_id",
        }
        for checkpoint in checkpoints:
            if not isinstance(checkpoint, dict) or set(checkpoint) != checkpoint_keys:
                raise ValueError
            if (
                not isinstance(checkpoint["producing_device"], int)
                or not isinstance(checkpoint["step"], int)
            ):
                raise ValueError

        evaluations = normalized["evaluations"]
        if not isinstance(evaluations, list) or not evaluations:
            raise ValueError
        for evaluation in evaluations:
            if (
                not isinstance(evaluation, dict)
                or set(evaluation) != PREDECESSOR_EVALUATION_KEYS
                or not isinstance(evaluation["maps"], list)
                or len(evaluation["maps"]) != len(map_ids)
                or evaluation["device"] not in device_labels
            ):
                raise ValueError

        cross_device = normalized["cross_device_checkpoint_summary"]
        if not isinstance(cross_device, list) or len(cross_device) != len(checkpoints):
            raise ValueError
        for row in cross_device:
            if (
                not isinstance(row, dict)
                or set(row) != PREDECESSOR_CROSS_DEVICE_KEYS
                or set(row["represented_zero_count_by_device"]) != set(device_labels)
            ):
                raise ValueError

        summary = normalized["summary"]
        integer_keys = {
            key
            for key in PREDECESSOR_SUMMARY_KEYS
            if key.endswith("_count")
            or key
            in {
                "checkpoint_count",
                "device_count",
                "device_map_evaluation_count",
                "map_count_per_checkpoint",
            }
        }
        if any(not isinstance(summary[key], int) for key in integer_keys):
            raise ValueError
        if (
            summary["checkpoint_count"] != len(checkpoints)
            or summary["device_count"] != len(devices)
            or summary["map_count_per_checkpoint"] != len(map_ids)
            or summary["device_map_evaluation_count"] != len(evaluations) * len(map_ids)
        ):
            raise ValueError
        return normalized
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("predecessor summary schema mismatch") from error


def validate_producing_device_routes(
    checkpoints: list[dict[str, Any]], devices: list[str]
) -> None:
    """Require every archived producing CUDA route before model construction."""

    try:
        required = {
            f"cuda:{int(checkpoint['producing_device'])}"
            for checkpoint in checkpoints
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("producing-device route metadata is malformed") from error
    missing = sorted(required - set(devices))
    if missing:
        raise ValueError(
            "producing-device route is absent from --devices: " + ", ".join(missing)
        )



def preflight_inputs(
    completed_run_root: Path,
    devices: list[str],
    predecessor_path: Path | None = None,
) -> dict[str, Any]:
    """Resolve and validate all archived inputs before importing torch."""

    predecessor = None
    predecessor_scientific = None
    resolved_predecessor = None
    if predecessor_path is not None:
        resolved_predecessor = predecessor_path.resolve()
        predecessor = json.loads(resolved_predecessor.read_text(encoding="utf-8"))
        predecessor_scientific = validate_predecessor_record(predecessor)

    run_root = completed_run_root.resolve()
    config, checkpoints = checkpoint_inventory(
        run_root, "COMPLETED_DECISION_RECORD"
    )
    if not checkpoints:
        raise ValueError("no completed checkpoints were found")
    validate_producing_device_routes(checkpoints, devices)
    return {
        "run_root": run_root,
        "config": config,
        "checkpoints": checkpoints,
        "predecessor_path": resolved_predecessor,
        "predecessor": predecessor,
        "predecessor_scientific": predecessor_scientific,
    }


def device_description(torch: Any, label: str) -> dict[str, Any]:
    device = torch.device(label)
    if device.type != "cuda":
        return {"label": label, "type": device.type, "name": "host CPU"}
    if (
        not torch.cuda.is_available()
        or device.index is None
        or device.index >= torch.cuda.device_count()
    ):
        raise RuntimeError(f"requested device is unavailable: {label}")
    properties = torch.cuda.get_device_properties(device)
    return {
        "label": label,
        "type": "cuda",
        "name": properties.name,
        "compute_capability": f"{properties.major}.{properties.minor}",
        "total_memory_bytes": int(properties.total_memory),
    }


def first_coalescence(stage_row: dict[str, dict[str, float | int]]) -> str | None:
    return next(
        (
            stage
            for stage in STAGES
            if int(stage_row[stage]["distinct_rows"]) == 1
        ),
        None,
    )


def layernorm_module(model: Any, layer: int, output_stage: str) -> Any:
    row_map = model.decoder.dec_layers[layer].mha1
    if output_stage == "S1":
        return row_map.layernorm1
    if output_stage == "S2out":
        return row_map.reslayerAs[0].layernormA
    if output_stage == "S3out":
        return row_map.reslayerAs[1].layernormA
    raise ValueError(f"stage is not a LayerNorm output: {output_stage}")


def affine_parameter_record(module: Any) -> dict[str, float | bool]:
    gain = module.weight.detach().double().cpu().numpy()
    bias = module.bias.detach().double().cpu().numpy()
    return {
        "epsilon": float(module.eps),
        "gain_l2_norm": float(np.linalg.norm(gain)),
        "gain_linf_norm": float(np.linalg.norm(gain, ord=np.inf)),
        "gain_all_zero": bool(np.all(gain == 0.0)),
        "bias_l2_norm": float(np.linalg.norm(bias)),
        "bias_linf_norm": float(np.linalg.norm(bias, ord=np.inf)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--completed-run-root", required=True)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda:0", "cuda:1"])
    parser.add_argument("--precision-bits", type=int, default=192)
    parser.add_argument("--refinedweb-root", default=legacy_path('/pldr-assets/refinedweb'))
    parser.add_argument("--predecessor")
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    started = time.perf_counter()
    commit, analysis_sources = committed_identity()
    preflight = preflight_inputs(
        Path(arguments.completed_run_root),
        list(arguments.devices),
        Path(arguments.predecessor) if arguments.predecessor else None,
    )
    run_root = preflight["run_root"]
    config = preflight["config"]
    checkpoints = preflight["checkpoints"]
    predecessor_path = preflight["predecessor_path"]
    predecessor = preflight["predecessor"]
    predecessor_scientific = preflight["predecessor_scientific"]

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    base_command = config["producer_command"]
    specification = {
        "layers": int(option(base_command, "--layers")),
        "heads": int(option(base_command, "--heads")),
        "dk": int(option(base_command, "--dk")),
        "contexts": int(option(base_command, "--contexts")),
        "context_length": int(option(base_command, "--context-length")),
        "max_sequence_length": int(option(base_command, "--max-sequence-length")),
    }
    reference_root = Path(option(base_command, "--reference-experiments")).resolve()
    dataset_shard = Path(option(base_command, "--dataset-shard")).resolve()
    refinedweb_root = Path(arguments.refinedweb_root).resolve()
    dataset_relative = dataset_shard.relative_to(refinedweb_root)
    probe_offset = int(option(base_command, "--probe-document-offset"))
    probe_count = specification["contexts"] * (
        specification["context_length"] + 1
    )
    token_values, probe_digest = probe_tokens(
        dataset_shard, probe_offset, probe_count
    )
    metadata = json.loads(
        (
            run_root
            / "trajectories"
            / config["trajectories"][0]["trajectory_id"]
            / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    if probe_digest != metadata["data"]["probe"]["byte_token_sha256"]:
        raise ValueError("reconstructed probe differs from producer metadata")

    sys.path.insert(0, str(reference_root))
    from pldr_model_v510 import PLDR_Model

    descriptions = [
        device_description(torch, label) for label in arguments.devices
    ]
    models: dict[tuple[str, str], Any] = {}
    probes: dict[str, Any] = {}
    for label in arguments.devices:
        device = torch.device(label)
        torch.manual_seed(0)
        models[(label, "float32")] = build_model(
            torch, PLDR_Model, specification, device, torch.float32
        )
        torch.manual_seed(0)
        models[(label, "float64")] = build_model(
            torch, PLDR_Model, specification, device, torch.float64
        )
        probes[label] = torch.tensor(
            token_values, dtype=torch.long, device=device
        ).reshape(
            specification["contexts"],
            specification["context_length"] + 1,
        )

    map_coordinates = [
        (context, layer, head, f"c{context}.L{layer}.H{head}")
        for context in range(specification["contexts"])
        for layer in range(specification["layers"])
        for head in range(specification["heads"])
    ]
    evaluations = []
    checkpoint_inputs = []
    for checkpoint_descriptor in checkpoints:
        checkpoint = torch.load(
            checkpoint_descriptor["absolute_path"],
            map_location="cpu",
            weights_only=False,
        )
        if (
            checkpoint.get("trajectory_id")
            != checkpoint_descriptor["trajectory_id"]
            or int(checkpoint.get("step", -1))
            != checkpoint_descriptor["step"]
        ):
            raise ValueError("checkpoint identity disagrees with its path")
        recorded = np.asarray(
            checkpoint["recorder"]["energies"], dtype=np.float64
        )[-1]
        if recorded.shape != (len(map_coordinates),):
            raise ValueError("checkpoint recorder has an unexpected map count")
        checkpoint_inputs.append(
            {
                key: value
                for key, value in checkpoint_descriptor.items()
                if key != "absolute_path"
            }
        )

        for device_label in arguments.devices:
            device = torch.device(device_label)
            model32 = models[(device_label, "float32")]
            model64 = models[(device_label, "float64")]
            model32.load_state_dict(checkpoint["model"], strict=True)
            model64.load_state_dict(checkpoint["model"], strict=True)
            model32.eval()
            model64.eval()
            stages32 = capture_stages(
                torch,
                model32,
                probes[device_label],
                specification["context_length"],
            )
            stages64_network = capture_stages(
                torch,
                model64,
                probes[device_label],
                specification["context_length"],
            )
            with torch.no_grad():
                stages64_pipeline = pipeline64_from_float32_s0(
                    torch, stages32, model32
                )

            rows = []
            for index, (context, layer, head, map_id) in enumerate(
                map_coordinates
            ):
                float32_row = stage_record(
                    stages32, context, layer, head
                )
                network64_row = stage_record(
                    stages64_network, context, layer, head
                )
                pipeline64_row = stage_record(
                    stages64_pipeline, context, layer, head
                )
                final_zero = float32_row["S3out"]["distinct_rows"] == 1
                if final_zero:
                    network64_row["S3out"]["represented_energy"] = (
                        exact_represented_energy_record(
                            map_tensor(stages64_network, context, layer, head, "S3out")
                        )
                    )
                    pipeline64_row["S3out"]["represented_energy"] = (
                        exact_represented_energy_record(
                            map_tensor(stages64_pipeline, context, layer, head, "S3out")
                        )
                    )
                row: dict[str, Any] = {
                    "map_id": map_id,
                    "map_index": index,
                    "recorded_energy": float(recorded[index]),
                    "float32": float32_row,
                    "float64_whole_network": network64_row,
                    "float64_from_float32_S0": pipeline64_row,
                    "float32_final_represented_zero": final_zero,
                }
                if final_zero:
                    first_stage = first_coalescence(float32_row)
                    transitions = {}
                    for output_stage, input_stage in LAYERNORM_INPUT_STAGE.items():
                        if float32_row[output_stage]["distinct_rows"] != 1:
                            continue
                        normalization = layernorm_module(
                            model32, layer, output_stage
                        )
                        exact_input = map_tensor(
                            stages32, context, layer, head, input_stage
                        )
                        gain = (
                            normalization.weight.detach().float().cpu().numpy()
                        )
                        interval = layernorm_row_energy_enclosure(
                            exact_input.float().cpu().numpy(),
                            gain,
                            float(normalization.eps),
                            precision_bits=arguments.precision_bits,
                        )
                        input_identical = (
                            float32_row[input_stage]["distinct_rows"] == 1
                        )
                        transitions[output_stage] = {
                            "input_stage": input_stage,
                            "classification": represented_input_class(
                                interval,
                                input_rows_identical=input_identical,
                            ),
                            "input_rows_identical": input_identical,
                            "exact_output_energy_enclosure": (
                                rational_interval_record(interval)
                            ),
                            "affine_parameters": affine_parameter_record(
                                normalization
                            ),
                        }
                    if first_stage not in transitions or "S3out" not in transitions:
                        raise RuntimeError(
                            "represented final zero lacks a LayerNorm transition"
                        )
                    row["first_float32_coalescence_stage"] = first_stage
                    row["layernorm_zero_transitions"] = transitions
                rows.append(row)

            direct = np.asarray(
                [row["float32"]["S3out"]["row_centered_energy"] for row in rows],
                dtype=np.float64,
            )
            evaluations.append(
                {
                    "run_id": config["run_id"],
                    "trajectory_id": checkpoint_descriptor["trajectory_id"],
                    "step": checkpoint_descriptor["step"],
                    "device": device_label,
                    "producing_device": checkpoint_descriptor["producing_device"],
                    "recorded_energy_bitwise_match_count": int(
                        sum(
                            observed.hex() == expected.hex()
                            for observed, expected in zip(direct, recorded)
                        )
                    ),
                    "maps": rows,
                }
            )
            del stages32, stages64_network, stages64_pipeline
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
        del checkpoint

    producing = [
        evaluation
        for evaluation in evaluations
        if evaluation["device"]
        == f"cuda:{evaluation['producing_device']}"
    ]
    producing_zero_rows = [
        row
        for evaluation in producing
        for row in evaluation["maps"]
        if row["float32_final_represented_zero"]
    ]
    classifications = Counter(
        row["layernorm_zero_transitions"]["S3out"]["classification"]
        for row in producing_zero_rows
    )
    first_stages = Counter(
        row["first_float32_coalescence_stage"]
        for row in producing_zero_rows
    )
    first_classifications = Counter(
        row["layernorm_zero_transitions"][
            row["first_float32_coalescence_stage"]
        ]["classification"]
        for row in producing_zero_rows
    )
    first_stage_diagnostics = {}
    for output_stage, input_stage in LAYERNORM_INPUT_STAGE.items():
        stage_rows = [
            row
            for row in producing_zero_rows
            if row["first_float32_coalescence_stage"] == output_stage
        ]
        if not stage_rows:
            continue
        transitions = [
            row["layernorm_zero_transitions"][output_stage]
            for row in stage_rows
        ]
        first_stage_diagnostics[output_stage] = {
            "input_stage": input_stage,
            "count": len(stage_rows),
            "represented_input_energy_minimum": min(
                row["float32"][input_stage]["row_centered_energy"]
                for row in stage_rows
            ),
            "represented_input_energy_maximum": max(
                row["float32"][input_stage]["row_centered_energy"]
                for row in stage_rows
            ),
            "exact_output_lower_minimum": min(
                transition["exact_output_energy_enclosure"]["lower_float"]
                for transition in transitions
            ),
            "exact_output_lower_maximum": max(
                transition["exact_output_energy_enclosure"]["lower_float"]
                for transition in transitions
            ),
            "gain_l2_norm_minimum": min(
                transition["affine_parameters"]["gain_l2_norm"]
                for transition in transitions
            ),
            "gain_l2_norm_maximum": max(
                transition["affine_parameters"]["gain_l2_norm"]
                for transition in transitions
            ),
            "bias_l2_norm_minimum": min(
                transition["affine_parameters"]["bias_l2_norm"]
                for transition in transitions
            ),
            "bias_l2_norm_maximum": max(
                transition["affine_parameters"]["bias_l2_norm"]
                for transition in transitions
            ),
        }
    def route_summary(route: str) -> dict[str, Any]:
        diagnostics = [
            row[route]["S3out"]["represented_energy"]
            for row in producing_zero_rows
        ]
        classifications = Counter(
            diagnostic["classification"] for diagnostic in diagnostics
        )
        exact = [
            float(diagnostic["exact_dyadic_energy"]["float"])
            for diagnostic in diagnostics
        ]
        positive = [value for value in exact if value > 0.0]
        rounded = [
            float(row[route]["S3out"]["row_centered_energy"])
            for row in producing_zero_rows
        ]
        return {
            "status": "CLASSIFIED" if diagnostics else "NO_MATCHING_MAPS",
            "represented_zero_count": classifications["REPRESENTED_ZERO"],
            "one_ulp_near_collapse_count": classifications[
                "ONE_ULP_NEAR_COLLAPSE"
            ],
            "positive_beyond_one_ulp_count": classifications[
                "POSITIVE_BEYOND_ONE_ULP"
            ],
            "exact_energy_minimum": min(exact) if exact else None,
            "exact_positive_energy_minimum": min(positive) if positive else None,
            "exact_energy_maximum": max(exact) if exact else None,
            "rounded_centered_sum_minimum": min(rounded) if rounded else None,
            "rounded_centered_sum_maximum": max(rounded) if rounded else None,
        }

    network64_summary = route_summary("float64_whole_network")
    pipeline64_summary = route_summary("float64_from_float32_S0")
    all_network64_final = [
        row["float64_whole_network"]["S3out"]
        for evaluation in producing
        for row in evaluation["maps"]
    ]
    cross_device_groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for evaluation in evaluations:
        key = (evaluation["trajectory_id"], evaluation["step"])
        cross_device_groups.setdefault(key, []).append(evaluation)
    cross_device = []
    for (trajectory_id, step), group in sorted(cross_device_groups.items()):
        zeros_by_device = {
            evaluation["device"]: sum(
                row["float32_final_represented_zero"]
                for row in evaluation["maps"]
            )
            for evaluation in group
        }
        status_by_device = {
            evaluation["device"]: [
                bool(row["float32_final_represented_zero"])
                for row in evaluation["maps"]
            ]
            for evaluation in group
        }
        reference_status = status_by_device[arguments.devices[0]]
        dependent = sum(
            any(
                status_by_device[label][index] != reference_status[index]
                for label in arguments.devices[1:]
            )
            for index in range(len(map_coordinates))
        )
        cross_device.append(
            {
                "trajectory_id": trajectory_id,
                "step": step,
                "represented_zero_count_by_device": zeros_by_device,
                "device_dependent_zero_status_count": dependent,
            }
        )

    payload = {
            "schema_version": "pldr-row-rg-stage-resolved-observer-v4",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "code_commit": commit,
            "analysis_sources": analysis_sources,
            "precision_bits": arguments.precision_bits,
            "arithmetic_policy": {
                "native_network": "float32",
                "whole_network_control": "float64",
                "pipeline_control": (
                    "float64 from the exact represented float32 S0"
                ),
                "rounded_centered_sum": "float64 row centering and summation",
                "represented_energy": (
                    "exact dyadic pairwise energy of each represented float64 array"
                ),
                "near_collapse": "maximum coordinatewise ordered-float span at most one ulp",
                "exact_input_enclosure": (
                    "binary-rational S3in and gain with outward dyadic "
                    "square-root bounds"
                ),
                "tf32_enabled": False,
            },
            "devices": descriptions,
            "architecture": specification,
            "map_ids": [row[3] for row in map_coordinates],
            "inputs": {
                "run_id": config["run_id"],
                "resolved_config": {
                    "path": "config-resolved.json",
                    "sha256": file_sha256(
                        run_root / "config-resolved.json"
                    ),
                },
                "checkpoints": checkpoint_inputs,
                "dataset": {
                    "root_role": "read-only refinedweb dataset",
                    "relative_path": dataset_relative.as_posix(),
                    "size_bytes": dataset_shard.stat().st_size,
                },
                "probe_document_offset": probe_offset,
                "probe_byte_token_sha256": probe_digest,
                "reference_sources": [
                    {
                        "relative_path": name,
                        "sha256": file_sha256(reference_root / name),
                        "size_bytes": (reference_root / name).stat().st_size,
                    }
                    for name in (
                        "pldr_model_v510.py",
                        "power_law_attention_layer_v510.py",
                    )
                ],
            },
            "evaluations": evaluations,
            "cross_device_checkpoint_summary": cross_device,
            "summary": {
                "checkpoint_count": len(checkpoints),
                "device_count": len(arguments.devices),
                "map_count_per_checkpoint": len(map_coordinates),
                "device_map_evaluation_count": (
                    len(evaluations) * len(map_coordinates)
                ),
                "producing_device_final_represented_zero_count": len(
                    producing_zero_rows
                ),
                "producing_device_exact_input_classification_counts": dict(
                    sorted(classifications.items())
                ),
                "producing_device_first_float32_coalescence_stage_counts": {
                    str(key): value
                    for key, value in sorted(first_stages.items())
                },
                "producing_device_first_coalescence_input_classification_counts": dict(
                    sorted(first_classifications.items())
                ),
                "producing_device_first_coalescence_diagnostics": (
                    first_stage_diagnostics
                ),
                "corresponding_float64_whole_network_status": (
                    network64_summary["status"]
                ),
                "corresponding_float64_whole_network_zero_count": (
                    network64_summary["represented_zero_count"]
                ),
                "corresponding_float64_whole_network_one_ulp_count": (
                    network64_summary["one_ulp_near_collapse_count"]
                ),
                "corresponding_float64_whole_network_positive_beyond_one_ulp_count": (
                    network64_summary["positive_beyond_one_ulp_count"]
                ),
                "corresponding_float64_whole_network_energy_minimum": (
                    network64_summary["exact_energy_minimum"]
                ),
                "corresponding_float64_whole_network_positive_energy_minimum": (
                    network64_summary["exact_positive_energy_minimum"]
                ),
                "corresponding_float64_whole_network_energy_maximum": (
                    network64_summary["exact_energy_maximum"]
                ),
                "corresponding_float64_whole_network_rounded_minimum": (
                    network64_summary["rounded_centered_sum_minimum"]
                ),
                "corresponding_float64_from_float32_S0_status": (
                    pipeline64_summary["status"]
                ),
                "corresponding_float64_from_float32_S0_zero_count": (
                    pipeline64_summary["represented_zero_count"]
                ),
                "corresponding_float64_from_float32_S0_one_ulp_count": (
                    pipeline64_summary["one_ulp_near_collapse_count"]
                ),
                "corresponding_float64_from_float32_S0_positive_beyond_one_ulp_count": (
                    pipeline64_summary["positive_beyond_one_ulp_count"]
                ),
                "corresponding_float64_from_float32_S0_energy_minimum": (
                    pipeline64_summary["exact_energy_minimum"]
                ),
                "corresponding_float64_from_float32_S0_positive_energy_minimum": (
                    pipeline64_summary["exact_positive_energy_minimum"]
                ),
                "corresponding_float64_from_float32_S0_energy_maximum": (
                    pipeline64_summary["exact_energy_maximum"]
                ),
                "corresponding_float64_from_float32_S0_rounded_minimum": (
                    pipeline64_summary["rounded_centered_sum_minimum"]
                ),
                "all_producing_device_float64_whole_network_final_zero_count": sum(
                    value["distinct_rows"] == 1 for value in all_network64_final
                ),
                "all_producing_device_float64_whole_network_final_rounded_centered_sum_minimum": min(
                    value["row_centered_energy"] for value in all_network64_final
                ),
                "all_producing_device_float64_whole_network_final_rounded_centered_sum_maximum": max(
                    value["row_centered_energy"] for value in all_network64_final
                ),
                "device_dependent_zero_status_count": sum(
                    row["device_dependent_zero_status_count"]
                    for row in cross_device
                ),
                "interpretation": {
                    "represented_zero_implies_layernorm_reached_real_face": False,
                    "inherited_input_zero_is_separated": True,
                    "whole_network_float64_is_a_numerical_control_not_ground_truth": True,
                    "finite_checkpoint_replay_establishes_asymptotic_phase": False,
                },
            },
        }
    if predecessor is not None:
        assert predecessor_path is not None
        assert predecessor_scientific is not None
        payload["predecessor"] = {
            "record_sha256": predecessor["record_sha256"],
            "file_sha256": file_sha256(predecessor_path),
            "size_bytes": predecessor_path.stat().st_size,
            "scientific_leaves_equal": predecessor_scientific == scientific_payload(payload),
        }
        if not payload["predecessor"]["scientific_leaves_equal"]:
            raise ValueError("stage scientific leaves differ from predecessor")
    record = seal_record(payload)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "record_sha256": record["record_sha256"],
                **record["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
