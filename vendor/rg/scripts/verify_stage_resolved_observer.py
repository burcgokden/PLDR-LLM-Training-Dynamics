#!/usr/bin/env python3
"""Independently verify a source-bound stage-resolved observer record."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from row_rgmap.analysis import file_sha256, verify_record_seal  # noqa: E402
from row_rgmap.provenance_v3 import git_blob_descriptor  # noqa: E402


STAGES = ("S0", "S1", "S2in", "S2out", "S3in", "S3out")
LAYERNORM_INPUT_STAGE = {
    "S1": "S0",
    "S2out": "S2in",
    "S3out": "S3in",
}
CLASSES = {
    "REPRESENTED_INPUT_ZERO_IMAGE",
    "REPRESENTED_INPUT_POSITIVE",
    "REPRESENTED_INPUT_UNRESOLVED",
}


def first_coalescence(stage_row: dict[str, dict[str, Any]]) -> str | None:
    return next(
        (
            stage
            for stage in STAGES
            if int(stage_row[stage]["distinct_rows"]) == 1
        ),
        None,
    )


def replay_exact_route_summary(
    producing_zeros: list[dict[str, Any]], route: str
) -> dict[str, Any]:
    """Reconstruct exact-route aggregates, including the empty case."""

    diagnostics = [
        row[route]["S3out"]["represented_energy"]
        for row in producing_zeros
    ]
    route_classes = Counter(
        diagnostic["classification"] for diagnostic in diagnostics
    )
    exact = [
        float(diagnostic["exact_dyadic_energy"]["float"])
        for diagnostic in diagnostics
    ]
    positive_exact = [value for value in exact if value > 0.0]
    rounded = [
        float(row[route]["S3out"]["row_centered_energy"])
        for row in producing_zeros
    ]
    return {
        "status": "CLASSIFIED" if diagnostics else "NO_MATCHING_MAPS",
        "zero_count": route_classes["REPRESENTED_ZERO"],
        "one_ulp_count": route_classes["ONE_ULP_NEAR_COLLAPSE"],
        "positive_count": route_classes["POSITIVE_BEYOND_ONE_ULP"],
        "minimum": min(exact) if exact else None,
        "positive_minimum": min(positive_exact) if positive_exact else None,
        "maximum": max(exact) if exact else None,
        "rounded_minimum": min(rounded) if rounded else None,
    }


def verify(
    record_path: Path,
    run_root: Path,
    reference_root: Path,
) -> dict[str, Any]:
    record = json.loads(record_path.read_text(encoding="utf-8"))
    verify_record_seal(record)
    schema = record.get("schema_version")
    if schema not in {
        "pldr-row-rg-stage-resolved-observer-v2",
        "pldr-row-rg-stage-resolved-observer-v3",
        "pldr-row-rg-stage-resolved-observer-v4",
    }:
        raise ValueError("unexpected stage-resolved record schema")
    commit = record.get("code_commit")
    sources = record.get("analysis_sources")
    if not isinstance(commit, str) or not isinstance(sources, list) or not sources:
        raise ValueError("analysis source registry is incomplete")
    for source in sources:
        repository_path = source.get("repository_path")
        if not isinstance(repository_path, str):
            raise ValueError("analysis source is not repository-relative")
        digest, size = git_blob_descriptor(ROOT, commit, repository_path)
        if (
            source.get("code_commit") != commit
            or source.get("sha256") != digest
            or source.get("size_bytes") != size
        ):
            raise ValueError(f"analysis source mismatch: {repository_path}")

    inputs = record["inputs"]
    config_path = run_root / inputs["resolved_config"]["path"]
    if file_sha256(config_path) != inputs["resolved_config"]["sha256"]:
        raise ValueError("resolved configuration digest mismatch")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("run_id") != inputs.get("run_id"):
        raise ValueError("run identifier mismatch")
    checkpoints = inputs.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("checkpoint registry is empty")
    checkpoint_keys = set()
    for descriptor in checkpoints:
        key = (descriptor["trajectory_id"], int(descriptor["step"]))
        if key in checkpoint_keys:
            raise ValueError(f"duplicate checkpoint descriptor: {key}")
        checkpoint_keys.add(key)
        path = (run_root / descriptor["path"]).resolve()
        try:
            path.relative_to(run_root.resolve())
        except ValueError as error:
            raise ValueError("checkpoint path escapes the run root") from error
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or file_sha256(path) != descriptor["sha256"]
        ):
            raise ValueError(f"checkpoint descriptor mismatch: {key}")
    for descriptor in inputs.get("reference_sources", []):
        path = reference_root / descriptor["relative_path"]
        if (
            not path.is_file()
            or path.stat().st_size != descriptor["size_bytes"]
            or file_sha256(path) != descriptor["sha256"]
        ):
            raise ValueError(
                f"reference source mismatch: {descriptor['relative_path']}"
            )

    map_ids = record.get("map_ids")
    evaluations = record.get("evaluations")
    devices = [row["label"] for row in record.get("devices", [])]
    if (
        not isinstance(map_ids, list)
        or len(map_ids) != len(set(map_ids))
        or not isinstance(evaluations, list)
        or not devices
    ):
        raise ValueError("map, evaluation, or device registry is invalid")
    expected_evaluations = {
        (trajectory, step, device)
        for trajectory, step in checkpoint_keys
        for device in devices
    }
    observed_evaluations = {
        (
            row["trajectory_id"],
            int(row["step"]),
            row["device"],
        )
        for row in evaluations
    }
    if observed_evaluations != expected_evaluations:
        raise ValueError("evaluation registry is incomplete or duplicated")

    producing = []
    for evaluation in evaluations:
        rows = evaluation.get("maps")
        if (
            not isinstance(rows, list)
            or [row.get("map_id") for row in rows] != map_ids
            or [row.get("map_index") for row in rows] != list(range(len(map_ids)))
        ):
            raise ValueError("evaluation map order differs from its registry")
        for row in rows:
            for route in ("float32", "float64_whole_network", "float64_from_float32_S0"):
                stages = row.get(route)
                if not isinstance(stages, dict) or set(stages) != set(STAGES):
                    raise ValueError("stage registry is incomplete")
                for stage in STAGES:
                    distinct = stages[stage].get("distinct_rows")
                    energy = stages[stage].get("row_centered_energy")
                    if (
                        not isinstance(distinct, int)
                        or not 1 <= distinct <= 64
                        or not isinstance(energy, (int, float))
                        or not math.isfinite(float(energy))
                        or float(energy) < 0.0
                    ):
                        raise ValueError("stage statistic is invalid")
            represented_zero = bool(row["float32_final_represented_zero"])
            final = row["float32"]["S3out"]
            if represented_zero != (
                final["distinct_rows"] == 1
                and float(final["row_centered_energy"]) == 0.0
            ):
                raise ValueError("represented-zero flag is inconsistent")
            if (
                schema == "pldr-row-rg-stage-resolved-observer-v3"
                and represented_zero
            ):
                for route in (
                    "float64_whole_network",
                    "float64_from_float32_S0",
                ):
                    final64 = row[route]["S3out"]
                    diagnostic = final64.get("represented_energy", {})
                    dyadic = diagnostic.get("exact_dyadic_energy", {})
                    try:
                        numerator = int(dyadic["numerator"])
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError("invalid exact dyadic numerator") from error
                    exponent = dyadic.get("power_of_two_exponent")
                    ulp_span = diagnostic.get("maximum_coordinate_ulp_span")
                    if (
                        numerator < 0
                        or not isinstance(exponent, int)
                        or not isinstance(ulp_span, int)
                        or ulp_span < 0
                    ):
                        raise ValueError("invalid represented-energy diagnostic")
                    exact = float(
                        Fraction(numerator, 1 << -exponent)
                        if exponent < 0
                        else Fraction(numerator << exponent, 1)
                    )
                    if dyadic.get("float") != exact:
                        raise ValueError("exact dyadic float projection is inconsistent")
                    expected_class = (
                        "REPRESENTED_ZERO"
                        if numerator == 0
                        else "ONE_ULP_NEAR_COLLAPSE"
                        if ulp_span <= 1
                        else "POSITIVE_BEYOND_ONE_ULP"
                    )
                    if diagnostic.get("classification") != expected_class:
                        raise ValueError("represented-energy classification is inconsistent")
                    if (numerator == 0) != (final64["distinct_rows"] == 1):
                        raise ValueError("exact energy and represented rows disagree")
            transitions = row.get("layernorm_zero_transitions")
            if represented_zero:
                first_stage = first_coalescence(row["float32"])
                if row.get("first_float32_coalescence_stage") != first_stage:
                    raise ValueError("first coalescence stage is inconsistent")
                expected_transition_stages = {
                    stage
                    for stage in LAYERNORM_INPUT_STAGE
                    if row["float32"][stage]["distinct_rows"] == 1
                }
                if (
                    not isinstance(transitions, dict)
                    or set(transitions) != expected_transition_stages
                    or first_stage not in transitions
                    or "S3out" not in transitions
                ):
                    raise ValueError("LayerNorm transition registry is incomplete")
                for output_stage, transition in transitions.items():
                    input_stage = LAYERNORM_INPUT_STAGE[output_stage]
                    if transition.get("input_stage") != input_stage:
                        raise ValueError("LayerNorm input stage is inconsistent")
                    classification = transition.get("classification")
                    if classification not in CLASSES:
                        raise ValueError("unknown exact-input classification")
                    input_identical = (
                        row["float32"][input_stage]["distinct_rows"] == 1
                    )
                    if transition.get("input_rows_identical") != input_identical:
                        raise ValueError("input-row identity flag is inconsistent")
                    enclosure = transition.get(
                        "exact_output_energy_enclosure", {}
                    )
                    lower = enclosure.get("lower_float")
                    upper = enclosure.get("upper_float")
                    if (
                        not isinstance(lower, (int, float))
                        or not isinstance(upper, (int, float))
                        or not math.isfinite(float(lower))
                        or not math.isfinite(float(upper))
                        or lower < 0.0
                        or upper < lower
                        or enclosure.get("lower_positive") != (lower > 0.0)
                        or enclosure.get("upper_zero") != (upper == 0.0)
                    ):
                        raise ValueError("exact-output enclosure is invalid")
                    if classification == "REPRESENTED_INPUT_ZERO_IMAGE":
                        if not input_identical or not enclosure.get("upper_zero"):
                            raise ValueError(
                                "inherited zero classification is invalid"
                            )
                    elif classification == "REPRESENTED_INPUT_POSITIVE":
                        if input_identical or not enclosure.get("lower_positive"):
                            raise ValueError(
                                "positive-input classification is invalid"
                            )
                    elif input_identical or enclosure.get("lower_positive"):
                        raise ValueError("unresolved classification is invalid")
                    parameters = transition.get("affine_parameters", {})
                    numeric_parameters = (
                        "epsilon",
                        "gain_l2_norm",
                        "gain_linf_norm",
                        "bias_l2_norm",
                        "bias_linf_norm",
                    )
                    if any(
                        not isinstance(parameters.get(name), (int, float))
                        or not math.isfinite(float(parameters[name]))
                        or float(parameters[name]) < 0.0
                        for name in numeric_parameters
                    ):
                        raise ValueError("affine-parameter statistic is invalid")
                    if parameters["epsilon"] <= 0.0:
                        raise ValueError("LayerNorm epsilon must be positive")
                    if parameters.get("gain_all_zero") != (
                        parameters["gain_l2_norm"] == 0.0
                    ):
                        raise ValueError("zero-gain flag is inconsistent")
            elif transitions is not None or row.get(
                "first_float32_coalescence_stage"
            ) is not None:
                raise ValueError("nonzero output has unexpected transition data")
        if evaluation["device"] == f"cuda:{evaluation['producing_device']}":
            producing.append(evaluation)

    producing_zeros = [
        row
        for evaluation in producing
        for row in evaluation["maps"]
        if row["float32_final_represented_zero"]
    ]
    classifications = Counter(
        row["layernorm_zero_transitions"]["S3out"]["classification"]
        for row in producing_zeros
    )
    first_stages = Counter(
        row["first_float32_coalescence_stage"]
        for row in producing_zeros
    )
    first_classifications = Counter(
        row["layernorm_zero_transitions"][
            row["first_float32_coalescence_stage"]
        ]["classification"]
        for row in producing_zeros
    )
    first_stage_diagnostics = {}
    for output_stage, input_stage in LAYERNORM_INPUT_STAGE.items():
        stage_rows = [
            row
            for row in producing_zeros
            if row["first_float32_coalescence_stage"] == output_stage
        ]
        if not stage_rows:
            continue
        stage_transitions = [
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
                for transition in stage_transitions
            ),
            "exact_output_lower_maximum": max(
                transition["exact_output_energy_enclosure"]["lower_float"]
                for transition in stage_transitions
            ),
            "gain_l2_norm_minimum": min(
                transition["affine_parameters"]["gain_l2_norm"]
                for transition in stage_transitions
            ),
            "gain_l2_norm_maximum": max(
                transition["affine_parameters"]["gain_l2_norm"]
                for transition in stage_transitions
            ),
            "bias_l2_norm_minimum": min(
                transition["affine_parameters"]["bias_l2_norm"]
                for transition in stage_transitions
            ),
            "bias_l2_norm_maximum": max(
                transition["affine_parameters"]["bias_l2_norm"]
                for transition in stage_transitions
            ),
        }
    def rounded_route_values(route: str) -> list[float]:
        return [
            float(row[route]["S3out"]["row_centered_energy"])
            for row in producing_zeros
        ]

    network64 = rounded_route_values("float64_whole_network")
    pipeline64 = rounded_route_values("float64_from_float32_S0")
    summary = record["summary"]
    expected_summary = {
        "checkpoint_count": len(checkpoints),
        "device_count": len(devices),
        "map_count_per_checkpoint": len(map_ids),
        "device_map_evaluation_count": len(evaluations) * len(map_ids),
        "producing_device_final_represented_zero_count": len(producing_zeros),
        "producing_device_exact_input_classification_counts": dict(
            sorted(classifications.items())
        ),
        "producing_device_first_float32_coalescence_stage_counts": {
            str(key): value for key, value in sorted(first_stages.items())
        },
        "producing_device_first_coalescence_input_classification_counts": dict(
            sorted(first_classifications.items())
        ),
        "producing_device_first_coalescence_diagnostics": (
            first_stage_diagnostics
        ),
    }
    if schema == "pldr-row-rg-stage-resolved-observer-v2":
        expected_summary.update(
            {
                "corresponding_float64_whole_network_zero_count": sum(
                    value == 0.0 for value in network64
                ),
                "corresponding_float64_whole_network_energy_minimum": min(
                    network64
                ),
                "corresponding_float64_whole_network_energy_maximum": max(
                    network64
                ),
                "corresponding_float64_from_float32_S0_zero_count": sum(
                    value == 0.0 for value in pipeline64
                ),
                "corresponding_float64_from_float32_S0_energy_minimum": min(
                    pipeline64
                ),
                "corresponding_float64_from_float32_S0_energy_maximum": max(
                    pipeline64
                ),
            }
        )
    else:
        for prefix, route in (
            (
                "corresponding_float64_whole_network",
                "float64_whole_network",
            ),
            (
                "corresponding_float64_from_float32_S0",
                "float64_from_float32_S0",
            ),
        ):
            replay = replay_exact_route_summary(producing_zeros, route)
            expected_summary.update(
                {
                    f"{prefix}_status": replay["status"],
                    f"{prefix}_zero_count": replay["zero_count"],
                    f"{prefix}_one_ulp_count": replay["one_ulp_count"],
                    f"{prefix}_positive_beyond_one_ulp_count": replay[
                        "positive_count"
                    ],
                    f"{prefix}_energy_minimum": replay["minimum"],
                    f"{prefix}_positive_energy_minimum": replay[
                        "positive_minimum"
                    ],
                    f"{prefix}_energy_maximum": replay["maximum"],
                    f"{prefix}_rounded_minimum": replay["rounded_minimum"],
                }
            )
        if schema == "pldr-row-rg-stage-resolved-observer-v4":
            all_network64_final = [
                row["float64_whole_network"]["S3out"]
                for evaluation in producing
                for row in evaluation["maps"]
            ]
            expected_summary.update(
                {
                    "all_producing_device_float64_whole_network_final_zero_count": sum(
                        value["distinct_rows"] == 1
                        for value in all_network64_final
                    ),
                    "all_producing_device_float64_whole_network_final_rounded_centered_sum_minimum": min(
                        value["row_centered_energy"]
                        for value in all_network64_final
                    ),
                    "all_producing_device_float64_whole_network_final_rounded_centered_sum_maximum": max(
                        value["row_centered_energy"]
                        for value in all_network64_final
                    ),
                }
            )
            predecessor = record.get("predecessor")
            if predecessor is not None and (
                not isinstance(predecessor, dict)
                or predecessor.get("scientific_leaves_equal") is not True
                or not isinstance(predecessor.get("record_sha256"), str)
                or len(predecessor["record_sha256"]) != 64
                or not isinstance(predecessor.get("file_sha256"), str)
                or len(predecessor["file_sha256"]) != 64
                or not isinstance(predecessor.get("size_bytes"), int)
                or predecessor["size_bytes"] <= 0
            ):
                raise ValueError("stage predecessor descriptor is invalid")
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(f"summary field does not replay: {key}")
    return {
        "record_sha256": record["record_sha256"],
        "code_commit": commit,
        "analysis_source_count": len(sources),
        **expected_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--completed-run-root", required=True)
    parser.add_argument("--reference-root", required=True)
    arguments = parser.parse_args()
    result = verify(
        Path(arguments.record).resolve(),
        Path(arguments.completed_run_root).resolve(),
        Path(arguments.reference_root).resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
