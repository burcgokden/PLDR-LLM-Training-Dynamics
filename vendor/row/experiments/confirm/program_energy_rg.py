"""Canonical blocking and discrete RG flow for owned energy comparisons."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


RG_EDGE_SCHEMA_VERSION = "pldr-program-energy-rg-edge-v1"
RG_FLOW_SCHEMA_VERSION = "pldr-program-energy-rg-flow-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest_object(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _matrix(value, name):
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or result.shape[0] == 0
        or not np.isfinite(result).all()
        or np.any(result < 0)
    ):
        raise ValueError(f"{name} must be a finite nonnegative square matrix")
    return result


def _vector(value, dimension, name, *, positive=False):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (dimension,)
        or not np.isfinite(result).all()
        or np.any(result <= 0 if positive else result < 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(
            f"{name} must be a finite {qualifier} vector of length {dimension}")
    return result


def _outward_matmul(left, right):
    """Conservative binary64 upper product for nonnegative matrices."""

    dimension = left.shape[1]
    raw = left @ right
    magnitude = np.abs(left) @ np.abs(right)
    unit = np.finfo(float).eps
    gamma = (dimension + 2) * unit
    if gamma >= 1:
        raise ValueError("matrix dimension is too large for the RG roundoff rule")
    upper = raw + gamma * magnitude
    return np.where(
        magnitude == 0.0, 0.0, np.nextafter(upper, np.inf))


def _outward_matvec(matrix, vector):
    dimension = matrix.shape[1]
    raw = matrix @ vector
    magnitude = np.abs(matrix) @ np.abs(vector)
    unit = np.finfo(float).eps
    gamma = (dimension + 2) * unit
    upper = raw + gamma * magnitude
    return np.where(
        magnitude == 0.0, 0.0, np.nextafter(upper, np.inf))


def _outward_add(left, right):
    raw = left + right
    magnitude = np.abs(left) + np.abs(right)
    upper = raw + 2.0 * np.finfo(float).eps * magnitude
    return np.where(
        magnitude == 0.0, 0.0, np.nextafter(upper, np.inf))


def _compose_components(later_operator, later_forcing,
                        earlier_operator, earlier_forcing):
    operator = _outward_matmul(later_operator, earlier_operator)
    forcing = _outward_add(
        _outward_matvec(later_operator, earlier_forcing),
        later_forcing,
    )
    return operator, forcing


def _normalize(operator, forcing, source_gauge, target_gauge):
    normalized_operator = (
        operator * source_gauge[np.newaxis, :]
        / target_gauge[:, np.newaxis]
    )
    normalized_forcing = forcing / target_gauge
    normalized_operator = np.where(
        operator == 0.0, 0.0,
        np.nextafter(normalized_operator, np.inf),
    )
    normalized_forcing = np.where(
        forcing == 0.0, 0.0,
        np.nextafter(normalized_forcing, np.inf),
    )
    return normalized_operator, normalized_forcing


def make_rg_edge(
    *,
    edge_id,
    start_index,
    source_state_sha256,
    target_state_sha256,
    comparison_owner_sha256,
    operator_upper,
    forcing_upper,
    source_gauge,
    target_gauge,
):
    """Construct one canonical RG input from a constructor-owned comparison."""

    operator = _matrix(operator_upper, "operator_upper")
    dimension = operator.shape[0]
    forcing = _vector(
        forcing_upper, dimension, "forcing_upper")
    source = _vector(
        source_gauge, dimension, "source_gauge", positive=True)
    target = _vector(
        target_gauge, dimension, "target_gauge", positive=True)
    if not isinstance(edge_id, str) or not edge_id:
        raise ValueError("edge_id must be a nonempty string")
    if not isinstance(start_index, int) or start_index < 0:
        raise ValueError("start_index must be a nonnegative integer")
    for name, value in (
        ("source_state_sha256", source_state_sha256),
        ("target_state_sha256", target_state_sha256),
        ("comparison_owner_sha256", comparison_owner_sha256),
    ):
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    record = {
        "schema_version": RG_EDGE_SCHEMA_VERSION,
        "edge_id": edge_id,
        "start_index": start_index,
        "stop_index": start_index + 1,
        "source_state_sha256": source_state_sha256,
        "target_state_sha256": target_state_sha256,
        "comparison_owner_sha256": comparison_owner_sha256,
        "operator_upper": operator.tolist(),
        "forcing_upper": forcing.tolist(),
        "source_gauge": source.tolist(),
        "target_gauge": target.tolist(),
    }
    record["record_sha256"] = digest_object(record)
    return record


def _edge_arrays(edge):
    required = {
        "schema_version", "edge_id", "start_index", "stop_index",
        "source_state_sha256", "target_state_sha256",
        "comparison_owner_sha256", "operator_upper", "forcing_upper",
        "source_gauge", "target_gauge", "record_sha256",
    }
    if not isinstance(edge, dict) or set(edge) != required:
        raise ValueError("RG edge has missing or unknown fields")
    unsigned = dict(edge)
    digest = unsigned.pop("record_sha256")
    if digest != digest_object(unsigned):
        raise ValueError("RG edge digest does not replay")
    if edge["schema_version"] != RG_EDGE_SCHEMA_VERSION:
        raise ValueError("unknown RG edge schema")
    if (
        not isinstance(edge["start_index"], int)
        or edge["stop_index"] != edge["start_index"] + 1
    ):
        raise ValueError("RG edge indices are malformed")
    for name in (
        "source_state_sha256", "target_state_sha256",
        "comparison_owner_sha256",
    ):
        if not HEX64.fullmatch(edge[name]):
            raise ValueError("RG edge contains an invalid digest")
    operator = _matrix(edge["operator_upper"], "operator_upper")
    dimension = operator.shape[0]
    return {
        "operator": operator,
        "forcing": _vector(
            edge["forcing_upper"], dimension, "forcing_upper"),
        "source_gauge": _vector(
            edge["source_gauge"], dimension, "source_gauge", positive=True),
        "target_gauge": _vector(
            edge["target_gauge"], dimension, "target_gauge", positive=True),
    }


def _validate_chain(edges):
    if not isinstance(edges, list) or not edges:
        raise ValueError("RG flow requires a nonempty edge list")
    arrays = [_edge_arrays(edge) for edge in edges]
    dimension = arrays[0]["operator"].shape[0]
    for index, (edge, values) in enumerate(zip(edges, arrays)):
        if values["operator"].shape != (dimension, dimension):
            raise ValueError("RG edge dimensions disagree")
        if index == 0:
            continue
        previous = edges[index - 1]
        previous_values = arrays[index - 1]
        if edge["start_index"] != previous["stop_index"]:
            raise ValueError("RG edge indices are not contiguous")
        if edge["source_state_sha256"] != previous["target_state_sha256"]:
            raise ValueError("RG state-digest chain is broken")
        if not np.array_equal(
            values["source_gauge"], previous_values["target_gauge"]
        ):
            raise ValueError("RG gauge chain is broken")
    return arrays


def _comparison_tolerance(*values):
    scale = max(
        1.0,
        *(float(np.max(np.abs(value))) for value in values if value.size),
    )
    dimension = max(value.size for value in values)
    return float(512.0 * np.finfo(float).eps * max(1, dimension) * scale)


def _block(edges, arrays):
    operator = arrays[0]["operator"]
    forcing = arrays[0]["forcing"]
    for values in arrays[1:]:
        operator, forcing = _compose_components(
            values["operator"], values["forcing"], operator, forcing)
    source_gauge = arrays[0]["source_gauge"]
    target_gauge = arrays[-1]["target_gauge"]
    normalized_operator, normalized_forcing = _normalize(
        operator, forcing, source_gauge, target_gauge)

    composed_normalized_operator, composed_normalized_forcing = _normalize(
        arrays[0]["operator"], arrays[0]["forcing"],
        arrays[0]["source_gauge"], arrays[0]["target_gauge"],
    )
    for values in arrays[1:]:
        later_operator, later_forcing = _normalize(
            values["operator"], values["forcing"],
            values["source_gauge"], values["target_gauge"],
        )
        composed_normalized_operator, composed_normalized_forcing = (
            _compose_components(
                later_operator, later_forcing,
                composed_normalized_operator, composed_normalized_forcing,
            )
        )
    covariance_residual = max(
        float(np.max(np.abs(
            normalized_operator - composed_normalized_operator))),
        float(np.max(np.abs(
            normalized_forcing - composed_normalized_forcing))),
    )
    tolerance = _comparison_tolerance(
        normalized_operator, normalized_forcing,
        composed_normalized_operator, composed_normalized_forcing,
    )
    gain = float(np.max(np.sum(normalized_operator, axis=1)))
    gain = 0.0 if gain == 0.0 else math.nextafter(gain, math.inf)
    duration = edges[-1]["stop_index"] - edges[0]["start_index"]
    log_gain_rate = None if gain == 0.0 else math.log(gain) / duration
    return {
        "start_index": edges[0]["start_index"],
        "stop_index": edges[-1]["stop_index"],
        "duration": duration,
        "source_state_sha256": edges[0]["source_state_sha256"],
        "target_state_sha256": edges[-1]["target_state_sha256"],
        "constituent_edge_sha256": [
            edge["record_sha256"] for edge in edges
        ],
        "operator_upper": operator.tolist(),
        "forcing_upper": forcing.tolist(),
        "source_gauge": source_gauge.tolist(),
        "target_gauge": target_gauge.tolist(),
        "normalized_operator_upper": normalized_operator.tolist(),
        "normalized_forcing_upper": normalized_forcing.tolist(),
        "weighted_gain_upper": gain,
        "log_gain_per_fine_step_upper": log_gain_rate,
        "gauge_covariance_residual": covariance_residual,
        "gauge_covariance_tolerance": tolerance,
    }


def _raw_order_sensitivity(first, second):
    first_values, second_values = _validate_chain([first, second])
    forward_operator, forward_forcing = _compose_components(
        second_values["operator"], second_values["forcing"],
        first_values["operator"], first_values["forcing"],
    )
    reverse_operator, reverse_forcing = _compose_components(
        first_values["operator"], first_values["forcing"],
        second_values["operator"], second_values["forcing"],
    )
    return max(
        float(np.max(np.abs(forward_operator - reverse_operator))),
        float(np.max(np.abs(forward_forcing - reverse_forcing))),
    )


def construct_rg_flow(edges, block_factor=2):
    """Block an aligned owned edge path and serialize its discrete RG flow."""

    if not isinstance(block_factor, int) or block_factor < 2:
        raise ValueError("block_factor must be an integer at least two")
    arrays = _validate_chain(edges)
    count = len(edges)
    power = count
    while power > 1 and power % block_factor == 0:
        power //= block_factor
    if power != 1:
        raise ValueError(
            "RG edge count must be an exact power of the block factor")

    levels = []
    block_size = 1
    while block_size <= count:
        blocks = []
        for start in range(0, count, block_size):
            stop = start + block_size
            blocks.append(_block(edges[start:stop], arrays[start:stop]))
        levels.append({
            "level": len(levels),
            "block_size": block_size,
            "block_count": len(blocks),
            "blocks": blocks,
        })
        block_size *= block_factor

    semigroup_residual = 0.0
    semigroup_tolerance = 0.0
    for level_index in range(1, len(levels)):
        previous = levels[level_index - 1]["blocks"]
        current = levels[level_index]["blocks"]
        for block_index, direct in enumerate(current):
            subblocks = previous[
                block_index * block_factor:
                (block_index + 1) * block_factor
            ]
            operator = np.asarray(subblocks[0]["operator_upper"], dtype=float)
            forcing = np.asarray(subblocks[0]["forcing_upper"], dtype=float)
            for later in subblocks[1:]:
                operator, forcing = _compose_components(
                    np.asarray(later["operator_upper"], dtype=float),
                    np.asarray(later["forcing_upper"], dtype=float),
                    operator, forcing,
                )
            direct_operator = np.asarray(
                direct["operator_upper"], dtype=float)
            direct_forcing = np.asarray(
                direct["forcing_upper"], dtype=float)
            residual = max(
                float(np.max(np.abs(direct_operator - operator))),
                float(np.max(np.abs(direct_forcing - forcing))),
            )
            tolerance = _comparison_tolerance(
                direct_operator, direct_forcing, operator, forcing)
            semigroup_residual = max(semigroup_residual, residual)
            semigroup_tolerance = max(semigroup_tolerance, tolerance)

    gauge_residual = max(
        block["gauge_covariance_residual"]
        for level in levels for block in level["blocks"]
    )
    gauge_tolerance = max(
        block["gauge_covariance_tolerance"]
        for level in levels for block in level["blocks"]
    )
    order_sensitivity = (
        _raw_order_sensitivity(edges[0], edges[1])
        if len(edges) >= 2 else 0.0
    )
    conditions = {
        "owned_digest_chain": True,
        "positive_gauge_chain": True,
        "aligned_complete_blocking": levels[-1]["block_count"] == 1,
        "affine_semigroup_enclosed": bool(
            semigroup_residual <= semigroup_tolerance),
        "gauge_covariance_enclosed": bool(
            gauge_residual <= gauge_tolerance),
        "temporal_order_serialized": all(
            edge["start_index"] == edges[0]["start_index"] + index
            for index, edge in enumerate(edges)
        ),
    }
    record = {
        "schema_version": RG_FLOW_SCHEMA_VERSION,
        "block_factor": block_factor,
        "fine_edges": edges,
        "levels": levels,
        "diagnostics": {
            "semigroup_residual": semigroup_residual,
            "semigroup_tolerance": semigroup_tolerance,
            "gauge_covariance_residual": gauge_residual,
            "gauge_covariance_tolerance": gauge_tolerance,
            "first_pair_order_sensitivity": order_sensitivity,
        },
        "conditions": conditions,
        "decision": "QUALIFIED" if all(conditions.values()) else "REJECTED",
    }
    record["record_sha256"] = digest_object(record)
    return record


def q0_rg_fixture():
    """Four binary-rational noncommuting edges for the RG algebra gate."""

    operators = (
        [[0.5, 0.25], [0.0, 0.5]],
        [[0.25, 0.0], [0.5, 0.5]],
        [[0.5, 0.0], [0.25, 0.25]],
        [[0.25, 0.5], [0.0, 0.5]],
    )
    forcings = (
        [0.125, 0.25],
        [0.25, 0.125],
        [0.0625, 0.125],
        [0.125, 0.0625],
    )
    gauges = (
        [1.0, 2.0],
        [2.0, 1.0],
        [1.0, 4.0],
        [4.0, 2.0],
        [2.0, 8.0],
    )
    states = [
        hashlib.sha256(f"q0-rg-state-{index}".encode()).hexdigest()
        for index in range(5)
    ]
    owner = hashlib.sha256(b"q0-rg-owned-comparison").hexdigest()
    return [
        make_rg_edge(
            edge_id=f"q0-rg-edge-{index}",
            start_index=index,
            source_state_sha256=states[index],
            target_state_sha256=states[index + 1],
            comparison_owner_sha256=owner,
            operator_upper=operators[index],
            forcing_upper=forcings[index],
            source_gauge=gauges[index],
            target_gauge=gauges[index + 1],
        )
        for index in range(4)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edges", required=True)
    parser.add_argument("--block-factor", type=int, default=2)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    with Path(arguments.edges).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    edges = payload.get("fine_edges") if isinstance(payload, dict) else payload
    flow = construct_rg_flow(edges, arguments.block_factor)
    serialized = json.dumps(flow, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    if flow["decision"] != "QUALIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
