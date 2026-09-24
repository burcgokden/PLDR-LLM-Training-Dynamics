#!/usr/bin/env python3
"""Independent replay for an owned affine RG-flow record.

This checker deliberately does not import ``program_energy_rg``.  It rebuilds
ordered affine blocks, positive-gauge normalization, gains, the aligned
semigroup comparison, and all serialized digests from the record alone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


EDGE_SCHEMA = "pldr-program-energy-rg-edge-v1"
FLOW_SCHEMA = "pldr-program-energy-rg-flow-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _close(left, right, tolerance):
    return bool(np.max(np.abs(left - right)) <= tolerance)


def _comparison_tolerance(*values):
    scale = max(
        1.0,
        *(float(np.max(np.abs(value))) for value in values if value.size),
    )
    dimension = max(value.size for value in values)
    return float(
        512.0 * np.finfo(float).eps * max(1, dimension) * scale)


def _matrix(value, dimension=None):
    result = np.asarray(value, dtype=float)
    if (
        result.ndim != 2
        or result.shape[0] != result.shape[1]
        or result.shape[0] == 0
        or (dimension is not None and result.shape != (dimension, dimension))
        or not np.isfinite(result).all()
        or np.any(result < 0)
    ):
        raise ValueError("independent RG replay found an invalid operator")
    return result


def _vector(value, dimension, *, positive=False):
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (dimension,)
        or not np.isfinite(result).all()
        or np.any(result <= 0 if positive else result < 0)
    ):
        raise ValueError("independent RG replay found an invalid vector")
    return result


def _edge_arrays(edge, dimension=None):
    required = {
        "schema_version", "edge_id", "start_index", "stop_index",
        "source_state_sha256", "target_state_sha256",
        "comparison_owner_sha256", "operator_upper", "forcing_upper",
        "source_gauge", "target_gauge", "record_sha256",
    }
    if not isinstance(edge, dict) or set(edge) != required:
        raise ValueError("independent RG replay found a malformed fine edge")
    unsigned = dict(edge)
    digest = unsigned.pop("record_sha256")
    if digest != _digest(unsigned):
        raise ValueError("independent RG fine-edge digest does not replay")
    if edge["schema_version"] != EDGE_SCHEMA:
        raise ValueError("independent RG replay found an unknown edge schema")
    if (
        not isinstance(edge["edge_id"], str)
        or not edge["edge_id"]
        or not isinstance(edge["start_index"], int)
        or edge["stop_index"] != edge["start_index"] + 1
    ):
        raise ValueError("independent RG replay found malformed edge ownership")
    for name in (
        "source_state_sha256", "target_state_sha256",
        "comparison_owner_sha256",
    ):
        if not isinstance(edge[name], str) or not HEX64.fullmatch(edge[name]):
            raise ValueError("independent RG replay found an invalid digest")
    operator = _matrix(edge["operator_upper"], dimension)
    size = operator.shape[0]
    return {
        "operator": operator,
        "forcing": _vector(edge["forcing_upper"], size),
        "source_gauge": _vector(
            edge["source_gauge"], size, positive=True),
        "target_gauge": _vector(
            edge["target_gauge"], size, positive=True),
    }


def _compose(later_operator, later_forcing,
             earlier_operator, earlier_forcing):
    return (
        later_operator @ earlier_operator,
        later_operator @ earlier_forcing + later_forcing,
    )


def _direct_block(edge_rows, array_rows):
    operator = array_rows[0]["operator"]
    forcing = array_rows[0]["forcing"]
    for values in array_rows[1:]:
        operator, forcing = _compose(
            values["operator"], values["forcing"], operator, forcing)
    source_gauge = array_rows[0]["source_gauge"]
    target_gauge = array_rows[-1]["target_gauge"]
    normalized_operator = (
        operator * source_gauge[np.newaxis, :]
        / target_gauge[:, np.newaxis]
    )
    normalized_forcing = forcing / target_gauge
    gain = float(np.max(np.sum(normalized_operator, axis=1)))
    duration = edge_rows[-1]["stop_index"] - edge_rows[0]["start_index"]
    rate = None if gain == 0.0 else math.log(gain) / duration
    return operator, forcing, normalized_operator, normalized_forcing, gain, rate


def _scalar_close(left, right, tolerance):
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def check_rg_flow(record):
    """Replay a serialized RG flow without trusting constructor decisions."""

    required = {
        "schema_version", "block_factor", "fine_edges", "levels",
        "diagnostics", "conditions", "decision", "record_sha256",
    }
    if not isinstance(record, dict) or set(record) != required:
        raise ValueError("RG flow has missing or unknown fields")
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256")
    if digest != _digest(unsigned):
        raise ValueError("RG flow digest does not replay")
    if record["schema_version"] != FLOW_SCHEMA:
        raise ValueError("unknown RG-flow schema")
    factor = record["block_factor"]
    edges = record["fine_edges"]
    if not isinstance(factor, int) or factor < 2 or not isinstance(edges, list):
        raise ValueError("independent RG replay found an invalid flow domain")
    if not edges:
        raise ValueError("independent RG replay found an empty flow")
    first = _edge_arrays(edges[0])
    dimension = first["operator"].shape[0]
    arrays = [first] + [
        _edge_arrays(edge, dimension) for edge in edges[1:]
    ]
    for index in range(1, len(edges)):
        if (
            edges[index]["start_index"] != edges[index - 1]["stop_index"]
            or edges[index]["source_state_sha256"]
                != edges[index - 1]["target_state_sha256"]
            or not np.array_equal(
                arrays[index]["source_gauge"],
                arrays[index - 1]["target_gauge"],
            )
        ):
            raise ValueError("independent RG replay found a broken owned chain")

    expected_sizes = []
    size = 1
    while size <= len(edges):
        if len(edges) % size != 0:
            raise ValueError("independent RG replay found incomplete blocking")
        expected_sizes.append(size)
        size *= factor
    if expected_sizes[-1] != len(edges):
        raise ValueError("fine-edge count is not a power of the block factor")
    levels = record["levels"]
    if not isinstance(levels, list) or len(levels) != len(expected_sizes):
        raise ValueError("independent RG replay found incomplete levels")

    block_checks = []
    covariance_residuals = []
    covariance_tolerances = []
    computed_blocks = []
    for level_index, (level, block_size) in enumerate(
        zip(levels, expected_sizes)
    ):
        required_level = {
            "level", "block_size", "block_count", "blocks",
        }
        if not isinstance(level, dict) or set(level) != required_level:
            raise ValueError("independent RG replay found a malformed level")
        expected_count = len(edges) // block_size
        if (
            level["level"] != level_index
            or level["block_size"] != block_size
            or level["block_count"] != expected_count
            or not isinstance(level["blocks"], list)
            or len(level["blocks"]) != expected_count
        ):
            raise ValueError("independent RG replay found incorrect level metadata")
        computed_level = []
        for block_index, serialized in enumerate(level["blocks"]):
            start = block_index * block_size
            stop = start + block_size
            direct = _direct_block(edges[start:stop], arrays[start:stop])
            operator, forcing, normalized_operator, normalized_forcing, gain, rate = direct
            recorded_operator = _matrix(
                serialized["operator_upper"], dimension)
            recorded_forcing = _vector(
                serialized["forcing_upper"], dimension)
            recorded_normalized_operator = _matrix(
                serialized["normalized_operator_upper"], dimension)
            recorded_normalized_forcing = _vector(
                serialized["normalized_forcing_upper"], dimension)
            composed_normalized_operator = (
                arrays[start]["operator"]
                * arrays[start]["source_gauge"][np.newaxis, :]
                / arrays[start]["target_gauge"][:, np.newaxis]
            )
            composed_normalized_forcing = (
                arrays[start]["forcing"]
                / arrays[start]["target_gauge"]
            )
            for values in arrays[start + 1:stop]:
                later_operator = (
                    values["operator"]
                    * values["source_gauge"][np.newaxis, :]
                    / values["target_gauge"][:, np.newaxis]
                )
                later_forcing = (
                    values["forcing"] / values["target_gauge"])
                composed_normalized_operator, composed_normalized_forcing = (
                    _compose(
                        later_operator, later_forcing,
                        composed_normalized_operator,
                        composed_normalized_forcing,
                    )
                )
            covariance_residual = max(
                float(np.max(np.abs(
                    recorded_normalized_operator
                    - composed_normalized_operator))),
                float(np.max(np.abs(
                    recorded_normalized_forcing
                    - composed_normalized_forcing))),
            )
            tolerance = _comparison_tolerance(
                recorded_normalized_operator,
                recorded_normalized_forcing,
                composed_normalized_operator,
                composed_normalized_forcing,
                recorded_operator,
                recorded_forcing,
            )
            serialized_covariance_tolerance = float(
                serialized["gauge_covariance_tolerance"])
            tolerance_owned = (
                0.5 * tolerance <= serialized_covariance_tolerance
                <= 2.0 * tolerance
            )
            covariance_residuals.append(covariance_residual)
            covariance_tolerances.append(serialized_covariance_tolerance)
            metadata = (
                serialized["start_index"] == edges[start]["start_index"]
                and serialized["stop_index"] == edges[stop - 1]["stop_index"]
                and serialized["duration"] == block_size
                and serialized["source_state_sha256"]
                    == edges[start]["source_state_sha256"]
                and serialized["target_state_sha256"]
                    == edges[stop - 1]["target_state_sha256"]
                and serialized["constituent_edge_sha256"] == [
                    edge["record_sha256"] for edge in edges[start:stop]
                ]
                and serialized["source_gauge"] == edges[start]["source_gauge"]
                and serialized["target_gauge"] == edges[stop - 1]["target_gauge"]
            )
            outward = (
                np.all(recorded_operator + tolerance >= operator)
                and np.all(recorded_forcing + tolerance >= forcing)
                and np.all(
                    recorded_normalized_operator + tolerance
                    >= normalized_operator)
                and np.all(
                    recorded_normalized_forcing + tolerance
                    >= normalized_forcing)
                and float(serialized["weighted_gain_upper"]) + tolerance
                    >= gain
            )
            numerical = (
                _close(recorded_operator, operator, tolerance)
                and _close(recorded_forcing, forcing, tolerance)
                and _close(
                    recorded_normalized_operator,
                    normalized_operator,
                    tolerance,
                )
                and _close(
                    recorded_normalized_forcing,
                    normalized_forcing,
                    tolerance,
                )
                and _scalar_close(
                    serialized["weighted_gain_upper"], gain, tolerance)
                and _scalar_close(
                    serialized["log_gain_per_fine_step_upper"],
                    rate,
                    tolerance,
                )
                and _scalar_close(
                    serialized["gauge_covariance_residual"],
                    covariance_residual,
                    tolerance,
                )
                and covariance_residual <= serialized_covariance_tolerance
                and tolerance_owned
            )
            block_checks.append(bool(metadata and outward and numerical))
            computed_level.append((recorded_operator, recorded_forcing))
        computed_blocks.append(computed_level)

    semigroup_residual = 0.0
    semigroup_tolerance = 0.0
    for level_index in range(1, len(computed_blocks)):
        prior = computed_blocks[level_index - 1]
        for block_index, direct in enumerate(computed_blocks[level_index]):
            pieces = prior[
                block_index * factor:(block_index + 1) * factor
            ]
            operator, forcing = pieces[0]
            for later_operator, later_forcing in pieces[1:]:
                operator, forcing = _compose(
                    later_operator, later_forcing, operator, forcing)
            semigroup_residual = max(
                semigroup_residual,
                float(np.max(np.abs(direct[0] - operator))),
                float(np.max(np.abs(direct[1] - forcing))),
            )
            semigroup_tolerance = max(
                semigroup_tolerance,
                _comparison_tolerance(
                    direct[0], direct[1], operator, forcing),
            )
    first_forward = _compose(
        arrays[1]["operator"], arrays[1]["forcing"],
        arrays[0]["operator"], arrays[0]["forcing"],
    ) if len(arrays) >= 2 else (arrays[0]["operator"], arrays[0]["forcing"])
    first_reverse = _compose(
        arrays[0]["operator"], arrays[0]["forcing"],
        arrays[1]["operator"], arrays[1]["forcing"],
    ) if len(arrays) >= 2 else first_forward
    order_sensitivity = max(
        float(np.max(np.abs(first_forward[0] - first_reverse[0]))),
        float(np.max(np.abs(first_forward[1] - first_reverse[1]))),
    )
    diagnostics = record["diagnostics"]
    recorded_semigroup_tolerance = float(
        diagnostics["semigroup_tolerance"])
    recorded_gauge_residual = float(
        diagnostics["gauge_covariance_residual"])
    recorded_gauge_tolerance = float(
        diagnostics["gauge_covariance_tolerance"])
    maximum_covariance_residual = max(covariance_residuals)
    maximum_covariance_tolerance = max(covariance_tolerances)
    diagnostic_checks = (
        _scalar_close(
            diagnostics["semigroup_residual"], semigroup_residual,
            max(semigroup_tolerance, 1e-15),
        )
        and 0.5 * semigroup_tolerance <= recorded_semigroup_tolerance
            <= 2.0 * semigroup_tolerance
        and _scalar_close(
            diagnostics["first_pair_order_sensitivity"],
            order_sensitivity,
            max(recorded_semigroup_tolerance, 1e-15),
        )
        and _scalar_close(
            recorded_gauge_residual,
            maximum_covariance_residual,
            maximum_covariance_tolerance,
        )
        and _scalar_close(
            recorded_gauge_tolerance,
            maximum_covariance_tolerance,
            max(maximum_covariance_tolerance * 1e-12, 1e-18),
        )
        and recorded_gauge_residual <= recorded_gauge_tolerance
    )
    expected_conditions = {
        "owned_digest_chain": True,
        "positive_gauge_chain": True,
        "aligned_complete_blocking": True,
        "affine_semigroup_enclosed": (
            semigroup_residual
            <= recorded_semigroup_tolerance
        ),
        "gauge_covariance_enclosed": (
            recorded_gauge_residual <= recorded_gauge_tolerance
        ),
        "temporal_order_serialized": all(
            edge["start_index"] == edges[0]["start_index"] + index
            for index, edge in enumerate(edges)
        ),
    }
    checks = {
        "flow_digest": True,
        "fine_edge_digests": True,
        "owned_chain": True,
        "complete_aligned_levels": True,
        "ordered_affine_blocks": all(block_checks),
        "positive_gauge_normalization": all(block_checks),
        "affine_semigroup": bool(
            semigroup_residual
            <= recorded_semigroup_tolerance),
        "diagnostics": bool(diagnostic_checks),
        "constructor_conditions_reproduced": (
            record["conditions"] == expected_conditions
        ),
        "qualified_decision_reproduced": (
            record["decision"]
            == ("QUALIFIED" if all(expected_conditions.values()) else "REJECTED")
        ),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = sorted(name for name, value in checks.items() if not value)
        raise ValueError("independent RG replay failed: " + ", ".join(failed))
    return {
        "schema_version": "pldr-independent-rg-replay-v1",
        "flow_sha256": record["record_sha256"],
        "checks": checks,
        "decision": "REPLAYED",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    with Path(arguments.record).open("r", encoding="utf-8") as stream:
        record = json.load(stream)
    result = check_rg_flow(record)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
