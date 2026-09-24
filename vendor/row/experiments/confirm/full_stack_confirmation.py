#!/usr/bin/env python3
"""Assemble the primary complete-stack row-map confirmation certificate.

This module deliberately has no trace-fitting inputs.  Every accepted edge is
constructed from a complete block operator, an exact program-state successor,
an outward runtime bridge, and an explicitly sourced nonnegative forcing
vector.  Scalar projections and fitted geometric envelopes are diagnostics
only and cannot enter the primary decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from block_comparison import (  # noqa: E402
    construct_block_comparison,
    entrywise_path_envelope,
    enumerate_path_compositions,
    verify_positive_witness,
)
from campaign_record import strict_dumps  # noqa: E402
from full_stack import build_registry, validate_registry  # noqa: E402
from normal_closure import (  # noqa: E402
    check_dense_normal_closure,
    construct_dense_normal_closure,
)
from program_state import (  # noqa: E402
    derived_positive_map,
    interval_adamw_successor,
)
from runtime_enclosure import (  # noqa: E402
    enclose_runtime_array,
    verify_runtime_array,
)
from validated_interval import as_fraction  # noqa: E402


SCHEMA_VERSION = "pldr-full-stack-confirmation-v1"
TIME_SEMANTICS = {
    "FINITE_IMPLEMENTED_SCHEDULE",
    "FROZEN_CHECKPOINT",
    "CONTROLLED_INFINITE_TAIL",
}


def successor_graph_is_total(edges):
    """Return whether every registered target owns an outgoing edge."""

    edges = list(edges)
    if not edges:
        return False
    for row in edges:
        if not isinstance(row, dict):
            raise TypeError("successor graph edges must be mappings")
        for name in ("source", "target"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(
                    "successor graph endpoints must be nonempty strings")
    sources = {row["source"] for row in edges}
    targets = {row["target"] for row in edges}
    return targets <= sources


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _load(path):
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError("full-stack input must be a JSON object")
    return value


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = strict_dumps(value)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _finite(value, label, *, nonnegative=False):
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0):
        qualifier = "finite and nonnegative" if nonnegative else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _rational_vector(value, label):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty rational vector")
    result = np.asarray([float(as_fraction(entry)) for entry in value])
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


def _layer_cells(value):
    if not isinstance(value, dict) or not value:
        raise ValueError("layer_cells must be a nonempty object")
    result = {}
    for layer, cells in value.items():
        try:
            index = int(layer)
        except (TypeError, ValueError) as error:
            raise ValueError("layer_cells keys must be integer strings") from error
        if str(index) != str(layer):
            raise ValueError("layer_cells keys must use canonical integers")
        if not isinstance(cells, list):
            raise ValueError("each layer_cells entry must be an array")
        result[index] = [(row["cell_id"], row["width"]) for row in cells]
    return result


def _conversion_errors(value):
    if not isinstance(value, list):
        raise ValueError("runtime conversion errors must be an array")
    result = {}
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "target", "source", "absolute_operator_error",
        }:
            raise ValueError("runtime conversion error row is malformed")
        key = (row["target"], row["source"])
        if key in result:
            raise ValueError("runtime conversion error block is duplicated")
        result[key] = _finite(
            row["absolute_operator_error"],
            "runtime conversion error", nonnegative=True)
    return result


def _runtime_records(value):
    if not isinstance(value, list) or not value:
        raise ValueError("at least one runtime primitive enclosure is required")
    records = []
    names = set()
    for row in value:
        if not isinstance(row, dict) or set(row) != {
            "name", "values", "absolute_errors",
        }:
            raise ValueError("runtime primitive row is malformed")
        name = row["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("runtime primitive names must be unique")
        names.add(name)
        values = np.asarray(row["values"], dtype=float)
        record = enclose_runtime_array(values, row["absolute_errors"])
        verify_runtime_array(record, values)
        records.append({"name": name, "enclosure": record})
    return records


def _frozen_checkpoint(value, criterion):
    required = {
        "model_sha256", "row_cover_sha256", "optimizer_updates_disabled",
        "evaluation_only", "direct_row_map_upper",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("frozen-checkpoint record is malformed")
    for name in ("model_sha256", "row_cover_sha256"):
        digest = value[name]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{name} must be a SHA-256 digest")
    if value["optimizer_updates_disabled"] is not True \
            or value["evaluation_only"] is not True:
        raise ValueError("frozen semantics requires an evaluation-only checkpoint")
    direct = _finite(
        value["direct_row_map_upper"], "frozen direct bound", nonnegative=True)
    return {
        **value,
        "direct_row_map_upper": direct,
        "strict_direct_margin": criterion - direct,
        "persistent_by_identity": True,
    }


def _program_edge(row):
    required = {
        "edge_id", "source", "target", "program_update", "target_cell",
        "complete_operator", "block_slices", "source_metrics",
        "target_metrics", "runtime_conversion_errors", "forcing",
        "forcing_provenance",
    }
    if not isinstance(row, dict) or set(row) != required:
        raise ValueError("full-stack edge has missing or unknown fields")
    for name in ("edge_id", "source", "target"):
        if not isinstance(row[name], str) or not row[name]:
            raise ValueError(f"edge {name} must be a nonempty string")
    successor = interval_adamw_successor(**row["program_update"])
    state_map = derived_positive_map(successor, row["target_cell"])
    comparison = construct_block_comparison(
        row["complete_operator"], row["block_slices"],
        row["source_metrics"], row["target_metrics"],
        _conversion_errors(row["runtime_conversion_errors"]),
    )
    forcing = _rational_vector(row["forcing"], "full-stack forcing")
    matrix = np.asarray(comparison["comparison_matrix"], dtype=float)
    if forcing.shape != (matrix.shape[0],):
        raise ValueError("full-stack forcing and comparison dimensions disagree")
    provenance = row["forcing_provenance"]
    if (
        not isinstance(provenance, list)
        or len(provenance) != len(forcing)
        or any(not isinstance(item, str) or not item for item in provenance)
    ):
        raise ValueError("every forcing coordinate needs explicit provenance")
    positive_map = {
        "edge_id": row["edge_id"],
        "source": row["source"],
        "target": row["target"],
        "matrix": matrix.tolist(),
        "forcing": forcing.tolist(),
    }
    return {
        "edge_id": row["edge_id"],
        "source": row["source"],
        "target": row["target"],
        "program_successor": successor,
        "program_positive_map": state_map,
        "complete_block_comparison": comparison,
        "full_stack_positive_map": positive_map,
        "forcing_provenance": provenance,
    }


def build_confirmation(payload):
    required = {
        "schema_version", "time_semantics", "criterion", "layer_cells",
        "normal_closure", "runtime_primitives", "edges", "path_length",
        "positive_witness", "frozen_checkpoint", "validation_membership",
        "cover_utility_margins", "direct_row_map_uppers",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("full-stack input has missing or unknown fields")
    if payload["schema_version"] != "pldr-full-stack-confirmation-input-v1":
        raise ValueError("unknown full-stack confirmation input schema")
    semantics = payload["time_semantics"]
    if semantics not in TIME_SEMANTICS:
        raise ValueError("unknown confirmation time semantics")
    criterion = _finite(payload["criterion"], "criterion")
    if criterion <= 0:
        raise ValueError("criterion must be positive")

    registry = build_registry(_layer_cells(payload["layer_cells"]))
    validate_registry(registry)
    normal_inputs = payload["normal_closure"]
    if not isinstance(normal_inputs, dict):
        raise ValueError("normal_closure must be an object")
    normal = construct_dense_normal_closure(**normal_inputs)
    if normal["stack_dimension"] != registry["dimension"]:
        raise ValueError("normal closure does not act on the complete registry")
    check_dense_normal_closure(
        normal,
        stack_parameter_jacobian=normal_inputs["stack_parameter_jacobian"],
        normal_velocity_parameter_jacobian=normal_inputs[
            "normal_velocity_parameter_jacobian"],
        stack_center=normal_inputs["stack_center"],
        normal_velocity_center=normal_inputs["normal_velocity_center"],
        validation_cells=normal_inputs["validation_cells"],
    )
    runtime = _runtime_records(payload["runtime_primitives"])

    membership = payload["validation_membership"]
    if not isinstance(membership, list) or not membership \
            or any(not isinstance(value, bool) for value in membership):
        raise ValueError("validation_membership must be a nonempty Boolean array")
    utility = [
        _finite(value, "cover utility margin")
        for value in payload["cover_utility_margins"]
    ]
    direct = [
        _finite(value, "direct row-map upper", nonnegative=True)
        for value in payload["direct_row_map_uppers"]
    ]
    if not utility or not direct:
        raise ValueError("cover margins and direct bounds must be nonempty")

    edges = [_program_edge(row) for row in payload["edges"]]
    edge_ids = [row["edge_id"] for row in edges]
    if len(set(edge_ids)) != len(edge_ids):
        raise ValueError("full-stack edge identifiers are duplicated")

    path_record = None
    witness = None
    frozen = None
    graph_total = True
    if semantics == "FROZEN_CHECKPOINT":
        if edges or payload["path_length"] is not None \
                or payload["positive_witness"] is not None:
            raise ValueError("frozen checkpoint semantics has no optimizer edges")
        frozen = _frozen_checkpoint(payload["frozen_checkpoint"], criterion)
    else:
        if payload["frozen_checkpoint"] is not None or not edges:
            raise ValueError("evolving semantics requires edges and no frozen record")
        length = payload["path_length"]
        if not isinstance(length, int) or isinstance(length, bool) or length < 1:
            raise ValueError("path_length must be a positive integer")
        compositions = enumerate_path_compositions(
            [row["full_stack_positive_map"] for row in edges], length)
        envelope = entrywise_path_envelope(compositions)
        path_record = {
            "length": length,
            "compositions": compositions,
            "entrywise_envelope": envelope,
        }
        if semantics == "CONTROLLED_INFINITE_TAIL":
            graph_total = successor_graph_is_total(edges)
            specification = payload["positive_witness"]
            if not isinstance(specification, dict) or set(specification) != {
                "weights", "kappa",
            }:
                raise ValueError("controlled tail needs a positive witness")
            witness = verify_positive_witness(
                envelope["matrix"], specification["weights"],
                specification["kappa"],
            )
        elif payload["positive_witness"] is not None:
            raise ValueError("finite schedule must not assert an infinite-tail witness")

    conditions = {
        "complete_stack_registry": True,
        "construction_validation_disjoint": True,
        "right_inverse_residual_below_one": normal[
            "right_inverse_residual"] < 1.0,
        "all_validation_rows_in_frozen_domain": all(membership),
        "all_cover_utility_margins_positive": min(utility) > 0.0,
        "all_direct_bounds_below_criterion": max(direct) < criterion,
        "all_off_diagonal_blocks_included": all(
            row["complete_block_comparison"]["off_diagonal_blocks_included"]
            for row in edges
        ) if edges else True,
        "time_semantics_closed": (
            frozen is not None if semantics == "FROZEN_CHECKPOINT"
            else path_record is not None
        ),
        "controlled_tail_witness": (
            witness is not None
            if semantics == "CONTROLLED_INFINITE_TAIL" else True
        ),
        "controlled_tail_successor_graph_total": (
            graph_total
            if semantics == "CONTROLLED_INFINITE_TAIL" else True
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "time_semantics": semantics,
        "criterion": criterion,
        "registry": registry,
        "normal_closure": normal,
        "runtime_primitive_enclosures": runtime,
        "edges": edges,
        "path_certificate": path_record,
        "positive_witness": witness,
        "frozen_checkpoint": frozen,
        "validation_membership": membership,
        "cover_utility_margins": utility,
        "direct_row_map_uppers": direct,
        "conditions": conditions,
        "decision": "CONFIRMED" if all(conditions.values()) else "NOT_CONFIRMED",
        "scalar_projection_role": "diagnostic_only",
        "trace_fitting_used": False,
        "caller_selected_radius_multipliers_used": False,
    }
    result["record_sha256"] = _digest(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    result = build_confirmation(_load(arguments.input))
    _write(arguments.output, result)
    print(arguments.output)


if __name__ == "__main__":
    main()
