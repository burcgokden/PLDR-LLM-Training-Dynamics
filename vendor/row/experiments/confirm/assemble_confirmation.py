"""Assemble primary measurements only from checked certificate DAG outputs."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path

import campaign_record
from certificate_graph import validate_graph
from estimator_manifest import load_manifest
from validated_interval import as_fraction


def _directed_float(value, projection):
    value = as_fraction(value)
    try:
        candidate = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("exact primary measurement is outside float range") from error
    if not math.isfinite(candidate):
        raise ValueError("exact primary measurement is outside float range")
    represented = Fraction.from_float(candidate)
    if projection == "upper" and represented < value:
        candidate = math.nextafter(candidate, math.inf)
    elif projection == "lower" and represented > value:
        candidate = math.nextafter(candidate, -math.inf)
    elif projection == "point" and represented != value:
        raise ValueError(
            "registered point measurement is not exactly representable; "
            "store the realized binary float or a registered endpoint")
    if not math.isfinite(candidate):
        raise ValueError("directed primary measurement is outside float range")
    return candidate


def finite_scalar_projection(value, projection):
    if projection not in {"upper", "lower", "point"}:
        raise ValueError("numeric projection must be upper, lower, or point")
    if isinstance(value, bool):
        raise ValueError("Boolean value is not a primary scalar")
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("primary measurement must be finite")
        return value
    if isinstance(value, (int, Fraction)):
        return _directed_float(value, projection)
    if isinstance(value, dict) and set(value) == {"numerator", "denominator"}:
        return _directed_float(value, projection)
    if isinstance(value, dict) and set(value) == {"lower", "upper"}:
        lower = as_fraction(value["lower"])
        upper = as_fraction(value["upper"])
        if lower > upper:
            raise ValueError("primary interval has reversed endpoints")
        if projection == "upper":
            return _directed_float(upper, "upper")
        if projection == "lower":
            return _directed_float(lower, "lower")
        if lower != upper:
            raise ValueError(
                "registered point projection cannot consume a nondegenerate interval")
        return _directed_float(lower, "point")
    raise ValueError(
        "primary measurement node is not a scalar or rational interval")


def assemble(protocol_id, graph, output_nodes, manifest):
    """Create the closed measurement array from owned, checked graph nodes.

    ``output_nodes`` maps each registered measurement name to one graph node.
    The node metadata must name the exact estimator-manifest identifier.  This
    prevents a caller from importing a free-standing number with a status
    string or arbitrary method label.
    """

    if protocol_id not in campaign_record.PROTOCOLS:
        raise ValueError("protocol must be E0 through E8")
    nodes = validate_graph(graph)
    required = set(campaign_record.MEASUREMENT_NAMES[protocol_id])
    if set(output_nodes) != required:
        raise ValueError("output-node map does not equal the registered measurement set")
    rows = []
    for name in campaign_record.MEASUREMENT_NAMES[protocol_id]:
        node_name = output_nodes[name]
        if node_name not in nodes:
            raise ValueError(f"measurement node is absent: {node_name}")
        node = nodes[node_name]
        estimator_id = f"{protocol_id}.{name}"
        if estimator_id not in manifest["estimators"]:
            raise ValueError(f"estimator is not registered: {estimator_id}")
        if node["metadata"].get("estimator_id") != estimator_id:
            raise ValueError(f"certificate node does not bind estimator {estimator_id}")
        owner = manifest["estimators"][estimator_id]
        if node["constructor"] != (
            owner["executable"] + ":" + owner["function"]
        ):
            raise ValueError(f"certificate node has the wrong owner for {estimator_id}")
        if node["constructor_sha256"] != owner["executable_sha256"]:
            raise ValueError(f"certificate node owner digest is stale: {estimator_id}")
        rows.append({
            "name": name,
            "value": finite_scalar_projection(
                node["value"], owner["numeric_projection"]),
            "unit": node["unit"],
            "method": estimator_id,
            "status": "OBSERVED",
            "reason_code": None,
        })
    return {
        "protocol_id": protocol_id,
        "certificate_graph_sha256": graph["graph_sha256"],
        "measurements": rows,
        "node_bindings": dict(output_nodes),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=campaign_record.PROTOCOLS,
                        required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--output-nodes", required=True)
    parser.add_argument("--estimator-manifest", required=True)
    parser.add_argument("--output", required=True)
    arguments = parser.parse_args()
    graph = json.loads(Path(arguments.graph).read_text(encoding="utf-8"))
    output_nodes = json.loads(
        Path(arguments.output_nodes).read_text(encoding="utf-8"))
    manifest = load_manifest(arguments.estimator_manifest)
    result = assemble(arguments.protocol, graph, output_nodes, manifest)
    Path(arguments.output).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
