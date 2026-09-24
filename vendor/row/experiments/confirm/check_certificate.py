"""Independent replay boundary for typed confirmation certificates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from certificate_graph import validate_graph
from small_pldr_certificate import build_small_certificate
from tail_comparison import (
    check_finite_entry_horizon,
    check_positive_certificate,
)
from validated_linear_algebra import check_residual_aware_lyapunov_certificate


def check_graph(graph):
    nodes = validate_graph(graph)
    for node in nodes.values():
        if node["derivation_kind"] == "exact_ldl_witness":
            check_residual_aware_lyapunov_certificate(node["value"])
        elif node["derivation_kind"] == "positive_comparison":
            check_positive_certificate(node["value"])
        elif node["derivation_kind"] == "finite_horizon":
            value = node["value"]
            check_finite_entry_horizon(value)
            if value.get("horizon_type") != "FINITE_REGISTERED_SCHEDULE":
                raise ValueError("finite-horizon node has the wrong horizon type")
            if not value.get("permanent_through_terminal"):
                raise ValueError("finite-horizon node does not establish permanent entry")
    return {
        "status": "PASS",
        "node_count": len(nodes),
        "output_count": len(graph["outputs"]),
        "graph_sha256": graph["graph_sha256"],
    }


def check_small_certificate(certificate):
    if certificate.get("schema_version") != "pldr-small-complete-certificate-v2":
        raise ValueError("small-certificate schema is stale")
    replay = check_graph(certificate["graph"])
    rebuilt = build_small_certificate()
    if rebuilt != certificate:
        raise ValueError("small complete certificate does not reconstruct exactly")
    return {**replay, "small_complete_example": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", required=True)
    parser.add_argument("--output")
    arguments = parser.parse_args()
    certificate = json.loads(
        Path(arguments.certificate).read_text(encoding="utf-8"))
    if certificate.get("schema_version") == "pldr-small-complete-certificate-v2":
        result = check_small_certificate(certificate)
    else:
        result = check_graph(certificate)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
