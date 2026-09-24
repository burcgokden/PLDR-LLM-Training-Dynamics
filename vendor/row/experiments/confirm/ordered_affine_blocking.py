"""Owned exact blocking of constructor-produced positive affine edges."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from constructive_block_comparison import EDGE_SCHEMA_VERSION
from validated_interval import as_fraction, rational_object


BLOCK_SCHEMA_VERSION = "pldr-ordered-affine-block-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _matrix(value):
    rows = tuple(tuple(as_fraction(entry) for entry in row) for row in value)
    if not rows or any(len(row) != len(rows) for row in rows):
        raise ValueError("affine comparison matrix must be square")
    if any(entry < 0 for row in rows for entry in row):
        raise ValueError("affine comparison matrix must be nonnegative")
    return rows


def _vector(value, size):
    result = tuple(as_fraction(entry) for entry in value)
    if len(result) != size or any(entry < 0 for entry in result):
        raise ValueError("affine forcing has the wrong dimension or sign")
    return result


def _multiply(left, right):
    size = len(left)
    return tuple(tuple(sum(
        (left[i][k] * right[k][j] for k in range(size)), Fraction(0))
        for j in range(size)) for i in range(size))


def _apply(matrix, vector):
    return tuple(sum(
        (matrix[i][j] * vector[j] for j in range(len(matrix))), Fraction(0))
        for i in range(len(matrix)))


def _matrix_object(value):
    return [[rational_object(entry) for entry in row] for row in value]


def _vector_object(value):
    return [rational_object(entry) for entry in value]


def _validate_edge(edge):
    if (
        not isinstance(edge, dict)
        or edge.get("schema_version") != EDGE_SCHEMA_VERSION
    ):
        raise ValueError("blocking input is not an owned energy edge")
    unsigned = dict(edge)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        raise ValueError("owned energy edge digest does not replay")
    matrix = _matrix(edge["comparison_matrix"])
    forcing = _vector(edge["forcing"], len(matrix))
    if len(edge["block_names"]) != len(matrix):
        raise ValueError("owned energy edge block registry is inconsistent")
    return matrix, forcing


def block_owned_edges(edges):
    """Compose resolved fine edges in temporal order."""

    edges = list(edges)
    if not edges:
        raise ValueError("ordered blocking needs at least one fine edge")
    first_matrix, first_forcing = _validate_edge(edges[0])
    size = len(first_matrix)
    product = tuple(tuple(Fraction(int(i == j)) for j in range(size))
                    for i in range(size))
    accumulated = tuple(Fraction(0) for _ in range(size))
    names = edges[0]["block_names"]
    fine_digests = []
    for index, edge in enumerate(edges):
        matrix, forcing = _validate_edge(edge)
        if edge["block_names"] != names:
            raise ValueError("fine edges do not share a block registry")
        if index and (
            edges[index - 1]["target_state_sha256"]
            != edge["source_state_sha256"]
        ):
            raise ValueError("fine edges do not form an owned state chain")
        accumulated = tuple(
            value + forcing[i]
            for i, value in enumerate(_apply(matrix, accumulated)))
        product = _multiply(matrix, product)
        fine_digests.append(edge["record_sha256"])
    result = {
        "schema_version": BLOCK_SCHEMA_VERSION,
        "source_state_sha256": edges[0]["source_state_sha256"],
        "target_state_sha256": edges[-1]["target_state_sha256"],
        "block_names": names,
        "comparison_matrix": _matrix_object(product),
        "forcing": _vector_object(accumulated),
        "fine_edge_record_sha256": fine_digests,
        "edge_count": len(edges),
        "composition_rule": "P2_P1_and_P2_d1_plus_d2",
    }
    result["record_sha256"] = _digest(result)
    return result


def normalize_endpoint_gauge(block, source_gauge, target_gauge):
    """Apply the exact positive diagonal endpoint gauge."""

    if block.get("schema_version") != BLOCK_SCHEMA_VERSION:
        raise ValueError("unknown affine block")
    unsigned = dict(block)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        raise ValueError("affine block digest does not replay")
    matrix = _matrix(block["comparison_matrix"])
    size = len(matrix)
    source = _vector(source_gauge, size)
    target = _vector(target_gauge, size)
    if any(value <= 0 for value in source + target):
        raise ValueError("endpoint gauges must be strictly positive")
    forcing = _vector(block["forcing"], size)
    normalized_matrix = tuple(tuple(
        matrix[i][j] * source[j] / target[i]
        for j in range(size)) for i in range(size))
    normalized_forcing = tuple(
        forcing[i] / target[i] for i in range(size))
    return {
        "source_gauge": _vector_object(source),
        "target_gauge": _vector_object(target),
        "comparison_matrix": _matrix_object(normalized_matrix),
        "forcing": _vector_object(normalized_forcing),
        "derivation": "target_gauge_inverse_times_edge_times_source_gauge",
    }


def infinity_norm_affine_upper(matrix, energy, forcing):
    """Return the corrected bound ||P|| infinity ||e|| infinity + ||d|| infinity."""

    matrix = _matrix(matrix)
    energy = _vector(energy, len(matrix))
    forcing = _vector(forcing, len(matrix))
    matrix_norm = max(sum(row, Fraction(0)) for row in matrix)
    energy_norm = max(energy)
    forcing_norm = max(forcing)
    return matrix_norm * energy_norm + forcing_norm
