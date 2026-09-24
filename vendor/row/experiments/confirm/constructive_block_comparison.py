"""Exact construction of every positive block comparison coefficient."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json

from validated_interval import as_fraction, rational_object
from validated_linear_algebra import identity, inverse, multiply, subtract


SCHEMA_VERSION = "pldr-constructive-block-comparison-v1"
EDGE_SCHEMA_VERSION = "pldr-owned-positive-energy-edge-v1"


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _nonnegative(value, label):
    value = as_fraction(value)
    if value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _matrix(value, label):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a nonempty matrix")
    rows = tuple(tuple(_nonnegative(entry, label) for entry in row)
                 for row in value)
    if any(len(row) != len(rows) for row in rows):
        raise ValueError(f"{label} must be square")
    return rows


def _matrix_object(value):
    return [[rational_object(entry) for entry in row] for row in value]


def _vector_object(value):
    return [rational_object(entry) for entry in value]


def construct_comparison(
        *, block_names, transformed_operator_norms, forcing_norms,
        structural_zeros):
    """Derive P,d by weighted Cauchy-Schwarz with frozen equal weights.

    Every ordered block pair must occur with a certified transformed
    operator norm or as a branch-proved structural zero. Uniform weights
    over active terms are a deterministic construction rule that uses no
    validation magnitudes. No comparison coefficient is supplied by the
    caller.
    """

    if (
        not isinstance(block_names, (list, tuple))
        or not block_names
        or any(not isinstance(name, str) or not name for name in block_names)
        or len(set(block_names)) != len(block_names)
    ):
        raise ValueError("block_names must be distinct nonempty strings")
    names = tuple(block_names)
    pairs = {(target, source) for target in names for source in names}
    if not isinstance(transformed_operator_norms, dict):
        raise ValueError("transformed_operator_norms must be a mapping")
    if not isinstance(structural_zeros, (list, tuple, set)):
        raise ValueError("structural_zeros must be a collection")
    zero_pairs = {tuple(value) for value in structural_zeros}
    norm_pairs = {tuple(value) for value in transformed_operator_norms}
    if norm_pairs & zero_pairs:
        raise ValueError("a block cannot be both nonzero and structurally zero")
    if norm_pairs | zero_pairs != pairs:
        raise ValueError("the dependency grid is incomplete or has unknown blocks")
    if set(forcing_norms) != set(names):
        raise ValueError("forcing_norms must cover every target block")

    comparison = [[Fraction(0) for _ in names] for _ in names]
    forcing = [Fraction(0) for _ in names]
    allocations = []
    block_records = []
    for target_index, target in enumerate(names):
        active = []
        for source in names:
            pair = (target, source)
            if pair in norm_pairs:
                gain = _nonnegative(
                    transformed_operator_norms[pair],
                    f"transformed operator norm {target} <- {source}")
                if gain == 0:
                    raise ValueError(
                        "zero operator bounds must be declared structural zeros")
                active.append(("block", source, gain))
        forcing_norm = _nonnegative(
            forcing_norms[target], f"forcing norm for {target}")
        if forcing_norm > 0:
            active.append(("forcing", None, forcing_norm))
        denominator = len(active)
        allocation = (
            Fraction(1, denominator) if denominator else Fraction(1))
        allocations.append({
            "target": target,
            "active_term_count": denominator,
            "uniform_young_weight": rational_object(allocation),
        })
        for source_index, source in enumerate(names):
            pair = (target, source)
            if pair in zero_pairs:
                block_records.append({
                    "target": target,
                    "source": source,
                    "structural_zero": True,
                    "transformed_operator_norm_upper": rational_object(0),
                    "young_weight": None,
                    "comparison_coefficient": rational_object(0),
                })
                continue
            gain = _nonnegative(
                transformed_operator_norms[pair], "operator norm")
            coefficient = gain ** 2 / allocation
            comparison[target_index][source_index] = coefficient
            block_records.append({
                "target": target,
                "source": source,
                "structural_zero": False,
                "transformed_operator_norm_upper": rational_object(gain),
                "young_weight": rational_object(allocation),
                "comparison_coefficient": rational_object(coefficient),
            })
        if forcing_norm > 0:
            forcing[target_index] = forcing_norm ** 2 / allocation

    result = {
        "schema_version": SCHEMA_VERSION,
        "block_names": list(names),
        "comparison_matrix": _matrix_object(tuple(map(tuple, comparison))),
        "forcing": _vector_object(tuple(forcing)),
        "young_allocations": allocations,
        "operator_blocks": block_records,
        "dependency_grid_complete": True,
        "derivation": (
            "weighted_cauchy_schwarz_from_transformed_block_norms_"
            "and_forcing_norms"),
    }
    result["record_sha256"] = _digest(result)
    return result


def construct_positive_witness(comparison_matrix):
    """Construct v=(I-P)^-1 one and the exact componentwise gain."""

    matrix = _matrix(comparison_matrix, "comparison matrix")
    size = len(matrix)
    resolvent = inverse(subtract(identity(size), matrix))
    witness_column = multiply(
        resolvent, tuple((Fraction(1),) for _ in range(size)))
    witness = tuple(row[0] for row in witness_column)
    if any(value <= 0 for value in witness):
        raise ValueError("the resolvent did not produce a positive witness")
    image = tuple(
        sum((matrix[i][j] * witness[j] for j in range(size)), Fraction(0))
        for i in range(size)
    )
    if any(image[i] + 1 != witness[i] for i in range(size)):
        raise ArithmeticError("positive resolvent identity failed")
    ratios = tuple(image[i] / witness[i] for i in range(size))
    kappa = max(ratios)
    if not 0 <= kappa < 1:
        raise ValueError("the constructed positive witness is not contractive")
    return {
        "weights": _vector_object(witness),
        "component_ratios": _vector_object(ratios),
        "kappa": rational_object(kappa),
        "strict_slack": rational_object(1 - kappa),
        "identity": "P_v_plus_one_equals_v",
    }


def bind_owned_edge(
        *, edge_id, source_state_sha256, target_state_sha256,
        comparison_record):
    """Bind a constructor-produced comparison to one resolved state edge."""

    for name, value in (
        ("source_state_sha256", source_state_sha256),
        ("target_state_sha256", target_state_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    if source_state_sha256 == target_state_sha256:
        raise ValueError("an update edge must have distinct state digests")
    if (
        not isinstance(comparison_record, dict)
        or comparison_record.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError("comparison_record was not produced by the constructor")
    unsigned = dict(comparison_record)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        raise ValueError("comparison_record digest does not replay")
    value = {
        "schema_version": EDGE_SCHEMA_VERSION,
        "edge_id": edge_id,
        "source_state_sha256": source_state_sha256,
        "target_state_sha256": target_state_sha256,
        "comparison_record_sha256": digest,
        "block_names": comparison_record["block_names"],
        "comparison_matrix": comparison_record["comparison_matrix"],
        "forcing": comparison_record["forcing"],
    }
    value["record_sha256"] = _digest(value)
    return value


def forced_affine_upper(comparison_matrix, current_energy, forcing):
    """Evaluate P e + d with the required positive forcing sign."""

    matrix = _matrix(comparison_matrix, "comparison matrix")
    current = tuple(_nonnegative(value, "current energy")
                    for value in current_energy)
    source = tuple(_nonnegative(value, "forcing") for value in forcing)
    if len(current) != len(matrix) or len(source) != len(matrix):
        raise ValueError("affine energy dimensions disagree")
    return tuple(
        sum((matrix[i][j] * current[j] for j in range(len(matrix))),
            Fraction(0)) + source[i]
        for i in range(len(matrix))
    )
