"""Full-block comparison and positive path composition.

Every diagonal and off-diagonal operator block contributes to the
nonnegative comparison matrix.  This prevents stable scalar projections or
stable diagonal blocks from hiding an unstable orthogonal or cross-layer
mode.
"""

from __future__ import annotations

from collections import defaultdict
import itertools
import math

import numpy as np

from full_stack import outward_upper, validate_block_slices


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return result


def _metric(value, size, name):
    result = _array(value, name, ndim=2)
    if result.shape != (size, size) or not np.allclose(
        result, result.T, atol=1e-12, rtol=1e-12,
    ):
        raise ValueError(f"{name} must be symmetric with the block dimension")
    eigenvalues = np.linalg.eigvalsh(result)
    if eigenvalues[0] <= 0:
        raise ValueError(f"{name} is not positive definite")
    return result


def metric_block_gain(operator_block, source_metric, target_metric):
    """Return the induced source-to-target quadratic-metric gain."""

    block = _array(operator_block, "operator block", ndim=2)
    target_size, source_size = block.shape
    source = _metric(source_metric, source_size, "source metric")
    target = _metric(target_metric, target_size, "target metric")
    source_cholesky = np.linalg.cholesky(source)
    target_cholesky = np.linalg.cholesky(target)
    transformed = (
        target_cholesky.T @ block
        @ np.linalg.inv(source_cholesky.T)
    )
    return outward_upper(np.linalg.norm(transformed, ord=2))


def construct_block_comparison(
        operator, block_slices, source_metrics, target_metrics,
        conversion_error_bounds=None):
    """Construct a complete nonnegative block-gain matrix.

    ``conversion_error_bounds`` may add a separately proved runtime-to-exact
    charge for each ordered block name pair.  It must be complete when
    supplied; an omitted off-diagonal error is not interpreted as zero.
    """

    matrix = _array(operator, "complete operator", ndim=2)
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("complete block operator must be square")
    dimension = matrix.shape[0]
    rows = list(block_slices)
    validate_block_slices(dimension, rows)
    names = [row["name"] for row in rows]
    if set(source_metrics) != set(names) or set(target_metrics) != set(names):
        raise ValueError("block metrics do not cover every direct-sum block")
    expected_errors = {(target, source) for target in names for source in names}
    if conversion_error_bounds is None:
        errors = {key: 0.0 for key in expected_errors}
    else:
        if set(conversion_error_bounds) != expected_errors:
            raise ValueError("conversion errors omit or add an operator block")
        errors = {}
        for key, value in conversion_error_bounds.items():
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError("block conversion errors must be nonnegative")
            errors[key] = value
    gains = np.zeros((len(rows), len(rows)), dtype=float)
    block_records = []
    for target_index, target in enumerate(rows):
        target_slice = slice(target["start"], target["stop"])
        target_metric = target_metrics[target["name"]]
        for source_index, source in enumerate(rows):
            source_slice = slice(source["start"], source["stop"])
            source_metric = source_metrics[source["name"]]
            operator_block = matrix[target_slice, source_slice]
            exact_gain = metric_block_gain(
                operator_block, source_metric, target_metric)
            charge = errors[(target["name"], source["name"])]
            accepted = outward_upper(exact_gain + charge)
            gains[target_index, source_index] = accepted
            block_records.append({
                "target": target["name"],
                "source": source["name"],
                "shape": list(operator_block.shape),
                "metric_gain": exact_gain,
                "runtime_conversion_charge": charge,
                "accepted_gain": accepted,
                "is_off_diagonal": target_index != source_index,
            })
    return {
        "schema_version": "pldr-complete-block-comparison-v1",
        "dimension": dimension,
        "block_names": names,
        "block_slices": rows,
        "comparison_matrix": gains.tolist(),
        "operator_blocks": block_records,
        "off_diagonal_blocks_included": True,
        "maximum_block_gain": float(np.max(gains)),
    }


def weighted_gain(comparison_matrix, weights):
    matrix = _array(comparison_matrix, "comparison matrix", ndim=2)
    weights = _array(weights, "comparison weights", ndim=1)
    if matrix.shape != (len(weights), len(weights)):
        raise ValueError("comparison matrix and weights disagree")
    if (matrix < 0).any() or (weights <= 0).any():
        raise ValueError("comparison matrix must be nonnegative and weights positive")
    return outward_upper(float(np.max((matrix @ weights) / weights)))


def verify_positive_witness(comparison_matrix, weights, kappa):
    matrix = _array(comparison_matrix, "comparison matrix", ndim=2)
    weights = _array(weights, "comparison weights", ndim=1)
    kappa = float(kappa)
    if not math.isfinite(kappa) or not 0 <= kappa < 1:
        raise ValueError("kappa must lie in [0, 1)")
    if matrix.shape != (len(weights), len(weights)):
        raise ValueError("comparison matrix and weights disagree")
    if (matrix < 0).any() or (weights <= 0).any():
        raise ValueError("positive witness has the wrong sign")
    ratios = (matrix @ weights) / weights
    if np.any(ratios > kappa):
        raise ValueError("P v is not componentwise below kappa v")
    return {
        "weights": weights.tolist(),
        "kappa": kappa,
        "component_ratios": ratios.tolist(),
        "maximum_ratio": outward_upper(float(np.max(ratios))),
        "strict_slack": float(kappa - np.max(ratios)),
    }


def compose_affine_maps(maps):
    """Compose nonnegative maps ``x' <= P x + d`` in temporal order."""

    maps = list(maps)
    if not maps:
        raise ValueError("at least one positive affine map is required")
    first_matrix = _array(maps[0]["matrix"], "positive matrix", ndim=2)
    if first_matrix.shape[0] != first_matrix.shape[1]:
        raise ValueError("positive affine matrix must be square")
    dimension = first_matrix.shape[0]
    product_matrix = np.eye(dimension)
    product_source = np.zeros(dimension)
    steps = []
    for index, value in enumerate(maps):
        if not isinstance(value, dict) or set(value) != {
            "edge_id", "source", "target", "matrix", "forcing",
        }:
            raise ValueError("positive affine map has missing or unknown fields")
        matrix = _array(value["matrix"], "positive matrix", ndim=2)
        forcing = _array(value["forcing"], "positive forcing", ndim=1)
        if matrix.shape != (dimension, dimension) or forcing.shape != (dimension,):
            raise ValueError("positive affine map dimensions disagree")
        if (matrix < 0).any() or (forcing < 0).any():
            raise ValueError("positive affine maps must be componentwise nonnegative")
        if index and maps[index - 1]["target"] != value["source"]:
            raise ValueError("positive affine maps do not form a path")
        product_source = matrix @ product_source + forcing
        product_matrix = matrix @ product_matrix
        steps.append(value["edge_id"])
    return {
        "edge_ids": steps,
        "source": maps[0]["source"],
        "target": maps[-1]["target"],
        "matrix": product_matrix.tolist(),
        "forcing": product_source.tolist(),
    }


def enumerate_path_compositions(edges, length):
    """Enumerate every admissible graph path of an exact positive length."""

    if not isinstance(length, int) or isinstance(length, bool) or length < 1:
        raise ValueError("path length must be a positive integer")
    edges = list(edges)
    if not edges:
        raise ValueError("positive-map graph has no edges")
    by_source = defaultdict(list)
    ids = set()
    for edge in edges:
        if edge["edge_id"] in ids:
            raise ValueError("positive-map edge identifiers are duplicated")
        ids.add(edge["edge_id"])
        by_source[edge["source"]].append(edge)
    paths = [[edge] for edge in edges]
    for _ in range(1, length):
        paths = [
            path + [successor]
            for path in paths
            for successor in by_source.get(path[-1]["target"], ())
        ]
    if not paths:
        raise ValueError("positive-map graph has no complete registered path")
    return [compose_affine_maps(path) for path in paths]


def entrywise_path_envelope(compositions):
    """Take an entrywise envelope over a nonempty set of path maps."""

    compositions = list(compositions)
    if not compositions:
        raise ValueError("path envelope needs at least one composition")
    matrices = [_array(row["matrix"], "path matrix", ndim=2)
                for row in compositions]
    forcings = [_array(row["forcing"], "path forcing", ndim=1)
                for row in compositions]
    if any(matrix.shape != matrices[0].shape for matrix in matrices) or any(
        forcing.shape != forcings[0].shape for forcing in forcings
    ):
        raise ValueError("path compositions have inconsistent dimensions")
    matrix = np.maximum.reduce(matrices)
    forcing = np.maximum.reduce(forcings)
    return {
        "matrix": matrix.tolist(),
        "forcing": forcing.tolist(),
        "path_count": len(compositions),
        "path_ids": [row["edge_ids"] for row in compositions],
    }


def block_diagonal(matrices):
    """Small dependency-free block-diagonal helper for qualification tests."""

    matrices = [_array(value, "block diagonal entry", ndim=2)
                for value in matrices]
    if not matrices:
        raise ValueError("block diagonal needs at least one matrix")
    rows = sum(value.shape[0] for value in matrices)
    columns = sum(value.shape[1] for value in matrices)
    result = np.zeros((rows, columns))
    row_offset = column_offset = 0
    for value in matrices:
        r, c = value.shape
        result[row_offset:row_offset + r, column_offset:column_offset + c] = value
        row_offset += r
        column_offset += c
    return result
