"""Derivative-defined normal closure for the complete row-Jacobian stack.

The constructor freezes one right-inverse witness and one reference forcing on
a construction partition.  Validation cells are then evaluated without
refitting either object.  A least-squares fit to a single pair ``(Z, Q)`` is
deliberately not an accepted input to this module.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from full_stack import outward_upper


SCHEMA_VERSION = "pldr-dense-normal-closure-v1"


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=float)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return result


def _partition(values, name):
    values = tuple(values)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise ValueError(f"{name} must contain nonempty identifiers")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicate identifiers")
    return values


def _roundoff_bounds(value):
    required = {
        "right_inverse_residual",
        "closure_complement",
        "validation_center_residual",
        "validation_derivative_residual",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError(
            "normal-closure runtime roundoff ledger is incomplete")
    result = {}
    for name, bound in value.items():
        bound = float(bound)
        if not math.isfinite(bound) or bound < 0:
            raise ValueError(
                "normal-closure runtime roundoff bounds must be nonnegative")
        result[name] = bound
    return result


def _digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _cell(value, dimension, parameter_dimension):
    required = {
        "cell_id", "stack", "normal_velocity", "stack_parameter_jacobian",
        "normal_velocity_parameter_jacobian", "cell_radius",
        "second_derivative_bound", "affine_residual_proof",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("normal validation cell has missing or unknown fields")
    if not isinstance(value["cell_id"], str) or not value["cell_id"]:
        raise ValueError("normal validation cell_id is invalid")
    stack = _array(value["stack"], "validation stack", ndim=1)
    velocity = _array(
        value["normal_velocity"], "validation normal velocity", ndim=1)
    stack_jacobian = _array(
        value["stack_parameter_jacobian"],
        "validation stack parameter Jacobian", ndim=2)
    velocity_jacobian = _array(
        value["normal_velocity_parameter_jacobian"],
        "validation velocity parameter Jacobian", ndim=2)
    if (
        stack.shape != (dimension,)
        or velocity.shape != (dimension,)
        or stack_jacobian.shape != (dimension, parameter_dimension)
        or velocity_jacobian.shape != (dimension, parameter_dimension)
    ):
        raise ValueError("normal validation cell dimensions disagree")
    radius = float(value["cell_radius"])
    second = float(value["second_derivative_bound"])
    if (
        not math.isfinite(radius) or radius < 0
        or not math.isfinite(second) or second < 0
    ):
        raise ValueError("normal validation radii must be finite and nonnegative")
    affine_proof = value["affine_residual_proof"]
    if not isinstance(affine_proof, bool):
        raise TypeError("affine_residual_proof must be Boolean")
    if second == 0.0 and not affine_proof:
        raise ValueError(
            "a zero nonlinear residual needs an exact affine-residual proof")
    if second > 0.0 and affine_proof:
        raise ValueError("an affine proof conflicts with a nonzero second derivative")
    return {
        "cell_id": value["cell_id"],
        "stack": stack,
        "normal_velocity": velocity,
        "stack_parameter_jacobian": stack_jacobian,
        "normal_velocity_parameter_jacobian": velocity_jacobian,
        "cell_radius": radius,
        "second_derivative_bound": second,
        "affine_residual_proof": affine_proof,
    }


def construct_dense_normal_closure(
        *, stack_parameter_jacobian, normal_velocity_parameter_jacobian,
        stack_center, normal_velocity_center, construction_partition,
        validation_partition, validation_cells, runtime_roundoff_bounds,
        rank_tolerance=1e-12):
    """Construct ``R``, ``K`` and a frozen forcing from derivatives.

    Let ``B = D_theta Psi`` and ``DQ = D_theta Q`` at the construction
    center.  The returned maps are ``R = pinv(B)``, ``K = DQ R`` and
    ``f = Q_bar - K Psi_bar``.  Acceptance requires ``||B R - I|| < 1``.
    The closure complement ``DQ - K B`` and validation residuals are measured
    independently and carried in the record; they are never set to zero by
    construction.
    """
    roundoff = _roundoff_bounds(runtime_roundoff_bounds)

    construction = _partition(construction_partition, "construction partition")
    validation = _partition(validation_partition, "validation partition")
    if set(construction) & set(validation):
        raise ValueError("construction and validation partitions overlap")
    B = _array(
        stack_parameter_jacobian, "stack parameter Jacobian", ndim=2)
    DQ = _array(
        normal_velocity_parameter_jacobian,
        "normal velocity parameter Jacobian", ndim=2)
    z = _array(stack_center, "stack center", ndim=1)
    q = _array(normal_velocity_center, "normal velocity center", ndim=1)
    dimension, parameter_dimension = B.shape
    if dimension < 1 or parameter_dimension < 1:
        raise ValueError("normal closure dimensions must be nonzero")
    if DQ.shape != B.shape or z.shape != (dimension,) or q.shape != (dimension,):
        raise ValueError("normal construction dimensions disagree")
    rank_tolerance = float(rank_tolerance)
    if not math.isfinite(rank_tolerance) or rank_tolerance <= 0:
        raise ValueError("rank_tolerance must be finite and positive")
    singular_values = np.linalg.svd(B, compute_uv=False)
    rank = int(np.sum(singular_values > rank_tolerance * singular_values[0]))
    if rank != dimension:
        raise ValueError(
            "complete stack Jacobian has no certified full-row-rank chart")
    right_inverse = np.linalg.pinv(B, rcond=rank_tolerance)
    inverse_residual = outward_upper(np.linalg.norm(
        B @ right_inverse - np.eye(dimension), ord=2)
        + roundoff["right_inverse_residual"])
    if not inverse_residual < 1.0:
        raise ValueError("approximate right-inverse residual is not below one")
    normal = DQ @ right_inverse
    complement = DQ - normal @ B
    complement_norm = outward_upper(
        np.linalg.norm(complement, ord=2) + roundoff["closure_complement"])
    forcing = q - normal @ z

    cells = [_cell(value, dimension, parameter_dimension)
             for value in validation_cells]
    if not cells:
        raise ValueError("normal closure needs at least one validation cell")
    if {value["cell_id"] for value in cells} != set(validation):
        raise ValueError("validation cells do not equal the frozen partition")
    residual_rows = []
    for value in cells:
        residual = value["normal_velocity"] - normal @ value["stack"] - forcing
        derivative = (
            value["normal_velocity_parameter_jacobian"]
            - normal @ value["stack_parameter_jacobian"]
        )
        residual_norm = outward_upper(
            np.linalg.norm(residual, ord=2)
            + roundoff["validation_center_residual"])
        derivative_norm = outward_upper(
            np.linalg.norm(derivative, ord=2)
            + roundoff["validation_derivative_residual"])
        nonlinear = value["second_derivative_bound"]
        radius = value["cell_radius"]
        envelope = outward_upper(
            residual_norm + derivative_norm * radius
            + 0.5 * nonlinear * radius * radius)
        residual_rows.append({
            "cell_id": value["cell_id"],
            "center_residual_norm": residual_norm,
            "first_derivative_residual_bound": derivative_norm,
            "second_derivative_residual_bound": nonlinear,
            "cell_radius": radius,
            "residual_envelope": envelope,
            "affine_residual_proof": value["affine_residual_proof"],
        })
    record = {
        "schema_version": SCHEMA_VERSION,
        "mode": "dense_complete_stack",
        "stack_dimension": dimension,
        "parameter_dimension": parameter_dimension,
        "construction_partition": list(construction),
        "validation_partition": list(validation),
        "rank_tolerance": rank_tolerance,
        "construction_rank": rank,
        "construction_singular_values": singular_values.tolist(),
        "right_inverse": right_inverse.tolist(),
        "right_inverse_residual": inverse_residual,
        "normal_operator": normal.tolist(),
        "closure_complement_norm": complement_norm,
        "reference_forcing": forcing.tolist(),
        "reference_forcing_rule": "Q_bar_minus_K_Psi_bar_once_on_construction",
        "validation_residuals": residual_rows,
        "maximum_validation_residual_envelope": max(
            row["residual_envelope"] for row in residual_rows),
        "runtime_roundoff_bounds": roundoff,
        "eigenvalue_summary_role": "diagnostic_only",
    }
    record["record_sha256"] = _digest(record)
    return record


def check_dense_normal_closure(
        record, *, stack_parameter_jacobian,
        normal_velocity_parameter_jacobian, stack_center,
        normal_velocity_center, validation_cells):
    """Independently replay a dense normal-closure construction."""

    if not isinstance(record, dict) or record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown normal-closure record")
    unsigned = dict(record)
    digest = unsigned.pop("record_sha256", None)
    if digest != _digest(unsigned):
        raise ValueError("normal-closure record digest does not replay")
    rebuilt = construct_dense_normal_closure(
        stack_parameter_jacobian=stack_parameter_jacobian,
        normal_velocity_parameter_jacobian=
            normal_velocity_parameter_jacobian,
        stack_center=stack_center,
        normal_velocity_center=normal_velocity_center,
        construction_partition=record["construction_partition"],
        validation_partition=record["validation_partition"],
        validation_cells=validation_cells,
        runtime_roundoff_bounds=record["runtime_roundoff_bounds"],
        rank_tolerance=record["rank_tolerance"],
    )
    if rebuilt != record:
        raise ValueError("normal-closure record differs from independent replay")
    return True


def residual_envelope(record, cell_id):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown normal-closure record")
    rows = [row for row in record["validation_residuals"]
            if row["cell_id"] == cell_id]
    if len(rows) != 1:
        raise ValueError("normal-closure cell is absent or duplicated")
    return rows[0]["residual_envelope"]
