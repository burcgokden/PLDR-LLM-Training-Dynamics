"""Program-bound complete row-Jacobian energy primitives.

The functions in this module consume row blocks produced by one owned
program successor.  They do not accept a comparison operator, contraction
factor, affine forcing vector, direct upper bound, target radius, or verdict.
Those quantities are derived from the source and target row-Jacobian fields
and from a closed primitive error ledger.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from full_stack import validate_registry


SCHEMA_VERSION = "pldr-program-energy-primitives-v1"
EDGE_SCHEMA_VERSION = "pldr-program-energy-edge-v1"
ERROR_TERMS = (
    "taylor_remainder",
    "row_cover_motion",
    "metric_conversion",
    "runtime_roundoff",
    "exogenous_input",
)


def canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest_object(value):
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def rational_object(value):
    """Serialize the exact binary rational represented by one float."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("a rationalized value must be finite")
    exact = Fraction.from_float(value)
    return {"numerator": exact.numerator, "denominator": exact.denominator}


def outward_upper(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("an outward upper bound must be finite and nonnegative")
    return value if value == 0 else math.nextafter(value, math.inf)


def outward_lower_nonnegative(value):
    """Round a finite nonnegative quantity toward negative infinity."""

    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("an outward lower bound must be finite and nonnegative")
    if value == 0:
        return 0.0
    lower = math.nextafter(value, -math.inf)
    if lower <= 0:
        raise ValueError("positive metric or energy lower bound underflowed")
    return lower


def runtime_reduction_bound(values, operation_count, unit_roundoff=2.0 ** -53):
    """Higham-style first-order reduction bound derived from primitives.

    The accepted bound is computed from the operation count, the declared
    machine unit roundoff, and the absolute input sum.  A reported floating
    result is not used as its own error radius.
    """

    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError("runtime reduction inputs must be a finite vector")
    if (
        not isinstance(operation_count, int)
        or isinstance(operation_count, bool)
        or operation_count < 0
    ):
        raise ValueError("operation_count must be a nonnegative integer")
    unit_roundoff = float(unit_roundoff)
    if not 0 < unit_roundoff < 1:
        raise ValueError("unit_roundoff must lie in (0, 1)")
    product = operation_count * unit_roundoff
    if product >= 1:
        raise ValueError("operation-count roundoff model is outside its domain")
    gamma = product / (1.0 - product)
    return outward_upper(gamma * float(np.sum(np.abs(array))))


def _vector(value, dimension, label):
    array = np.asarray(value, dtype=float)
    if array.shape != (dimension,) or not np.isfinite(array).all():
        raise ValueError(f"{label} is nonfinite or has the wrong dimension")
    return array


def _metric_blocks(registry, value):
    if not isinstance(value, list) or len(value) != registry["block_count"]:
        raise ValueError("metric list must contain one entry per registry block")
    metrics = []
    for block, row in zip(registry["blocks"], value):
        if not isinstance(row, dict) or "block_index" not in row or "kind" not in row:
            raise ValueError("metric block has missing or unknown fields")
        if row["block_index"] != block["block_index"]:
            raise ValueError("metric block order disagrees with the registry")
        size = block["size"]
        if row["kind"] == "scaled_identity":
            if set(row) != {"block_index", "kind", "scale"}:
                raise ValueError("scaled-identity metric is malformed")
            scale = float(row["scale"])
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError("scaled-identity metric must be positive")
            metrics.append({"kind": "scaled_identity", "data": scale,
                            "minimum": scale, "maximum": scale})
        elif row["kind"] == "diagonal":
            if set(row) != {"block_index", "kind", "diagonal"}:
                raise ValueError("diagonal metric is malformed")
            diagonal = np.asarray(row["diagonal"], dtype=float)
            if (
                diagonal.shape != (size,)
                or not np.isfinite(diagonal).all()
                or np.any(diagonal <= 0)
            ):
                raise ValueError("diagonal metric must be finite and positive")
            metrics.append({"kind": "diagonal", "data": diagonal,
                            "minimum": float(np.min(diagonal)),
                            "maximum": float(np.max(diagonal))})
        elif row["kind"] == "dense_tiny_fixture":
            if set(row) != {"block_index", "kind", "matrix"} or size > 256:
                raise ValueError("dense metrics are restricted to the tiny fixture")
            matrix = np.asarray(row["matrix"], dtype=float)
            if (
                matrix.shape != (size, size)
                or not np.isfinite(matrix).all()
                or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12)
            ):
                raise ValueError("dense energy metric is invalid")
            eigenvalues = np.linalg.eigvalsh(matrix)
            if eigenvalues[0] <= 0:
                raise ValueError("dense energy metric must be positive definite")
            metrics.append({"kind": "dense_tiny_fixture", "data": matrix,
                            "minimum": float(eigenvalues[0]),
                            "maximum": float(eigenvalues[-1])})
        else:
            raise ValueError("unknown energy metric representation")
    return metrics


def _quadratic(metric, left, right=None):
    right = left if right is None else right
    if metric["kind"] == "scaled_identity":
        return metric["data"] * float(left @ right)
    if metric["kind"] == "diagonal":
        return float((left * metric["data"]) @ right)
    return float(left @ metric["data"] @ right)


def _block_slice(block):
    return slice(block["offset"], block["offset"] + block["size"])


def block_energies(registry, stack, metric_blocks):
    """Return the complete energy of every registered full matrix block."""

    validate_registry(registry)
    stack = _vector(stack, registry["dimension"], "row-Jacobian stack")
    metrics = _metric_blocks(registry, metric_blocks)
    rows = []
    for block, metric in zip(registry["blocks"], metrics):
        vector = stack[_block_slice(block)]
        energy = _quadratic(metric, vector)
        rows.append({
            "block_index": block["block_index"],
            "layer": block["layer"],
            "cell_id": block["cell_id"],
            "energy": outward_upper(max(0.0, energy)),
            "minimum_metric_eigenvalue_lower":
                outward_lower_nonnegative(metric["minimum"]),
        })
    return rows


def total_energy(registry, stack, metric_blocks):
    return outward_upper(sum(
        row["energy"] for row in block_energies(
            registry, stack, metric_blocks)
    ))


def exact_energy_increment(registry, source, target, metric_blocks):
    """Compute both sides of the fixed-metric polarization identity."""

    validate_registry(registry)
    dimension = registry["dimension"]
    source = _vector(source, dimension, "source stack")
    target = _vector(target, dimension, "target stack")
    increment = target - source
    metrics = _metric_blocks(registry, metric_blocks)
    source_energy = 0.0
    target_energy = 0.0
    increment_energy = 0.0
    cross = 0.0
    for block, metric in zip(registry["blocks"], metrics):
        section = _block_slice(block)
        z = source[section]
        dz = increment[section]
        zp = target[section]
        source_energy += _quadratic(metric, z)
        target_energy += _quadratic(metric, zp)
        increment_energy += _quadratic(metric, dz)
        cross += _quadratic(metric, z, dz)
    left = target_energy - source_energy
    right = 2.0 * cross + increment_energy
    operation_count = max(1, 8 * dimension)
    roundoff = runtime_reduction_bound(
        np.concatenate([source, target, increment]), operation_count)
    residual = abs(left - right)
    return {
        "source_energy": outward_upper(max(0.0, source_energy)),
        "source_energy_lower":
            outward_lower_nonnegative(max(0.0, source_energy))
            if source_energy > 0 else 0.0,
        "target_energy": outward_upper(max(0.0, target_energy)),
        "increment_energy": outward_upper(max(0.0, increment_energy)),
        "cross_term": float(cross),
        "left_increment": float(left),
        "right_increment": float(right),
        "identity_residual": outward_upper(residual),
        "derived_roundoff_bound": roundoff,
        "identity_enclosed": residual <= roundoff,
        "operation_count": operation_count,
    }


def _primitive_error_ledger(value):
    if not isinstance(value, dict) or set(value) != set(ERROR_TERMS):
        raise ValueError("primitive error ledger is incomplete")
    result = {}
    for name in ERROR_TERMS:
        row = value[name]
        if not isinstance(row, dict) or set(row) != {
            "observed_norm", "certified_upper", "derivation",
        }:
            raise ValueError(f"{name} ledger row is malformed")
        observed = float(row["observed_norm"])
        upper = float(row["certified_upper"])
        if (
            not math.isfinite(observed) or observed < 0
            or not math.isfinite(upper) or upper < 0
            or not isinstance(row["derivation"], str)
            or not row["derivation"]
        ):
            raise ValueError(f"{name} ledger row is invalid")
        result[name] = {
            "observed_norm": observed,
            "certified_upper": upper,
            "derivation": row["derivation"],
            "enclosed": observed <= upper,
        }
    return result


def construct_program_energy_edge(primitives):
    """Derive one energy edge from a single owned source/successor pair."""

    required = {
        "schema_version", "edge_id", "source_state_sha256",
        "target_state_sha256", "successor_owner_sha256", "registry",
        "source_stack", "target_stack", "actual_update_jvp",
        "metric_blocks", "primitive_errors", "block_completeness",
        "construction_partition_sha256", "validation_partition_sha256",
    }
    if not isinstance(primitives, dict) or set(primitives) != required:
        raise ValueError("program-energy primitives have missing or unknown fields")
    if primitives["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unknown program-energy primitive schema")
    for name in (
        "source_state_sha256", "target_state_sha256",
        "successor_owner_sha256", "construction_partition_sha256",
        "validation_partition_sha256",
    ):
        value = primitives[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{name} is not a lowercase SHA-256 digest")
    if (
        primitives["construction_partition_sha256"]
        == primitives["validation_partition_sha256"]
    ):
        raise ValueError("construction and validation partitions must be distinct")
    registry = primitives["registry"]
    validate_registry(registry)
    dimension = registry["dimension"]
    source = _vector(primitives["source_stack"], dimension, "source stack")
    target = _vector(primitives["target_stack"], dimension, "target stack")
    jvp = _vector(
        primitives["actual_update_jvp"], dimension, "actual-update JVP")
    remainder = target - source - jvp
    errors = _primitive_error_ledger(primitives["primitive_errors"])
    metrics = _metric_blocks(registry, primitives["metric_blocks"])
    identity = exact_energy_increment(
        registry, source, target, primitives["metric_blocks"])
    remainder_norm = float(np.linalg.norm(remainder))
    vector_remainder_upper = (
        errors["taylor_remainder"]["certified_upper"]
        + errors["row_cover_motion"]["certified_upper"]
        + errors["runtime_roundoff"]["certified_upper"]
        + errors["exogenous_input"]["certified_upper"]
    )
    metric_charge = errors["metric_conversion"]["certified_upper"]
    all_errors_enclosed = all(row["enclosed"] for row in errors.values())
    ideal_stack = source + jvp
    ideal_energy = outward_upper(sum(
        _quadratic(metric, ideal_stack[_block_slice(block)])
        for block, metric in zip(registry["blocks"], metrics)
    ))
    maximum_metric_eigenvalue = outward_upper(
        max(metric["maximum"] for metric in metrics))

    source_energy = identity["source_energy"]
    target_energy = identity["target_energy"]
    metric_remainder_upper = outward_upper(
        math.sqrt(maximum_metric_eigenvalue) * vector_remainder_upper)
    energy_charge = outward_upper(
        metric_charge
        + 2.0 * math.sqrt(ideal_energy) * metric_remainder_upper
        + metric_remainder_upper ** 2
    )
    source_energy_lower = identity["source_energy_lower"]
    if source_energy_lower > 0:
        q_squared = outward_upper(ideal_energy / source_energy_lower)
        affine_source = energy_charge
    else:
        q_squared = 0.0
        affine_source = outward_upper(ideal_energy + energy_charge)
    derived_upper = outward_upper(
        q_squared * source_energy + affine_source)

    completeness = primitives["block_completeness"]
    if not isinstance(completeness, list):
        raise ValueError("block_completeness must be a list")
    expected = {
        (block["layer"], block["cell_id"], output, input_index)
        for block in registry["blocks"]
        for output in range(block["width"])
        for input_index in range(block["width"])
    }
    observed = set()
    for row in completeness:
        if not isinstance(row, list) or len(row) != 4:
            raise ValueError("block completeness coordinate is malformed")
        observed.add((int(row[0]), str(row[1]), int(row[2]), int(row[3])))
    complete = observed == expected

    conditions = {
        "complete_registry": complete,
        "energy_identity_enclosed": identity["identity_enclosed"],
        "actual_jvp_remainder_enclosed": (
            remainder_norm <= vector_remainder_upper
        ),
        "every_primitive_error_enclosed": all_errors_enclosed,
        "derived_energy_step_holds": target_energy <= derived_upper,
        "state_bindings_well_formed": True,
    }
    record = {
        "schema_version": EDGE_SCHEMA_VERSION,
        "edge_id": primitives["edge_id"],
        "bindings": {
            "source_state_sha256": primitives["source_state_sha256"],
            "target_state_sha256": primitives["target_state_sha256"],
            "successor_owner_sha256": primitives["successor_owner_sha256"],
            "construction_partition_sha256":
                primitives["construction_partition_sha256"],
            "validation_partition_sha256":
                primitives["validation_partition_sha256"],
        },
        "registry": registry,
        "source_stack": source.tolist(),
        "target_stack": target.tolist(),
        "actual_update_jvp": jvp.tolist(),
        "jvp_remainder": remainder.tolist(),
        "jvp_remainder_norm": outward_upper(remainder_norm),
        "jvp_remainder_upper": outward_upper(vector_remainder_upper),
        "metric_blocks": primitives["metric_blocks"],
        "source_block_energies": block_energies(
            registry, source, primitives["metric_blocks"]),
        "target_block_energies": block_energies(
            registry, target, primitives["metric_blocks"]),
        "energy_identity": identity,
        "primitive_errors": errors,
        "derived_comparison": {
            "q_squared": q_squared,
            "affine_source": affine_source,
            "energy_upper": derived_upper,
            "ideal_actual_jvp_energy": ideal_energy,
            "maximum_metric_eigenvalue": maximum_metric_eigenvalue,
            "metric_remainder_upper": metric_remainder_upper,
            "negative_work_upper":
                outward_signed_upper(
                    ideal_energy - source_energy_lower + energy_charge),
            "caller_supplied": False,
        },
        "conditions": conditions,
        "decision": "QUALIFIED" if all(conditions.values()) else "REJECTED",
    }
    record["record_sha256"] = digest_object(record)
    return record


def outward_signed_upper(value):
    """Round a finite signed quantity toward positive infinity."""

    value = float(value)
    if not math.isfinite(value):
        raise ValueError("a signed outward upper bound must be finite")
    return math.nextafter(value, math.inf)
