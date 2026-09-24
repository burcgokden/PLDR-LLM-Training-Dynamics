"""Canonical complete state and owned branch-resolved successor bindings."""

from __future__ import annotations

import hashlib
import json

import numpy as np


STATE_SCHEMA_VERSION = "pldr-complete-program-state-v1"
EDGE_SCHEMA_VERSION = "pldr-owned-branch-successor-v1"

STATE_FIELDS = (
    "model_tensors",
    "optimizer_tensors",
    "optimizer_step",
    "optimizer_groups",
    "scheduler_state",
    "amp_state",
    "gradient_accumulation",
    "rng_states",
    "data_state",
    "intervention_state",
    "branch_signature",
    "physical_rows",
    "row_registry",
    "model_config",
    "operation_order",
    "code_manifest",
)

UPDATE_OPERATION_ORDER = (
    "restore_rng",
    "load_registered_batch",
    "forward",
    "loss",
    "backward",
    "clip",
    "update_first_moment",
    "update_second_moment",
    "bias_correction",
    "adaptive_preconditioner",
    "decoupled_weight_decay",
    "parameter_update",
    "scheduler_update",
    "recapture_physical_rows",
)


def _canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def digest_object(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def tensor_record(name, value):
    """Serialize one tensor with metadata and its canonical raw bytes."""

    if not isinstance(name, str) or not name:
        raise ValueError("tensor name must be a nonempty string")
    if hasattr(value, "detach"):
        value = value.detach().cpu().contiguous().numpy()
    array = np.ascontiguousarray(np.asarray(value))
    if array.dtype.hasobject:
        raise ValueError("object tensors are not canonical program state")
    raw = array.tobytes(order="C")
    byte_order = array.dtype.byteorder
    if byte_order == "=":
        byte_order = "little" if np.little_endian else "big"
    elif byte_order == "|":
        byte_order = "not_applicable"
    elif byte_order == "<":
        byte_order = "little"
    else:
        byte_order = "big"
    return {
        "name": name,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "byte_order": byte_order,
        "data_hex": raw.hex(),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _validate_tensor_records(values, label):
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a list")
    names = set()
    required = {
        "name", "dtype", "shape", "byte_order", "data_hex", "data_sha256",
    }
    for record in values:
        if not isinstance(record, dict) or set(record) != required:
            raise ValueError(f"{label} contains a malformed tensor record")
        if not isinstance(record["name"], str) or not record["name"]:
            raise ValueError(f"{label} contains an invalid tensor name")
        if record["name"] in names:
            raise ValueError(f"{label} contains duplicate tensor names")
        names.add(record["name"])
        if (
            not isinstance(record["shape"], list)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in record["shape"]
            )
        ):
            raise ValueError(f"{label} contains an invalid tensor shape")
        try:
            raw = bytes.fromhex(record["data_hex"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{label} contains invalid tensor bytes") from error
        if hashlib.sha256(raw).hexdigest() != record["data_sha256"]:
            raise ValueError(f"{label} tensor digest does not match its bytes")
        dtype = np.dtype(record["dtype"])
        dtype_byte_order = dtype.byteorder
        if dtype_byte_order == "=":
            expected_byte_order = "little" if np.little_endian else "big"
        elif dtype_byte_order == "|":
            expected_byte_order = "not_applicable"
        elif dtype_byte_order == "<":
            expected_byte_order = "little"
        else:
            expected_byte_order = "big"
        if record["byte_order"] != expected_byte_order:
            raise ValueError(
                f"{label} tensor byte order disagrees with its dtype")
        count = int(np.prod(record["shape"], dtype=np.int64))
        if len(raw) != count * dtype.itemsize:
            raise ValueError(f"{label} tensor byte count disagrees with metadata")


def build_complete_state(**fields):
    """Validate and digest every state component used by one live update."""

    if set(fields) != set(STATE_FIELDS):
        raise ValueError("complete state has missing or unknown fields")
    _validate_tensor_records(fields["model_tensors"], "model_tensors")
    _validate_tensor_records(fields["optimizer_tensors"], "optimizer_tensors")
    _validate_tensor_records(fields["physical_rows"], "physical_rows")
    step = fields["optimizer_step"]
    if not isinstance(step, int) or isinstance(step, bool) or step < 0:
        raise ValueError("optimizer_step must be a nonnegative integer")
    for name in (
        "optimizer_groups", "scheduler_state", "amp_state",
        "gradient_accumulation", "rng_states", "data_state",
        "intervention_state", "branch_signature", "row_registry",
        "model_config", "code_manifest",
    ):
        if not isinstance(fields[name], dict):
            raise ValueError(f"{name} must be a mapping")
    if (
        not isinstance(fields["operation_order"], list)
        or not fields["operation_order"]
        or any(not isinstance(value, str) or not value
               for value in fields["operation_order"])
    ):
        raise ValueError("operation_order must be a nonempty string list")
    value = {"schema_version": STATE_SCHEMA_VERSION, **fields}
    value["state_sha256"] = digest_object(value)
    return value


def validate_complete_state(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"schema_version", *STATE_FIELDS, "state_sha256"}
        or value.get("schema_version") != STATE_SCHEMA_VERSION
    ):
        raise ValueError("unknown or malformed complete program state")
    unsigned = dict(value)
    digest = unsigned.pop("state_sha256")
    if digest != digest_object(unsigned):
        raise ValueError("complete state digest does not replay")
    rebuilt_fields = {
        name: unsigned[name] for name in STATE_FIELDS
    }
    rebuilt = build_complete_state(**rebuilt_fields)
    if rebuilt != value:
        raise ValueError("complete state does not replay canonically")
    return True


def bind_owned_successor(
        *, edge_id, source_state, target_state, update_direction_sha256,
        constructor_sha256, operation_trace):
    """Bind one executed successor without subtracting its discrete state."""

    validate_complete_state(source_state)
    validate_complete_state(target_state)
    if not isinstance(edge_id, str) or not edge_id:
        raise ValueError("edge_id must be a nonempty string")
    if target_state["optimizer_step"] != source_state["optimizer_step"] + 1:
        raise ValueError("target is not the next optimizer-clock state")
    if source_state["branch_signature"] != target_state["branch_signature"]:
        raise ValueError(
            "continuous differentiation crossed a branch; split the edge")
    if tuple(operation_trace) != UPDATE_OPERATION_ORDER:
        raise ValueError("operation trace does not match the implemented update order")
    if (
        tuple(source_state["operation_order"]) != tuple(operation_trace)
        or tuple(target_state["operation_order"]) != tuple(operation_trace)
    ):
        raise ValueError("complete states do not own the declared operation trace")
    for label, value in (
        ("update_direction_sha256", update_direction_sha256),
        ("constructor_sha256", constructor_sha256),
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"{label} is not a lowercase SHA-256 digest")
    record = {
        "schema_version": EDGE_SCHEMA_VERSION,
        "edge_id": edge_id,
        "source_state_sha256": source_state["state_sha256"],
        "target_state_sha256": target_state["state_sha256"],
        "source_optimizer_step": source_state["optimizer_step"],
        "target_optimizer_step": target_state["optimizer_step"],
        "branch_signature": source_state["branch_signature"],
        "update_direction_sha256": update_direction_sha256,
        "constructor_sha256": constructor_sha256,
        "operation_trace": list(operation_trace),
        "continuous_displacement_only": True,
    }
    record["record_sha256"] = digest_object(record)
    return record


def verify_replayed_target(edge, replayed_target_state):
    """Check an independently executed target against the owned edge."""

    if (
        not isinstance(edge, dict)
        or edge.get("schema_version") != EDGE_SCHEMA_VERSION
    ):
        raise ValueError("unknown owned-successor record")
    unsigned = dict(edge)
    digest = unsigned.pop("record_sha256", None)
    if digest != digest_object(unsigned):
        raise ValueError("owned-successor digest does not replay")
    validate_complete_state(replayed_target_state)
    if replayed_target_state["state_sha256"] != edge["target_state_sha256"]:
        raise ValueError("independent replay did not regenerate the owned target")
    return {
        "schema_version": "pldr-owned-successor-replay-v1",
        "edge_record_sha256": edge["record_sha256"],
        "replayed_target_state_sha256": replayed_target_state["state_sha256"],
        "decision": "REPLAYED",
    }
