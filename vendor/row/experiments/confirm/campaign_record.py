"""Strict raw-record contract for the row-map confirmation program.

Raw records are closed at every structural level, reject duplicate JSON keys
and nonfinite numbers, carry one global optimizer clock, and bind every
protocol and data dependency by a digest recomputed from its on-disk path.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

from campaign_unit_key import (
    canonical_unit_key,
    schema_object as unit_key_schema_object,
    validate_unit_key,
)

SCHEMA_VERSION = "pldr-row-map-direct-certificate-v8"
PROTOCOLS = tuple(f"E{index}" for index in range(9))
BINDING_NAMES = (
    "protocol", "schema", "registration", "estimator_manifest",
    "source", "model", "data_order", "tokenizer",
)
UNIT_KEY_NAMES = (
    "seed", "checkpoint", "layer_bundle", "tensor_group",
    "certificate_box", "schedule_cell", "architecture_cell",
    "block_size", "arm",
)
MEASUREMENT_NAMES = {
    "E0": (
        "direct_row_map_upper", "grid_jacobian_upper", "cover_remainder",
        "cover_identity_error", "heldout_jacobian_ratio",
        "heldout_cover_radius_ratio",
        "mixed_derivative_enclosure_ratio",
        "third_derivative_enclosure_ratio", "numerical_error",
        "cover_utility_margin", "validation_domain_membership_indicator",
    ),
    "E1": (
        "adamw_ledger_residual", "lifted_recurrence_residual",
        "defect_recomposition_residual",
        "coordinate_motion_enclosure_ratio",
        "taylor_remainder_enclosure_ratio",
        "intervention_boundary_residual",
        "applied_learning_rate", "bias_corrected_step",
        "schedule_increment", "decay_coefficient",
        "optimizer_state_step",
    ),
    "E2": (
        "lower_normal_edge", "upper_normal_edge", "jury_margin_min",
        "jury_lower_margin", "jury_upper_margin", "spectral_loading",
        "normal_operator_self_adjoint_residual",
        "normal_linearization_residual_ratio",
        "nonlinear_residual_enclosure_ratio",
        "normal_operator_identity_residual",
    ),
    "E3": (
        "nominal_lyapunov_residual", "metric_eigenvalue_min",
        "nominal_q", "structured_family_q", "structured_family_slack",
        "maximum_corner_h_gain", "multiaffine_reconstruction_residual",
        "primitive_box_enclosure_ratio", "path_window_gain",
        "observed_h_gain",
    ),
    "E4": (
        "forcing_norm", "complete_defect_norm",
        "complete_disturbance_norm", "persistent_disturbance_floor",
        "disturbance_envelope", "envelope_ratio", "geometric_rate",
        "positive_comparison_witness_ratio",
        "defect_recomposition_residual",
    ),
    "E5": (
        "initial_membership_margin", "lifted_budget_slack",
        "path_window_length", "path_window_gain",
        "block_lifted_budget_slack", "prefix_budget_slack_min",
        "auxiliary_budget_slack_min", "heldout_image_margin",
        "one_step_escape_count", "family_box_enclosure_ratio",
    ),
    "E6": (
        "predicted_entry_step", "observed_entry_step", "entry_error",
        "residual_floor", "direct_row_map_upper", "criterion",
        "sustained_entry_indicator",
        "path_window_length", "path_window_gain",
        "block_product_convolution_upper",
        "schedule_product_convolution_upper",
        "schedule_predicted_entry_step",
    ),
    "E7": (
        "robust_q", "complete_disturbance_envelope",
        "predicted_entry_step", "direct_row_map_upper", "heldout_loss",
        "intervention_dose", "jury_lower_margin",
        "jury_upper_margin", "spectral_loading",
        "schedule_product_convolution_upper",
        "avalanche_event_rate", "avalanche_mean_size",
        "avalanche_mean_support",
    ),
    "E8": (
        "model_width", "model_depth", "direct_row_map_upper",
        "criterion",
        "predicted_entry_step", "observed_entry_step", "heldout_loss",
        "baseline_indicator", "model_score", "theorem_bound_ratio",
        "maximum_learning_rate", "warmup_steps", "anneal_floor",
        "source_order_parameter", "source_rmse", "tensor_mean_abs",
        "row_input_spread", "downstream_lipschitz",
        "order_parameter_upper", "order_parameter_bridge_ratio",
        "avalanche_event_rate", "avalanche_mean_size",
        "avalanche_mean_duration", "avalanche_mean_support",
        "avalanche_tail_comparison", "finite_size_scaling_error",
        "rg_block_size", "rg_density_identity_residual",
        "rg_layernorm_scale_residual", "rg_within_covariance_trace",
        "rg_deductive_closure_upper", "rg_deductive_closure_ratio",
        "rg_softmax_aggregation_residual",
        "rg_repeated_block_semigroup_residual",
        "rg_full_layer_closure_upper", "rg_full_layer_closure_ratio",
    ),
}
STATUS = ("OBSERVED", "MISSING", "NOT_APPLICABLE", "ERROR")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class RecordError(ValueError):
    pass


def _exact_mapping(value, names, label):
    if not isinstance(value, dict):
        raise RecordError(f"{label} must be an object")
    missing = sorted(set(names) - set(value))
    extra = sorted(set(value) - set(names))
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("unknown " + ", ".join(extra))
        raise RecordError(f"{label}: {'; '.join(parts)}")
    return value


def _identifier(value, label):
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise RecordError(f"{label} is not a valid identifier")
    return value


def _integer(value, label, lower=0):
    if not isinstance(value, int) or isinstance(value, bool) or value < lower:
        raise RecordError(f"{label} must be an integer at least {lower}")
    return value


def _number(value, label):
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise RecordError(f"{label} must be a finite number")
    return float(value)


def _string(value, label, allow_empty=False):
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RecordError(f"{label} must be a nonempty string")
    return value


def _optional_string(value, label):
    if value is not None:
        _string(value, label)
    return value


def _duplicates_rejected(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise RecordError(f"duplicate JSON key: {name}")
        result[name] = value
    return result


def strict_load(path):
    path = Path(path)

    def reject_constant(value):
        raise RecordError(f"nonfinite JSON token: {value}")

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=_duplicates_rejected,
                parse_constant=reject_constant,
            )
    except json.JSONDecodeError as error:
        raise RecordError(f"invalid JSON: {error}") from error


def strict_dumps(value):
    return json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
        separators=(",", ": "),
    ) + "\n"


def write_strict(path, value, *, exclusive=False):
    path = Path(path)
    validate_record(value)
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        stream.write(strict_dumps(value))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def binding_for(path, root):
    root = Path(root).resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise RecordError("binding path lies outside the declared root") from error
    if not path.is_file():
        raise RecordError(f"binding is not a regular file: {relative}")
    return {"path": relative.as_posix(), "sha256": sha256_file(path)}


def verify_binding(binding, root, label):
    binding = _exact_mapping(binding, ("path", "sha256"), label)
    relative = Path(_string(binding["path"], f"{label}.path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise RecordError(f"{label}.path is not a safe relative path")
    digest = _string(binding["sha256"], f"{label}.sha256")
    if not _SHA256.fullmatch(digest):
        raise RecordError(f"{label}.sha256 is not lowercase SHA-256")
    root = Path(root).resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise RecordError(f"{label}.path escapes the binding root") from error
    if not target.is_file():
        raise RecordError(f"{label}.path does not exist")
    realized = sha256_file(target)
    if realized != digest:
        raise RecordError(
            f"{label} is stale: recorded {digest}, realized {realized}")
    return target


def _validate_clocks(value):
    value = _exact_mapping(value, (
        "global_optimizer_step", "segment_index", "segment_local_step",
        "segment_global_offset", "checkpoint_global_step",
    ), "clocks")
    for name in value:
        _integer(value[name], f"clocks.{name}")
    if (
        value["global_optimizer_step"]
        != value["segment_global_offset"] + value["segment_local_step"]
    ):
        raise RecordError("global clock disagrees with segment clock")
    if value["checkpoint_global_step"] != value["global_optimizer_step"]:
        raise RecordError("checkpoint clock disagrees with global clock")


def _validate_bindings(value):
    value = _exact_mapping(value, BINDING_NAMES, "bindings")
    for name, binding in value.items():
        binding = _exact_mapping(
            binding, ("path", "sha256"), f"bindings.{name}")
        _string(binding["path"], f"bindings.{name}.path")
        digest = _string(binding["sha256"], f"bindings.{name}.sha256")
        if not _SHA256.fullmatch(digest):
            raise RecordError(
                f"bindings.{name}.sha256 is not lowercase SHA-256")


def _validate_tensor_groups(value):
    if not isinstance(value, list) or not value:
        raise RecordError("tensor_groups must be a nonempty array")
    names = set()
    for index, group in enumerate(value):
        label = f"tensor_groups[{index}]"
        group = _exact_mapping(group, ("name", "parameters"), label)
        name = _identifier(group["name"], f"{label}.name")
        if name in names:
            raise RecordError(f"duplicate tensor group: {name}")
        names.add(name)
        parameters = group["parameters"]
        if not isinstance(parameters, list) or not parameters:
            raise RecordError(f"{label}.parameters must be nonempty")
        for pindex, parameter in enumerate(parameters):
            _string(parameter, f"{label}.parameters[{pindex}]")
        if len(set(parameters)) != len(parameters):
            raise RecordError(f"{label}.parameters contains duplicates")


def _validate_intervention(value):
    value = _exact_mapping(value, (
        "arm", "assigned_dose", "applied_after_optimizer", "description",
    ), "intervention")
    _identifier(value["arm"], "intervention.arm")
    if value["assigned_dose"] is not None:
        _number(value["assigned_dose"], "intervention.assigned_dose")
    if not isinstance(value["applied_after_optimizer"], bool):
        raise RecordError(
            "intervention.applied_after_optimizer must be Boolean")
    _string(value["description"], "intervention.description")


def _validate_measurements(protocol_id, value):
    if not isinstance(value, list):
        raise RecordError("measurements must be an array")
    rows = {}
    for index, row in enumerate(value):
        label = f"measurements[{index}]"
        row = _exact_mapping(row, (
            "name", "value", "unit", "method", "status", "reason_code",
        ), label)
        name = _identifier(row["name"], f"{label}.name")
        if name in rows:
            raise RecordError(f"duplicate measurement: {name}")
        rows[name] = row
        _string(row["unit"], f"{label}.unit")
        _string(row["method"], f"{label}.method")
        if row["status"] not in STATUS:
            raise RecordError(f"{label}.status is invalid")
        _optional_string(row["reason_code"], f"{label}.reason_code")
        if row["status"] == "OBSERVED":
            _number(row["value"], f"{label}.value")
            if row["reason_code"] is not None:
                raise RecordError(
                    f"{label}.reason_code must be null when observed")
        else:
            if row["value"] is not None:
                raise RecordError(
                    f"{label}.value must be null when not observed")
            if row["reason_code"] is None:
                raise RecordError(
                    f"{label}.reason_code is required when not observed")
    required = set(MEASUREMENT_NAMES[protocol_id])
    if set(rows) != required:
        missing = sorted(required - set(rows))
        extra = sorted(set(rows) - required)
        raise RecordError(
            f"measurements do not match {protocol_id}: "
            f"missing={missing}, unknown={extra}")


def _validate_artifacts(value):
    if not isinstance(value, list):
        raise RecordError("artifacts must be an array")
    names = set()
    for index, row in enumerate(value):
        label = f"artifacts[{index}]"
        row = _exact_mapping(
            row, ("name", "path", "sha256", "media_type"), label)
        name = _identifier(row["name"], f"{label}.name")
        if name in names:
            raise RecordError(f"duplicate artifact: {name}")
        names.add(name)
        _string(row["path"], f"{label}.path")
        digest = _string(row["sha256"], f"{label}.sha256")
        if not _SHA256.fullmatch(digest):
            raise RecordError(f"{label}.sha256 is invalid")
        _string(row["media_type"], f"{label}.media_type")


def _validate_environment(value):
    value = _exact_mapping(value, (
        "python", "torch", "numpy", "device", "dtype", "git_commit",
        "source_tree_clean",
    ), "environment")
    for name in ("python", "torch", "numpy", "device", "dtype"):
        _string(value[name], f"environment.{name}")
    commit = _string(value["git_commit"], "environment.git_commit")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RecordError("environment.git_commit must be a full commit hash")
    if not isinstance(value["source_tree_clean"], bool):
        raise RecordError("environment.source_tree_clean must be Boolean")


def validate_record(record):
    record = _exact_mapping(record, (
        "schema_version", "campaign_id", "protocol_id", "run_id", "seed",
        "unit_key", "clocks", "bindings", "tensor_groups", "intervention",
        "measurements", "artifacts", "environment",
    ), "record")
    if record["schema_version"] != SCHEMA_VERSION:
        raise RecordError("schema_version is missing or stale")
    _identifier(record["campaign_id"], "campaign_id")
    protocol_id = record["protocol_id"]
    if protocol_id not in PROTOCOLS:
        raise RecordError("protocol_id must be E0 through E8")
    _identifier(record["run_id"], "run_id")
    _integer(record["seed"], "seed")
    _validate_clocks(record["clocks"])
    validate_unit_key(record["unit_key"], record, RecordError)
    _validate_bindings(record["bindings"])
    _validate_tensor_groups(record["tensor_groups"])
    _validate_intervention(record["intervention"])
    _validate_measurements(protocol_id, record["measurements"])
    _validate_artifacts(record["artifacts"])
    _validate_environment(record["environment"])
    return record


def load_and_verify(path, binding_root, artifact_root=None):
    record = validate_record(strict_load(path))
    for name in BINDING_NAMES:
        verify_binding(record["bindings"][name], binding_root,
                       f"bindings.{name}")
    artifact_root = (
        Path(artifact_root).resolve()
        if artifact_root is not None else Path(path).resolve().parent
    )
    for index, row in enumerate(record["artifacts"]):
        binding = {"path": row["path"], "sha256": row["sha256"]}
        verify_binding(binding, artifact_root, f"artifacts[{index}]")
    return record


def measurement_map(record, *, require_observed=False):
    validate_record(record)
    result = {row["name"]: row for row in record["measurements"]}
    if require_observed:
        missing = [
            name for name, row in result.items()
            if row["status"] != "OBSERVED"
        ]
        if missing:
            raise RecordError(
                "required measurements are unavailable: " + ", ".join(missing))
    return result


def schema_object():
    """Return the machine-readable closed JSON Schema companion."""
    measurement_names = sorted({
        name for names in MEASUREMENT_NAMES.values() for name in names
    })
    binding = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "sha256"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "sha256": {
                "type": "string", "pattern": "^[0-9a-f]{64}$",
            },
        },
    }
    measurement = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "name", "value", "unit", "method", "status", "reason_code",
        ],
        "properties": {
            "name": {"enum": measurement_names},
            "value": {"type": ["number", "null"]},
            "unit": {"type": "string", "minLength": 1},
            "method": {"type": "string", "minLength": 1},
            "status": {"enum": list(STATUS)},
            "reason_code": {"type": ["string", "null"]},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_VERSION,
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "campaign_id", "protocol_id", "run_id",
            "seed", "unit_key", "clocks", "bindings", "tensor_groups",
            "intervention", "measurements", "artifacts", "environment",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "campaign_id": {"type": "string", "pattern": _IDENTIFIER.pattern},
            "protocol_id": {"enum": list(PROTOCOLS)},
            "run_id": {"type": "string", "pattern": _IDENTIFIER.pattern},
            "seed": {"type": "integer", "minimum": 0},
            "unit_key": unit_key_schema_object(),
            "clocks": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "global_optimizer_step", "segment_index",
                    "segment_local_step", "segment_global_offset",
                    "checkpoint_global_step",
                ],
                "properties": {
                    name: {"type": "integer", "minimum": 0}
                    for name in (
                        "global_optimizer_step", "segment_index",
                        "segment_local_step", "segment_global_offset",
                        "checkpoint_global_step",
                    )
                },
            },
            "bindings": {
                "type": "object",
                "additionalProperties": False,
                "required": list(BINDING_NAMES),
                "properties": {name: binding for name in BINDING_NAMES},
            },
            "tensor_groups": {
                "type": "array", "minItems": 1,
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "parameters"],
                    "properties": {
                        "name": {
                            "type": "string", "pattern": _IDENTIFIER.pattern,
                        },
                        "parameters": {
                            "type": "array", "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                    },
                },
            },
            "intervention": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "arm", "assigned_dose", "applied_after_optimizer",
                    "description",
                ],
                "properties": {
                    "arm": {
                        "type": "string", "pattern": _IDENTIFIER.pattern,
                    },
                    "assigned_dose": {"type": ["number", "null"]},
                    "applied_after_optimizer": {"type": "boolean"},
                    "description": {"type": "string", "minLength": 1},
                },
            },
            "measurements": {
                "type": "array", "items": measurement,
            },
            "artifacts": {
                "type": "array",
                "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["name", "path", "sha256", "media_type"],
                    "properties": {
                        "name": {
                            "type": "string", "pattern": _IDENTIFIER.pattern,
                        },
                        "path": {"type": "string", "minLength": 1},
                        "sha256": {
                            "type": "string",
                            "pattern": "^[0-9a-f]{64}$",
                        },
                        "media_type": {"type": "string", "minLength": 1},
                    },
                },
            },
            "environment": {
                "type": "object", "additionalProperties": False,
                "required": [
                    "python", "torch", "numpy", "device", "dtype",
                    "git_commit", "source_tree_clean",
                ],
                "properties": {
                    **{
                        name: {"type": "string", "minLength": 1}
                        for name in (
                            "python", "torch", "numpy", "device", "dtype",
                        )
                    },
                    "git_commit": {
                        "type": "string", "pattern": "^[0-9a-f]{40}$",
                    },
                    "source_tree_clean": {"type": "boolean"},
                },
            },
        },
    }
