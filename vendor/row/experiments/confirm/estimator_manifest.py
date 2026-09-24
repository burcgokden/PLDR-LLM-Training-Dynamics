"""Closed ownership registry for every primary confirmation measurement."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import campaign_record


SCHEMA = "pldr-estimator-manifest-v2"
OWNERS = {
    "E0": ("experiments/confirm/measure_row_map.py", "measure_checkpoint"),
    "E1": (
        "experiments/confirm/measure_optimizer_transport.py",
        "measure_live_snapshot",
    ),
    "E2": (
        "experiments/confirm/measure_direct_certificate.py",
        "measure_live_e2",
    ),
    "E3": (
        "experiments/confirm/measure_direct_certificate.py",
        "measure_live_e3",
    ),
    "E4": (
        "experiments/confirm/measure_direct_certificate.py",
        "measure_live_e4",
    ),
    "E5": (
        "experiments/confirm/measure_direct_certificate.py",
        "measure_live_e5",
    ),
    "E6": (
        "experiments/confirm/measure_direct_certificate.py",
        "measure_live_e6",
    ),
    "E7": (
        "experiments/confirm/measure_intervention.py",
        "measure_live_e7",
    ),
    "E8": (
        "experiments/confirm/measure_scale_transfer.py",
        "measure_live_e8",
    ),
}

LOWER_ENDPOINTS = {
    "E0.cover_utility_margin",
    "E0.validation_domain_membership_indicator",
    "E2.lower_normal_edge",
    "E2.jury_margin_min",
    "E2.jury_lower_margin",
    "E2.jury_upper_margin",
    "E3.metric_eigenvalue_min",
    "E3.structured_family_slack",
    "E5.initial_membership_margin",
    "E5.lifted_budget_slack",
    "E5.block_lifted_budget_slack",
    "E5.prefix_budget_slack_min",
    "E5.auxiliary_budget_slack_min",
    "E5.heldout_image_margin",
    "E6.criterion",
}
POINT_ESTIMATES = {
    "E1.applied_learning_rate",
    "E1.bias_corrected_step",
    "E1.schedule_increment",
    "E1.decay_coefficient",
    "E1.optimizer_state_step",
    "E2.upper_normal_edge",
    "E6.observed_entry_step",
    "E6.entry_error",
    "E6.sustained_entry_indicator",
    "E5.path_window_length",
    "E6.path_window_length",
    "E8.model_width",
    "E8.model_depth",
    "E8.observed_entry_step",
    "E8.heldout_loss",
    "E8.baseline_indicator",
    "E8.model_score",
    "E8.maximum_learning_rate",
    "E8.warmup_steps",
    "E8.anneal_floor",
    "E8.source_order_parameter",
    "E8.source_rmse",
    "E8.tensor_mean_abs",
    "E8.row_input_spread",
    "E8.downstream_lipschitz",
    "E8.avalanche_event_rate",
    "E8.avalanche_mean_size",
    "E8.avalanche_mean_duration",
    "E8.avalanche_mean_support",
    "E8.avalanche_tail_comparison",
    "E8.finite_size_scaling_error",
    "E8.rg_block_size",
}


def numeric_projection(protocol, name):
    identifier = f"{protocol}.{name}"
    if protocol == "E7" or identifier in POINT_ESTIMATES:
        return "point"
    if identifier in LOWER_ENDPOINTS:
        return "lower"
    return "upper"


def _defined_functions(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_object(root=None):
    root = Path(root or Path(__file__).resolve().parents[2]).resolve()
    estimators = {}
    for protocol, names in campaign_record.MEASUREMENT_NAMES.items():
        executable, function = OWNERS[protocol]
        executable_path = root / executable
        if function not in _defined_functions(executable_path):
            raise ValueError(
                f"registered estimator function is absent: {executable}:{function}")
        digest = sha256_file(executable_path)
        for name in names:
            identifier = f"{protocol}.{name}"
            estimators[identifier] = {
                "protocol_id": protocol,
                "measurement_name": name,
                "executable": executable,
                "executable_sha256": digest,
                "function": function,
                "permitted_parent_types": [
                    "bound_root_artifact", "validated_certificate_node",
                    "immutable_empirical_array",
                ],
                "numeric_projection": numeric_projection(protocol, name),
                "output_type": "finite_scalar_with_registered_projection",
                "checker": "experiments/confirm/check_certificate.py",
                "primary": True,
            }
    return {
        "schema_version": SCHEMA,
        "checker": "experiments/confirm/check_certificate.py",
        "estimators": estimators,
    }


def validate_manifest(manifest, root=None):
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version", "checker", "estimators",
    }:
        raise ValueError("estimator manifest has missing or unknown fields")
    if manifest["schema_version"] != SCHEMA:
        raise ValueError("estimator manifest schema is stale")
    expected = manifest_object(root)
    if manifest != expected:
        raise ValueError("estimator manifest is stale or incomplete")
    return True


def load_manifest(path, root=None):
    with Path(path).open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    validate_manifest(manifest, root)
    return manifest
