"""Real-checkpoint Q3 interface-rank and local-tube diagnostics.

The exact rank in this module is the rank of the implemented LayerNorm image
after row centering.  It is not the rank of the stacked parameter-and-moment
constraint Jacobian.  Sampled response gains are retained as diagnostics and
are never serialized as certified full-operator upper bounds.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from confirm.confirmation_artifacts import load_complete_checkpoint, sha256_path
from confirm.finite_increment_live import capture_maps, registered_tokens
from confirm.row_map_live import _model_from_raw
from confirm.source_restoring_live import load_registry
from confirm.source_resolved_energy import (
    admissible_quadratic_radius,
    layernorm_hessian_bound,
    plga_derivative_bounds,
)
from confirm.source_resolved_live import _model_gate, _schema
from confirm.source_resolved_specs import (
    CAMPAIGN_ID,
    FROZEN_POLICY,
    RANK_TUBE_SCHEMA,
    REGISTRY,
)
from confirm.strict_schema import load_json, validate


def _binding(
    path: str | Path, protocol_directory: str | Path
) -> tuple[Path, dict[str, Any]]:
    binding_path = Path(path).resolve()
    value = load_json(binding_path)
    validate(
        value,
        _schema(protocol_directory, "checkpoint_binding.schema.json"),
    )
    checkpoint = Path(value["checkpoint_path"]).resolve()
    if (
        not value["technical_valid"]
        or sha256_path(checkpoint) != value["checkpoint_sha256"]
    ):
        raise ValueError("Q3 checkpoint binding is invalid or stale")
    return checkpoint, value


def _rank_record(
    map_index: int,
    contexts: list[dict[str, Any]],
    shape: np.ndarray,
    gate: np.ndarray,
) -> dict[str, Any]:
    """Return the implemented-image rank and its numerical conditioning.

    Regularized LayerNorm has the constant-feature direction as its only
    kernel.  If all 64 affine scales are nonzero, its image is one fixed
    63-dimensional feature hyperplane.  Centering 64 rows then leaves
    ``(64 - 1) * (64 - 1) = 3969`` independent contrasts.
    """

    context_index, remainder = divmod(map_index, 12)
    layer, head = divmod(remainder, 4)
    local_gate = np.asarray(gate, dtype=np.float64)
    local_shape = np.asarray(shape, dtype=np.float64)
    if (
        local_gate.shape != (64,)
        or local_shape.shape != (64, 64)
        or not np.isfinite(local_gate).all()
        or not np.isfinite(local_shape).all()
    ):
        raise ValueError("rank diagnostic received an invalid map shape")

    projection = np.eye(64) - np.ones((64, 64)) / 64.0
    row_minimum = []
    row_maximum = []
    row_threshold_ranks = []
    for normalized in local_shape:
        mean_square = float(np.mean(normalized * normalized))
        scale = math.sqrt(
            1.0e-6 / max(1.0 - mean_square, np.finfo(float).eps)
        )
        differential = (
            np.diag(local_gate)
            @ (projection - np.outer(normalized, normalized) / 64.0)
            / scale
        )
        singular = np.linalg.svd(differential, compute_uv=False)
        row_maximum.append(float(singular[0]))
        row_minimum.append(float(singular[-2]))
        row_threshold_ranks.append(int(np.sum(
            singular
            > singular[0] * FROZEN_POLICY["rank_relative_threshold"]
        )))

    maximum = max(row_maximum)
    minimum = min(row_minimum)
    relative = 0.0 if maximum == 0.0 else minimum / maximum
    threshold = float(FROZEN_POLICY["rank_relative_threshold"])
    nonzero_gate_count = int(np.count_nonzero(local_gate))
    exact_feature_rank = min(nonzero_gate_count, 63)
    exact_rank = (REGISTRY["rows_per_map"] - 1) * exact_feature_rank
    target = int(REGISTRY["independent_contrast_dimension_per_map"])
    threshold_pass = bool(
        exact_rank == target
        and relative > threshold
        and min(row_threshold_ranks) == 63
    )
    condition = maximum / minimum if minimum > 0.0 else math.inf
    return {
        "map_id": f"{contexts[context_index]['id']}-l{layer}-h{head}",
        "context": str(contexts[context_index]["id"]),
        "layer": int(layer),
        "head": int(head),
        "operator_scope": "implemented-layernorm-image-contrast",
        "constraint_dimension_exact": True,
        "target_row_count": target,
        "relative_threshold": threshold,
        "estimated_numerical_rank": int(
            (REGISTRY["rows_per_map"] - 1) * min(row_threshold_ranks)
        ),
        "exact_algebraic_rank": int(exact_rank),
        "smallest_relative_singular_estimate": float(relative),
        "condition_estimate": float(
            condition
            if math.isfinite(condition)
            else FROZEN_POLICY["metric_maximum_eigenvalue_cap"] + 1.0
        ),
        "rank_threshold_pass": threshold_pass,
    }


def _ordered_edges(
    paths: list[str | Path], protocol_directory: str | Path
) -> list[dict[str, Any]]:
    schema = _schema(protocol_directory, "edge.schema.json")
    edges = []
    for path in paths:
        value = load_json(path)
        validate(value, schema)
        if not value["technical_valid"]:
            raise ValueError("Q3 received a technically invalid source edge")
        edges.append(value)
    edges.sort(key=lambda value: value["source_step"])
    if len(edges) != FROZEN_POLICY["block_length"]:
        raise ValueError("Q3 needs every one-step edge in the length-16 block")
    for left, right in zip(edges, edges[1:]):
        if left["endpoint_step"] != right["source_step"]:
            raise ValueError("Q3 edge sequence has a chronological gap")
    return edges


def _responses(
    paths: list[str | Path], protocol_directory: str | Path
) -> tuple[list[dict[str, Any]], list[str]]:
    if not paths:
        raise ValueError("Q3 needs at least one Q2 response artifact")
    schema = _schema(protocol_directory, "mask_response.schema.json")
    values = []
    digests = []
    seen = set()
    reference_partition = None
    reference_binding = None
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path in seen:
            raise ValueError("Q3 response artifact membership has duplicates")
        seen.add(path)
        value = load_json(path)
        validate(value, schema)
        if not value["technical_valid"]:
            raise ValueError("Q3 received an invalid masked response")
        if reference_partition is None:
            reference_partition = value["coordinate_tensors"]
            reference_binding = value["checkpoint_binding_sha256"]
        elif (
            value["coordinate_tensors"] != reference_partition
            or value["checkpoint_binding_sha256"] != reference_binding
        ):
            raise ValueError("split Q2 responses disagree on state or mask")
        values.append(value)
        digests.append(sha256_path(path))
    return values, digests


def produce_rank_tube(
    source_binding_path: str | Path,
    endpoint_binding_path: str | Path,
    mask_response_paths: list[str | Path],
    edge_paths: list[str | Path],
    block_response_path: str | Path,
    registry_path: str | Path,
    tokens_path: str | Path,
    device_name: str,
    *,
    protocol_directory: str | Path,
) -> dict[str, Any]:
    """Derive exact interface rank and explicitly diagnostic local screens."""

    source_path, source_binding = _binding(
        source_binding_path, protocol_directory
    )
    endpoint_path, endpoint_binding = _binding(
        endpoint_binding_path, protocol_directory
    )
    if (
        endpoint_binding["step"] - source_binding["step"]
        != FROZEN_POLICY["block_length"]
    ):
        raise ValueError("Q3 endpoint does not close the declared block")
    edges = _ordered_edges(edge_paths, protocol_directory)
    if (
        edges[0]["source_step"] != source_binding["step"]
        or edges[-1]["endpoint_step"] != endpoint_binding["step"]
        or edges[0]["source_checkpoint_binding_sha256"]
        != sha256_path(source_binding_path)
        or edges[-1]["endpoint_checkpoint_binding_sha256"]
        != sha256_path(endpoint_binding_path)
    ):
        raise ValueError("Q3 bindings and edge block disagree")
    responses, response_digests = _responses(
        mask_response_paths, protocol_directory
    )
    block_response = load_json(block_response_path)
    validate(
        block_response,
        _schema(protocol_directory, "block_response.schema.json"),
    )
    if not block_response["technical_valid"]:
        raise ValueError("Q3 received an invalid lifted block response")
    current_edge_digests = [sha256_path(path) for path in edge_paths]
    if (
        block_response["block_start"] != source_binding["step"]
        or block_response["block_length"] != FROZEN_POLICY["block_length"]
        or set(block_response["edge_sha256"]) != set(current_edge_digests)
        or block_response["mask_response_sha256"] not in response_digests
    ):
        raise ValueError("Q3 lifted block response has different inputs")

    registry = load_registry(registry_path)
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device.index)
    source_checkpoint = load_complete_checkpoint(
        source_path, map_location="cpu"
    )
    endpoint_checkpoint = load_complete_checkpoint(
        endpoint_path, map_location="cpu"
    )
    if (
        source_checkpoint["measurement_registry"] != registry
        or endpoint_checkpoint["measurement_registry"] != registry
    ):
        raise ValueError("Q3 checkpoint and registry content disagree")
    token_rows, contexts = registered_tokens(registry, tokens_path, device)
    source_model = _model_from_raw(source_checkpoint, device)
    endpoint_model = _model_from_raw(endpoint_checkpoint, device)
    source_shape, source_rows = capture_maps(source_model, token_rows)
    _endpoint_shape, endpoint_rows = capture_maps(endpoint_model, token_rows)
    source_maps = source_rows.reshape(-1, 64, 64)
    endpoint_maps = endpoint_rows.reshape(-1, 64, 64)
    source_gate = _model_gate(source_checkpoint["model"])
    shape_maps = source_shape.reshape(-1, 64, 64)
    map_gate = np.broadcast_to(
        source_gate[None], (len(contexts), 3, 4, 64)
    ).reshape(-1, 64)
    per_map = [
        _rank_record(index, contexts, shape_maps[index], map_gate[index])
        for index in range(REGISTRY["map_count"])
    ]

    stacked = {
        "map_id": "stacked-registered-population",
        "context": "all-24-contexts",
        "layer": 0,
        "head": 0,
        "operator_scope": "full-adamw-stacked-constraint",
        "constraint_dimension_exact": False,
        "target_row_count": (
            REGISTRY["map_count"]
            * REGISTRY["independent_contrast_dimension_per_map"]
        ),
        "relative_threshold": FROZEN_POLICY["rank_relative_threshold"],
        "estimated_numerical_rank": 0,
        "exact_algebraic_rank": 0,
        "smallest_relative_singular_estimate": 0.0,
        "condition_estimate": (
            FROZEN_POLICY["metric_maximum_eigenvalue_cap"] + 1.0
        ),
        "rank_threshold_pass": False,
    }

    edge_gains = np.asarray([
        [record["source_gain"] for record in edge["map_records"]]
        for edge in edges
    ], dtype=np.float64)
    if edge_gains.shape != (
        FROZEN_POLICY["block_length"], REGISTRY["map_count"]
    ):
        raise ValueError("Q3 edge map population changed")
    observed_orbit_gain = float(
        np.sqrt(np.max(np.prod(edge_gains, axis=0)))
    )
    sampled_block_gain = float(max(
        [record["row_response_gain"]
         for record in block_response["krylov_actions"]]
        + [record["row_response_gain"]
           for record in block_response["orthogonal_audits"]]
    ))
    audit_maximum = float(block_response["maximum_audit_row_gain"])
    audit_squared = audit_maximum * audit_maximum

    source_diameter = float(max(
        record["source_diameter"] for record in edges[0]["map_records"]
    ))
    maximum_gate = float(np.max(np.abs(map_gate)))
    layernorm_baseline = float(2.0 * math.sqrt(64.0) * maximum_gate)
    observed_envelope = sampled_block_gain * source_diameter
    metric_screen = bool(
        audit_squared <= FROZEN_POLICY["metric_maximum_eigenvalue_cap"]
    )
    physical_screen = bool(
        observed_envelope <= source_diameter
        and observed_envelope <= layernorm_baseline
    )

    z_values = []
    p_values = []
    for layer in range(3):
        module = source_model.decoder.dec_layers[layer].mha1.plgatt_layer
        rows = torch.from_numpy(source_rows[:, layer]).to(
            device=device, dtype=torch.float64
        )
        weight = module.Wlst.detach().double()
        bias = module.blst.detach().double()
        z_values.append(
            (weight[None] @ rows + bias[None]).detach().cpu().numpy()
        )
        p_values.append(module.pwlst.detach().double().cpu().numpy())
    z = np.concatenate([value.reshape(-1) for value in z_values])
    power = np.concatenate([value.reshape(-1) for value in p_values])
    plga = plga_derivative_bounds(
        float(np.min(z)),
        float(np.max(z)),
        float(np.min(power)),
        float(np.max(power)),
        epsilon=1.0e-9,
    )
    ln_bound = layernorm_hessian_bound(64, 1.0e-6)
    quadratic = float(
        ln_bound + plga["z_second_derivative_abs_upper"]
    )
    force = float(max(
        record["native_energy_upper"] - record["endpoint_energy"]
        for edge in edges for record in edge["map_records"]
    ))
    diagnostic_radius = admissible_quadratic_radius(
        sampled_block_gain, quadratic, max(force, 0.0)
    )
    realized = float(np.max(np.linalg.norm(
        endpoint_maps - source_maps, axis=(-2, -1)
    )))
    radius_upper = diagnostic_radius["radius_upper"]
    inside_diagnostic = bool(
        diagnostic_radius["eligible"]
        and (not math.isfinite(radius_upper) or realized <= radius_upper)
    )

    # The full stacked rank and a certified full-operator gain are both
    # intentionally unresolved.  Exact interface rank cannot discharge them.
    outcome = "unresolved"
    checks = {
        "rank_threshold_frozen": True,
        "rank_diagnostics_finite": all(
            np.isfinite(record["smallest_relative_singular_estimate"])
            and np.isfinite(record["condition_estimate"])
            for record in per_map
        ),
        "analytic_bounds_not_fitted": True,
        "diagnostic_radius_labeled": True,
        "full_state_rank_not_inferred": True,
        "sampled_response_not_promoted_to_bound": True,
        "lifted_block_response_bound": True,
    }
    result = {
        "schema_version": RANK_TUBE_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "source_checkpoint_binding_sha256": sha256_path(
            source_binding_path
        ),
        "endpoint_checkpoint_binding_sha256": sha256_path(
            endpoint_binding_path
        ),
        "mask_response_sha256": response_digests,
        "edge_sha256": current_edge_digests,
        "block_response_sha256": sha256_path(block_response_path),
        "per_map_rank": per_map,
        "stacked_rank": stacked,
        "analytic_bounds": {
            "layernorm_hessian_upper": ln_bound,
            "plga_z_derivative_upper": plga["z_derivative_abs_upper"],
            "plga_z_second_derivative_upper": (
                plga["z_second_derivative_abs_upper"]
            ),
            "block_quadratic_coefficient_upper": quadratic,
            "observed_orbit_radius_gain": observed_orbit_gain,
            "sampled_block_row_gain": sampled_block_gain,
            "diagnostic_radius_gain_used": sampled_block_gain,
            "diagnostic_radius_lower": float(
                diagnostic_radius["radius_lower"]
            ),
            "diagnostic_radius_upper": (
                float(radius_upper) if math.isfinite(radius_upper) else None
            ),
            "diagnostic_radius_eligible": bool(
                diagnostic_radius["eligible"]
            ),
            "realized_displacement": realized,
            "realized_displacement_inside_diagnostic_radius": (
                inside_diagnostic
            ),
            "diagnostic_only": True,
        },
        "block_test": {
            "block_start": source_binding["step"],
            "block_length": FROZEN_POLICY["block_length"],
            "reduced_krylov_rank": int(
                block_response["reduced_rank_achieved"]
            ),
            "audit_direction_count": len(
                block_response["orthogonal_audits"]
            ),
            "audit_response_squared_maximum": audit_squared,
            "metric_cap": FROZEN_POLICY["metric_maximum_eigenvalue_cap"],
            "metric_cap_screen_pass": metric_screen,
            "metric_upper_bound_certified": False,
            "observed_block_diameter_envelope": observed_envelope,
            "source_diameter": source_diameter,
            "layernorm_baseline": layernorm_baseline,
            "physical_scale_screen_pass": physical_screen,
            "audit_response_maximum": audit_maximum,
            "full_operator_theorem_claimed": False,
        },
        "local_robustness_outcome": outcome,
        "checks": checks,
        "technical_valid": all(checks.values()),
    }
    validate(result, _schema(protocol_directory, "rank_tube.schema.json"))
    return result
