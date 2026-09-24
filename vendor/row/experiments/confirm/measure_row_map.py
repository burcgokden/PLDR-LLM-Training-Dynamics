#!/usr/bin/env python3
"""Measure the live convex-tube row-map certificate for one E0 unit."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(EXPERIMENTS))

import instrument  # noqa: E402
from measure_tilt import build_model  # noqa: E402
from train_run import create_masks, PROBE_RESERVE_CHUNKS  # noqa: E402

from campaign_record import strict_dumps  # noqa: E402
from row_domain import (  # noqa: E402
    cover_segment_hull,
    interval_hull,
    points_in_rational_boxes,
    tile_interval_hull,
    validate_construction_validation_split,
)
from row_map_jacobian import (  # noqa: E402
    _row_map_value,
    basis_row_jacobian,
    dense_row_jacobian,
    resource_cost,
)
from validated_interval import (  # noqa: E402
    RationalInterval,
    as_fraction,
    sqrt_enclosure,
)
from validated_row_map import (  # noqa: E402
    certify_joint_row_map_norms,
    model_row_map_specification,
)


def _run_config(run_directory):
    path = Path(run_directory) / "log.jsonl"
    with path.open("r", encoding="utf-8") as stream:
        row = json.loads(stream.readline())
    if row.get("event") != "config":
        raise ValueError("run log does not start with a config record")
    return row


def _probe_batch(config, token_path, device, batch_size, offset):
    tokens = np.memmap(token_path, dtype=np.uint16, mode="r")
    context = int(config["ctx"])
    chunks = tokens[: len(tokens) // context * context].reshape(-1, context)
    probe_start = len(chunks) - PROBE_RESERVE_CHUNKS + int(offset)
    if not 0 <= probe_start <= len(chunks) - batch_size:
        raise ValueError("probe batch lies outside the token array")
    rows = np.asarray(
        chunks[probe_start:probe_start + batch_size], dtype=np.int64)
    inputs = torch.from_numpy(rows[:, :-1]).to(device)
    return inputs, create_masks(inputs, device)


def _fraction(value):
    if not isinstance(value, dict) or set(value) != {"numerator", "denominator"}:
        raise ValueError("certificate scalar is not an exact rational")
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def _upper_float(value):
    value = as_fraction(value)
    result = float(value)
    if not math.isfinite(result):
        raise OverflowError("certified E0 bound lies outside finite float range")
    if Fraction.from_float(result) < value:
        result = math.nextafter(result, math.inf)
    if not math.isfinite(result):
        raise OverflowError("certified E0 bound lies outside finite float range")
    return result


def _ratio(observed, upper):
    observed = float(observed)
    upper = float(upper)
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError("observed norm must be finite and nonnegative")
    if not math.isfinite(upper) or upper < 0.0:
        raise ValueError("certified norm must be finite and nonnegative")
    if upper == 0.0:
        return 0.0 if observed == 0.0 else float(np.finfo(float).max)
    return observed / upper


def _rational_object(value):
    value = as_fraction(value)
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _log10_fraction(value):
    value = as_fraction(value)
    if value <= 0:
        return -float(np.finfo(float).max)
    return math.log10(value.numerator) - math.log10(value.denominator)


def _exact_ratio(observed, upper):
    observed = float(observed)
    upper = as_fraction(upper)
    if not math.isfinite(observed) or observed < 0.0:
        raise ValueError("observed norm must be finite and nonnegative")
    if upper < 0:
        raise ValueError("certified norm must be nonnegative")
    if upper == 0:
        return 0.0 if observed == 0.0 else float(np.finfo(float).max)
    return _upper_float(Fraction.from_float(observed) / upper)


def _box_radius_upper(box):
    """Exact outward upper bound for a box's midpoint radius."""

    intervals = [
        RationalInterval.from_object(entry)
        for entry in box
    ]
    squared = sum(
        ((entry.width / 2) ** 2 for entry in intervals), Fraction(0))
    return sqrt_enclosure(squared)[1]


def _directional_third(model, layer_index, row):
    width = row.numel()
    direction = torch.ones_like(row)
    direction[1::2] = -1
    direction /= torch.linalg.vector_norm(direction)
    output_direction = torch.arange(
        1, width + 1, dtype=row.dtype, device=row.device)
    output_direction[1::2] *= -1
    output_direction /= torch.linalg.vector_norm(output_direction)
    scalar = torch.zeros((), dtype=row.dtype, device=row.device,
                         requires_grad=True)
    value = output_direction @ _row_map_value(
        model, layer_index, row + scalar * direction)
    first = torch.autograd.grad(value, scalar, create_graph=True)[0]
    second = torch.autograd.grad(first, scalar, create_graph=True)[0]
    third = torch.autograd.grad(second, scalar)[0]
    return abs(float(third.detach().cpu()))


def _center_tensor(center, device):
    return torch.tensor(
        [float(as_fraction(value)) for value in center],
        dtype=torch.float64,
        device=device,
    )


def _parameter_displacement(model_before, model_after, layer_index):
    before = model_before.decoder.dec_layers[layer_index].mha1.reslayerAs
    after = model_after.decoder.dec_layers[layer_index].mha1.reslayerAs
    squared = sum((
        ((right.detach().cpu().double() - left.detach().cpu().double()) ** 2).sum()
        for left, right in zip(before.parameters(), after.parameters())
    ), torch.zeros((), dtype=torch.float64))
    return float(torch.sqrt(squared))


def _measurement(name, value, unit, method):
    return {
        "name": name,
        "value": float(value),
        "unit": unit,
        "method": method,
        "status": "OBSERVED",
        "reason_code": None,
    }


def measure_checkpoint(*, run_directory, checkpoint, successor_checkpoint,
                       token_path, device, batch_size, probe_offset,
                       rows_per_layer, heldout_rows_per_layer,
                       coordinate_padding, subdivisions_per_axis,
                       maximum_boxes, resource_action_limit,
                       resource_storage_bytes_limit, output_rows=None,
                       expected_segment_updates=1,
                       cover_strategy="convex_interval_hull",
                       subdivisions_per_segment=1,
                       physical_criterion=1.0,
                       reserved_tail_budget=0.0):
    if rows_per_layer < 1 or heldout_rows_per_layer < 1:
        raise ValueError("registered and held-out row counts must be positive")
    if coordinate_padding < 0 or not math.isfinite(coordinate_padding):
        raise ValueError("coordinate padding must be finite and nonnegative")
    if not math.isfinite(physical_criterion) or physical_criterion <= 0:
        raise ValueError("physical criterion must be finite and positive")
    if not math.isfinite(reserved_tail_budget) or reserved_tail_budget < 0:
        raise ValueError("reserved tail budget must be finite and nonnegative")
    config = _run_config(run_directory)
    before_payload = torch.load(
        checkpoint, map_location=device, weights_only=True)
    after_payload = torch.load(
        successor_checkpoint, map_location=device, weights_only=True)
    before_step = int(before_payload["step"])
    after_step = int(after_payload["step"])
    if expected_segment_updates not in {0, 1}:
        raise ValueError("expected_segment_updates must be zero or one")
    if after_step != before_step + expected_segment_updates:
        raise ValueError(
            "E0 joint segment checkpoint spacing disagrees with its mode")

    model_before = build_model(config, device)
    model_after = build_model(config, device)
    model_before.load_state_dict(before_payload["model"])
    model_after.load_state_dict(after_payload["model"])
    model_before = model_before.to(device=device, dtype=torch.float64).eval()
    model_after = model_after.to(device=device, dtype=torch.float64).eval()
    inputs, mask = _probe_batch(
        config, token_path, device, batch_size, probe_offset)
    generator = torch.Generator().manual_seed(int(config["seed"]) + 41017)
    rows = instrument.visited_rows(
        model_before, inputs, mask,
        max_rows_per_layer=rows_per_layer + heldout_rows_per_layer,
        generator=generator,
    )

    layers = []
    row_artifact = {}
    for layer_index in sorted(rows):
        available = rows[layer_index].to(dtype=torch.float64)
        required = rows_per_layer + heldout_rows_per_layer
        if len(available) < required:
            raise ValueError(
                f"layer {layer_index} has {len(available)} rows, "
                f"fewer than the required {required}")
        registered = available[:rows_per_layer]
        heldout = available[rows_per_layer:required]
        construction_ids = [
            f"layer-{layer_index}:probe-row-{index}"
            for index in range(rows_per_layer)
        ]
        validation_ids = [
            f"layer-{layer_index}:probe-row-{rows_per_layer + index}"
            for index in range(heldout_rows_per_layer)
        ]
        validate_construction_validation_split(construction_ids, validation_ids)
        domain_rows = registered.detach().cpu().tolist()
        hull = interval_hull(domain_rows, padding=coordinate_padding)
        if cover_strategy == "registered_pairwise_segment_hull":
            if coordinate_padding != 0:
                raise ValueError(
                    "the registered segment hull requires zero ambient padding")
            tiling = cover_segment_hull(
                domain_rows,
                subdivisions_per_segment,
                maximum_boxes=maximum_boxes,
            )
        elif cover_strategy == "convex_interval_hull":
            tiling = tile_interval_hull(
                hull, subdivisions_per_axis, maximum_boxes=maximum_boxes)
        else:
            raise ValueError("unknown row-cover strategy")
        if tiling["status"] != "CERTIFIED":
            raise RuntimeError(
                "registered convex row cover is infeasible: "
                + tiling["reason_code"])
        costs = resource_cost(tiling["box_count"], hull.dimension)
        actions = 2 * costs["jacobian_actions"]
        storage = 2 * costs["jacobian_storage_bytes"]
        if actions > resource_action_limit:
            raise RuntimeError("registered E0 Jacobian action limit exceeded")
        if storage > resource_storage_bytes_limit:
            raise RuntimeError("registered E0 Jacobian storage limit exceeded")

        specification_before = model_row_map_specification(
            model_before, layer_index)
        specification_after = model_row_map_specification(
            model_after, layer_index)
        exact_certificates = []
        basis_rows = []
        dense_rows = []
        successor_dense_rows = []
        numerical_errors = []
        spectral_norms = []
        cover_radius_exacts = []
        mixed_actions = []
        third_actions = []
        for box, center in zip(tiling["boxes"], tiling["centers"]):
            exact = certify_joint_row_map_norms(
                specification_before, specification_after, box)
            exact_certificates.append(exact)
            cover_radius_exacts.append(
                _fraction(exact["derivative_lipschitz_upper"])
                * _box_radius_upper(box))
            point = _center_tensor(center, device)
            basis = basis_row_jacobian(model_before, layer_index, point)
            dense = dense_row_jacobian(model_before, layer_index, point)
            successor_dense = dense_row_jacobian(
                model_after, layer_index, point)
            error = float(torch.linalg.matrix_norm(basis - dense, ord=2))
            basis_rows.append(basis.detach().cpu().numpy())
            dense_rows.append(dense.detach().cpu().numpy())
            successor_dense_rows.append(
                successor_dense.detach().cpu().numpy())
            numerical_errors.append(error)
            spectral_norms.append(float(torch.linalg.matrix_norm(
                basis, ord=2)) + error)
            mixed_actions.append(float(torch.linalg.matrix_norm(
                successor_dense - dense, ord=2)))
            third_actions.append(_directional_third(
                model_before, layer_index, point))

        direct_exact = max(
            _fraction(item["jacobian_operator_upper"])
            for item in exact_certificates)
        derivative_exact = max(
            _fraction(item["derivative_lipschitz_upper"])
            for item in exact_certificates)
        mixed_exact = max(
            _fraction(item["parameter_row_mixed_upper"])
            for item in exact_certificates)
        third_exact = max(
            _fraction(item["joint_third_derivative_upper"])
            for item in exact_certificates)
        direct_upper = _upper_float(direct_exact)
        derivative_upper = _upper_float(derivative_exact)
        mixed_upper = _upper_float(mixed_exact)
        cover_remainder_exact = max(cover_radius_exacts)
        grid_upper = max(spectral_norms)
        cover_remainder = _upper_float(cover_remainder_exact)
        if grid_upper > direct_upper:
            raise ArithmeticError(
                "live Jacobian escaped the exact rational norm certificate")

        heldout_norms = []
        for point in heldout:
            jacobian = dense_row_jacobian(
                model_before, layer_index, point)
            heldout_norms.append(float(torch.linalg.matrix_norm(
                jacobian, ord=2)))
        heldout_array = heldout.detach().cpu().numpy()
        center_array = np.asarray([
            [float(as_fraction(value)) for value in center]
            for center in tiling["centers"]
        ])
        nearest_distances = np.min(np.linalg.norm(
            heldout_array[:, None, :] - center_array[None, :, :], axis=2),
            axis=1,
        )
        h_upper = _upper_float(_fraction(tiling["h_upper"]))
        validation_membership = points_in_rational_boxes(
            heldout_array.tolist(), tiling["boxes"])
        validation_membership_indicator = float(all(
            row["member"] for row in validation_membership))
        if h_upper == 0.0:
            heldout_cover_radius_ratio = (
                0.0 if np.max(nearest_distances) == 0.0
                else float(np.finfo(float).max))
        else:
            heldout_cover_radius_ratio = float(
                np.max(nearest_distances) / h_upper)
        heldout_jacobian_ratio = _ratio(
            max(heldout_norms), direct_upper)
        mixed_ratio = _exact_ratio(max(mixed_actions), mixed_exact)
        third_ratio = _exact_ratio(max(third_actions), third_exact)
        if mixed_ratio > 1.0 + 1e-12 or third_ratio > 1.0 + 1e-12:
            raise ArithmeticError("live derivative escaped its joint bound")

        row_artifact[f"layer_{layer_index}_registered"] = (
            registered.detach().cpu().numpy())
        row_artifact[f"layer_{layer_index}_heldout"] = (
            heldout.detach().cpu().numpy())
        row_artifact[f"layer_{layer_index}_construction"] = (
            registered.detach().cpu().numpy())
        row_artifact[f"layer_{layer_index}_validation"] = (
            heldout.detach().cpu().numpy())
        row_artifact[f"layer_{layer_index}_cover_centers"] = center_array
        cover_utility_margin = (
            physical_criterion - cover_remainder - reserved_tail_budget)
        layers.append({
            "layer": layer_index,
            "n_registered_rows": len(registered),
            "n_heldout_rows": len(heldout),
            "n_construction_rows": len(registered),
            "n_validation_rows": len(heldout),
            "construction_row_ids": construction_ids,
            "validation_row_ids": validation_ids,
            "width": len(domain_rows[0]),
            "domain": tiling["domain"],
            "cover_strategy": cover_strategy,
            "tiling": tiling,
            "parameter_segment": {
                "checkpoint_before": before_step,
                "checkpoint_after": after_step,
                "row_map_parameter_displacement_l2":
                    _parameter_displacement(
                        model_before, model_after, layer_index),
            },
            "exact_joint_certificates": exact_certificates,
            "basis_jacobians": np.asarray(basis_rows).tolist(),
            "dense_jacobians": np.asarray(dense_rows).tolist(),
            "successor_dense_jacobians":
                np.asarray(successor_dense_rows).tolist(),
            "basis_dense_error_max": max(numerical_errors),
            "heldout_spectral_norms": heldout_norms,
            "heldout_nearest_center_distances": nearest_distances.tolist(),
            "heldout_cover_radius_ratio": heldout_cover_radius_ratio,
            "validation_membership_witnesses": validation_membership,
            "validation_domain_membership_indicator":
                validation_membership_indicator,
            "heldout_jacobian_ratio": heldout_jacobian_ratio,
            "mixed_parameter_row_segment_actions": mixed_actions,
            "directional_third_derivatives": third_actions,
            "direct_upper": direct_upper,
            "grid_upper": grid_upper,
            "cover_radius_uppers": [
                _rational_object(value) for value in cover_radius_exacts],
            "cover_utility_margin": cover_utility_margin,
            "cover_remainder": cover_remainder,
            "derivative_lipschitz_upper": derivative_upper,
            "mixed_parameter_row_upper": mixed_upper,
            "joint_third_derivative_upper":
                _rational_object(third_exact),
            "joint_third_derivative_log10_upper":
                _log10_fraction(third_exact),
            "mixed_derivative_enclosure_ratio": mixed_ratio,
            "third_derivative_enclosure_ratio": third_ratio,
            "cover_identity_error": 0.0,
            "cover_recomposition_method":
                "independent_center_norm_plus_interval_radius_per_cell",
            "resource_cost": {
                **costs,
                "joint_checkpoint_jacobian_actions": actions,
                "joint_checkpoint_jacobian_storage_bytes": storage,
            },
        })

    if output_rows is not None:
        np.savez_compressed(output_rows, **row_artifact)
    direct_upper = max(row["direct_upper"] for row in layers)
    grid_upper = max(row["grid_upper"] for row in layers)
    remainder = max(row["cover_remainder"] for row in layers)
    numerical_error = max(row["basis_dense_error_max"] for row in layers)
    cover_identity_error = max(row["cover_identity_error"] for row in layers)
    heldout_jacobian_ratio = max(
        row["heldout_jacobian_ratio"] for row in layers)
    heldout_cover_radius_ratio = max(
        row["heldout_cover_radius_ratio"] for row in layers)
    mixed_ratio = max(
        row["mixed_derivative_enclosure_ratio"] for row in layers)
    third_ratio = max(
        row["third_derivative_enclosure_ratio"] for row in layers)
    validation_membership_indicator = min(
        row["validation_domain_membership_indicator"] for row in layers)
    cover_utility_margin = min(row["cover_utility_margin"] for row in layers)
    measurements = [
        _measurement(
            "direct_row_map_upper", direct_upper,
            "output_row_unit_per_input_row_unit",
            "exact_rational_joint_segment_norm_bound"),
        _measurement(
            "grid_jacobian_upper", grid_upper,
            "output_row_unit_per_input_row_unit",
            "complete_basis_jvp_and_dense_svd_diagnostic"),
        _measurement(
            "cover_remainder", remainder,
            "output_row_unit_per_input_row_unit",
            "independent_derivative_modulus_times_exact_cell_radius"),
        _measurement(
            "cover_identity_error", cover_identity_error,
            "output_row_unit_per_input_row_unit",
            "compatibility_field_excluded_from_primary_decision"),
        _measurement(
            "cover_utility_margin", cover_utility_margin,
            "output_row_unit_per_input_row_unit",
            "criterion_minus_independent_cover_remainder_minus_reserved_tail"),
        _measurement(
            "validation_domain_membership_indicator",
            validation_membership_indicator,
            "indicator", "exact_rational_frozen_box_union_membership"),
        _measurement(
            "heldout_jacobian_ratio", heldout_jacobian_ratio,
            "ratio", "dense_heldout_operator_over_certified_upper"),
        _measurement(
            "heldout_cover_radius_ratio", heldout_cover_radius_ratio,
            "ratio", "exact_convex_partition_membership_witness"),
        _measurement(
            "mixed_derivative_enclosure_ratio", mixed_ratio,
            "ratio", "checkpoint_segment_jacobian_motion_over_joint_bound"),
        _measurement(
            "third_derivative_enclosure_ratio", third_ratio,
            "ratio", "directional_third_derivative_over_joint_bound"),
        _measurement(
            "numerical_error", numerical_error,
            "output_row_unit_per_input_row_unit",
            "basis_dense_spectral_discrepancy"),
    ]
    return {
        "schema_version": "pldr-row-map-direct-cover-v5",
        "checkpoint_step": before_step,
        "successor_checkpoint_step": after_step,
        "normalizer": 1.0,
        "units": "physical_row_units",
        "physical_criterion": float(physical_criterion),
        "reserved_tail_budget": float(reserved_tail_budget),
        "layers": layers,
        "measurements": measurements,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-directory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--successor-checkpoint", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--probe-offset", type=int, default=0)
    parser.add_argument("--rows-per-layer", type=int, default=16)
    parser.add_argument("--heldout-rows-per-layer", type=int, default=8)
    parser.add_argument("--coordinate-padding", type=float, default=2.0)
    parser.add_argument("--subdivisions-per-axis", type=int, default=1)
    parser.add_argument("--subdivisions-per-segment", type=int, default=1)
    parser.add_argument(
        "--cover-strategy",
        choices=("convex_interval_hull", "registered_pairwise_segment_hull"),
        default="convex_interval_hull",
    )
    parser.add_argument("--maximum-boxes", type=int, default=1)
    parser.add_argument("--resource-action-limit", type=int, default=2000000)
    parser.add_argument(
        "--resource-storage-bytes-limit", type=int, default=2147483648)
    parser.add_argument("--output-rows")
    parser.add_argument(
        "--expected-segment-updates", type=int, choices=(0, 1), default=1)
    parser.add_argument("--physical-criterion", type=float, default=1.0)
    parser.add_argument("--reserved-tail-budget", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = measure_checkpoint(
        run_directory=args.run_directory,
        checkpoint=args.checkpoint,
        successor_checkpoint=args.successor_checkpoint,
        token_path=args.tokens,
        device=args.device,
        batch_size=args.batch_size,
        probe_offset=args.probe_offset,
        rows_per_layer=args.rows_per_layer,
        heldout_rows_per_layer=args.heldout_rows_per_layer,
        coordinate_padding=args.coordinate_padding,
        subdivisions_per_axis=args.subdivisions_per_axis,
        subdivisions_per_segment=args.subdivisions_per_segment,
        cover_strategy=args.cover_strategy,
        maximum_boxes=args.maximum_boxes,
        resource_action_limit=args.resource_action_limit,
        physical_criterion=args.physical_criterion,
        reserved_tail_budget=args.reserved_tail_budget,
        resource_storage_bytes_limit=args.resource_storage_bytes_limit,
        output_rows=args.output_rows,
        expected_segment_updates=args.expected_segment_updates,
    )
    Path(args.output).write_text(strict_dumps(result), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
