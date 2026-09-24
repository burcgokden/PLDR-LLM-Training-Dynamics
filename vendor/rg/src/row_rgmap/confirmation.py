"""Registered finite-horizon decisions for consecutive PLDR energy paths."""

from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
import math
import re
from typing import Any

import numpy as np

from .statistics import normal_ks_distance, standardized_moments


_MAP_PATTERN = re.compile(r"^c(?P<context>[0-9]+)[.]L(?P<layer>[0-9]+)[.]H(?P<head>[0-9]+)$")


def _parse_registry(map_ids) -> list[dict[str, int]]:
    registry = []
    for value in np.asarray(map_ids).tolist():
        map_id = str(value)
        match = _MAP_PATTERN.fullmatch(map_id)
        if match is None:
            raise ValueError(f"map id does not match c{{context}}.L{{layer}}.H{{head}}: {map_id}")
        registry.append(
            {
                "map_id": map_id,
                "context": int(match.group("context")),
                "layer": int(match.group("layer")),
                "head": int(match.group("head")),
            }
        )
    if len({row["map_id"] for row in registry}) != len(registry):
        raise ValueError("map registry contains duplicates")
    return registry


def _batch_summary(values: np.ndarray, batch_length: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    usable = len(values) // batch_length * batch_length
    batches = (
        values[:usable].reshape(-1, batch_length).mean(axis=1)
        if usable else np.empty(0, dtype=np.float64)
    )
    return {
        "batch_length": batch_length,
        "used_updates": int(usable),
        "discarded_updates": int(len(values) - usable),
        "batch_count": int(len(batches)),
        "batch_means": batches,
        "mean": float(values.mean()),
        "variance": float(values.var()),
        "lag1_correlation": (
            float(np.corrcoef(values[:-1], values[1:])[0, 1])
            if len(values) > 2 and values[:-1].std() > 0.0
            and values[1:].std() > 0.0 else None
        ),
    }


def _normal_ks_calibration(
    sample_count: int, *, replicates: int, quantile: float, generator
) -> float:
    if sample_count < 2:
        raise ValueError("KS calibration needs at least two samples")
    distances = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = generator.normal(size=sample_count)
        sample = (sample - sample.mean()) / sample.std()
        distances[index] = normal_ks_distance(sample)
    return float(np.quantile(distances, quantile))


def _block_law_diagnostics(
    values: np.ndarray,
    block_sizes: list[int],
    calibration: dict,
    generator,
    calibration_cache: dict[int, float],
) -> list[dict]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    mean = float(values.mean())
    centered = values - mean
    rows = []
    for block_size in block_sizes:
        count = len(centered) // block_size
        if count < 4:
            continue
        sums = centered[: count * block_size].reshape(count, block_size).sum(axis=1)
        standard_deviation = float(sums.std())
        if standard_deviation == 0.0:
            rows.append(
                {
                    "block_size": block_size,
                    "block_count": count,
                    "degenerate": True,
                    "normal_ks_distance": None,
                    "normal_ks_95_calibration": None,
                }
            )
            continue
        normalized = (sums - sums.mean()) / standard_deviation
        moments = standardized_moments(normalized)
        if count not in calibration_cache:
            calibration_cache[count] = _normal_ks_calibration(
                count,
                replicates=int(calibration["replicates"]),
                quantile=float(calibration["quantile"]),
                generator=generator,
            )
        threshold = calibration_cache[count]
        rows.append(
            {
                "block_size": block_size,
                "block_count": count,
                "degenerate": False,
                "normal_ks_distance": normal_ks_distance(normalized),
                "normal_ks_95_calibration": threshold,
                "within_matched_normal_ks_calibration": (
                    normal_ks_distance(normalized) <= threshold
                ),
                "skewness": moments["skewness"],
                "excess_kurtosis": moments["excess_kurtosis"],
            }
        )
    return rows


def _excursion_census(path: np.ndarray) -> dict[str, Any]:
    path = np.asarray(path, dtype=np.float64).reshape(-1)
    face_indices = np.flatnonzero(path == 0.0)
    rows = []
    for left, right in zip(face_indices[:-1], face_indices[1:]):
        restart = float(path[left + 1]) if left + 1 <= right else 0.0
        segment = path[left + 1 : right + 1]
        peak = float(segment.max()) if len(segment) else 0.0
        amplification = peak / restart if restart > 0.0 else 0.0
        rows.append(
            {
                "face_start": int(left),
                "face_stop": int(right),
                "restart": restart,
                "maximum_amplification": amplification,
                "restart_times_maximum_amplification": restart * amplification,
            }
        )
    return {
        "face_state_count": int(len(face_indices)),
        "complete_excursion_count": int(len(rows)),
        "maximum_restart_amplification_product": (
            max(row["restart_times_maximum_amplification"] for row in rows)
            if rows else None
        ),
        "excursions": rows,
    }


def analyze_confirmation(
    energies,
    trajectory_ids,
    map_ids,
    protocol: dict,
) -> dict[str, Any]:
    """Apply a versioned finite-horizon decision protocol.

    The time-series resampling unit is a nonoverlapping batch of updates.
    Maps within a trajectory-layer cell are averaged before batch means are
    formed, so maps are not treated as independent temporal replicates.
    """

    array = np.asarray(energies, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] < 3:
        raise ValueError("energies must have shape (trajectory, time, map)")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("energies must be finite and nonnegative")
    trajectory_names = [str(value) for value in np.asarray(trajectory_ids).tolist()]
    registry = _parse_registry(map_ids)
    if len(trajectory_names) != array.shape[0] or len(registry) != array.shape[2]:
        raise ValueError("trajectory or map registry dimensions do not match")

    rules = protocol["decision_rules"]
    split_fraction = float(rules["estimation_fraction"])
    batch_length = int(rules["time_batch_length"])
    minimum_batches = int(rules["minimum_time_batches"])
    familywise_alpha = float(rules["familywise_alpha"])
    if not 0.0 < split_fraction < 1.0:
        raise ValueError("estimation_fraction must be between zero and one")
    if batch_length < 2 or minimum_batches < 2:
        raise ValueError("time-batch requirements are invalid")

    positive_edges = (array[:, :-1, :] > 0.0) & (array[:, 1:, :] > 0.0)
    log_gain = np.full(positive_edges.shape, np.nan, dtype=np.float64)
    log_gain[positive_edges] = np.log(
        array[:, 1:, :][positive_edges] / array[:, :-1, :][positive_edges]
    )
    edge_count = log_gain.shape[1]
    split = int(math.floor(edge_count * split_fraction))
    if split < batch_length * minimum_batches:
        raise ValueError("estimation segment is too short for registered batches")
    if edge_count - split < batch_length * minimum_batches:
        raise ValueError("holdout segment is too short for registered batches")

    by_layer: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(registry):
        by_layer[row["layer"]].append(index)
    cell_total = len(trajectory_names) * len(by_layer)
    critical_value = NormalDist().inv_cdf(
        1.0 - familywise_alpha / cell_total
    )
    calibration_seed = int(rules["gaussian_calibration"]["seed"])
    generator = np.random.default_rng(calibration_seed)
    calibration_cache: dict[int, float] = {}

    cells = []
    for trajectory_index, trajectory_id in enumerate(trajectory_names):
        for layer, indices in sorted(by_layer.items()):
            cell_edges = log_gain[trajectory_index][:, indices]
            complete_positive = bool(np.isfinite(cell_edges).all())
            if not complete_positive:
                cells.append(
                    {
                        "trajectory_id": trajectory_id,
                        "layer": layer,
                        "map_count": len(indices),
                        "positive_excursion_complete": False,
                        "flow_classification": "FACE_INTERRUPTED",
                    }
                )
                continue
            series = cell_edges.mean(axis=1)
            estimation = series[:split]
            holdout = series[split:]
            estimation_batches = _batch_summary(estimation, batch_length)
            holdout_batches = _batch_summary(holdout, batch_length)
            if (
                estimation_batches["batch_count"] < minimum_batches
                or holdout_batches["batch_count"] < minimum_batches
            ):
                raise ValueError("registered cell has too few time batches")
            batch_means = holdout_batches.pop("batch_means")
            estimation_batch_values = estimation_batches.pop("batch_means")
            standard_error = float(
                batch_means.std(ddof=1) / math.sqrt(len(batch_means))
            )
            holdout_mean = float(holdout.mean())
            upper = holdout_mean + critical_value * standard_error
            lower = holdout_mean - critical_value * standard_error
            estimation_mean = float(estimation.mean())
            if estimation_mean < 0.0 and upper < 0.0:
                classification = "SUBCRITICAL"
            elif estimation_mean > 0.0 and lower > 0.0:
                classification = "SUPERCRITICAL"
            else:
                classification = "UNRESOLVED"

            estimation_variance = float(estimation.var())
            holdout_variance = float(holdout.var())
            variance_ratio = (
                holdout_variance / estimation_variance
                if estimation_variance > 0.0 else None
            )
            shift_denominator = math.sqrt(
                float(estimation_batch_values.var(ddof=1))
                / len(estimation_batch_values)
                + float(batch_means.var(ddof=1)) / len(batch_means)
            )
            split_shift_z = (
                abs(holdout_mean - estimation_mean) / shift_denominator
                if shift_denominator > 0.0 else None
            )
            stationarity_rules = rules["stationarity"]
            stationarity_pass = (
                split_shift_z is not None
                and split_shift_z <= float(stationarity_rules["maximum_split_shift_z"])
                and variance_ratio is not None
                and float(stationarity_rules["minimum_variance_ratio"])
                <= variance_ratio
                <= float(stationarity_rules["maximum_variance_ratio"])
                and (
                    holdout_batches["lag1_correlation"] is not None
                    and abs(holdout_batches["lag1_correlation"])
                    <= float(stationarity_rules["maximum_absolute_lag1_correlation"])
                )
            )

            law_rows = _block_law_diagnostics(
                holdout,
                [int(value) for value in protocol["block_sizes"]],
                rules["gaussian_calibration"],
                generator,
                calibration_cache,
            )
            usable_laws = [row for row in law_rows if not row["degenerate"]]
            final_law = usable_laws[-1] if usable_laws else None
            first_law = usable_laws[0] if usable_laws else None
            gaussian_rules = rules["gaussian_calibration"]
            law_pass = (
                final_law is not None
                and final_law["block_count"] >= int(gaussian_rules["minimum_blocks"])
                and final_law["within_matched_normal_ks_calibration"]
                and abs(final_law["excess_kurtosis"])
                <= float(gaussian_rules["maximum_absolute_excess_kurtosis"])
                and first_law is not None
                and final_law["normal_ks_distance"]
                <= float(gaussian_rules["maximum_final_to_initial_ks_ratio"])
                * first_law["normal_ks_distance"]
            )
            map_means = cell_edges[split:, :].mean(axis=0)
            cells.append(
                {
                    "trajectory_id": trajectory_id,
                    "layer": layer,
                    "map_count": len(indices),
                    "positive_excursion_complete": True,
                    "estimation_mean_log_gain_per_update": estimation_mean,
                    "holdout_mean_log_gain_per_update": holdout_mean,
                    "holdout_map_mean_log_gain_range": [
                        float(map_means.min()), float(map_means.max())
                    ],
                    "familywise_one_sided_confidence_level": 1.0 - familywise_alpha,
                    "bonferroni_critical_value": critical_value,
                    "holdout_batch_mean_standard_error": standard_error,
                    "holdout_lower_confidence_bound": lower,
                    "holdout_upper_confidence_bound": upper,
                    "estimation": estimation_batches,
                    "holdout": holdout_batches,
                    "split_variance_ratio": variance_ratio,
                    "split_mean_shift_z": split_shift_z,
                    "stationarity_gate_passed": stationarity_pass,
                    "block_law_diagnostics": law_rows,
                    "gaussian_block_law_gate_passed": law_pass,
                    "flow_classification": classification,
                }
            )

    complete_cells = [row for row in cells if row["positive_excursion_complete"]]
    all_subcritical = (
        len(complete_cells) == len(cells)
        and all(
            row["flow_classification"] == "SUBCRITICAL"
            and row["estimation_mean_log_gain_per_update"] < 0.0
            for row in complete_cells
        )
    )
    all_supercritical = (
        len(complete_cells) == len(cells)
        and all(
            row["flow_classification"] == "SUPERCRITICAL"
            for row in complete_cells
        )
    )
    classifications = sorted({row["flow_classification"] for row in cells})
    heterogeneous = len(classifications) > 1
    stationary = (
        len(complete_cells) == len(cells)
        and all(row["stationarity_gate_passed"] for row in complete_cells)
    )
    gaussian_law = (
        stationary
        and all(row["gaussian_block_law_gate_passed"] for row in complete_cells)
    )
    source_exercised = bool(np.any(array == 0.0))
    universality_identified = gaussian_law and not source_exercised

    if all_subcritical:
        flow_phrase = "ALL_CELLS_SUBCRITICAL"
    elif all_supercritical:
        flow_phrase = "ALL_CELLS_SUPERCRITICAL"
    elif heterogeneous:
        flow_phrase = "HETEROGENEOUS_CELL_FLOW"
    else:
        flow_phrase = "CELL_FLOW_UNRESOLVED"
    class_phrase = (
        "FINITE_VARIANCE_CLASS_CONSISTENT"
        if universality_identified else "CLASS_NOT_IDENTIFIED"
    )

    excursion_rows = []
    for trajectory_index, trajectory_id in enumerate(trajectory_names):
        for map_index, registry_row in enumerate(registry):
            census = _excursion_census(array[trajectory_index, :, map_index])
            if census["face_state_count"]:
                excursion_rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "map_id": registry_row["map_id"],
                        **census,
                    }
                )

    return {
        "schema_version": "pldr-row-rg-confirmation-decision-v1",
        "inferential_scope": "FINITE_HORIZON_REGISTERED_RUN",
        "calibration": {
            "status": "nominal_for_native_dependent_batches",
            "student_reference": "Conditional on all sign-selection/design information, within-cell holdout batch means are independent normal draws with common mean and positive variance.",
            "chisquare_reference": "Independent normal design batch means; variance stability or a transport bound is additionally needed to project holdout width.",
            "multiplicity": "Bonferroni needs valid marginal pivots, not independence between cells; joint design/holdout coverage requires combined error allocation.",
            "native_dependence_justification_established": False,
            "precision_projection_is_not_power_or_realized_width_guarantee": True,
        },
        "edge_count": edge_count,
        "estimation_edge_count": split,
        "holdout_edge_count": edge_count - split,
        "registry": registry,
        "cells": cells,
        "cell_flow_classifications": {
            name: sum(row["flow_classification"] == name for row in cells)
            for name in (
                "SUBCRITICAL", "SUPERCRITICAL", "UNRESOLVED", "FACE_INTERRUPTED"
            )
        },
        "all_cells_subcritical_on_registered_finite_horizon": all_subcritical,
        "cell_flow_is_heterogeneous": heterogeneous,
        "stationarity_gate_passed_for_every_cell": stationary,
        "gaussian_block_law_gate_passed_for_every_cell": gaussian_law,
        "source_sector_exercised": source_exercised,
        "excursion_census": excursion_rows,
        "universality_identified_on_registered_domain": universality_identified,
        "outcome": f"{flow_phrase}_{class_phrase}",
        "limitations": {
            "finite_horizon_is_not_an_asymptotic_collapse_proof": True,
            "batch_means_do_not_make_training_stationary": True,
            "normal_law_consistency_is_not_a_unique_domain_of_attraction_proof": True,
            "maps_within_a_cell_are_not_treated_as_independent_time_series": True,
        },
    }
