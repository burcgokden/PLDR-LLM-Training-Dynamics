"""Registered descriptive diagnostics for the v3 PLDR campaign."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _segment(update: int, burn: int, design: int) -> str:
    if update <= burn:
        return "BURN_IN"
    if update <= burn + design:
        return "DESIGN"
    return "HOLDOUT"


def block_prediction_diagnostics(
    design: Any, holdout: Any, block_sizes: list[int]
) -> list[dict[str, Any]]:
    """Report design and holdout block means, variances, and quantiles."""

    design_array = np.asarray(design, dtype=np.float64).reshape(-1)
    holdout_array = np.asarray(holdout, dtype=np.float64).reshape(-1)
    rows = []
    probabilities = np.asarray([0.05, 0.5, 0.95], dtype=np.float64)
    for block_size in block_sizes:
        design_count = len(design_array) // block_size
        holdout_count = len(holdout_array) // block_size
        design_sums = design_array[: design_count * block_size].reshape(
            design_count, block_size
        ).sum(axis=1)
        holdout_sums = holdout_array[: holdout_count * block_size].reshape(
            holdout_count, block_size
        ).sum(axis=1)
        design_variance = float(design_sums.var(ddof=1))
        holdout_variance = float(holdout_sums.var(ddof=1))
        rows.append(
            {
                "block_size": int(block_size),
                "design_block_count": int(design_count),
                "holdout_block_count": int(holdout_count),
                "design_mean_log_gain_per_update": float(
                    design_sums.mean() / block_size
                ),
                "holdout_mean_log_gain_per_update": float(
                    holdout_sums.mean() / block_size
                ),
                "design_block_sum_variance": design_variance,
                "holdout_block_sum_variance": holdout_variance,
                "holdout_to_design_variance_ratio": (
                    holdout_variance / design_variance
                    if design_variance > 0.0
                    else None
                ),
                "design_block_sum_quantiles": [
                    float(value)
                    for value in np.quantile(design_sums, probabilities)
                ],
                "holdout_block_sum_quantiles": [
                    float(value)
                    for value in np.quantile(holdout_sums, probabilities)
                ],
                "quantile_probabilities": probabilities.tolist(),
            }
        )
    return rows


def large_increment_excursion_ledger(
    energies: Any,
    trajectory_ids: list[str],
    registry: list[dict[str, Any]],
    log_gain: Any,
    rules: dict[str, Any],
    burn: int,
    design: int,
) -> dict[str, Any]:
    """Enumerate every predeclared large layer increment and its clusters."""

    array = np.asarray(energies, dtype=np.float64)
    gains = np.asarray(log_gain, dtype=np.float64)
    layer = int(rules["layer"])
    threshold = float(rules["absolute_log_gain_threshold"])
    cluster_gap = int(rules["cluster_gap_updates"])
    indices = [
        index for index, row in enumerate(registry) if int(row["layer"]) == layer
    ]
    events: list[dict[str, Any]] = []
    per_map_counts: list[dict[str, Any]] = []
    censored = 0
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        for map_index in indices:
            map_id = str(registry[map_index]["map_id"])
            series = gains[trajectory_index, :, map_index]
            censored += int(np.sum(~np.isfinite(series)))
            event_indices = np.flatnonzero(
                np.isfinite(series) & (np.abs(series) >= threshold)
            )
            per_map_counts.append(
                {
                    "trajectory_id": trajectory_id,
                    "map_id": map_id,
                    "event_count": int(len(event_indices)),
                }
            )
            for edge_index in event_indices.tolist():
                update = edge_index + 1
                events.append(
                    {
                        "trajectory_id": trajectory_id,
                        "map_id": map_id,
                        "update": update,
                        "segment": _segment(update, burn, design),
                        "log_gain": float(series[edge_index]),
                        "energy_before": float(
                            array[trajectory_index, edge_index, map_index]
                        ),
                        "energy_after": float(
                            array[trajectory_index, edge_index + 1, map_index]
                        ),
                    }
                )
    events.sort(key=lambda row: (row["trajectory_id"], row["update"], row["map_id"]))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[str(event["trajectory_id"])].append(event)
    clusters = []
    for trajectory_id, rows in sorted(grouped.items()):
        active: list[dict[str, Any]] = []
        for event in rows:
            if active and int(event["update"]) - int(active[-1]["update"]) > cluster_gap:
                clusters.append(_cluster_row(trajectory_id, active))
                active = []
            active.append(event)
        if active:
            clusters.append(_cluster_row(trajectory_id, active))
    return {
        "layer": layer,
        "definition": "absolute per-map log gain at least the registered threshold",
        "absolute_log_gain_threshold": threshold,
        "cluster_gap_updates": cluster_gap,
        "map_series_count": len(per_map_counts),
        "event_count": len(events),
        "cluster_count": len(clusters),
        "censored_log_edge_count": censored,
        "per_map_counts": per_map_counts,
        "events": events,
        "clusters": clusters,
    }


def _cluster_row(
    trajectory_id: str, events: list[dict[str, Any]]
) -> dict[str, Any]:
    extreme = max(events, key=lambda row: abs(float(row["log_gain"])))
    return {
        "trajectory_id": trajectory_id,
        "first_update": int(events[0]["update"]),
        "last_update": int(events[-1]["update"]),
        "event_count": len(events),
        "map_ids": sorted({str(row["map_id"]) for row in events}),
        "maximum_absolute_log_gain": abs(float(extreme["log_gain"])),
        "extreme_event": extreme,
    }


def face_excursion_ledger(
    energies: Any, trajectory_ids: list[str], map_ids: list[str]
) -> dict[str, Any]:
    """Enumerate exact-face closures, restarts, and restart excursions."""

    array = np.asarray(energies, dtype=np.float64)
    closures = []
    restarts = []
    excursions = []
    for trajectory_index, trajectory_id in enumerate(trajectory_ids):
        for map_index, map_id in enumerate(map_ids):
            series = array[trajectory_index, :, map_index]
            closure_steps = np.flatnonzero(
                (series[:-1] > 0.0) & (series[1:] == 0.0)
            ) + 1
            restart_steps = np.flatnonzero(
                (series[:-1] == 0.0) & (series[1:] > 0.0)
            ) + 1
            for step in closure_steps.tolist():
                closures.append(
                    {
                        "trajectory_id": trajectory_id,
                        "map_id": map_id,
                        "face_step": int(step),
                        "energy_before": float(series[step - 1]),
                    }
                )
            for step in restart_steps.tolist():
                restart = float(series[step])
                future_faces = np.flatnonzero(series[step + 1 :] == 0.0)
                next_face = (
                    int(step + 1 + future_faces[0])
                    if len(future_faces)
                    else None
                )
                stop = next_face if next_face is not None else len(series)
                positive_excursion = series[step:stop]
                peak_offset = int(np.argmax(positive_excursion))
                peak = float(positive_excursion[peak_offset])
                amplification = peak / restart
                row = {
                    "trajectory_id": trajectory_id,
                    "map_id": map_id,
                    "restart_step": int(step),
                    "next_face_step": next_face,
                    "right_censored": next_face is None,
                    "restart_energy_r": restart,
                    "peak_energy": peak,
                    "first_peak_step": int(step + peak_offset),
                    "amplification_M": amplification,
                    "restart_times_amplification": restart * amplification,
                }
                restarts.append(
                    {
                        "trajectory_id": trajectory_id,
                        "map_id": map_id,
                        "restart_step": int(step),
                        "restart_energy": restart,
                    }
                )
                excursions.append(row)
    return {
        "exact_face_state_count": int(np.sum(array == 0.0)),
        "closure_count": len(closures),
        "restart_count": len(restarts),
        "excursion_count": len(excursions),
        "right_censored_excursion_count": sum(
            bool(row["right_censored"]) for row in excursions
        ),
        "closures": closures,
        "restarts": restarts,
        "excursions": excursions,
        "source_sector_exercised": bool(restarts),
    }
