"""Analysis routines shared by executed and future confirmation runs."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .core import block_edges, canonical_edges
from .statistics import ordinary_linear_fit, standardized_moments, through_origin_fit


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def seal_record(value: dict) -> dict:
    if "record_sha256" in value:
        raise ValueError("record is already sealed")
    result = dict(value)
    result["record_sha256"] = hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()
    return result


def verify_record_seal(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("sealed record must be a JSON object")
    observed = value.get("record_sha256")
    if not isinstance(observed, str):
        raise ValueError("sealed record has no record_sha256")
    unsigned = dict(value)
    unsigned.pop("record_sha256")
    expected = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ValueError("record SHA-256 does not replay")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_weakening(reference: float, current: float) -> float | None:
    """Return ``1 - current / reference`` when the reference is nonzero."""

    return None if reference == 0.0 else 1.0 - current / reference


def _retained_cell_envelopes(observer: dict) -> list[dict]:
    cells = observer.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError("retained observer cells are missing")
    grouped: dict[tuple[str, int], list[dict]] = {}
    for cell in cells:
        key = (str(cell["trajectory"]), int(cell["layer"]))
        grouped.setdefault(key, []).append(cell)

    rows = []
    for (trajectory, layer), group in sorted(grouped.items()):
        by_step: dict[int, list[dict]] = {}
        for cell in group:
            by_step.setdefault(int(cell["step"]), []).append(cell)
        steps = sorted(by_step)
        if len(steps) < 2:
            raise ValueError("each retained trajectory-layer cell needs two steps")

        def envelope(step: int) -> dict:
            entries = by_step[step]
            lows = [float(entry["minimum_positive_energy"]) for entry in entries]
            highs = [float(entry["maximum_positive_energy"]) for entry in entries]
            if min(lows) <= 0.0 or min(highs) <= 0.0:
                raise ValueError("retained cell envelopes must be positive")
            return {
                "step": step,
                "minimum_positive_energy": min(lows),
                "maximum_positive_energy": max(highs),
                "collections": sorted(str(entry["collection"]) for entry in entries),
                "floor_censored_positive_maps": sum(
                    int(entry["floor_censored_positive_maps"]) for entry in entries
                ),
            }

        first = envelope(steps[0])
        last = envelope(steps[-1])
        lower_change = math.log10(
            last["minimum_positive_energy"] / first["maximum_positive_energy"]
        )
        upper_change = math.log10(
            last["maximum_positive_energy"] / first["minimum_positive_energy"]
        )
        if upper_change < 0.0:
            classification = "DECREASED_ENVELOPE"
        elif lower_change > 0.0:
            classification = "INCREASED_ENVELOPE"
        else:
            classification = "OVERLAPPING_ENVELOPES"
        rows.append(
            {
                "trajectory": trajectory,
                "layer": layer,
                "first": first,
                "last": last,
                "log10_change_bounds": [lower_change, upper_change],
                "classification": classification,
                "decreased_by_at_least_ten_orders": upper_change <= -10.0,
            }
        )
    return rows


def analyze_retained_summary(
    payload: dict, *, source_sha256: str, source_path: str | None = None
) -> dict:
    """Analyze the supplied aggregate horizon census without pseudoreplication."""

    block_census = payload.get("block_gain_census")
    horizons_payload = block_census.get("horizons") if isinstance(
        block_census, dict
    ) else None
    scope = payload.get("inferential_scope")
    observer = payload.get("observer_census")
    if (
        not isinstance(horizons_payload, dict)
        or not isinstance(scope, dict)
        or not isinstance(observer, dict)
    ):
        raise ValueError("retained summary has an unknown schema")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in "0123456789abcdef" for character in source_sha256)
    ):
        raise ValueError("source_sha256 must be a lowercase SHA-256 digest")
    required_scope = {
        "block_counts_are_descriptive_not_independent_samples": True,
        "block_windows_overlap": True,
        "finite_windows_do_not_establish_an_all_future_premise": True,
    }
    if any(scope.get(name) is not value for name, value in required_scope.items()):
        raise ValueError("retained summary does not preserve inferential guards")

    rows = []
    for key, value in horizons_payload.items():
        horizon = int(key)
        quantiles = value["endpoint_gain_quantiles"]
        resolved = value["resolved"]
        resolved_quantiles = resolved["endpoint_gain_quantiles"]
        excursion = value["maximum_relative_excursion_quantiles"]
        censored = value["censored_endpoint_population"]
        windows = value["windows"]
        median = float(quantiles["median"])
        p05 = float(quantiles["p05"])
        p95 = float(quantiles["p95"])
        maximum = float(quantiles["maximum"])
        comparisons = int(value["comparisons"])
        contracting_fraction = float(value["contracting_fraction"])
        window_expanding_fractions = [
            float(window["expanding"]) / int(window["comparisons"])
            for window in windows
        ]
        window_maxima = [float(window["maximum_gain"]) for window in windows]
        if (
            horizon < 1
            or comparisons < 1
            or not 0.0 <= contracting_fraction <= 1.0
            or median <= 0.0
            or not 0.0 <= p05 <= median <= p95 <= maximum
            or len(windows) != int(value["contributing_windows"])
        ):
            raise ValueError("retained horizon census is inconsistent")
        rows.append(
            {
                "horizon": horizon,
                "comparisons": comparisons,
                "contracting_fraction": contracting_fraction,
                "median_gain": median,
                "p05_gain": p05,
                "p95_gain": p95,
                "maximum_gain": maximum,
                "effective_log_gain_per_step": math.log(median) / horizon,
                "every_window_contains_expansion": bool(
                    value["every_window_contains_expansion"]
                ),
                "censored_endpoint_comparisons": int(censored["comparisons"]),
                "resolved_comparisons": int(resolved["comparisons"]),
                "resolved_median_gain": float(resolved_quantiles["median"]),
                "resolved_p05_gain": float(resolved_quantiles["p05"]),
                "resolved_p95_gain": float(resolved_quantiles["p95"]),
                "maximum_relative_excursion": {
                    "median": float(excursion["median"]),
                    "p95": float(excursion["p95"]),
                    "maximum": float(excursion["maximum"]),
                },
                "window_maximum_gain_range": [
                    min(window_maxima), max(window_maxima)
                ],
                "window_expanding_fraction_range": [
                    min(window_expanding_fractions),
                    max(window_expanding_fractions),
                ],
            }
        )
    rows.sort(key=lambda row: row["horizon"])
    expected = [1, 2, 4, 8, 16, 32]
    if [row["horizon"] for row in rows] != expected:
        raise ValueError("retained horizons differ from the registered scale grid")

    block_sizes = [row["horizon"] for row in rows]
    log_medians = [math.log(row["median_gain"]) for row in rows]
    fit = through_origin_fit(block_sizes, log_medians)
    intercept_fit = ordinary_linear_fit(block_sizes, log_medians)
    rate = float(fit["slope"])
    for row in rows:
        row["fitted_median_gain"] = math.exp(rate * row["horizon"])
        row["log_fit_residual"] = (
            math.log(row["median_gain"]) - rate * row["horizon"]
        )

    power_fit = None
    if all(value < 0.0 for value in log_medians):
        power_fit = ordinary_linear_fit(
            np.log(np.asarray(block_sizes, dtype=np.float64)),
            np.log(-np.asarray(log_medians, dtype=np.float64)),
        )
        power_fit["power_exponent"] = power_fit["slope"]

    effective_rates = [row["effective_log_gain_per_step"] for row in rows]
    strongest_rate = min(effective_rates)
    scale_drift = {
        "b1_rate": effective_rates[0],
        "b32_rate": effective_rates[-1],
        "relative_weakening_from_b1_to_b32": (
            relative_weakening(effective_rates[0], effective_rates[-1])
        ),
        "relative_weakening_from_strongest_to_b32": (
            relative_weakening(strongest_rate, effective_rates[-1])
        ),
        "through_origin_residual_signs": [
            -1 if row["log_fit_residual"] < 0.0 else
            1 if row["log_fit_residual"] > 0.0 else 0
            for row in rows
        ],
        "b1_homogeneous_prediction_at_b32": rows[0]["median_gain"] ** 32,
        "observed_b32_median": rows[-1]["median_gain"],
    }

    all_straddle = all(
        row["p05_gain"] < 1.0 < row["p95_gain"] for row in rows
    )
    all_windows_expand = all(
        row["every_window_contains_expansion"] for row in rows
    )
    all_medians_contract = all(row["median_gain"] < 1.0 for row in rows)
    exact_zeros = int(observer["all_exact_zero_maps"])
    map_count = int(observer["all_maps"])
    effect_floor = float(observer["effect_floor"])
    collection_rows = observer.get("collections", {})
    floor_censored = sum(
        int(collection_rows[name]["floor_censored_positive_maps"])
        for name in sorted(collection_rows)
    )
    if (
        map_count < 1
        or not 0 <= exact_zeros <= map_count
        or not 0 <= floor_censored <= map_count
        or effect_floor <= 0.0
    ):
        raise ValueError("retained observer census is inconsistent")
    uniform_contraction = all(row["maximum_gain"] < 1.0 for row in rows)
    cell_envelopes = _retained_cell_envelopes(observer)
    cell_counts = {
        name: sum(row["classification"] == name for row in cell_envelopes)
        for name in (
            "DECREASED_ENVELOPE",
            "INCREASED_ENVELOPE",
            "OVERLAPPING_ENVELOPES",
        )
    }
    ten_order_decreases = sum(
        row["decreased_by_at_least_ten_orders"] for row in cell_envelopes
    )
    outcome = (
        "AGGREGATE_MEDIANS_BELOW_ONE_WITH_HETEROGENEOUS_CELLS"
        if (
            rate < 0.0
            and all_medians_contract
            and all_straddle
            and all_windows_expand
            and (
                cell_counts["INCREASED_ENVELOPE"] > 0
                or cell_counts["OVERLAPPING_ENVELOPES"] > 0
            )
        )
        else "NO_REGISTERED_AGGREGATE_MEDIAN_CONTRACTION"
    )
    record = {
        "schema_version": "pldr-row-rg-retained-scale-analysis-v3",
        "source": {
            "path": source_path,
            "file_sha256": source_sha256,
            "schema_version": payload.get("schema_version"),
            "content_sha256": payload.get("content_sha256"),
            "target_release_id": payload.get("target_release_id"),
        },
        "inferential_status": "DESCRIPTIVE_AGGREGATE",
        "overlapping_comparison_count": int(sum(row["comparisons"] for row in rows)),
        "observer_map_count": map_count,
        "exact_face_map_count": exact_zeros,
        "effect_floor": effect_floor,
        "floor_censored_positive_map_count": floor_censored,
        "collection_census": collection_rows,
        "block_all_stored_energies_strictly_positive": bool(
            block_census["all_stored_energies_strictly_positive"]
        ),
        "horizons": rows,
        "median_log_flow_fit": fit,
        "median_log_flow_fit_with_intercept": intercept_fit,
        "negative_log_median_power_fit": power_fit,
        "scale_drift": scale_drift,
        "median_log_flow_rate_per_update": rate,
        "median_e_folding_updates": (-1.0 / rate if rate < 0.0 else None),
        "cell_energy_envelopes": cell_envelopes,
        "cell_envelope_classification_counts": cell_counts,
        "cells_decreased_by_at_least_ten_orders": ten_order_decreases,
        "all_registered_medians_contract": all_medians_contract,
        "all_registered_quantile_bands_straddle_one": all_straddle,
        "every_window_contains_expansion_at_every_horizon": all_windows_expand,
        "uniform_contraction_observed": uniform_contraction,
        "per_orbit_lyapunov_rate_computed": False,
        "source_sector_exercised": False,
        "universality_identified": False,
        "outcome": outcome,
        "scope": {
            "comparisons_are_not_treated_as_independent": True,
            "fit_is_descriptive_and_has_no_sampling_p_value": True,
            "finite_windows_do_not_prove_asymptotic_collapse": True,
            "aggregate_quantiles_do_not_test_the_exact_semigroup": True,
            "pooled_median_is_not_an_orbitwise_lyapunov_exponent": True,
            "floor_censored_positive_values_are_not_face_hits": True,
        },
    }
    return seal_record(record)

def _numeric_summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return {"count": 0}
    if not np.isfinite(array).all():
        raise ValueError("summary values must be finite")
    return {
        "count": int(len(array)),
        "minimum": float(np.min(array)),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def _gain_summary(values: list[float], block_size: int) -> dict:
    array = np.asarray(values, dtype=np.float64)
    summary = _numeric_summary(values)
    if not len(array):
        return summary
    if np.any(array < 0.0):
        raise ValueError("gain samples must be nonnegative")
    positive = array[array > 0.0]
    summary.update(
        {
            "zero_count": int(np.sum(array == 0.0)),
            "contracting_fraction": float(np.mean(array < 1.0)),
            "marginal_fraction": float(np.mean(array == 1.0)),
            "expanding_fraction": float(np.mean(array > 1.0)),
            "positive_log_gain_count": int(len(positive)),
            "conditional_positive_mean_log_gain_per_step": (
                float(np.mean(np.log(positive)) / block_size)
                if len(positive) else None
            ),
        }
    )
    if len(positive) >= 4:
        logs = np.log(positive)
        if float(np.var(logs)) > 0.0:
            summary["positive_log_gain"] = standardized_moments(logs)
        else:
            summary["positive_log_gain"] = {
                "count": int(len(logs)),
                "mean": float(logs[0]),
                "standard_deviation": 0.0,
                "skewness": None,
                "excess_kurtosis": None,
                "degenerate": True,
            }
    return summary


def analyze_energy_array(energies, block_sizes=(1, 2, 4, 8, 16, 32)) -> dict:
    """Analyze raw row-energy trajectories under aligned exact blocking.

    Input shape is ``(trajectory, time, map)``.  A two-dimensional input is
    interpreted as one trajectory.  All zeros remain exact zeros.
    """

    array = np.asarray(energies, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, :, :]
    if (
        array.ndim != 3
        or array.shape[0] < 1
        or array.shape[1] < 2
        or array.shape[2] < 1
    ):
        raise ValueError("energies must have shape (trajectory, time, map)")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("energies must be finite and nonnegative")
    raw_sizes = tuple(block_sizes)
    if any(
        not isinstance(value, (int, np.integer)) or isinstance(value, bool)
        for value in raw_sizes
    ):
        raise ValueError("block_sizes must contain integers")
    sizes = tuple(int(value) for value in raw_sizes)
    if not sizes or sizes[0] != 1 or any(value < 1 for value in sizes):
        raise ValueError("block_sizes must begin at one and remain positive")
    if tuple(sorted(set(sizes))) != sizes:
        raise ValueError("block_sizes must be strictly increasing")

    trajectories, times, maps = array.shape
    levels = []
    maximum_semigroup_residual = 0.0
    for block_size in sizes:
        starts = np.arange(0, times - block_size, block_size)
        if not len(starts):
            raise ValueError("trajectory is too short for a requested block size")
        homogeneous_gains: list[float] = []
        transported_sources: list[float] = []
        face_free_gains: list[float] = []
        endpoint_ratios: list[float] = []
        source_fractions: list[float] = []
        homogeneous_fractions: list[float] = []
        block_start_faces = 0
        block_endpoint_restarts = 0
        blocks_touching_face = 0
        blocks_with_transported_source = 0
        level_residual = 0.0
        for trajectory in range(trajectories):
            for map_index in range(maps):
                path = array[trajectory, :, map_index]
                fine_edges = canonical_edges(path.tolist())
                for start in starts:
                    stop = int(start + block_size)
                    direct = block_edges(fine_edges[int(start):stop])
                    source = float(path[start])
                    target = float(path[stop])
                    predicted = direct.act(source)
                    scale = max(abs(source), abs(target), 1e-300)
                    level_residual = max(
                        level_residual, abs(predicted - target) / scale
                    )
                    homogeneous_gains.append(direct.gain)
                    transported_sources.append(direct.source)
                    if direct.source > 0.0:
                        blocks_with_transported_source += 1
                    touches_face = bool(np.any(path[int(start):stop + 1] == 0.0))
                    if touches_face:
                        blocks_touching_face += 1
                    else:
                        face_free_gains.append(direct.gain)
                    if source == 0.0:
                        block_start_faces += 1
                        if target > 0.0:
                            block_endpoint_restarts += 1
                    if source > 0.0:
                        endpoint_ratios.append(target / source)
                    if target > 0.0:
                        source_fractions.append(direct.source / target)
                        homogeneous_fractions.append(
                            direct.gain * source / target
                        )
        maximum_semigroup_residual = max(
            maximum_semigroup_residual, level_residual
        )
        level = {
            "block_size": block_size,
            "aligned_blocks_per_map": int(len(starts)),
            "total_blocks": int(trajectories * maps * len(starts)),
            "block_start_face_count": block_start_faces,
            "block_endpoint_restart_count": block_endpoint_restarts,
            "blocks_touching_exact_face": blocks_touching_face,
            "blocks_with_transported_source": blocks_with_transported_source,
            "semigroup_replay_relative_residual": level_residual,
            "homogeneous_gain": _gain_summary(
                homogeneous_gains, block_size
            ),
            "transported_source": _numeric_summary(transported_sources),
            "face_free_gain": _gain_summary(face_free_gains, block_size),
            "positive_start_endpoint_ratio": _gain_summary(
                endpoint_ratios, block_size
            ),
            "transported_source_fraction_at_positive_endpoint":
                _numeric_summary(source_fractions),
            "homogeneous_fraction_at_positive_endpoint":
                _numeric_summary(homogeneous_fractions),
        }
        levels.append(level)

    return seal_record(
        {
            "schema_version": "pldr-row-rg-energy-array-analysis-v2",
            "inferential_status": "DESCRIPTIVE_CENSUS",
            "shape": {
                "trajectories": trajectories,
                "times": times,
                "maps": maps,
            },
            "block_sizes": list(sizes),
            "fine_exact_face_state_count": int(np.sum(array == 0.0)),
            "fine_face_closure_count": int(
                np.sum((array[:, :-1, :] > 0.0) & (array[:, 1:, :] == 0.0))
            ),
            "fine_face_restart_count": int(
                np.sum((array[:, :-1, :] == 0.0) & (array[:, 1:, :] > 0.0))
            ),
            "levels": levels,
            "maximum_semigroup_replay_relative_residual": maximum_semigroup_residual,
            "scope": {
                "aligned_blocks_are_not_treated_as_independent": True,
                "endpoint_ratios_are_not_identified_with_homogeneous_gain_across_faces": True,
                "no_sampling_p_values_are_reported": True,
            },
        }
    )
