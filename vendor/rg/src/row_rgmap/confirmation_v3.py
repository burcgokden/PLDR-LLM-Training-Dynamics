"""Power-aware finite-scale analysis for the four-trajectory campaign.

The registered unit of temporal inference is a nonoverlapping batch of the
trajectory-layer mean of per-map log gains.  Maps in a cell are retained for
heterogeneity diagnostics, but are never counted as independent time
replicates.  Gaussian fixed-law checks use matched-size parametric bootstrap
calibration and a campaign-wide Holm correction.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
from typing import Any

import numpy as np
from scipy import stats

from .diagnostics_v3 import (
    block_prediction_diagnostics,
    large_increment_excursion_ledger,
)
from .statistics import normal_ks_distance, ordinary_linear_fit, standardized_moments


SCHEMA = "pldr-row-rg-confirmation-decision-v3"
PROTOCOL_SCHEMA = "pldr-row-rg-confirmation-protocol-v3"
_MAP_PATTERN = re.compile(
    r"^c(?P<context>[0-9]+)[.]L(?P<layer>[0-9]+)[.]H(?P<head>[0-9]+)$"
)


def _parse_registry(map_ids: Any) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for raw in np.asarray(map_ids).tolist():
        map_id = str(raw)
        match = _MAP_PATTERN.fullmatch(map_id)
        if match is None:
            raise ValueError(f"invalid map identifier: {map_id}")
        rows.append(
            {
                "map_id": map_id,
                "context": int(match.group("context")),
                "layer": int(match.group("layer")),
                "head": int(match.group("head")),
            }
        )
    if len({row["map_id"] for row in rows}) != len(rows):
        raise ValueError("map registry contains duplicates")
    return rows


def _correlation(values: np.ndarray, lag: int) -> float | None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if lag < 1 or len(values) <= lag:
        return None
    left = values[:-lag]
    right = values[lag:]
    if left.std() == 0.0 or right.std() == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _batch_values(values: np.ndarray, length: int) -> tuple[np.ndarray, int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    usable = len(values) // length * length
    if usable == 0:
        return np.empty(0, dtype=np.float64), len(values)
    batches = values[:usable].reshape(-1, length).mean(axis=1)
    return batches, len(values) - usable


def _series_diagnostics(values: np.ndarray, batch_length: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    batches, discarded = _batch_values(values, batch_length)
    batch_lag1 = _correlation(batches, 1)
    ess = None
    capped_ess = None
    if batch_lag1 is not None and batch_lag1 > -1.0:
        ess = float(len(batches) * (1.0 - batch_lag1) / (1.0 + batch_lag1))
        capped_ess = float(min(len(batches), max(1.0, ess)))
    return {
        "update_count": int(len(values)),
        "mean": float(values.mean()),
        "variance": float(values.var()),
        "batch_length": batch_length,
        "batch_count": int(len(batches)),
        "discarded_updates": int(discarded),
        "batch_mean_standard_deviation": (
            float(batches.std(ddof=1)) if len(batches) > 1 else None
        ),
        "batch_long_run_variance_estimate": (
            float(batch_length * batches.var(ddof=1))
            if len(batches) > 1 else None
        ),
        "update_autocorrelation": {
            str(lag): _correlation(values, lag)
            for lag in (1, 8, 64, batch_length)
        },
        "batch_lag1_correlation": batch_lag1,
        "ar1_effective_batch_count": ess,
        "ar1_effective_batch_count_capped": capped_ess,
    }


def _trend_t(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(values) < 3:
        return None, None
    x = np.linspace(-0.5, 0.5, len(values), dtype=np.float64)
    fit = ordinary_linear_fit(x, values)
    fitted = fit["intercept"] + fit["slope"] * x
    residual = values - fitted
    denominator = float(np.sum(x * x))
    variance = float(np.sum(residual * residual) / (len(values) - 2))
    if denominator == 0.0 or variance == 0.0:
        return None, float(fit["slope"])
    standard_error = math.sqrt(variance / denominator)
    return abs(float(fit["slope"])) / standard_error, float(fit["slope"])


def _split_t(values: np.ndarray) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    midpoint = len(values) // 2
    left = values[:midpoint]
    right = values[midpoint:]
    if len(left) < 2 or len(right) < 2:
        return None, None
    denominator = math.sqrt(
        float(left.var(ddof=1)) / len(left)
        + float(right.var(ddof=1)) / len(right)
    )
    statistic = (
        abs(float(right.mean() - left.mean())) / denominator
        if denominator > 0.0 else None
    )
    ratio = (
        float(right.var(ddof=1) / left.var(ddof=1))
        if left.var(ddof=1) > 0.0 else None
    )
    return statistic, ratio


def _design_stability(
    values: np.ndarray, batch_length: int, rules: dict[str, Any]
) -> dict[str, Any]:
    batches, _ = _batch_values(values, batch_length)
    split_t, variance_ratio = _split_t(batches)
    trend_t, trend_slope = _trend_t(batches)
    passed = (
        split_t is not None
        and split_t <= float(rules["maximum_split_t"])
        and trend_t is not None
        and trend_t <= float(rules["maximum_trend_t"])
        and variance_ratio is not None
        and float(rules["minimum_variance_ratio"])
        <= variance_ratio
        <= float(rules["maximum_variance_ratio"])
    )
    return {
        "split_mean_t": split_t,
        "batch_trend_t": trend_t,
        "batch_trend_slope": trend_slope,
        "split_variance_ratio": variance_ratio,
        "passed": bool(passed),
    }


def _cross_segment_stability(
    design: np.ndarray,
    holdout: np.ndarray,
    batch_length: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    design_batches, _ = _batch_values(design, batch_length)
    holdout_batches, _ = _batch_values(holdout, batch_length)
    denominator = math.sqrt(
        float(design_batches.var(ddof=1)) / len(design_batches)
        + float(holdout_batches.var(ddof=1)) / len(holdout_batches)
    )
    mean_shift_t = (
        abs(float(holdout_batches.mean() - design_batches.mean())) / denominator
        if denominator > 0.0 else None
    )
    variance_ratio = (
        float(holdout_batches.var(ddof=1) / design_batches.var(ddof=1))
        if design_batches.var(ddof=1) > 0.0 else None
    )
    passed = (
        mean_shift_t is not None
        and mean_shift_t <= float(rules["maximum_design_holdout_shift_t"])
        and variance_ratio is not None
        and float(rules["minimum_design_holdout_variance_ratio"])
        <= variance_ratio
        <= float(rules["maximum_design_holdout_variance_ratio"])
    )
    return {
        "mean_shift_t": mean_shift_t,
        "batch_variance_ratio": variance_ratio,
        "passed": bool(passed),
    }


def _calibration_seed(base: int, sample_count: int) -> int:
    payload = f"{base}:{sample_count}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _normal_calibration(
    sample_count: int, replicates: int, base_seed: int
) -> dict[str, np.ndarray]:
    if sample_count < 4 or replicates < 99:
        raise ValueError("normal calibration is too small")
    generator = np.random.default_rng(_calibration_seed(base_seed, sample_count))
    distances = np.empty(replicates, dtype=np.float64)
    kurtoses = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        sample = generator.normal(size=sample_count)
        sample = (sample - sample.mean()) / sample.std()
        distances[index] = normal_ks_distance(sample)
        kurtoses[index] = standardized_moments(sample)["excess_kurtosis"]
    return {"ks": distances, "kurtosis": kurtoses}


def _law_rows(
    values: np.ndarray,
    block_sizes: list[int],
    rules: dict[str, Any],
    cache: dict[int, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    rows: list[dict[str, Any]] = []
    for block_size in block_sizes:
        count = len(centered) // block_size
        if count < int(rules["minimum_blocks"]):
            raise ValueError("registered law scale has too few blocks")
        sums = centered[: count * block_size].reshape(count, block_size).sum(axis=1)
        if sums.std() == 0.0:
            raise ValueError("registered law scale is degenerate")
        normalized = (sums - sums.mean()) / sums.std()
        distance = normal_ks_distance(normalized)
        moments = standardized_moments(normalized)
        if count not in cache:
            cache[count] = _normal_calibration(
                count,
                int(rules["bootstrap_replicates"]),
                int(rules["seed"]),
            )
        calibration = cache[count]
        p_value = float(
            (1 + np.sum(calibration["ks"] >= distance))
            / (len(calibration["ks"]) + 1)
        )
        kurtosis_low, kurtosis_high = np.quantile(
            calibration["kurtosis"],
            [float(rules["diagnostic_band_alpha"]) / 2.0,
             1.0 - float(rules["diagnostic_band_alpha"]) / 2.0],
        )
        rows.append(
            {
                "block_size": block_size,
                "block_count": count,
                "normal_ks_distance": distance,
                "normal_ks_bootstrap_p_value": p_value,
                "normal_ks_holm_adjusted_p_value": None,
                "normal_ks_rejected": None,
                "skewness": moments["skewness"],
                "excess_kurtosis": moments["excess_kurtosis"],
                "matched_normal_excess_kurtosis_band": [
                    float(kurtosis_low), float(kurtosis_high)
                ],
            }
        )
    return rows


def _holm(rows: list[dict[str, Any]], alpha: float) -> None:
    ordered = sorted(
        enumerate(rows), key=lambda item: item[1]["normal_ks_bootstrap_p_value"]
    )
    running = 0.0
    count = len(ordered)
    for rank, (_, row) in enumerate(ordered):
        adjusted = min(
            1.0,
            max(running, (count - rank) * row["normal_ks_bootstrap_p_value"]),
        )
        running = adjusted
        row["normal_ks_holm_adjusted_p_value"] = adjusted
        row["normal_ks_rejected"] = bool(adjusted <= alpha)


def _variance_scaling(values: np.ndarray, block_sizes: list[int]) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    centered = values - values.mean()
    rows = []
    for block_size in block_sizes:
        count = len(centered) // block_size
        sums = centered[: count * block_size].reshape(count, block_size).sum(axis=1)
        if count < 4 or sums.var(ddof=1) <= 0.0:
            raise ValueError("variance-scaling block is unavailable")
        rows.append(
            {
                "block_size": block_size,
                "block_count": count,
                "block_sum_variance": float(sums.var(ddof=1)),
            }
        )
    fit = ordinary_linear_fit(
        np.log([row["block_size"] for row in rows]),
        np.log([row["block_sum_variance"] for row in rows]),
    )
    return {
        "rows": rows,
        "variance_exponent": float(fit["slope"]),
        "fluctuation_exponent_H": float(fit["slope"]) / 2.0,
        "fit": fit,
    }


def _window_means(
    values: np.ndarray, start_step: int, window_length: int
) -> list[dict[str, Any]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    count = len(values) // window_length
    return [
        {
            "first_update": start_step + index * window_length + 1,
            "last_update": start_step + (index + 1) * window_length,
            "midpoint_update": (
                start_step + (index + 0.5) * window_length
            ),
            "mean_log_gain_per_update": float(
                values[index * window_length : (index + 1) * window_length].mean()
            ),
        }
        for index in range(count)
    ]


def _regular_drift_prediction(
    design_windows: list[dict[str, Any]],
    holdout: np.ndarray,
    holdout_start: int,
    rules: dict[str, Any],
) -> dict[str, Any]:
    rates = np.asarray(
        [row["mean_log_gain_per_update"] for row in design_windows],
        dtype=np.float64,
    )
    times = np.asarray(
        [row["midpoint_update"] for row in design_windows], dtype=np.float64
    )
    eligible = bool(len(rates) >= 3 and np.all(rates < 0.0))
    if not eligible:
        return {
            "eligible": False,
            "reason": "design-window rates are not uniformly negative",
            "heldout_prediction_passed": False,
        }
    fit = ordinary_linear_fit(np.log(times), np.log(-rates))
    gamma = -float(fit["slope"])
    coefficient = math.exp(float(fit["intercept"]))
    update_times = np.arange(
        holdout_start + 1, holdout_start + len(holdout) + 1, dtype=np.float64
    )
    predicted = -coefficient * np.power(update_times, -gamma)
    predicted_cumulative = float(predicted.sum())
    observed_cumulative = float(np.sum(holdout))
    relative_error = abs(observed_cumulative - predicted_cumulative) / max(
        abs(predicted_cumulative), 1e-300
    )
    passed = (
        float(fit["r_squared_about_mean"] or 0.0)
        >= float(rules["minimum_design_r_squared"])
        and float(rules["minimum_gamma"]) <= gamma <= float(rules["maximum_gamma"])
        and math.copysign(1.0, observed_cumulative)
        == math.copysign(1.0, predicted_cumulative)
        and relative_error <= float(rules["maximum_cumulative_relative_error"])
    )
    return {
        "eligible": True,
        "model": "mean log gain = -c t^{-gamma}",
        "coefficient_c": coefficient,
        "gamma": gamma,
        "design_fit": fit,
        "predicted_holdout_cumulative_log_gain": predicted_cumulative,
        "observed_holdout_cumulative_log_gain": observed_cumulative,
        "cumulative_relative_error": relative_error,
        "heldout_prediction_passed": bool(passed),
    }


def _energy_checkpoints(
    energy: np.ndarray, indices: list[int], steps: list[int]
) -> list[dict[str, Any]]:
    return [
        {
            "step": step,
            "mean_energy": float(energy[step, indices].mean()),
            "median_energy": float(np.median(energy[step, indices])),
            "minimum_energy": float(energy[step, indices].min()),
            "maximum_energy": float(energy[step, indices].max()),
        }
        for step in steps
    ]


def analyze_confirmation_v3(
    energies: Any,
    trajectory_ids: Any,
    map_ids: Any,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Apply the immutable v3 design to consecutive energy trajectories."""

    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unknown v3 confirmation protocol schema")
    array = np.asarray(energies, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] < 3 or array.shape[2] < 1:
        raise ValueError("energies must have shape (trajectory, time, map)")
    if not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("energies must be finite and nonnegative")
    names = [str(value) for value in np.asarray(trajectory_ids).tolist()]
    registry = _parse_registry(map_ids)
    if len(names) != array.shape[0] or len(registry) != array.shape[2]:
        raise ValueError("trajectory or map registry dimensions do not match")
    if len(set(names)) != len(names):
        raise ValueError("trajectory identifiers must be unique")

    segments = protocol["segments"]
    burn = int(segments["burn_in_updates"])
    design_count = int(segments["design_updates"])
    holdout_count = int(segments["holdout_updates"])
    expected = burn + design_count + holdout_count
    if array.shape[1] != expected + 1 or protocol["expected_updates"] != expected:
        raise ValueError("energy horizon differs from the registered segments")
    analysis_rules = protocol["analysis"]
    batch_length = int(analysis_rules["time_batch_length"])
    if design_count % batch_length or holdout_count % batch_length:
        raise ValueError("registered segments must contain complete time batches")

    positive = array > 0.0
    edge_positive = positive[:, :-1, :] & positive[:, 1:, :]
    log_gain = np.full(edge_positive.shape, np.nan, dtype=np.float64)
    log_gain[edge_positive] = (
        np.log(array[:, 1:, :][edge_positive])
        - np.log(array[:, :-1, :][edge_positive])
    )
    by_layer: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(registry):
        by_layer[int(row["layer"])].append(index)
    cell_count = len(names) * len(by_layer)
    if cell_count != int(analysis_rules["familywise_cell_count"]):
        raise ValueError("registered familywise cell count differs from the data")

    holdout_batches = holdout_count // batch_length
    critical = float(
        stats.t.ppf(
            1.0 - float(analysis_rules["familywise_alpha"]) / cell_count,
            holdout_batches - 1,
        )
    )
    law_cache: dict[int, dict[str, np.ndarray]] = {}
    all_law_rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    checkpoint_steps = [0, burn, burn + design_count, expected]
    window_length = int(analysis_rules["window_length"])
    large_increment = large_increment_excursion_ledger(
        array,
        names,
        registry,
        log_gain,
        analysis_rules["excursion_events"],
        burn,
        design_count,
    )

    for trajectory_index, trajectory_id in enumerate(names):
        for layer, indices in sorted(by_layer.items()):
            cell_edges = log_gain[trajectory_index][:, indices]
            complete = bool(np.isfinite(cell_edges).all())
            base: dict[str, Any] = {
                "trajectory_id": trajectory_id,
                "layer": layer,
                "map_count": len(indices),
                "positive_excursion_complete": complete,
                "censored_log_edge_count": int(np.sum(~np.isfinite(cell_edges))),
                "energy_checkpoints": _energy_checkpoints(
                    array[trajectory_index], indices, checkpoint_steps
                ),
            }
            if not complete:
                base.update(
                    {
                        "flow_classification": "FACE_INTERRUPTED",
                        "phase_classification": "FACE_INTERRUPTED",
                        "finite_variance_fixed_law_consistent": False,
                    }
                )
                cells.append(base)
                continue
            series = cell_edges.mean(axis=1)
            design = series[burn : burn + design_count]
            holdout = series[burn + design_count :]
            design_diag = _series_diagnostics(design, batch_length)
            holdout_diag = _series_diagnostics(holdout, batch_length)
            design_batches, _ = _batch_values(design, batch_length)
            observed_batches, _ = _batch_values(holdout, batch_length)
            design_stability = _design_stability(
                design, batch_length, analysis_rules["stationarity"]
            )
            cross_stability = _cross_segment_stability(
                design,
                holdout,
                batch_length,
                analysis_rules["stationarity"],
            )
            temporal_stability = bool(
                design_stability["passed"] and cross_stability["passed"]
            )
            design_mean = float(design.mean())
            selected_direction = "NEGATIVE" if design_mean < 0.0 else "POSITIVE"
            variance_degrees = len(design_batches) - 1
            variance_lower_quantile = float(stats.chi2.ppf(
                float(analysis_rules["familywise_alpha"]) / cell_count,
                variance_degrees,
            ))
            design_batch_variance_upper = float(
                variance_degrees * design_batches.var(ddof=1)
                / variance_lower_quantile
            )
            projected_se = math.sqrt(
                design_batch_variance_upper / holdout_batches
            )
            projected_mde = critical * projected_se
            powered = projected_mde <= float(analysis_rules["target_mde_per_update"])
            holdout_mean = float(holdout.mean())
            standard_error = float(
                observed_batches.std(ddof=1) / math.sqrt(len(observed_batches))
            )
            lower = holdout_mean - critical * standard_error
            upper = holdout_mean + critical * standard_error
            if not powered:
                flow = "UNDERPOWERED"
            elif selected_direction == "NEGATIVE" and upper < 0.0:
                flow = "NEGATIVE_DRIFT"
            elif selected_direction == "POSITIVE" and lower > 0.0:
                flow = "POSITIVE_DRIFT"
            else:
                flow = "UNRESOLVED"
            if not temporal_stability:
                phase = "NONAUTONOMOUS"
            elif flow == "NEGATIVE_DRIFT":
                phase = "SUBCRITICAL"
            elif flow == "POSITIVE_DRIFT":
                phase = "SUPERCRITICAL"
            else:
                phase = flow

            law_rows = _law_rows(
                holdout,
                [int(value) for value in analysis_rules["law_block_sizes"]],
                analysis_rules["gaussian_fixed_law"],
                law_cache,
            )
            all_law_rows.extend(law_rows)
            block_predictions = block_prediction_diagnostics(
                design, holdout,
                [int(value) for value in analysis_rules["law_block_sizes"]],
            )
            design_scaling = _variance_scaling(
                design,
                [int(value) for value in analysis_rules["variance_block_sizes"]],
            )
            holdout_scaling = _variance_scaling(
                holdout,
                [int(value) for value in analysis_rules["variance_block_sizes"]],
            )
            scaling_rules = analysis_rules["variance_scaling"]
            scaling_consistent = (
                float(scaling_rules["minimum_variance_exponent"])
                <= holdout_scaling["variance_exponent"]
                <= float(scaling_rules["maximum_variance_exponent"])
                and abs(
                    holdout_scaling["variance_exponent"]
                    - design_scaling["variance_exponent"]
                ) <= float(scaling_rules["maximum_design_holdout_exponent_shift"])
            )
            design_windows = _window_means(design, burn, window_length)
            holdout_windows = _window_means(
                holdout, burn + design_count, window_length
            )
            regular_prediction = _regular_drift_prediction(
                design_windows,
                holdout,
                burn + design_count,
                analysis_rules["regular_drift_prediction"],
            )
            map_means = cell_edges[burn + design_count :, :].mean(axis=0)
            log_telescoping = float(np.sum(holdout))
            endpoint_telescoping = float(
                np.mean(
                    np.log(array[trajectory_index, -1, indices])
                    - np.log(array[trajectory_index, burn + design_count, indices])
                )
            )
            base.update(
                {
                    "design_mean_log_gain_per_update": design_mean,
                    "selected_holdout_direction": selected_direction,
                    "design_batch_variance": float(design_batches.var(ddof=1)),
                    "design_batch_variance_simultaneous_upper": design_batch_variance_upper,
                    "design_variance_degrees_of_freedom": variance_degrees,
                    "design_variance_chi_square_lower_quantile": variance_lower_quantile,
                    "projected_holdout_standard_error_from_design": projected_se,
                    "projected_holdout_mde": projected_mde,
                    "adequately_powered": bool(powered),
                    "student_t_critical_value": critical,
                    "holdout_mean_log_gain_per_update": holdout_mean,
                    "holdout_batch_mean_standard_error": standard_error,
                    "holdout_lower_simultaneous_bound": lower,
                    "holdout_upper_simultaneous_bound": upper,
                    "holdout_map_sign_counts": {
                        "negative": int(np.sum(map_means < 0.0)),
                        "zero": int(np.sum(map_means == 0.0)),
                        "positive": int(np.sum(map_means > 0.0)),
                    },
                    "holdout_map_mean_log_gain_range": [
                        float(map_means.min()), float(map_means.max())
                    ],
                    "design_diagnostics": design_diag,
                    "holdout_diagnostics": holdout_diag,
                    "design_update_standardized_moments": standardized_moments(
                        design
                    ),
                    "holdout_update_standardized_moments": standardized_moments(
                        holdout
                    ),
                    "design_stability": design_stability,
                    "design_holdout_stability": cross_stability,
                    "temporal_domain_stable": temporal_stability,
                    "design_window_means": design_windows,
                    "holdout_window_means": holdout_windows,
                    "regular_drift_prediction": regular_prediction,
                    "block_law_diagnostics": law_rows,
                    "block_prediction_diagnostics": block_predictions,
                    "design_variance_scaling": design_scaling,
                    "holdout_variance_scaling": holdout_scaling,
                    "finite_variance_scaling_consistent": bool(scaling_consistent),
                    "cumulative_log_identity_relative_residual": (
                        abs(log_telescoping - endpoint_telescoping)
                        / max(abs(endpoint_telescoping), 1e-300)
                    ),
                    "flow_classification": flow,
                    "phase_classification": phase,
                }
            )
            cells.append(base)

    _holm(all_law_rows, float(analysis_rules["familywise_alpha"]))
    for cell in cells:
        if not cell["positive_excursion_complete"]:
            continue
        gaussian_pass = not any(
            row["normal_ks_rejected"] for row in cell["block_law_diagnostics"]
        )
        cell["gaussian_fixed_law_not_rejected"] = gaussian_pass
        cell["finite_variance_fixed_law_consistent"] = bool(
            cell["temporal_domain_stable"]
            and cell["adequately_powered"]
            and gaussian_pass
            and cell["finite_variance_scaling_consistent"]
        )

    source_exercised = bool(np.any(array == 0.0))
    counts = {
        value: sum(cell["flow_classification"] == value for cell in cells)
        for value in (
            "NEGATIVE_DRIFT", "POSITIVE_DRIFT", "UNRESOLVED",
            "UNDERPOWERED", "FACE_INTERRUPTED",
        )
    }
    phase_counts = {
        value: sum(cell["phase_classification"] == value for cell in cells)
        for value in (
            "SUBCRITICAL", "SUPERCRITICAL", "UNRESOLVED",
            "UNDERPOWERED", "NONAUTONOMOUS", "FACE_INTERRUPTED",
        )
    }
    complete_cells = [cell for cell in cells if cell["positive_excursion_complete"]]
    finite_variance_consistent = bool(
        len(complete_cells) == len(cells)
        and all(cell["finite_variance_fixed_law_consistent"] for cell in complete_cells)
    )
    return {
        "schema_version": SCHEMA,
        "inferential_scope": "REGISTERED_FINITE_SCALE_HOLDOUT",
        "calibration": {
            "status": "nominal_for_native_dependent_batches",
            "student_reference": "Conditional on all sign-selection/design information, within-cell holdout batch means are independent normal draws with common mean and positive variance.",
            "chisquare_reference": "Independent normal design batch means; variance stability or a transport bound is additionally needed to project holdout width.",
            "multiplicity": "Bonferroni needs valid marginal pivots, not independence between cells; joint design/holdout coverage requires combined error allocation.",
            "native_dependence_justification_established": False,
            "precision_projection_is_not_power_or_realized_width_guarantee": True,
        },
        "segments": {
            "burn_in_updates": burn,
            "design_updates": design_count,
            "holdout_updates": holdout_count,
        },
        "trajectory_count": len(names),
        "cell_count": len(cells),
        "registry": registry,
        "large_increment_excursion_ledger": large_increment,
        "cells": cells,
        "flow_classification_counts": counts,
        "phase_classification_counts": phase_counts,
        "every_cell_adequately_powered": bool(
            complete_cells and all(cell["adequately_powered"] for cell in complete_cells)
        ),
        "every_temporal_domain_stable": bool(
            complete_cells and all(cell["temporal_domain_stable"] for cell in complete_cells)
        ),
        "normal_ks_familywise_hypothesis_count": len(all_law_rows),
        "normal_ks_familywise_rejection_count": sum(
            bool(row["normal_ks_rejected"]) for row in all_law_rows
        ),
        "finite_variance_fixed_law_consistent_on_registered_scales": (
            finite_variance_consistent
        ),
        "universality_class_identified": False,
        "source_sector_exercised": source_exercised,
        "minimum_energy": float(array.min()),
        "exact_face_state_count": int(np.sum(array == 0.0)),
        "maximum_cumulative_log_identity_relative_residual": max(
            (
                float(cell.get("cumulative_log_identity_relative_residual", 0.0))
                for cell in cells
            ),
            default=0.0,
        ),
        "interpretation": {
            "finite_scale_nonrejection_is_not_asymptotic_identification": True,
            "nonstationary_cells_receive_no_stationary_phase_label": True,
            "map_signs_are_heterogeneity_diagnostics_not_time_replicates": True,
            "power_is_computed_from_design_variance_only": True,
            "holdout_does_not_retune_the_registered_model": True,
        },
    }
