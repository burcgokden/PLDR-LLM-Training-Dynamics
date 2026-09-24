"""Regression tests for the observer-stratum and block-gain reducers."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.analyze_observer_block import (
    block_gain_census,
    cell_stratum,
    classify_energies,
)
from analysis.check_observer_block import (
    scalar_quantiles,
    type7_quantile,
)


@pytest.mark.parametrize(
    ("values", "probability", "expected"),
    [
        ([7.0], 0.0, 7.0),
        ([7.0], 1.0, 7.0),
        ([0.0, 10.0], 0.05, 0.5),
        ([0.0, 10.0], 0.50, 5.0),
        ([0.0, 10.0], 0.95, 9.5),
        ([4.0, 1.0, 3.0, 2.0, 0.0], 0.25, 1.0),
    ],
)
def test_scalar_type7_quantile_boundary_fixtures(
    values: list[float], probability: float, expected: float,
) -> None:
    assert type7_quantile(values, probability) == expected


def test_scalar_type7_quantile_reports_complete_surface() -> None:
    assert scalar_quantiles([0.0, 10.0]) == {
        "p05": 0.5,
        "median": 5.0,
        "p95": 9.5,
        "maximum": 10.0,
    }


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_scalar_type7_quantile_rejects_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        type7_quantile([1.0, 2.0], probability)


def test_scalar_type7_quantile_rejects_empty_population() -> None:
    with pytest.raises(ValueError):
        type7_quantile([], 0.5)


def test_observer_partition_separates_zero_censored_and_resolved() -> None:
    values = np.array([
        [0.0, 1.0e-15, 1.0e-12, 1.0e-11],
        [0.0, 4.0e-13, 1.0e-12, 2.0e-12],
    ] * 3, dtype=np.float64).reshape(24, 1)
    values = np.repeat(values, 4, axis=1)
    partition = classify_energies(values, 1.0e-12)
    assert np.count_nonzero(partition["exact"]) == 24
    assert np.count_nonzero(partition["censored"]) == 48
    assert np.count_nonzero(partition["resolved"]) == 24
    assert np.all(
        partition["exact"] | partition["censored"] | partition["resolved"]
    )


def test_observer_partition_zero_floor_and_adjacent_float_boundaries() -> None:
    floor = 1.0e-12
    samples = np.asarray(
        [
            0.0,
            np.nextafter(0.0, np.inf),
            np.nextafter(floor, 0.0),
            floor,
            np.nextafter(floor, np.inf),
        ],
        dtype=np.float64,
    )
    values = np.resize(samples, 24 * 4).reshape(24, 4)
    partition = classify_energies(values, floor)
    flat_values = values.ravel()
    assert np.array_equal(partition["exact"].ravel(), flat_values == 0.0)
    assert np.array_equal(
        partition["censored"].ravel(),
        (flat_values > 0.0) & (flat_values <= floor),
    )
    assert np.array_equal(
        partition["resolved"].ravel(),
        flat_values > floor,
    )


def test_effect_floor_must_not_be_reclassified_as_exact_zero() -> None:
    values = np.full((24, 4), 1.0e-15, dtype=np.float64)
    partition = classify_energies(values, 1.0e-12)
    assert not np.any(partition["exact"])
    assert np.all(partition["censored"])
    assert cell_stratum({
        "exact": 0, "censored": 96, "resolved": 0,
    }, 96) == "floor-censored-positive"


@pytest.mark.parametrize("bad_floor", [0.0, -1.0, np.inf, np.nan])
def test_observer_partition_rejects_invalid_floor(bad_floor: float) -> None:
    with pytest.raises(ValueError):
        classify_energies(np.ones((24, 4)), bad_floor)


def test_block_census_retains_expanding_comparisons_at_every_horizon() -> None:
    windows = []
    for index in range(18):
        base = 1.0 + 0.01 * index
        sequence = np.full((65, 3), base, dtype=np.float64)
        sequence[:, 0] *= np.linspace(1.0, 0.8, 65)
        sequence[:, 1] *= np.linspace(1.0, 1.2, 65)
        sequence[:, 2] *= 1.0 + 0.01 * np.sin(np.arange(65))
        windows.append({
            "trajectory": "ABC"[index % 3],
            "center": index,
            "steps": list(range(65)),
            "energy": sequence,
        })
    result = block_gain_census(windows, 1.0e-12)
    assert result["all_stored_energies_strictly_positive"]
    for horizon in (1, 2, 4, 8, 16, 32):
        row = result["horizons"][str(horizon)]
        assert row["expanding"] > 0
        assert row["windows_with_expansion"] == 18
        assert row["every_window_contains_expansion"]
        assert not row["uniform_stored_block_contraction_observed"]


def test_block_census_keeps_censored_population_separate() -> None:
    windows = []
    for index in range(18):
        sequence = np.ones((65, 2), dtype=np.float64)
        sequence[:, 0] *= 2.0e-12
        sequence[:, 1] *= 5.0e-15
        sequence[1:, 0] *= 0.9
        sequence[1:, 1] *= 1.1
        windows.append({
            "trajectory": "A",
            "center": index,
            "steps": list(range(65)),
            "energy": sequence,
        })
    result = block_gain_census(windows, 1.0e-12)
    one = result["horizons"]["1"]
    assert one["resolved"]["comparisons"] > 0
    assert one["censored_endpoint_population"]["comparisons"] > 0
    assert one["censored_endpoint_population"]["expanding"] > 0
