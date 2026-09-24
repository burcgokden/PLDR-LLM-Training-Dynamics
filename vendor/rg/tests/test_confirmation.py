import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

import numpy as np

from scripts.check_formal_soundness import forbidden_escape_hatches
from row_rgmap.confirmation import (
    _normal_ks_calibration,
    analyze_confirmation,
)
from row_rgmap.statistics import (
    normal_ks_distance,
    ordinary_linear_fit,
    through_origin_fit,
)


ROOT = Path(__file__).resolve().parents[1]


def _protocol() -> dict:
    return {
        "schema_version": "pldr-row-rg-confirmation-protocol-v2",
        "expected_updates": 32,
        "block_sizes": [1, 2, 4],
        "decision_rules": {
            "estimation_fraction": 0.5,
            "time_batch_length": 4,
            "minimum_time_batches": 2,
            "familywise_alpha": 0.05,
            "stationarity": {
                "maximum_split_shift_z": 3.0,
                "minimum_variance_ratio": 0.5,
                "maximum_variance_ratio": 2.0,
                "maximum_absolute_lag1_correlation": 0.2,
            },
            "gaussian_calibration": {
                "seed": 17,
                "replicates": 20,
                "quantile": 0.95,
                "minimum_blocks": 4,
                "maximum_absolute_excess_kurtosis": 1.5,
                "maximum_final_to_initial_ks_ratio": 0.8,
            },
        },
    }


def _energy_from_log_edges(log_edges: np.ndarray) -> np.ndarray:
    log_edges = np.asarray(log_edges, dtype=np.float64)
    if log_edges.ndim == 1:
        log_edges = log_edges[:, None]
    log_energy = np.concatenate(
        [
            np.zeros((1, log_edges.shape[1]), dtype=np.float64),
            np.cumsum(log_edges, axis=0),
        ],
        axis=0,
    )
    return np.exp(log_energy)[None, :, :]


class CalibrationTests(unittest.TestCase):
    def test_normal_null_uses_fitted_location_and_scale(self):
        observed = _normal_ks_calibration(
            16,
            replicates=12,
            quantile=0.75,
            generator=np.random.default_rng(23),
        )
        generator = np.random.default_rng(23)
        distances = []
        for _ in range(12):
            sample = generator.normal(size=16)
            sample = (sample - sample.mean()) / sample.std()
            distances.append(normal_ks_distance(sample))
        expected = float(np.quantile(distances, 0.75))
        self.assertAlmostEqual(observed, expected)


class ConfirmationDecisionTests(unittest.TestCase):
    def test_subcritical_label_requires_negative_estimation_mean(self):
        update = np.arange(16, dtype=np.float64)
        estimation = 0.02 + 0.001 * np.sin(update)
        holdout = -0.02 + 0.001 * np.cos(update)
        energies = _energy_from_log_edges(
            np.concatenate([estimation, holdout])
        )
        result = analyze_confirmation(
            energies,
            np.asarray(["trajectory"]),
            np.asarray(["c0.L0.H0"]),
            _protocol(),
        )
        cell = result["cells"][0]
        self.assertLess(cell["holdout_upper_confidence_bound"], 0.0)
        self.assertGreater(cell["estimation_mean_log_gain_per_update"], 0.0)
        self.assertEqual(cell["flow_classification"], "UNRESOLVED")
        self.assertFalse(
            result["all_cells_subcritical_on_registered_finite_horizon"]
        )

    def test_all_supercritical_outcome_is_reported(self):
        update = np.arange(32, dtype=np.float64)
        energies = _energy_from_log_edges(
            0.02 + 0.001 * np.sin(0.73 * update)
        )
        result = analyze_confirmation(
            energies,
            np.asarray(["trajectory"]),
            np.asarray(["c0.L0.H0"]),
            _protocol(),
        )
        self.assertEqual(
            result["cell_flow_classifications"]["SUPERCRITICAL"], 1
        )
        self.assertTrue(result["outcome"].startswith("ALL_CELLS_SUPERCRITICAL_"))

    def test_consecutive_face_hits_have_zero_amplification(self):
        path = np.exp(-0.01 * np.arange(33, dtype=np.float64))
        path[[5, 6, 10]] = 0.0
        result = analyze_confirmation(
            path[None, :, None],
            np.asarray(["trajectory"]),
            np.asarray(["c0.L0.H0"]),
            _protocol(),
        )
        self.assertTrue(result["source_sector_exercised"])
        self.assertEqual(
            result["cell_flow_classifications"]["FACE_INTERRUPTED"], 1
        )
        first = result["excursion_census"][0]["excursions"][0]
        self.assertEqual(first["restart"], 0.0)
        self.assertEqual(first["maximum_amplification"], 0.0)
        self.assertEqual(
            first["restart_times_maximum_amplification"], 0.0
        )


class StatisticsRegressionTests(unittest.TestCase):
    def test_degenerate_fits_are_json_safe(self):
        through = through_origin_fit([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        ordinary = ordinary_linear_fit([1.0, 2.0, 3.0], [4.0, 4.0, 4.0])
        self.assertTrue(through["degenerate_fit"])
        self.assertIsNone(through["r_squared_about_mean"])
        self.assertTrue(ordinary["degenerate_fit"])
        self.assertIsNone(ordinary["r_squared_about_mean"])
        json.dumps(
            {"through": through, "ordinary": ordinary},
            allow_nan=False,
        )




if __name__ == "__main__":
    unittest.main()
