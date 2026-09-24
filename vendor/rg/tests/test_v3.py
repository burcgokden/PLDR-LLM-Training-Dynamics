import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import stats

from row_rgmap.confirmation_v3 import analyze_confirmation_v3
from row_rgmap.diagnostics_v3 import face_excursion_ledger
from row_rgmap.recorder import RowEnergyRecorder
from scripts.run_confirmation_v3 import (
    _append_event,
    _event_rows,
    _exact_block_census,
    _missing_final_artifacts,
    _next_directory,
)
from scripts.run_stage_resolved_observer import (
    exact_dyadic_row_energy,
    maximum_coordinate_ulp_span,
)
from scripts.pldr_energy_producer_v3 import (
    _derivative_pair,
    _gate_shape_residuals,
    _plga_map,
)


def _protocol() -> dict:
    return {
        "schema_version": "pldr-row-rg-confirmation-protocol-v3",
        "protocol_id": "unit-test",
        "expected_updates": 2048,
        "segments": {
            "burn_in_updates": 512,
            "design_updates": 512,
            "holdout_updates": 1024,
        },
        "analysis": {
            "familywise_alpha": 0.05,
            "familywise_cell_count": 1,
            "time_batch_length": 64,
            "window_length": 128,
            "target_mde_per_update": 0.01,
            "law_block_sizes": [128, 256],
            "variance_block_sizes": [32, 64, 128],
            "excursion_events": {
                "layer": 1,
                "absolute_log_gain_threshold": 1.0,
                "cluster_gap_updates": 16,
            },
            "stationarity": {
                "maximum_split_t": 10.0,
                "maximum_trend_t": 10.0,
                "minimum_variance_ratio": 0.01,
                "maximum_variance_ratio": 100.0,
                "maximum_design_holdout_shift_t": 10.0,
                "minimum_design_holdout_variance_ratio": 0.01,
                "maximum_design_holdout_variance_ratio": 100.0,
            },
            "gaussian_fixed_law": {
                "seed": 19,
                "bootstrap_replicates": 99,
                "minimum_blocks": 4,
                "diagnostic_band_alpha": 0.05,
            },
            "variance_scaling": {
                "minimum_variance_exponent": -10.0,
                "maximum_variance_exponent": 10.0,
                "maximum_design_holdout_exponent_shift": 10.0,
            },
            "regular_drift_prediction": {
                "minimum_design_r_squared": 0.0,
                "minimum_gamma": -10.0,
                "maximum_gamma": 10.0,
                "maximum_cumulative_relative_error": 100.0,
            },
        },
    }


def _energies(log_edges: np.ndarray) -> np.ndarray:
    logs = np.concatenate([[0.0], np.cumsum(log_edges)])
    return np.exp(logs)[None, :, None]


class ConfirmationV3Tests(unittest.TestCase):
    def test_negative_holdout_drift_uses_student_t(self):
        generator = np.random.default_rng(7)
        log_edges = -0.002 + generator.normal(scale=0.001, size=2048)
        result = analyze_confirmation_v3(
            _energies(log_edges),
            np.asarray(["trajectory"]),
            np.asarray(["c0.L1.H0"]),
            _protocol(),
        )
        cell = result["cells"][0]
        expected = stats.t.ppf(0.95, 15)
        self.assertAlmostEqual(cell["student_t_critical_value"], expected)
        self.assertEqual(cell["flow_classification"], "NEGATIVE_DRIFT")
        self.assertTrue(cell["adequately_powered"])
        self.assertEqual(cell["holdout_map_sign_counts"]["negative"], 1)
        self.assertLess(
            cell["cumulative_log_identity_relative_residual"], 1e-12
        )
        self.assertIn("excess_kurtosis", cell["holdout_update_standardized_moments"])
        self.assertGreater(
            cell["design_diagnostics"]["batch_long_run_variance_estimate"], 0.0
        )
        self.assertEqual(len(cell["block_prediction_diagnostics"]), 2)
        self.assertEqual(result["large_increment_excursion_ledger"]["map_series_count"], 1)
        self.assertEqual(result["large_increment_excursion_ledger"]["event_count"], 0)

    def test_nonstationarity_prevents_stationary_phase_label(self):
        protocol = _protocol()
        protocol["analysis"]["stationarity"]["maximum_trend_t"] = 1.0
        update = np.arange(2048, dtype=np.float64)
        log_edges = -0.004 + 2e-6 * update + 1e-4 * np.sin(update)
        result = analyze_confirmation_v3(
            _energies(log_edges),
            np.asarray(["trajectory"]),
            np.asarray(["c0.L0.H0"]),
            protocol,
        )
        cell = result["cells"][0]
        self.assertFalse(cell["temporal_domain_stable"])
        self.assertEqual(cell["phase_classification"], "NONAUTONOMOUS")

    def test_exact_face_interrupts_log_analysis(self):
        path = _energies(-0.001 * np.ones(2048))
        path[0, 1000, 0] = 0.0
        result = analyze_confirmation_v3(
            path,
            np.asarray(["trajectory"]),
            np.asarray(["c0.L0.H0"]),
            _protocol(),
        )
        self.assertEqual(result["cells"][0]["flow_classification"], "FACE_INTERRUPTED")
        self.assertTrue(result["source_sector_exercised"])


class V3InfrastructureTests(unittest.TestCase):
    def test_next_directory_parses_directory_and_json_suffixes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "analysis-0001").mkdir()
            (root / "analysis-0002.json").touch()
            (root / "analysis-unrelated").touch()
            self.assertEqual(
                _next_directory(root, "analysis").name, "analysis-0003"
            )

    def test_recorder_checkpoint_roundtrip(self):
        recorder = RowEnergyRecorder(["map"])
        recorder.record(4, {"map": np.asarray([[0.0], [2.0]])})
        recorder.record(5, {"map": np.asarray([[0.0], [1.0]])})
        restored = RowEnergyRecorder(["map"])
        restored.load_state_dict(recorder.state_dict())
        restored.record(6, {"map": np.asarray([[0.0], [0.5]])})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "record.npz"
            restored.write_npz(output, "trajectory")
            with np.load(output, allow_pickle=False) as payload:
                self.assertEqual(payload["steps"].tolist(), [4, 5, 6])

    def test_append_only_event_chain_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.jsonl"
            _append_event(path, "FIRST", {"value": 1})
            _append_event(path, "SECOND", {"value": 2})
            self.assertEqual(len(_event_rows(path)), 2)
            rows = path.read_text(encoding="utf-8").splitlines()
            first = json.loads(rows[0])
            first["payload"]["value"] = 3
            rows[0] = json.dumps(first, sort_keys=True, separators=(",", ":"))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not replay"):
                _event_rows(path)

    def test_analysis_only_completeness_guard_lists_missing_artifacts(self):
        config = {
            "trajectories": [
                {"trajectory_id": "t0"},
                {"trajectory_id": "t1"},
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = root / "trajectories" / "t0"
            paths.mkdir(parents=True)
            for name in ("energies.npz", "metrics.npz", "metadata.json"):
                (paths / name).touch()
            missing = _missing_final_artifacts(config, root)
        self.assertEqual(
            missing,
            [
                "trajectories/t1/energies.npz",
                "trajectories/t1/metrics.npz",
                "trajectories/t1/metadata.json",
            ],
        )

    def test_fast_block_census_retains_transported_source(self):
        energies = np.asarray([[[2.0], [1.0], [0.0], [0.5], [0.25]]])
        result = _exact_block_census(energies, [1, 2, 4])
        block = result["levels"][-1]
        self.assertEqual(block["blocks_touching_exact_face"], 1)
        self.assertEqual(block["blocks_with_transported_source"], 1)
        self.assertEqual(block["transported_source"]["median"], 0.25)
        ledger = face_excursion_ledger(
            energies, ["trajectory"], ["c0.L1.H0"]
        )
        self.assertEqual(ledger["closure_count"], 1)
        self.assertEqual(ledger["restart_count"], 1)
        excursion = ledger["excursions"][0]
        self.assertEqual(excursion["amplification_M"], 1.0)
        self.assertEqual(excursion["restart_times_amplification"], 0.5)

    def test_exact_dyadic_energy_separates_zero_and_one_ulp_rows(self):
        identical = np.ones((4, 3), dtype=np.float64)
        self.assertEqual(exact_dyadic_row_energy(identical), (0, 0))
        self.assertEqual(maximum_coordinate_ulp_span(identical), 0)

        successor = np.nextafter(np.float64(1.0), np.float64(2.0))
        one_ulp = np.asarray(
            [[1.0], [1.0], [1.0], [successor]], dtype=np.float64
        )
        self.assertEqual(exact_dyadic_row_energy(one_ulp), (3, -106))
        self.assertEqual(maximum_coordinate_ulp_span(one_ulp), 1)

        two_ulp = one_ulp.copy()
        two_ulp[-1, 0] = np.nextafter(successor, np.float64(2.0))
        self.assertGreater(exact_dyadic_row_energy(two_ulp)[0], 0)
        self.assertEqual(maximum_coordinate_ulp_span(two_ulp), 2)

    def test_gate_shape_backward_error_is_conditioned_near_face(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch is optional for this algebraic regression")

        before = torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 4.0, 2.0, 3.0],
            ],
            dtype=torch.float32,
        )
        gamma = torch.full((4,), 1e-9, dtype=torch.float32)
        bias = torch.tensor([1.0, -2.0, 3.0, -4.0], dtype=torch.float32)
        epsilon = 1e-6
        centered = before - before.mean(dim=-1, keepdim=True)
        shape = centered / torch.sqrt(
            epsilon + (centered * centered).mean(dim=-1, keepdim=True)
        )
        after = (shape * gamma + bias).float()
        residuals = _gate_shape_residuals(
            torch, before, after, gamma, bias, epsilon
        )
        self.assertLess(residuals["map_backward_relative"], 2e-6)
        self.assertEqual(
            residuals["quotient_energy_symmetric_relative"], 1.0
        )

    def test_plga_derivative_is_projected_to_the_row_quotient(self):
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError:
            self.skipTest("PyTorch is optional for this derivative regression")

        generator = torch.Generator().manual_seed(41)
        weight = torch.randn((3, 3), generator=generator, dtype=torch.float64)
        bias = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        power = 0.8 + torch.rand((3, 2), generator=generator, dtype=torch.float64)
        coupling = torch.randn((3, 3), generator=generator, dtype=torch.float64)
        coupling_bias = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        parameters = (weight, bias, power, coupling, coupling_bias)
        point = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        direction = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        direction -= direction.mean(dim=0, keepdim=True)
        probe = torch.randn((3, 2), generator=generator, dtype=torch.float64)
        probe -= probe.mean(dim=0, keepdim=True)
        action, adjoint = _derivative_pair(torch, functional, parameters, point)
        epsilon = 1e-6
        finite_difference = (
            _plga_map(functional, parameters, point + epsilon * direction)
            - _plga_map(functional, parameters, point - epsilon * direction)
        ) / (2.0 * epsilon)
        finite_difference -= finite_difference.mean(dim=0, keepdim=True)
        self.assertTrue(
            torch.allclose(action(direction), finite_difference, rtol=2e-8, atol=2e-8)
        )
        self.assertAlmostEqual(
            float(torch.sum(action(direction) * probe)),
            float(torch.sum(direction * adjoint(probe))),
            places=10,
        )
        constant_direction = torch.ones_like(direction)
        self.assertEqual(float(torch.linalg.vector_norm(action(constant_direction))), 0.0)
        self.assertLess(float(torch.linalg.vector_norm(adjoint(probe).mean(dim=0))), 1e-12)


if __name__ == "__main__":
    unittest.main()
