import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from scripts.analyze_energy_npz import load
from scripts.run_two_gpu_confirmation import execute as execute_two_gpu
from row_rgmap.analysis import (
    analyze_energy_array,
    seal_record,
    verify_record_seal,
)
from row_rgmap.core import (
    Edge,
    block_edges,
    canonical_edges,
    clean_flow,
    compose,
    homogeneous_block,
    normalize,
    replay,
)
from row_rgmap.recorder import RowEnergyRecorder, row_centered_energy


class AffineCocycleTests(unittest.TestCase):
    def test_composition_acts_chronologically(self):
        first = Edge(0.8, 0.1)
        second = Edge(1.2, 0.3)
        blocked = compose(second, first)
        self.assertAlmostEqual(blocked.act(2.0), second.act(first.act(2.0)))

    def test_associativity(self):
        a = Edge(0.8, 0.1)
        b = Edge(1.2, 0.3)
        c = Edge(0.7, 0.2)
        left = compose(c, compose(b, a))
        right = compose(compose(c, b), a)
        self.assertAlmostEqual(left.gain, right.gain)
        self.assertAlmostEqual(left.source, right.source)

    def test_gauge_covariance(self):
        first = Edge(0.8, 0.1)
        second = Edge(1.2, 0.3)
        left = compose(normalize(second, 3.0, 5.0), normalize(first, 2.0, 3.0))
        right = normalize(compose(second, first), 2.0, 5.0)
        self.assertAlmostEqual(left.gain, right.gain)
        self.assertAlmostEqual(left.source, right.source)

    def test_canonical_replay_across_zero(self):
        energies = [2.0, 1.0, 0.0, 0.25, 0.125]
        edges = canonical_edges(energies)
        self.assertEqual(replay(energies[0], edges), energies)
        blocked = block_edges(edges)
        self.assertEqual(blocked.act(energies[0]), energies[-1])

    def test_homogeneous_closed_form(self):
        edge = homogeneous_block(0.9, 0.2, 7)
        self.assertAlmostEqual(edge.gain, 0.9**7)
        self.assertAlmostEqual(edge.source, 0.2 * (1.0 - 0.9**7) / 0.1)

    def test_clean_continuous_semigroup(self):
        q = 0.97
        self.assertAlmostEqual(clean_flow(clean_flow(q, 3.0), 5.0), clean_flow(q, 15.0))
        with self.assertRaises(ValueError):
            clean_flow(0.0, 0.0)


class EnergyAnalysisTests(unittest.TestCase):
    def test_record_seal_detects_tampering(self):
        record = seal_record({"answer": 42})
        verify_record_seal(record)
        record["answer"] = 43
        with self.assertRaisesRegex(ValueError, "does not replay"):
            verify_record_seal(record)

    def test_raw_array_semigroup(self):
        q = 0.99
        times = 65
        base = q ** np.arange(times, dtype=float)
        energies = np.stack([base, 2.0 * base], axis=1)[None, :, :]
        result = analyze_energy_array(energies)
        self.assertLess(result["maximum_semigroup_replay_relative_residual"], 1e-14)
        for level in result["levels"]:
            expected = q ** level["block_size"]
            self.assertAlmostEqual(level["homogeneous_gain"]["median"], expected)
            self.assertAlmostEqual(level["face_free_gain"]["median"], expected)
            self.assertAlmostEqual(
                level["positive_start_endpoint_ratio"]["median"], expected
            )
            self.assertEqual(level["blocks_touching_exact_face"], 0)

    def test_raw_array_separates_face_source_from_gain(self):
        energies = np.asarray([[[2.0], [1.0], [0.0], [0.5], [0.25]]])
        result = analyze_energy_array(energies, block_sizes=(1, 2, 4))
        level = result["levels"][-1]
        self.assertEqual(result["fine_face_closure_count"], 1)
        self.assertEqual(result["fine_face_restart_count"], 1)
        self.assertEqual(level["blocks_touching_exact_face"], 1)
        self.assertEqual(level["blocks_with_transported_source"], 1)
        self.assertEqual(level["homogeneous_gain"]["median"], 0.0)
        self.assertEqual(level["transported_source"]["median"], 0.25)
        self.assertEqual(level["positive_start_endpoint_ratio"]["median"], 0.125)
        self.assertEqual(level["face_free_gain"]["count"], 0)
        self.assertEqual(
            level["transported_source_fraction_at_positive_endpoint"]["median"],
            1.0,
        )
        self.assertEqual(
            level["homogeneous_fraction_at_positive_endpoint"]["median"], 0.0
        )

    def test_block_size_types_are_strict(self):
        energies = np.ones((1, 3, 1), dtype=np.float64)
        with self.assertRaises(ValueError):
            analyze_energy_array(energies, block_sizes=(1, 2.0))

    def test_npz_contract_rejects_non_native_energy_dtype(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            np.savez(
                path,
                energies=np.ones((1, 3, 1), dtype=np.float32),
                steps=np.arange(3, dtype=np.int64),
                trajectory_ids=np.asarray(["a"]),
                map_ids=np.asarray(["m"]),
            )
            with self.assertRaisesRegex(ValueError, "native float64"):
                load(path)


class RecorderTests(unittest.TestCase):
    def test_row_centered_energy(self):
        matrix = np.asarray([[0.0, 4.0], [2.0, 4.0]])
        self.assertEqual(row_centered_energy(matrix), 2.0)
        self.assertEqual(row_centered_energy(np.ones((3, 2))), 0.0)
        self.assertEqual(
            row_centered_energy(np.full((3, 2), 1.0e308)), 0.0
        )
        with self.assertRaisesRegex(ValueError, "must be real"):
            row_centered_energy(np.ones((2, 2), dtype=np.complex128))

    def test_recorder_writes_strict_contract(self):
        recorder = RowEnergyRecorder(["map-a", "map-b"])
        recorder.record(
            7,
            {
                "map-a": np.asarray([[0.0], [2.0]]),
                "map-b": np.ones((2, 1)),
            },
        )
        recorder.record(
            8,
            {
                "map-a": np.asarray([[0.0], [1.0]]),
                "map-b": np.asarray([[0.0], [2.0]]),
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "energies.npz"
            recorder.write_npz(path, "trajectory-a")
            energies, steps, trajectories, maps = load(path)
            self.assertEqual(energies.dtype, np.dtype(np.float64))
            self.assertEqual(energies.shape, (1, 2, 2))
            self.assertEqual(steps.tolist(), [7, 8])
            self.assertEqual(trajectories.tolist(), ["trajectory-a"])
            self.assertEqual(maps.tolist(), ["map-a", "map-b"])


class TwoGpuRunnerTests(unittest.TestCase):
    def test_registered_runner_end_to_end_and_resume(self):
        fixture = Path(__file__).with_name("fixture_energy_producer.py")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "schema_version":
                            "pldr-row-rg-confirmation-protocol-v2",
                        "expected_updates": 32,
                        "block_sizes": [1, 2, 4],
                        "identity_tolerances": {
                            "float64_affine_replay_relative": 1e-12,
                            "float32_gate_shape_relative": 1e-12,
                            "plga_decomposition_relative": 1e-12,
                        },
                        "decision_rules": {
                            "estimation_fraction": 0.5,
                            "time_batch_length": 4,
                            "minimum_time_batches": 2,
                            "familywise_alpha": 0.05,
                            "stationarity": {
                                "maximum_split_shift_z": 100.0,
                                "minimum_variance_ratio": 0.0,
                                "maximum_variance_ratio": 1e6,
                                "maximum_absolute_lag1_correlation": 1.0,
                            },
                            "gaussian_calibration": {
                                "seed": 11,
                                "replicates": 20,
                                "quantile": 0.95,
                                "minimum_blocks": 4,
                                "maximum_absolute_excess_kurtosis": 100.0,
                                "maximum_final_to_initial_ks_ratio": 10.0,
                            },
                            "plga_transfer": {
                                "maximum_constant_input_defect_norm": 1e-6,
                                "maximum_decomposition_relative_residual": 1e-12,
                                "maximum_sampled_directional_gain": 100.0,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "schema_version":
                            "pldr-row-rg-two-gpu-confirmation-v2",
                        "run_id": "deterministic-fixture",
                        "devices": [0, 1],
                        "expected_updates": 32,
                        "producer_timeout_seconds": 60,
                        "protocol_path": str(protocol_path),
                        "producer_command": [
                            sys.executable,
                            str(fixture),
                            "--logical-device",
                            "cuda:0",
                            "--physical-device",
                            "{device}",
                            "--steps",
                            "32",
                            "--seed",
                            "{device}",
                            "--output",
                            "{output}",
                            "--metadata-output",
                            "{metadata}",
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "result"
            missing_output = root / "missing-result"
            with self.assertRaisesRegex(FileNotFoundError, "complete existing"):
                execute_two_gpu(config_path, missing_output, reanalyze=True)
            self.assertFalse((missing_output / "device-0-producer.log").exists())
            record = execute_two_gpu(config_path, output)
            self.assertEqual(
                record["schema_version"],
                "pldr-row-rg-two-gpu-confirmation-result-v3",
            )
            self.assertEqual(
                record["exact_rg_analysis"]["shape"]["trajectories"], 2
            )
            self.assertTrue((output / "combined-energies.npz").is_file())
            self.assertTrue((output / "confirmation-result.json").is_file())
            self.assertTrue((output / "run-manifest.json").is_file())
            verify_record_seal(record)
            state = json.loads(
                (output / "run-state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["status"], "COMPLETE")
            self.assertIn(
                "outcome", record["registered_decision"]
            )

            reanalyzed = execute_two_gpu(config_path, output, reanalyze=True)
            self.assertEqual(
                reanalyzed["registered_decision"],
                record["registered_decision"],
            )
            resumed = execute_two_gpu(config_path, output, resume=True)
            self.assertEqual(resumed, reanalyzed)
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                execute_two_gpu(
                    config_path, output, resume=True, reanalyze=True
                )
            protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
            protocol["decision_rules"]["gaussian_calibration"]["seed"] = 12
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different protocol"):
                execute_two_gpu(config_path, output, resume=True)
            with self.assertRaises(FileExistsError):
                execute_two_gpu(config_path, output)


if __name__ == "__main__":
    unittest.main()
