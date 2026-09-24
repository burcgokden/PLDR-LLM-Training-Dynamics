import json
import io
from contextlib import redirect_stderr
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from row_rgmap.analysis import file_sha256, seal_record
from row_rgmap.provenance_v3 import git_blob_descriptor
import scripts.run_confirmation_v3 as runner
from scripts.verify_registered_artifacts import event_rows, verify


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).with_name("fixture_v3_producer.py")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class ProductionRunnerStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.behavior_path = self.workspace / "behavior.json"
        write_json(self.behavior_path, {})
        self.protocol_path = self.workspace / "protocol.json"
        self.protocol = self._protocol()
        write_json(self.protocol_path, self.protocol)
        self.qualification_path = self.workspace / "qualification.json"
        self.qualification = self._qualification()
        write_json(self.qualification_path, self.qualification)
        self.config_path = self.workspace / "config.json"
        self.config = self._config()
        write_json(self.config_path, self.config)
        self.run_root = self.workspace / "run"

    def _protocol(self) -> dict:
        return {
            "schema_version": "pldr-row-rg-confirmation-protocol-v3",
            "protocol_id": "production-runner-fixture-v3",
            "expected_updates": 32,
            "segments": {
                "burn_in_updates": 8,
                "design_updates": 8,
                "holdout_updates": 16,
            },
            "block_sizes": [1, 2, 4, 8],
            "observer_effect_floor": 1e-12,
            "identity_tolerances": {
                "float64_affine_replay_relative": 1e-12,
                "float32_gate_shape_map_backward_relative": 1e-6,
                "plga_decomposition_relative": 1e-12,
                "constrained_plga_defect_norm": 1e-9,
            },
            "analysis": {
                "familywise_alpha": 0.05,
                "familywise_cell_count": 12,
                "time_batch_length": 4,
                "window_length": 8,
                "target_mde_per_update": 0.01,
                "law_block_sizes": [2, 4],
                "variance_block_sizes": [1, 2],
                "excursion_events": {
                    "layer": 1,
                    "absolute_log_gain_threshold": 1.0,
                    "cluster_gap_updates": 2,
                },
                "stationarity": {
                    "maximum_split_t": 1e6,
                    "maximum_trend_t": 1e6,
                    "minimum_variance_ratio": 0.0,
                    "maximum_variance_ratio": 1e12,
                    "maximum_design_holdout_shift_t": 1e6,
                    "minimum_design_holdout_variance_ratio": 0.0,
                    "maximum_design_holdout_variance_ratio": 1e12,
                },
                "gaussian_fixed_law": {
                    "seed": 19,
                    "bootstrap_replicates": 99,
                    "minimum_blocks": 4,
                    "diagnostic_band_alpha": 0.05,
                },
                "variance_scaling": {
                    "minimum_variance_exponent": -100.0,
                    "maximum_variance_exponent": 100.0,
                    "maximum_design_holdout_exponent_shift": 100.0,
                },
                "regular_drift_prediction": {
                    "minimum_design_r_squared": 0.0,
                    "minimum_gamma": -100.0,
                    "maximum_gamma": 100.0,
                    "maximum_cumulative_relative_error": 1e6,
                },
            },
            "plga_analysis": {
                "checkpoint_steps": [0, 16, 32],
                "direct_defect_tolerance": 1e-6,
            },
        }

    def _qualification(self) -> dict:
        repository_path = "src/row_rgmap/analysis.py"
        digest, size = git_blob_descriptor(
            REPOSITORY_ROOT, self.commit, repository_path
        )
        return seal_record({
            "schema_version": "pldr-row-rg-confirmation-v3-qualification-v1",
            "protocol_id": self.protocol["protocol_id"],
            "protocol_file_sha256": file_sha256(self.protocol_path),
            "code_commit": self.commit,
            "gates": {"fixture_contract_passed": True},
            "all_launch_gates_passed": True,
            "qualification_sources": [{
                "repository_path": repository_path,
                "code_commit": self.commit,
                "sha256": digest,
                "size_bytes": size,
            }],
        })

    def _config(self) -> dict:
        trajectories = []
        for index in range(4):
            trajectories.append({
                "trajectory_id": f"fixture-t{index}",
                "wave": index // 2,
                "device": index % 2,
                "seed": 700 + index,
                "train_document_offset": 1000 + 100 * index,
            })
        return {
            "schema_version": "pldr-row-rg-four-trajectory-run-v3",
            "run_id": "production-runner-fixture-v3",
            "devices": [0, 1],
            "producer_timeout_seconds": 60,
            "protocol_path": str(self.protocol_path),
            "qualification": {
                "path": str(self.qualification_path),
                "file_sha256": file_sha256(self.qualification_path),
                "record_sha256": self.qualification["record_sha256"],
            },
            "trajectories": trajectories,
            "producer_command": [
                sys.executable,
                str(FIXTURE),
                "--physical-device", "{device}",
                "--trajectory-id", "{trajectory_id}",
                "--seed", "{seed}",
                "--train-document-offset", "{train_document_offset}",
                "--output", "{output}",
                "--metrics-output", "{metrics}",
                "--metadata-output", "{metadata}",
                "--checkpoint-dir", "{checkpoint_dir}",
                "--protocol", "{protocol}",
                "--code-commit", "{code_commit}",
                "--behavior", str(self.behavior_path),
            ],
        }

    def _execute(self, root: Path | None = None, **modes):
        selected_root = self.run_root if root is None else root
        with (
            patch.object(
                runner, "_git_identity", return_value=(self.commit, "")
            ),
            patch.object(
                runner,
                "_source_digest_at_commit",
                side_effect=lambda path, _commit: file_sha256(path),
            ),
        ):
            return runner.execute(
                self.config_path, selected_root, **modes
            )

    def _events(self, root: Path | None = None) -> list[dict]:
        selected_root = self.run_root if root is None else root
        return event_rows(selected_root / "events.jsonl")

    def _set_behavior(self, value: dict[str, str]) -> None:
        write_json(self.behavior_path, value)

    @staticmethod
    def _scientific_leaves(result: dict) -> dict:
        return {
            key: result[key]
            for key in (
                "exact_rg_analysis",
                "face_excursion_ledger",
                "registered_decision",
                "loss_analysis",
                "plga_analysis",
            )
        }
    def test_01_fresh_run_completes_and_verifies(self):
        result = self._execute()
        self.assertEqual(result["analysis_id"], "analysis-0001")
        self.assertEqual(verify(self.run_root)["selected_analysis_id"], "analysis-0001")
        self.assertEqual(
            sum(row["event"] == "PRODUCER_COMPLETED" for row in self._events()),
            4,
        )

    def test_02_existing_root_without_mode_refuses_before_write(self):
        self._execute()
        before = (self.run_root / "events.jsonl").read_bytes()
        with self.assertRaisesRegex(FileExistsError, "explicit continuation"):
            self._execute()
        self.assertEqual((self.run_root / "events.jsonl").read_bytes(), before)

    def test_03_analyze_only_incomplete_lists_missing_artifacts(self):
        self._set_behavior({"fixture-t0": "fail"})
        with self.assertRaises(RuntimeError):
            self._execute()
        self._set_behavior({})
        before = (self.run_root / "events.jsonl").read_bytes()
        attempts = len(list(self.run_root.glob("trajectories/*/producer-attempt-*.log")))
        with self.assertRaisesRegex(
            runner.ActiveRootContinuationError,
            "analyze-only requires complete final producer artifacts",
        ):
            self._execute(analyze_only=True)
        self.assertEqual((self.run_root / "events.jsonl").read_bytes(), before)
        self.assertEqual(
            len(list(self.run_root.glob("trajectories/*/producer-attempt-*.log"))),
            attempts,
        )


    def test_04_forced_producer_failure_records_attempt(self):
        self._set_behavior({"fixture-t0": "fail"})
        with self.assertRaisesRegex(RuntimeError, "producer failures"):
            self._execute()
        failures = [
            row for row in self._events() if row["event"] == "PRODUCER_FAILED"
        ]
        self.assertEqual(len(failures), 1)
        payload = failures[0]["payload"]
        self.assertEqual(payload["trajectory_id"], "fixture-t0")
        self.assertEqual(payload["return_code"], 17)
        log = self.run_root / payload["attempt_log"]
        self.assertEqual(payload["attempt_log_sha256"], file_sha256(log))

    def test_05_interruption_is_distinct_and_chain_valid(self):
        self._set_behavior({"fixture-t0": "interrupt"})
        with self.assertRaisesRegex(RuntimeError, "producer failures"):
            self._execute()
        interrupted = [
            row
            for row in self._events()
            if row["event"] == "PRODUCER_INTERRUPTED"
        ]
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0]["payload"]["return_code"], 130)
        self.assertEqual(
            interrupted[0]["payload"]["termination"]["classification"],
            "EXTERNAL_TERMINATION",
        )
        self.assertTrue(
            (
                self.run_root
                / "trajectories/fixture-t0/checkpoints/checkpoint-step000004.pt"
            ).is_file()
        )

    def test_06_resume_reproduces_uninterrupted_scientific_leaves(self):
        self._set_behavior({"fixture-t0": "interrupt"})
        with self.assertRaises(RuntimeError):
            self._execute()
        self._set_behavior({})
        resumed = self._execute(resume=True)
        uninterrupted = self._execute(root=self.workspace / "clean-run")
        self.assertEqual(
            self._scientific_leaves(resumed),
            self._scientific_leaves(uninterrupted),
        )
        self.assertEqual(
            verify(self.run_root)["selected_analysis_id"], "analysis-0001"
        )

    def test_07_complete_root_resume_is_a_verified_no_op(self):
        first = self._execute()
        before = (self.run_root / "events.jsonl").read_bytes()
        resumed = self._execute(resume=True)
        self.assertEqual(resumed["record_sha256"], first["record_sha256"])
        self.assertEqual((self.run_root / "events.jsonl").read_bytes(), before)

    def test_08_relocated_copy_analyzes_without_absolute_root_identity(self):
        self._execute()
        relocated = self.workspace / "relocated" / "copied-run"
        shutil.copytree(self.run_root, relocated)
        result = self._execute(root=relocated, analyze_only=True)
        self.assertEqual(result["analysis_id"], "analysis-0002")
        self.assertEqual(
            verify(relocated)["selected_analysis_id"], "analysis-0002"
        )

    def test_09_sequential_analyses_extend_the_manifest_chain(self):
        first = self._execute()
        second = self._execute(analyze_only=True)
        third = self._execute(analyze_only=True)
        self.assertEqual(
            [
                path.name
                for path in sorted(
                    (self.run_root / "manifests").glob("*.json")
                )
            ],
            ["manifest-0001.json", "manifest-0002.json", "manifest-0003.json"],
        )
        self.assertEqual(
            self._scientific_leaves(first), self._scientific_leaves(second)
        )
        self.assertEqual(
            self._scientific_leaves(second), self._scientific_leaves(third)
        )
        self.assertEqual(
            verify(self.run_root)["selected_analysis_id"], "analysis-0003"
        )

    def test_10_mutated_resolved_protocol_is_rejected_at_preflight(self):
        self._execute()
        before = (self.run_root / "events.jsonl").read_bytes()
        path = self.run_root / "protocol-resolved.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["observer_effect_floor"] = 2e-12
        write_json(path, value)
        with self.assertRaisesRegex(
            runner.FinalizedRootMutationError, "resolved protocol"
        ):
            self._execute(resume=True)
        self.assertEqual((self.run_root / "events.jsonl").read_bytes(), before)

    def test_11_mutated_producer_metadata_is_rejected_at_preflight(self):
        self._execute()
        before = (self.run_root / "events.jsonl").read_bytes()
        path = self.run_root / "trajectories/fixture-t0/metadata.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["seed"] += 1
        write_json(path, value)
        with self.assertRaisesRegex(
            runner.FinalizedRootMutationError, "metadata"
        ):
            self._execute(resume=True)
        self.assertEqual((self.run_root / "events.jsonl").read_bytes(), before)

    def test_12_damaged_finalized_roots_never_relaunch_producers(self):
        self._execute()
        broken = self.workspace / "broken-event"
        missing = self.workspace / "missing-artifact"
        shutil.copytree(self.run_root, broken)
        shutil.copytree(self.run_root, missing)

        event_path = broken / "events.jsonl"
        rows = event_path.read_text(encoding="utf-8").splitlines()
        row = json.loads(rows[1])
        row["previous_event_sha256"] = "0" * 64
        rows[1] = json.dumps(row, sort_keys=True, separators=(",", ":"))
        event_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        attempts = len(list(broken.glob("trajectories/*/producer-attempt-*.log")))
        with self.assertRaisesRegex(
            runner.FinalizedRootMutationError, "use a new output root"
        ):
            self._execute(root=broken, resume=True)
        self.assertEqual(
            len(list(broken.glob("trajectories/*/producer-attempt-*.log"))),
            attempts,
        )

        (missing / "trajectories/fixture-t0/energies.npz").unlink()
        attempts = len(list(missing.glob("trajectories/*/producer-attempt-*.log")))
        with self.assertRaisesRegex(
            runner.FinalizedRootMutationError, "final producer artifacts are missing"
        ):
            self._execute(root=missing, resume=True)
        self.assertEqual(
            len(list(missing.glob("trajectories/*/producer-attempt-*.log"))),
            attempts,
        )

    def test_13_imported_source_mismatch_is_rejected(self):
        self._set_behavior({"fixture-t0": "bad_source"})
        with self.assertRaisesRegex(ValueError, "imported producer source mismatch"):
            self._execute()

    def test_14_asymmetric_wave_completion_is_emitted_once(self):
        self._set_behavior({"fixture-t1": "interrupt"})
        with self.assertRaises(RuntimeError):
            self._execute()
        first_events = self._events()
        self.assertEqual(
            sum(
                row["event"] == "PRODUCER_COMPLETED"
                and row["payload"].get("trajectory_id") == "fixture-t0"
                for row in first_events
            ),
            1,
        )
        self.assertEqual(
            sum(row["event"] == "PRODUCER_INTERRUPTED" for row in first_events),
            1,
        )
        self._set_behavior({})
        self._execute(resume=True)
        final_events = self._events()
        for trajectory_id in ("fixture-t0", "fixture-t1"):
            self.assertEqual(
                sum(
                    row["event"] == "PRODUCER_COMPLETED"
                    and row["payload"].get("trajectory_id") == trajectory_id
                    for row in final_events
                ),
                1,
            )
        verify(self.run_root)

    def test_15_signal_mechanism_classification_does_not_infer_cause(self):
        cases = (
            (-2, "SIGINT", "subprocess-negative-signal"),
            (130, "SIGINT", "conventional-128-plus-signal"),
            (-9, "SIGKILL", "subprocess-negative-signal"),
            (137, "SIGKILL", "conventional-128-plus-signal"),
        )
        for return_code, expected_signal, expected_encoding in cases:
            event, metadata = runner._termination_metadata(return_code)
            self.assertEqual(event, "PRODUCER_INTERRUPTED")
            self.assertEqual(metadata["classification"], "EXTERNAL_TERMINATION")
            self.assertEqual(metadata["signal_name"], expected_signal)
            self.assertEqual(metadata["cause_attribution"], "UNDETERMINED")
            self.assertEqual(metadata["encoding"], expected_encoding)
        event, metadata = runner._termination_metadata(17)
        self.assertEqual(event, "PRODUCER_FAILED")
        self.assertEqual(metadata, {"classification": "PROCESS_EXIT", "exit_status": 17})

    def test_16_active_error_uses_public_exit_two_channel(self):
        error = runner.ActiveRootContinuationError("active root refusal")
        stderr = io.StringIO()
        argv = [
            "run_confirmation_v3.py",
            "--config", str(self.config_path),
            "--output-root", str(self.run_root),
            "--analyze-only",
        ]
        with (
            patch.object(sys, "argv", argv),
            patch.object(runner, "execute", side_effect=error),
            redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            runner.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertEqual(stderr.getvalue(), "runner error: active root refusal\n")

if __name__ == "__main__":
    unittest.main()
