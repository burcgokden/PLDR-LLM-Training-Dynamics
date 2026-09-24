from __future__ import annotations

import copy
import hashlib
import math
import os
import subprocess
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch

import torch

from row_rgmap.analysis import seal_record
from row_rgmap.provenance_v3 import git_blob_descriptor
from row_rgmap.staged_validation import (
    RELOCATION_SCHEMA,
    TOKEN_SELECTION_RULE,
    staged_source_paths_for_commit,
    derive_scientific_payload,
    lexical_file_census,
    resolve_descriptor,
    validate_config_shape,
    validate_embedding_intervention,
    validate_staged_source_bindings,
    validate_resource_accounting,
    validate_scientific_payload,
    validate_staged_relocation,
)

ROOT = Path(__file__).resolve().parents[1]


def staged_config() -> dict:
    return {
        "schema_version": "pldr-row-rg-staged-closure-config-v1",
        "stage_id": "synthetic-stage",
        "source_run": {"anchor": "data_root", "path": "source"},
        "predecessor_closure": {"anchor": "data_root", "path": "predecessor"},
        "source_step": 10,
        "block_sizes": [3, 6],
        "semigroup_split": 3,
        "gpu_hour_cap": 1.0,
        "embedding_shift": {
            "model_key": "decoder.embedding.weight",
            "coordinate": 0,
            "offset": 0.125,
            "token_selection_rule": TOKEN_SELECTION_RULE,
        },
        "branch_ids": [
            "baseline-direct",
            "gauge-consistent",
            "gauge-params-only",
            "probe-null-embedding-shift",
        ],
        "candidate_states": [
            {
                "candidate_id": "candidate",
                "challenge_branch": "gauge-params-only",
                "description": "synthetic candidate",
            }
        ],
        "splits": {"design": ["T0"], "holdout": ["T1"]},
        "trajectories": [
            {"trajectory_id": "T0", "device": 0},
            {"trajectory_id": "T1", "device": 1},
        ],
    }


def defect(scale: int, different: int, *, conditional: bool = False) -> dict:
    row = {
        "block_size": scale,
        "map_count": 2,
        "bitwise_different_map_count": different,
        "maximum_absolute_energy_defect": 0.0 if different == 0 else 0.25,
        "maximum_symmetric_relative_energy_defect": 0.0 if different == 0 else 0.5,
    }
    if conditional:
        row["conditional_equality_prediction_rejected"] = bool(different)
    return row


def scientific_payload(outcome: str) -> tuple[dict, dict]:
    config = staged_config()
    source_exact = outcome != "NOT_EVALUABLE"
    different = 1 if outcome == "REJECTED" else 0
    candidate_rows = []
    baseline_rows = []
    gauge_rows = []
    semigroup_rows = []
    for trajectory_id, split in (("T0", "design"), ("T1", "holdout")):
        scales = []
        for scale in config["block_sizes"]:
            row = defect(scale, different if scale == 3 else 0, conditional=True)
            row["conditional_equality_prediction_rejected"] = bool(
                source_exact and row["bitwise_different_map_count"]
            )
            scales.append(row)
        candidate_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "challenge_branch": "gauge-params-only",
                "source_candidate_bitwise_equal": source_exact,
                "scales": scales,
            }
        )
        baseline_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "all_steps_bitwise_equal": True,
                "state_count": 7,
                "map_count": 12,
            }
        )
        gauge_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "source_canonical_transition_state_equal": True,
                "successor_scales": [
                    defect(scale, 0) for scale in config["block_sizes"]
                ],
                "final_canonical_transition_state_equal": True,
            }
        )
        semigroup_rows.append(
            {
                "trajectory_id": trajectory_id,
                "split": split,
                "restart_step": 13,
                "tail_energy_arrays_bitwise_equal": True,
                "final_canonical_transition_state_equal": True,
            }
        )
    payload = derive_scientific_payload(
        config,
        {"candidate": candidate_rows},
        baseline_rows,
        gauge_rows,
        semigroup_rows,
    )
    return config, payload


def resource_fixture() -> tuple[dict, dict, dict]:
    route_a = {
        "wall_seconds": 2.0,
        "timing": {"total_route_elapsed_seconds": 2.0},
    }
    route_b = {
        "wall_seconds": 1.0,
        "timing": {"total_route_elapsed_seconds": 1.0},
    }
    accounting = {
        "source_remeasurement_cuda_wall_seconds": 3.0,
        "continuation_producer_reported_wall_seconds": 9.0,
        "cap_gpu_hours": 1.0,
        "cap_passed": True,
    }
    manifest = {
        "source_remeasurement": {
            "cuda_wall_seconds": 3.0,
            "trajectories": [
                {
                    "original_route": route_a,
                    "branches": [{"route": route_b}],
                }
            ],
        },
        "branches": {"T0": [{"wall_seconds": 4.0}]},
        "semigroup_branches": {"T0": [{"wall_seconds": 5.0}]},
        "gpu_time_accounting": accounting,
        "gpu_seconds": 12.0,
        "gpu_hours": 12.0 / 3600.0,
    }
    record = {
        "gpu_time_accounting": copy.deepcopy(accounting),
        "gpu_seconds": 12.0,
        "gpu_hours": 12.0 / 3600.0,
    }
    config = {"gpu_hour_cap": 1.0}
    return manifest, config, record


class OutcomeNeutralDecisionTests(unittest.TestCase):
    def test_all_three_candidate_outcomes_validate(self) -> None:
        expected_stage = {
            "REJECTED": "COMPLETED_STOPPED_AT_CLOSURE",
            "NOT_REJECTED": "PROMOTED_TO_LOCAL_FLOW",
            "NOT_EVALUABLE": "COMPLETED_STOPPED_AT_CLOSURE",
        }
        for outcome, stage in expected_stage.items():
            with self.subTest(outcome=outcome):
                config, payload = scientific_payload(outcome)
                validated = validate_scientific_payload(payload, config)
                self.assertEqual(
                    validated["candidate_results"][0]["status"], outcome
                )
                self.assertEqual(validated["promotion"]["stage_status"], stage)

    def test_mixed_split_does_not_pass_promotion_gate(self) -> None:
        config, payload = scientific_payload("REJECTED")
        candidate_rows = copy.deepcopy(
            payload["candidate_results"][0]["trajectories"]
        )
        holdout = candidate_rows[1]
        for row in holdout["scales"]:
            row["bitwise_different_map_count"] = 0
            row["maximum_absolute_energy_defect"] = 0.0
            row["maximum_symmetric_relative_energy_defect"] = 0.0
            row["conditional_equality_prediction_rejected"] = False
        controls = payload["controls"]
        mixed = derive_scientific_payload(
            config,
            {"candidate": candidate_rows},
            controls["baseline_replay"]["trajectories"],
            controls["complete_state_gauge_quotient"]["trajectories"],
            controls["direct_vs_iterated_semigroup"]["trajectories"],
        )
        candidate = mixed["candidate_results"][0]
        self.assertEqual(candidate["splits"]["design"]["status"], "REJECTED")
        self.assertEqual(candidate["splits"]["holdout"]["status"], "NOT_REJECTED")
        self.assertEqual(candidate["status"], "NOT_REJECTED")
        self.assertFalse(candidate["promotion_gate_passed"])
        self.assertEqual(
            mixed["promotion"]["stage_status"],
            "COMPLETED_STOPPED_AT_CLOSURE",
        )
        self.assertIn("promotion gate", mixed["promotion"]["stop_reason"])
        validate_scientific_payload(mixed, config)

    def test_derived_aggregate_and_type_mutations_fail(self) -> None:
        config, payload = scientific_payload("REJECTED")
        mutations = []

        value = copy.deepcopy(payload)
        value["candidate_results"][0]["status"] = "NOT_REJECTED"
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["candidate_results"][0]["splits"]["design"]["rejected_scales"] = []
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["promotion"]["promoted_candidate_ids"] = ["candidate"]
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["controls"]["baseline_replay"]["passed"] = 1
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["conclusion"]["scope"] = "altered"
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["candidate_results"][0]["trajectories"][0]["scales"][0][
            "conditional_equality_prediction_rejected"
        ] = False
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["promotion"]["local_reduced_flow_authorized"] = True
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["promotion"]["critical_surface_authorized"] = True
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["promotion"]["stage_status"] = "PROMOTED_TO_LOCAL_FLOW"
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["promotion"]["stop_reason"] = None
        mutations.append(value)

        value = copy.deepcopy(payload)
        value["unknown"] = "field"
        mutations.append(value)

        for value in mutations:
            with self.subTest(mutation=value):
                with self.assertRaises(ValueError):
                    validate_scientific_payload(value, config)

    def test_config_registry_and_numeric_mutations_fail(self) -> None:
        config = staged_config()
        mutations = []

        value = copy.deepcopy(config)
        value["unknown"] = 1
        mutations.append(value)

        value = copy.deepcopy(config)
        value["branch_ids"].append("unregistered-branch")
        mutations.append(value)

        value = copy.deepcopy(config)
        value["semigroup_split"] = 4
        mutations.append(value)

        value = copy.deepcopy(config)
        value["candidate_states"].append(copy.deepcopy(value["candidate_states"][0]))
        mutations.append(value)

        value = copy.deepcopy(config)
        value["trajectories"][1]["trajectory_id"] = "T0"
        mutations.append(value)

        value = copy.deepcopy(config)
        value["splits"]["holdout"] = ["T0"]
        mutations.append(value)

        value = copy.deepcopy(config)
        value["embedding_shift"]["token_selection_rule"] = "largest eligible token"
        mutations.append(value)

        for offset in (None, True, math.inf, math.nan, 0.0):
            value = copy.deepcopy(config)
            value["embedding_shift"]["offset"] = offset
            mutations.append(value)

        for value in mutations:
            with self.subTest(mutation=value):
                with self.assertRaises(ValueError):
                    validate_config_shape(value)


class ResourceAccountingTests(unittest.TestCase):
    def test_every_gpu_leaf_is_summed(self) -> None:
        manifest, config, record = resource_fixture()
        result = validate_resource_accounting(manifest, config, record)
        self.assertEqual(result["gpu_seconds"], 12.0)
        self.assertTrue(result["cap_passed"])

    def test_leaf_total_cap_and_record_mutations_fail(self) -> None:
        manifest, config, record = resource_fixture()
        cases = []

        value = copy.deepcopy(manifest)
        value["source_remeasurement"]["trajectories"][0]["original_route"][
            "timing"
        ]["total_route_elapsed_seconds"] = 1.5
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["source_remeasurement"]["trajectories"][0]["branches"][0][
            "route"
        ]["wall_seconds"] = -1.0
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["branches"]["T0"][0]["wall_seconds"] = True
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["semigroup_branches"]["T0"][0]["wall_seconds"] = math.nan
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["source_remeasurement"]["cuda_wall_seconds"] = 4.0
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["gpu_time_accounting"][
            "continuation_producer_reported_wall_seconds"
        ] = 8.0
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["gpu_time_accounting"]["cap_passed"] = False
        cases.append((value, config, record))

        value = copy.deepcopy(manifest)
        value["gpu_time_accounting"]["cap_gpu_hours"] = True
        cases.append((value, config, record))

        changed_record = copy.deepcopy(record)
        changed_record["gpu_seconds"] = 13.0
        cases.append((manifest, config, changed_record))

        changed_record = copy.deepcopy(record)
        changed_record["gpu_time_accounting"]["cap_passed"] = 1
        cases.append((manifest, config, changed_record))

        for changed_manifest, changed_config, changed_record in cases:
            with self.subTest(manifest=changed_manifest, record=changed_record):
                with self.assertRaises(ValueError):
                    validate_resource_accounting(
                        changed_manifest, changed_config, changed_record
                    )


class SourceRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        paths = staged_source_paths_for_commit(ROOT, self.commit)
        sources = []
        for repository_path in paths:
            digest, size = git_blob_descriptor(
                ROOT, self.commit, repository_path
            )
            sources.append(
                {
                    "repository_path": repository_path,
                    "code_commit": self.commit,
                    "sha256": digest,
                    "size_bytes": size,
                }
            )
        self.manifest = {
            "code_commit": self.commit,
            "analysis_sources": sources,
        }

    def test_exact_commit_source_registry_validates(self) -> None:
        self.assertTrue(
            validate_staged_source_bindings(self.manifest, ROOT)
        )

    def test_missing_reordered_extra_and_digest_mutations_fail(self) -> None:
        cases = []
        missing = copy.deepcopy(self.manifest)
        missing["analysis_sources"].pop()
        cases.append(missing)

        reordered = copy.deepcopy(self.manifest)
        first = reordered["analysis_sources"].pop(0)
        reordered["analysis_sources"].insert(1, first)
        cases.append(reordered)

        extra = copy.deepcopy(self.manifest)
        extra["analysis_sources"].append(
            copy.deepcopy(extra["analysis_sources"][0])
        )
        cases.append(extra)

        altered = copy.deepcopy(self.manifest)
        altered["analysis_sources"][0]["sha256"] = "0" * 64
        cases.append(altered)

        for value in cases:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_staged_source_bindings(value, ROOT)
class EmbeddingInterventionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.specification = {
            "model_key": "decoder.embedding.weight",
            "coordinate": 0,
            "offset": 0.125,
        }
        self.selection = {
            "selected_token_id": 1,
            "next_input_occurrences": 2,
            "absent_from_probe_inputs": True,
        }
        self.source = {
            "code_commit": "old",
            "protocol_sha256": "old-protocol",
            "model": {
                "decoder.embedding.weight": torch.zeros((3, 2), dtype=torch.float32),
                "unchanged": torch.ones(1),
            },
            "optimizer": {"step": 10},
        }
        self.rewritten = copy.deepcopy(self.source)
        self.rewritten["code_commit"] = "new"
        self.rewritten["protocol_sha256"] = "new-protocol"
        self.rewritten["model"]["decoder.embedding.weight"][1, 0] += 0.125
        before = self.source["model"]["decoder.embedding.weight"][1, 0]
        after = self.rewritten["model"]["decoder.embedding.weight"][1, 0]
        self.serialized = {
            "operation": "source-null-embedding-shift",
            "role": "gauge-canonical-optimizer-candidate-closure-test",
            "model_key": "decoder.embedding.weight",
            "token_id": 1,
            "coordinate": 0,
            "offset": 0.125,
            "before_hex": before.numpy().tobytes().hex(),
            "after_hex": after.numpy().tobytes().hex(),
            "selection": self.selection,
        }

    def test_exact_single_coordinate_edit_validates(self) -> None:
        self.assertTrue(
            validate_embedding_intervention(
                self.source,
                self.rewritten,
                self.specification,
                self.selection,
                self.serialized,
            )
        )

    def test_intervention_and_selection_metadata_mutations_fail(self) -> None:
        cases = []

        specification = copy.deepcopy(self.specification)
        specification["offset"] = 0.25
        cases.append(
            (self.rewritten, specification, self.selection, self.serialized)
        )

        specification = copy.deepcopy(self.specification)
        specification["model_key"] = "missing.embedding.weight"
        serialized = copy.deepcopy(self.serialized)
        serialized["model_key"] = specification["model_key"]
        cases.append(
            (self.rewritten, specification, self.selection, serialized)
        )

        specification = copy.deepcopy(self.specification)
        specification["coordinate"] = 1
        serialized = copy.deepcopy(self.serialized)
        serialized["coordinate"] = 1
        cases.append(
            (self.rewritten, specification, self.selection, serialized)
        )

        rewritten = copy.deepcopy(self.rewritten)
        rewritten["model"]["decoder.embedding.weight"][2, 1] += 0.125
        cases.append(
            (rewritten, self.specification, self.selection, self.serialized)
        )

        serialized = copy.deepcopy(self.serialized)
        serialized["after_hex"] = "00" * (
            len(serialized["after_hex"]) // 2
        )
        cases.append(
            (self.rewritten, self.specification, self.selection, serialized)
        )


        serialized = copy.deepcopy(self.serialized)
        serialized["selection"]["next_input_occurrences"] = 3
        cases.append(
            (self.rewritten, self.specification, self.selection, serialized)
        )

        serialized = copy.deepcopy(self.serialized)
        serialized["selection"]["next_batch_byte_token_sha256"] = "0" * 64
        cases.append(
            (self.rewritten, self.specification, self.selection, serialized)
        )

        rewritten = copy.deepcopy(self.rewritten)
        rewritten["code_commit"] = self.source["code_commit"]
        cases.append(
            (rewritten, self.specification, self.selection, self.serialized)
        )
        rewritten = copy.deepcopy(self.rewritten)
        rewritten["optimizer"]["step"] = 11
        cases.append(
            (rewritten, self.specification, self.selection, self.serialized)
        )

        selection = copy.deepcopy(self.selection)
        selection["selected_token_id"] = 0
        serialized = copy.deepcopy(self.serialized)
        serialized["token_id"] = 0
        serialized["selection"] = selection
        cases.append((self.rewritten, self.specification, selection, serialized))

        for rewritten, specification, selection, serialized in cases:
            with self.subTest(
                specification=specification,
                selection=selection,
                serialized=serialized,
            ):
                with self.assertRaises(ValueError):
                    validate_embedding_intervention(
                        self.source,
                        rewritten,
                        specification,
                        selection,
                        serialized,
                    )


class LexicalRelocationTests(unittest.TestCase):
    @staticmethod
    def make_tree(root: Path) -> set[PurePosixPath]:
        (root / "nested").mkdir()
        (root / "a.txt").write_text("hello", encoding="utf-8")
        (root / "nested" / "b.bin").write_bytes(b"xy")
        return {PurePosixPath("a.txt"), PurePosixPath("nested/b.bin")}

    def test_exact_tree_and_same_size_byte_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_tree(root)
            census = lexical_file_census(root, expected)
            self.assertEqual(census["file_count"], 2)
            descriptor = {
                "path": "a.txt",
                "size_bytes": 5,
                "sha256": hashlib.sha256(b"hello").hexdigest(),
            }
            self.assertEqual(resolve_descriptor(root, descriptor, "fixture"), root / "a.txt")
            escaped = copy.deepcopy(descriptor)
            escaped["path"] = "../escape"
            with self.assertRaises(ValueError):
                resolve_descriptor(root, escaped, "fixture")
            (root / "a.txt").write_text("HELLO", encoding="utf-8")
            with self.assertRaises(ValueError):
                resolve_descriptor(root, descriptor, "fixture")

    def test_extra_missing_nonregular_and_all_symlink_forms_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_tree(root)
            (root / "extra").write_text("x", encoding="utf-8")
            with self.assertRaises(ValueError):
                lexical_file_census(root, expected)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_tree(root)
            (root / "a.txt").unlink()
            with self.assertRaises(ValueError):
                lexical_file_census(root, expected)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = self.make_tree(root)
            os.mkfifo(root / "fifo")
            with self.assertRaises(ValueError):
                lexical_file_census(root, expected)

        targets = ("a.txt", "missing", "/etc/passwd", "nested")
        for target in targets:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                expected = self.make_tree(root)
                link = root / "link"
                os.symlink(target, link)
                with self.assertRaises(ValueError):
                    lexical_file_census(root, expected)

    def test_relocation_record_mutations_fail(self) -> None:
        census = {"file_count": 2, "total_size_bytes": 7, "exact": True}
        manifest = {
            "code_commit": "abc",
            "record_sha256": "manifest-digest",
            "inputs": {},
            "branches": {},
            "semigroup_branches": {},
        }
        record = {"record_sha256": "result-digest"}
        relocation = seal_record(
            {
                "schema_version": RELOCATION_SCHEMA,
                "completed_at_utc": "2000-01-01T00:00:00+00:00",
                "code_commit": "abc",
                "input_run_manifest_record_sha256": "manifest-digest",
                "input_analysis_record_sha256": "result-digest",
                "source_census": census,
                "relocated_census": census,
                "scientific_payload_equal": True,
                "temporary_copy_removed": True,
            }
        )
        with patch(
            "row_rgmap.staged_validation.lexical_file_census",
            return_value=census,
        ):
            self.assertEqual(
                validate_staged_relocation(
                    relocation, record, manifest, Path("/unused")
                ),
                census,
            )
            mutations = []
            value = copy.deepcopy(relocation)
            value["unexpected"] = True
            mutations.append(seal_record({k: v for k, v in value.items() if k != "record_sha256"}))
            value = copy.deepcopy(relocation)
            value["scientific_payload_equal"] = False
            mutations.append(seal_record({k: v for k, v in value.items() if k != "record_sha256"}))
            value = copy.deepcopy(relocation)
            value["source_census"]["total_size_bytes"] = 8
            mutations.append(seal_record({k: v for k, v in value.items() if k != "record_sha256"}))
            value = copy.deepcopy(relocation)
            value["relocated_census"]["file_count"] = 3
            mutations.append(
                seal_record(
                    {k: v for k, v in value.items() if k != "record_sha256"}
                )
            )
            for value in mutations:
                with self.subTest(value=value):
                    with self.assertRaises(ValueError):
                        validate_staged_relocation(
                            value, record, manifest, Path("/unused")
                        )


if __name__ == "__main__":
    unittest.main()
