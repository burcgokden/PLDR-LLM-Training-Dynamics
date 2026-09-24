import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from scripts.analyze_energy_closure_probe_v2 import (
    resolve_data_root_locator as analyze_locator,
    vector_defect,
    verify_checkpoint_intervention,
)
from scripts.pldr_energy_producer_v3 import _short_byte_tokens
from scripts.run_energy_closure_probe_v2 import (
    BRANCH_SPECS,
    apply_branch_intervention,
    independent_probe_tokens,
    independent_row_centered_energy,
    resolve_data_root_locator as run_locator,
)
from scripts.verify_energy_closure_relocation import (
    scientific_payload as relocation_scientific_payload,
)


class FakeDataset:
    def __init__(self, rows):
        self.rows = [{"content": value} for value in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class PortableLocatorTests(unittest.TestCase):
    def test_relative_locator_resolves_in_both_consumers(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "executed" / "run"
            target.mkdir(parents=True)
            locator = {"anchor": "data_root", "path": "executed/run"}
            self.assertEqual(run_locator(root, locator), target)
            self.assertEqual(analyze_locator(root, locator), target)

    def test_absolute_parent_and_symlink_escapes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            container = Path(temporary)
            root = container / "root"
            root.mkdir()
            outside = container / "outside"
            outside.mkdir()
            (root / "link").symlink_to(outside, target_is_directory=True)
            invalid = (
                {"anchor": "data_root", "path": str(outside)},
                {"anchor": "data_root", "path": "../outside"},
                {"anchor": "data_root", "path": "link"},
            )
            for consumer in (run_locator, analyze_locator):
                for locator in invalid:
                    with self.subTest(consumer=consumer.__module__, locator=locator):
                        with self.assertRaisesRegex(ValueError, "relative|escape|traverse"):
                            consumer(root, locator)


class IndependentSourceObserverTests(unittest.TestCase):
    def test_probe_extraction_matches_the_producer_contract(self):
        dataset = FakeDataset(["ab", "çd", "efgh"])
        expected_values, expected_metadata = _short_byte_tokens(dataset, 0, 8)
        values, metadata = independent_probe_tokens(dataset, 0, 8)
        self.assertEqual(values, expected_values)
        self.assertEqual(metadata, expected_metadata)

    @unittest.skipUnless(torch is not None, "PyTorch is unavailable")
    def test_energy_reduction_matches_the_registered_contract(self):
        from row_rgmap.recorder import row_centered_energy

        matrix = torch.tensor(
            [[1.0, -2.0, 4.0], [3.0, 5.0, -1.0], [0.5, 8.0, 2.0]],
            dtype=torch.float32,
        )
        self.assertEqual(
            independent_row_centered_energy(matrix),
            row_centered_energy(matrix),
        )


@unittest.skipUnless(torch is not None, "PyTorch is unavailable")
class CheckpointInterventionTests(unittest.TestCase):
    def baseline(self):
        model = {
            (
                f"decoder.dec_layers.{index}.mha1.reslayerAs.1."
                "layernormA.bias"
            ): torch.tensor([0.5, -1.0], dtype=torch.float32)
            for index in range(3)
        }
        model["decoder.embedding.weight"] = torch.tensor(
            [[1.0, 2.0]], dtype=torch.float32
        )
        return {
            "model": model,
            "optimizer": {
                "state": {
                    0: {
                        "step": torch.tensor(16.0),
                        "exp_avg": torch.tensor([0.2, -0.3], dtype=torch.float32),
                        "exp_avg_sq": torch.tensor([0.4, 0.5], dtype=torch.float32),
                    }
                }
            },
            "rng": (1, 2, 3),
        }

    def test_every_registered_intervention_replays_independently(self):
        baseline = self.baseline()
        for branch in BRANCH_SPECS:
            with self.subTest(branch=branch):
                checkpoint = copy.deepcopy(baseline)
                apply_branch_intervention(checkpoint, branch)
                verify_checkpoint_intervention(baseline, checkpoint, branch)

    def test_undeclared_change_is_rejected(self):
        baseline = self.baseline()
        checkpoint = copy.deepcopy(baseline)
        apply_branch_intervention(checkpoint, "zero-first-moment")
        checkpoint["model"]["decoder.embedding.weight"].add_(1.0)
        with self.assertRaisesRegex(ValueError, "undeclared"):
            verify_checkpoint_intervention(
                baseline, checkpoint, "zero-first-moment"
            )


class BitwiseDecisionRuleTests(unittest.TestCase):
    def test_signed_zero_is_detected_by_exact_rule(self):
        defect = vector_defect(
            np.asarray([0.0], dtype=np.float64),
            np.asarray([-0.0], dtype=np.float64),
        )
        self.assertEqual(defect["maximum_absolute_energy_defect"], 0.0)
        self.assertEqual(defect["bitwise_different_map_count"], 1)


class RelocationComparisonTests(unittest.TestCase):
    def test_only_declared_provenance_fields_are_excluded(self):
        baseline = {
            "schema_version": "example",
            "analysis_sources": [{"sha256": "a" * 64}],
            "code_commit": "b" * 40,
            "completed_at_utc": "2000-01-01T00:00:00+00:00",
            "record_sha256": "c" * 64,
            "conclusion": {"closure_rejected": True},
        }
        relocated = copy.deepcopy(baseline)
        relocated["analysis_sources"] = [{"sha256": "d" * 64}]
        relocated["code_commit"] = "e" * 40
        relocated["completed_at_utc"] = "2000-01-01T00:01:00+00:00"
        relocated["record_sha256"] = "f" * 64
        self.assertEqual(
            relocation_scientific_payload(baseline),
            relocation_scientific_payload(relocated),
        )
        relocated["conclusion"]["closure_rejected"] = False
        self.assertNotEqual(
            relocation_scientific_payload(baseline),
            relocation_scientific_payload(relocated),
        )


if __name__ == "__main__":
    unittest.main()
