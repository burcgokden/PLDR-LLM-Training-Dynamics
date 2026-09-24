from __future__ import annotations

import copy
import json
from pathlib import Path
import unittest

import torch

from row_rgmap.analysis import seal_record
from scripts.run_staged_predictive_closure import (
    apply_source_null_embedding_shift,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[1]


class StagedPredictiveProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "configs" / "staged-predictive-closure-v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_frozen_config_passes(self) -> None:
        validate_config(self.config)

    def test_scale_split_candidate_and_intervention_mutations_fail(self) -> None:
        mutations = []
        scale = copy.deepcopy(self.config)
        scale["block_sizes"].pop()
        mutations.append(scale)
        split = copy.deepcopy(self.config)
        split["splits"]["holdout"][0] = split["splits"]["design"][0]
        mutations.append(split)
        candidate = copy.deepcopy(self.config)
        candidate["candidate_states"][2]["challenge_branch"] = "gauge-params-only"
        mutations.append(candidate)
        intervention = copy.deepcopy(self.config)
        intervention["embedding_shift"]["offset"] = 1.0
        mutations.append(intervention)
        for value in mutations:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_config(value)

    def test_embedding_shift_changes_exactly_one_coordinate(self) -> None:
        checkpoint = {
            "model": {"decoder.embedding.weight": torch.zeros((5, 3))}
        }
        selection = {
            "selected_token_id": 2,
            "next_input_occurrences": 1,
            "absent_from_probe_inputs": True,
        }
        specification = self.config["embedding_shift"]
        before = checkpoint["model"]["decoder.embedding.weight"].clone()
        details = apply_source_null_embedding_shift(
            checkpoint, selection, specification
        )
        difference = checkpoint["model"]["decoder.embedding.weight"] - before
        self.assertEqual(int(torch.count_nonzero(difference)), 1)
        self.assertEqual(float(difference[2, 0]), 0.125)
        self.assertEqual(details["token_id"], 2)



if __name__ == "__main__":
    unittest.main()
