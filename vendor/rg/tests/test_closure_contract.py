import unittest

from row_rgmap.closure_contract import (
    BRANCH_SPECS,
    branch_specification,
    decision_status,
    validate_serialized_branch_specification,
)
from scripts import analyze_energy_closure_probe_v3 as analyzer
from scripts import run_energy_closure_probe_v3 as runner


class ClosureContractTests(unittest.TestCase):
    def test_runner_and_analyzer_use_the_shared_registry(self):
        self.assertIs(runner.BRANCH_SPECS, BRANCH_SPECS)
        self.assertIs(analyzer.BRANCH_SPECS, BRANCH_SPECS)
        specification = branch_specification("sign-gauge-consistent")
        self.assertIn("co_transform_first_moment", specification)
        self.assertNotIn("co_transformed_first_moment", specification)
        specification.update(
            {
                "changed_tensor_count": 60,
                "changed_element_count": 296448,
                "model_keys": ["fixture"],
                "model_keys_sha256": "0" * 64,
            }
        )
        validate_serialized_branch_specification(
            "sign-gauge-consistent", specification
        )

    def test_branch_contract_rejects_wrong_or_legacy_key(self):
        wrong = branch_specification("sign-gauge-consistent")
        wrong["co_transform_first_moment"] = False
        with self.assertRaisesRegex(ValueError, "differs from registry"):
            validate_serialized_branch_specification(
                "sign-gauge-consistent", wrong
            )
        legacy = branch_specification("sign-gauge-consistent")
        legacy["co_transformed_first_moment"] = legacy.pop(
            "co_transform_first_moment"
        )
        with self.assertRaisesRegex(ValueError, "differs from registry"):
            validate_serialized_branch_specification(
                "sign-gauge-consistent", legacy
            )

    def test_serialized_and_measured_gauge_support_names_are_mapped(self):
        declared = {
            "changed_tensor_count": 20,
            "changed_element_count": 98816,
            "model_keys_sha256": "a" * 64,
            "model_keys": ["layer0"],
        }
        measured = {
            "tensor_count": 20,
            "element_count": 98816,
            "model_keys_sha256": "a" * 64,
        }
        self.assertTrue(
            analyzer.serialized_gauge_support_matches(
                declared, measured, ["layer0"]
            )
        )
        measured["tensor_count"] = 21
        self.assertFalse(
            analyzer.serialized_gauge_support_matches(
                declared, measured, ["layer0"]
            )
        )

    def test_decision_vocabulary_is_outcome_neutral(self):
        self.assertEqual(
            decision_status(eligible=True, rejected=True), "REJECTED"
        )
        self.assertEqual(
            decision_status(eligible=True, rejected=False), "NOT_REJECTED"
        )
        self.assertEqual(
            decision_status(eligible=False, rejected=True), "NOT_EVALUABLE"
        )


if __name__ == "__main__":
    unittest.main()
