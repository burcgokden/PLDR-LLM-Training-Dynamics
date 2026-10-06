import builtins
import copy
from pathlib import Path
from unittest import mock
import unittest

from row_rgmap.analysis import seal_record
from scripts.run_stage_resolved_observer import (
    preflight_inputs,
    validate_predecessor_record,
    validate_producing_device_routes,
)
from scripts.verify_stage_resolved_observer import replay_exact_route_summary


class StageResolvedRendererTests(unittest.TestCase):




    def test_predecessor_schema_and_producing_route_preflight(self):
        import json

        predecessor_path = Path(
            "data:rg/executed/"
            "stage-resolved-observer-v7-20260903/result.json"
        )
        predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
        normalized = validate_predecessor_record(predecessor)
        self.assertEqual(normalized["summary"]["checkpoint_count"], 16)

        for key in sorted(normalized["summary"]):
            mutated = copy.deepcopy(predecessor)
            del mutated["summary"][key]
            mutated = seal_record(
                {name: value for name, value in mutated.items() if name != "record_sha256"}
            )
            with self.subTest(missing_summary_key=key), self.assertRaisesRegex(
                ValueError, "predecessor summary schema mismatch"
            ):
                validate_predecessor_record(mutated)

        incompatible = seal_record(
            {
                "schema_version": "pldr-row-rg-stage-resolved-observer-v2",
                "summary": {},
            }
        )
        with self.assertRaisesRegex(
            ValueError, "predecessor summary schema mismatch"
        ):
            validate_predecessor_record(incompatible)

        checkpoints = [{"producing_device": 0}, {"producing_device": 1}]
        validate_producing_device_routes(
            checkpoints, ["cpu", "cuda:0", "cuda:1"]
        )
        with self.assertRaisesRegex(ValueError, "producing-device route"):
            validate_producing_device_routes(checkpoints, ["cpu", "cuda:0"])


    def test_route_preflight_completes_before_any_torch_import(self):
        run_root = Path(
            "data:rg/executed/"
            "stage-a-runner-smoke-v7-20260903/run"
        )
        original_import = builtins.__import__
        torch_imports = []

        def guarded_import(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                torch_imports.append(name)
                raise AssertionError("torch imported during input preflight")
            return original_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=guarded_import):
            preflight = preflight_inputs(
                run_root, ["cpu", "cuda:0", "cuda:1"]
            )
            self.assertEqual(len(preflight["checkpoints"]), 16)
            with self.assertRaisesRegex(ValueError, "producing-device route"):
                preflight_inputs(run_root, ["cpu", "cuda:0"])
        self.assertEqual(torch_imports, [])

    def test_exact_route_verifier_summary_handles_empty_and_classified_rows(self):
        route = "float64_whole_network"
        empty = replay_exact_route_summary([], route)
        self.assertEqual(empty["status"], "NO_MATCHING_MAPS")
        self.assertEqual(empty["zero_count"], 0)
        self.assertIsNone(empty["minimum"])
        self.assertIsNone(empty["positive_minimum"])

        def row(classification: str, exact: float, rounded: float) -> dict:
            return {
                route: {
                    "S3out": {
                        "row_centered_energy": rounded,
                        "represented_energy": {
                            "classification": classification,
                            "exact_dyadic_energy": {"float": exact},
                        },
                    }
                }
            }

        replay = replay_exact_route_summary(
            [
                row("REPRESENTED_ZERO", 0.0, 3.8e-44),
                row("ONE_ULP_NEAR_COLLAPSE", 2.5e-48, 3.9e-44),
                row("POSITIVE_BEYOND_ONE_ULP", 4e-30, 4e-30),
            ],
            route,
        )
        self.assertEqual(replay["status"], "CLASSIFIED")
        self.assertEqual(replay["zero_count"], 1)
        self.assertEqual(replay["one_ulp_count"], 1)
        self.assertEqual(replay["positive_count"], 1)
        self.assertEqual(replay["minimum"], 0.0)
        self.assertEqual(replay["positive_minimum"], 2.5e-48)
        self.assertEqual(replay["rounded_minimum"], 3.8e-44)

if __name__ == "__main__":
    unittest.main()
