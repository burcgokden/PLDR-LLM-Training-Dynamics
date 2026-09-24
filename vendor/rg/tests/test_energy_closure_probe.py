import unittest

import numpy as np

import copy

try:
    import torch
except ImportError:
    torch = None

from scripts.analyze_energy_closure_probe import closure_defect, object_fingerprint


class EnergyClosureDefectTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None, "PyTorch is unavailable")
    def test_checkpoint_fingerprint_omits_only_first_moment(self) -> None:
        baseline = {
            "model": {"weight": torch.tensor([1.0, 2.0])},
            "optimizer": {
                "state": {
                    0: {
                        "step": torch.tensor(16.0),
                        "exp_avg": torch.tensor([0.2, -0.3]),
                        "exp_avg_sq": torch.tensor([0.4, 0.5]),
                    }
                }
            },
            "rng": (1, 2, 3),
        }
        intervention = copy.deepcopy(baseline)
        intervention["optimizer"]["state"][0]["exp_avg"].zero_()
        self.assertNotEqual(
            object_fingerprint(baseline), object_fingerprint(intervention)
        )
        self.assertEqual(
            object_fingerprint(baseline, omit_first_moment=True),
            object_fingerprint(intervention, omit_first_moment=True),
        )
        intervention["optimizer"]["state"][0]["exp_avg_sq"].add_(1.0)
        self.assertNotEqual(
            object_fingerprint(baseline, omit_first_moment=True),
            object_fingerprint(intervention, omit_first_moment=True),
        )

    def test_zero_defect_and_changed_map_census(self):
        same = closure_defect(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])
        )
        self.assertEqual(same["bitwise_different_map_count"], 0)
        self.assertEqual(same["maximum_absolute_energy_defect"], 0.0)
        changed = closure_defect(
            np.array([0.0, 1.0, 2.0]), np.array([0.0, 3.0, 1.0])
        )
        self.assertEqual(changed["bitwise_different_map_count"], 2)
        self.assertEqual(changed["maximum_absolute_energy_defect"], 2.0)
        self.assertAlmostEqual(
            changed["maximum_symmetric_relative_energy_defect"], 1.0
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "equal one-dimensional"):
            closure_defect(np.zeros(2), np.zeros(3))


if __name__ == "__main__":
    unittest.main()
