import copy
import unittest

import numpy as np

from scripts.analyze_observer_regime import (
    content_record, excursion_summary, scientific_payload,
)


class ObserverRegimeRendererTests(unittest.TestCase):



    def test_scientific_compatibility_ignores_schema_status_annotations(self):
        baseline = {
            "observer_floor": 1e-12,
            "absolute_route_discrepancy_bound_at_audited_checkpoints": 2e-4,
            "route_bound_coverage": {"checkpoint_map_evaluations": 1},
            "represented_arithmetic": {
                "face_excursions": {"excursion_count": 2, "peak_maximum": 1e-13},
                "represented_zero_state_count": 3,
            },
            "cell_observer_bands_and_power": [],
            "power_summary": {
                "minimum_required_holdout_updates": 100,
                "maximum_required_holdout_updates": 200,
            },
            "regular_drift_heldout_prediction_stage_count": 0,
            "interpretation": {"fixed_floor_establishes_zero_limit": False},
        }
        annotated = copy.deepcopy(baseline)
        annotated["represented_arithmetic"]["face_excursions"][
            "status"
        ] = "OBSERVED_EXCURSIONS"
        annotated["power_summary"][
            "status"
        ] = "ESTIMATED_FROM_COMPLETE_POSITIVE_CELLS"
        self.assertEqual(
            scientific_payload(baseline), scientific_payload(annotated)
        )
        annotated["power_summary"]["maximum_required_holdout_updates"] = 201
        self.assertNotEqual(
            scientific_payload(baseline), scientific_payload(annotated)
        )

if __name__ == "__main__":
    unittest.main()
