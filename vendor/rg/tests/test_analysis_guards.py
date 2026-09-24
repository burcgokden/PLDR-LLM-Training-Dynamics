import unittest

from row_rgmap.analysis import relative_weakening
from scripts.analyze_retained_summary import _format_optional_percent


class RetainedAnalysisGuardTests(unittest.TestCase):
    def test_relative_weakening_preserves_regular_case(self):
        self.assertAlmostEqual(relative_weakening(-2.0, -1.0), 0.5)

    def test_zero_reference_is_not_applicable(self):
        self.assertIsNone(relative_weakening(0.0, 0.0))
        self.assertEqual(_format_optional_percent(None), r"\text{not applicable}")


if __name__ == "__main__":
    unittest.main()
