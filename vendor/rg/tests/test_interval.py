import unittest

from row_rgmap.interval import (
    IntervalAffineEdge,
    NonnegativeInterval,
    block_interval_edges,
    certified_face_edge,
    certified_positive_gain_interval,
)


class NonnegativeIntervalTests(unittest.TestCase):
    def test_interval_affine_composition_is_associative_up_to_rounding(self):
        first = IntervalAffineEdge(
            NonnegativeInterval(0.4, 0.5),
            NonnegativeInterval(0.0, 0.1),
        )
        second = IntervalAffineEdge(
            NonnegativeInterval(1.1, 1.3),
            NonnegativeInterval(0.2, 0.4),
        )
        third = IntervalAffineEdge(
            NonnegativeInterval(0.7, 0.9),
            NonnegativeInterval(0.0, 0.2),
        )
        left = third.compose(second).compose(first)
        right = third.compose(second.compose(first))
        for left_value, right_value in (
            (left.gain.lower, right.gain.lower),
            (left.gain.upper, right.gain.upper),
            (left.source.lower, right.source.lower),
            (left.source.upper, right.source.upper),
        ):
            self.assertAlmostEqual(left_value, right_value, places=14)
        blocked = block_interval_edges([first, second, third])
        self.assertAlmostEqual(blocked.gain.lower, left.gain.lower, places=14)
        self.assertAlmostEqual(blocked.gain.upper, left.gain.upper, places=14)
        self.assertAlmostEqual(blocked.source.lower, left.source.lower, places=14)
        self.assertAlmostEqual(blocked.source.upper, left.source.upper, places=14)

    def test_exact_edges_and_actions_are_enclosed_after_blocking(self):
        first = IntervalAffineEdge(
            NonnegativeInterval(0.4, 0.6),
            NonnegativeInterval(0.1, 0.2),
        )
        second = IntervalAffineEdge(
            NonnegativeInterval(1.5, 2.0),
            NonnegativeInterval(0.0, 0.1),
        )
        blocked = second.compose(first)
        exact_gain = 1.8 * 0.5
        exact_source = 1.8 * 0.15 + 0.05
        self.assertTrue(blocked.contains(exact_gain, exact_source))
        energy = NonnegativeInterval(2.0, 3.0)
        image = blocked.act(energy)
        self.assertTrue(image.contains(exact_gain * 2.5 + exact_source))

    def test_certified_branch_constructors(self):
        positive = certified_positive_gain_interval(
            NonnegativeInterval(2.0, 2.5),
            NonnegativeInterval(1.0, 1.2),
        )
        self.assertTrue(positive.gain.contains(1.1 / 2.2))
        self.assertEqual(positive.source, NonnegativeInterval.singleton(0.0))
        face = certified_face_edge(
            NonnegativeInterval.singleton(0.0),
            NonnegativeInterval(0.2, 0.3),
        )
        self.assertTrue(face.contains(0.0, 0.25))

    def test_near_face_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "strictly positive"):
            certified_positive_gain_interval(
                NonnegativeInterval(0.0, 1e-12),
                NonnegativeInterval(0.0, 1e-10),
            )


if __name__ == "__main__":
    unittest.main()
