import unittest
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np

from row_rgmap.observer import (
    dyadic_sqrt_enclosure,
    enclosure_class,
    layernorm_row_energy_enclosure,
    represented_input_class,
)


class LayerNormObserverTests(unittest.TestCase):
    def test_dyadic_sqrt_encloses(self):
        value = Fraction(2, 1)
        lower, upper = dyadic_sqrt_enclosure(value, 64)
        self.assertLessEqual(lower * lower, value)
        self.assertGreaterEqual(upper * upper, value)
        self.assertLess(upper - lower, Fraction(1, 1 << 63))

    def test_identical_rows_are_certified_exact_face(self):
        values = np.asarray([[1.0, -2.0], [1.0, -2.0]], dtype=np.float32)
        interval = layernorm_row_energy_enclosure(
            values, np.ones(2, dtype=np.float32), 1e-6
        )
        self.assertEqual(interval, (Fraction(0), Fraction(0)))
        self.assertEqual(enclosure_class(interval), "CERTIFIED_EXACT_FACE")

    def test_nonconstant_rows_are_certified_positive(self):
        values = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        gain = np.asarray([0.25, -0.5], dtype=np.float32)
        lower, upper = layernorm_row_energy_enclosure(values, gain, 1e-6)
        # For these symmetric rows the radicals cancel after squaring, so the
        # ideal energy has this exact rational form. A NumPy float64 reference
        # may sit an ulp outside the much tighter rational enclosure.
        reference = Fraction(5, 32) / (
            Fraction(1, 4) + Fraction.from_float(1e-6)
        )
        self.assertLessEqual(lower, reference)
        self.assertGreaterEqual(upper, reference)
        self.assertEqual(enclosure_class((lower, upper)), "CERTIFIED_POSITIVE")


    def test_represented_zero_class_separates_inherited_zero_from_positive_input(self):
        self.assertEqual(
            represented_input_class((Fraction(0), Fraction(0)), input_rows_identical=True),
            "REPRESENTED_INPUT_ZERO_IMAGE",
        )
        self.assertEqual(
            represented_input_class((Fraction(1, 8), Fraction(1, 4)), input_rows_identical=False),
            "REPRESENTED_INPUT_POSITIVE",
        )
        self.assertEqual(
            represented_input_class((Fraction(0), Fraction(1, 4)), input_rows_identical=False),
            "REPRESENTED_INPUT_UNRESOLVED",
        )

    def test_zero_gain_annihilates_row_variation(self):
        values = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        interval = layernorm_row_energy_enclosure(
            values, np.zeros(2, dtype=np.float32), 1e-6
        )
        self.assertEqual(enclosure_class(interval), "CERTIFIED_EXACT_FACE")

    def test_negative_values_are_rounded_outward(self):
        values = np.asarray(
            [[0.0, 1.0, 2.0], [2.0, -1.0, 0.5], [1.0, 0.0, -2.0]],
            dtype=np.float32,
        )
        gain = np.asarray([-0.25, 0.5, -1.5], dtype=np.float32)
        lower, upper = layernorm_row_energy_enclosure(
            values, gain, 1e-6, precision_bits=96
        )
        with localcontext() as context:
            context.prec = 100
            rows = [
                [Decimal.from_float(float(value)) for value in row]
                for row in values
            ]
            gains = [Decimal.from_float(float(value)) for value in gain]
            eps = Decimal.from_float(1e-6)
            outputs = []
            for row in rows:
                mean = sum(row) / len(row)
                centered = [value - mean for value in row]
                variance = sum(value * value for value in centered) / len(row)
                root = (variance + eps).sqrt()
                outputs.append(
                    [value * gamma / root for value, gamma in zip(centered, gains)]
                )
            means = [sum(column) / len(outputs) for column in zip(*outputs)]
            reference = sum(
                (value - means[index]) ** 2
                for row in outputs
                for index, value in enumerate(row)
            )
            lower_decimal = Decimal(lower.numerator) / Decimal(lower.denominator)
            upper_decimal = Decimal(upper.numerator) / Decimal(upper.denominator)
        self.assertLessEqual(lower_decimal, reference)
        self.assertGreaterEqual(upper_decimal, reference)


if __name__ == "__main__":
    unittest.main()
