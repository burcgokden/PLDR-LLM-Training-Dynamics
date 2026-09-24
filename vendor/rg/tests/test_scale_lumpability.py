import unittest

import numpy as np


def kernel_power(kernel: np.ndarray, scale: int) -> np.ndarray:
    return np.linalg.matrix_power(kernel, scale)


def is_strongly_lumpable(
    kernel: np.ndarray, observable: np.ndarray
) -> bool:
    classes = sorted(set(int(value) for value in observable))
    for source_class in classes:
        states = np.flatnonzero(observable == source_class)
        reference = None
        for state in states:
            pushforward = np.asarray(
                [
                    kernel[state, observable == target_class].sum()
                    for target_class in classes
                ]
            )
            if reference is None:
                reference = pushforward
            elif not np.array_equal(reference, pushforward):
                return False
    return True


class ScaleIndexedLumpabilityTests(unittest.TestCase):
    def test_three_cycle_closes_at_scale_three_but_not_scale_one(self):
        kernel = np.asarray(
            [
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        )
        observable = np.asarray([0, 0, 1])
        self.assertFalse(is_strongly_lumpable(kernel, observable))
        self.assertTrue(
            is_strongly_lumpable(kernel_power(kernel, 3), observable)
        )


if __name__ == "__main__":
    unittest.main()

