import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from verify_scaling_native_step import bitwise_equal as replay_equal
from verify_scaling_raw import bitwise_equal as raw_equal


def test_byte_equality_distinguishes_numerically_equal_representations():
    positive = np.array([0.0], dtype=np.float64)
    negative = np.array([-0.0], dtype=np.float64)
    assert np.array_equal(positive, negative)
    first_nan = np.array([0x7ff8000000000000], dtype=np.uint64).view(np.float64)
    second_nan = np.array([0x7ff8000000000001], dtype=np.uint64).view(np.float64)
    source = np.arange(12, dtype=np.float64).reshape(3, 4).T
    for compare in (replay_equal, raw_equal):
        assert compare(positive, positive.copy())
        assert not compare(positive, negative)
        assert not compare(positive, positive.astype(np.float32))
        assert not compare(positive, positive.reshape(1, 1))
        assert compare(first_nan, first_nan.copy())
        assert not compare(first_nan, second_nan)
        assert compare(source, np.ascontiguousarray(source))
