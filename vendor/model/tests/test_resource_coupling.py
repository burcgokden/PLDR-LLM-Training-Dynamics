"""Exact finite probability checks for source edits, including target uniformity."""
from fractions import Fraction
from itertools import permutations
import numpy as np
from model_rg.resource_coupling import couple_resource, edit_moments


def test_resource_coupling_uniformity_and_hamming_moments():
    # Enumerate the entire random experiment, including every possible refill.
    population = tuple(range(5)); removed = {0, 1}; later = {2, 3, 4}
    law = {}; mean = Fraction(0); second = Fraction(0)
    for x in permutations(population, 2):
        slots = [i for i, v in enumerate(x) if v in removed]
        kept = set(x)-removed
        choices = list(permutations(later-kept, len(slots)))
        weight = Fraction(1, 20*len(choices))
        for replacement in choices:
            y = list(x)
            for i, v in zip(slots, replacement): y[i] = v
            law[tuple(y)] = law.get(tuple(y), Fraction(0))+weight
            edits = sum(a != b for a, b in zip(x, y))
            mean += weight*edits; second += weight*edits*edits
    assert set(law.values()) == {Fraction(1, 6)}
    assert mean == Fraction(4, 5) and second-mean*mean == Fraction(9, 25)
    np.testing.assert_allclose(edit_moments(5, 2, 2), [float(mean), float(second-mean*mean)])
    rng = np.random.default_rng(115201)
    for _ in range(32):
        x, y = couple_resource(rng, population, removed, 2)
        assert len(set(x)) == len(set(y)) == 2 and not (set(y)&removed)
        assert np.all(x[~np.isin(x, list(removed))] == y[~np.isin(x, list(removed))])
