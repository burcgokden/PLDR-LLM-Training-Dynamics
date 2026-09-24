"""A maximal sequence coupling that also controls source-position edits."""
import numpy as np


def couple_resource(rng, remaining, removed, length):
    remaining = np.asarray(remaining)
    removed = set(int(x) for x in removed)
    if not removed.issubset(set(int(x) for x in remaining)):
        raise ValueError('Removed indices must belong to the parent resource')
    later = remaining[np.array([int(x) not in removed for x in remaining])]
    if length > len(later):
        raise ValueError('The successor resource cannot support the sequence')
    return draw_pair(rng, remaining, later, removed, length)


def draw_pair(rng, remaining, later, removed, length):
    """Keep surviving slots; refill removed slots uniformly without replacement."""
    x = rng.choice(remaining, length, replace=False)
    hit = np.fromiter((int(v) in removed for v in x), dtype=bool, count=length)
    y = x.copy()
    occupied = set(int(v) for v in x[~hit])
    for slot in np.flatnonzero(hit):
        while True:
            value = int(later[rng.integers(len(later))])
            if value not in occupied:
                y[slot] = value
                occupied.add(value)
                break
    return x, y


def edit_moments(population, removed, length):
    r, k, ell = population, removed, length
    if not (r > 1 and 0 <= k <= r and 0 <= ell <= r-k):
        raise ValueError('Invalid finite-resource sizes')
    mean = ell*k/r
    variance = ell*(k/r)*(1-k/r)*(r-ell)/(r-1)
    return mean, variance
