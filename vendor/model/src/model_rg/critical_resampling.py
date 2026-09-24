"""Whole-realization uncertainty for conditional vector fluctuations."""
import numpy as np


def replica_distances(values):
    """Squared distances from direct differences, in per-coordinate units.

    Axis zero indexes complete independent realizations. No subtraction of
    Gram entries is used, so identical represented vectors have exact zero
    distance even in the presence of large shared coordinate offsets.
    """
    x = np.asarray(values, dtype=np.float64)
    if x.ndim < 2 or x.shape[0] < 2 or not np.isfinite(x).all():
        raise ValueError('Expected at least two finite vector realizations')
    x = x.reshape(x.shape[0], -1)
    if not x.shape[1]:
        raise ValueError('Empty observation vector')
    distances = np.zeros((len(x), len(x)))
    with np.errstate(over='ignore', invalid='ignore'):
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                delta = x[i]-x[j]
                distances[i,j] = distances[j,i] = np.mean(delta*delta)
    if not np.isfinite(distances).all():
        raise ValueError('Nonfinite pair distance')
    return distances


def replica_counts(seeds, draws=10000, seed=9152592):
    if seeds < 2 or draws < 1:
        raise ValueError('Invalid ensemble or resampling size')
    return np.random.default_rng(seed).multinomial(seeds, np.full(seeds, 1/seeds), size=draws)


def resampled_variance(distances, counts):
    """Unbiased count-weighted variance via nonnegative unordered pairs.

    Reuse the same count rows across paired controls, sizes, times and fields.
    The distances already contain the declared coordinate normalization.
    """
    distances = np.asarray(distances, dtype=float)
    counts = np.asarray(counts)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError('Expected a square distance matrix')
    n = len(distances)
    if n < 2 or counts.ndim != 2 or counts.shape[1] != n or counts.dtype.kind not in 'iu':
        raise ValueError('Expected integer replica counts')
    if (not np.isfinite(distances).all() or np.any(distances < 0)
            or not np.array_equal(distances, distances.T) or np.any(np.diag(distances) != 0)
            or np.any(counts < 0) or np.any(counts > n) or not np.all(counts.sum(1) == n)):
        raise ValueError('Invalid pair distances or resampling counts')
    result = np.zeros(len(counts))
    weights = counts.astype(np.float64)
    for i in range(n):
        for j in range(i+1, n):
            result += weights[:,i]*weights[:,j]*distances[i,j]
    result /= n*(n-1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite resampled variance')
    return result


def exact_replica_counts(seeds):
    """All empirical bootstrap count vectors and integer ordered-draw masses."""
    import math
    if type(seeds) is not int or not 2 <= seeds <= 8:
        raise ValueError('Exact enumeration supports two through eight seeds')
    def compositions(total, length):
        if length == 1:
            yield (total,)
        else:
            for first in range(total+1):
                for tail in compositions(total-first, length-1):
                    yield (first,)+tail
    counts = np.asarray(list(compositions(seeds, seeds)), dtype=np.int64)
    multiplicities = np.array([math.factorial(seeds)//math.prod(math.factorial(int(v)) for v in row)
                               for row in counts], dtype=np.int64)
    return counts, multiplicities


def weighted_inverse_cdf(values, multiplicities, probabilities=(.025,.5,.975)):
    values, multiplicities = np.asarray(values), np.asarray(multiplicities)
    valid = np.isfinite(values)
    if not np.any(valid):
        return [None]*len(probabilities)
    order = np.argsort(values[valid], kind='stable')
    v, w = values[valid][order], multiplicities[valid][order]
    cdf = np.cumsum(w)/w.sum()
    return v[np.searchsorted(cdf, probabilities, side='left')].tolist()


def vector_statistics(values):
    """Per-coordinate trace statistics for seed/context/layer/head/vector data."""
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 5 or x.shape[0] < 2 or min(x.shape[1:]) < 1 or not np.isfinite(x).all():
        raise ValueError('Expected finite seed/context/layer/head/vector values')
    s, _, _, n, d = x.shape
    centered = x-x.mean(0, keepdims=True)
    q = x.mean(3)
    dq = q-q.mean(0, keepdims=True)
    variance = np.mean(dq*dq)*s/(s-1)
    single = np.mean(centered*centered)*s/(s-1)
    chi = n*variance
    # The ensemble mean direction is a descriptive orthogonal decomposition,
    # not an independently selected response direction.
    reference = q.mean(0)
    norm = np.linalg.norm(reference, axis=-1, keepdims=True)
    unit = np.divide(reference, norm, out=np.zeros_like(reference), where=norm > 0)
    parallel = n*np.mean(np.sum(dq*unit, axis=-1)**2)*s/((s-1)*d)
    perpendicular = chi-parallel
    if perpendicular < -1e-12*(1+chi):
        raise ValueError('Orthogonal decomposition lost positivity')
    return dict(vector_dimension=d, susceptibility=float(chi), intensive_variance=float(variance),
                one_head_variance=float(single), cross_head_covariance=float((chi-single)/(n-1)) if n > 1 else 0.,
                enhancement=float(chi/single) if single > 0 else None,
                mean_square_amplitude=float(np.mean(x*x)),
                mean_by_seed=np.sqrt(np.mean(q*q, axis=(1, 2, 3))).tolist(),
                parallel_susceptibility=float(parallel), perpendicular_susceptibility=float(max(perpendicular, 0.)),
                covariance_identity_residual=float(chi-single-(n-1)*((chi-single)/(n-1))) if n > 1 else 0.,
                units='Context/layer-averaged covariance trace divided by vector dimension; fixed native component units.')


def binary_variance_decomposition(values, labels, scale):
    """Exact within/between decomposition with the same unbiased divisor.

    Labels describe each entire realization. They do not turn its individual
    contexts into independent phase assignments or experimental replicas.
    """
    x = np.asarray(values, dtype=float)
    labels = np.asarray(labels)
    if x.ndim < 2 or len(x) < 2 or labels.shape != (len(x),) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected vector realizations and one binary label each')
    if not np.isfinite(x).all() or not np.isfinite(scale) or scale < 0:
        raise ValueError('Invalid fluctuation observation')
    x = x.reshape(len(x), -1)
    mean = x.mean(0)
    total = scale*np.mean(np.sum((x-mean)**2, axis=0))/(len(x)-1)
    within = between = 0.
    counts = []
    for label in [0, 1]:
        group = x[labels == label]
        counts.append(len(group))
        if not len(group):
            continue
        group_mean = group.mean(0)
        within += scale*np.mean(np.sum((group-group_mean)**2, axis=0))/(len(x)-1)
        between += scale*len(group)*np.mean((group_mean-mean)**2)/(len(x)-1)
    return dict(susceptibility=float(total), within_susceptibility=float(within),
                between_susceptibility=float(between), label_counts=counts,
                between_fraction=float(between/total) if total > 0 else None,
                decomposition_residual=float(total-within-between))
