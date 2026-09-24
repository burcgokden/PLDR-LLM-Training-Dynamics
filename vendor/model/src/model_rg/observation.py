"""Precision and whole-seed uncertainty in fixed collective observation units."""
import itertools
import numpy as np


def sample_susceptibility(q, heads):
    """q has seed as first axis; all remaining coordinates are averaged."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[0] < 2 or not np.isfinite(q).all():
        raise ValueError('At least two finite seed fields are required')
    return float(heads * np.var(q, axis=0, ddof=1).mean())


def observation_transfer(reference, observed, heads):
    x, y = np.asarray(reference, dtype=float), np.asarray(observed, dtype=float)
    if x.shape != y.shape:
        raise ValueError('Reference and observation must have identical pairing')
    chi = sample_susceptibility(x, heads)
    other = sample_susceptibility(y, heads)
    err = sample_susceptibility(y-x, heads)
    upper = heads*x.shape[0]/(x.shape[0]-1)*float(np.mean((y-x)**2))
    bound = float(2*np.sqrt(chi*err)+err)
    return dict(reference=chi, observed=other, signed_difference=other-chi,
                error_susceptibility=err, uncentered_error_bound=upper,
                absolute_bound=bound,
                relative_difference=(other-chi)/chi if chi else None,
                relative_bound=bound/chi if chi else None)


def paired_horizon(before, after, heads, seed_ids, resamples=20000, rng_seed=660001):
    x, y = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if x.shape != y.shape or x.shape[0] != len(seed_ids) or len(set(seed_ids)) != len(seed_ids):
        raise ValueError('Unique matched seed identities and field shapes are required')
    s = len(seed_ids)
    if s < 3:
        raise ValueError('Jackknife variance needs at least three seeds')
    # Remove a common coordinate origin for stable Gram computations.
    def gram(a):
        a = (a-a.mean(0)).reshape(s, -1)
        return a@a.T/a.shape[1]
    g = gram(y)-gram(x)
    if s == 4:
        indices = np.array(list(itertools.product(range(s), repeat=s)))
        method = 'all 4^4 equally likely ordered empirical resamples'
    else:
        indices = np.random.default_rng(rng_seed).integers(0, s, (resamples, s))
        method = 'Monte Carlo whole-seed paired empirical resampling'
    weights = np.eye(s, dtype=int)[indices].sum(1)
    samples = heads/(s-1)*(weights@np.diag(g)-np.einsum('bi,ij,bj->b',weights,g,weights)/s)
    leave = np.array([sample_susceptibility(np.delete(y,i,0),heads)-sample_susceptibility(np.delete(x,i,0),heads) for i in range(s)])
    delta = sample_susceptibility(y,heads)-sample_susceptibility(x,heads)
    result = dict(heads=heads, seeds=seed_ids, independent_seeds=s,
                  before=sample_susceptibility(x,heads), after=sample_susceptibility(y,heads), difference=delta,
                  jackknife_se=float(np.sqrt((s-1)/s*np.sum((leave-leave.mean())**2))),
                  percentiles=np.quantile(samples,[.025,.975]).tolist(),
                  leave_one_out=leave.tolist(), leave_one_out_range=[float(leave.min()),float(leave.max())],
                  seed_mean_changes=(y-x).reshape(s,-1).mean(1).tolist(),
                  method=method, resamples=len(samples), rng_seed=rng_seed if s!=4 else None,
                  coverage='Empirical resampling diagnostic; no guaranteed population or simultaneous coverage.')
    return result, dict(indices=indices, weights=weights, differences=samples, gram_difference=g)


def screen_decomposition(rows, heads, threshold):
    """Separate below-screen and remaining contributions without changing the estimand."""
    r = np.asarray(rows,dtype=float)
    mask = r < threshold
    low = np.where(mask,r,0).mean(-1)
    high = np.where(mask,0,r).mean(-1)
    chi = sample_susceptibility(low+high,heads)
    a,b = sample_susceptibility(low,heads),sample_susceptibility(high,heads)
    return dict(threshold=threshold, fraction=float(mask.mean()), susceptibility=chi,
                below=a, above=b, cross=chi-a-b,
                removal_relative=abs(chi-b)/chi if chi else None,
                median=float(np.median(r)), maximum=float(r.max()))
