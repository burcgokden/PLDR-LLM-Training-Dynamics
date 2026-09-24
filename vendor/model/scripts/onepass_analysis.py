"""Declared identities and finite-ensemble summaries for the single-pass study."""
import itertools
import numpy as np


GROUP_KEYS = ('schedule_horizon', 'recipe', 'heads', 'step', 'shared_seed', 'stream_seed')


def group_key(case):
    return tuple(case[k] for k in GROUP_KEYS)


def group_record(key):
    return dict(zip(GROUP_KEYS, key, strict=True))


def label(key):
    horizon, recipe, heads, step, shared, stream = key
    return f'H{horizon}_{recipe}_N{heads}_t{step}_c{shared}_b{stream}'


def draws(count):
    if count not in (1, 2, 4):
        raise AssertionError('The declared study uses one, two or four initialization identities')
    return np.asarray(list(itertools.product(range(count), repeat=count)), dtype=int)


def empirical_mean(values):
    count = len(values)
    indices = draws(count)
    record = dict(seed_values=values, selected_seeds=count, defined_seeds=sum(v is not None for v in values))
    if any(v is None for v in values):
        return dict(**record, mean=None, empirical_percentiles=None, leave_one_out=None)
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all():
        raise AssertionError('Nonfinite single-pass summary')
    return dict(**record, mean=float(x.mean()),
        empirical_percentiles=np.quantile(x[indices].mean(-1), [.025, .975]).tolist() if count > 1 else None,
        leave_one_out=[float(np.delete(x, i).mean()) for i in range(count)] if count > 1 else None)


def statistics(values, heads):
    q = np.asarray(values, dtype=float)
    count = len(q)
    indices = draws(count)
    if not np.isfinite(q).all():
        raise AssertionError('Nonfinite collective field')
    x = q.reshape(count, -1)
    cohort = x.mean(-1)
    answer = dict(mean=float(x.mean()), independent_seeds=count, contexts=q.shape[1],
        cohort_means=cohort.tolist(), susceptibility=None, central_second=None, central_fourth=None,
        fourth_ratio=None, empirical_percentiles=None, mean_empirical_percentiles=None,
        jackknife_se=None, leave_one_out=None, bootstrap_samples=0, bootstrap_zero_fraction=None,
        cohort_susceptibility=None, cohort_fourth_ratio=None, cohort_binder=None,
        finite_seed_gaussian_binder_mean=None)
    if count == 1:
        return answer, np.array([], dtype=float), cohort
    centered = x-x.mean(0)
    m2 = float(np.mean(centered**2)); m4 = float(np.mean(centered**4))
    gram = centered@centered.T/x.shape[1]
    distance = np.maximum(np.diag(gram)[:, None]+np.diag(gram)[None, :]-2*gram, 0)
    np.fill_diagonal(distance, 0)
    weights = np.asarray([np.bincount(row, minlength=count) for row in indices], dtype=float)
    boot = heads*np.einsum('bi,ij,bj->b', weights, distance, weights)/(2*count*(count-1))
    d = cohort-cohort.mean(); a = float(np.mean(d*d)); b = float(np.mean(d**4))
    leave = np.asarray([heads*np.var(np.delete(x, i, axis=0), axis=0, ddof=1).mean()
                        for i in range(count)]) if count > 2 else None
    answer.update(susceptibility=float(heads*np.var(x, axis=0, ddof=1).mean()),
        central_second=m2, central_fourth=m4, fourth_ratio=m4/m2**2 if m2 else None,
        empirical_percentiles=np.quantile(boot, [.025, .975]).tolist(),
        mean_empirical_percentiles=np.quantile(cohort[indices].mean(-1), [.025, .975]).tolist(),
        jackknife_se=float(np.sqrt((count-1)/count*np.sum((leave-leave.mean())**2))) if leave is not None else None,
        leave_one_out=leave.tolist() if leave is not None else None,
        bootstrap_samples=len(indices), bootstrap_zero_fraction=float(np.mean(boot == 0)),
        cohort_susceptibility=float(heads*np.var(cohort, ddof=1)),
        cohort_fourth_ratio=b/a**2 if a else None, cohort_binder=1-b/(3*a**2) if a else None,
        finite_seed_gaussian_binder_mean=2/(count+1))
    return answer, boot, cohort


def arithmetic(left, right, heads):
    rows = {}
    for name in sorted(left):
        x, y = left[name], right[name]
        if x.shape != y.shape:
            raise AssertionError('The two arithmetic programs use different coordinates')
        error = x-y
        report = dict(rms_field_difference=float(np.sqrt(np.mean(error**2))),
            max_field_difference=float(np.max(np.abs(error))),
            float32_susceptibility=None, float64_susceptibility=None, absolute_difference=None,
            centered_error_susceptibility=None, susceptibility_error_bound=None, relative_bound=None,
            meets_one_percent_or_1e_minus8_absolute=None)
        if len(x) > 1:
            cx = float(heads*np.var(x, axis=0, ddof=1).mean())
            cy = float(heads*np.var(y, axis=0, ddof=1).mean())
            ce = float(heads*np.var(error, axis=0, ddof=1).mean())
            bound = float(2*np.sqrt(cy*ce)+ce)
            if abs(cx-cy) > bound*(1+1e-7)+1e-20:
                raise AssertionError('Paired arithmetic variance bound failed')
            report.update(float32_susceptibility=cx, float64_susceptibility=cy,
                absolute_difference=abs(cx-cy), centered_error_susceptibility=ce,
                susceptibility_error_bound=bound, relative_bound=bound/cy if cy else None,
                meets_one_percent_or_1e_minus8_absolute=bool(bound <= max(.01*cy, 1e-8)))
        rows[name] = report
    return rows


def phases(job):
    stop = job['steps']; horizon = job['schedule_horizon']; warmup = job['profile']['warmup_steps']
    if horizon == 0:
        return [('constant', 0, stop)]
    candidates = [('warmup', 0, min(warmup, stop)),
                  ('annealing', min(warmup, stop), min(horizon, stop)),
                  ('floor', min(horizon, stop), stop)]
    return [(name, begin, end) for name, begin, end in candidates if end > begin]


def windows(job, widths):
    result = []
    for phase, start, stop in phases(job):
        result.append(dict(phase=phase, kind='complete_observed_phase', begin=start, end=stop, width=stop-start))
        for width in widths:
            for begin in range(start, stop-width+1, width):
                result.append(dict(phase=phase, kind='phase_local_block', begin=begin, end=begin+width, width=width))
        if phase == 'floor':
            if (stop-start) % 2:
                raise AssertionError('The declared floor halves require an even interval')
            half = (stop-start)//2
            for part in range(2):
                begin = start+part*half
                result.append(dict(phase=phase, kind='floor_half', begin=begin, end=begin+half, width=half))
    return result
