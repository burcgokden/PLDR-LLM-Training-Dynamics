"""State-conditioned moment fitting and whole-path calibration.

Budgets are nested within each fit cohort. No assessment path enters a fit.
The archive control retains both its old mean and old physical covariance.
"""
import numpy as np

from model_rg.moment_rg import forecast


def increments(paths, observables):
    p = np.asarray(paths, dtype=float)
    if p.ndim != 3 or p.shape[1:] != (5, 2):
        raise ValueError('Expected paths x five horizons x risk/entropy')
    return np.diff(p[:, :, :observables], axis=1).transpose(0, 2, 1).reshape(len(p), -1)


def fit(paths, archived):
    result = {}
    for count, target in [(1, 'risk'), (2, 'risk_entropy')]:
        f = forecast(increments(paths, count))
        scale = np.asarray(f['scale']); outer = np.outer(scale, scale)
        corr = np.asarray(archived[target]['correlation'])
        transferred = (.75*corr + .25*np.diag(np.diag(corr)) + .001*np.eye(len(corr)))*outer
        result[target] = dict(mean=f['mean'], covariances={
            'local': (np.asarray(f['regularized'])*outer).tolist(),
            'diagonal': (np.asarray(f['diagonal'])*outer).tolist(),
            'transferred': transferred.tolist()})
    return result


def archive_fit(archived, no_change=False):
    result = {}
    for target, item in archived.items():
        c = np.asarray(item['archive_covariance'])
        d = np.maximum(np.sqrt(np.diag(c)), 1e-6)
        # This is a zero-acquisition baseline, regularized once in archive units.
        c = .75*c + .25*np.diag(np.diag(c)) + .001*np.diag(d*d)
        result[target] = dict(mean=(np.zeros(len(c)).tolist() if no_change else item['mean']),
                             covariances={'archive': c.tolist()})
    return result


def calibrate(paths, fitted):
    m = np.cumsum(fitted['risk']['mean'])
    b = np.tril(np.ones((4, 4)))
    c = np.asarray(fitted['risk']['covariances']['local'])
    a = np.sqrt(np.maximum(np.diag(b@c@b.T), 1e-12))
    y = np.asarray(paths)[:, 1:, 0] - np.asarray(paths)[:, 0, None, 0]
    scores = np.max(np.abs(y-m)/a, axis=1)
    return dict(mean=m.tolist(), scale=a.tolist(), radius=float(scores.max()), scores=scores.tolist())
