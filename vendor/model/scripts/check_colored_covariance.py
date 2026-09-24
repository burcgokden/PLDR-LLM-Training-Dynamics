#!/usr/bin/env python
"""Finite covariance qualifications and quadrature of the conditional spectral class."""
import argparse
from fractions import Fraction
from itertools import product
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from model_rg.provenance import sha256, write_json


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    f = Fraction
    atoms = list(product([f(-1), f(1)], repeat=4))
    x = [(a[0]+a[1])/2 for a in atoms]
    var = lambda xs: sum((v-sum(xs)/len(xs))**2 for v in xs)/len(xs)
    assert 2*var(x) == 1 and 2*var([f(3, 2)*v for v in x]) == f(9, 4)
    rng = np.random.default_rng(4110911); errors = []; omitted = []
    for _ in range(64):
        latent = rng.normal(size=(37, 6)); latent -= latent.mean(0)
        initial = latent@rng.normal(size=(6, 3)); state = initial.copy()
        matrices = []; noises = []
        for t in range(4):
            a = rng.normal(size=(3, 3))/2; e = latent@rng.normal(size=(6, 3))
            cross = state.T@e/37
            no_cross = a@(state.T@state/37)@a.T + e.T@e/37
            new = state@a.T+e
            exact = new.T@new/37
            full = no_cross+a@cross+cross.T@a.T
            errors.append(float(np.max(np.abs(exact-full))))
            omitted.append(float(np.max(np.abs(exact-no_cross))))
            np.testing.assert_allclose(exact, full, atol=2e-10, rtol=2e-12)
            matrices.append(a); noises.append(e); state = new
        transforms = [None]*5; acc = np.eye(3)
        for t in reversed(range(4)):
            transforms[t+1] = acc.copy(); acc = acc@matrices[t]
        transforms[0] = acc
        sources = [initial, *noises]
        blocked = sum(transforms[i]@(sources[i].T@sources[j]/37)@transforms[j].T
                      for i in range(5) for j in range(5))
        errors.append(float(np.max(np.abs(blocked-state.T@state/37))))
        np.testing.assert_allclose(blocked, state.T@state/37, atol=2e-10, rtol=2e-12)
    spectral = []
    for alpha in [-.5, 0., .5]:
        for delta in [.01, .001, .0001, .00001]:
            lam = 1-delta
            def smooth(w):
                ratio = 1. if w == 0 else 2*np.sin(w/2)/w
                return ratio**alpha/(delta**2+4*lam*np.sin(w/2)**2)
            value, error = quad(smooth, 0., np.pi, weight='alg', wvar=(alpha, 0),
                                epsabs=1e-9, epsrel=1e-10, limit=500)
            value /= np.pi; error /= np.pi
            asymptotic = delta**(alpha-1)/(2*np.cos(np.pi*alpha/2))
            if alpha == 0:
                np.testing.assert_allclose(value, 1/(1-lam**2), rtol=1e-10)
            spectral.append(dict(alpha=alpha, gap=delta, variance=value,
                                 leading=asymptotic, ratio=value/asymptotic, quadrature_error=error))
        ratios = [abs(r['ratio']-1) for r in spectral if r['alpha'] == alpha]
        assert all(ratios[i+1] < ratios[i] for i in range(3))
        assert ratios[-1] < .01
    result = dict(status='passed', schema='colored-covariance-mathematics-v1',
                  exact_atoms=16, state_noise_counterexample={'actual':'9/4', 'without_cross':'5/4'},
                  matrix_cases=64, max_covariance_error=max(errors),
                  max_omitted_cross_error=max(omitted), spectral=spectral,
                  checker_sha256=sha256(__file__),
                  scope='Finite mathematical and quadrature checks. No native stationarity, '
                        'spectral exponent, Gaussianity or thermodynamic limit is inferred.')
    write_json(out, result)
    print('Passed 64 correlated matrix cases and 12 spectral quadratures.', flush=True)


if __name__ == '__main__': main()
