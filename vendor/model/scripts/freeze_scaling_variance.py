#!/usr/bin/env python
"""Freeze finite-width variance predictions before additional initializations execute."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

from analyze_size_time import empirical_weights
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


OBSERVABLES = ['row_native', 'absolute_row_energy', 'common_centroid',
               'mean_metric_contrast', 'attention', 'operator_rms',
               'prediction_entropy', 'nll', 'logit_projection']


def fit_variance(widths, chi, query):
    widths, chi, query = map(lambda x:np.asarray(x, dtype=float), (widths, chi, query))
    if np.any(chi <= 0):
        raise ValueError('Positive reference susceptibility is required')
    fits = {}
    for name, design, target in [
        ('finite_common', np.column_stack((widths, np.ones_like(widths))), np.column_stack((query, np.ones_like(query)))),
        ('independent_head', np.column_stack((np.ones_like(widths), 1/widths)), np.column_stack((np.ones_like(query), 1/query)))]:
        # Fixed relative-residual least squares; the empirical chi is a weight,
        # not a calibrated sampling variance. Both coefficients are nonnegative.
        coef, _ = nnls(design/chi[:, None], np.ones_like(chi))
        fitted, predicted = design@coef, target@coef
        fits[name] = dict(coefficients=coef.tolist(), prediction=predicted.tolist(),
                          reference=fitted.tolist(), relative_rmse=float(np.sqrt(np.mean((fitted/chi-1)**2))))
    exponent, intercept = np.polyfit(np.log(widths), np.log(chi), 1)
    prediction = np.exp(intercept)*query**exponent
    fitted = np.exp(intercept)*widths**exponent
    fits['free_count_power'] = dict(coefficients=[float(np.exp(intercept)), float(exponent)],
        prediction=prediction.tolist(), reference=fitted.tolist(),
        relative_rmse=float(np.sqrt(np.mean((fitted/chi-1)**2))))
    return fits


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    a = p.parse_args();study = Path(a.root)/a.study
    analysis = study/'analysis/baseline-head-covariance'
    meta = json.loads((analysis/'manifest.json').read_text())
    if meta['status'] != 'complete' or sha256(analysis/'results.json') != meta['results_sha256']:
        raise AssertionError('Reference common-platform analysis changed')
    report = json.loads((analysis/'results.json').read_text())
    protocol = study/'protocols/width-replication.json'
    spec = json.loads(protocol.read_text())
    for job in spec['jobs']:
        if (study/'runs'/job['run_id']).exists():
            raise AssertionError('A selected confirmation trajectory has already started')
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False)
    bind_run(out, [analysis/'manifest.json', analysis/'results.json', analysis/'measurements.npz', protocol], vars(a))
    records, raw = [], {}
    train_widths, test_widths = [2, 4, 8, 14], [2, 4, 8, 24]
    lookup = {r['heads']:r for r in report['conditions'] if r['seed_count'] == 4 and r['step'] == 16384}
    with np.load(analysis/'measurements.npz') as z:
        for name in OBSERVABLES:
            chi = np.array([lookup[n]['observables'][name]['susceptibility'] for n in train_widths])
            fits = fit_variance(train_widths, chi, test_widths)
            boots = np.stack([z[f'float32_N{n}_g1_t16384_s4_c640011_b640001_{name}_bootstrap'] for n in train_widths])
            if boots.shape != (4, len(empirical_weights(4))):
                raise AssertionError('Whole-seed calibration resampling changed')
            valid = np.all(boots > 0, axis=0)
            forecasts = {key:[] for key in fits}
            for row in boots[:, valid].T:
                for key, result in fit_variance(train_widths, row, test_widths).items():
                    forecasts[key].append(result['prediction'])
            for key, result in fits.items():
                values = np.asarray(forecasts[key])
                raw[name+'_'+key+'_bootstrap'] = values
                result['prediction_empirical_percentiles'] = np.quantile(values, [.025, .975], axis=0).T.tolist()
            # This uses each width's own first four seeds; it is a replication
            # baseline and has no width-holdout interpretation at N=24.
            fits['same_width_replication'] = dict(
                prediction=[lookup[n]['observables'][name]['susceptibility'] for n in test_widths],
                scope='The first-four estimate at each target width, including the already observed N24 panel.')
            raw[name+'_same_width_replication_bootstrap'] = np.stack([
                z[f'float32_N{n}_g1_t16384_s4_c640011_b640001_{name}_bootstrap'] for n in test_widths], axis=1)
            records.append(dict(observable=name, train_widths=train_widths, train_susceptibility=chi.tolist(),
                test_widths=test_widths, models=fits, calibration_bootstrap_nonzero=int(valid.sum()),
                calibration_bootstrap_zero_excluded=int((~valid).sum())))
    frozen_at = datetime.now(timezone.utc).isoformat()
    result = dict(schema='frozen-finite-variance-predictions-v1', status='complete', frozen_at=frozen_at,
        step=16384, calibration_initializations=list(range(640101, 640105)),
        target_initializations=list(range(640105, 640109)), target_widths=test_widths, forecasts=records,
        targets=spec['jobs'], target_protocol_sha256=sha256(protocol),
        models=dict(finite_common='chi(N)=A*N+B, A,B >= 0', independent_head='chi(N)=A+B/N, A,B >= 0',
                    free_count_power='chi(N)=A*N^k, A>0, free real k',
                    same_width_replication='Use the measured first-four chi at that same width'),
        fitting='Finite common and independent-head models minimize squared relative residuals at N=2,4,8,14. The free power minimizes squared log residuals. All four alternatives and all nine observations are retained.',
        uncertainty='The 256 exact calibration resamples keep the same initialization multiplicities across widths. Resamples with zero susceptibility cannot be log fitted and are explicitly counted; these finite empirical percentiles are not coverage-certified confidence intervals.',
        scope='All target initializations at 16384 are new completions. N24 is withheld from fitting the first three models, but its earlier first-four observations were available during design; this is a frozen finite model extrapolation and fresh-initialization check, not a blind discovery of an unseen width. A free count power is not a critical exponent.')
    write_json(out/'results.json', result);np.savez_compressed(out/'measurements.npz', **raw)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), raw_sha256=sha256(out/'measurements.npz'),
        frozen_at=frozen_at, observations=len(records), scalar_forecasts=len(records)*4*4))
    print('Frozen', len(records)*4*4, 'variance predictions before all 16 target trajectories', flush=True)


if __name__ == '__main__':main()
