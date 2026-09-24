#!/usr/bin/env python3
"""Paired seed resampling across every control, width and recorded horizon."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.critical_resampling import replica_distances, replica_counts, resampled_variance, binary_variance_decomposition
from model_rg.provenance import sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def quantiles(x):
    a = np.asarray(x)
    a = a[np.isfinite(a)]
    return np.quantile(a, [.025, .5, .975]).tolist() if len(a) else [None, None, None]


def crossing_draws(controls, profiles, level):
    g = np.asarray(controls, dtype=float)
    y = np.asarray(profiles, dtype=float)
    crosses = (y[:, :-1] > level) & (y[:, 1:] <= level)
    number = crosses.sum(1)
    index = np.argmax(crosses, axis=1)
    rows = np.arange(len(y))
    left = y[rows, index]
    right = y[rows, index+1]
    value = np.full(len(y), np.nan)
    valid = number == 1
    value[valid] = g[index[valid]]+(level-left[valid])*(g[index[valid]+1]-g[index[valid]])/(right[valid]-left[valid])
    interior = valid & (g[index] > 0)
    return value, interior, number


def peak_draws(controls, profiles):
    g = np.asarray(controls, dtype=float)
    y = np.asarray(profiles, dtype=float)
    index = np.argmax(y, axis=1)
    peak = y[np.arange(len(y)), index]
    positive = peak > 0
    left = np.full(len(y), np.nan)
    right = left.copy()
    for k in range(len(g)-1):
        yl = y[:, k]
        yr = y[:, k+1]
        level = peak*.5
        hit_left = (k < index) & (yl <= level) & (level < yr)
        hit_right = (k >= index) & (yl > level) & (level >= yr) & ~np.isfinite(right)
        for mask, result in [(hit_left, left), (hit_right, right)]:
            result[mask] = g[k]+(level[mask]-yl[mask])*(g[k+1]-g[k])/(yr[mask]-yl[mask])
    return dict(peak=peak, index=index, controls=g[index], width=right-left,
                interior=positive & (index > 0) & (index < len(g)-1))


def analyze(study, analysis_path, output):
    if output.exists():
        raise FileExistsError(output)
    p = json.loads((study/'protocol.json').read_text())
    a = json.loads(analysis_path.read_text())
    if a['status'] != 'complete' or a['protocol_sha256'] != sha256(study/'protocol.json'):
        raise ValueError('A complete bound native analysis is required')
    seeds = sorted(p['design']['seeds'])
    counts = replica_counts(len(seeds))
    checked = {str(analysis_path): sha256(analysis_path), str(study/'protocol.json'): sha256(study/'protocol.json')}
    groups = defaultdict(list)
    for job in p['jobs']:
        folder = study/'runs'/job['run_id']
        m = json.loads((folder/'manifest.json').read_text())
        checked[str(folder/'manifest.json')] = sha256(folder/'manifest.json')
        if m['status'] != 'complete':
            continue
        if sha256(folder/'observations.npz') != m['artifacts']['observations.npz']:
            raise ValueError('Changed paired native observations')
        checked[str(folder/'observations.npz')] = m['artifacts']['observations.npz']
        groups[(job['environment'], job['heads'], job['control'])].append((job['seed'], folder/'observations.npz'))
    values = {}
    mixtures = []
    passage = []
    for (e, n, g), rows in sorted(groups.items()):
        rows.sort()
        if [s for s, _ in rows] != seeds:
            continue
        arrays = [np.load(path) for _, path in rows]
        try:
            data = np.stack([r['heads'] for r in arrays])
            steps = arrays[0]['steps']
            initial_row = data[:, 0, ..., 0].mean((1, 2, 3))
            trajectory = data[..., 0].mean((2, 3, 4))
            for i, seed in enumerate(seeds):
                for level in [.1, .5, .9]:
                    hits = np.flatnonzero(trajectory[i] <= level*initial_row[i])
                    onset = int(hits[0]) if len(hits) else None
                    passage.append(dict(environment=e, heads=n, control=g, seed=seed, fraction=level,
                                        first_observed_step=int(steps[onset]) if onset is not None else None,
                                        crossing_bracket=[int(steps[max(onset-1, 0)]), int(steps[onset])] if onset is not None else None,
                                        subsequent_recrossing=bool(np.any(trajectory[i, onset+1:] > level*initial_row[i])) if onset is not None else None,
                                        censoring_step=int(steps[-1])))
            for k, t in enumerate(steps):
                if int(t) not in p['observations']['milestone_steps']:
                    continue
                for label, coordinate in [('row', 0), ('attention', 1), ('operator', 2), ('common_amplitude', None)]:
                    if coordinate is None:
                        energy = data[:, k, ..., 3]
                        x = np.sqrt(np.maximum(energy-data[:, k, ..., 0]*np.maximum(energy, 1e-30), 0.))
                    else:
                        x = data[:, k, ..., coordinate]
                    q = x.mean(-1)
                    chi = n*resampled_variance(replica_distances(q), counts)
                    means = counts@q.mean((1, 2))/len(seeds)
                    values[(e, n, g, int(t), label)] = (chi, means)
                    if label == 'row':
                        labels = (q.mean((1, 2)) <= .5*initial_row).astype(int)
                        result = binary_variance_decomposition(q, labels, n)
                        result.update(environment=e, heads=n, control=g, step=int(t),
                                      classification='Current whole-run row mean at or below half its own initial mean.',
                                      seed_ids=seeds, labels=labels.tolist())
                        mixtures.append(result)
        finally:
            for array in arrays:
                array.close()
    controls = sorted(p['design']['controls'])
    curves = defaultdict(dict)
    for (e, n, g, t, field), value in values.items():
        curves[(e, n, t, field)][g] = value
    profiles = []
    size_groups = defaultdict(list)
    crossing_groups = defaultdict(list)
    for (e, n, t, field), curve in sorted(curves.items()):
        if set(curve) != set(controls):
            continue
        chi = np.stack([curve[g][0] for g in controls], axis=1)
        peaks = peak_draws(controls, chi)
        record = dict(environment=e, heads=n, step=t, field=field,
                      peak_control_percentile_95=quantiles(peaks['controls']),
                      peak_control_counts={str(g): int(np.sum(peaks['controls'] == g)) for g in controls},
                      peak_susceptibility_percentile_95=quantiles(peaks['peak']),
                      peak_interior_fraction=float(peaks['interior'].mean()),
                      half_width_resolved_fraction=float(np.isfinite(peaks['width']).mean()),
                      resolved_half_width_percentile_95=quantiles(peaks['width']))
        if field == 'row' and controls[0] == 0:
            mean = np.stack([curve[g][1] for g in controls], axis=1)
            ratio = mean/mean[:, 0, None]
            crossing, interior, number = crossing_draws(controls, ratio, .5)
            record.update(half_mean_crossing_percentile_95=quantiles(crossing),
                          unique_half_mean_crossing_fraction=float(np.mean(number == 1)),
                          nonzero_lower_bracket_fraction=float(interior.mean()))
            crossing_groups[(e, n)].append((t, crossing, interior))
        profiles.append(record)
        size_groups[(e, t, field)].append((n, peaks))
    sizes = []
    for (e, t, field), rows in sorted(size_groups.items()):
        rows.sort(key=lambda r: r[0])
        if len(rows) < 3:
            continue
        n = np.array([v[0] for v in rows], dtype=float)
        peak = np.stack([v[1]['peak'] for v in rows], axis=1)
        valid = np.all(peak > 0, axis=1)
        logn = np.log(n)
        centered = logn-logn.mean()
        slopes = np.full(len(peak), np.nan)
        slopes[valid] = np.log(peak[valid])@centered/(centered@centered)
        training = logn[:-1]-logn[:-1].mean()
        predictions = np.full(len(peak), np.nan)
        train_slopes = np.log(peak[valid, :-1])@training/(training@training)
        predictions[valid] = np.exp(np.log(peak[valid, :-1]).mean(1)+train_slopes*(logn[-1]-logn[:-1].mean()))
        sizes.append(dict(environment=e, step=t, field=field, heads=n.astype(int).tolist(),
                          effective_log_size_slope_percentile_95=quantiles(slopes),
                          largest_size_prediction_percentile_95=quantiles(predictions),
                          paired_largest_prediction_to_observation_percentile_95=quantiles(np.divide(predictions, peak[:, -1], out=np.full(len(peak), np.nan), where=peak[:, -1] > 0)),
                          all_peaks_interior_fraction=float(np.all(np.stack([v[1]['interior'] for v in rows], axis=1), axis=1).mean()),
                          scope='A joint-seed interval for a finite sampled-peak slope. The largest width is excluded only from the prediction fit; this does not establish universal exponents.'))
    drifts = []
    for (e, n), rows in sorted(crossing_groups.items()):
        rows = [r for r in rows if r[0] >= 256]
        if len(rows) < 4:
            continue
        # Retain every trailing milestone window containing at least four times.
        # An unresolved early horizon is not silently discarded from one fit.
        for offset in range(len(rows)-3):
            selected = rows[offset:]
            times = np.array([r[0] for r in selected], dtype=float)
            crossing = np.stack([r[1] for r in selected], axis=1)
            interior = np.stack([r[2] for r in selected], axis=1)
            valid = interior.all(1) & np.isfinite(crossing).all(1) & np.all(crossing > 0, axis=1)
            slopes = np.full(len(crossing), np.nan)
            centered = np.log(times)-np.log(times).mean()
            slopes[valid] = np.log(crossing[valid])@centered/(centered@centered)
            drifts.append(dict(environment=e, heads=n, times=times.astype(int).tolist(),
                               all_times_nonzero_bracket_fraction=float(valid.mean()),
                               log_control_vs_log_time_slope_percentile_95=quantiles(slopes),
                               scope='Descriptive kinetic drift. All trailing milestone windows with at least four times are retained; each fit requires jointly resolved positive-control brackets.'))
    result = dict(schema='critical-joint-analysis-v1', status='complete', study=str(study),
                  seed_ids=seeds, bootstrap_draws=len(counts), bootstrap_seed=9152592,
                  quantile_probabilities=[.025, .5, .975],
                  bootstrap_coupling='One multinomial seed-count row is shared across every width, control, time and source partition.',
                  variance_method='Direct pair differences; exact zero profiles unresolved.',
                  failed_paths=a.get('failed_paths',a.get('numerical_failures',[])), declared_paths=len(p['jobs']),
                  uncertainty_scope='Finite-seed descriptive percentiles; unresolved and boundary bootstrap mass remains explicit.',
                  profiles=profiles, size_slopes=sizes, crossing_drifts=drifts,
                  binary_row_decompositions=mixtures, first_passages=passage, checked_sha256=checked,
                  source_sha256={name: sha256(REPO/name) for name in ['scripts/analyze_critical_joint.py',
                      'src/model_rg/critical_resampling.py', 'src/model_rg/provenance.py']})
    write_json(output, result)
    print(json.dumps({'status': 'complete', 'profiles': len(profiles), 'size_diagnostics': len(sizes), 'bootstrap_draws': len(counts)}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.study.resolve(), a.analysis.resolve(), a.output.resolve())


if __name__ == '__main__':
    main()
