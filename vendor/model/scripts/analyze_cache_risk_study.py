#!/usr/bin/env python3
"""Portable complete-grid reduction of fixed-panel cache risk and prefix transfer."""
import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict, finite_array, array_discrepancy
from cache_risk_validation import validate_operators
from analyze_operator_cache import paired_metrics

REPO = Path(__file__).resolve().parents[1]
ANALYSIS_SOURCES = ['scripts/analyze_cache_risk_study.py', 'scripts/cache_risk_validation.py', 'scripts/numerical_validation.py', 'scripts/analyze_operator_cache.py', 'src/model_rg/provenance.py']


def analyze(study, output):
    if output.exists(): raise FileExistsError(output)
    p = load_json_strict((study/'protocol.json').read_text()); ph = sha256(study/'protocol.json')
    if p['schema'] != 'cache-risk-v1' or p['status'] != 'frozen_before_acquisition':
        raise ValueError('Wrong frozen protocol')
    if len(p['arms']) != 28 or len({a['name'] for a in p['arms']}) != 28:
        raise ValueError('Incomplete arm grid')
    checked = {'protocol.json': ph}
    def bind(path, expected):
        digest = sha256(path)
        if digest != expected: raise ValueError('Changed artifact '+str(path))
        checked[str(path.relative_to(study))] = digest
    bind(study/'inputs.npz', p['selection_sha256'])
    for name, digest in p['source_sha256'].items():
        bind(study/'executed-source'/name, digest)
    if len(set(p['document_hashes'])) != 128: raise ValueError('Source overlap')
    with np.load(study/'inputs.npz') as a: blocks = a['blocks']
    if blocks.shape != (128, 129): raise ValueError('Wrong prefix panel')
    groups = defaultdict(list); manifests = {}; counts = {k: 0 for k in p['expected_calls']}
    for job in p['jobs']:
        groups[job['heads'], job['control']].append(job)
        folder = study/'runs'/job['run_id']; path = folder/'manifest.json'
        m = load_json_strict(path.read_text()); checked[str(path.relative_to(study))] = sha256(path)
        if m['status'] != 'complete' or m['job'] != job or m['protocol_sha256'] != ph or m['training_updates'] != 0:
            raise ValueError('Foreign or incomplete acquisition')
        expected = dict(exact_own_operator_replay=True, exact_native_restoration=True,
            native_generator_calls=40, cached_generator_calls=0, prefix_lengths=[32, 64, 128], longest_batch_one=True)
        if m['qualification'] != expected or m['peak_allocated_bytes'] >= p['memory_ceiling_bytes']:
            raise ValueError('Missing native or memory qualification')
        for name, digest in m['artifacts'].items(): bind(folder/name, digest)
        for key in counts: counts[key] += m['calls'][key]
        manifests[job['run_id']] = m
    if set(groups) != {(8, 0), (8, 1.5), (24, 0), (24, 1.5)} or counts != p['expected_calls']:
        raise ValueError('Incomplete condition grid or call accounting')
    operator_scan = validate_operators(study, p)
    cells = []; operator_error = 0.
    for (heads, control), jobs in sorted(groups.items()):
        if len(jobs) != 6 or len({j['seed'] for j in jobs}) != 6:
            raise ValueError('Incomplete initialization block')
        jobs = sorted(jobs, key=lambda j: j['seed'])
        for length in p['prefix_lengths']:
            native = np.stack([np.load(study/'runs'/j['run_id']/f'native-L{length}.npy') for j in jobs])
            if native.shape != (6, 64, p['vocabulary']): raise ValueError('Wrong native dimensions')
            for arm in p['arms']:
                if arm['length'] != length: continue
                cached = np.stack([np.load(study/'runs'/j['run_id']/(arm['name']+'.npy')) for j in jobs])
                result = paired_metrics(native, cached, blocks[64:, length])
                risks = []; biases = []; scatters = []; logit_energy = []
                for i, job in enumerate(jobs):
                    folder = study/'runs'/job['run_id']
                    with np.load(folder/f'operators-L{length}.npz') as a:
                        scatter = finite_array(a['scatter'], 'operator scatter')
                        risk = finite_array(a[arm['name']+'-risk'], 'operator risk')
                        bias = finite_array(a[arm['name']+'-bias'], 'operator displacement')
                        err = array_discrepancy(risk, scatter+bias, 'operator identity', atol=2e-12)
                        operator_error = max(operator_error, err)
                        if err > 2e-12 or min(risk.min(), scatter.min(), bias.min()) < 0:
                            raise ValueError('Fixed-panel risk identity failed')
                        risks.append(risk.tolist()); biases.append(bias.tolist()); scatters.append(scatter.tolist())
                    dz = native[i].astype(float)-cached[i].astype(float)
                    dz -= dz.mean(-1, keepdims=True)
                    logit_energy.append(float(np.mean(np.sum(dz*dz, axis=-1))))
                total_risk = np.sum(risks, axis=-1)
                result.update(arm, heads=heads, control=control, replicas=6, contexts=64,
                    operator_risk_by_seed_layer=risks, operator_scatter_by_seed_layer=scatters,
                    operator_displacement_by_seed_layer=biases,
                    mean_operator_risk=float(np.mean(risks)), mean_operator_scatter=float(np.mean(scatters)),
                    mean_operator_displacement=float(np.mean(biases)),
                    per_seed_centered_logit_energy=logit_energy,
                    empirical_gain_by_seed=[float(np.sqrt(v/r)) if r > 0 else None for v, r in zip(logit_energy, total_risk)])
                result['targets_met'] = bool(result['relative_centered_rms'] <= .25 and result['mean_kl'] <= .03)
                cells.append(result)
                print(f"{heads} {control} {arm['name']} R={result['relative_centered_rms']:.6f} KL={result['mean_kl']:.7f}", flush=True)
                del cached
            del native
    if len(cells) != 112: raise ValueError('Missing condition/arm cell')
    result = dict(status='passed', schema='cache-risk-analysis-v2', protocol_sha256=ph,
        cells=cells, calls=counts, new_training_updates=0, new_training_replicas=0,
        worker_seconds=sum(m['elapsed_seconds'] for m in manifests.values()),
        peak_allocated_bytes=max(m['peak_allocated_bytes'] for m in manifests.values()),
        maximum_operator_identity_error=operator_error,
        scientific_target_cells=sum(c['targets_met'] for c in cells), total_cells=len(cells),
        checked_sha256=checked, analysis_source_sha256=sha256(__file__),
        analysis_sources_sha256={n: sha256(REPO/n) for n in ANALYSIS_SOURCES}, operator_scan=operator_scan,
        scope='Fixed assessment and calibration panels. Relative centered errors use the same six paired training seeds in every arm. Empirical logit/operator gains are secant diagnostics, not uniform Lipschitz estimates. No new timing or stochastic calibration-size law.')
    write_json(output, result)
    print({k: result[k] for k in ['status', 'calls', 'scientific_target_cells', 'maximum_operator_identity_error', 'worker_seconds']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args(); analyze(a.study.resolve(), a.output.resolve())
