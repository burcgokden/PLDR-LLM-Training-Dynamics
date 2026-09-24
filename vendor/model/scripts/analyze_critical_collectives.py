#!/usr/bin/env python3
"""Resolve complete common fields and predictive distributions at every endpoint."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.critical_resampling import replica_distances, replica_counts, resampled_variance, vector_statistics
from model_rg.provenance import sha256, write_json

REPO = Path(__file__).resolve().parents[1]


def analyze(root, output):
    if output.exists():
        raise FileExistsError(output)
    p = json.loads((root/'protocol.json').read_text())
    if p['schema'] != 'critical-collective-observation-v1' or p['role'] != 'observation' or p['training_updates'] != 0:
        raise ValueError('Expected a zero-training native observation protocol')
    ph = sha256(root/'protocol.json')
    parent = Path(p['study'])
    native = json.loads((parent/'protocol.json').read_text())
    checked = {str(root/'protocol.json'): ph}
    for path, digest in p['input_sha256'].items():
        # Checkpoints were already replayed by the observer. Hashing them again
        # independently checks that the observation still refers to that state.
        if sha256(path) != digest:
            raise ValueError('Changed observation parent: '+path)
        checked[path] = digest
    for path, digest in p['producer_sha256'].items():
        if sha256(REPO/path) != digest:
            raise ValueError('Changed observation producer: '+path)
        checked[str(REPO/path)] = digest
    groups = defaultdict(list)
    replay_errors = []
    for job in p['jobs']:
        folder = root/'runs'/job['run_id']
        manifest = json.loads((folder/'manifest.json').read_text())
        if manifest['status'] != 'complete' or manifest['job'] != job or manifest['protocol_sha256'] != ph:
            raise ValueError('Incomplete or unbound collective observation')
        if manifest['training_updates'] != 0 or manifest['role'] != 'observation':
            raise ValueError('An observation acquired training updates')
        raw = folder/'collectives.npz'
        if sha256(raw) != manifest['observations_sha256']:
            raise ValueError('Changed collective array')
        checked[str(raw)] = sha256(raw)
        checked[str(folder/'manifest.json')] = sha256(folder/'manifest.json')
        groups[(job['environment'], job['heads'], job['control'])].append((job['seed'], raw))
        replay_errors.append([manifest['maximum_normalized_head_replay_error'], manifest['maximum_logit_replay_error']])
    seeds = sorted(native['design']['seeds'])
    counts = replica_counts(len(seeds))
    cells = []
    unresolved = []
    for (e, n, g), rows in sorted(groups.items()):
        rows.sort()
        if [s for s, _ in rows] != seeds:
            unresolved.append(dict(environment=e, heads=n, control=g, available_seeds=[s for s, _ in rows],
                                   reason='The full declared independent-initialization ensemble is unavailable.'))
            continue
        arrays = [np.load(path) for _, path in rows]
        try:
            common = np.stack([a['common_centroids'] for a in arrays])
            operators = np.stack([a['operator_projections'] for a in arrays])
            operator_mean = np.stack([a['mean_operators'] for a in arrays])
            if common.shape != (len(seeds), 64, 5, n, 64) or operators.shape != (len(seeds), 64, 5, n, 8):
                raise ValueError('Unexpected vector field dimensions')
            if operator_mean.shape != (len(seeds), 64, 5, 64, 64):
                raise ValueError('Unexpected complete mean operator dimensions')
            for field, x in [('common_vector', common), ('operator_projection', operators)]:
                stat = vector_statistics(x)
                gram = replica_distances(x.mean(3))
                draws = n*resampled_variance(gram, counts)
                stat.update(field=field, environment=e, heads=n, control=g, step=native['design']['steps'],
                            seed_ids=seeds, susceptibility_percentile_95=np.quantile(draws, [.025, .975]).tolist(),
                            bootstrap_draws=len(counts), bootstrap_unit='Complete paired training initialization.',
                            analysis_role='exploratory')
                cells.append(stat)
            gram = replica_distances(operator_mean)
            chi = n*resampled_variance(gram,np.ones((1,len(seeds)),dtype=np.int64))[0]
            cells.append(dict(field='complete_mean_operator', environment=e, heads=n, control=g,
                              step=native['design']['steps'], seed_ids=seeds,
                              susceptibility=float(chi), intensive_variance=float(chi/n),
                              susceptibility_percentile_95=np.quantile(n*resampled_variance(gram, counts), [.025, .975]).tolist(),
                              units='Per native matrix entry, averaged across contexts and layers.',
                              one_head_variance=None, analysis_role='exploratory',
                              scope='Complete mean operator fluctuations; individual full operators were not retained.'))
            logits = np.stack([a['logits'].astype(float) for a in arrays])
            logp = logits-logsumexp(logits, axis=-1, keepdims=True)
            probability = np.exp(logp)
            embedding = 2*np.exp(.5*logp)
            # This trace sums vocabulary coordinates, rather than dividing by
            # vocabulary size. Contexts retain their fixed uniform weights.
            gram = replica_distances(embedding)*embedding.shape[-1]
            chi = n*resampled_variance(gram,np.ones((1,len(seeds)),dtype=np.int64))[0]
            log_mixture = logsumexp(logp, axis=0)-np.log(len(seeds))
            js = np.mean(np.sum(probability*(logp-log_mixture), axis=-1))
            cells.append(dict(field='predictive_sqrt_probability', environment=e, heads=n, control=g,
                              step=native['design']['steps'], seed_ids=seeds,
                              susceptibility=float(chi), intensive_variance=float(chi/n),
                              susceptibility_percentile_95=np.quantile(n*resampled_variance(gram, counts), [.025, .975]).tolist(),
                              empirical_jensen_shannon=float(js),
                              mean_nll_by_seed=[float(a['nll'].mean()) for a in arrays],
                              units='N times the full covariance trace of 2*sqrt(p), averaged over eight fixed contexts.',
                              analysis_role='exploratory',
                              scope='A finite between-initialization predictive law, not a response to a conjugate training field.'))
        finally:
            for a in arrays:
                a.close()
    result = dict(schema='critical-collective-analysis-v1', status='complete', study=str(root), parent_study=str(parent),
                  protocol_sha256=ph, training_updates=0, observed_paths=len(p['jobs']),
                  excluded_parent_paths=p['excluded'], unresolved_cells=unresolved, cells=cells,
                  maximum_normalized_head_replay_error=max((v[0] for v in replay_errors), default=0.),
                  maximum_logit_replay_error=max((v[1] for v in replay_errors), default=0.), checked_sha256=checked,
                  source_sha256={name: sha256(REPO/name) for name in ['scripts/analyze_critical_collectives.py',
                      'src/model_rg/critical_resampling.py', 'src/model_rg/provenance.py']},
                  uncertainty_scope='Percentiles of a whole-seed descriptive bootstrap. With few training seeds they are not calibrated population tail bounds.',
                  conclusion='Complete common vectors distinguish amplitude concentration from concentration of the full common field.')
    write_json(output, result)
    print(json.dumps({key: result[key] for key in ['status', 'observed_paths', 'training_updates']}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    analyze(a.study.resolve(), a.output.resolve())


if __name__ == '__main__':
    main()
