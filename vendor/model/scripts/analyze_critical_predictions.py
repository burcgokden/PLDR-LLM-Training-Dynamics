#!/usr/bin/env python3
"""Full predictive-law variability at every acquired native milestone.

The initial predictive law is retained in the same units as trained laws.
An extensive trace after a variance-normalized random readout is not by
itself a collective critical fluctuation. All complete seed cells are kept.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.critical_resampling import replica_distances, replica_counts, resampled_variance
from model_rg.provenance import sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def analyze(study, scalar_path, output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((study/'protocol.json').read_text())
    scalar=json.loads(scalar_path.read_text())
    if scalar['status']!='complete' or scalar['protocol_sha256']!=sha256(study/'protocol.json'):
        raise ValueError('Completed native scalar reconstruction is required')
    checked={str(study/'protocol.json'):sha256(study/'protocol.json'),str(scalar_path):sha256(scalar_path)}
    groups=defaultdict(list)
    for job in p['jobs']:
        folder=study/'runs'/job['run_id'];m=json.loads((folder/'manifest.json').read_text())
        if m['job']!=job or m['protocol_sha256']!=scalar['protocol_sha256'] or m['role']!='scientific':
            raise ValueError('Unbound predictive record')
        if m['status']!='complete':continue
        path=folder/'observations.npz'
        if sha256(path)!=m['artifacts']['observations.npz']:raise ValueError('Predictive acquisition changed')
        checked[str(path)]=sha256(path);checked[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
        groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],path))
    seeds=sorted(p['design']['seeds']);counts=replica_counts(len(seeds));cells=[];maximum_error=0.
    initial_laws={}
    for (e,n,g),rows in sorted(groups.items()):
        rows.sort()
        if [s for s,_ in rows]!=seeds:continue
        arrays=[np.load(path) for _,path in rows]
        try:
            times=sorted({int(name.removeprefix('logits_')) for name in arrays[0].files if name.startswith('logits_')})
            baseline=None
            for t in times:
                logits=np.stack([a[f'logits_{t}'].astype(float) for a in arrays])
                if logits.shape[:2]!=(len(seeds),8):raise ValueError('Unexpected seed/context axes')
                if t==0:
                    key=(e,n)
                    if key in initial_laws:
                        np.testing.assert_array_equal(logits,initial_laws[key])
                    else:initial_laws[key]=logits.copy()
                logp=logits-logsumexp(logits,axis=-1,keepdims=True)
                probability=np.exp(logp);embedding=2*np.exp(.5*logp)
                gram=replica_distances(embedding)*embedding.shape[-1]
                variance=float(resampled_variance(gram,np.ones((1,len(seeds)),dtype=np.int64))[0]);chi=n*variance
                # Independent positive-affinity normalization and unordered pairs.
                weights=np.exp(logits-logits.max(-1,keepdims=True))
                roots=np.sqrt(weights/weights.sum(-1,keepdims=True))
                pair_sum=0.
                for i in range(len(seeds)):
                    for j in range(i):
                        pair_sum+=float(np.mean(.5*np.sum((roots[i]-roots[j])**2,axis=-1)))
                independent=8*n*pair_sum/(len(seeds)*(len(seeds)-1))
                error=abs(chi-independent)/(1+abs(chi));maximum_error=max(maximum_error,error)
                if error>2e-12 or variance>4+2e-12 or variance<0:
                    raise ValueError('Predictive trace reconstruction or simplex bound failed')
                if t==0:baseline=variance
                mixture=logsumexp(logp,axis=0)-np.log(len(seeds))
                js=float(np.mean(np.sum(probability*(logp-mixture),axis=-1)))
                if js < -2e-12 or js > np.log(len(seeds))+2e-12:
                    raise ValueError('Empirical Jensen-Shannon bound failed')
                cells.append(dict(environment=e,heads=n,control=g,step=t,seed_ids=seeds,
                    susceptibility=chi,intensive_variance=variance,
                    susceptibility_percentile_95=np.quantile(n*resampled_variance(gram,counts),[.025,.975]).tolist(),
                    empirical_jensen_shannon=js,
                    variance_relative_to_initial=variance/baseline if baseline else None,
                    variance_change_from_initial=variance-baseline if baseline is not None else None,
                    heldout_nll_by_seed=[float(a[f'evaluation_nll_{t}'].mean()) for a in arrays],
                    generator_clock=.0003*g*t,other_clock=.0006*t/n,
                    consumed_fraction=32*t/p['source']['population_blocks'],analysis_role='exploratory'))
        finally:
            for a in arrays:a.close()
    result=dict(schema='critical-predictive-milestones-v1',status='complete',study=str(study),cells=cells,
        maximum_normalized_pair_error=maximum_error,checked_sha256=checked,
        source_sha256={name:sha256(REPO/name) for name in ['scripts/analyze_critical_predictions.py',
            'src/model_rg/critical_resampling.py','src/model_rg/provenance.py']},
        units='N times the full vocabulary-summed covariance trace of 2*sqrt(p), averaged over eight fixed contexts.',
        uncertainty_scope='Descriptive resampling of complete independent training initializations; initial and trained laws share their units.',
        interpretation='An extensive predictive trace can arise from variance-normalized random readout. Its initial magnitude, evolution, head-sector coupling and response must be assessed separately.')
    write_json(output,result)
    print(json.dumps({'status':'complete','predictive_cells':len(cells),'maximum_pair_error':maximum_error}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze(a.study.resolve(),a.analysis.resolve(),a.output.resolve())
