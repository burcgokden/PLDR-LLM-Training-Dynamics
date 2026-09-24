#!/usr/bin/env python3
"""Reduce native operator fluctuations using the verified head-sign action.

A balanced orthogonal sign quadrature integrates all degree-two head-sign
moments exactly. The resulting representatives are not new independent
training realizations. The original six-seed covariance stays separate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]


def sign_quadrature(heads):
    if heads<1:raise ValueError('Positive head count required')
    size=1<<int(heads).bit_length()
    h=np.array([[-1 if (i&j).bit_count()%2 else 1 for j in range(1,heads+1)] for i in range(size)],dtype=float)
    np.testing.assert_array_equal(h.sum(0),np.zeros(heads))
    np.testing.assert_array_equal(h.T@h,size*np.eye(heads))
    return h


def analyze(root,qualification,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((root/'protocol.json').read_text());linear=json.loads((root/'analysis.json').read_text())
    q=json.loads(qualification.read_text())
    if linear['status']!='complete' or q['schema']!='critical-head-sign-analysis-v1' or not q['all_pairs_passed']:
        raise ValueError('Complete endpoint and native symmetry records are required')
    if any(r['maximum_normalized_state_error']>2e-6 or r['maximum_normalized_gradient_error']>2e-6 for r in q['records']):
        raise ValueError('Native symmetry qualification exceeded its declared envelope')
    checked={str(root/'protocol.json'):sha256(root/'protocol.json'),str(root/'analysis.json'):sha256(root/'analysis.json'),str(qualification):sha256(qualification)}
    for name,digest in q['checked_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed native symmetry record')
        checked[name]=digest
    parent=Path(p['study']);native=json.loads((parent/'protocol.json').read_text());seeds=sorted(native['design']['seeds'])
    checked[str(parent/'protocol.json')]=sha256(parent/'protocol.json');groups=defaultdict(list)
    for job in p['jobs']:
        folder=root/'runs'/job['run_id'];m=json.loads((folder/'manifest.json').read_text());raw=folder/'collectives.npz'
        if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha256(root/'protocol.json'):
            raise ValueError('Unbound native endpoint')
        if sha256(raw)!=m['observations_sha256']:raise ValueError('Changed operator observation')
        checked[str(raw)]=m['observations_sha256'];checked[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
        groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],raw))
    index={(c['environment'],c['heads'],c['control']):c for c in linear['cells'] if c['field']=='complete_mean_operator'}
    cells=[];unresolved=[];maximum_error=0.
    for (e,n,g),rows in sorted(groups.items()):
        rows.sort()
        if [seed for seed,_ in rows]!=seeds:unresolved.append(dict(environment=e,heads=n,control=g));continue
        power=[];projection=[]
        for _,path in rows:
            with np.load(path) as a:
                power.append(float(np.mean(a['head_fields'][...,2]**2)))
                projection.append(a['operator_projections'])
        z=np.stack(projection);h=sign_quadrature(n)
        signed=np.einsum('ra,sclad->srcld',h,z,optimize=True)/n
        centered=signed-signed.mean(axis=(0,1),keepdims=True)
        direct=float(n*np.mean(centered**2));prediction=float(np.mean(z**2))
        error=abs(direct-prediction);maximum_error=max(maximum_error,error)
        if error>2e-11*max(prediction,1e-30):raise ValueError('Exact sign quadrature differs from one-head projected power')
        actual=index[(e,n,g)]['susceptibility'];energy=float(np.mean(power))
        cells.append(dict(environment=e,heads=n,control=g,step=native['design']['steps'],seed_ids=seeds,
            original_unbiased_operator_susceptibility=actual,
            orbit_averaged_operator_susceptibility=energy,mean_one_head_operator_power=energy,
            original_to_one_head_power_ratio=actual/energy if energy else None,
            sign_quadrature_representatives=len(h),projected_quadrature_susceptibility=direct,
            projected_one_head_power=prediction,quadrature_reconstruction_error=error,
            quadrature_mean_maximum_absolute=float(np.max(np.abs(signed.mean(axis=(0,1)))))))
    result=dict(schema='critical-head-sign-observation-v1',status='complete',study=str(root),parent_study=str(parent),
        cells=cells,unresolved_cells=unresolved,additional_training_updates=0,maximum_quadrature_reconstruction_error=maximum_error,
        checked_sha256=checked,source_sha256={'scripts/analyze_critical_sign_observations.py':sha256(__file__)},
        qualification=str(qualification),role='Structural consequence of the native head-sign action; original finite-sample ratios are descriptive.',
        units='Complete-operator power per matrix entry; projected quantities per one of the eight fixed orthonormal coordinates.',
        scope='The sign quadrature integrates first and second orbit moments algebraically. It is not additional native training, additional independent seeds, or a full-distribution quadrature. The original unbiased covariance is retained separately from the orbit-averaged population covariance.')
    write_json(output,result);print(json.dumps({'status':'complete','cells':len(cells),'maximum_quadrature_error':maximum_error}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','qualification','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.study.resolve(),a.qualification.resolve(),a.output.resolve())
