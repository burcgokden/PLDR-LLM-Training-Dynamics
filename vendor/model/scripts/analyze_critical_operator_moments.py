#!/usr/bin/env python3
"""Exploratory nonlinear operator observations from completed saved endpoints.

The fixed eight-dimensional projection basis was chosen before coarse
training. Squared projected coordinates and total operator energy are
additional descriptive observations, selected after inspecting the
complete linear operator mean. They do not replace the primary row field.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from model_rg.critical_resampling import vector_statistics,replica_distances,replica_counts,resampled_variance

REPO=Path(__file__).resolve().parents[1]


def analyze(root,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((root/'protocol.json').read_text());ph=sha256(root/'protocol.json')
    verification=json.loads((root/'verification.json').read_text())
    if verification['status']!='passed' or p['role']!='observation' or p['training_updates']!=0:
        raise ValueError('Completed independently reconstructed observations are required')
    checked={str(root/'protocol.json'):ph,str(root/'verification.json'):sha256(root/'verification.json')}
    parent=Path(p['study']);native=json.loads((parent/'protocol.json').read_text());groups=defaultdict(list)
    for job in p['jobs']:
        folder=root/'runs'/job['run_id'];m=json.loads((folder/'manifest.json').read_text());raw=folder/'collectives.npz'
        if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=ph or m['training_updates']!=0:
            raise ValueError('Unbound complete endpoint observation')
        if sha256(raw)!=m['observations_sha256']:raise ValueError('Changed raw endpoint observation')
        checked[str(raw)]=sha256(raw);checked[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
        groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],raw))
    seeds=sorted(native['design']['seeds']);counts=replica_counts(len(seeds));cells=[];unresolved=[];max_error=0.
    for (e,n,g),rows in sorted(groups.items()):
        rows.sort()
        if [s for s,_ in rows]!=seeds:
            unresolved.append(dict(environment=e,heads=n,control=g));continue
        energy=[];projections=[]
        for _,path in rows:
            with np.load(path) as a:
                energy.append(a['head_fields'][...,2:3]**2)
                z=a['operator_projections']
                projections.append((z[...,None]*z[... ,None,:]).reshape(*z.shape[:-1],64))
        for name,values in [('operator_energy',energy),('projected_operator_second_moment',projections)]:
            x=np.stack(values);stat=vector_statistics(x);q=x.mean(3)
            independent=n*sum(float(np.mean((q[i]-q[j])**2)) for i in range(len(seeds)) for j in range(i))/(len(seeds)*(len(seeds)-1))
            error=abs(independent-stat['susceptibility']);max_error=max(max_error,error)
            if error>2e-11*max(stat['susceptibility'],1e-30):raise ValueError('Independent quadratic-field covariance differs')
            gram=replica_distances(q)
            stat.update(field=name,environment=e,heads=n,control=g,step=native['design']['steps'],seed_ids=seeds,
                susceptibility_percentile_95=np.quantile(n*resampled_variance(gram,counts),[.025,.975]).tolist(),
                analysis_role='postselection_exploratory',mean=float(x.mean()),
                field_units='Mean squared native operator entry' if name=='operator_energy' else 'All 8 by 8 ordered products in the fixed orthonormal operator projection basis',
                variance_units='Squared units of the named quadratic field; not comparable numerically to linear operator variance.')
            cells.append(stat)
    result=dict(schema='critical-operator-moments-v1',status='complete',study=str(root),parent_study=str(parent),
        cells=cells,unresolved_cells=unresolved,additional_training_updates=0,maximum_independent_pair_error=max_error,
        checked_sha256=checked,source_sha256={name:sha256(REPO/name) for name in ['scripts/analyze_critical_operator_moments.py','src/model_rg/critical_resampling.py','src/model_rg/provenance.py']},
        role='Exploratory nonlinear observation, selected after the completed coarse linear-operator profiles.',
        scope='No universal exponent or concentration of the full empirical head law is inferred from one first or second moment.')
    write_json(output,result);print(json.dumps({'status':'complete','cells':len(cells),'additional_training_updates':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.study.resolve(),a.output.resolve())
