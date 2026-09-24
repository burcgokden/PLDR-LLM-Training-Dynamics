#!/usr/bin/env python3
"""Separate absolute transverse energy from changes in its normalization."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.critical_onepass import conditional_statistics
from model_rg.provenance import sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def analyze(study,analysis_path,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((study/'protocol.json').read_text());a=json.loads(analysis_path.read_text())
    ph=sha256(study/'protocol.json')
    if a['status']!='complete' or a['protocol_sha256']!=ph:raise ValueError('A complete bound native analysis is required')
    checked={str(study/'protocol.json'):ph,str(analysis_path):sha256(analysis_path)}
    groups=defaultdict(list)
    for job in p['jobs']:
        folder=study/'runs'/job['run_id'];mp=folder/'manifest.json';m=json.loads(mp.read_text())
        if m['job']!=job or m['protocol_sha256']!=ph or m['role']!='scientific':raise ValueError('Native binding differs')
        checked[str(mp)]=sha256(mp)
        if m['status']!='complete':continue
        raw=folder/'observations.npz'
        if sha256(raw)!=m['artifacts']['observations.npz']:raise ValueError('Changed native energy observations')
        checked[str(raw)]=sha256(raw);groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],raw))
    seeds=sorted(p['design']['seeds']);cells=[];budgets=[];unresolved=[];max_error=0.
    for (e,n,g),rows in sorted(groups.items()):
        rows.sort()
        if [s for s,_ in rows]!=seeds:
            unresolved.append(dict(environment=e,heads=n,control=g));continue
        data=[];steps=None
        for _,path in rows:
            with np.load(path) as raw:
                if steps is None:steps=raw['steps']
                np.testing.assert_array_equal(raw['steps'],steps);data.append(raw['heads'])
        data=np.stack(data);ratio=data[...,0];total=data[...,3];denominator=np.maximum(total,1e-30)
        centered=ratio*denominator;common=total-centered
        if np.any(common < -2e-12*(1+np.abs(total))):raise ValueError('Orthogonal common energy is negative')
        common=np.maximum(common,0.)
        for k,t in enumerate(steps):
            for label,values in [('total_energy',total),('centered_energy',centered),('common_energy',common)]:
                stat=conditional_statistics(values[:,k]);base=float(values[:,0].mean())
                stat.update(environment=e,heads=n,control=g,step=int(t),field=label,seed_ids=seeds,
                    mean_relative_to_initial=stat['mean']/base if base>0 else None,analysis_role='exploratory_geometry')
                cells.append(stat)
            mask=(centered[:,0]>0)&(centered[:,k]>0)&(ratio[:,0]>0)&(ratio[:,k]>0)
            fraction=float(mask.mean());entry=dict(environment=e,heads=n,control=g,step=int(t),
                strictly_positive_pair_fraction=fraction,zero_centered_energy_fraction=float(np.mean(centered[:,k]==0)),
                denominator_floor_fraction=float(np.mean(total[:,k]<1e-30)))
            if mask.any():
                dr=np.log(ratio[:,k][mask])-np.log(ratio[:,0][mask])
                dc=np.log(centered[:,k][mask])-np.log(centered[:,0][mask])
                dd=np.log(denominator[:,k][mask])-np.log(denominator[:,0][mask])
                residual=float(np.max(np.abs(dr-dc+dd)));max_error=max(max_error,residual)
                if residual>2e-12:raise ValueError('The relative-energy log budget failed')
                entry.update(mean_log_row_change=float(dr.mean()),mean_log_centered_energy_change=float(dc.mean()),
                    mean_log_denominator_change=float(dd.mean()),maximum_log_identity_error=residual,
                    averaging_scope='The explicitly reported subset of strictly positive initial/final row-energy pairs.')
            budgets.append(entry)
    result=dict(schema='critical-native-energy-v1',status='complete',study=str(study),cells=cells,
        log_budgets=budgets,unresolved_cells=unresolved,maximum_log_identity_error=max_error,checked_sha256=checked,
        source_sha256={name:sha256(REPO/name) for name in ['scripts/analyze_critical_energy.py','src/model_rg/critical_onepass.py','src/model_rg/provenance.py']},
        interpretation='Absolute centered energy and total normalization are reported separately. A falling relative row ratio alone does not specify absolute transverse contraction.',
        role='Exploratory geometry of the recorded native fields; this does not replace the primary relative-row endpoint.')
    write_json(output,result);print(json.dumps({'status':'complete','cells':len(cells),'maximum_log_identity_error':max_error}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['study','analysis','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.study.resolve(),a.analysis.resolve(),a.output.resolve())
