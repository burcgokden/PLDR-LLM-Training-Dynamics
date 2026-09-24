#!/usr/bin/env python3
"""Independent endpoint-energy and adjacent-interval readout reconstruction."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from numerical_validation import load_json_strict,array_discrepancy
from matched_clock_coverage import read as read_strict, census


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def verify(study,parent,analysis):
    a=read_strict(analysis);p=read_strict(study/'protocol.json');parent=parent or Path(p['parent_transport']);err=0.;cases=[]
    if a['status']!='passed' or len(a['cells'])!=8 or a['protocol_sha256']!=sha(study/'protocol.json'):raise ValueError('Wrong result grid')
    expected={(n,g,9163401) for n in [4,8,14,24] for g in [1.5,2.]}
    census(p['jobs'],['heads','control','seed'],expected,'quadrature protocol')
    census(a['cells'],['heads','control','seed'],expected,'quadrature cells')
    if a['row_absolute_target']!=p['row_absolute_target'] or p['row_absolute_target']!=1e-5:raise ValueError('Changed quadrature target')
    jobs={(j['heads'],j['control'],j['seed']):j for j in p['jobs']}
    for c in a['cells']:
        if any(c[k]!=v for k,v in jobs[c['heads'],c['control'],c['seed']].items()):raise ValueError('Quadrature run binding')
        if [v['intervals'] for v in c['levels']]!=[1,2,4,8,16,32] or any(type(v['intervals']) is not int for v in c['levels']):raise ValueError('Quadrature level census')
    for kind,root in [('quadrature',study),('transport',parent)]:
        member='quadrature.npz' if kind=='quadrature' else 'observations.npz'
        expected_paths={'protocol.json'}|{'runs/'+j['run_id']+'/'+name for j in p['jobs'] for name in ['manifest.json',member]}
        if set(a['checked_sha256'][kind])!=expected_paths:raise ValueError('Quadrature observation census')
        for name,h in a['checked_sha256'][kind].items():
            if sha(root/name)!=h:raise ValueError('Changed observations')
    def compare(x,y):
        nonlocal err
        err=max(err,array_discrepancy(np.asarray(x),np.asarray(y),'independent finite transport',atol=4e-12))
    for c in a['cells']:
        with np.load(parent/'runs'/c['run_id']/'observations.npz') as z:A=z['A'].astype(np.float64);d0=float(z['gradient_dot'][3])
        # Orthogonal energy complement, independently of the centered-matrix formula.
        total=np.einsum('...ij,...ij->...',A,A);sums=A.sum(-2);common=np.einsum('...j,...j->...',sums,sums)/64
        row=(1-common/total).mean((1,2,3));delta=float(row[3]-row[0])
        with np.load(study/'runs'/c['run_id']/'quadrature.npz') as z:f=z['fractions'];d=z['derivative'];measured=z['row'].mean((1,2,3))
        compare(c['incoming_directional_estimate'],d0)
        compare(c['incoming_row'],row[0]);compare(c['endpoint_row'],row[3]);compare(c['observed_increment'],delta);compare(c['node_rows'],measured);compare(c['node_derivatives'],d)
        for level in c['levels']:
            stride=32//level['intervals'];value=sum(float((d[k]+d[k+stride])*(f[k+stride]-f[k])/2) for k in range(0,32,stride))
            compare(level['integrated_increment'],value);compare(level['absolute_error'],abs(value-delta))
        value=sum(float((d[k]+d[k+1])*(f[k+1]-f[k])/2) for k in range(32))
        compare(c['integrated_increment'],value);compare(c['predicted_endpoint'],row[0]+value);compare(c['absolute_error'],abs(value-delta))
        if c['target_met']!=bool(abs(value-delta)<=a['row_absolute_target']):raise ValueError('Changed accuracy decision')
        cases.append(dict(heads=c['heads'],control=c['control'],absolute_error=abs(value-delta)))
    return dict(status='passed',cells=8,levels=48,maximum_difference=err,cases=cases,analysis_sha256=sha(analysis),verifier_sha256=sha(Path(__file__)),scope='Independent raw-matrix energy-complement endpoints and adjacent-interval integration of all saved native derivative nodes; no new model calls.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=verify(a.study,a.parent,a.analysis);a.output.write_text(json.dumps(r,indent=2)+'\n');print(dict(status='passed',cells=8,maximum_difference=r['maximum_difference']))
