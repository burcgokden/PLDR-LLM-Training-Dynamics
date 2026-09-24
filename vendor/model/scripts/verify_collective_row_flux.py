#!/usr/bin/env python3
"""Independent common-row energy reconstruction of every finite native row flux."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from numerical_validation import load_json_strict, finite_array, array_discrepancy


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def verify(study,analysis):
    a=load_json_strict(analysis.read_text());p=load_json_strict((study/'protocol.json').read_text())
    expected={(n,g,9163401) for n in [4,8,14,24] for g in [1.5,2.]}
    if a['status']!='passed' or a['schema']!='collective-row-flux-v1' or len(a['cells'])!=8 or {(c['heads'],c['control'],c['seed']) for c in a['cells']}!=expected:raise ValueError('Incomplete result grid')
    if a['protocol_sha256']!=sha(study/'protocol.json') or p['scientific_updates']!=8 or p['native_forwards']!=40:raise ValueError('Foreign acquisition')
    for name,h in a['checked_sha256'].items():
        if sha(study/name)!=h:raise ValueError('Changed observation '+name)
    parent=Path(p['panel_reuse']['parent'])
    with np.load(parent/'selection.npz') as z:order=z['order'].copy()
    error=0.;pairs=0;cases=[]
    def compare(x,y):
        nonlocal error
        error=max(error,array_discrepancy(np.asarray(x),np.asarray(y),'independent finite row energy',atol=4e-12))
    for c in a['cells']:
        folder=study/'runs'/c['run_id'];m=load_json_strict((folder/'manifest.json').read_text())
        job={k:c[k] for k in ['heads','control','seed','run_id','steps']}
        if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=sha(study/'protocol.json') or m['incoming_row_replay_error']!=0 or m['endpoint_restored'] is not True:raise ValueError('Incomplete endpoint replay')
        if m['artifacts']!={'observations.npz':sha(folder/'observations.npz')} or m['scientific_updates']!=1 or m['native_forwards']!=5:raise ValueError('Invalid acquisition inventory')
        with np.load(folder/'observations.npz') as z:
            raw=finite_array(z['A'],'native matrices').astype(np.float64);stored=finite_array(z['row'],'native row coordinates');source=z['next_document_rows']
            if not np.array_equal(z['fractions'],[0,-.5,.5,1]):raise ValueError('Incorrect endpoint selection')
        if raw.shape!=(4,8,5,c['heads'],64,64):raise ValueError('Wrong matrix geometry')
        cursor=32*c['steps']
        if not np.array_equal(source,order[cursor:cursor+32]) or np.intersect1d(source,order[:cursor]).size:raise ValueError('Not a next-unused-block continuation')
        if c['next_document_rows']!=source.tolist():raise ValueError('Changed source identities')
        A=raw[0];B=raw[3];D=B-A;d=A.shape[-2]
        # Use common-row energy complements, not centered-matrix products.
        E=np.einsum('...ij,...ij->...',A,A);En=np.einsum('...ij,...ij->...',B,B)
        DD=np.einsum('...ij,...ij->...',D,D);AD=np.einsum('...ij,...ij->...',A,D)
        SA=A.sum(-2);SB=B.sum(-2);SD=D.sum(-2)
        common=np.einsum('...j,...j->...',SA,SA)/d
        common_next=np.einsum('...j,...j->...',SB,SB)/d
        common_cross=np.einsum('...j,...j->...',SA,SD)/d
        common_increment=np.einsum('...j,...j->...',SD,SD)/d
        if (E<=0).any() or (En<=0).any():raise ValueError('Degenerate energy')
        u=1-common/E;un=1-common_next/En;delta=un-u
        linear=2*((common/E)*AD-common_cross)/En
        quadratic=((common/E)*DD-common_increment)/En
        compare(u,stored[0]);compare(un,stored[3]);compare(delta,linear+quadratic)
        values=dict(incoming_row=u.mean(),outgoing_row=un.mean(),increment=delta.mean(),linear_flux=linear.mean(),quadratic_flux=quadratic.mean(),
            transverse_cross=(2*(AD-common_cross)/En).mean(),normalization_cross=(-2*u*AD/En).mean(),
            transverse_quadratic=((DD-common_increment)/En).mean(),normalization_quadratic=(-u*DD/En).mean(),
            relative_matrix_energy=(DD/E).mean(),physical_row_rate=c['steps']*delta.mean(),per_context_increment=delta.mean((1,2)))
        for key,value in values.items():compare(c[key],value)
        pairs+=u.size;cases.append(dict(heads=c['heads'],control=c['control'],increment=float(delta.mean())))
    if pairs!=a['head_context_pairs'] or a['scientific_updates']!=8 or a['native_forward_calls']!=40:raise ValueError('Incorrect accounting')
    return dict(status='passed',cells=8,head_context_pairs=pairs,maximum_difference=error,cases=cases,analysis_sha256=sha(analysis),
        protocol_sha256=sha(study/'protocol.json'),verifier_sha256=sha(__file__),
        scope='Independent common-row energy complement of all eight recorded updates, source-cursor checks and endpoint replay. Exact decomposition is not an autonomous prediction or a population replication.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=verify(a.study,a.analysis);a.output.write_text(json.dumps(r,indent=2)+'\n');print({k:r[k] for k in ['status','cells','head_context_pairs','maximum_difference']})
