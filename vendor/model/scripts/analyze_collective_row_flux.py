#!/usr/bin/env python3
"""Exact finite row-energy decomposition of all eight recorded native updates."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,finite_array,array_discrepancy


def analyze(study):
    p=load_json_strict((study/'protocol.json').read_text());ph=sha256(study/'protocol.json')
    expected={(n,g,9163401) for n in [4,8,14,24] for g in [1.5,2.]}
    if p['schema']!='collective-row-clock-v1' or len(p['jobs'])!=8 or {(j['heads'],j['control'],j['seed']) for j in p['jobs']}!=expected:raise ValueError('Incomplete native grid')
    checked={'protocol.json':ph};cells=[];maximum=0.;seconds=0.;peak=0
    for j in p['jobs']:
        folder=study/'runs'/j['run_id'];m=load_json_strict((folder/'manifest.json').read_text());path=folder/'observations.npz'
        if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=ph or m['scientific_updates']!=1 or m['native_forwards']!=5 or m['artifacts']!={'observations.npz':sha256(path)}:raise ValueError('Incomplete acquisition')
        for name in ['manifest.json','observations.npz']:checked[str((folder/name).relative_to(study))]=sha256(folder/name)
        with np.load(path) as z:
            raw=finite_array(z['A'],'native matrices').astype(float);recorded=finite_array(z['row'],'native rows');source=z['next_document_rows']
            if not np.array_equal(z['fractions'],[0,-.5,.5,1]):raise ValueError('Foreign state pairing')
        if raw.shape!=(4,8,5,j['heads'],64,64) or recorded.shape!=(4,8,5,j['heads']) or source.shape!=(32,) or len(np.unique(source))!=32:raise ValueError('Wrong field or source geometry')
        A=raw[0];D=raw[3]-A;next_A=raw[3];E=(A*A).sum((-2,-1));En=(next_A*next_A).sum((-2,-1))
        if (E<=0).any() or (En<=0).any():raise ValueError('Degenerate row energy')
        PA=A-A.mean(-2,keepdims=True);PD=D-D.mean(-2,keepdims=True)
        r=(PA*PA).sum((-2,-1))/E;rn=((PA+PD)**2).sum((-2,-1))/En
        lt=2*(PA*PD).sum((-2,-1))/En;ln=-2*r*(A*D).sum((-2,-1))/En
        qt=(PD*PD).sum((-2,-1))/En;qn=-r*(D*D).sum((-2,-1))/En
        linear=lt+ln;quadratic=qt+qn;change=rn-r
        maximum=max(maximum,array_discrepancy(r,recorded[0],'incoming native row',atol=3e-12),array_discrepancy(rn,recorded[3],'outgoing native row',atol=3e-12),array_discrepancy(change,linear+quadratic,'finite row flux',atol=3e-12))
        cells.append(dict(**j,incoming_row=float(r.mean()),outgoing_row=float(rn.mean()),increment=float(change.mean()),
            linear_flux=float(linear.mean()),quadratic_flux=float(quadratic.mean()),transverse_cross=float(lt.mean()),normalization_cross=float(ln.mean()),
            transverse_quadratic=float(qt.mean()),normalization_quadratic=float(qn.mean()),relative_matrix_energy=float(((D*D).sum((-2,-1))/E).mean()),
            physical_row_rate=float(j['steps']*change.mean()),per_context_increment=change.mean((1,2)).tolist(),
            maximum_identity_error=float(abs(change-linear-quadratic).max()),next_document_rows=source.tolist()))
        seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes'])
    return dict(status='passed',schema='collective-row-flux-v1',protocol_sha256=ph,checked_sha256=checked,cells=cells,
        scientific_updates=8,native_forward_calls=40,head_context_pairs=4000,maximum_identity_error=maximum,worker_seconds=seconds,peak_allocated_bytes=peak,
        analysis_source_sha256=sha256(__file__),scope='Deterministic finite-energy decomposition of every recorded one-step native continuation. One retained initialization and eight paired contexts per width/control; no population independence, derivative bound or autonomous closure claim.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=analyze(a.study);write_json(a.output,r)
    for c in r['cells']:print({k:c[k] for k in ['heads','control','increment','linear_flux','quadratic_flux','relative_matrix_energy','maximum_identity_error']})
    print(dict(maximum_identity_error=r['maximum_identity_error'],pairs=r['head_context_pairs']))
