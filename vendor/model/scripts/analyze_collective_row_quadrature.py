#!/usr/bin/env python3
"""Reconstruct finite native readout transport from all saved derivative nodes."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,finite_array,array_discrepancy
from matched_clock_coverage import read as read_strict, census


def analyze(study,parent=None):
    p=read_strict(study/'protocol.json');parent=parent or Path(p['parent_transport']);base=load_json_strict((parent/'protocol.json').read_text())
    ph=sha256(study/'protocol.json');bh=sha256(parent/'protocol.json')
    expected={(n,g,9163401) for n in [4,8,14,24] for g in [1.5,2.]}
    census(p['jobs'],['heads','control','seed'],expected,'quadrature jobs')
    if p['row_absolute_target']!=1e-5:raise ValueError('Changed quadrature target')
    if p['schema']!='collective-row-quadrature-v1' or p['intervals']!=32 or p['jobs']!=base['jobs'] or len(p['jobs'])!=8:raise ValueError('Incomplete fixed grid')
    checked={'quadrature':{'protocol.json':ph},'transport':{'protocol.json':bh}};cells=[];err=0.;seconds=0.;peak=0;calls=0
    for j in p['jobs']:
        f=study/'runs'/j['run_id'];b=parent/'runs'/j['run_id'];m=load_json_strict((f/'manifest.json').read_text());bm=load_json_strict((b/'manifest.json').read_text())
        if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=ph or m['native_forwards']!=34 or m['optimizer_replay_updates']!=1 or m['transport_replay_error']>1e-9:raise ValueError('Incomplete quadrature point')
        if bm['status']!='complete' or bm['job']!=j or bm['protocol_sha256']!=bh or bm['scientific_updates']!=1 or bm['native_forwards']!=5:raise ValueError('Incomplete transport point')
        for name in ['manifest.json','quadrature.npz']:checked['quadrature'][str((f/name).relative_to(study))]=sha256(f/name)
        for name in ['manifest.json','observations.npz']:checked['transport'][str((b/name).relative_to(parent))]=sha256(b/name)
        if m['artifacts']!={'quadrature.npz':sha256(f/'quadrature.npz')} or bm['artifacts']!={'observations.npz':sha256(b/'observations.npz')}:raise ValueError('Changed artifact')
        with np.load(f/'quadrature.npz') as z:q={k:finite_array(z[k],k) for k in z.files}
        with np.load(b/'observations.npz') as z:A=z['A'].astype(float);next_rows=z['next_document_rows'];original_rows=z['row'];incoming_directional=float(z['gradient_dot'][3])
        if set(q)!={'fractions','row','derivative','minimum_power','next_document_rows'} or q['row'].shape!=(33,8,5,j['heads']) or q['derivative'].shape!=(33,) or not np.array_equal(q['fractions'],np.linspace(0,1,33)):raise ValueError('Wrong node geometry')
        if not np.array_equal(q['next_document_rows'],next_rows) or len(np.unique(next_rows))!=32 or q['minimum_power'].min()<=1e-30:raise ValueError('Invalid source or normalization')
        # This direct projection calculation is separate from automatic differentiation.
        c=A-A.mean(-2,keepdims=True);r=(c*c).mean((-2,-1))/(A*A).mean((-2,-1))
        err=max(err,array_discrepancy(r,original_rows,'raw row field',atol=3e-12))
        for node,old in [(0,0),(16,2),(32,3)]:err=max(err,array_discrepancy(q['row'][node],r[old],'replayed row',atol=3e-12))
        means=q['row'].mean((1,2,3));increment=float(r[3].mean()-r[0].mean());levels=[]
        for n in [1,2,4,8,16,32]:
            values=q['derivative'][::32//n];estimate=float((values.sum()-.5*(values[0]+values[-1]))/n)
            levels.append(dict(intervals=n,integrated_increment=estimate,absolute_error=abs(estimate-increment)))
        estimate=levels[-1]['integrated_increment'];simpson=(4*estimate-levels[-2]['integrated_increment'])/3
        cells.append(dict(**j,incoming_directional_estimate=incoming_directional,incoming_row=float(means[0]),endpoint_row=float(means[-1]),observed_increment=increment,
            levels=levels,integrated_increment=estimate,predicted_endpoint=float(means[0])+estimate,absolute_error=abs(estimate-increment),
            relative_increment_error=abs(estimate-increment)/max(abs(increment),1e-30),simpson_increment=simpson,simpson_absolute_error=abs(simpson-increment),
            target_met=bool(abs(estimate-increment)<=p['row_absolute_target']),transport_replay_error=m['transport_replay_error'],
            minimum_power=float(q['minimum_power'].min()),node_rows=means.tolist(),node_derivatives=q['derivative'].tolist()))
        calls+=m['native_forwards'];seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes'])
    return dict(status='passed',schema='collective-row-quadrature-analysis-v1',protocol_sha256=ph,parent_protocol_sha256=bh,checked_sha256=checked,
        cells=cells,scientific_updates=8,initial_transport_forwards=40,optimizer_replay_updates=8,quadrature_forwards=calls,
        row_absolute_target=p['row_absolute_target'],worker_seconds=seconds,peak_allocated_bytes=peak,maximum_reconstruction_error=err,
        analysis_source_sha256=sha256(__file__),scope='All eight local parameter-path transports and all six nested quadrature levels. Saved native derivatives are integrated on CPU and compared with independent raw-matrix endpoint observations. Neither a uniform derivative bound nor autonomous reduced training is inferred.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--parent',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=analyze(a.study,a.parent);write_json(a.output,r)
    for c in r['cells']:print({k:c[k] for k in ['heads','control','incoming_row','observed_increment','integrated_increment','absolute_error','relative_increment_error','simpson_absolute_error','target_met']})
