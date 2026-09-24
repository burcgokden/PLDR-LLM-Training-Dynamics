#!/usr/bin/env python3
"""CPU reconstruction of every declared native collective-row transport check."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,finite_array,array_discrepancy


def analyze(study):
    p=load_json_strict((study/'protocol.json').read_text());ph=sha256(study/'protocol.json')
    expected={(n,g,9163401) for n in [4,8,14,24] for g in [1.5,2.]}
    if p['schema']!='collective-row-clock-v1' or len(p['jobs'])!=8 or {(j['heads'],j['control'],j['seed']) for j in p['jobs']}!=expected:raise ValueError('Incomplete design')
    checked={'protocol.json':ph};rows=[];err=0.;calls=updates=0;seconds=0.;peak=0
    for j in p['jobs']:
        folder=study/'runs'/j['run_id'];m=load_json_strict((folder/'manifest.json').read_text());path=folder/'observations.npz'
        if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=ph or m['native_forwards']!=5 or m['scientific_updates']!=1 or not m['endpoint_restored']:raise ValueError('Incomplete native point')
        if m['artifacts']!={'observations.npz':sha256(path)} or m['incoming_row_replay_error']>1e-9:raise ValueError('Foreign observations')
        for name in ['manifest.json','observations.npz']:checked[str((folder/name).relative_to(study))]=sha256(folder/name)
        with np.load(path) as a:
            values={k:finite_array(a[k],k) for k in a.files}
        if set(values)!={'fractions','row','A','minimum_power','gradient_dot','displacement_norm','prediction','next_document_rows'}:raise ValueError('Missing observation role')
        if not np.array_equal(values['fractions'],[0,-.5,.5,1]) or values['A'].shape!=(4,8,5,j['heads'],64,64) or values['row'].shape!=(4,8,5,j['heads']):raise ValueError('Wrong geometry')
        if values['next_document_rows'].shape!=(32,) or len(np.unique(values['next_document_rows']))!=32:raise ValueError('Repeated continuation source')
        if (values['minimum_power']<=1e-30).any():raise ValueError('Degenerate row normalization')
        A=values['A'].astype(float);power=(A*A).mean((-2,-1));center=A-A.mean(-2,keepdims=True);r=(center*center).mean((-2,-1))/power
        err=max(err,array_discrepancy(r,values['row'],'native row reconstruction',atol=3e-12));mean=r.mean((1,2,3));dot=values['gradient_dot']
        err=max(err,array_discrepancy(mean[0]+dot,values['prediction'],'incoming derivative predictions',atol=3e-12))
        half=float(mean[2]-mean[0]);full=float(mean[3]-mean[0]);central=float(mean[2]-mean[1]);d=float(dot[3]);h=1/j['steps']
        remainder_half=float(half-dot[2]);remainder_full=float(full-d)
        rows.append(dict(**j,incoming_row=float(mean[0]),next_row=float(mean[3]),linear_prediction=float(mean[0]+d),
            full_increment=full,autograd_increment=d,central_increment=central,central_relative_difference=abs(central-d)/max(abs(d),1e-30),
            first_order_relative_error=abs(remainder_full)/max(abs(full),1e-30),half_remainder=remainder_half,full_remainder=remainder_full,
            remainder_ratio=abs(remainder_half/remainder_full) if remainder_full else None,
            projected_descent_rate=-d/h,observed_descent_rate=-full/h,displacement_norm=float(values['displacement_norm'][3]),
            minimum_power=float(values['minimum_power'].min()),incoming_replay_error=m['incoming_row_replay_error'],next_document_rows=values['next_document_rows'].tolist()))
        calls+=m['native_forwards'];updates+=m['scientific_updates'];seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes'])
    if calls!=40 or updates!=8:raise ValueError('Wrong native ledger')
    return dict(status='passed',schema='collective-row-clock-analysis-v1',protocol_sha256=ph,checked_sha256=checked,rows=rows,
        scientific_updates=updates,native_forward_calls=calls,worker_seconds=seconds,peak_allocated_bytes=peak,maximum_reconstruction_error=err,
        analysis_source_sha256=sha256(__file__),scope='Eight state-conditioned one-step continuations and symmetric parameter controls, with one retained seed and eight paired contexts per condition. The central secant checks the automatic directional derivative locally; no population fit, uniform Hessian bound, new independent panel or autonomous training closure is inferred.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    r=analyze(a.study);write_json(a.output,r)
    for row in r['rows']:print({k:row[k] for k in ['heads','control','incoming_row','full_increment','projected_descent_rate','first_order_relative_error','central_relative_difference','remainder_ratio']})
    print(dict(updates=r['scientific_updates'],forwards=r['native_forward_calls'],seconds=r['worker_seconds'],peak_gib=r['peak_allocated_bytes']/2**30,error=r['maximum_reconstruction_error']))
