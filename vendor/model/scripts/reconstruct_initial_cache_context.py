#!/usr/bin/env python3
"""Reconstruct the paired initial/endpoint diagnostic from retained observations."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
from numerical_validation import load_json_strict,array_discrepancy
from analyze_operator_cache import paired_metrics
from verify_cache_state_transfer import predictive
from cache_state_contract import read

REPO=Path(__file__).resolve().parents[1]
SOURCES=['scripts/reconstruct_initial_cache_context.py','scripts/analyze_operator_cache.py',
         'scripts/verify_cache_state_transfer.py','scripts/verify_cache_risk_study.py',
         'scripts/numerical_validation.py','scripts/cache_state_contract.py','src/model_rg/provenance.py']


def reconstruct(initial,endpoint,output):
    if output.exists():raise FileExistsError(output)
    protocols={t:read(root/'protocol.json') for t,root in [(0,initial),(4096,endpoint)]}
    if sha256(endpoint/'inputs.npz')!=protocols[4096]['selection_sha256']:
        raise ValueError('Changed endpoint target panel')
    with np.load(endpoint/'inputs.npz',allow_pickle=False) as z:targets=z['blocks'][16:,64]
    checked={str(endpoint/'inputs.npz'):sha256(endpoint/'inputs.npz')};cells=[];counts={};maximum=0.
    for heads in [8,24]:
        for time,control in [(0,None),(4096,0),(4096,1.5)]:
            root=initial if time==0 else endpoint;p=protocols[time]
            jobs=sorted([j for j in p['jobs'] if j['heads']==heads and (time==0 or j['control']==control)],key=lambda j:j['seed'])
            if len(jobs)!=6:raise ValueError('Incomplete paired panel')
            xs=[];ys=[]
            for job in jobs:
                folder=root/'runs'/job['run_id'];m=load_json_strict((folder/'manifest.json').read_text())
                if m['status']!='complete' or m['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Unbound observation')
                if time==0:
                    if m['initial_parameter_sha256']!=job['initial_parameter_sha256'] or m['initial_logit_replay_max_error']!=0:raise ValueError('Initial state replay failed')
                    if m['qualification']['cached_generator_calls']!=0 or m['qualification']['native_generator_calls']!=40:raise ValueError('Native bypass qualification missing')
                    for key,value in m['calls'].items():counts[key]=counts.get(key,0)+value
                for path in [root/'protocol.json',folder/'manifest.json']:
                    checked[str(path)]=sha256(path)
                path=folder/'observations.npz'
                if sha256(path)!=m['artifacts']['observations.npz']:raise ValueError('Changed observation')
                checked[str(path)]=sha256(path)
                with np.load(path,allow_pickle=False) as z:xs.append(z['native']);ys.append(z['cached'])
            x=np.stack(xs);y=np.stack(ys);cell=paired_metrics(x,y,targets);independent=predictive(x,y,targets)
            for key,value in independent.items():maximum=max(maximum,array_discrepancy(value,cell[key],key,atol=4e-12,rtol=2e-10))
            cell.update(heads=heads,time=time,control=control,seeds=[j['seed'] for j in jobs]);cells.append(cell)
    contrasts=[]
    for c in cells:
        if c['time']==0:continue
        initial_cell=next(x for x in cells if x['heads']==c['heads'] and x['time']==0)
        if c['seeds']!=initial_cell['seeds']:raise ValueError('Initial/endpoint pairing differs')
        delta=np.array(c['per_seed_mean_kl'])-initial_cell['per_seed_mean_kl']
        contrasts.append(dict(heads=c['heads'],control=c['control'],seeds=c['seeds'],per_seed_kl_final_minus_initial=delta.tolist(),mean_kl_final_minus_initial=float(delta.mean())))
    if counts!=protocols[0]['expected_forward_calls']:raise ValueError('Initial forward ledger differs')
    result=dict(status='passed',schema='cache-initial-context-reconstruction-v1',cells=cells,contrasts=contrasts,maximum_predictive_difference=maximum,
        initial_acquisition_calls=counts,initial_successful_forwards=sum(counts.values()),new_forward_calls=0,new_training_updates=0,new_training_replicas=0,
        checked_sha256=checked,source_sha256={n:sha256(REPO/n) for n in SOURCES})
    write_json(output,result);print({'status':'passed','cells':len(cells),'maximum_difference':maximum,'initial_calls':counts},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--initial',type=Path,required=True);p.add_argument('--endpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();reconstruct(a.initial.resolve(),a.endpoint.resolve(),a.output.resolve())
