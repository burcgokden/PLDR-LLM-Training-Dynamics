#!/usr/bin/env python3
"""Reduce all state-transfer observations; retain every policy and context."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict, array_discrepancy, finite_array
from analyze_operator_cache import paired_metrics
from cache_risk_validation import real_array
from cache_state_contract import (validate_protocol, inventory, validate_arrays, CONTRACT,
                                 ANALYSIS_SOURCES as SOURCES, digest)

REPO=Path(__file__).resolve().parents[1]


def analyze(study,output):
    if output.exists():raise FileExistsError(output)
    p=validate_protocol(study);ph=sha256(study/'protocol.json')
    checked, admitted_counts, admitted_seconds, admitted_peak = inventory(study,p)
    validate_arrays(study,p)
    def bind(path,h):
        if sha256(path)!=h:raise ValueError('Changed input '+str(path))
        checked[str(path.relative_to(study))]=h
    bind(study/'inputs.npz',p['selection_sha256'])
    for name,h in p['source_sha256'].items():bind(study/'executed-source'/name,h)
    counts={k:0 for k in p['expected_calls']};seconds=0.;peak=0;manifests={}
    for job in p['jobs']:
        folder=study/'runs'/job['run_id'];path=folder/'manifest.json';m=load_json_strict(path.read_text());bind(path,sha256(path))
        if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=ph or m['training_updates']!=0:raise ValueError('Invalid run')
        if (m['initial_parameter_sha256'],m['initial_shared_sha256'])!=(job['initial_parameter_sha256'],job['initial_shared_sha256']):raise ValueError('Initial identity differs')
        expected=[dict(state=s,length=l,own_replay=True,restoration=True,native_generator_calls=40,cached_generator_calls=0) for s in p['states'] for l in p['prefix_lengths']]
        if m['qualification']!=expected or not m['longest_batch_one']:raise ValueError('Incomplete native qualification')
        for name,h in m['artifacts'].items():bind(folder/name,h)
        for k in counts:counts[k]+=m['calls'][k]
        seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes']);manifests[job['run_id']]=m
    if counts!=p['expected_calls'] or seconds>p['worker_hour_cap']*3600 or peak>=p['memory_ceiling_bytes']:raise ValueError('Accounting or resource gate failed')
    with np.load(study/'inputs.npz',allow_pickle=False) as a:blocks=a['blocks']
    if blocks.shape!=(80,129) or blocks.dtype.kind not in 'iu':raise ValueError('Wrong block geometry')
    cells=[];max_identity=0.
    for heads in [8,24]:
        jobs=sorted([j for j in p['jobs'] if j['heads']==heads],key=lambda j:j['seed'])
        if len(jobs)!=6 or len({j['seed'] for j in jobs})!=6:raise ValueError('Incomplete seed panel')
        for state in p['states']:
            for length in p['prefix_lengths']:
                for policy in (['recalibrated'] if state=='initial' else ['recalibrated','initial_cache']):
                    native=[];cached=[];stats={k:[] for k in ['scatter','displacement','risk','drift','initial_displacement','cross','recalibrated_displacement']};energy=[]
                    for job in jobs:
                        folder=study/'runs'/job['run_id']
                        x=real_array(np.load(folder/f'{state}-L{length}-native.npy',allow_pickle=False),'native',(64,32000))
                        y=real_array(np.load(folder/f'{state}-L{length}-{policy}.npy',allow_pickle=False),'cached',(64,32000))
                        native.append(x);cached.append(y)
                        dz=x-y;dz-=dz.mean(-1,keepdims=True);energy.append(float(np.mean(np.sum(dz*dz,axis=-1))))
                        with np.load(folder/'caches.npz',allow_pickle=False) as ca, np.load(folder/f'{state}-L{length}-operators.npz',allow_pickle=False) as op, np.load(folder/f'initial-L{length}-operators.npz',allow_pickle=False) as ini:
                            mean=real_array(op['mean'],'mean',(5,heads,64,64));m0=real_array(ini['mean'],'initial mean',(5,heads,64,64))
                            c0=real_array(ca[f'initial-L{length}'],'initial cache',(5,3,1,heads,64,64))[:,2,0]
                            ct=real_array(ca[f'{state}-L{length}'],'state cache',(5,3,1,heads,64,64))[:,2,0]
                            for k in ['scatter','displacement','risk']:
                                key=k if k=='scatter' else policy+'-'+k
                                value=real_array(op[key],key,(5,))
                                if np.any(value<0):raise ValueError('Negative squared statistic')
                                stats[k].append(value)
                            delta=mean-m0;offset=m0-c0
                            stats['drift'].append(np.mean(delta*delta,axis=(1,2,3)))
                            stats['initial_displacement'].append(np.mean(offset*offset,axis=(1,2,3)))
                            stats['cross'].append(2*np.mean(delta*offset,axis=(1,2,3)))
                            stats['recalibrated_displacement'].append(np.mean((mean-ct)**2,axis=(1,2,3)))
                            max_identity=max(max_identity,array_discrepancy(stats['risk'][-1],stats['scatter'][-1]+stats['displacement'][-1],'cache budget',atol=3e-12))
                            if policy=='initial_cache':
                                max_identity=max(max_identity,array_discrepancy(stats['displacement'][-1],stats['drift'][-1]+stats['initial_displacement'][-1]+stats['cross'][-1],'state transport',atol=3e-12))
                    cell=paired_metrics(np.stack(native),np.stack(cached),blocks[16:,length]);del native,cached
                    cell.update(heads=heads,state=state,updates=0 if state=='initial' else 4096,control=None if state=='initial' else float(state[1:]),length=length,policy=policy,seeds=[j['seed'] for j in jobs],contexts=64)
                    cell.update(operator_by_seed_layer={k:np.asarray(v).tolist() for k,v in stats.items()},operator_means={k:float(np.mean(v)) for k,v in stats.items()},centered_logit_energy_by_seed=energy)
                    cell['empirical_gain_by_seed']=[float(np.sqrt(e/np.sum(r))) if np.sum(r)>0 else None for e,r in zip(energy,stats['risk'])]
                    cell['targets_met']=bool(cell['relative_centered_rms']<=.25 and cell['mean_kl']<=.03)
                    cells.append(cell)
                    print({k:cell[k] for k in ['heads','state','length','policy','relative_centered_rms','mean_kl','targets_met']},flush=True)
    result=dict(status='passed',schema='cache-state-transfer-analysis-v2',contract=CONTRACT,
        coverage_sha256=digest(checked),external_sha256=p['input_sha256'],cells=cells,total_cells=len(cells),calls=counts,training_updates=0,new_training_replicas=0,
        worker_seconds=seconds,peak_allocated_bytes=peak,maximum_identity_error=max_identity,protocol_sha256=ph,checked_sha256=checked,
        analysis_sources_sha256={n:sha256(REPO/n) for n in SOURCES},scope=p['conditioning'])
    if len(cells)!=30:raise ValueError('Incomplete cell grid')
    write_json(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();analyze(a.study.resolve(),a.output.resolve())
