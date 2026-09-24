#!/usr/bin/env python
"""Measure both last-target and all-target risks at a selected scheduled state."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');p.add_argument('--case',required=True)
    p.add_argument('--selection',default='onepass-observation-selection.json')
    a=p.parse_args();root=Path(a.root).resolve();study=root/a.study
    selection=study/'protocols'/a.selection;spec=json.loads(selection.read_text())
    cases=[c for c in spec['cases'] if c['name']==a.case]
    if len(cases)!=1:raise AssertionError('A unique frozen scheduled observation is required')
    case=cases[0]
    spec=dict(spec,**{k:case[k] for k in ['training_cohort','training_cohort_sha256','training_cohort_manifest']})
    parent=Path(case['parent_manifest']);meta=json.loads(parent.read_text())
    checkpoint=Path(case['state']);reference=study/'measurements'/case['name']/'manifest.json'
    rm=json.loads(reference.read_text())
    if meta['status']!='complete' or rm['status']!='complete':raise AssertionError('Complete native and CPU parents are required')
    if sha256(checkpoint)!=meta['saved_states'][str(case['step'])]['sha256'] or sha256(checkpoint)!=rm['case']['state_sha256']:
        raise AssertionError('The selected checkpoint changed')
    for folder,m in [(parent.parent,meta),(reference.parent,rm)]:
        if sha256(folder/'measurements.npz')!=m['raw_sha256']:raise AssertionError('A complete native or CPU parent changed')
    cohort=Path(spec['training_cohort']);source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    probe=root/'controlled-study-20260905/data/short'
    if sha256(cohort)!=spec['training_cohort_sha256']:raise AssertionError('The frozen training-law cohort changed')
    out=study/'measurements'/('risk-'+case['name'].removeprefix('cpu-'));out.mkdir(parents=True,exist_ok=False)
    inputs=[selection,cohort,Path(spec['training_cohort_manifest']),checkpoint,parent,parent.parent/'measurements.npz',
            reference,reference.parent/'measurements.npz',source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',
            probe/'tokens.npy',probe/'offsets.npy']
    bind_run(out,inputs,vars(a));torch.set_num_threads(4);before=time.time()
    saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
    for key in ['heads','seed','recipe','shared_seed','stream_seed']:
        if saved['arguments'][key]!=case[key] or rm['condition'][key]!=case[key]:raise AssertionError('Risk condition changed: '+key)
    if saved['step']!=case['step']:raise AssertionError('Risk horizon changed')
    model=TrainingModel(source,case['heads'],case['seed'],'cpu');model.model.load_state_dict(saved['model']);del saved
    model.model.eval().requires_grad_(False)
    with np.load(cohort) as z:training=z['crops'].copy()
    pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy');rows=np.arange(512,1024)
    heldout=np.asarray(pt[rows[:,None],po[rows,None]+np.arange(65)])
    raw={};results={}
    with torch.no_grad():
        for label,crops in [('training',training),('heldout',heldout)]:
            last_nll=[];last_entropy=[];all_nll=[];all_entropy=[]
            last_objectives=[];all_objectives=[];last_logits_discrepancy=[]
            for begin in range(0,len(crops),32):
                batch=torch.tensor(crops[begin:begin+32],dtype=torch.long)
                logits=model.forward(batch[:,:64],capture=True).logits[:,-1]
                lp=logits.log_softmax(-1)
                nll=-lp.gather(1,batch[:,64,None]).squeeze(1)
                entropy=-(lp.exp()*lp).sum(-1)
                last_nll.append(nll.numpy());last_entropy.append(entropy.numpy())
                last_objectives.append(float(torch.nn.functional.cross_entropy(logits,batch[:,64])))
                if begin==0:
                    raw[label+'_first_last_logits']=logits.numpy().copy()
                    raw[label+'_first_last_targets']=batch[:,64].numpy().copy()
                # The full-position vocabulary projection is a distinct arithmetic
                # program. Preserve the last-only observation for paired comparison.
                model.eta=None;model.capture=False;model.head_outputs={}
                full=model.model(batch[:,:64],use_cache=False,logits_to_keep=0).logits
                targets=batch[:,1:];mask=targets.ne(0)
                values=torch.nn.functional.cross_entropy(full.transpose(1,2),targets,reduction='none')
                full_lp=full.log_softmax(-1);entropies=-(full_lp.exp()*full_lp).sum(-1)
                if not all(torch.isfinite(v).all() for v in [nll,entropy,values,entropies]):
                    raise FloatingPointError('Nonfinite frozen scheduled risk')
                if not mask.any():raise AssertionError('No valid target in an all-target batch')
                all_nll.append(values.numpy());all_entropy.append(entropies.numpy())
                all_objectives.append(float((values*mask).sum()/mask.sum()))
                difference=(full[:,-1]-logits).double()
                last_logits_discrepancy.append([float(difference.abs().max()),float(difference.square().mean().sqrt())])
                if begin==0:
                    raw[label+'_first_all_logits']=full[:2].numpy().copy()
                    raw[label+'_first_all_targets']=targets[:2].numpy().copy()
                del logits,lp,full,full_lp,values,entropies
            for name,values in [('last_nll',last_nll),('last_entropy',last_entropy),('all_nll',all_nll),('all_entropy',all_entropy)]:
                raw[label+'_'+name]=np.concatenate(values)
            raw[label+'_all_mask']=crops[:,1:]!=0
            raw[label+'_native_last_batch_loss']=np.array(last_objectives)
            raw[label+'_native_all_batch_loss']=np.array(all_objectives)
            raw[label+'_last_projection_discrepancy']=np.array(last_logits_discrepancy)
            mask=raw[label+'_all_mask'];count=int(mask.sum())
            results[label]=dict(contexts=len(crops),valid_all_targets=count,
                last_nll=float(raw[label+'_last_nll'].astype(float).mean()),
                last_entropy=float(raw[label+'_last_entropy'].astype(float).mean()),
                all_token_weighted_nll=float(np.sum(raw[label+'_all_nll'].astype(float)*mask)/count),
                all_token_weighted_entropy=float(np.sum(raw[label+'_all_entropy'].astype(float)*mask)/count),
                native_last_batch_mean_loss=float(np.mean(last_objectives)),
                native_all_batch_mean_loss=float(np.mean(all_objectives)))
    with np.load(reference.parent/'measurements.npz') as z:fields=z['float32_fields']
    errors=np.stack([raw['heldout_last_nll']-fields[:,25],raw['heldout_last_entropy']-fields[:,24]])
    maximum=float(np.max(np.abs(errors)))
    if maximum>1e-5:raise AssertionError('The common CPU risk and collective observation differ')
    with np.load(parent.parent/'measurements.npz') as z:online=z['losses']
    start=max(meta['start_step'],case['step']-spec['risk']['online_window']);end=case['step']
    raw['preceding_online_losses']=online[start-meta['start_step']:end-meta['start_step']]
    result=dict(schema='scheduled-frozen-risk-v1',status='complete',case=case,cohorts=results,
        heldout_reference_maximum_error=maximum,
        heldout_reference_bitwise_equal=all(raw['heldout_last_'+name].dtype==fields[:,column].dtype and
            raw['heldout_last_'+name].tobytes()==np.ascontiguousarray(fields[:,column]).tobytes()
            for name,column in [('nll',25),('entropy',24)]),
        online_window=dict(begin=start,end=end,preupdate_mean_loss=float(raw['preceding_online_losses'].mean())),
        scope='CPU float32 full native forwards, batch32. Last-target observations and all-position vocabulary projections are retained separately. All-token CE masks target0. The training cohort is a frozen uniform sample of 1024 blocks already consumed by this checkpoint, selected without replacement before training. Its membership changes with the checkpoint horizon. Frozen-state risk and evolving online loss are distinct. Neither objective changes the immutable checkpoint.')
    np.savez_compressed(out/'measurements.npz',**raw);write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(schema='scheduled-risk-observation-v1',status='complete',case=case,
        binding_sha256=sha256(out/'binding.json'),raw_sha256=sha256(out/'measurements.npz'),
        results_sha256=sha256(out/'results.json'),seconds=time.time()-before))
    print(case['name'],results,'heldout reference error',maximum,flush=True)


if __name__=='__main__':main()
