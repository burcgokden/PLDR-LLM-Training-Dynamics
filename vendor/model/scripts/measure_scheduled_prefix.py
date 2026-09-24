#!/usr/bin/env python
"""Measure a selected scheduled state under the frozen prefix/suffix interventions."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.native import NativeModel
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908')
    p.add_argument('--case',required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;protocol=study/'protocols/scheduled-prefix-selection.json'
    spec=json.loads(protocol.read_text());probe=root/'controlled-study-20260905/data/short'
    cases=[c for c in spec['cases'] if c['name']==a.case]
    if len(cases)!=1:raise AssertionError('A unique scheduled prefix case is required')
    for path,digest in spec['inputs_sha256'].items():
        if sha256(path)!=digest:raise AssertionError('A selected prefix input changed')
    for path,digest in spec['producer_sources'].items():
        if sha256(Path(__file__).resolve().parents[1]/path)!=digest:raise AssertionError('A frozen prefix producer changed')
    rows=np.array(spec['rows']);pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
    crops=np.asarray(pt[rows[:,None],po[rows,None]+np.arange(65)])
    torch.set_num_threads(4);records=[];ledger=study/('launcher-'+a.case+'.json')
    if ledger.exists():raise FileExistsError(ledger)
    write_json(ledger,dict(status='running',protocol_sha256=sha256(protocol),records=records))
    for case in cases:
        out=study/'measurements'/case['name'];out.mkdir(parents=True,exist_ok=False)
        inputs=[protocol,probe/'tokens.npy',probe/'offsets.npy']
        source=Path(case['source']);inputs+=[source/'modeling_pldrllm.py',source/'configuration_pldrllm.py']
        if case['kind']=='pretrained':inputs+=[source/'model.safetensors',source/'config.json']
        else:inputs+=[Path(case['state']),Path(case['parent_manifest'])]
        for name,signature in case.get('input_sha256',{}).items():
            if sha256(name)!=signature:raise AssertionError('A selected prefix probe input changed')
        parent=json.loads(Path(case['parent_manifest']).read_text())
        if parent['status']!='complete' or sha256(case['state'])!=parent['saved_states'][str(case['step'])]['sha256']:
            raise AssertionError('The selected scheduled prefix checkpoint changed')
        for key in ['heads','seed','recipe','shared_seed','stream_seed']:
            if parent['arguments'][key]!=case[key]:raise AssertionError('The selected prefix condition changed')
        bind_run(out,inputs,vars(a));before=time.time()
        if case['kind']=='pretrained':model=NativeModel(source,'cpu')
        else:
            saved=torch.load(case['state'],map_location='cpu',mmap=True,weights_only=True)
            if saved['step']!=case['step']:raise AssertionError('Selected prefix probe state changed')
            model=TrainingModel(source,saved['arguments']['heads'],saved['arguments']['seed'],'cpu')
            model.model.load_state_dict(saved['model']);del saved
        model.model.eval().requires_grad_(False)
        raw=dict(rows=rows,crops=crops,donor_index=np.roll(np.arange(len(rows)),1));conditions=[]

        @torch.no_grad()
        def evaluate(ids,position):
            model.eta=None;model.capture=False;model.head_outputs={}
            output=model.model(torch.as_tensor(ids,dtype=torch.long),use_cache=False,
                logits_to_keep=0,output_pldr_attentions=True)
            z=output.logits[:,position].numpy().copy()
            aa=np.stack([v[0].numpy().copy() for v in output.pldr_attentions],axis=1)
            gg=np.stack([v[5].numpy().copy() for v in output.pldr_attentions],axis=1)
            if not all(np.isfinite(x).all() for x in [z,aa,gg]):raise FloatingPointError('Nonfinite prefix probe')
            return z,aa,gg

        def retain(label,values):
            for name,value in zip(['logits','A','G'],values,strict=True):raw[label+'_'+name]=value

        for length in spec['prefix_lengths']:
            full=evaluate(crops[:,:64],length-1);retain(f'L{length}_full',full)
            variants={'prefix':crops[:,:length].copy(),'noop':crops[:,:64].copy()}
            if length<64:
                target_swap=crops[:,:64].copy();target_swap[:,length]=crops[raw['donor_index'],length]
                suffix_swap=crops[:,:64].copy();suffix_swap[:,length:]=crops[raw['donor_index'],length:64]
                variants.update(target_swap=target_swap,suffix_swap=suffix_swap)
            target=crops[:,length];z0=torch.from_numpy(full[0]).double();lp0=z0.log_softmax(-1)
            base_nll=-lp0[torch.arange(len(rows)),torch.from_numpy(target).long()]
            for name,ids in variants.items():
                values=evaluate(ids,length-1);label=f'L{length}_{name}';retain(label,values);raw[label+'_inputs']=ids
                zz=torch.from_numpy(values[0]).double();lp=zz.log_softmax(-1)
                nll=-lp[torch.arange(len(rows)),torch.from_numpy(target).long()]
                kl=(lp0.exp()*(lp0-lp)).sum(-1)
                aa=values[1].astype(float);gg=values[2].astype(float);g0=full[2].astype(float)
                row=np.mean((aa-aa.mean(-2,keepdims=True))**2,axis=(-2,-1))/np.maximum(np.mean(aa*aa,axis=(-2,-1)),1e-30)
                gscale=np.mean((gg*gg+g0*g0)/2,axis=(-2,-1))
                gerror=np.sqrt(np.mean((gg-g0)**2,axis=(-2,-1))/np.maximum(gscale,1e-30))
                raw[label+'_kl']=kl.numpy();raw[label+'_nll']=nll.numpy();raw[label+'_row_ratio']=row
                raw[label+'_relative_G_difference']=gerror
                if name=='noop' and not all(x.dtype==y.dtype and x.tobytes()==y.tobytes() for x,y in zip(full,values,strict=True)):
                    raise AssertionError('Identical full inputs did not reproduce native outputs bytewise')
                conditions.append(dict(length=length,variant=name,full_window_nll=float(base_nll.mean()),
                    nll=float(nll.mean()),nll_change=float((nll-base_nll).mean()),mean_predictive_kl=float(kl.mean()),
                    maximum_logit_difference=float(np.max(np.abs(values[0].astype(float)-full[0]))),
                    mean_row_ratio=float(row.mean()),mean_relative_G_difference=float(gerror.mean())))
        np.savez_compressed(out/'measurements.npz',**raw)
        result=dict(schema='scheduled-prefix-mechanism-probe-v1',status='complete',case=case,conditions=conditions,
            scope='Descriptive frozen-state CPU float32 probes. Changing only future tokens at fixed sequence length directly tests the prefix consistency of parallel native predictions. A prefix-only forward also changes the available context length. The next target is outside the prefix and remains fixed when suffix or target input tokens are replaced. These probes do not by themselves establish a training shortcut, reasoning ability or criticality.')
        write_json(out/'results.json',result)
        write_json(out/'manifest.json',dict(status='complete',case=case,binding_sha256=sha256(out/'binding.json'),
            raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json'),seconds=time.time()-before))
        records.append(dict(name=case['name'],manifest_sha256=sha256(out/'manifest.json')))
        write_json(ledger,dict(status='running',protocol_sha256=sha256(protocol),records=records))
        print(case['name'],conditions,flush=True);del model,raw
    write_json(ledger,dict(status='complete',protocol_sha256=sha256(protocol),records=records))


if __name__=='__main__':main()
