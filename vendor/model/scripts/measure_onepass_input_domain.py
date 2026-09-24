#!/usr/bin/env python
"""Capture the native shared-row transport over the declared proper-prefix domain."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def summaries(raw,lengths):
    full=raw['L64_stages'].astype(float);records=[]
    for length in lengths:
        x=raw[f'L{length}_stages'].astype(float)
        row=np.mean((x-x.mean(-2,keepdims=True))**2,axis=(0,3,4,5))
        context=np.mean((x-x.mean(0,keepdims=True))**2,axis=(0,3,4,5))
        separation=np.mean((x-full)**2,axis=(0,3,4,5))
        ratio=lambda e:[float(np.sqrt(v[-1]/v[0])) if v[0]>0 else None for v in e]
        records.append(dict(length=length,row_energy=row.tolist(),context_energy=context.tolist(),
            prefix_separation_energy=separation.tolist(),row_transport_factor=ratio(row),
            context_transport_factor=ratio(context),prefix_transport_factor=ratio(separation)))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/onepass-input-domain-selection.json';spec=json.loads(selection.read_text())
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('A selected input-domain producer changed')
    for name,digest in spec['inputs_sha256'].items():
        if sha256(name)!=digest:raise AssertionError('A selected input-domain observation changed')
    ledger=study/'launcher-onepass-input-domain.json'
    if ledger.exists():raise FileExistsError(ledger)
    records=[];write_json(ledger,dict(status='running',selection_sha256=sha256(selection),records=records))
    torch.set_num_threads(4)
    for case in spec['cases']:
        for name,digest in case['input_sha256'].items():
            if sha256(name)!=digest:raise AssertionError('A selected input-domain checkpoint changed')
        reference=study/'measurements'/case['reference_prefix']
        with np.load(reference/'measurements.npz') as z:crops=z['crops'].copy();rows=z['rows'].copy()
        if rows.tolist()!=spec['rows']:raise AssertionError('Selected input-domain contexts changed')
        out=study/'measurements'/case['name'];out.mkdir(parents=True,exist_ok=False)
        bind_run(out,[selection,reference/'manifest.json',reference/'measurements.npz',*map(Path,case['input_sha256'])],vars(a))
        before=time.time();saved=torch.load(case['state'],map_location='cpu',mmap=True,weights_only=True)
        if saved['step']!=case['step']:raise AssertionError('Input-domain checkpoint time changed')
        model=TrainingModel(case['source'],saved['arguments']['heads'],saved['arguments']['seed'],'cpu')
        model.model.load_state_dict(saved['model']);del saved
        model.model.eval().requires_grad_(False);captured={};handles=[]
        def hook(name):
            def retain(module,args,output):captured[name]=output.detach().numpy().copy()
            return retain
        for layer,decoder in enumerate(model.model.decoder.dec_layers):
            handles.append(decoder.mha1.layernorm1.register_forward_hook(hook((layer,0))))
            for index,unit in enumerate(decoder.mha1.reslayerAs,1):handles.append(unit.register_forward_hook(hook((layer,index))))
        raw=dict(rows=rows,crops=crops);elements=0
        with torch.no_grad(), np.load(reference/'measurements.npz') as z:
            for length in spec['lengths']:
                captured.clear();model.eta=None;model.capture=False;model.head_outputs={}
                output=model.model(torch.tensor(crops[:,:length],dtype=torch.long),use_cache=False,logits_to_keep=0,output_pldr_attentions=True)
                if len(captured)!=45:raise AssertionError('A decoder or shared row unit was omitted')
                stages=np.stack([np.stack([captured[l,j] for j in range(9)],axis=1) for l in range(5)],axis=1)
                values=dict(A=stages[:,:,8],G=np.stack([v[5].numpy().copy() for v in output.pldr_attentions],axis=1),
                            logits=output.logits[:,-1].numpy().copy())
                for name,value in values.items():
                    expected=z[f'L{length}_prefix_'+name]
                    if value.dtype!=expected.dtype or value.shape!=expected.shape or value.tobytes()!=expected.tobytes():
                        raise AssertionError('Native instrumentation changed a verified prefix endpoint')
                    elements+=value.size
                raw[f'L{length}_stages']=stages
                raw[f'L{length}_G']=values['G'];raw[f'L{length}_logits']=values['logits']
        for handle in handles:handle.remove()
        result=dict(status='complete',case=case,lengths=summaries(raw,spec['lengths']),
            native_prefix_replay_byte_equal=True,native_endpoint_elements=elements,
            scope='Eight fixed held-out contexts at each of four proper-prefix lengths. Native normalized-Gram and all eight residual stages are retained at every decoder/head. Finite row/context/separation factors are descriptive input-domain observations, not uniform derivative bounds or training-critical eigenvalues.')
        np.savez_compressed(out/'measurements.npz',**raw);write_json(out/'results.json',result)
        write_json(out/'manifest.json',dict(status='complete',case=case,binding_sha256=sha256(out/'binding.json'),
            results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),seconds=time.time()-before))
        records.append(dict(name=case['name'],manifest_sha256=sha256(out/'manifest.json')))
        write_json(ledger,dict(status='running',selection_sha256=sha256(selection),records=records))
        print('Captured and replayed every selected input-domain endpoint',case['name'],flush=True)
        del model,raw
    write_json(ledger,dict(status='complete',selection_sha256=sha256(selection),records=records))


if __name__=='__main__':main()
