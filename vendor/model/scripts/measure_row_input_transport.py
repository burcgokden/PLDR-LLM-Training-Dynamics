#!/usr/bin/env python
"""Locate observed row and suffix contraction inside the shared native row maps."""
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
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    selection=study/'protocols/row-input-transport-selection.json';spec=json.loads(selection.read_text())
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest:raise AssertionError('The row/input producer changed')
    for path,digest in spec['inputs_sha256'].items():
        if sha256(path)!=digest:raise AssertionError('The row/input selection changed')
    ledger=study/'launcher-row-input-transport.json'
    if ledger.exists():raise FileExistsError(ledger)
    records=[];write_json(ledger,dict(status='running',selection_sha256=sha256(selection),records=records))
    probe=root/'controlled-study-20260905/data/short'
    tokens=np.load(probe/'tokens.npy',mmap_mode='r');offsets=np.load(probe/'offsets.npy');rows=np.array(spec['rows'])
    crops=tokens[rows[:,None],offsets[rows,None]+np.arange(64)]
    suffix=crops.copy();suffix[:,32:]=crops[np.roll(np.arange(len(rows)),1),32:]
    torch.set_num_threads(4)
    for case in spec['cases']:
        for path,digest in case['input_sha256'].items():
            if sha256(path)!=digest:raise AssertionError('A frozen row/input state changed')
        out=study/'measurements'/case['name'];out.mkdir(parents=True,exist_ok=False)
        reference=study/'measurements'/case['reference_prefix'];rm=json.loads((reference/'manifest.json').read_text())
        if rm['status']!='complete' or sha256(reference/'measurements.npz')!=rm['raw_sha256']:raise AssertionError('Changed prefix reference')
        bind_run(out,[selection,*map(Path,case['input_sha256']),reference/'manifest.json',reference/'measurements.npz'],vars(a))
        before=time.time()
        if case['kind']=='pretrained':model=NativeModel(case['source'],'cpu')
        else:
            saved=torch.load(case['state'],map_location='cpu',mmap=True,weights_only=True)
            if saved['step']!=case['step']:raise AssertionError('The row/input horizon changed')
            model=TrainingModel(case['source'],saved['arguments']['heads'],saved['arguments']['seed'],'cpu')
            model.model.load_state_dict(saved['model']);del saved
        model.model.eval().requires_grad_(False);captured={};handles=[]
        def hook(name):
            def retain(module,args,output):captured[name]=output.detach().numpy().copy()
            return retain
        for layer,module in enumerate(model.model.decoder.dec_layers):
            handles.append(module.mha1.layernorm1.register_forward_hook(hook((layer,0))))
            for index,unit in enumerate(module.mha1.reslayerAs,1):handles.append(unit.register_forward_hook(hook((layer,index))))
        raw=dict(rows=rows,full_inputs=crops,suffix_inputs=suffix)
        with torch.no_grad():
            for label,ids in [('full',crops),('suffix',suffix)]:
                captured.clear();model.eta=None;model.capture=False
                output=model.model(torch.tensor(ids,dtype=torch.long),use_cache=False,logits_to_keep=0,output_pldr_attentions=True)
                if len(captured)!=45:raise AssertionError('The complete five-layer/eight-unit row path was not captured')
                raw[label+'_stages']=np.stack([np.stack([captured[l,j] for j in range(9)],axis=1) for l in range(5)],axis=1)
                raw[label+'_G']=np.stack([v[5].numpy().copy() for v in output.pldr_attentions],axis=1)
                raw[label+'_logits']=output.logits[:,31].numpy().copy()
        for h in handles:h.remove()
        with np.load(reference/'measurements.npz') as z:
            for label,ref in [('full','L32_full'),('suffix','L32_suffix_swap')]:
                for name,value in [('A',raw[label+'_stages'][:,:,8]),('G',raw[label+'_G']),('logits',raw[label+'_logits'])]:
                    other=z[ref+'_'+name]
                    if value.dtype!=other.dtype or value.shape!=other.shape or value.tobytes()!=other.tobytes():
                        raise AssertionError('Stage capture changed the native prefix emission')
        x=raw['full_stages'].astype(float);y=raw['suffix_stages'].astype(float)
        if not np.isfinite(x).all() or not np.isfinite(y).all():raise FloatingPointError('Nonfinite native row path')
        row_energy=np.mean((x-x.mean(-2,keepdims=True))**2,axis=(0,3,4,5))
        context_energy=np.mean((x-x.mean(0,keepdims=True))**2,axis=(0,3,4,5))
        suffix_energy=np.mean((x-y)**2,axis=(0,3,4,5))
        row_factors=np.sqrt(np.divide(row_energy[:,1:],row_energy[:,:-1],out=np.zeros_like(row_energy[:,1:]),where=row_energy[:,:-1]>0))
        context_factors=np.sqrt(np.divide(context_energy[:,1:],context_energy[:,:-1],out=np.zeros_like(context_energy[:,1:]),where=context_energy[:,:-1]>0))
        suffix_factors=np.sqrt(np.divide(suffix_energy[:,1:],suffix_energy[:,:-1],out=np.zeros_like(suffix_energy[:,1:]),where=suffix_energy[:,:-1]>0))
        raw.update(row_energy=row_energy,context_energy=context_energy,suffix_energy=suffix_energy,
            row_secant_factors=row_factors,context_secant_factors=context_factors,suffix_secant_factors=suffix_factors,
            row_factor_defined=row_energy[:,:-1]>0,context_factor_defined=context_energy[:,:-1]>0,suffix_factor_defined=suffix_energy[:,:-1]>0)
        def overall(energy):
            return [float(np.sqrt(x[-1]/x[0])) if x[0]>0 else None for x in energy]
        result=dict(status='complete',case=case,native_prefix_replay_byte_equal=True,
            stages=['normalized_input_Gram',*[f'residual_unit_{j}' for j in range(1,9)]],
            row_energy=row_energy.tolist(),context_energy=context_energy.tolist(),suffix_energy=suffix_energy.tolist(),
            overall_row_factor=overall(row_energy),
            overall_context_factor=overall(context_energy),
            overall_suffix_factor=overall(suffix_energy),
            scope='Descriptive decomposition of five previously observed input/output states. Stagewise finite-cloud row energy, context variance and paired suffix secants are not uniform Lipschitz constants or critical exponents. Exact endpoint replay qualifies passive capture; Undefined overall factors are null. Zero-denominator stepwise factors use a0 sentinel with an explicit false definition mask; they are not measured zero contractions.')
        np.savez_compressed(out/'measurements.npz',**raw);write_json(out/'results.json',result)
        write_json(out/'manifest.json',dict(status='complete',case=case,binding_sha256=sha256(out/'binding.json'),
            raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json'),seconds=time.time()-before))
        records.append(dict(name=case['name'],manifest_sha256=sha256(out/'manifest.json')))
        write_json(ledger,dict(status='running',selection_sha256=sha256(selection),records=records))
        print(case['name'],'row factors',result['overall_row_factor'],'suffix factors',result['overall_suffix_factor'],flush=True)
        del model,raw,x,y
    write_json(ledger,dict(status='complete',selection_sha256=sha256(selection),records=records))


if __name__=='__main__':main()
