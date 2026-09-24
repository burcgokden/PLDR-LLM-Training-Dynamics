#!/usr/bin/env python
"""Test mean-preserving internal row projections in the complete predictive graph."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import observations, predictive_kl, parameter_digest, fix_shared_generator
from model_rg.native import NativeModel
from model_rg.variance_family import normalize_variance_initialization
from model_rg.provenance import sha256, write_json
from model_rg.row_projection import project_with_exceptions
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case',required=True)
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906';protocol=study/'protocols/row-projection.json'
    spec=json.loads(protocol.read_text());case=next(c for c in spec['cases'] if c['name']==args.case)
    out=study/'analysis'/case['output_name'];out.mkdir(parents=True,exist_ok=False)
    source=Path(case.get('model_dir',root/'assets/PLDR-LLM-v51-SOC-110M-1'));probe=root/'controlled-study-20260905/data/short'
    inputs=[protocol,source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',probe/'tokens.npy',probe/'offsets.npy']
    if case.get('model_dir'):inputs += [source/'model.safetensors',source/'config.json']
    else:
        inputs.append(Path(case['initial_reference']))
        if case['checkpoint']:inputs.append(Path(case['checkpoint']))
    bind_run(out,inputs,vars(args));torch.set_num_threads(4);started=time.time()
    if case.get('model_dir'):model=NativeModel(source,'cpu')
    else:
        model=TrainingModel(source,case['heads'],case['seed'],'cpu')
        if case['checkpoint']:
            saved=torch.load(case['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
            if saved['step']!=case['step']:raise AssertionError('Projection checkpoint horizon changed')
            model.model.load_state_dict(saved['model']);del saved
        else:
            normalize_variance_initialization(model);fix_shared_generator(model,640011)
            if parameter_digest(model)!=json.loads(Path(case['initial_reference']).read_text())['initial_parameter_sha256']:raise AssertionError('Initial projection state changed')
    model.model.eval().requires_grad_(False)
    tokens=np.load(probe/'tokens.npy');offsets=np.load(probe/'offsets.npy');rows=np.arange(*spec['rows'])
    crops=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    raw=dict(cohort=rows,crops=crops);records=[];baseline=None;broadcast=None
    def evaluate():
        fs=[];hs=[];zs=[]
        with torch.no_grad():
            for i in range(0,len(rows),32):
                batch=torch.tensor(crops[i:i+32],dtype=torch.long)
                f,h,z,_=observations(model,batch[:,:64],batch[:,64])
                if not all(torch.isfinite(value).all() for value in [f,h,z]):raise FloatingPointError('Nonfinite projected native emission')
                fs.append(f.numpy());hs.append(h.numpy());zs.append(z.numpy())
        return np.concatenate(fs),np.concatenate(hs),np.concatenate(zs)
    for variant in spec['variants']:
        handles=[];indices={};errors={}
        for layer,decoder in enumerate(model.model.decoder.dec_layers):
            units=decoder.mha1.reslayerAs
            if variant.startswith('terminal_'):
                count={'terminal_mean':0,'terminal_keep1':1,'terminal_keep4':4}[variant]
                def terminal(module,inputs,output,layer=layer,count=count):
                    projected,kept=project_with_exceptions(output,count)
                    indices.setdefault(layer,[]).append(kept.cpu().numpy())
                    errors.setdefault(layer,[]).append((output-projected).square().mean((-1,-2)).cpu().numpy())
                    return projected
                handles.append(units[-1].register_forward_hook(terminal))
            elif variant.startswith('initial_mean_'):
                compressed=variant=='initial_mean_compressed'
                def initial(module,inputs,compressed=compressed):
                    matrix=inputs[0][0];mean=matrix.mean(-2,keepdim=True)
                    return ([mean if compressed else mean.expand_as(matrix)],)
                handles.append(units[0].register_forward_pre_hook(initial))
                if compressed:
                    def expand(module,inputs,output):return output.expand(*output.shape[:-2],64,64)
                    handles.append(units[-1].register_forward_hook(expand))
        try:fields,heads,logits=evaluate()
        finally:
            for handle in handles:handle.remove()
        if baseline is None:baseline=logits.copy()
        kl=predictive_kl(torch.tensor(baseline),torch.tensor(logits)).numpy()
        raw.update({variant+'_fields':fields,variant+'_heads':heads,variant+'_logits':logits,variant+'_kl':kl})
        for layer,values in indices.items():raw[variant+f'_L{layer}_kept_indices']=np.concatenate(values)
        for layer,values in errors.items():raw[variant+f'_L{layer}_row_projection_mse']=np.concatenate(values)
        record=dict(variant=variant,mean_kl=float(kl.mean()),maximum_kl=float(kl.max()),mean_nll=float(fields[:,25].mean()),mean_row_energy=float(heads[...,2].mean()))
        if variant=='initial_mean_broadcast':broadcast=logits.copy()
        if variant=='initial_mean_compressed':
            arithmetic=predictive_kl(torch.tensor(broadcast),torch.tensor(logits)).numpy()
            raw['compressed_broadcast_kl']=arithmetic;record['maximum_broadcast_compressed_kl']=float(arithmetic.max())
        records.append(record);print(case['output_name'],record,'seconds',round(time.time()-started,1),flush=True)
    fields,heads,logits=evaluate()
    np.testing.assert_array_equal(logits,baseline);np.testing.assert_array_equal(fields,raw['base_fields']);np.testing.assert_array_equal(heads,raw['base_heads'])
    for key,value in raw.items():
        if not np.isfinite(value).all():raise AssertionError('Nonfinite projection record: '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(schema='model-wide-row-projection-v1',case=case,records=records,seconds=time.time()-started,
        interpretation='Native CPU float32 full-graph predictions under explicit row projections. Terminal exception projections preserve each generated row mean in real arithmetic; their retained indices are stored. Initial-mean variants also approximate the nonlinear centroid flow. Predictive accuracy and preservation of row susceptibility are distinct requirements.'))
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
