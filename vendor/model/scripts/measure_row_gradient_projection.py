#!/usr/bin/env python
"""Compare native shared loss forces under complete-graph metric row reductions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run, stable_kl
from model_rg.criticality import sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.row_projection import project_with_exceptions
from model_rg.training import TrainingModel


def comparison(base, other):
    a=base.double();b=other.double();difference=b-a
    aa=float(torch.dot(a,a));bb=float(torch.dot(b,b));dd=float(torch.dot(difference,difference));ab=float(torch.dot(a,b))
    return dict(reference_squared_norm=aa,squared_norm=bb,squared_difference=dd,inner_product=ab,
                relative_error=(dd/aa)**.5 if aa>0 else None,
                cosine=ab/(aa*bb)**.5 if aa>0 and bb>0 else None)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case',required=True);parser.add_argument('--output')
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906'
    protocol=study/'protocols/row-gradient-projection.json';spec=json.loads(protocol.read_text());case=next(c for c in spec['cases'] if c['name']==args.case)
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';reference=study/'analysis'/case['adjoint_reference'];token_path=root/'data/refinedweb-4608/tokens.npy'
    inputs=[protocol,Path(case['checkpoint']),token_path,source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',reference/'manifest.json',reference/'results.json',reference/'measurements.npz']
    out=Path(args.output) if args.output else study/'analysis'/case['name'];out.mkdir(parents=True,exist_ok=False)
    bind_run(out,inputs,vars(args));torch.set_num_threads(4);started=time.time()
    checkpoint=torch.load(case['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
    if checkpoint['step']!=case['step'] or any(checkpoint['arguments'][key]!=case[key] for key in ['heads','seed']):raise AssertionError('Gradient checkpoint identity changed')
    if checkpoint['arguments']['stream_seed']!=640001 or checkpoint['arguments']['multiplier']!=2:raise AssertionError('Gradient training family changed')
    model=TrainingModel(source,case['heads'],case['seed'],'cpu');model.model.load_state_dict(checkpoint['model']);del checkpoint
    model.model.eval().requires_grad_(True)
    all_rows,all_offsets,all_crops=sample_batches(np.load(token_path),640001,case['batch_step']+1)
    rows=all_rows[-1].copy();offsets=all_offsets[-1].copy();crops=all_crops[-1].copy();del all_rows,all_offsets,all_crops
    ref=np.load(reference/'measurements.npz');ref_result=json.loads((reference/'results.json').read_text())
    for key,value in [('rows',rows),('offsets',offsets),('crops',crops)]:np.testing.assert_array_equal(value,ref[key])
    batch=torch.tensor(crops,dtype=torch.long);parameters=[(n,p) for n,p in model.model.named_parameters() if 'reslayerAs' in n]
    raw=dict(rows=rows,offsets=offsets,crops=crops);records=[];reference_gradients={};reference_logits=None;layout=[];offset=0
    for name,p in parameters:
        layout.append(dict(name=name,shape=list(p.shape),start=offset,stop=offset+p.numel()));offset+=p.numel()
    for variant in spec['variants']:
        model.model.zero_grad(set_to_none=True);handles=[];captured={}
        count={'terminal_keep4':4,'terminal_mean':0,'terminal_identity':64}.get(variant)
        for layer,decoder in enumerate(model.model.decoder.dec_layers):
            def hook(module,inputs,output,layer=layer,count=count):
                captured[f'L{layer}_terminal_rows']=output.detach().numpy().copy()
                if count is None:return output
                projected,indices=project_with_exceptions(output,count)
                captured[f'L{layer}_indices']=indices.numpy().copy()
                return projected
            handles.append(decoder.mha1.reslayerAs[-1].register_forward_hook(hook))
        try:logits=model.logits(batch[:,:64]);loss=torch.nn.functional.cross_entropy(logits,batch[:,64])
        finally:
            for handle in handles:handle.remove()
        loss.backward()
        gradient=torch.cat([p.grad.detach().reshape(-1) for _,p in parameters]).clone()
        total_norm=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
        clipped=torch.cat([p.grad.detach().reshape(-1) for _,p in parameters]).clone()
        if not torch.isfinite(gradient).all() or not torch.isfinite(clipped).all():raise FloatingPointError('Nonfinite shared loss force')
        if variant=='base':
            reference_logits=logits.detach().clone();reference_gradients={'unclipped':gradient,'clipped':clipped}
            np.testing.assert_array_equal(logits.detach().numpy(),ref['native_logits'])
            np.testing.assert_allclose(float(total_norm),ref_result['total_gradient_norm'],rtol=2e-6,atol=1e-8)
            np.testing.assert_allclose(float(loss.detach()),ref_result['training_loss'],rtol=0,atol=0)
        kl=stable_kl(reference_logits,logits.detach()).numpy()
        record=dict(variant=variant,loss=float(loss.detach()),total_gradient_norm=float(total_norm),mean_kl=float(kl.mean()),maximum_kl=float(kl.max()),
                    unclipped=comparison(reference_gradients['unclipped'],gradient),clipped=comparison(reference_gradients['clipped'],clipped))
        raw.update({variant+'_logits':logits.detach().numpy(),variant+'_gradient':gradient.numpy(),variant+'_clipped_gradient':clipped.numpy(),variant+'_kl':kl})
        for name,value in captured.items():raw[variant+'_'+name]=value
        if variant=='terminal_identity':
            for key in ['logits','gradient','clipped_gradient']:
                np.testing.assert_array_equal(raw[variant+'_'+key],raw['base_'+key])
            np.testing.assert_array_equal(raw[variant+'_kl'],0)
        records.append(record);print(case['name'],variant,record['clipped']['relative_error'],record['mean_kl'],round(time.time()-started,1),flush=True)
        del logits,loss,gradient,clipped
    for key,value in raw.items():
        if not np.isfinite(value).all():raise AssertionError('Nonfinite reduction record: '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(schema='native-row-gradient-projection-v1',case=case,shared_parameter_layout=layout,records=records,seconds=time.time()-started,interpretation=spec['interpretation']))
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
