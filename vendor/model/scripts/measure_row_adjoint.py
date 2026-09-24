#!/usr/bin/env python
"""Measure loss-adjoint transport at actual native rows and matched training batches."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import sample_batches
from model_rg.provenance import sha256, write_json
from model_rg.row_projection import project_with_exceptions
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case',required=True);parser.add_argument('--protocol',default='row-adjoint.json')
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906';protocol=study/'protocols'/args.protocol
    spec=json.loads(protocol.read_text());case=next(c for c in spec['cases'] if c['name']==args.case)
    out=study/'analysis'/case['name'];out.mkdir(parents=True,exist_ok=False);source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    token_path=root/'data/refinedweb-4608/tokens.npy';inputs=[protocol,Path(case['checkpoint']),token_path,source/'modeling_pldrllm.py',source/'configuration_pldrllm.py']
    bind_run(out,inputs,vars(args));torch.set_num_threads(4);started=time.time()
    checkpoint=torch.load(case['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
    if checkpoint['step']!=case['step']:raise AssertionError('Adjoint horizon changed')
    for key in ['heads','seed']:
        if checkpoint['arguments'][key]!=case[key]:raise AssertionError('Adjoint model identity changed')
    if checkpoint['arguments']['stream_seed']!=640001 or checkpoint['arguments']['multiplier']!=2:raise AssertionError('Adjoint training law changed')
    model=TrainingModel(source,case['heads'],case['seed'],'cpu');model.model.load_state_dict(checkpoint['model']);del checkpoint
    model.model.eval().requires_grad_(True)
    tokens=np.load(token_path);step=case.get('batch_step',case['step'])
    all_rows,all_offsets,all_crops=sample_batches(tokens,640001,step+1)
    rows=all_rows[-1].copy();offsets=all_offsets[-1].copy();crops=all_crops[-1].copy()
    del all_rows,all_offsets,all_crops
    captured={};units={};handles=[]
    for layer,decoder in enumerate(model.model.decoder.dec_layers):
        for index,unit in enumerate(decoder.mha1.reslayerAs):
            key=f'L{layer}_U{index}';units[key]=unit
            def hook(module,inputs,output,key=key):
                x=inputs[0][0];x.retain_grad();output.retain_grad();captured[key]=(x,output)
            handles.append(unit.register_forward_hook(hook))
    batch=torch.tensor(crops,dtype=torch.long);logits=model.logits(batch[:,:64]);loss=torch.nn.functional.cross_entropy(logits,batch[:,64])
    for handle in handles:handle.remove()
    loss.backward()
    gradients=[p.grad for p in model.model.parameters() if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack([g.norm() for g in gradients]));clipping=min(1.,1/(float(norm)+1e-6))
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite native training adjoint')
    raw=dict(rows=rows,offsets=offsets,crops=crops,native_logits=logits.detach().numpy());records=[]
    for key,original in units.items():
        native_x,native_y=captured[key];x=native_x.detach().double().requires_grad_(True);output_adjoint=native_y.grad.detach().double()
        unit=copy.deepcopy(original).double().eval().requires_grad_(False)
        smooth_output=unit([x]);smooth_adjoint=torch.autograd.grad(smooth_output,x,grad_outputs=output_adjoint)[0].detach()
        input_adjoint=native_x.grad.detach().double();vin=input_adjoint.square().sum((-1,-2));vout=output_adjoint.square().sum((-1,-2))
        error=(smooth_adjoint-input_adjoint).square().sum((-1,-2)).sqrt()
        _,indices=project_with_exceptions(native_y.detach(),4)
        mask=torch.zeros(native_y.shape[:-1],dtype=torch.bool);mask.scatter_(-1,indices,True)
        input_row_energy=input_adjoint.square().sum(-1);output_row_energy=output_adjoint.square().sum(-1)
        group_grad=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in original.parameters() if p.grad is not None]))
        resolved=vin>1e-40
        record=dict(unit=key,input_adjoint_energy=float(vin.sum()),output_adjoint_energy=float(vout.sum()),
            aggregate_adjoint_gain=float((vin.sum()/vout.sum()).sqrt()) if float(vout.sum())>0 else None,
            maximum_local_vjp_relative_error=float((error[resolved]/vin[resolved].sqrt()).max()) if resolved.any() else None,
            maximum_local_vjp_absolute_error=float(error.max()),parameter_gradient_norm=float(group_grad),clipped_parameter_gradient_norm=float(group_grad)*clipping,
            retained4_input_adjoint_energy_fraction=float((input_row_energy*mask).sum()/vin.sum()) if float(vin.sum())>0 else None,
            retained4_output_adjoint_energy_fraction=float((output_row_energy*mask).sum()/vout.sum()) if float(vout.sum())>0 else None)
        for label,value in dict(native_input=native_x.detach(),native_output=native_y.detach(),input_adjoint=input_adjoint,
                                 output_adjoint=output_adjoint,smooth_input_adjoint=smooth_adjoint,retained4_indices=indices).items():raw[key+'_'+label]=value.numpy()
        records.append(record);print(case['name'],key,record['aggregate_adjoint_gain'],round(time.time()-started,1),flush=True)
        del unit,x,smooth_output,smooth_adjoint
    for key,value in raw.items():
        if not np.isfinite(value).all():raise AssertionError('Nonfinite adjoint record: '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(schema='native-metric-adjoint-v1',case=case,training_loss=float(loss.detach()),
        total_gradient_norm=float(norm),clipping_factor=clipping,records=records,seconds=time.time()-started,interpretation=spec['interpretation']))
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
