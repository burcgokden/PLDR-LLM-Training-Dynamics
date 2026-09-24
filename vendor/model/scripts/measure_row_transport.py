#!/usr/bin/env python
"""Measure centered row transport and nonlinear covariance defects in native metric units."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import observations, parameter_digest, fix_shared_generator
from model_rg.variance_family import normalize_variance_initialization
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--case',required=True);parser.add_argument('--protocol',default='row-transport.json')
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906';protocol=study/'protocols'/args.protocol
    spec=json.loads(protocol.read_text());case=next(c for c in spec['cases'] if c['name']==args.case)
    out=study/'analysis'/case['name'];out.mkdir(parents=True,exist_ok=False)
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';probe=root/'controlled-study-20260905/data/short'
    inputs=[protocol,Path(case['initial_reference']),source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',probe/'tokens.npy',probe/'offsets.npy']
    if case['checkpoint']:inputs.append(Path(case['checkpoint']))
    bind_run(out,inputs,vars(args));torch.set_num_threads(4);started=time.time()
    model=TrainingModel(source,case['heads'],case['seed'],'cpu')
    initial_match=None
    if case['checkpoint']:
        checkpoint=torch.load(case['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        if checkpoint['step']!=case['step']:raise AssertionError('Row-transport horizon changed')
        if any(checkpoint['arguments'][key]!=case[key] for key in ['heads','seed']):raise AssertionError('Row-transport identity changed')
        model.model.load_state_dict(checkpoint['model']);del checkpoint
    else:
        normalize_variance_initialization(model);fix_shared_generator(model,640011)
        initial_match=parameter_digest(model)==json.loads(Path(case['initial_reference']).read_text())['initial_parameter_sha256']
        if not initial_match:raise AssertionError('CPU initialization does not reproduce the recorded initial state')
    model.model.eval();model.model.requires_grad_(False)
    rows=np.arange(*spec['rows']);tokens=np.load(probe/'tokens.npy');offsets=np.load(probe/'offsets.npy')
    crops=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    ids=torch.tensor(crops[:,:64],dtype=torch.long);targets=torch.tensor(crops[:,64],dtype=torch.long)
    captured={};handles=[];units={}
    for layer,decoder in enumerate(model.model.decoder.dec_layers):
        for index,unit in enumerate(decoder.mha1.reslayerAs):
            key=f'L{layer}_U{index}';units[key]=unit
            def hook(module,inputs,output,key=key):captured[key]=(inputs[0][0].detach().clone(),output.detach().clone())
            handles.append(unit.register_forward_hook(hook))
    with torch.no_grad():fields,head_fields,logits,_=observations(model,ids,targets)
    for handle in handles:handle.remove()
    raw=dict(cohort=rows,crops=crops,native_fields=fields.numpy(),native_head_fields=head_fields.numpy(),native_logits=logits.numpy(),radii=np.array(spec['radii']))
    records=[]
    for key,original in units.items():
        native_input,native_output=captured[key];x=native_input.double();y_native=native_output.double()
        unit=copy.deepcopy(original).double().eval().requires_grad_(False)
        mu=x.mean(-2);delta=x-mu.unsqueeze(-2)
        function=lambda row:unit([row])
        with torch.no_grad():
            y=unit([x]);mean_y=y.mean(-2);centered=y-mean_y.unsqueeze(-2);mean_image=unit([mu])
            jacobian=torch.func.vmap(torch.func.jacrev(function))(mu.reshape(-1,64)).reshape(*mu.shape[:-1],64,64)
            linear=delta@jacobian.transpose(-1,-2);residual=centered-linear
            vin=delta.square().mean((-1,-2));vout=centered.square().mean((-1,-2));vlinear=linear.square().mean((-1,-2))
            cross=2*(linear*residual).mean((-1,-2));verror=residual.square().mean((-1,-2))
            torch.testing.assert_close(vout,vlinear+cross+verror,rtol=1e-9,atol=1e-15)
            covariance=delta.transpose(-1,-2)@delta/64
            cross_matrix=linear.transpose(-1,-2)@residual/64
            residual_covariance=residual.transpose(-1,-2)@residual/64
            output_covariance=centered.transpose(-1,-2)@centered/64
            transported=jacobian@covariance@jacobian.transpose(-1,-2)
            torch.testing.assert_close(output_covariance,transported+cross_matrix+cross_matrix.transpose(-1,-2)+residual_covariance,rtol=1e-8,atol=1e-12)
            singular=torch.linalg.svdvals(jacobian)[...,0]
            valid=vin>1e-24;response_valid=vlinear>1e-24
            empirical=torch.where(valid,(vout/vin.clamp_min(1e-300)).sqrt(),0)
            directional=torch.where(valid,(vlinear/vin.clamp_min(1e-300)).sqrt(),0)
            errors=[];absolute_errors=[]
            for radius in spec['radii']:
                output=unit([mu.unsqueeze(-2)+radius*delta]);output-=output.mean(-2,keepdim=True)
                error=(output/radius-linear).square().mean((-1,-2)).sqrt()
                absolute_errors.append(error.numpy());errors.append(torch.where(response_valid,error/vlinear.sqrt().clamp_min(1e-300),0).numpy())
        for label,value in dict(native_input=native_input,native_output=native_output,jacobian=jacobian,
                                 smooth_output=y,linear_centered=linear,centered_residual=residual,
                                 centroid_defect=mean_y-mean_image,variance_input=vin,variance_output=vout,
                                 variance_linear=vlinear,variance_cross=cross,variance_residual=verror,
                                 spectral_norm=singular,empirical_gain=empirical,directional_gain=directional,
                                 input_resolved=valid,response_resolved=response_valid).items():raw[key+'_'+label]=value.numpy()
        raw[key+'_amplitude_relative_errors']=np.array(errors);raw[key+'_amplitude_absolute_errors']=np.array(absolute_errors)
        difference=(y-y_native).square().mean((-1,-2)).sqrt()
        raw[key+'_native_smooth_rms_difference']=difference.numpy()
        records.append(dict(unit=key,input_variance_mean=float(vin.mean()),output_variance_mean=float(vout.mean()),
            linear_variance_mean=float(vlinear.mean()),signed_cross_mean=float(cross.mean()),residual_variance_mean=float(verror.mean()),
            spectral_norm_median=float(singular.median()),spectral_norm_max=float(singular.max()),
            native_smooth_max_rms_difference=float(difference.max()),resolved_input_points=int(valid.sum()),resolved_response_points=int(response_valid.sum()),
            measured_points=valid.numel(),maximum_relative_errors_by_radius=[float(np.max(error[response_valid.numpy()])) if response_valid.any() else None for error in errors]))
        print(case['name'],key,'V',records[-1]['input_variance_mean'],records[-1]['output_variance_mean'],'seconds',round(time.time()-started,1),flush=True)
    for key,array in raw.items():
        if not np.isfinite(array).all():raise AssertionError('Nonfinite row transport: '+key)
    np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'results.json',dict(schema='metric-row-transport-v1',case=case,records=records,initial_parameter_identity=initial_match,seconds=time.time()-started,
        interpretation='Full native CPU float32 states supply the observed row inputs. Each residual metric unit is differentiated in float64 at its actual input row mean. Centered nonlinear residuals and signed covariance cross terms remain in the exact transport identity. Finite row-unit depth, head-count size and training time are distinct scales.'))
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
