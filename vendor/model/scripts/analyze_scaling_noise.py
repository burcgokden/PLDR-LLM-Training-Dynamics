#!/usr/bin/env python
"""Reconstruct conditional drift, local response and forcing-fit diagnostics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def relative_error(x,y):
    return float(np.linalg.norm(x-y)/max(np.linalg.norm(y),1e-30))


def delta_kl(logits,delta):
    logp=logits-logsumexp(logits,axis=-1,keepdims=True)
    probability=np.exp(logp)
    centered=delta-np.sum(probability*delta,axis=-1,keepdims=True)
    if np.max(np.abs(centered))>50:
        return logsumexp(logp+delta,axis=-1)-np.sum(probability*delta,axis=-1)
    small=np.abs(centered)<1e-3
    residual=np.empty_like(centered)
    x=centered[small]
    residual[small]=x*x*(.5+x*(1/6+x*(1/24+x*(1/120+x/720))))
    residual[~small]=np.expm1(centered[~small])-centered[~small]
    mean=np.sum(probability*centered,axis=-1)
    return np.log1p(mean+np.sum(probability*residual,axis=-1))-mean


def reconstruct_case(path, raw_out):
    meta=json.loads((path/'manifest.json').read_text())
    result=json.loads((path/'results.json').read_text())
    for filename,key in [('measurements.npz','raw_sha256'),('generator-velocities.npy','velocities_sha256'),
                         ('results.json','results_sha256'),('binding.json','binding_sha256')]:
        if sha256(path/filename)!=meta[key]:raise AssertionError('Bound conditional measurement changed')
    velocity=np.load(path/'generator-velocities.npy',mmap_mode='r')
    count=velocity.shape[0]
    if count!=32:raise AssertionError('Fresh batch count changed')
    group_records=[]
    for target in result['groups']:
        group=target['group'];mean2=second=trace=0.;coordinates=0
        for name,(lo,hi) in zip(result['parameter_names'],result['parameter_slices'],strict=True):
            if group!='all_generator' and ('reslayerAs' in name)!=(group=='shared_metric_network'):continue
            values=np.asarray(velocity[:,lo:hi])
            second+=float(np.mean(values*values,axis=0).sum())
            mean2+=float((values.mean(0)**2).sum())
            trace+=float(np.var(values,axis=0,ddof=1).sum())
            coordinates+=hi-lo
        calculated=dict(second_raw_moment=second,squared_sample_mean=mean2,unbiased_noise_trace=trace,
                        unbiased_squared_drift=mean2-trace/count)
        for key,value in calculated.items():np.testing.assert_allclose(value,target[key],rtol=1e-12,atol=1e-12)
        group_records.append(dict(group=group,coordinates=coordinates,**calculated,
            squared_drift_to_noise_trace=(mean2-trace/count)/trace if trace>0 else None,
            second_moment_per_coordinate=second/coordinates,
            noise_trace_per_coordinate=trace/coordinates))
    del velocity
    derivatives=[];forcing=[];primal_errors=[];arithmetic={}
    with np.load(path/'measurements.npz') as z:
        base=z['base_fields'];logits=z['base_logits']
        probability=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
        normalized=z['base_normalized_rows']
        amplitudes=sorted(float(x) for x in z['amplitudes'] if x>0)
        vectors=z['projected_common_forcing']
        for label in ['native_base_fields','dtype_base_fields']:
            arithmetic[label]=dict(relative_l2=relative_error(z[label],base),
                                   max_absolute=float(np.max(np.abs(z[label]-base))))
        for batch in range(8):
            first=z[f'b{batch}_jvp_fields'];half_second=z[f'b{batch}_second_fields']/2
            tangent=z[f'b{batch}_jvp_normalized_rows'];v=z[f'b{batch}_jvp_logits']
            centered=v-np.sum(probability*v,axis=-1,keepdims=True)
            fisher=np.sum(probability*centered**2,axis=-1)
            raw_out[f'{path.name}_b{batch}_fisher']=fisher
            primal_errors.append(dict(batch=batch,
                fields=float(np.max(np.abs(z[f'b{batch}_jvp_primal_fields']-base))),
                logits=float(np.max(np.abs(z[f'b{batch}_jvp_primal_logits']-logits)))))
            for amplitude in amplitudes:
                plus=z[f'b{batch}_a{amplitude}_fields'];minus=z[f'b{batch}_a{-amplitude}_fields']
                odd=(plus-minus)/(2*amplitude);even=(plus+minus-2*base)/(2*amplitude**2)
                kp=z[f'b{batch}_a{amplitude}_kl'];km=z[f'b{batch}_a{-amplitude}_kl']
                for value,sign in [(kp,amplitude),(km,-amplitude)]:
                    recomputed=delta_kl(logits,z[f'b{batch}_a{sign}_logit_delta'])
                    np.testing.assert_allclose(value,recomputed,rtol=1e-7,atol=2e-14)
                curvature=(kp+km)/amplitude**2
                item=dict(batch=batch,amplitude=amplitude,
                    first_relative_error=relative_error(odd,first),
                    half_second_relative_error=relative_error(even,half_second),
                    first_field_relative_errors=[relative_error(odd[...,i],first[...,i]) for i in range(5)],
                    half_second_field_relative_errors=[relative_error(even[...,i],half_second[...,i]) for i in range(5)],
                    fisher_relative_error=relative_error(curvature,fisher),
                    mean_fisher=float(fisher.mean()),mean_secant_curvature=float(curvature.mean()),
                    mean_positive_kl=float(kp.mean()),mean_negative_kl=float(km.mean()))
                item['local_response_diagnostic']=bool(item['first_relative_error']<=.01 and
                    item['half_second_relative_error']<=.01 and item['fisher_relative_error']<=.05)
                producer=next(r for r in result['response_moments'] if r['batch']==batch and r['amplitude']==amplitude)
                np.testing.assert_allclose(item['first_relative_error'],producer['odd_relative_error'],rtol=1e-12,atol=1e-15)
                np.testing.assert_allclose(item['half_second_relative_error'],producer['even_relative_error'],rtol=1e-12,atol=1e-15)
                derivatives.append(item)
            for layer in range(5):
                vector=vectors[batch,layer]
                record=dict(batch=batch,layer=layer)
                for label,selection in [('calibration',slice(0,8)),('validation',slice(8,16))]:
                    x=normalized[selection,layer];y=tangent[selection,layer]
                    predicted=vector-vector.mean()-x*np.sum(x*vector,axis=-1,keepdims=True)
                    residual=float(np.sum((predicted-y)**2));total=float(np.sum(y*y))
                    target=result['common_tangent_fits'][batch]['layers'][layer]
                    np.testing.assert_allclose(residual,target[label+'_residual_squared'],rtol=1e-11,atol=1e-15)
                    np.testing.assert_allclose(total,target[label+'_tangent_squared'],rtol=1e-11,atol=1e-15)
                    record[label+'_residual_fraction']=residual/total if total else None
                forcing.append(record)
        mean=vectors.mean(0);noise=vectors-mean
        raw_out[path.name+'_common_forcing_vectors']=vectors
        raw_out[path.name+'_common_forcing_covariance']=np.einsum('bld,ble->lde',noise,noise)/7
    return dict(case=meta['case'],groups=group_records,derivatives=derivatives,forcing=forcing,
        primal_errors=primal_errors,paired_arithmetic=arithmetic,native_update_control=result['native_update_control'])


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True);p.add_argument('--study',default='critical-scaling-20260906')
    p.add_argument('--output',required=True);p.add_argument('--wait',action='store_true')
    a=p.parse_args();study=Path(a.root)/a.study
    launcher=study/'launcher-conditional-noise-jvp.json'
    while True:
        status=json.loads(launcher.read_text())['status']
        if status=='complete':break
        if not a.wait or status!='running':raise RuntimeError('Conditional noise executor: '+status)
        time.sleep(30)
    protocol=study/'protocols/conditional-noise-jvp.json'
    spec=json.loads(protocol.read_text());paths=[study/'measurements'/c['name'] for c in spec['cases']]
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    inputs=[protocol,launcher]
    for path in paths:
        inputs.extend(path/name for name in ['manifest.json','results.json','measurements.npz','generator-velocities.npy'])
    bind_run(out,inputs,vars(a))
    raw={};records=[]
    for path in paths:
        record=reconstruct_case(path,raw);records.append(record)
        print(path.name,'reconstructed',flush=True)
    summaries=[]
    groups=defaultdict(list)
    for record in records:groups[record['case']['heads'],record['case']['step']].append(record)
    for (n,t),items in sorted(groups.items()):
        if sorted(x['case']['seed'] for x in items)!=list(range(640101,640105)):
            raise AssertionError('Conditional-noise seed panel changed')
        local={}
        for amplitude in sorted({d['amplitude'] for x in items for d in x['derivatives']}):
            selected=[d for x in items for d in x['derivatives'] if d['amplitude']==amplitude]
            local[str(amplitude)]=dict(responses=len(selected),
                meeting_local_diagnostic=sum(x['local_response_diagnostic'] for x in selected),
                first_error_median=float(np.median([x['first_relative_error'] for x in selected])),
                half_second_error_median=float(np.median([x['half_second_relative_error'] for x in selected])),
                fisher_error_median=float(np.median([x['fisher_relative_error'] for x in selected])))
        summaries.append(dict(heads=n,step=t,seed_count=4,locality=local,
            shared_drift_to_noise_by_seed=[next(g for g in x['groups'] if g['group']=='shared_metric_network')['squared_drift_to_noise_trace'] for x in items],
            heldout_forcing_residual_median_by_layer=[float(np.median([f['validation_residual_fraction']
                for x in items for f in x['forcing'] if f['layer']==layer])) for layer in range(5)]))
    result=dict(schema='conditional-noise-reconstruction-v1',status='complete',states=records,conditions=summaries,
        diagnostic='First and half-second field derivatives within 1 percent and symmetric KL curvature within 5 percent define a numerical local-response diagnostic, not a statistical confidence set.',
        units='Each state has 32 IID fresh batches for conditional Adam moments and eight of those batches for paired emission. Four initialization identities form a matched panel. These unit counts are not interchangeable.',
        scope='All 24 selected states and all 13 positive magnitudes retained. Conditional incoming-moment drift does not exclude a separately justified fast-memory homogenization. A common forcing fit does not certify isotropy, closure, or a native critical mechanism.')
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),states=len(records)))


if __name__=='__main__':main()
