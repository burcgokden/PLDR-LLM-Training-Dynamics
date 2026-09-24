#!/usr/bin/env python
"""Measure active Adam denominator scales in the recorded native state family."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import generator_parameter
from model_rg.provenance import sha256, write_json


def moment_groups(checkpoint):
    optimizer=checkpoint['optimizer'];states=optimizer['state'];model=checkpoint['model']
    # Native state_dict preserves named-parameter insertion order. Non-parameter
    # buffers do not belong to any optimizer group and are excluded below.
    buffers=('rotary_emb.inv_freq','cos_cached','sin_cached')
    candidates=[name for name in model if not any(part in name for part in buffers)]
    group_names=[[name for name in candidates if generator_parameter(name)],
                 [name for name in candidates if not generator_parameter(name)]]
    groups=defaultdict(lambda:dict(coordinates=0,active_coordinates=0,zero_second_moment=0,
        active_below_epsilon=0,active_second_sum=0.,adaptive_ratio_squared_sum=0.,nonzero_first_at_zero_second=0,
        decay_squared_sum=0.,active_adaptive_below_decay=0))
    for names,group in zip(group_names,optimizer['param_groups'],strict=True):
        if len(names)!=len(group['params']):raise AssertionError('Recorded parameter order does not match optimizer groups')
        for name,index in zip(names,group['params'],strict=True):
            state=states[index];first=state['exp_avg'];second=state['exp_avg_sq'];counter=int(state['step'])
            if tuple(first.shape)!=tuple(model[name].shape) or counter!=checkpoint['step']:raise AssertionError('Optimizer parameter identity or counter changed')
            if tuple(first.shape)!=tuple(second.shape) or (second<0).any():raise AssertionError('Invalid moment state')
            label='shared_metric' if 'reslayerAs' in name else ('head_power' if generator_parameter(name) else 'remaining')
            out=groups[label];n=first.numel();active=second>0;active_count=int(active.sum())
            corrected=second.double()/(1-group['betas'][1]**counter);scale=corrected.sqrt()
            ratio=(first.double()/(1-group['betas'][0]**counter))/(scale+group['eps'])
            out['coordinates']+=n;out['active_coordinates']+=active_count;out['zero_second_moment']+=n-active_count
            out['active_below_epsilon']+=int((active & (scale<group['eps'])).sum())
            out['active_second_sum']+=float(corrected.sum());out['adaptive_ratio_squared_sum']+=float(ratio.square().sum())
            out['nonzero_first_at_zero_second']+=int(((~active)&(first!=0)).sum())
            decay=model[name].double()*group['weight_decay']
            out['decay_squared_sum']+=float(decay.square().sum())
            out['active_adaptive_below_decay']+=int((active & (ratio.abs()<decay.abs())).sum())
    for out in groups.values():
        n=out['coordinates'];active=out['active_coordinates']
        out['zero_second_fraction']=out['zero_second_moment']/n
        out['active_below_epsilon_fraction']=out['active_below_epsilon']/active if active else None
        out['active_gradient_scale_rms']=(out['active_second_sum']/active)**.5 if active else None
        out['adaptive_ratio_rms']=(out['adaptive_ratio_squared_sum']/n)**.5
        out['decay_rms']=(out['decay_squared_sum']/n)**.5
        out['adaptive_decay_norm_ratio']=(out['adaptive_ratio_squared_sum']/out['decay_squared_sum'])**.5 if out['decay_squared_sum']>0 else None
        out['active_adaptive_below_decay_fraction']=out['active_adaptive_below_decay']/active if active else None
    return dict(groups)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--pilot-only',action='store_true');args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906'
    protocol=study/'protocols/optimizer-scales.json';spec=json.loads(protocol.read_text());cases=[c for c in spec['cases'] if not args.pilot_only or c['multiplier']==2]
    inputs=[protocol]
    for case in cases:
        path=Path(case['checkpoint']);inputs.extend([path,path.parent/'manifest.json'])
        if case['step']==(8192 if case['multiplier']==2 else 16384):inputs.append(Path(case['trajectory'])/'measurements.npz')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);bind_run(out,inputs,vars(args));torch.set_num_threads(4);rows=[]
    for case in cases:
        checkpoint=torch.load(case['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        if checkpoint['step']!=case['step']:raise AssertionError('Optimizer horizon changed')
        for key in ['heads','seed','multiplier']:
            if checkpoint['arguments'][key]!=case[key]:raise AssertionError('Optimizer state identity changed')
        for group in checkpoint['optimizer']['param_groups']:
            if group['eps']!=1e-8 or tuple(group['betas'])!=(.9,.95) or group['weight_decay']!=.01:raise AssertionError('Native optimizer constants changed')
        result=dict(case=case,groups=moment_groups(checkpoint));del checkpoint
        if case['step']==(8192 if case['multiplier']==2 else 16384):
            with np.load(Path(case['trajectory'])/'measurements.npz') as raw:
                norms=raw['gradient_norms'][-256:].astype(float)
                np.testing.assert_allclose(norms[:,0]**2,np.sum(norms[:,1:]**2,axis=1),rtol=5e-6,atol=1e-9)
                result['last_256_clipping']=dict(median_total_norm=float(np.median(norms[:,0])),
                    median_total_norm_per_sqrt_heads=float(np.median(norms[:,0])/np.sqrt(case['heads'])),
                    fraction_clipped=float(np.mean(norms[:,0]+1e-6>1)),
                    median_generator_norm=float(np.median(norms[:,1])),median_remaining_norm=float(np.median(norms[:,2])))
        rows.append(result);print(case['name'],result['groups']['shared_metric']['active_below_epsilon_fraction'],flush=True)
    write_json(out/'results.json',dict(schema='native-optimizer-scale-analysis-v1',cases=rows,pilot_only=args.pilot_only,interpretation=spec['interpretation']))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json')))


if __name__=='__main__':main()
