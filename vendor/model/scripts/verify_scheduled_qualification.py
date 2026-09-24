#!/usr/bin/env python
"""Check full GPU checkpoint recovery and the scheduled qualification inventory."""
import argparse
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256, write_json
from verify_scaling_raw import BoundFiles, bitwise_equal, load_json
from verify_scheduled_raw import verify_selection


def identical(left,right):
    if isinstance(left,torch.Tensor):
        return bitwise_equal(left.detach().cpu().numpy(),right.detach().cpu().numpy())
    if isinstance(left,dict):
        return left.keys()==right.keys() and all(identical(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(identical(x,y) for x,y in zip(left,right))
    return left==right


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-20260908');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;repo=Path(__file__).resolve().parents[1]
    output=study/'verification/gpu-qualification.json'
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4);files=BoundFiles();records=verify_selection(root,study,'qualification',files)
    selection=load_json(study/'protocols/regime-qualification-selection.json');comparison=selection['comparison']
    left=study/'runs'/comparison['continuous'];right=study/'runs'/comparison['resumed']
    lm=load_json(left/'manifest.json');rm=load_json(right/'manifest.json')
    if lm['arguments']['device']!=rm['arguments']['device'] or lm['arguments']['device']!='cuda:1':
        raise AssertionError('Native resume qualification changed the GPU')
    continuous=torch.load(left/'final-training-state.pt',map_location='cpu',mmap=True,weights_only=True)
    resumed=torch.load(right/'final-training-state.pt',map_location='cpu',mmap=True,weights_only=True)
    for key in ['model','optimizer','scheduler','supervised_target_counts','recipe','initial_parameter_sha256','step']:
        if not identical(continuous[key],resumed[key]):raise AssertionError('GPU checkpoint recovery differs: '+key)
    counts=dict(parameter_tensors=len(continuous['model']),parameter_elements=sum(x.numel() for x in continuous['model'].values()),
                adam_states=len(continuous['optimizer']['state']))
    del continuous,resumed
    arrays={};tail_fields=['losses','gradient_norms','applied_learning_rates','supervised_targets_per_update',
                           'shared_update_projection','shared_clipped_gradient_projection','shared_update_moments']
    with np.load(left/'measurements.npz') as x,np.load(right/'measurements.npz') as y:
        for name in tail_fields:
            if not bitwise_equal(x[name][512:],y[name]):raise AssertionError('Resumed GPU trajectory differs: '+name)
            arrays[name]=int(y[name].size)
        endpoints=[name for name in y.files if name.endswith('_1024') or name.startswith('source_')]
        for name in endpoints:
            if not bitwise_equal(x[name],y[name]):raise AssertionError('Resumed endpoint observation differs: '+name)
            arrays[name]=int(y[name].size)
    resources=[]
    for name in selection['qualification_phases'][0]:
        meta=load_json(study/'runs'/name/'manifest.json')
        resources.append(dict(run_id=name,recipe=meta['arguments']['recipe'],heads=meta['arguments']['heads'],
            updates=meta['completed_step']-meta['start_step'],seconds=meta['seconds'],peak_cuda_gb=meta['peak_cuda_gb'],
            seconds_per_update_including_observations=meta['seconds']/(meta['completed_step']-meta['start_step'])))
    for n in [4,14]:
        metas=[load_json(study/'runs'/f'qualification-{name}-h{n}-continuous'/'manifest.json')
               for name in ['controlled','reference1']]
        if metas[0]['initial_parameter_sha256']!=metas[1]['initial_parameter_sha256']:
            raise AssertionError('Paired reference and controlled initial states differ')
    write_json(output,dict(schema='scheduled-gpu-qualification-v1',status='complete',native_updates=5120,
        qualification_artifacts=6,raw_checks=records,checkpoint_recovery=dict(**counts,after=512,through=1024,
            byte_equal=True,emission_arrays=arrays,emission_elements=sum(arrays.values())),resources=resources,
        selection_sha256=sha256(study/'protocols/regime-qualification-selection.json'),
        verified_files={key[0]:value for key,value in files.cache.items()},
        verifier_sources={str(path.relative_to(repo)):sha256(path) for path in [Path(__file__),
                          repo/'scripts/verify_scheduled_raw.py',repo/'scripts/verify_scaling_raw.py']},
        scope='GPU float32 full-batch qualification. One reference1 N14 native trajectory is checked through all512 resumed updates, including the final parameters, moments, counters, scheduler and recorded emissions. Resource timings include fixed diagnostic and output costs.'))
    print('GPU reference execution and full native checkpoint recovery qualified',resources,flush=True)


if __name__=='__main__':main()
