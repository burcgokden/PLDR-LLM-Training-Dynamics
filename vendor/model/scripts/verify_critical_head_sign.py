#!/usr/bin/env python3
"""Independently verify the saved native head-sign qualification states."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256,write_json

REPO=Path(__file__).resolve().parents[1]


def equal_bytes(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and a.detach().contiguous().numpy().tobytes()==b.detach().contiguous().numpy().tobytes()


def verify(root,output):
    if output.exists():raise FileExistsError(output)
    p=json.loads((root/'protocol.json').read_text());a=json.loads((root/'analysis.json').read_text())
    if p['schema']!='critical-head-sign-check-v1' or p['role']!='qualification' or a['scientific_updates']!=0 or not a['all_pairs_passed']:
        raise ValueError('Wrong native symmetry role or status')
    checked={str(root/'protocol.json'):sha256(root/'protocol.json'),str(root/'analysis.json'):sha256(root/'analysis.json')}
    for names,base in [(p['source_sha256'],REPO),(p['input_sha256'],None),(a['checked_sha256'],None)]:
        for name,digest in names.items():
            path=base/name if base else Path(name)
            if sha256(path)!=digest:raise ValueError('Changed symmetry input '+str(path))
            checked[str(path)]=digest
    parent=Path(p['study']);selection=np.load(parent/'selection.npz');comparisons=0;tensors=0;updates=0
    if len(a['records'])!=len(p['cases']) or a['reconstructed_paths']!=len(p['cases']):raise ValueError('Paired-case count differs')
    for case,record in zip(p['cases'],a['records']):
        if record['case']!=case or record['role']!='qualification' or record['scientific_updates']!=0 or record['status']!='passed':
            raise ValueError('Symmetry comparison role or case differs')
        folder=root/'runs'/f'h{case["heads"]}-s{case["seed"]}-{case["pattern"]}'
        actual=json.loads((folder/'manifest.json').read_text())
        if actual!=record or actual['protocol_sha256']!=sha256(root/'protocol.json'):raise ValueError('Unbound saved comparison')
        np.testing.assert_array_equal(actual['source_blocks'],selection['blocks_0'][:case['steps']])
        if len(set(np.asarray(actual['source_blocks']).ravel().tolist()))!=32*case['steps']:raise ValueError('Source blocks repeated')
        if len(actual['trace'])!=case['steps'] or actual['replay_updates']!=2*case['steps']:raise ValueError('Prefix cost or observation count differs')
        for key in ['losses','gradient_norms']:
            values=np.asarray(actual[key]);np.testing.assert_array_equal(values[:,0],values[:,1])
        for row in actual['trace']:
            if any(value!=0 for key,value in row.items() if key!='step'):raise ValueError('A live native forward comparison was not exact')
        if actual['maximum_normalized_gradient_error']!=0 or not actual['all_compared_gradient_tensor_bytes_equal']:
            raise ValueError('A live transformed gradient comparison was not exact')
        states=[torch.load(folder/(label+'.pt'),map_location='cpu',mmap=True,weights_only=False) for label in ['native','transformed']]
        left,right=states
        if left['model'].keys()!=right['model'].keys():raise ValueError('Model inventory differs')
        prefixes=sorted(name[:-len('.wq.weight')] for name in left['model'] if name.endswith('.wq.weight'))
        if len(prefixes)!=5:raise ValueError('Expected five query blocks')
        masks={};declared=[];n=case['heads']
        for layer,prefix in enumerate(prefixes):
            signs=[-1. if (case['pattern']=='single_first' and layer==0 and h==0) or
                    (case['pattern']=='checkerboard' and (layer+h)%2==0) else 1. for h in range(n)]
            sign=torch.tensor(signs,dtype=torch.float32);declared.append(signs)
            masks[prefix+'.wq.weight']=sign.repeat_interleave(64)[:,None]
            if prefix+'.wq.bias' in left['model']:masks[prefix+'.wq.bias']=sign.repeat_interleave(64)
            masks[prefix+'.plgatt_layer.alst']=sign[:,None,None];masks[prefix+'.plgatt_layer.balst']=sign[:,None,None]
        np.testing.assert_array_equal(declared,actual['head_signs'])
        for name,value in left['model'].items():
            expected=value*masks[name] if name in masks else value
            if not equal_bytes(expected,right['model'][name]):raise ValueError('Saved native parameter symmetry differs: '+name)
            tensors+=1
        if left['optimizer']['param_groups']!=right['optimizer']['param_groups'] or left['optimizer']['state'].keys()!=right['optimizer']['state'].keys():
            raise ValueError('Saved optimizer group or participation differs')
        names=actual['optimizer_parameter_names']
        if len(set(names.values()))!=len(names) or set(names)!={str(i) for i in left['optimizer']['state']}:
            raise ValueError('Optimizer coordinate binding is incomplete')
        for index,state in left['optimizer']['state'].items():
            name=names[str(index)];other=right['optimizer']['state'][index]
            if name not in left['model'] or state.keys()!=other.keys():raise ValueError('Unknown optimizer coordinate')
            for key,value in state.items():
                expected=value*masks[name] if key=='exp_avg' and name in masks else value
                if not equal_bytes(expected,other[key]):raise ValueError('Saved native Adam symmetry differs: '+name+' '+key)
                tensors+=1
        if not equal_bytes(left['rng'],right['rng']):raise ValueError('Coupled branch random states differ')
        updates+=actual['replay_updates'];comparisons+=1
    if updates!=p['replay_updates'] or updates!=a['replay_updates']:raise ValueError('Replay subtotal differs')
    result=dict(schema='critical-head-sign-verification-v1',status='passed',study=str(root),parent_study=str(parent),
        comparison_pairs=comparisons,native_prefix_executions=2*comparisons,replay_updates=updates,scientific_updates=0,
        saved_state_tensors_independently_compared=tensors,all_saved_state_tensor_bytes_equal=True,
        checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Saved complete model/Adam/RNG states and source-prefix identities are independently reconstructed. Live forward and gradient byte comparisons are source-bound acquisition records; their full per-step tensors are not retained.')
    write_json(output,result);print(json.dumps({k:result[k] for k in ['status','comparison_pairs','native_prefix_executions','replay_updates']}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.study.resolve(),a.output.resolve())
