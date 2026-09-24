#!/usr/bin/env python3
"""Check native CPU inference sign actions at selected trained checkpoints.

Every comparison uses the same GPU-trained parameter state and fixed source
contexts on both CPU branches. This checks the inference action, not CPU/GPU
trajectory equivalence. No training update is executed.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.provenance import sha256,write_json
from model_rg.criticality import parameter_digest
import check_critical_sign_symmetry as signs
import run_critical_onepass as native

REPO=native.REPO
SELF='scripts/check_critical_endpoint_sign.py'


@torch.no_grad()
def forward(model,ids):
    result=model.forward(ids,capture=True)
    return dict(logits=result.logits[:8,-1].detach().clone(),
        metric=[r[0].detach().clone() for r in result.pldr_attentions],
        operator=[r[5].detach().clone() for r in result.pldr_attentions],
        attention=[r[6].detach().clone() for r in result.pldr_attentions])


def check(parent,output,head_counts,controls):
    if output.exists() or output==native.ROOT or not output.is_relative_to(native.ROOT):
        raise ValueError('Use a fresh authorized observation directory')
    p=native.admit(parent);seeds=sorted(p['design']['seeds'])[:2]
    jobs=[j for j in p['jobs'] if j['heads'] in head_counts and j['control'] in controls and j['seed'] in seeds]
    if len(jobs)!=len(head_counts)*len(controls)*len(seeds)*len(p['design']['environments']):
        raise ValueError('The declared endpoint panel does not match the frozen native design')
    sources={**p['source_sha256'],SELF:sha256(__file__),
        'scripts/check_critical_sign_symmetry.py':sha256(REPO/'scripts/check_critical_sign_symmetry.py')}
    inputs={str(parent/'protocol.json'):sha256(parent/'protocol.json'),str(parent/'selection.npz'):sha256(parent/'selection.npz'),**p['input_sha256']}
    for j in jobs:
        folder=parent/'runs'/j['run_id'];m=json.loads((folder/'manifest.json').read_text())
        if m['status']!='complete' or m['job']!=j or m['protocol_sha256']!=sha256(parent/'protocol.json'):
            raise ValueError('An endpoint is not a completed bound native state')
        for name in ['manifest.json','final-state.pt']:
            digest=sha256(folder/name)
            if name=='final-state.pt' and digest!=m['artifacts'][name]:raise ValueError('Changed native checkpoint')
            inputs[str(folder/name)]=digest
    output.mkdir(parents=True)
    protocol=dict(schema='critical-endpoint-sign-check-v1',role='inference_qualification',parent_study=str(parent),
        jobs=jobs,head_counts=head_counts,controls=controls,seeds=seeds,patterns=['single_first','checkerboard'],
        source_sha256=sources,input_sha256=inputs,device='cpu',dtype='float32',threads=2,
        scientific_updates=0,training_updates=0,contexts=16,full_vocabulary_contexts=8,
        scope='Paired CPU inference at identical trained parameter states. No CPU/GPU emission or trajectory equivalence is assumed.')
    write_json(output/'protocol.json',protocol);ph=sha256(output/'protocol.json')
    for name,digest in sources.items():
        target=output/'executed-source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/name).read_bytes())
        if sha256(target)!=digest:raise ValueError('Changed source while freezing inference check')
    with np.load(parent/'selection.npz') as selected:ids=torch.tensor(selected['probes'][:16,:64],dtype=torch.long)
    torch.set_num_threads(2);records=[];checked={**inputs,str(output/'protocol.json'):ph}
    for job in jobs:
        started=time.monotonic();folder=output/'runs'/job['run_id'];folder.mkdir(parents=True)
        state=torch.load(parent/'runs'/job['run_id']/'final-state.pt',map_location='cpu',mmap=True,weights_only=False)
        model=native.TrainingModel(native.NATIVE,job['heads'],job['seed'],'cpu')
        model.model.load_state_dict(state['model']);model.model.eval();del state
        original=parameter_digest(model)
        parent_manifest=json.loads((parent/'runs'/job['run_id']/'manifest.json').read_text())
        if original!=parent_manifest['final_parameter_sha256']:raise ValueError('Loaded endpoint identity differs')
        rng=torch.get_rng_state().clone();base=forward(model,ids);logits=[base['logits'].numpy()];comparisons=[]
        for pattern in protocol['patterns']:
            masks,head_signs=signs.sign_masks(model,pattern);signs.transform(model,masks)
            changed=forward(model,ids);errors={};exact={}
            errors['logits'],exact['logits']=signs.tensor_error(base['logits'],changed['logits'])
            for field in ['metric','operator','attention']:
                values=[];bytes_equal=[]
                for layer,value in enumerate(base[field]):
                    expected=value*torch.from_numpy(head_signs[layer])[None,:,None,None] if field=='operator' else value
                    error,equal=signs.tensor_error(expected,changed[field][layer]);values.append(error);bytes_equal.append(equal)
                errors[field]=max(values);exact[field]=all(bytes_equal)
            signs.transform(model,masks)
            if parameter_digest(model)!=original:raise ValueError('Sign involution did not restore the trained parameters')
            if not torch.equal(torch.get_rng_state(),rng):raise ValueError('Inference comparison consumed random draws')
            logits.append(changed['logits'].numpy());comparisons.append(dict(pattern=pattern,normalized_errors=errors,
                tensor_bytes_equal=exact,parameters_restored=True,rng_unchanged=True))
            del changed,masks
        np.savez_compressed(folder/'predictive-laws.npz',logits=np.stack(logits))
        record=dict(schema=protocol['schema'],status='complete',role=protocol['role'],job=job,protocol_sha256=ph,
            training_updates=0,scientific_updates=0,native_forward_calls=3,comparisons=comparisons,
            checkpoint_sha256=inputs[str(parent/'runs'/job['run_id']/'final-state.pt')],
            predictive_arrays_sha256=sha256(folder/'predictive-laws.npz'),runtime_seconds=time.monotonic()-started)
        write_json(folder/'manifest.json',record)
        checked[str(folder/'manifest.json')]=sha256(folder/'manifest.json')
        checked[str(folder/'predictive-laws.npz')]=record['predictive_arrays_sha256']
        # Independently reduce the retained complete predictive arrays.
        with np.load(folder/'predictive-laws.npz') as arrays:
            predictions=arrays['logits']
            maximum=float(np.max(np.abs(predictions[1:]-predictions[0])))
            if not np.isfinite(predictions).all() or maximum>2e-6:raise ValueError('Trained-endpoint predictive sign action differs')
        if max(error for r in comparisons for error in r['normalized_errors'].values())>2e-6:
            raise ValueError('Trained-endpoint internal sign action differs')
        records.append(record);print('ENDPOINT SIGN',job['run_id'],'maximum logit difference',maximum,flush=True)
        del model,base,logits,predictions
    for name,digest in sources.items():
        if sha256(REPO/name)!=digest:raise ValueError('Executing inference source changed')
        checked[str(REPO/name)]=digest
    maximum=max(error for r in records for pair in r['comparisons'] for error in pair['normalized_errors'].values())
    exact=all(value for r in records for pair in r['comparisons'] for value in pair['tensor_bytes_equal'].values())
    write_json(output/'analysis.json',dict(schema='critical-endpoint-sign-analysis-v1',status='complete',study=str(output),parent_study=str(parent),
        role='inference_qualification',device='cpu',trained_endpoints=len(records),sign_comparisons=2*len(records),
        native_forward_calls=3*len(records),training_updates=0,scientific_updates=0,
        maximum_normalized_forward_difference=maximum,all_forward_tensor_bytes_equal=exact,
        records=records,checked_sha256=checked,source_sha256={SELF:sha256(__file__)},
        scope='Paired CPU inference at retained trained states; all full predictive comparison arrays are saved. Internal tensor comparisons are source-bound live records. Sign representatives are not extra scientific trajectories.'))
    print('Completed trained-endpoint sign checks',len(records),'maximum difference',maximum,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['parent','output']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--heads',type=int,nargs='+',required=True);p.add_argument('--controls',type=float,nargs='+',required=True)
    a=p.parse_args();check(a.parent.resolve(),a.output.resolve(),a.heads,a.controls)
