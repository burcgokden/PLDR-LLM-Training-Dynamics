#!/usr/bin/env python
"""Execute every predeclared frozen-weight arithmetic cell on the full paired cohort."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import predictive_kl
from model_rg.observation import observation_transfer, screen_decomposition
from model_rg.precision import preserve_response_dtype
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='observation-closure-20260906');a=p.parse_args()
    root=Path(a.root);study=root/a.study;protocol_path=study/'protocols/arithmetic.json'
    protocol=json.loads(protocol_path.read_text());source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    probe=root/'controlled-study-20260905/data/short';rows=np.array(protocol['rows'])
    pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
    crops=np.asarray(pt[rows[:,None],po[rows,None]+np.arange(65)])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    records=[];start_all=time.time();ledger=study/'launcher-arithmetic.json'
    if ledger.exists():raise FileExistsError(ledger)
    write_json(ledger,dict(status='running',protocol_sha256=sha256(protocol_path),records=records))
    for cell in protocol['cells']:
        h,g,step=cell['heads'],cell['multiplier'],cell['step']
        out=study/'runs'/f'arithmetic-h{h}-g{g}-t{step}';out.mkdir(exist_ok=False)
        parents=[root/cell['parent_study']/'runs'/cell['parent_pattern'].format(seed=s) for s in protocol['seeds']]
        inputs=[protocol_path,probe/'tokens.npy',probe/'offsets.npy',source/'modeling_pldrllm.py',source/'configuration_pldrllm.py']
        inputs += [p/name for p in parents for name in ['final-training-state.pt','measurements.npz','manifest.json']]
        bind_run(out,inputs,dict(**vars(a),cell=cell))
        raw=dict(rows=rows,crops=crops);seed_records=[];started=time.time()
        def measure(m):
            energies=[];totals=[];logits=[]
            with torch.no_grad():
                for k in range(0,len(rows),16):
                    result=m.forward(torch.tensor(crops[k:k+16,:64],dtype=torch.long),capture=True)
                    mat=torch.stack([v[0] for v in result.pldr_attentions],1).double()
                    energies.append((mat-mat.mean(-2,keepdim=True)).square().mean((-1,-2)).numpy())
                    totals.append(mat.square().mean((-1,-2)).numpy())
                    logits.append(result.logits[:,-1].double().numpy())
            e,t=np.concatenate(energies),np.concatenate(totals)
            return e,t,e/np.maximum(t,1e-30),np.concatenate(logits)
        for seed,parent in zip(protocol['seeds'],parents,strict=True):
            m=TrainingModel(source,h,seed,'cpu')
            saved=torch.load(parent/'final-training-state.pt',map_location='cpu',mmap=True,weights_only=True)
            assert saved['step']==step and saved['arguments']['seed']==seed and saved['arguments']['multiplier']==g
            m.model.load_state_dict(saved['model']);del saved
            m.model.eval().requires_grad_(False)
            e32,t32,r32,z32=measure(m)
            m.model.double();preserve_response_dtype(m)
            e64,t64,r64,z64=measure(m)
            with np.load(parent/'measurements.npz') as z:
                archived=z[f'heads_{step}'][...,2].astype(float)
            kl=predictive_kl(torch.from_numpy(z32),torch.from_numpy(z64)).numpy()
            for name,value in dict(E32=e32,T32=t32,R32=r32,E64=e64,T64=t64,R64=r64,Rarchive=archived,KL=kl).items():
                if not np.isfinite(value).all():raise ValueError('Nonfinite '+name)
                raw[f's{seed}_{name}']=value
            seed_records.append(dict(seed=seed,mean_kl=float(kl.mean()),max_kl=float(kl.max())))
            print(out.name,seed,round(time.time()-started,1),flush=True);del m
        fields={name:np.stack([raw[f's{s}_{name}'] for s in protocol['seeds']]) for name in ['Rarchive','R32','R64']}
        transfer=observation_transfer(fields['Rarchive'].mean(-1),fields['R64'].mean(-1),h)
        assert abs(transfer['signed_difference'])<=transfer['absolute_bound']+1e-24
        agree=(transfer['relative_bound'] is not None and transfer['relative_bound']<=protocol['relative_tolerance'] and transfer['absolute_bound']<=protocol['absolute_tolerance'])
        result=dict(cell=cell,status='complete',independent_seeds=protocol['seeds'],contexts=len(rows),
            transfer=transfer,statistic_transfer=observation_transfer(fields['Rarchive'].mean(-1),fields['R32'].mean(-1),h),
            screens={name:[screen_decomposition(field,h,t) for t in protocol['screen_levels']] for name,field in fields.items()},
            precision_status='paired arithmetic agreement' if agree else 'insufficient arithmetic resolution at declared tolerance',
            seeds=seed_records,seconds=time.time()-started,protocol_sha256=sha256(protocol_path),
            scope=protocol['precision_scope'])
        np.savez_compressed(out/'measurements.npz',**raw);write_json(out/'results.json',result)
        write_json(out/'manifest.json',dict(schema='arithmetic-observation-v1',status='complete',cell=cell,
            binding_sha256=sha256(out/'binding.json'),raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json'),seconds=result['seconds']))
        records.append(dict(run_id=out.name,manifest_sha256=sha256(out/'manifest.json'),status='complete'))
        write_json(ledger,dict(status='running',protocol_sha256=sha256(protocol_path),records=records))
    write_json(ledger,dict(status='complete',protocol_sha256=sha256(protocol_path),records=records,seconds=time.time()-start_all))


if __name__=='__main__':main()
