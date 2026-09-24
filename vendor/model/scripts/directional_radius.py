#!/usr/bin/env python
"""Development-only radius selection at fixed incoming states; no native updates."""
import argparse
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.criticality import generator_parameter
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256,write_json
from model_rg.schedules import loss as native_loss
from model_rg.training import TrainingModel
from directional_study import observe


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',required=True);a=p.parse_args()
    root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    old=root/'outer-transfer-20260911';spec=json.loads((old/'protocol.json').read_text())
    with np.load(old/'data/panels.npz') as f:panel=f['evaluation'];donors=f['donors']
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[];start=time.perf_counter();inputs={};amplitudes=[1e-6,3e-6,1e-5,3e-5]
    write_json(out/'protocol.json',dict(status='frozen',amplitudes=amplitudes,
        scope='Development-only instantaneous radius sweep, same complete incoming states and panels; zero training updates.',
        selection='Smallest amplitude with maximum initial q-secant discrepancy at most 10% across both parameter families and all eight states. If none qualifies, retain a finite-pulse description.',
        producer_sha256=sha256(__file__)))
    for case in spec['cases']:
        parent=old/'runs'/case['name']/'incoming-state.pt';inputs[str(parent)]=sha256(parent)
        ck=torch.load(parent,map_location='cpu',weights_only=False)
        model=TrainingModel(root/'assets/PLDR-LLM-v51-SOC-110M-1',case['heads'],case['seed'],a.device)
        model.model.load_state_dict(ck['model']);optimizer,scheduler=optimizer_and_scheduler(model,case['profile'])
        optimizer.load_state_dict(copy.deepcopy(ck['optimizer']));scheduler.load_state_dict(copy.deepcopy(ck['scheduler']))
        optimizer.zero_grad(set_to_none=True);model.model.train()
        loss=native_loss(model,torch.as_tensor(donors,dtype=torch.long,device=a.device),case['profile']);loss.backward()
        named=list(model.model.named_parameters())
        directions={};norms={}
        for family in ['generator','body']:
            group=[(n,p) for n,p in named if generator_parameter(n)==(family=='generator')]
            norms[family]=sum(p.detach().double().square().sum() for n,p in group).sqrt().item()
            gn=sum(p.grad.detach().double().square().sum() for n,p in group).sqrt().item()
            directions[family]={n:-p.grad.detach().clone()/gn for n,p in group}
        optimizer.zero_grad(set_to_none=True)
        arrays={}
        for family in directions:
            for h in amplitudes:
                q=[]
                for multiplier in [1.,-1.,.5,-.5]:
                    model.model.load_state_dict(ck['model'])
                    with torch.no_grad():
                        for n,p in named:
                            if n in directions[family]:p.add_(directions[family][n],alpha=h*multiplier*norms[family])
                    obs=observe(model,panel);q.append(obs['q'])
                    arrays[f'{family}-{h}-{multiplier}']=obs['q']
                large=(q[0]-q[1])/2;small=q[2]-q[3]
                error=float(np.linalg.norm(large-small)/max(np.linalg.norm(small),1e-30))
                signal=float(np.sqrt(np.mean((small/2)**2)))
                rows.append(dict(case=case['name'],family=family,amplitude=h,discrepancy=error,half_signal_rms=signal))
        np.savez(out/(case['name']+'.npz'),**arrays)
        print(case['name'],[(r['family'],r['amplitude'],round(r['discrepancy'],5)) for r in rows if r['case']==case['name']],flush=True)
        del model,optimizer,scheduler,named,directions,ck
        torch.cuda.empty_cache()
    summary=[dict(amplitude=h,maximum_discrepancy=max(r['discrepancy'] for r in rows if r['amplitude']==h),
        passed=sum(r['discrepancy']<=.1 and r['half_signal_rms']>5e-7 for r in rows if r['amplitude']==h)) for h in amplitudes]
    eligible=[r['amplitude'] for r in summary if r['passed']==16]
    write_json(out/'results.json',dict(status='complete',native_updates=0,records=rows,summary=summary,
        selected_amplitude=min(eligible) if eligible else None,elapsed_seconds=time.perf_counter()-start,
        producer_sha256=sha256(__file__),inputs_sha256=inputs))
    print(summary,flush=True)

if __name__=='__main__':main()
