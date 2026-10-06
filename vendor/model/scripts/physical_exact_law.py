#!/usr/bin/env python
"""Enumerate complete finite PLDR configuration laws and verify error bounds."""
from companion_paths import configured_path
import argparse
import itertools
import json
from pathlib import Path
import sys
import numpy as np
import sentencepiece as spm
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import spin_tokens,context_tokens,prefix_batch,selected_forward
from model_rg.lattice import observables,critical_temperature
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))


def logsoftmax(x):
    shifted=x-x.max(-1,keepdims=True)
    return shifted-np.log(np.exp(shifted).sum(-1,keepdims=True))


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--case',required=True)
    a=ap.parse_args();study=Path(a.study)
    parent=study/'confirmation'/a.case;manifest=json.loads((parent/'manifest.json').read_text())
    if manifest['status']!='complete':raise ValueError('Complete model required')
    case=manifest['case'];checkpoint=manifest['checkpoints'][-1]
    if sha256(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Changed checkpoint')
    out=study/'exact-laws'/a.case;out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'))
    alphabet=spin_tokens(processor);adapter=TrainingModel(source,case['heads'],case['seed'],'cpu')
    state=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
    adapter.model.load_state_dict(state['model']);del state
    adapter.model.requires_grad_(False).eval();rows=[]
    for q in [2,3]:
        L=2;V=4;x=np.asarray(list(itertools.product(range(q),repeat=V)),dtype='uint8').reshape(-1,L,L)
        obs=observables(x,q);flat=x.reshape(-1,V)
        for ratio in [.94,1.,1.06]:
            meta=context_tokens(processor,q,L,ratio);letters=alphabet[:q]
            logsteps=[]
            for j in range(V):
                inputs=prefix_batch(x,j,meta,letters,'cpu')
                logits,_=selected_forward(adapter,inputs,letters)
                logsteps.append(logsoftmax(logits.double().numpy()))
            logsteps=np.stack(logsteps,1)
            logq=np.take_along_axis(logsteps,flat[:,:,None],2)[:,:,0].sum(1)
            model_prob=np.exp(logq)
            if abs(model_prob.sum()-1)>2e-6:raise ValueError('Generated joint law is not normalized')
            T=critical_temperature(q)*ratio
            ground_log=-obs['energy']/T;ground_log=logsoftmax(ground_log[None])[0]
            p=np.exp(ground_log)
            kl=float(np.sum(p*(ground_log-logq)));tv=float(.5*np.abs(p-model_prob).sum())
            prefix_terms=[]
            for j in range(V):
                total=0.
                groups={tuple(row[:j]) for row in flat}
                for prefix in groups:
                    selected=np.all(flat[:,:j]==np.asarray(prefix,dtype='uint8'),axis=1)
                    mass=p[selected].sum();conditional=np.array([p[selected & (flat[:,j]==c)].sum()/mass for c in range(q)])
                    index=np.flatnonzero(selected)[0]
                    total+=mass*np.sum(conditional*(np.log(conditional)-logsteps[index,j]))
                prefix_terms.append(float(total))
            if abs(sum(prefix_terms)-kl)>2e-6:raise ValueError('Prefix chain identity failed')
            bound=min(1.,np.sqrt(max(kl,0.)/2))
            errors={k:float(abs(p@obs[k]-model_prob@obs[k])) for k in ['m','m2','m4']}
            if tv>bound+2e-6 or max(errors.values())>tv+2e-6:raise ValueError('Finite-law inequality failed')
            path=out/f'q{q}-r{ratio:.3f}.npz'
            np.savez(path,configurations=x,ground_log_probability=ground_log,model_log_conditionals=logsteps,
                     model_log_probability=logq)
            rows.append(dict(q=q,L=L,ratio=ratio,configurations=len(x),kl=kl,tv=tv,pinsker_bound=bound,
                prefix_kl=prefix_terms,moment_absolute_errors=errors,
                reference_chi=float(V*(p@obs['m2'])),model_chi=float(V*(model_prob@obs['m2'])),
                raw=path.name,raw_sha256=sha256(path)))
    result=dict(status='passed',case=case,checkpoint=checkpoint,device='cpu',rows=rows,
        producer_sha256=sha256(__file__),normalization_tolerance=2e-6,
        scope='All 16 binary and 81 ternary configurations on a periodic 2x2 lattice. This is a finite-law check, not a thermodynamic exponent measurement.')
    write_json(out/'verification.json',result);print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
