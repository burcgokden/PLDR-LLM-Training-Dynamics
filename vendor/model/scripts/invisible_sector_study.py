#!/usr/bin/env python
"""Predeclared native test of an input-support null sector and its decay law."""
from companion_paths import legacy_path
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.schedules import loss as native_loss, clip as native_clip
from model_rg.provenance import sha256,write_json
from finite_response_study import observe

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MODEL_RG_DATA_ROOT',legacy_path('/pldr-data/model'))).resolve()
OUT=ROOT/'finite-response-20260912/invisible-sector'


def require(flag,message):
    if not flag: raise ValueError(message)


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    parent=ROOT/'outer-transfer-20260911';prior=json.loads((parent/'protocol.json').read_text())
    cases=[c for c in prior['cases'] if c['corpus']==0 and c['identity']==0]
    inputs={str(parent/p):sha256(parent/p) for p in ['protocol.json','data/panels.npz','data/corpus-0.npy']}
    corpus=np.load(parent/'data/corpus-0.npy',mmap_mode='r')
    with np.load(parent/'data/panels.npz') as f:panel=f['evaluation']
    for c in cases:
        checkpoint=parent/'runs'/c['name']/'incoming-state.pt';inputs[str(checkpoint)]=sha256(checkpoint)
        order=np.random.default_rng(c['stream_seed']).permutation(524288)
        rng=np.random.default_rng(913140000+c['heads'])
        blocks=np.stack([rng.choice(order[65536:],32*8,replace=False).reshape(8,32) for _ in range(2)])
        # The domain is fixed before any perturbed outcomes are collected.
        ids=blocks.ravel();contexts=corpus[(ids//8)[:,None],(64*(ids%8))[:,None]+np.arange(64)]
        require(not (contexts==0).any() and not (panel[:,:64]==0).any(),'Token 0 is outside the specified domain')
        path=OUT/(c['name']+'-blocks.npy');np.save(path,blocks);inputs[str(path)]=sha256(path)
    native=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    for p in native.glob('*.py'):inputs[str(p)]=sha256(p)
    names=set(prior['sources'])|{'scripts/invisible_sector_study.py','scripts/finite_response_study.py','scripts/verify_invisible_sector.py'}
    sources={str(REPO/n):sha256(REPO/n) for n in names}
    for n in names:
        dest=OUT/'executed-source'/n;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/n).read_bytes())
    write_json(OUT/'protocol.json',dict(status='frozen',schema='native-invisible-sector-v1',cases=cases,
        token=0,horizon=8,source_replicates=2,batch_size=32,times=list(range(9)),relative_pulse=.01,
        seed_base=913140000,inputs=inputs,sources=sources,native_updates=64,replay_updates=16,
        primary='Every vocabulary observation and every nonintervened model/Adam tensor must be bitwise equal for the baseline and pulse. The intervened embedding row must exactly follow the native per-step weight-decay multiplication predicted before the first update.',
        conditioning='Two paired widths, one complete initialization and corpus, two fresh without-replacement source sequences. Token 0 is absent from all declared training inputs and evaluation prefixes; its input-embedding moments must start at zero. Other input laws can observe this row.',
        limits='A finite source-support mechanism test. No native gap or critical exponent is estimated; the slow factor is imposed by the body learning rate and weight decay.'))
    print('Frozen 64 native and 16 replay updates',flush=True)


def worker(heads,device):
    spec=json.loads((OUT/'protocol.json').read_text())
    for path,digest in {**spec['inputs'],**spec['sources']}.items():require(sha256(path)==digest,'Changed bound source/input')
    case=next(c for c in spec['cases'] if c['heads']==heads);dest=OUT/case['name'];dest.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    parent=ROOT/'outer-transfer-20260911';ck=torch.load(parent/'runs'/case['name']/'incoming-state.pt',map_location='cpu',weights_only=False)
    corpus=np.load(parent/'data/corpus-0.npy',mmap_mode='r')
    with np.load(parent/'data/panels.npz') as f:panel=f['evaluation']
    blocks=np.load(OUT/(case['name']+'-blocks.npy'))
    model=TrainingModel(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1',heads,case['seed'],device)
    optimizer,scheduler=optimizer_and_scheduler(model,case['profile'])
    embedding=model.model.get_input_embeddings().weight
    require(embedding is not model.model.get_output_embeddings().weight,'Untied embeddings required')
    embedding_name=next(n for n,p in model.model.named_parameters() if p is embedding)
    def restore():
        optimizer.zero_grad(set_to_none=True);model.capture=False;model.eta=None;model.head_outputs={}
        model.model.load_state_dict(ck['model']);optimizer.load_state_dict(copy.deepcopy(ck['optimizer']));scheduler.load_state_dict(copy.deepcopy(ck['scheduler']))
    def hash_remaining():
        h=hashlib.sha256()
        for n,p in model.model.state_dict().items():
            h.update(n.encode());v=p.detach().cpu().numpy()
            h.update(v[1:].tobytes() if n==embedding_name else v.tobytes())
        # No moment coordinates are omitted, including the intervened row.
        for n,p in model.model.named_parameters():
            for k,v in sorted(optimizer.state[p].items()):
                h.update((n+k).encode());h.update(v.detach().cpu().numpy().tobytes() if isinstance(v,torch.Tensor) else repr(v).encode())
        h.update(json.dumps(scheduler.state_dict(),sort_keys=True).encode())
        return h.hexdigest()
    start=time.perf_counter();records=[];all_arrays={}
    for source,arm in [(s,a) for s in range(2) for a in ['zero','pulse']]+[(0,'replay')]:
        restore()
        require(not optimizer.state[embedding]['exp_avg'][0].any() and not optimizer.state[embedding]['exp_avg_sq'][0].any(),'Initial row moments are not zero')
        if arm=='pulse':
            with torch.no_grad():embedding[0].mul_(1+spec['relative_pulse'])
        group=next(g for g in optimizer.param_groups if any(p is embedding for p in g['params']))
        factor=1-group['lr']*group['weight_decay']
        # Build the complete row forecast before observing any continued state.
        predicted=[embedding[0].detach().clone()]
        for t in range(8):predicted.append(predicted[-1].clone().mul_(factor))
        observations=[observe(model,panel)];rows=[embedding[0].detach().cpu().numpy().copy()];digests=[hash_remaining()];losses=[]
        for t,ids in enumerate(blocks[source],1):
            values=corpus[(ids//8)[:,None],(64*(ids%8))[:,None]+np.arange(65)]
            require(not (values[:,:64]==0).any(),'Unexpected support change')
            model.model.train();optimizer.zero_grad(set_to_none=True)
            loss=native_loss(model,torch.as_tensor(values,dtype=torch.long,device=device),case['profile'])
            loss.backward();require(not embedding.grad[0].any(),'Null input row acquired a gradient')
            native_clip(model,case['profile']);optimizer.step();scheduler.step()
            require(group['lr']==case['profile']['body_peak'],'Unexpected changing body rate')
            rows.append(embedding[0].detach().cpu().numpy().copy());digests.append(hash_remaining());losses.append(float(loss.detach()))
            observations.append(observe(model,panel))
        arr={k:np.stack([v[k] for v in observations]) for k in observations[0]}
        arr.update(embedding_rows=np.array(rows),predicted_rows=torch.stack(predicted).cpu().numpy(),blocks=blocks[source],losses=losses)
        path=dest/f'source{source}-{arm}.npz';np.savez(path,**arr)
        require(np.array_equal(arr['embedding_rows'],arr['predicted_rows']),'Native row forecast differs')
        all_arrays[source,arm]=arr
        records.append(dict(source=source,arm=arm,path=str(path),sha256=sha256(path),unaffected_digests=digests,decay_factor=factor))
        print(case['name'],source,arm,round(time.perf_counter()-start,1),'seconds',flush=True)
    for source in range(2):
        x=all_arrays[source,'zero'];y=all_arrays[source,'pulse']
        require(all(np.array_equal(x[k],y[k]) for k in ['q','logits','losses']), 'An invisible pulse changed a prediction')
        a=next(r for r in records if r['source']==source and r['arm']=='zero');b=next(r for r in records if r['source']==source and r['arm']=='pulse')
        require(a['unaffected_digests']==b['unaffected_digests'],'An unaffected state changed')
    require(all(np.array_equal(all_arrays[0,'zero'][k],all_arrays[0,'replay'][k]) for k in all_arrays[0,'zero']),'Replay differs')
    write_json(dest/'results.json',dict(status='complete',case=case,seconds=time.perf_counter()-start,records=records,
        protocol_sha256=sha256(OUT/'protocol.json'),native_updates=32,replay_updates=8,embedding_name=embedding_name))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','worker']);p.add_argument('--heads',type=int);p.add_argument('--device');a=p.parse_args()
    if a.action=='prepare':prepare()
    else:worker(a.heads,a.device)
