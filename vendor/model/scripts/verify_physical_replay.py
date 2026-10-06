#!/usr/bin/env python
"""Independent reconstruction of 512 updates in every physical training arm."""
from companion_paths import configured_path
import argparse
import gc
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
import sentencepiece as spm
import torch
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import spin_tokens,context_tokens,prefix_batch,selected_forward
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))


def compare(actual,expected,label):
    rows=[]
    if actual.keys()!=expected.keys():raise ValueError('Tensor inventory differs: '+label)
    for name,a in actual.items():
        b=expected[name]
        if torch.is_tensor(a):
            a=a.detach().cpu().double();b=b.detach().cpu().double()
            diff=(a-b).abs();maximum=float(diff.max()) if a.numel() else 0.
            relative=float(torch.linalg.vector_norm(diff)/torch.linalg.vector_norm(b).clamp_min(1e-30))
            rows.append(dict(name=str(name),maximum_absolute=maximum,relative_l2=relative))
            if not np.isfinite(relative) or relative>2e-6:raise ValueError('Replay tensor disagreement: '+label+'/'+str(name))
        elif isinstance(a,dict):rows.extend([dict(name=str(name)+'/'+r['name'],maximum_absolute=r['maximum_absolute'],relative_l2=r['relative_l2']) for r in compare(a,b,label+'/'+str(name))])
        elif a!=b:raise ValueError('Replay state value differs: '+label+'/'+str(name))
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    base=Path(a.study);spec=json.loads((base/'confirmation/protocol.json').read_text())
    out=base/'replay';out.mkdir(exist_ok=False)
    ds=json.loads((base/'training-data/manifest.json').read_text())
    draws=np.load(base/'confirmation/draws.npz')
    cases=[c for c in spec['cases'] if c['seed']==640101]
    raw=[np.memmap(c['path'],mode='r',dtype=np.uint8,shape=(c['chains']*c['samples_per_chain'],c['L'],c['L'])) for c in ds['cells']]
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'));alphabet=spin_tokens(processor)
    metadata=[context_tokens(processor,c['q'],c['L'],c['temperature_ratio']) for c in ds['cells']]
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    results=[]
    for case in cases:
        start=time.perf_counter();parent=base/'confirmation'/case['name']
        waiting=time.monotonic()
        while not (parent/'manifest.json').exists():
            if time.monotonic()-waiting>4*3600:raise TimeoutError('Training path incomplete')
            time.sleep(10)
        manifest=json.loads((parent/'manifest.json').read_text())
        expected_record=next(c for c in manifest['checkpoints'] if c['step']==512)
        if sha256(case['state'])!=case['state_sha256'] or sha256(expected_record['path'])!=expected_record['sha256']:
            raise ValueError('Replay input changed')
        adapter=TrainingModel(source,case['heads'],case['seed'],a.device)
        state=torch.load(case['state'],map_location='cpu',weights_only=False)
        adapter.model.load_state_dict(state['model']);del state;gc.collect()
        if case['arm']=='frozen-generator':
            for name,p in adapter.model.named_parameters():
                if '.reslayerAs.' in name:p.requires_grad_(False)
        parameters=[p for p in adapter.model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(parameters,lr=spec['learning_rate'],betas=(.9,.95),eps=1e-8,weight_decay=.01)
        rng=np.random.default_rng(1760000);trace=[]
        for t in range(512):
            cell=ds['cells'][int(draws['cell'][t])];j=int(draws['site'][t]);x=np.asarray(raw[cell['id']][draws['index'][t]]).copy()
            if case['arm']=='shuffled':
                for row in x.reshape(32,-1):rng.shuffle(row)
            inputs=prefix_batch(x,j,metadata[cell['id']],alphabet[:cell['q']],a.device)
            target=torch.as_tensor(x.reshape(32,-1)[:,j].astype('int64'),device=a.device)
            lr=spec['learning_rate']*((t+1)/128 if t<128 else .2+.4*(1+math.cos(math.pi*(t-128)/(spec['steps']-128))))
            optimizer.param_groups[0]['lr']=lr
            adapter.model.train();optimizer.zero_grad(set_to_none=True)
            logits,_=selected_forward(adapter,inputs,alphabet[:cell['q']]);loss=torch.nn.functional.cross_entropy(logits,target)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
            trace.append([t+1,cell['id'],j,float(loss.detach()),float(norm),lr])
        actual_trace=np.asarray(trace);reference_trace=np.load(parent/'training-trace.npy')[:512]
        if not np.allclose(actual_trace,reference_trace,rtol=2e-6,atol=1e-8):raise ValueError('Replay trajectory differs')
        expected=torch.load(expected_record['path'],map_location='cpu',weights_only=False)
        model_rows=compare(adapter.model.state_dict(),expected['model'],'model')
        optimizer_rows=compare(optimizer.state_dict()['state'],expected['optimizer']['state'],'optimizer')
        if optimizer.state_dict()['param_groups']!=expected['optimizer']['param_groups']:raise ValueError('Optimizer groups differ')
        np.save(out/(case['name']+'-trace.npy'),actual_trace)
        result=dict(case=case,status='passed',updates=512,input_state_sha256=case['state_sha256'],
            expected_checkpoint=expected_record,model=model_rows,optimizer=optimizer_rows,
            maximum_trace_difference=float(np.max(np.abs(actual_trace-reference_trace))),runtime_seconds=time.perf_counter()-start)
        write_json(out/(case['name']+'.json'),result);results.append(result)
        print('replay passed',case['name'],flush=True)
        del expected,optimizer,adapter,parameters;gc.collect();torch.cuda.empty_cache()
    write_json(out/'verification.json',dict(status='passed',schema='physical-replay-v1',
        role='Arithmetic reproduction; excluded from scientific trajectory counts.',updates=512*len(cases),
        verifier_sha256=sha256(__file__),protocol_sha256=sha256(base/'confirmation/protocol.json'),results=results))


if __name__=='__main__':main()
