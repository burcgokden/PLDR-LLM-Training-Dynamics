#!/usr/bin/env python
"""Execute reset, single-pass native branches with full optimizer restoration."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.provenance import sha256, write_json
from model_rg.schedules import loss as native_loss, clip as native_clip
from model_rg.training import TrainingModel


def observe(model, optimizer, crops, keep_logits=False):
    model.model.eval()
    batch = torch.as_tensor(crops, dtype=torch.long, device=model.device)
    with torch.no_grad():
        out = model.forward(batch[:, :64], capture=True)
        logits = out.logits[:, -1].double(); logp = logits.log_softmax(-1)
        nll = -logp.gather(1, batch[:, 64, None]).squeeze(1)
        entropy = -(logp.exp()*logp).sum(-1)
        names = ['risk', 'predictive_entropy']+[f'logit_projection_{k}' for k in range(8)]
        values = [nll.mean(), entropy.mean(), *((logits @ model.logit_projection.double()).mean(0))]
        per_layer = []
        for l, att in enumerate(out.pldr_attentions):
            matrix = att[0].double(); operator = att[5].double()
            center = matrix.mean(-2, keepdim=True)
            total = matrix.square().mean((-2, -1))
            ratio = (matrix-center).square().mean((-2,-1))/total.clamp_min(1e-30)
            fields = [ratio.mean(), center.square().mean().sqrt(), operator.square().mean().sqrt()]
            per_layer.append(torch.stack(fields))
            names += [f'layer{l}_{key}' for key in ['row_ratio','centroid_rms','operator_rms']]
            values += fields
        row_dimension = len(values)
        for g, group in enumerate(optimizer.param_groups):
            count = sum(p.numel() for p in group['params'])
            for key in ['exp_avg', 'exp_avg_sq']:
                total = sum(optimizer.state[p][key].double().square().sum() for p in group['params'])
                values.append((total/count).sqrt()); names.append(f'group{g}_{key}_rms')
        q = torch.stack(values).cpu().numpy()
        result = dict(q=q, nll=nll.cpu().numpy(), entropy=entropy.cpu().numpy(),
                      layers=torch.stack(per_layer).cpu().numpy())
        if keep_logits: result['logits'] = logits.float().cpu().numpy()
    model.capture = False; model.head_outputs = {}; del out
    return result, names, row_dimension


def digest_state(model, optimizer, scheduler):
    h = hashlib.sha256()
    for name, p in model.model.named_parameters():
        h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
        for key, value in sorted(optimizer.state[p].items()):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes() if isinstance(value, torch.Tensor)
                     else repr(value).encode())
    h.update(json.dumps(scheduler.state_dict(),sort_keys=True).encode())
    h.update(json.dumps([{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],
                        sort_keys=True).encode())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True)
    p.add_argument('--study',required=True); p.add_argument('--case',required=True)
    p.add_argument('--device',required=True); p.add_argument('--qualification',action='store_true')
    a=p.parse_args(); root=Path(a.root).resolve(); study=Path(a.study).resolve()
    protocol=study/'protocol.json'; spec=json.loads(protocol.read_text())
    case=next(c for c in spec['cases'] if c['name']==a.case)
    repo=Path(__file__).resolve().parents[1]
    for name,digest in spec['producer_sources'].items():
        if sha256(repo/name)!=digest: raise AssertionError('Frozen producer changed: '+name)
    if sha256(case['parent'])!=case['parent_sha256']: raise AssertionError('Incoming state changed')
    if sha256(case['parent_verification'])!=case['parent_verification_sha256']:
        raise AssertionError('Incoming verification changed')
    out=study/('qualification' if a.qualification else 'runs')/case['name']
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    started=time.time()
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    inputs=[protocol,Path(case['parent']),Path(case['parent_verification'])]
    inputs += [Path(n) for n in spec['inputs_sha256']]
    inputs += [source/'configuration_pldrllm.py',source/'modeling_pldrllm.py']
    bind_run(out,inputs,vars(a))
    for name,digest in spec['inputs_sha256'].items():
        if sha256(name)!=digest: raise AssertionError('Frozen input changed: '+name)
    parent=torch.load(case['parent'],map_location='cpu',mmap=True,weights_only=True)
    assert parent['step']==case['step'] and parent['arguments']['seed']==case['seed']
    assert parent['recipe']==case['profile']
    model=TrainingModel(source,case['heads'],case['seed'],a.device)
    optimizer,scheduler=optimizer_and_scheduler(model,parent['recipe'])
    def restore():
        optimizer.zero_grad(set_to_none=True)
        model.capture=False; model.eta=None; model.head_outputs={}
        model.model.load_state_dict(parent['model'])
        optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
        scheduler.load_state_dict(copy.deepcopy(parent['scheduler']))
    restore()
    initial_digest=digest_state(model,optimizer,scheduler)
    data=np.load(root/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    probe=root/'controlled-study-20260905/data/short'
    tokens=np.load(probe/'tokens.npy',mmap_mode='r'); offsets=np.load(probe/'offsets.npy')
    cohort=np.array(spec['cohort']); crops=tokens[cohort[:,None],offsets[cohort,None]+np.arange(65)]
    order=np.random.default_rng(parent['arguments']['stream_seed']+1000).permutation(4194304)
    remaining=order[32*case['step']:]
    horizons=[0,1,4] if a.qualification else spec['horizons']
    branches=2 if a.qualification else spec['branches']
    rng=np.random.default_rng(spec['branch_seed']+case['seed']*37+case['heads']*101+case['step'])
    block_ids=np.stack([rng.choice(remaining,32*max(horizons),replace=False).reshape(-1,32)
                        for _ in range(branches)])
    np.savez_compressed(out/'sampling.npz',block_ids=block_ids,cohort=cohort,evaluation_crops=crops)
    first_final=None; records=[]
    for branch in range(branches+(1 if a.qualification else 0)):
        replay=branch==branches; b=0 if replay else branch
        restore()
        if digest_state(model,optimizer,scheduler)!=initial_digest:
            raise AssertionError('Incomplete incoming optimizer/model restoration')
        observations=[]; losses=[]; norms=[]
        base,names,row_dimension=observe(model,optimizer,crops,keep_logits=b==0)
        observations.append(base)
        for t,blocks in enumerate(block_ids[b],1):
            array=data[(blocks//8)[:,None],(64*(blocks%8))[:,None]+np.arange(65)]
            batch=torch.as_tensor(array,dtype=torch.long,device=model.device)
            model.model.train(); optimizer.zero_grad(set_to_none=True)
            loss=native_loss(model,batch,parent['recipe'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite native loss')
            loss.backward()
            norm=torch.linalg.vector_norm(torch.stack([q.grad.norm() for q in model.model.parameters()
                                                      if q.grad is not None]))
            native_clip(model,parent['recipe']); optimizer.step(); scheduler.step()
            losses.append(float(loss.detach())); norms.append(float(norm))
            if t in horizons:
                observations.append(observe(model,optimizer,crops,keep_logits=b==0)[0])
        arrays={k:np.stack([o[k] for o in observations]) for k in observations[0]}
        arrays.update(losses=np.array(losses),gradient_norms=np.array(norms),horizons=np.array(horizons))
        if not all(np.isfinite(v).all() for v in arrays.values()): raise FloatingPointError('Nonfinite observations')
        final_digest=digest_state(model,optimizer,scheduler)
        if replay:
            if final_digest!=first_final: raise AssertionError('Reset native replay differs')
            with np.load(out/'branch-00.npz') as original:
                for k,v in arrays.items():
                    if v.dtype!=original[k].dtype or v.shape!=original[k].shape or v.tobytes()!=original[k].tobytes():
                        raise AssertionError('Reset replay output differs: '+k)
        else:
            path=out/f'branch-{b:02d}.npz'; np.savez_compressed(path,**arrays)
            records.append(dict(branch=b,raw=path.name,sha256=sha256(path),final_state_sha256=final_digest))
        if branch==0:first_final=final_digest
        print(case['name'],'replay' if replay else f'branch {b+1}/{branches}',
              'seconds',round(time.time()-started,1),flush=True)
    restore(); restored=observe(model,optimizer,crops,keep_logits=True)[0]
    with np.load(out/'branch-00.npz') as first:
        for key,value in restored.items():
            if first[key][0].tobytes()!=value.tobytes(): raise AssertionError('Parent emission restoration differs')
    result=dict(schema='conditional-law-closure-branches-v1',status='complete',case=case,
        protocol_sha256=sha256(protocol),branches=branches,horizons=horizons,
        native_updates=(branches+(1 if a.qualification else 0))*max(horizons),
        scientific_updates=0 if a.qualification else branches*max(horizons),
        coordinate_names=names,row_predictive_dimension=row_dimension,
        records=records,initial_state_sha256=initial_digest,restored_parent_bitwise=True,
        reset_replay_bitwise=True if a.qualification else None,
        sampling_sha256=sha256(out/'sampling.npz'),
        started_at=datetime.fromtimestamp(started,timezone.utc).isoformat(),seconds=time.time()-started,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(model.device),
        scope=spec['conditional_law'],observation=spec['observation'])
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(schema=result['schema'],status='complete',
        binding_sha256=sha256(out/'binding.json'),results_sha256=sha256(out/'results.json'),
        sampling_sha256=sha256(out/'sampling.npz'),branch_files={r['raw']:r['sha256'] for r in records}))
    print('complete',case['name'],flush=True)


if __name__=='__main__':
    main()
