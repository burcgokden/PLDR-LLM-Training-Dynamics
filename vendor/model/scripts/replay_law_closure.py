#!/usr/bin/env python
"""Independent native replay of a complete selected 64-update branch."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--study',required=True);parser.add_argument('--heads',type=int,choices=[4,14],required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--seed',type=int,default=640101)
    parser.add_argument('--step',type=int,default=8192);args=parser.parse_args()
    root=Path(args.root).resolve();study=Path(args.study).resolve();repo=Path(__file__).resolve().parents[1]
    target=study/f'replay-h{args.heads}.json'
    if target.exists():raise FileExistsError(target)
    spec=json.loads((study/'protocol.json').read_text())
    case=next(c for c in spec['cases'] if c['heads']==args.heads and c['seed']==args.seed and c['step']==args.step)
    run=study/'runs'/case['name'];result=json.loads((run/'results.json').read_text())
    assert sha256(case['parent'])==case['parent_sha256']
    parent=torch.load(case['parent'],map_location='cpu',mmap=True,weights_only=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.monotonic();source=root/'assets/PLDR-LLM-v51-SOC-110M-1'
    adapter=TrainingModel(source,args.heads,case['seed'],args.device)
    adapter.model.load_state_dict(parent['model']);profile=parent['recipe']
    assert profile['objective']=='last_external_target' and profile['clipping']=='norm' and profile['total_steps']==0
    named=list(adapter.model.named_parameters())
    def generator(n):return 'reslayerAs' in n or 'plgatt_layer' in n or 'layernormA' in n
    groups=[dict(params=[v for n,v in named if generator(n)],lr=profile['generator_peak']),
            dict(params=[v for n,v in named if not generator(n)],lr=profile['body_peak'])]
    optimizer=torch.optim.AdamW(groups,betas=tuple(profile['betas']),eps=profile['epsilon'],
                                weight_decay=profile['weight_decay'],foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']));scheduler.load_state_dict(copy.deepcopy(parent['scheduler']))
    with np.load(run/'sampling.npz') as f:blocks=f['block_ids'][0];evaluation=f['evaluation_crops']
    tokens=np.load(root/'data/refinedweb-onepass-524288/tokens.npy',mmap_mode='r')
    saved=np.load(run/'branch-00.npz');compared=0
    def check_emission(horizon):
        nonlocal compared
        adapter.model.eval()
        with torch.no_grad():
            ids=torch.tensor(evaluation[:,:64],dtype=torch.long,device=args.device)
            logits=adapter.logits(ids).cpu().numpy()
        expected=saved['logits'][list(saved['horizons']).index(horizon)]
        if logits.dtype!=expected.dtype or logits.tobytes()!=expected.tobytes():
            raise AssertionError('Independent vocabulary replay differs at horizon '+str(horizon))
        compared+=logits.size
    check_emission(0)
    for step,indices in enumerate(blocks,1):
        rows=indices//8;offsets=indices%8*64
        crops=np.stack([tokens[r,o:o+65] for r,o in zip(rows,offsets,strict=True)])
        batch=torch.tensor(crops,dtype=torch.long,device=args.device)
        adapter.model.train();optimizer.zero_grad(set_to_none=True)
        logits=adapter.model(batch[:,:64],use_cache=False,logits_to_keep=1).logits[:,-1]
        loss=torch.nn.functional.cross_entropy(logits,batch[:,64]);loss.backward()
        np.testing.assert_array_equal(np.array(float(loss.detach())),saved['losses'][step-1])
        torch.nn.utils.clip_grad_norm_(adapter.model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();scheduler.step()
        if step in saved['horizons']:check_emission(step)
    # The canonical signature serializes every parameter and every optimizer tensor.
    digest=hashlib.sha256()
    for name,value in adapter.model.named_parameters():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        for key,entry in sorted(optimizer.state[value].items()):
            digest.update(key.encode())
            digest.update(entry.detach().cpu().contiguous().numpy().tobytes() if isinstance(entry,torch.Tensor)
                          else repr(entry).encode())
    digest.update(json.dumps(scheduler.state_dict(),sort_keys=True).encode())
    digest.update(json.dumps([{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups],sort_keys=True).encode())
    assert digest.hexdigest()==result['records'][0]['final_state_sha256']
    write_json(target,dict(status='passed',native_updates=64,case=case['name'],device=args.device,
        full_state_bitwise=True,full_vocabulary_logits_bitwise=True,compared_logit_coordinates=compared,
        final_state_sha256=digest.hexdigest(),seconds=time.monotonic()-start,
        sources={n:sha256(repo/n) for n in ['scripts/replay_law_closure.py','src/model_rg/native.py','src/model_rg/training.py']},
        inputs={str(p):sha256(p) for p in [Path(case['parent']),run/'sampling.npz',run/'branch-00.npz',run/'results.json']},
        scope='Independent sampler reconstruction and ordinary native last-target loss/AdamW execution, '
            'without importing the branch producer or analyzer. This is a replay of one recorded '
            'branch, not an additional independent scientific training realization.'))
    print('Passed independent 64-update replay',case['name'],flush=True)


if __name__=='__main__':main()
