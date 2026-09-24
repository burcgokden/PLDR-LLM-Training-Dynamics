#!/usr/bin/env python
"""Native selected-output parity and bounded GPU execution qualification."""
from companion_paths import legacy_path
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import sentencepiece as spm
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.training import TrainingModel
from model_rg.physical_native import context_tokens, spin_tokens, prefix_batch, selected_forward, generate
from model_rg.provenance import sha256, write_json

ROOT=Path(legacy_path('/pldr-data/model'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--heads',type=int,required=True)
    ap.add_argument('--device',required=True);a=ap.parse_args()
    study=Path(a.study);out=study/f'parity-h{a.heads}.json'
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    processor=spm.SentencePieceProcessor(model_file=str(source/'tokenizer.model'))
    alphabet=spin_tokens(processor);meta=context_tokens(processor,3,16,1.)
    model=TrainingModel(source,a.heads,640101,a.device)
    state_path=ROOT/f'scheduled-training-feasible-20260908/runs/compact-reference1-h{a.heads}-s640101/final-training-state.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=False)
    model.model.load_state_dict(state['model']);del state
    rng=np.random.default_rng(170101)
    configurations=rng.integers(0,3,(32,16,16),dtype=np.uint8)
    small=prefix_batch(configurations[:4],16,meta,alphabet,model.device)
    model.model.eval();model.model.zero_grad(set_to_none=True)
    full=model.forward(small).logits[:,-1,alphabet]
    selected,_=selected_forward(model,small,alphabet)
    error=float((full-selected).abs().max())
    torch.testing.assert_close(full,selected,rtol=2e-5,atol=2e-5)
    target=torch.as_tensor([0,1,2,0],device=model.device)
    torch.nn.functional.cross_entropy(full,target).backward()
    reference={n:p.grad.detach().clone() for n,p in model.model.named_parameters() if p.grad is not None}
    model.model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(selected,target).backward()
    numerator=0.;denominator=0.
    for n,p in model.model.named_parameters():
        if p.grad is not None:
            numerator+=float((p.grad-reference[n]).double().square().sum())
            denominator+=float(reference[n].double().square().sum())
    relative_gradient=(numerator/max(denominator,1e-30))**.5
    if relative_gradient>2e-5:raise ValueError('Selected-output gradient mismatch')
    del reference,full,selected
    model.model.zero_grad(set_to_none=True)
    optimizer=torch.optim.AdamW(model.model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    rows=[];torch.cuda.reset_peak_memory_stats(model.device)
    for site in [0,31,127,255]:
        inputs=prefix_batch(configurations,site,meta,alphabet,model.device)
        target=torch.as_tensor(configurations.reshape(32,-1)[:,site].astype('int64'),device=model.device)
        model.model.train();torch.cuda.synchronize();start=time.perf_counter()
        losses=[]
        for _ in range(3):
            fresh=rng.integers(0,3,(32,16,16),dtype=np.uint8)
            inputs=prefix_batch(fresh,site,meta,alphabet,model.device)
            target=torch.as_tensor(fresh.reshape(32,-1)[:,site].astype('int64'),device=model.device)
            optimizer.zero_grad(set_to_none=True)
            logits,_=selected_forward(model,inputs,alphabet)
            loss=torch.nn.functional.cross_entropy(logits,target);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        torch.cuda.synchronize()
        rows.append(dict(site=site,tokens=int(inputs.shape[1]),seconds_per_update=(time.perf_counter()-start)/3,losses=losses))
    start=time.perf_counter();samples=generate(model,context_tokens(processor,3,4,1.),alphabet,4,32,170123)
    generation_seconds=time.perf_counter()-start
    np.save(study/f'qualification-generated-h{a.heads}.npy',samples)
    from physical_design import sources as source_inventory, assets, runtime
    sources=source_inventory()
    report=dict(schema='physical-parity-qualification-v2',native_assets=assets(),runtime=runtime(),status='passed',heads=a.heads,device=a.device,alphabet=alphabet,metadata_length=len(meta),
        max_logit_error=error,relative_gradient_error=relative_gradient,profile=rows,
        generation_32_L4_seconds=generation_seconds,peak_gib=torch.cuda.max_memory_allocated(model.device)/2**30,
        qualification_updates=12,incoming_state=str(state_path),incoming_state_sha256=sha256(state_path),
        sources=sources,torch=torch.__version__,numpy=np.__version__)
    write_json(out,report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
