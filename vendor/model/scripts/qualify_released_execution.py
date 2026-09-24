"""Execute bounded branch-specific qualification using distinct RefinedWeb blocks."""
import argparse,gc,json,sys,time
from pathlib import Path
import numpy as np
import torch
import sentencepiece as spm
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.native import NativeModel
from model_rg.physical_native import prefix_batch,context_tokens,spin_tokens,selected_forward
from model_rg.provenance import sha256,write_json
from model_rg.released_qualification import ROOT,SCHEMA,contract,configure,destination,qualify_path,need
from train_released_lowrank import inject,merged


def main(a):
    study=destination(a.study);configure();torch.set_num_threads(4)
    need(a.device in ['cuda:0','cuda:1'],'Unsupported device')
    before=contract(study,a.model,a.branch)
    out=qualify_path(study,a.model,a.branch).parent;need(not out.exists(),'Qualification destination exists')
    out.mkdir(parents=True);torch.cuda.set_device(a.device);torch.manual_seed(2026091320)
    start=time.perf_counter();assets=ROOT/f'assets/PLDR-LLM-v51-SOC-110M-{a.model}'
    model=NativeModel(assets,a.device);model.model.requires_grad_(True);torch.cuda.reset_peak_memory_stats(a.device)
    proc=spm.SentencePieceProcessor(model_file=str(assets/'tokenizer.model'));alphabet=spin_tokens(proc)
    x=np.random.default_rng(2026091320).integers(0,3,(2,8,8),dtype=np.uint8)
    inp=prefix_batch(x,31,context_tokens(proc,3,8,1.),alphabet,a.device)
    changed=x.copy();changed.reshape(2,-1)[:,31:]=(changed.reshape(2,-1)[:,31:]+1)%3
    need(torch.equal(inp,prefix_batch(changed,31,context_tokens(proc,3,8,1.),alphabet,a.device)),'Prefix leaks target/suffix')
    target=torch.tensor(x.reshape(2,-1)[:,31].astype('int64'),device=a.device)
    full=model.forward(inp).logits[:,-1,alphabet];value=full.detach().clone()
    torch.nn.functional.cross_entropy(full,target).backward()
    gradients={n:p.grad.detach().clone() for n,p in model.model.named_parameters() if p.grad is not None}
    model.model.zero_grad(set_to_none=True);del full
    selected,_=selected_forward(model,inp,alphabet);logerr=float((selected.detach()-value).abs().max())
    torch.testing.assert_close(selected,value,rtol=2e-5,atol=2e-5)
    torch.nn.functional.cross_entropy(selected,target).backward();num=den=0.
    for n,p in model.model.named_parameters():
        if p.grad is not None:num+=float((p.grad-gradients[n]).double().square().sum());den+=float(gradients[n].double().square().sum())
    graderr=(num/max(den,1e-30))**.5;need(graderr<2e-5,'Selected gradient mismatch')
    del gradients,selected,value;model.model.zero_grad(set_to_none=True);gc.collect()
    names=[]
    if a.branch=='factor':
        model.model.requires_grad_(False);names=inject(model.model)
    params=[p for p in model.model.parameters() if p.requires_grad]
    if a.branch=='factor':need(all('.lora_' in n for n,p in model.model.named_parameters() if p.requires_grad),'Unexpected trainable matrix')
    opt=torch.optim.AdamW(params,lr=1e-4 if a.branch=='factor' else 1e-5,betas=(.9,.95),eps=1e-8,weight_decay=.01)
    with np.load(study/'data/language.npz') as z:
        tokens=z['technical-train-tokens'][:96,:65].copy();ids=z['technical-train-ids'][:96].copy()
        probe=z['technical-validation-tokens'][:4,:64].copy()
    need(len(np.unique(ids))==96,'Repeated qualification document')
    trace=[]
    for step in range(3):
        inp=torch.tensor(tokens[step*32:(step+1)*32,:64],device=a.device,dtype=torch.long)
        target=torch.tensor(tokens[step*32:(step+1)*32,64],device=a.device,dtype=torch.long)
        model.model.train();opt.zero_grad(set_to_none=True);logits=model.forward(inp).logits[:,-1]
        loss=torch.nn.functional.cross_entropy(logits,target);loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
        need(bool(torch.isfinite(loss) and torch.isfinite(norm)),'Nonfinite native update');opt.step()
        trace.append([step+1,float(loss.detach()),float(norm)])
    merge_error=0.
    if a.branch=='factor':
        native=NativeModel(assets,a.device);native.model.load_state_dict(merged(model.model,names),strict=True)
        model.model.eval();xx=torch.tensor(probe,device=a.device,dtype=torch.long)
        with torch.no_grad():
            left=model.forward(xx).logits;right=native.forward(xx).logits
            torch.testing.assert_close(left,right,rtol=3e-5,atol=3e-5);merge_error=float((left-right).abs().max())
    np.savez_compressed(out/'trace.npz',trace=np.asarray(trace),document_ids=ids,block_numbers=np.zeros(96,dtype=np.int64))
    write_json(out/'measurements.json',dict(status='passed',model=a.model,branch=a.branch,qualification_updates=3,scientific_updates=0,
        distinct_training_blocks=96,prefix_exclusion=True,selected_logit_error=logerr,selected_gradient_error=graderr,
        trainable_parameters=sum(p.numel() for p in params),targets=names,merge_error=merge_error,
        seconds=time.perf_counter()-start,peak_gib=torch.cuda.max_memory_allocated(a.device)/2**30))
    need(before==contract(study,a.model,a.branch),'Qualification inputs changed during execution')
    write_json(out/'verification.json',dict(schema=SCHEMA,status='passed',contract=before,
        artifacts={n:sha256(out/n) for n in ['measurements.json','trace.npz']}))
    print(json.dumps(json.loads((out/'measurements.json').read_text())),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--model',type=int,choices=[1,4,5],default=5);p.add_argument('--branch',choices=['full','factor'],required=True);p.add_argument('--device',required=True);main(p.parse_args())
