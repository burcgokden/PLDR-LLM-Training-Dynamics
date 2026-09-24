#!/usr/bin/env python
"""Completed-design worker: full-parameter PLDR training and fixed inference probes."""
import argparse
import copy
import time
from pathlib import Path
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.provenance import environment, sha256, source_manifest, write_json


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--source',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--heads',type=int,required=True)
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--steps',type=int,default=256);ap.add_argument('--batch',type=int,default=32)
    ap.add_argument('--probe-documents',type=int,default=512)
    args=ap.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source_files=source_manifest()
    tokens=np.load(Path(args.data)/'tokens.npy')
    rng=np.random.default_rng(args.seed+70000)
    rows=rng.integers(0,3072,size=(args.steps+4,args.batch))
    offsets=rng.integers(0,449,size=rows.shape)
    batches=tokens[rows[:,:,None],offsets[:,:,None]+np.arange(65)[None,None,:]]
    probe_rows=np.arange(3584,3584+args.probe_documents)
    m=TrainingModel(args.source,args.heads,args.seed,args.device)
    optimizer=torch.optim.AdamW(m.model.parameters(),lr=3e-4,betas=(.9,.95),eps=1e-8,
                               weight_decay=.01,foreach=False)
    m.config.to_json_file(str(out/'config.json'))
    np.savez_compressed(out/'sampling.npz',rows=rows,offsets=offsets,probe_rows=probe_rows)
    checkpoints=sorted(set([0]+[x for x in [32,128,256] if x<=args.steps]+[args.steps]))
    losses=[];gradnorms=[];features=[];gnorms=[];energies=[];times=[]
    checkpoint_hashes={};start=time.time()

    def step(index):
        m.model.train();optimizer.zero_grad(set_to_none=True)
        batch=torch.tensor(batches[index],dtype=torch.long,device=args.device)
        z=m.logits(batch[:,:64])
        loss=torch.nn.functional.cross_entropy(z,batch[:,64])
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        return float(loss.detach()),float(norm)

    def snapshot(t):
        m.model.eval();f=[];g=[];a=[]
        with torch.no_grad():
            for chunk in np.array_split(probe_rows,max(1,(len(probe_rows)+15)//16)):
                batch=torch.tensor(tokens[chunk,:65],dtype=torch.long,device=args.device)
                y,diag=m.features(batch[:,:64],batch[:,64])
                f.append(y.cpu().numpy());g.append(diag['g_rms'].cpu().numpy())
                a.append(diag['a_row_energy'].cpu().numpy())
        features.append(np.concatenate(f));gnorms.append(np.concatenate(g));energies.append(np.concatenate(a))
        if not all(np.isfinite(x[-1]).all() for x in [features,gnorms,energies]):
            raise RuntimeError('Nonfinite inference probes')
        times.append(time.time()-start)
        print(f'heads={args.heads} seed={args.seed} step={t} probe_nll={features[-1][:,-1].mean():.6f} seconds={times[-1]:.1f}',flush=True)

    snapshot(0)
    for t in range(args.steps):
        loss,norm=step(t);losses.append(loss);gradnorms.append(norm)
        if t+1 in checkpoints:snapshot(t+1)
    checkpoint={'model':{k:v.detach().cpu() for k,v in m.model.state_dict().items()},
                'optimizer':copy.deepcopy(optimizer.state_dict()),'step':args.steps,
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(args.device),
                'arguments':vars(args)}
    # One native torch checkpoint includes optimizer moments. It is locally produced.
    torch.save(checkpoint,out/'final-training-state.pt')
    checkpoint_hashes['final-training-state.pt']=sha256(out/'final-training-state.pt')
    # Replay four fixed minibatches in one call sequence and in two groups. The
    # full augmented state is restored at the grouping boundary.
    before=copy.deepcopy(m.model.state_dict());opt_before=copy.deepcopy(optimizer.state_dict())
    for t in range(args.steps,args.steps+4):step(t)
    direct={k:v.detach().clone() for k,v in m.model.state_dict().items()}
    direct_opt=copy.deepcopy(optimizer.state_dict())
    m.model.load_state_dict(before);optimizer.load_state_dict(opt_before)
    for t in range(args.steps,args.steps+2):step(t)
    boundary={'model':copy.deepcopy(m.model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict())}
    m.model.load_state_dict(boundary['model']);optimizer.load_state_dict(boundary['optimizer'])
    for t in range(args.steps+2,args.steps+4):step(t)
    primal=max(float((v-direct[k]).abs().max()) for k,v in m.model.state_dict().items())
    moment=max(float((state[k]-direct_opt['state'][i][k]).abs().max())
               for i,state in optimizer.state_dict()['state'].items() for k in state if torch.is_tensor(state[k]))
    m.model.load_state_dict(before);optimizer.load_state_dict(opt_before)
    raw=out/'training.npz'
    np.savez_compressed(raw,steps=np.array(checkpoints),features=np.stack(features),g_rms=np.stack(gnorms),
                        a_row_energy=np.stack(energies),losses=np.array(losses),gradient_norms=np.array(gradnorms),
                        snapshot_seconds=np.array(times))
    meta={'schema':'model-rg-training-v1','arguments':vars(args),'source_files':source_files,
          'environment':environment(),'parameters':sum(p.numel() for p in m.model.parameters()),
          'trainable_parameters':sum(p.numel() for p in m.model.parameters() if p.requires_grad),
          'feature_names':m.feature_names(),'checkpoints':checkpoints,'train_documents':[0,3072],
          'probe_documents':[3584,3584+args.probe_documents],'length':64,'learning_rate':3e-4,
          'optimizer':{'name':'AdamW','betas':[.9,.95],'epsilon':1e-8,'weight_decay':.01,'clip_norm':1.},
          'data_manifest_sha256':sha256(Path(args.data)/'manifest.json'),
          'native_source_sha256':sha256(Path(args.source)/'modeling_pldrllm.py'),
          'native_config_source_sha256':sha256(Path(args.source)/'configuration_pldrllm.py'),
          'raw_sha256':sha256(raw),'sampling_sha256':sha256(out/'sampling.npz'),
          'training_state_sha256':checkpoint_hashes,'seconds':time.time()-start,
          'peak_cuda_gb':torch.cuda.max_memory_allocated(args.device)/2**30,
          'temporal_blocking':{'steps':4,'groups':[2,2],'parameter_max_abs_error':primal,'optimizer_max_abs_error':moment}}
    write_json(out/'manifest.json',meta)
    print('Completed',out,flush=True)


if __name__=='__main__':main()
