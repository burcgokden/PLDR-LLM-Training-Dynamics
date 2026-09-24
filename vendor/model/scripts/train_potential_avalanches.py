#!/usr/bin/env python3
"""Single-pass native PLDR potential activity with fixed-input observations.

This measures excursions, not a preclassified SOC process. The selected matrix
coordinates are nested in head count. All-coordinate summaries are also kept.
"""
from companion_paths import legacy_path
import argparse
import copy
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.training import TrainingModel
from model_rg.variance_family import normalize_variance_initialization
from model_rg.criticality import parameter_digest
from model_rg.provenance import sha256, write_json, environment

ROOT = Path(legacy_path('/pldr-data/model'))
STUDY = ROOT/'potential-avalanche-20260913'
REPO = Path(__file__).resolve().parents[1]
FIELDS = ['log_potential_increment_rms','exponent_contribution_rms',
          'base_contribution_rms','contribution_cross_mean',
          'log_potential_increment_mean','relative_potential_increment_rms',
          'curvature_increment_rms','curvature_rms','metric_row_ratio',
          'potential_rms','potential_base_min','potential_base_max',
          'log_identity_max_error','exponent_increment_rms']


def lr_at(step, steps, warmup, peak, schedule):
    if schedule == 'constant': return peak
    if step <= warmup: return peak * step / warmup
    if schedule == 'warm_plateau': return peak
    if schedule == 'warm_cosine':
        return peak * (.1 + .45*(1+math.cos(math.pi*(step-warmup)/(steps-warmup))))
    raise ValueError(schedule)


@torch.no_grad()
def observe(model, probes, coordinate, previous=None):
    model.model.eval()
    out = model.forward(probes[:, :64], capture=True)
    bases, powers, curves, metrics = [], [], [], []
    for a, base, power, _, _, curve, _ in out.pldr_attentions:
        bases.append(base.double()); powers.append(power.detach().clone().double())
        curves.append(curve.double()); metrics.append(a.double())
    base = torch.stack(bases, 1)
    power = torch.stack(powers, 0)[None]
    curve = torch.stack(curves, 1); metric = torch.stack(metrics, 1)
    logbase = base.log(); logpotential = logbase*power
    if not all(torch.isfinite(x).all() for x in [base, power, curve, metric, logpotential]):
        raise FloatingPointError('Nonfinite native observation')
    # Treat the observed FP32 base and exponent as exact inputs to this FP64
    # finite identity. This is not a claim of exact real-arithmetic training.
    zero = torch.zeros_like(logpotential)
    if previous is None:
        delta, exponent, moved_base, dp = zero, zero, zero, torch.zeros_like(power)
        dg = torch.zeros_like(curve)
    else:
        oldlog, oldpower, oldcurve = previous
        dp = power-oldpower
        exponent = dp*oldlog
        moved_base = power*(logbase-oldlog)
        delta = logpotential-oldpower*oldlog
        dg = curve-oldcurve
    rms = lambda v: v.square().mean((-2,-1)).sqrt()
    mean = lambda v: v.mean((-2,-1))
    row = mean((metric-metric.mean(-2,keepdim=True)).square())/mean(metric.square()).clamp_min(1e-30)
    fields = torch.stack([rms(delta),rms(exponent),rms(moved_base),mean(exponent*moved_base),
        mean(delta),rms(delta.expm1()),rms(dg),rms(curve),row,rms(logpotential.exp()),
        base.amin((-2,-1)),base.amax((-2,-1)),
        (delta-exponent-moved_base).abs().amax((-2,-1)),rms(dp).expand(base.shape[:3])],-1)
    logp = out.logits[:, -1].double().log_softmax(-1)
    nll = -logp.gather(1,probes[:,64,None]).squeeze(1)
    # Fixed 128 entries per layer/head, identical across widths and contexts.
    selected_log = logbase.flatten(-2).index_select(-1,coordinate).float().cpu().numpy()
    selected_power = power[0].flatten(-2).index_select(-1,coordinate).float().cpu().numpy()
    return (fields.cpu().numpy(), selected_log, selected_power, nll.cpu().numpy(),
            (logbase,power,curve))


def main():
    global STUDY
    p=argparse.ArgumentParser()
    p.add_argument('--run-id', required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--study', type=Path, default=STUDY)
    a=p.parse_args()
    STUDY=a.study.resolve()
    specpath=STUDY/'protocols'/('pilot.json' if a.pilot else 'training.json')
    spec=json.loads(specpath.read_text())
    matches=[j for j in spec['jobs'] if j['run_id']==a.run_id]
    if len(matches)!=1: raise ValueError('Expected one frozen job')
    job=matches[0]
    for name,digest in spec['producer_sources'].items():
        if sha256(REPO/name)!=digest: raise ValueError('Producer changed: '+name)
    outdir=STUDY/('qualification' if a.pilot else 'runs')/a.run_id
    outdir.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    source=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    data=ROOT/'data/refinedweb-onepass-524288/tokens.npy'
    probe=ROOT/'controlled-study-20260905/data/short'
    binding=dict(job=job,protocol_sha256=sha256(specpath),producer_sources=spec['producer_sources'],
        native_sources={f:sha256(source/f) for f in ['modeling_pldrllm.py','configuration_pldrllm.py']},
        environment=environment(),data_selection_sha256=sha256(STUDY/'selection.npz'))
    write_json(outdir/'binding.json',binding)
    selection=np.load(STUDY/'selection.npz')
    # Same stream within each initialization seed across widths and schedules.
    rows=selection[f'rows_{job["seed"]}'][:job['steps']]
    offsets=selection[f'offsets_{job["seed"]}'][:job['steps']]
    token=np.load(data,mmap_mode='r')
    batches=token[rows[:,:,None],offsets[:,:,None]+np.arange(65)]
    if len(np.unique(rows*8+offsets//64)) != rows.size: raise ValueError('Repeated training block')
    probes=torch.as_tensor(selection['probes'],dtype=torch.long,device=a.device)
    coordinates=torch.as_tensor(selection['coordinates'],dtype=torch.long,device=a.device)
    model=TrainingModel(source,job['heads'],job['seed'],a.device)
    normalize_variance_initialization(model)
    initial=parameter_digest(model)
    # Native AdamW settings are fixed across conditions; only schedule changes.
    optimizer=torch.optim.AdamW(model.model.parameters(),lr=job['peak'],betas=(.9,.95),
                                eps=1e-5,weight_decay=.1,foreach=False)
    params=list(model.model.parameters())
    start=time.time(); rawfields=[]; logbases=[]; powers=[]; probe_nll=[]
    losses=[]; gradnorm=[]; rates=[]; evals={}; completed=0
    state=None

    def snapshot():
        nonlocal state
        f,b,p,n,state=observe(model,probes[:2],coordinates,state)
        rawfields.append(f);logbases.append(b);powers.append(p);probe_nll.append(n)

    @torch.no_grad()
    def evaluate(step):
        model.model.eval(); vals=[]
        for i in range(0,len(probes),16):
            z=model.forward(probes[i:i+16,:64]).logits[:, -1]
            vals.append(torch.nn.functional.cross_entropy(z,probes[i:i+16,64],reduction='none').cpu().numpy())
        evals[str(step)]=np.concatenate(vals).tolist()

    snapshot(); evaluate(0)
    try:
        for index in range(job['steps']):
            step=index+1
            rate=lr_at(step,job['steps'],job['warmup'],job['peak'],job['schedule'])
            for group in optimizer.param_groups:group['lr']=rate
            model.model.train();optimizer.zero_grad(set_to_none=True)
            batch=torch.as_tensor(batches[index],dtype=torch.long,device=a.device)
            z=model.forward(batch[:,:64]).logits[:, -1]
            loss=torch.nn.functional.cross_entropy(z,batch[:,64])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward()
            norm=torch.stack([v.grad.detach().square().sum() for v in params if v.grad is not None]).sum().sqrt()
            if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradient')
            torch.nn.utils.clip_grad_value_(params,1.,foreach=False)
            optimizer.step(); completed=step
            losses.append(float(loss.detach()));gradnorm.append(float(norm));rates.append(rate)
            snapshot()
            if step%256==0:
                write_json(outdir/'progress.json',dict(step=step,steps=job['steps'],seconds=time.time()-start))
                print(json.dumps(dict(run_id=a.run_id,step=step,seconds=round(time.time()-start,1))),flush=True)
            if step in {job['warmup'],job['steps']//2,job['steps']}:
                evaluate(step)
            if step in spec.get('save_steps',[]):
                torch.save(dict(model=model.model.state_dict(),optimizer=optimizer.state_dict(),step=step,job=job),
                           outdir/f'state-{step}.pt')
        status='complete'; error=None
    except FloatingPointError as e:
        status='numerical_failure';error=str(e)
    np.savez_compressed(outdir/'activity.npz',fields=np.asarray(rawfields),
        logbase=np.asarray(logbases),power=np.asarray(powers),probe_nll=np.asarray(probe_nll),
        loss=losses,gradient_norm=gradnorm,lr=rates,rows=rows[:completed],offsets=offsets[:completed])
    torch.save(dict(model=model.model.state_dict(),optimizer=optimizer.state_dict(),step=completed,job=job),
               outdir/'final-state.pt')
    final=dict(binding, status=status,error=error,completed_steps=completed,field_names=FIELDS,
        parameter_count=sum(p.numel() for p in params),initial_parameter_sha256=initial,
        final_parameter_sha256=parameter_digest(model),evaluation_nll=evals,
        runtime_seconds=time.time()-start,max_cuda_memory_bytes=torch.cuda.max_memory_allocated(a.device),
        artifacts={f:sha256(outdir/f) for f in ['activity.npz','final-state.pt']})
    write_json(outdir/'manifest.json',final)
    print(json.dumps({k:final[k] for k in ['status','runtime_seconds','parameter_count','max_cuda_memory_bytes']}),flush=True)
    if status!='complete': raise SystemExit(2)


if __name__=='__main__':main()
