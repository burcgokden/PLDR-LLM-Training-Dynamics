"""Locked finite-pulse assessment on new single-pass conditional source sequences.

The float64 control keeps stored float32 weights, Adam moments, donor direction,
and cached rotary constants as exact starting values, promotes arithmetic, and
removes the executed forced-float32 attention and reference-RoPE input casts.
It is a different numerical implementation, not a real-arithmetic certificate.
"""
from companion_paths import configured_path
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import textwrap
import time
import types
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(os.environ.get('MODEL_RG_DATA_ROOT',configured_path('data:model'))).resolve()
OUT=ROOT/'finite-response-20260912/assessment'
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.training import TrainingModel
from model_rg.onepass_regimes import optimizer_and_scheduler
from model_rg.criticality import generator_parameter
from model_rg.schedules import loss as native_loss
from measure_law_closure import digest_state

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,obj):
    Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    old=ROOT/'outer-transfer-20260911'
    parent=json.loads((old/'protocol.json').read_text())
    cases=[c for c in parent['cases'] if c['corpus']==0 and c['identity']==0]
    inputs={str(old/'protocol.json'):sha(old/'protocol.json'),str(old/'data/panels.npz'):sha(old/'data/panels.npz'),
        str(old/'data/corpus-0.npy'):sha(old/'data/corpus-0.npy')}
    for c in cases:
        state=old/'runs'/c['name']/'incoming-state.pt';inputs[str(state)]=sha(state)
        order=np.random.default_rng(c['stream_seed']).permutation(524288)
        rng=np.random.default_rng(913130000+c['heads'])
        blocks=np.stack([rng.choice(order[65536:],32*8,replace=False).reshape(8,32) for _ in range(4)])
        if any(len(np.unique(b))!=256 for b in blocks) or np.isin(blocks,order[:65536]).any():raise ValueError('Source overlap')
        np.save(OUT/(c['name']+'-blocks.npy'),blocks)
        inputs[str(OUT/(c['name']+'-blocks.npy'))]=sha(OUT/(c['name']+'-blocks.npy'))
    native=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    for name in ['modeling_pldrllm.py','configuration_pldrllm.py']:inputs[str(native/name)]=sha(native/name)
    names=set(parent['sources'])|{'scripts/directional_study.py','scripts/measure_law_closure.py'}
    sources={str(REPO/n):sha(REPO/n) for n in names}
    arms=[dict(name='zero',family=None,amplitude=0.,multiplier=0.)]
    for family in ['generator','body']:
        for h in [3e-7]:
            for tag,m in [('plus',1.),('minus',-1.),('halfplus',.5),('halfminus',-.5)]:
                arms.append(dict(name=f'{family}-{h}-{tag}',family=family,amplitude=h,multiplier=m))
    spec=dict(schema='finite-response-assessment-v1',status='frozen',created_at=datetime.now(timezone.utc).isoformat(),
        cases=cases,arms=arms,times=[0,1,2,4,8],horizon=8,batch_size=32,source_replicates=4,
        modes=['native32','arithmetic64'],native_updates=576,arithmetic_control_updates=576,replay_updates=32,
        primary='Locked h=3e-7 from separate instantaneous development. Both directions at both incoming states, four fresh sources, complete eight-update path. Report every native and control cell. Uniform assessment succeeds only if every native cell is resolved and d<=0.1. No extension unless the complete primary succeeds.',
        comparison='Equal weighting on times [0,1,2,4,8] and twelve unit-scaled coordinates: squared norm sum/(5*12). d=norm(O_h-2O_half)/norm(2O_half), d<=0.1 and norm(O_half)>5e-7. A signal below the floor is unresolved, never a pass.',
        precision_control='Stored parameters, moments, fixed float32 donor directions and cached rotary constants promoted exactly. Only executed softmax and reference-RoPE input forced casts removed in memory. Counter values unchanged.',
        limits='Four new conditional source sequences at each of two paired widths, one shared corpus/initialization identity; paired precision controls and signs are dependent. No population spectrum or independent-model replication.',
        pulse_convention='Materialize delta=direction*coefficient, then add_(delta) in each native parameter dtype.',
        source_seed_base=913130000, gate=dict(relative_tolerance=.1,signal_floor=5e-7),
        selected_radius_basis='Separate incoming-state diagnostic; time-zero repeats are not new confirmation. Fresh assessment is the complete eight-update path.',
        activation_contract='Native SiLU and x*SiLU(x)+1e-9 are smooth in real arithmetic. Diagnostics monitor power-base minima, finite outputs and Adam second-moment zeros; they do not assert absence of all possible derivative singularities.',
        producer_sha256=sha(__file__),sources=sources,inputs=inputs)
    names.add('scripts/finite_response_study.py')
    for n in sorted(names):
        target=OUT/'executed-source'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/n).read_bytes())
    spec['sources']={str(REPO/n):sha(REPO/n) for n in names}
    spec['analysis_source_sha256']=sha(REPO/'scripts/analyze_finite_response.py')
    write(OUT/'protocol.json',spec)
    print('Frozen 576 native and 576 arithmetic-control updates plus 32 replay updates',flush=True)

def observe(model,crops):
    model.model.eval()
    with torch.no_grad():
        x=torch.as_tensor(crops,dtype=torch.long,device=model.device)
        out=model.forward(x[:,:64],capture=True)
        logits=out.logits[:,-1].double();logp=logits.log_softmax(-1)
        nll=-logp.gather(1,x[:,64,None]).squeeze(1)
        entropy=-(logp.exp()*logp).sum(-1)
        common=[];transverse=[]
        for layer in out.pldr_attentions:
            matrix=layer[0].double();center=matrix.mean(-2,keepdim=True)
            common.append(center.square().mean((-2,-1)))
            transverse.append((matrix-center).square().mean((-2,-1)))
        common=torch.stack(common);transverse=torch.stack(transverse)
        q=torch.cat((torch.stack([nll.mean(),entropy.mean()]),torch.log1p(common.mean((1,2)).sqrt()),torch.log1p(transverse.mean((1,2)).sqrt())))
        ans={k:v.cpu().numpy() for k,v in dict(q=q,nll=nll,entropy=entropy,common_ms=common,transverse_ms=transverse,logits=logits).items()}
    model.capture=False;model.head_outputs={}
    return ans

def promote_control(model):
    if not model.config.reference_rope:raise ValueError('Expected reference cached rotary path')
    model.model.double()
    src=inspect.getsource(model._attention)
    if src.count('dtype=torch.float32')!=1:raise ValueError('Softmax cast contract changed')
    ns=dict(model.module.__dict__)
    exec(src.replace('dtype=torch.float32','dtype=query.dtype'),ns)
    model._attention=ns['eager_attention_forward']
    count=0
    for module in model.model.modules():
        if module.__class__.__name__=='RotaryPositionalEmbeddings':
            src=textwrap.dedent(inspect.getsource(type(module).forward))
            if src.count('x.float()')!=1:raise ValueError('Rotary cast contract changed')
            ns=dict(model.module.__dict__);exec(src.replace('x.float()','x'),ns)
            module.forward=types.MethodType(ns['forward'],module);count+=1
    if count!=5:raise ValueError(f'Expected five reference rotary modules, found {count}')
    return dict(softmax_casts_removed=1,rotary_input_casts_removed=count,cached_rotary_constants='unchanged exact stored values promoted to float64')

def worker(heads,device):
    spec=json.loads((OUT/'protocol.json').read_text())
    if sha(__file__)!=spec['producer_sha256']:raise ValueError('Changed producer')
    for p,h in {**spec['sources'],**spec['inputs']}.items():
        if sha(p)!=h:raise ValueError('Changed input '+p)
    case=next(c for c in spec['cases'] if c['heads']==heads)
    dest=OUT/case['name'];dest.mkdir(exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    old=ROOT/'outer-transfer-20260911';ck=torch.load(old/'runs'/case['name']/'incoming-state.pt',map_location='cpu',weights_only=False)
    if ck['step']!=2048:raise ValueError('Wrong incoming age')
    corpus=np.load(old/'data/corpus-0.npy',mmap_mode='r')
    with np.load(old/'data/panels.npz') as f:crops=f['evaluation'];donors=f['donors']
    all_blocks=np.load(OUT/(case['name']+'-blocks.npy'));start=time.perf_counter();results=[]
    directions=None;norms={}
    for mode in spec['modes']:
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        model=TrainingModel(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1',heads,case['seed'],device)
        promotion=promote_control(model) if mode=='arithmetic64' else None
        diagnostics=[]
        original_activation=model.module.iSwiGLU
        def monitored_activation(x):
            y=original_activation(x)
            with torch.no_grad():
                diagnostics.append([float(x.detach().abs().min()),float(y.detach().min()),int((~torch.isfinite(y)).sum())])
            return y
        model.module.iSwiGLU=monitored_activation
        optimizer,scheduler=optimizer_and_scheduler(model,case['profile'])
        def restore():
            optimizer.zero_grad(set_to_none=True);model.capture=False;model.eta=None;model.head_outputs={}
            model.model.load_state_dict(ck['model']);optimizer.load_state_dict(copy.deepcopy(ck['optimizer']));scheduler.load_state_dict(copy.deepcopy(ck['scheduler']))
        restore();initial=digest_state(model,optimizer,scheduler)
        if mode=='native32':
            model.model.train();loss=native_loss(model,torch.as_tensor(donors,dtype=torch.long,device=device),case['profile']);loss.backward()
            directions={}
            for family in ['generator','body']:
                group=[(n,p) for n,p in model.model.named_parameters() if generator_parameter(n)==(family=='generator')]
                norms[family]=sum(p.detach().double().square().sum() for n,p in group).sqrt().item()
                gn=sum(p.grad.detach().double().square().sum() for n,p in group).sqrt().item()
                directions[family]={n:(-p.grad.detach()/gn).cpu() for n,p in group}
            del group,loss
            optimizer.zero_grad(set_to_none=True);torch.save(dict(directions=directions,norms=norms),dest/'directions.pt')
        directions_gpu={family:{n:d.to(device=device,dtype=next(model.model.parameters()).dtype) for n,d in group.items()} for family,group in directions.items()}
        modeout=dest/mode;modeout.mkdir()
        zero=None;zero_digest=None;zero_grads=None
        jobs=[(source,arm) for source in range(spec['source_replicates']) for arm in spec['arms']]
        jobs.append((0,dict(name='zero-replay',family=None,amplitude=0.,multiplier=0.)))
        for source,arm in jobs:
            blocks=all_blocks[source]
            diagnostics.clear()
            if time.perf_counter()-start>1800:raise TimeoutError('30 minute worker ceiling reached')
            restore()
            if digest_state(model,optimizer,scheduler)!=initial:raise ValueError('Restore failed')
            family=arm['family'];displacement_error=0.
            if family:
                coeff=arm['amplitude']*arm['multiplier']*norms[family];err2=0.
                with torch.no_grad():
                    for name,p in model.model.named_parameters():
                        if name not in directions_gpu[family]:continue
                        prior=p.clone();delta=directions_gpu[family][name]*coeff;p.add_(delta)
                        err2+=(p.double()-prior.double()-delta.double()).square().sum().item()
                    del prior,delta
                displacement_error=np.sqrt(err2)/abs(coeff)
            observations=[observe(model,crops)];gradnorms=[];losses=[];moment_stats=[]
            for t,ids in enumerate(blocks,1):
                model.model.train();optimizer.zero_grad(set_to_none=True)
                arr=corpus[(ids//8)[:,None],(64*(ids%8))[:,None]+np.arange(65)]
                loss=native_loss(model,torch.as_tensor(arr,dtype=torch.long,device=device),case['profile'])
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite native loss')
                loss.backward()
                with torch.no_grad():
                    values=[state['exp_avg_sq'] for state in optimizer.state.values() if 'exp_avg_sq' in state]
                    moment_stats.append([min(float(v.min()) for v in values),sum(int((v==0).sum()) for v in values),sum(v.numel() for v in values)])
                gn=torch.nn.utils.clip_grad_norm_(model.model.parameters(),1.,error_if_nonfinite=True)
                gradnorms.append(float(gn));losses.append(float(loss.detach()));optimizer.step();scheduler.step()
                if t in spec['times']:observations.append(observe(model,crops))
            arrays={k:np.stack([r[k] for r in observations]) for k in observations[0]}
            final=digest_state(model,optimizer,scheduler)
            if arm['name']=='zero' and source==0:zero=arrays;zero_digest=final;zero_grads=gradnorms
            if arm['name']=='zero-replay':
                if final!=zero_digest or gradnorms!=zero_grads or not all(np.array_equal(v,zero[k]) for k,v in arrays.items()):raise ValueError('Replay failed')
            path=modeout/(f'source{source}-'+arm['name']+'.npz');np.savez(path,**arrays,gradnorms=gradnorms,losses=losses,blocks=blocks,times=spec['times'],activation_diagnostics=np.array(diagnostics),moment_stats=np.array(moment_stats))
            results.append(dict(mode=mode,source=source,arm=arm['name'],file=str(path),sha256=sha(path),final_digest=final,
                relative_pulse_rounding_error=displacement_error,gradnorms=gradnorms,clipped=[v>1 for v in gradnorms]))
            print(case['name'],mode,source,arm['name'],round(time.perf_counter()-start,1),'seconds',flush=True)
        write(modeout/'metadata.json',dict(mode=mode,promotion=promotion,incoming_digest=initial,replay_bitwise=True,
            parameter_dtypes=sorted({str(p.dtype) for p in model.model.parameters()}),
            moment_dtypes=sorted({str(v.dtype) for s in optimizer.state.values() for k,v in s.items() if k in ['exp_avg','exp_avg_sq']}),
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device)))
        del model,optimizer,scheduler,directions_gpu,arrays,observations,loss,gn
        torch.cuda.empty_cache()
    write(dest/'results.json',dict(status='complete',case=case['name'],seconds=time.perf_counter()-start,records=results,
        native_updates=288,arithmetic_control_updates=288,replay_updates=16,protocol_sha256=sha(OUT/'protocol.json'),directions_sha256=sha(dest/'directions.pt')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker']);ap.add_argument('--heads',type=int);ap.add_argument('--device')
    a=ap.parse_args()
    if a.action=='prepare':prepare()
    else:worker(a.heads,a.device)
