#!/usr/bin/env python3
"""Matched single-pass offset/first-moment interventions, with bitwise replay.

Preparation freezes inputs and executable identities. Profile and scientific
stages are distinct; the latter requires a completed 64-step width-14 profile.
"""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
from companion_paths import legacy_path
import argparse, copy, hashlib, inspect, json, os, subprocess, sys, textwrap, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from factorial_admission import preflight, freeze_corpus, producer_inventory
from model_rg.training import TrainingModel
from model_rg.schedules import optimizer_and_scheduler, loss, clip
from model_rg.controlled import stable_kl
from model_rg.provenance import sha256, write_json, environment
from train_potential_avalanches import observe, FIELDS
from model_rg.deductive_activity import DeductiveActivity, TENSOR_NAMES, STAT_NAMES

from factorial_admission import POLICY

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))
POTENTIAL=ROOT/'potential-avalanche-20260913'
DEFAULT=ROOT/'potential-factorial-disjoint-20260914'
BRANCHES=[('native_keep',1e-9,False),('raised_keep',1e-6,False),
          ('native_reset',1e-9,True),('raised_reset',1e-6,True),('replay',1e-9,False)]


def digest_state(state):
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous();h.update(str((tuple(y.shape),y.dtype)).encode());h.update(y.numpy().tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(state);return h.hexdigest()


def prepare(study,role="scientific"):
    if role not in ['scientific','qualification']:raise ValueError('Invalid role')
    study.mkdir(parents=True,exist_ok=True)
    if (study/'protocol.json').exists():raise FileExistsError('Frozen protocol already exists')
    cases=json.loads((POTENTIAL/'protocols/relaxation.json').read_text())['cases']
    token=ROOT/'data/refinedweb-onepass-524288/tokens.npy'
    data_meta=json.loads((token.parent/'manifest.json').read_text())
    if sha256(token)!=data_meta['tokens_sha256']:raise ValueError('Changed training corpus')
    selection=np.load(POTENTIAL/'selection.npz')
    streams={}
    for case in cases:
        checkpoint=Path(case['checkpoint']);case['checkpoint_sha256']=sha256(checkpoint)
        case['parent_manifest']=str(checkpoint.parent/'manifest.json')
        case['parent_manifest_sha256']=sha256(case['parent_manifest'])
        if case['kind']=='early':
            parent=json.loads(Path(case['parent_manifest']).read_text());seed=parent['job']['seed']
            stream_seed=seed+500000;start=2048
            history_path=checkpoint.parent/'activity.npz';history=np.load(history_path)
        else:
            saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
            start=saved['step'];stream_seed=saved['arguments']['stream_seed'];del saved
            history_path=checkpoint.parent/'sampling.npz';history=np.load(history_path)
        blocks=np.random.default_rng(stream_seed+1000).permutation(524288*8)[:(start+256)*32].reshape(-1,32)
        if not np.array_equal(history['rows'],blocks[:start]//8) or not np.array_equal(history['offsets'],64*(blocks[:start]%8)):
            raise ValueError('Parent consumption prefix differs')
        if np.unique(blocks).size!=blocks.size:raise ValueError('Repeated source block')
        streams[case['name']+'_rows']=blocks[start+128:]//8;streams[case['name']+'_offsets']=64*(blocks[start+128:]%8)
        case.update(start_step=int(start),unique_combined_blocks=int((start+128)*32),stream_seed=int(stream_seed),
            suffix_start_step=int(start+128),parent_history=str(history_path),parent_history_sha256=sha256(history_path))
    np.savez_compressed(study/'selection.npz',probes=selection['probes'],coordinates=selection['coordinates'],**streams)
    source_names=producer_inventory(REPO)
    assets=ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
    spec=dict(schema='potential-factorial-v2',role=role,runtime_policy=POLICY,reserved_updates=128,
        source_policy='permutation-prefix-reserved-suffix-v1',cases=cases,steps=128,profile_steps=64,corpus=freeze_corpus(ROOT),
        branches=[dict(name=n,offset=e,reset_first_moment=r) for n,e,r in BRANCHES],
        producer_sources={p:sha256(REPO/p) for p in source_names},
        native_sources={str(assets/n):sha256(assets/n) for n in ['modeling_pldrllm.py','configuration_pldrllm.py']},
        selection_sha256=sha256(study/'selection.npz'),token_sha256=sha256(token),environment=environment(),
        units='Four fixed incoming states. Branches are paired interventions, not independent pretraining replicates.',
        law='Continue the saved optimizer clock, reserve the next 4096 source blocks, and consume the following 4096 distinct blocks. No model update is performed on reserved blocks. Distinct unused source blocks per path; matched branches share the same suffix. Early paths keep their terminal positive floor rate; long paths continue the saved schedule and original objective.',
        intervention='Native positive offset 1e-9 or 1e-6 crossed with retained or zeroed first Adam moment. Preserve second moment, counter, weight decay and scheduler. Each branch computes its own nonzero native loss gradient.',
        observations='All-coordinate signed symmetric decomposition and five native tensor summaries on two fixed probes at every update; fixed 128-coordinate panel. Full vocabulary logits and 64 held-out external targets at origin and endpoint.',
        profile='64 native updates and 64 unobserved replay updates at 14 heads, memory ceiling 20 GiB; unchanged-offset forward/gradient equivalence. Profile updates excluded from scientific counts.',
        interpretation='Report every paired effect and interaction separately by state. No causal mediation, population confidence interval, critical point or exponent is inferred from this finite factorial.',
        primary='Per-state integrated log-potential activity, curvature activity, summed step KL and endpoint held-out NLL. Report immediate offset change separately. Symmetric corner effects are descriptive allocations.')
    write_json(study/'ROLE.json',dict(role=role,scientific_updates=2048 if role=='scientific' else 0,
        qualification_updates=128 if role=='scientific' else 2688,replay_updates=512 if role=='scientific' else 0))
    write_json(study/'protocol.json',spec)
    print(study/'protocol.json',flush=True)


def run_case(study,case_name,device,profile=False):
    spec,case,saved,selected,token,checked_inputs=preflight(study,case_name,profile,REPO,ROOT)
    out=study/('profile' if profile else 'runs')/case_name
    if out.exists():raise FileExistsError('Run destination already exists')
    job=saved['arguments'] if case['kind']=='continuation_parent' else saved['job']
    steps=spec['profile_steps'] if profile else spec['steps']
    rows=selected[case_name+'_rows'][:steps];offsets=selected[case_name+'_offsets'][:steps]
    batches=token[rows[:,:,None],offsets[:,:,None]+np.arange(65)]
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_device(device);torch.cuda.init();torch.cuda.reset_peak_memory_stats(device)
    probes=torch.as_tensor(selected['probes'],dtype=torch.long,device=device)
    coords=torch.as_tensor(selected['coordinates'],dtype=torch.long,device=device)
    model=TrainingModel(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1',job['heads'],job['seed'],device)
    modules=[m for m in model.model.modules() if hasattr(type(m),'cg_align_one')]
    classes=set(type(m) for m in modules)
    originals={c:c.cg_align_one for c in classes}
    source={c:textwrap.dedent(inspect.getsource(f)) for c,f in originals.items()}
    if len(modules)!=5 or any(s.count('epsilonAdj=1e-9')!=1 for s in source.values()):raise ValueError('Native offset target changed')
    def set_offset(offset):
        for c in classes:
            namespace={};code=source[c].replace('epsilonAdj=1e-9','epsilonAdj='+repr(offset))
            exec(compile(code,'<bound-native-offset>','exec'),originals[c].__globals__,namespace)
            c.cg_align_one=namespace['cg_align_one']
    def restore_native():
        for c,f in originals.items():c.cg_align_one=f
    @torch.no_grad()
    def evaluate():
        model.model.eval();logits=[]
        for i in range(0,len(probes),8):logits.append(model.forward(probes[i:i+8,:64]).logits[:,-1].detach().cpu())
        return torch.cat(logits).numpy()
    def objective(batch):
        return loss(model,batch,saved['recipe']) if case['kind']=='continuation_parent' else torch.nn.functional.cross_entropy(model.forward(batch[:,:64]).logits[:,-1],batch[:,64])
    out.mkdir(parents=True,exist_ok=False)
    t0=time.perf_counter();records={};qualification={}
    branches=[BRANCHES[0],BRANCHES[-1]] if profile else BRANCHES
    for name,offset,reset in branches:
        model.model.load_state_dict(saved['model']);restore_native();model.model.eval()
        if case['kind']=='continuation_parent':opt,sched=optimizer_and_scheduler(model,saved['recipe'])
        else:
            opt=torch.optim.AdamW(model.model.parameters(),lr=job['peak'],betas=(.9,.95),eps=1e-5,weight_decay=.1,foreach=False);sched=None
        opt.load_state_dict(copy.deepcopy(saved['optimizer']))
        if sched is not None:sched.load_state_dict(saved['scheduler'])
        before=evaluate();_,_,_,_,native_state=observe(model,probes[:2],coords)
        if profile and name=='native_keep':
            batch=torch.as_tensor(batches[0],dtype=torch.long,device=device)
            model.model.train();opt.zero_grad(set_to_none=True);objective(batch).backward()
            old_grad=digest_state([p.grad for p in model.model.parameters()])
            set_offset(offset);model.model.train();opt.zero_grad(set_to_none=True);objective(batch).backward()
            if old_grad!=digest_state([p.grad for p in model.model.parameters()]):raise ValueError('Unchanged-offset gradient mismatch')
            qualification['unchanged_offset_gradient_bitwise']=True
        set_offset(offset)
        if reset:
            preserved=digest_state([{k:v for k,v in s.items() if k!='exp_avg'} for s in opt.state.values()])
            for state in opt.state.values():state['exp_avg'].zero_()
            if preserved!=digest_state([{k:v for k,v in s.items() if k!='exp_avg'} for s in opt.state.values()]):raise ValueError('Reset changed another state coordinate')
        after=evaluate()
        if offset==1e-9 and not np.array_equal(before,after):raise ValueError('Unchanged-offset forward mismatch')
        if profile and name=='native_keep':qualification['unchanged_offset_forward_bitwise']=True
        fields=[];bases=[];powers=[];nll=[];symmetric=[];logits=[];rates=[];losses=[];state=None
        deductive=DeductiveActivity(model,probes[:2],coords)
        def snapshot():
            nonlocal state
            previous=state
            f,b,p,n,state=observe(model,probes[:2],coords,state)
            fields.append(f);bases.append(b);powers.append(p);nll.append(n)
            if previous is None:previous=state
            b0,p0,_=previous;b1,p1,_=state;dp=p1-p0;db=b1-b0
            u=dp*(b0+b1)/2;v=(p0+p1)*db/2;corner=dp*db;d=p1*b1-p0*b0
            symmetric.append(torch.stack([u.square().mean((-2,-1)),v.square().mean((-2,-1)),
                2*(u*v).mean((-2,-1)),corner.square().mean((-2,-1)),
                (d-u-v).abs().amax((-2,-1))],-1).cpu().numpy())
            deductive.snapshot();logits.append(deductive.previous_logits.cpu().numpy())
        if name!='replay':snapshot()
        _,_,_,_,intervened=observe(model,probes[:2],coords)
        q0=native_state[0]*native_state[1];q1=intervened[0]*intervened[1]
        jump=float((q1-q0).square().mean().sqrt())
        start=time.perf_counter()
        for k in range(steps):
            model.model.train();opt.zero_grad(set_to_none=True)
            batch=torch.as_tensor(batches[k],dtype=torch.long,device=device);value=objective(batch)
            if not torch.isfinite(value):raise FloatingPointError('Nonfinite gradient forcing')
            value.backward()
            if case['kind']=='continuation_parent':clip(model,saved['recipe'])
            else:torch.nn.utils.clip_grad_value_(model.model.parameters(),1.,foreach=False)
            rates.append([g['lr'] for g in opt.param_groups]);losses.append(float(value.detach()));opt.step()
            if sched is not None:sched.step()
            if name!='replay':snapshot()
            if torch.cuda.max_memory_allocated(device)>20*1024**3:raise RuntimeError('Profiled memory ceiling exceeded')
        final=evaluate();final_state=dict(model=model.model.state_dict(),optimizer=opt.state_dict(),scheduler=sched.state_dict() if sched else None)
        state_hash=digest_state(final_state)
        arrays=dict(rows=rows,offsets=offsets,lr=np.asarray(rates),loss=np.asarray(losses),
            initial_native_logits=before,initial_intervened_logits=after,final_logits=final,targets=selected['probes'][:,64])
        if name!='replay':arrays.update(fields=np.asarray(fields),logbase=np.asarray(bases),power=np.asarray(powers),
            probe_nll=np.asarray(nll),symmetric=np.asarray(symmetric),step_logits=np.asarray(logits),**deductive.arrays())
        if not all(np.isfinite(x).all() for x in arrays.values()):raise FloatingPointError('Nonfinite observation')
        np.savez_compressed(out/(name+'.npz'),**arrays)
        records[name]=dict(offset=offset,reset_first_moment=reset,steps=steps,final_state_sha256=state_hash,
            initial_logpotential_jump_rms=jump,runtime_seconds=time.perf_counter()-start,
            artifact_sha256=sha256(out/(name+'.npz')),
            objective=saved['recipe']['objective'] if sched else 'last_external_target',
            incoming_rates=[float(g['lr']) for g in saved['optimizer']['param_groups']])
        if name=='replay':
            if state_hash!=records['native_keep']['final_state_sha256']:raise ValueError('Observer changed complete final state')
            records[name]['complete_state_replay_bitwise']=True
        print(case_name,name,round(records[name]['runtime_seconds'],2),flush=True)
    write_json(out/'manifest.json',dict(status='complete',case=case,protocol_sha256=sha256(study/'protocol.json'),
        producer_sources=spec['producer_sources'],role='qualification' if profile else spec['role'],branches=records,steps_per_branch=steps,profile=profile,
        qualification=qualification,checked_inputs=checked_inputs,max_cuda_memory_bytes=torch.cuda.max_memory_allocated(device),
        runtime_seconds=time.perf_counter()-t0,field_names=FIELDS,tensor_names=TENSOR_NAMES,tensor_stat_names=STAT_NAMES,
        environment=environment()))


def queue(study):
    spec=json.loads((study/'protocol.json').read_text())
    cases=spec['cases']
    for case in cases:
        preflight(study,case['name'],False,REPO,ROOT)
    (study/'logs').mkdir(exist_ok=True)
    def worker(device,items):
        for case in items:
            cmd=[sys.executable,str(Path(__file__).resolve()),'worker','--study',str(study),'--case',case['name'],'--device',f'cuda:{device}']
            env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            with (study/'logs'/(case['name']+'.log')).open('x') as log:
                dispatch_worker(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            print(case['name'],'completed',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,d,cases[d::2]) for d in range(2)]
        for f in futures:f.result()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    actions=p.add_subparsers(dest='action',required=True)
    for action in ['prepare','profile','run','worker']:
        command=actions.add_parser(action)
        command.add_argument('--study',type=Path,default=DEFAULT)
        if action=='prepare':
            command.add_argument('--role',choices=['scientific','qualification'],default='scientific',
                help='Freeze the execution role in the new protocol; execution actions inherit it.')
        if action in ['profile','worker']:
            command.add_argument('--device',default='cuda:0')
        if action=='worker':
            command.add_argument('--case',required=True)
    a=p.parse_args(argv)
    if a.action=='prepare':prepare(a.study,a.role)
    elif a.action=='profile':run_case(a.study,'reference1',a.device,True)
    elif a.action=='run':queue(a.study)
    else:run_case(a.study,a.case,a.device)


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    main()
