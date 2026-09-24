#!/usr/bin/env python3
"""Completed-design producer: initial-to-trained cache transport on fresh prefixes."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
import time
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.criticality import fix_shared_generator, parameter_digest
from model_rg.variance_family import normalize_variance_initialization
from model_rg.inference_interventions import fixed_operators, operator_cache
from numerical_validation import load_json_strict, finite_array
from context_reservations import reserve
from context_categorical_study import ROOT, NATIVE, PARENT, PROBES, CORPUS
from cache_state_contract import SOURCES, STATES, CONTRACT, validate_protocol, artifacts, read

REPO = Path(__file__).resolve().parents[1]
def prepare(study, start, reproduce=None):
    if study.exists(): raise FileExistsError(study)
    receipt = reserve(study, 80, start, reproduce=reproduce)
    rows = receipt['context_rows']; hashes = receipt['document_hashes']
    exclusions = [Path(x) for x in receipt['imported_protocols']]
    parent = read(PARENT/'protocol.json')
    selected = [j for j in parent['jobs'] if j['heads'] in [8, 24] and j['control'] in [0, 1.5]]
    jobs = []; bound = {}
    for heads in [8, 24]:
        for seed in sorted({j['seed'] for j in selected if j['heads'] == heads}):
            endpoints = {}
            for control in [0, 1.5]:
                job = next(j for j in selected if (j['heads'], j['seed'], j['control']) == (heads, seed, control))
                path = PARENT/'runs'/job['run_id']/'manifest.json'; m = read(path)
                if m['status'] != 'complete' or m['job'] != job: raise ValueError('Invalid parent state')
                bound[str(path)] = sha256(path)
                endpoints[f'g{control:g}'] = dict(run_id=job['run_id'], checkpoint_sha256=m['artifacts']['final-state.pt'])
                initial = m['initial_parameter_sha256']; shared = m['initial_shared_sha256']
                if control == 0: initial_reference = (initial, shared, job['shared_seed'])
                elif (initial, shared, job['shared_seed']) != initial_reference: raise ValueError('Unpaired initialization')
            jobs.append(dict(run_id=f'h{heads}-s{seed}', heads=heads, seed=seed,
                shared_seed=initial_reference[2], initial_parameter_sha256=initial_reference[0],
                initial_shared_sha256=initial_reference[1], endpoints=endpoints))
    if len(jobs) != 12: raise ValueError('Incomplete paired grid')
    for path in [PROBES/'tokens.npy', PROBES/'records.json', CORPUS/'records.json',
                 PARENT/'protocol.json', NATIVE/'modeling_pldrllm.py', NATIVE/'configuration_pldrllm.py', *exclusions]:
        bound[str(path)] = sha256(path)
    study.mkdir(parents=True)
    blocks = np.array(np.load(PROBES/'tokens.npy', mmap_mode='r')[rows, :129])
    np.savez_compressed(study/'inputs.npz', blocks=blocks, rows=rows)
    p = dict(schema='cache-state-transfer-v3', design_id='rg-cache-state-transfer-v1',
        implementation=CONTRACT, excluded_context_protocols=sorted(str(x) for x in exclusions),
        status='frozen_before_acquisition', parent=str(PARENT), jobs=jobs, states=STATES,
        prefix_lengths=[32, 64, 128], contexts=64, calibration_contexts=16, vocabulary=32000,
        document_hashes=hashes, context_rows=rows, selection_sha256=sha256(study/'inputs.npz'),
        source_sha256={name: sha256(REPO/name) for name in SOURCES}, input_sha256=bound,
        cache_rule='Mean A, A_LM, G in float64 over 16 calibration documents, cast float32; one cache per state and length.',
        comparison='Every state uses its own recalibrated cache. Each trained endpoint also uses the cache of its paired initial state. All arms retained; no assumption of small stale-cache error.',
        hypotheses=['Exact empirical risk separates assessment scatter and cache displacement.',
          'Initial-to-trained displacement equals mean drift, initial calibration displacement and their signed cross term.',
          'State-specific recalibration meets fixed aggregate centered RMS 0.25 and mean KL 0.03 nats on the fresh panel.'],
        targets=dict(relative_centered_rms=.25, mean_kl=.03),
        target_exclusion='At length L, only tokens [:L] enter the model; token L is an external NLL target.',
        conditioning='Fixed corpus, source order and shared generator initialization. Six paired wide initializations per width; controls, ages, prefixes and documents do not add training replicas.',
        training_updates=0, new_training_replicas=0, expected_calls=dict(calibration=216,
            native_assessment=864, recalibrated_assessment=864, initial_cache_assessment=576, qualification=504),
        worker_hour_cap=2, memory_ceiling_bytes=22*1024**3, disk_budget_bytes=4*1024**3)
    write_json(study/'reservation.json', receipt)
    p['reservation_sha256'] = sha256(study/'reservation.json')
    write_json(study/'protocol.json', p)
    for name in SOURCES:
        dest=study/'executed-source'/name; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes((REPO/name).read_bytes())
    validate_protocol(study, acquisition=True)
    print({'study':str(study), 'calls':p['expected_calls'], 'documents':rows}, flush=True)


def admit(study, index=None):
    return validate_protocol(study, acquisition=True, index=index)


@torch.no_grad()
def worker(study, index, device):
    if type(index) is not int: raise ValueError('An explicit integer worker index is required')
    if device not in ['cuda:0','cuda:1']: raise ValueError('Invalid worker device')
    p=admit(study,index); job=p['jobs'][index]; out=study/'runs'/job['run_id']; out.mkdir(parents=True, exist_ok=False)
    started=time.monotonic(); counts={k:0 for k in p['expected_calls']}; artifacts=[]
    m=dict(status='started', job=job, protocol_sha256=sha256(study/'protocol.json'), training_updates=0, device=device)
    write_json(out/'manifest.json',m)
    try:
        torch.set_num_threads(2); torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
        torch.cuda.reset_peak_memory_stats(device)
        if torch.cuda.mem_get_info(device)[0] < 4*1024**3: raise RuntimeError('Insufficient free device memory')
        model=TrainingModel(NATIVE,job['heads'],job['seed'],device)
        normalize_variance_initialization(model); shared=fix_shared_generator(model,job['shared_seed'])
        initial=parameter_digest(model)
        if (initial,shared)!=(job['initial_parameter_sha256'],job['initial_shared_sha256']): raise ValueError('Initial parameter identity differs')
        model.model.eval().requires_grad_(False)
        with np.load(study/'inputs.npz') as a: blocks=a['blocks']
        def forward(x,role,capture=False):
            counts[role]+=1; value=model.forward(x,capture=capture)
            if torch.cuda.max_memory_allocated(device)>=p['memory_ceiling_bytes']: raise RuntimeError('Memory ceiling exceeded')
            if time.monotonic()-started>600: raise RuntimeError('Per-worker time exceeded')
            if not torch.isfinite(value.logits).all(): raise ValueError('Nonfinite native output')
            return value
        caches={}; statistics={}; qualification=[]
        for state_name in STATES:
            if state_name!='initial':
                source=job['endpoints'][state_name]; path=PARENT/'runs'/source['run_id']/'final-state.pt'
                if sha256(path)!=source['checkpoint_sha256']: raise ValueError('Changed endpoint')
                state=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
                model.model.load_state_dict(state['model'],strict=True); del state
            # Admit longest prefix with batch one before batch eight for each state.
            x=torch.as_tensor(blocks[:1,:128].copy(),device=device,dtype=torch.long)
            y=forward(x,'qualification',True); own=operator_cache(y); z=y.logits[:,-1].clone(); del y
            with fixed_operators(model,own): q=forward(x,'qualification').logits[:,-1]
            if not torch.equal(z,q): raise ValueError('Longest-prefix own replay differs')
            del x,own,z,q
            for length in p['prefix_lengths']:
                x=torch.as_tensor(blocks[:,:length].copy(),device=device,dtype=torch.long)
                parts=[]
                for offset in [0,8]:
                    y=forward(x[offset:offset+8],'calibration',True); parts.append(operator_cache(y).cpu()); del y
                cache=torch.cat(parts,dim=2).double().mean(2,keepdim=True).float(); del parts
                caches[state_name,length]=cache
                calls=[0]
                def counted(*_): calls[0]+=1
                handles=[layer.register_forward_hook(counted) for dec in model.model.decoder.dec_layers for layer in dec.mha1.reslayerAs]
                y=forward(x[:8],'qualification',True); own=operator_cache(y); z=y.logits[:,-1].clone(); del y
                if calls[0]!=40: raise ValueError('Native generator count')
                calls[0]=0
                with fixed_operators(model,own): q=forward(x[:8],'qualification').logits[:,-1]
                if calls[0] or not torch.equal(z,q): raise ValueError('Exact own replay failed')
                restored=forward(x[:8],'qualification').logits[:,-1]
                if calls[0]!=40 or not torch.equal(z,restored): raise ValueError('Exact restoration failed')
                calls[0]=0
                with fixed_operators(model,cache.to(device)): q=forward(x[:8],'qualification').logits[:,-1]
                if calls[0]: raise ValueError('Generator bypass failed')
                for handle in handles: handle.remove()
                qualification.append(dict(state=state_name,length=length,own_replay=True,restoration=True,native_generator_calls=40,cached_generator_calls=0))
                del own,z,q,restored
                gs=[]; logits=[]
                for offset in range(16,80,8):
                    y=forward(x[offset:offset+8],'native_assessment',True)
                    logits.append(y.logits[:,-1].cpu().numpy()); gs.append(operator_cache(y)[:,2].cpu()); del y
                name=f'{state_name}-L{length}-native.npy';np.save(out/name,np.concatenate(logits));artifacts.append(name);del logits
                g=torch.cat(gs,dim=1).double();del gs
                mean=g.mean(1)
                values=dict(mean=mean.numpy(),powers=(g*g).mean(dim=(2,3,4)).numpy(),scatter=((g-mean[:,None])**2).mean(dim=(1,2,3,4)).numpy())
                for policy in (['recalibrated'] if state_name=='initial' else ['recalibrated','initial_cache']):
                    c=cache if policy=='recalibrated' else caches['initial',length];cg=c[:,2,0].double()
                    values[policy+'-risk']=((g-cg[:,None])**2).mean(dim=(1,2,3,4)).numpy()
                    values[policy+'-displacement']=((mean-cg)**2).mean(dim=(1,2,3)).numpy()
                    logits=[]
                    with fixed_operators(model,c.to(device)):
                        for offset in range(16,80,8):
                            y=forward(x[offset:offset+8],policy+'_assessment');logits.append(y.logits[:,-1].cpu().numpy());del y
                    name=f'{state_name}-L{length}-{policy}.npy';np.save(out/name,np.concatenate(logits));artifacts.append(name);del logits
                name=f'{state_name}-L{length}-operators.npz';np.savez_compressed(out/name,**values);artifacts.append(name)
                statistics[state_name,length]=values['mean'];del g,mean,x,values
        name='caches.npz';np.savez_compressed(out/name,**{f'{s}-L{l}':c.numpy() for (s,l),c in caches.items()});artifacts.append(name)
        expected={k:v//12 for k,v in p['expected_calls'].items()}
        if counts!=expected: raise ValueError('Forward ledger differs')
        m.update(status='complete',initial_parameter_sha256=initial,initial_shared_sha256=shared,qualification=qualification,
            longest_batch_one=True,artifacts={n:sha256(out/n) for n in artifacts})
    except Exception as error:
        m.update(status='failed',error=repr(error));raise
    finally:
        m.update(calls=counts,elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(out/'manifest.json',m)
    print({'run':job['run_id'],'seconds':m['elapsed_seconds'],'calls':counts},flush=True)


def run(study):
    p=admit(study)
    logs=study/'logs';logs.mkdir(exist_ok=True)
    def queue(device,indices):
        for index in indices:
            folder=study/'runs'/p['jobs'][index]['run_id']
            if folder.exists():
                m=read(folder/'manifest.json')
                if m['status']!='complete' or m['protocol_sha256']!=sha256(study/'protocol.json') or m['job']!=p['jobs'][index]: raise ValueError('Partial or foreign run')
                if set(m['artifacts'])!=artifacts(): raise ValueError('Incomplete completed artifact coverage')
                if any(sha256(folder/n)!=h for n,h in m['artifacts'].items()): raise ValueError('Changed completed output')
                continue
            elapsed=sum(read(f).get('elapsed_seconds',0) for f in (study/'runs').glob('*/manifest.json'))
            if elapsed+1200>p['worker_hour_cap']*3600: raise RuntimeError('Worker budget exceeded')
            if sum(f.stat().st_size for f in study.rglob('*') if f.is_file())>p['disk_budget_bytes']: raise RuntimeError('Disk budget exceeded')
            with (logs/(p['jobs'][index]['run_id']+'.log')).open('x') as log:
                dispatch_worker([sys.executable,__file__,'worker','--study',str(study),'--index',str(index),'--device',device],
                    cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),
                    stdout=log,stderr=subprocess.STDOUT,check=True,timeout=660)
            print('COMPLETE '+p['jobs'][index]['run_id'],flush=True)
    widest=next(i for i,j in enumerate(p['jobs']) if j['heads']==24)
    queue('cuda:0',[widest]); remaining=[i for i in range(12) if i!=widest]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,f'cuda:{d}',remaining[d::2]) for d in range(2)]
        for future in futures: future.result()


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','worker','run'])
    p.add_argument('--study',type=Path,required=True);p.add_argument('--start',type=int)
    p.add_argument('--index',type=int);p.add_argument('--device',default='cuda:0');p.add_argument('--reproduce-panel',type=Path);a=p.parse_args();study=a.study.resolve()
    if study==ROOT or not study.is_relative_to(ROOT): raise ValueError('Authorized experiment destination required')
    if a.action=='prepare':prepare(study,a.start,a.reproduce_panel)
    elif a.action=='run':run(study)
    else:worker(study,a.index,a.device)
