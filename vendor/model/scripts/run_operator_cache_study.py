#!/usr/bin/env python3
"""Frozen paired native/cached PLGA inference on unused proper-prefix documents."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.inference_interventions import fixed_operators, operator_cache
from numerical_validation import load_json_strict, finite_array
from context_reservations import reserve
from context_execution_contract import attach, admit as shared_admit
from context_categorical_study import ROOT, NATIVE, PARENT, PROBES, CORPUS

REPO=Path(__file__).resolve().parents[1]
SOURCES=['scripts/run_operator_cache_study.py','scripts/analyze_operator_cache.py',
 'scripts/context_categorical_study.py','scripts/numerical_validation.py',
 'src/model_rg/training.py','src/model_rg/native.py','src/model_rg/inference_interventions.py',
 'src/model_rg/provenance.py']

def prepare(study,start,reproduce=None):
    if study.exists():raise FileExistsError(study)
    receipt = reserve(study, 144, start, reproduce=reproduce)
    rows = np.array(receipt['context_rows']); hashes = receipt['document_hashes']
    exclusions = [Path(x) for x in receipt['imported_protocols']]
    tokens=np.load(PROBES/'tokens.npy',mmap_mode='r');offsets=np.load(PROBES/'offsets.npy')
    blocks=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    parent=load_json_strict((PARENT/'protocol.json').read_text())
    jobs=[dict(j) for j in parent['jobs'] if j['heads'] in [8,24] and j['control'] in [0,1.5]]
    if len(jobs)!=24:raise ValueError('Incomplete endpoint grid')
    bound={str(f):sha256(f) for f in [PARENT/'protocol.json',PROBES/'records.json',PROBES/'tokens.npy',
        PROBES/'offsets.npy',CORPUS/'records.json',NATIVE/'modeling_pldrllm.py',NATIVE/'configuration_pldrllm.py',*exclusions]}
    for job in jobs:
        path=PARENT/'runs'/job['run_id']/'manifest.json';m=load_json_strict(path.read_text())
        if m['status']!='complete' or m['job']!=job:raise ValueError('Invalid parent')
        job['checkpoint_sha256']=m['artifacts']['final-state.pt'];bound[str(path)]=sha256(path)
    study.mkdir(parents=True)
    np.savez_compressed(study/'inputs.npz',blocks=blocks,rows=rows)
    p=dict(schema='operator-cache-v1',status='frozen_before_acquisition',parent=str(PARENT),
        jobs=jobs,contexts=128,calibration_contexts=16,context_rows=rows.tolist(),document_hashes=hashes,
        calibration_rows=rows[:16].tolist(),evaluation_rows=rows[16:].tolist(),batch_size=8,
        prefix_length=64,external_target_position=64,vocabulary=32000,
        cache_rule='Float64 arithmetic mean of native A, A_LM and G over 16 calibration prefixes, cast to float32. Fixed per checkpoint; no assessment refit.',
        inference='Author external-G branch; recomputed Q/K/V and causal mask; no KV cache; no metric-generator execution.',
        primary_targets=dict(centered_rms=.25,mean_kl_nats=.03,median_latency_reduction=.10),
        timing=dict(warmup_pairs=10,measured_pairs=30,alternating_order=True,synchronized=True,
            unit='Batch of eight 64-token prefixes; final-token logits only; steady-state model call, excluding one-time cache installation.'),
        selection='All 24 retained endpoints; all 128 assessment contexts; no outcome selection.',
        independence='Six paired initialization identities per condition; contexts and timing repeats are not extra training replicas.',
        arithmetic='float32 native; TF32 disabled; float64 cache mean and reductions',
        training_updates=0,new_training_replicas=0,expected_calls=dict(calibration=48,assessment=768,timing=1920,qualification=96),
        worker_hour_cap=3,input_sha256=bound,selection_sha256=sha256(study/'inputs.npz'),
        source_sha256={n:sha256(REPO/n) for n in SOURCES})
    write_json(study/'protocol.json',p)
    for name in SOURCES:
        dest=study/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((REPO/name).read_bytes())
    attach(study, p, receipt)
    print(json.dumps({'study':str(study),'protocol_sha256':sha256(study/'protocol.json'),'calls':p['expected_calls']}),flush=True)

def admit(study, index=None):
    return shared_admit(study, index)


@torch.no_grad()
def worker(study,index,device):
    if type(index) is not int or device not in ['cuda:0','cuda:1']: raise ValueError('Explicit worker index/device required')
    p=admit(study,index);job=p['jobs'][index];out=study/'runs'/job['run_id'];out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();counts=dict(calibration=0,assessment=0,timing=0,qualification=0)
    m=dict(status='started',job=job,protocol_sha256=sha256(study/'protocol.json'),device=device,training_updates=0)
    write_json(out/'manifest.json',m)
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_device(device)
        checkpoint=Path(p['parent'])/'runs'/job['run_id']/'final-state.pt'
        if sha256(checkpoint)!=job['checkpoint_sha256']:raise ValueError('Changed checkpoint')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model=TrainingModel(NATIVE,job['heads'],job['seed'],device)
        model.model.load_state_dict(state['model'],strict=True);model.model.eval().requires_grad_(False);del state
        with np.load(study/'inputs.npz') as a:blocks=a['blocks']
        tensors=torch.as_tensor(blocks[:,:64].copy(),dtype=torch.long,device=device)
        # Calibration excludes the target token and all assessment documents.
        torch.cuda.synchronize(device);beg=time.perf_counter();parts=[]
        for i in [0,8]:
            output=model.forward(tensors[i:i+8],capture=True);counts['calibration']+=1
            parts.append(operator_cache(output).double().sum(2,keepdim=True));del output
        cache=((parts[0]+parts[1])/16).float();del parts
        torch.cuda.synchronize(device);build_seconds=time.perf_counter()-beg
        np.savez_compressed(out/'cache.npz',cache=cache.cpu().numpy())
        # Every checkpoint: native replay, restoration and actual generator bypass.
        gen_calls=[0]
        def counted(*_):gen_calls[0]+=1
        handles=[layer.register_forward_hook(counted) for dec in model.model.decoder.dec_layers for layer in dec.mha1.reslayerAs]
        x=tensors[:8]
        native=model.forward(x,capture=True);counts['qualification']+=1
        own=operator_cache(native);z=native.logits[:,-1].clone();del native
        if gen_calls[0]!=40:raise ValueError('Incomplete native generator execution')
        gen_calls[0]=0
        with fixed_operators(model,own):
            exact=model.forward(x).logits[:,-1];counts['qualification']+=1
        if gen_calls[0]!=0 or not torch.equal(z,exact):raise ValueError('Native external-G replay is not exact')
        restored=model.forward(x).logits[:,-1];counts['qualification']+=1
        if gen_calls[0]!=40 or not torch.equal(z,restored):raise ValueError('Native state restoration differs')
        gen_calls[0]=0
        with fixed_operators(model,cache):
            check=model.forward(x).logits[:,-1];counts['qualification']+=1
        if gen_calls[0]!=0 or not torch.isfinite(check).all():raise ValueError('Cached path did not bypass the generator')
        for h in handles:h.remove()
        del own,z,exact,restored,check
        observations={name:[] for name in ['native','cached']}
        for offset in range(16,144,8):
            names=['native','cached'] if (offset//8+index)%2==0 else ['cached','native']
            for name in names:
                with fixed_operators(model,cache) if name=='cached' else nullcontext():
                    z=model.forward(tensors[offset:offset+8]).logits[:,-1].cpu().numpy();counts['assessment']+=1
                finite_array(z,'assessment logits');observations[name].append(z)
        np.savez_compressed(out/'observations.npz',**{k:np.concatenate(v) for k,v in observations.items()})
        del observations,z
        # Paired, alternating, synchronized timings of the same resident model.
        timings={name:[] for name in ['native','cached']};peaks={name:[] for name in timings}
        x=tensors[16:24]
        for iteration in range(40):
            names=['native','cached'] if (iteration+index)%2==0 else ['cached','native']
            for name in names:
                with fixed_operators(model,cache) if name=='cached' else nullcontext():
                    torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
                    begin=time.perf_counter();output=model.forward(x)
                    torch.cuda.synchronize(device);elapsed=time.perf_counter()-begin
                    peak=torch.cuda.max_memory_allocated(device);del output
                    counts['timing']+=1
                if iteration>=10:timings[name].append(elapsed);peaks[name].append(peak)
        np.savez_compressed(out/'timing.npz',**{k:np.asarray(v) for k,v in timings.items()},
                            **{k+'_peak_bytes':np.asarray(v) for k,v in peaks.items()})
        m.update(status='complete',cache_build_seconds=build_seconds,cache_bytes=cache.numel()*cache.element_size(),
            qualification=dict(exact_own_operator_replay=True,exact_native_restoration=True,native_generator_calls=40,cached_generator_calls=0),
            artifacts={name:sha256(out/name) for name in ['observations.npz','cache.npz','timing.npz']},
            checkpoint=str(checkpoint),checkpoint_sha256=job['checkpoint_sha256'])
    except Exception as error:
        m.update(status='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        m.update(calls=counts,elapsed_seconds=time.monotonic()-started)
        write_json(out/'manifest.json',m)
    print(json.dumps({'run':job['run_id'],'seconds':m['elapsed_seconds'],'counts':counts}),flush=True)

def run(study):
    p=admit(study)
    for name,digest in p['input_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Changed scientific input '+name)
    logs=study/'logs';logs.mkdir(exist_ok=True)
    def queue(device,indices):
        for index in indices:
            folder=study/'runs'/p['jobs'][index]['run_id']
            if folder.exists():
                m=load_json_strict((folder/'manifest.json').read_text())
                if m['status']!='complete' or m['job']!=p['jobs'][index] or m['protocol_sha256']!=sha256(study/'protocol.json'):
                    raise ValueError('Retained incomplete/foreign run')
                if any(sha256(folder/n)!=h for n,h in m['artifacts'].items()):raise ValueError('Changed retained artifact')
                continue
            elapsed=sum(load_json_strict(f.read_text()).get('elapsed_seconds',0) for f in (study/'runs').glob('*/manifest.json')) if (study/'runs').exists() else 0
            if elapsed>=p['worker_hour_cap']*3600:raise RuntimeError('Declared worker budget exhausted')
            with (logs/(p['jobs'][index]['run_id']+'.log')).open('x') as log:
                dispatch_worker([sys.executable,__file__,'worker','--study',str(study),'--index',str(index),'--device',device],
                    cwd=REPO,env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('COMPLETE '+p['jobs'][index]['run_id'],flush=True)
    queue('cuda:0',[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,f'cuda:{d}',list(range(1,24))[d::2]) for d in range(2)]
        for future in futures:future.result()

if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','run','worker'])
    parser.add_argument('--study',type=Path,required=True);parser.add_argument('--start',type=int)
    parser.add_argument('--index',type=int);parser.add_argument('--device',default='cuda:0');parser.add_argument('--reproduce-panel',type=Path);a=parser.parse_args()
    study=a.study.resolve()
    if study==ROOT or not study.is_relative_to(ROOT):raise ValueError('Use the authorized experiment root')
    if a.action=='prepare':prepare(study,a.start,a.reproduce_panel)
    elif a.action=='run':run(study)
    else:worker(study,a.index,a.device)
