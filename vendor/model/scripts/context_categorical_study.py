#!/usr/bin/env python3
"""Frozen predictive resolutions on disjoint proper-prefix RefinedWeb contexts."""
from companion_paths import child_pythonpath, dispatch_worker, validate_worker_cli
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor
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
from numerical_validation import load_json_strict, finite_array
from context_reservations import reserve
from context_execution_contract import attach, admit as shared_admit

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(legacy_path('/pldr-data/model'))
NATIVE = ROOT/'assets/PLDR-LLM-v51-SOC-110M-1'
PARENT = ROOT/'critical-onepass-refinement-20260914'
DICTIONARY = ROOT/'categorical-visibility-20260915'
PROBES = ROOT/'controlled-study-20260905/data/short'
CORPUS = ROOT/'data/refinedweb-onepass-524288'


def prepare(destination, start, count, sizes, excluded_studies=(), reproduce=None):
    if destination.exists() or not destination.is_relative_to(ROOT) or count < 8 or count % 8:
        raise ValueError('Use a fresh study and a positive multiple of eight contexts')
    p = load_json_strict((PARENT/'protocol.json').read_text())
    jobs = [j for j in p['jobs'] if j['heads'] in [8, 24] and j['control'] in [0, 1.5]]
    if len(jobs) != 24:
        raise ValueError('Expected the complete 24-checkpoint panel')
    receipt = reserve(destination, count, start, exclusions=excluded_studies, reproduce=reproduce)
    rows = np.array(receipt['context_rows'])
    hashes = receipt['document_hashes']
    exclusion_inputs = [Path(x) for x in receipt['imported_protocols']]
    tokens = np.load(PROBES/'tokens.npy', mmap_mode='r')
    offsets = np.load(PROBES/'offsets.npy')
    blocks = tokens[rows[:,None], offsets[rows,None]+np.arange(65)]
    with np.load(DICTIONARY/'reference.npz') as a:
        reference = a['reference'].mean(0)
    if not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError('Invalid development-only reference')
    order = np.argsort(-reference, kind='stable')
    if sorted(set(sizes)) != sizes or not all(0 < k < len(order) for k in sizes):
        raise ValueError('Ordered, distinct proper vocabulary resolutions required')
    destination.mkdir(parents=True)
    np.savez_compressed(destination/'inputs.npz', blocks=blocks, rows=rows,
                        reference=reference, order=order)
    sources = ['scripts/context_categorical_study.py', 'scripts/numerical_validation.py',
               'src/model_rg/training.py', 'src/model_rg/native.py', 'src/model_rg/provenance.py']
    inputs = [PARENT/'protocol.json', DICTIONARY/'protocol.json', DICTIONARY/'reference.npz',
              PROBES/'tokens.npy', PROBES/'offsets.npy', PROBES/'records.json',
              CORPUS/'records.json', NATIVE/'modeling_pldrllm.py', NATIVE/'configuration_pldrllm.py', *exclusion_inputs]
    bound = {str(path): sha256(path) for path in inputs}
    for job in jobs:
        folder = PARENT/'runs'/job['run_id']
        m = load_json_strict((folder/'manifest.json').read_text())
        if m['status'] != 'complete' or m['job'] != job:
            raise ValueError('Uncompleted incoming state')
        bound[str(folder/'manifest.json')] = sha256(folder/'manifest.json')
        # Actual checkpoint bytes are checked again before each native call.
        job['checkpoint_sha256'] = m['artifacts']['final-state.pt']
    protocol = dict(schema='context-categorical-v1', role='observation', jobs=jobs,
        parent=str(PARENT), contexts=count, context_rows=rows.tolist(), document_hashes=hashes,
        excluded_context_protocols=[str(path) for path in exclusion_inputs],
        sizes=sizes, vocabulary=len(order), replicas=6, batch_size=8,
        development_reference='Mean over the eight development contexts of the fixed six-initialization N=14, g=0, T=2048 reference. No new-context predictions enter the dictionary.',
        input_sha256=bound, selection_sha256=sha256(destination/'inputs.npz'),
        source_sha256={name:sha256(REPO/name) for name in sources},
        target=.25, training_updates=0, arithmetic='float32 native, float64 reduction; TF32 disabled',
        scientific_hypotheses=['Nested KL increments and channel variances are monotone.',
            'The finite bias/variance budget holds under a context-independent reference.',
            'Report centered RMS target outcomes at every frozen resolution; no universal context accuracy is assumed.'],
        independence='Six complete conditional initialization replicas per width/control cell. Contexts and paired controls are not extra training replicas.',
        failure_policy='Retain every frozen checkpoint and context; a failure prevents a complete analysis.',
        expected_forward_calls=len(jobs)*count//8, qualification_forward_calls=2)
    write_json(destination/'protocol.json', protocol)
    snapshot=destination/'executed-source'
    for name in sources:
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((REPO/name).read_bytes())
    attach(destination, protocol, receipt)
    print(json.dumps({'study':str(destination),'checkpoints':len(jobs),'contexts':count,'forward_calls':protocol['expected_forward_calls']}),flush=True)


def admit(study, index=None):
    return shared_admit(study, index)


@torch.no_grad()
def worker(study, index, device):
    if type(index) is not int or device not in ['cuda:0','cuda:1']: raise ValueError('Explicit worker index/device required')
    p=admit(study, index);job=p['jobs'][index];target=study/'runs'/job['run_id']
    target.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();calls=0;qualification_calls=0
    manifest=dict(status='started',job=job,protocol_sha256=sha256(study/'protocol.json'),
                  training_updates=0,device=device)
    write_json(target/'manifest.json',manifest)
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_device(device);torch.cuda.reset_peak_memory_stats(device)
        checkpoint=Path(p['parent'])/'runs'/job['run_id']/'final-state.pt'
        if sha256(checkpoint)!=job['checkpoint_sha256']:raise ValueError('Changed incoming checkpoint')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        model=TrainingModel(NATIVE,job['heads'],job['seed'],device)
        model.model.load_state_dict(state['model'],strict=True);model.model.eval()
        del state
        with np.load(study/'inputs.npz') as a:blocks=a['blocks']
        logits=[]
        for offset in range(0,len(blocks),8):
            batch=torch.as_tensor(blocks[offset:offset+8,:64],dtype=torch.long,device=device)
            z=model.forward(batch).logits[:,-1].cpu().numpy();calls+=1
            finite_array(z,'fresh native logits');logits.append(z)
            if index==0 and offset==0:
                # Replay same logical float32 observations before admitting the panel.
                for _ in range(2):
                    repeated=model.forward(batch).logits[:,-1].cpu().numpy();qualification_calls+=1
                    if z.dtype!=repeated.dtype or z.shape!=repeated.shape or z.tobytes()!=repeated.tobytes():
                        raise ValueError('Fresh-context logical-byte replay differs')
        np.savez_compressed(target/'observations.npz',logits=np.concatenate(logits))
        manifest.update(status='complete',observations_sha256=sha256(target/'observations.npz'),
            checkpoint=str(checkpoint),checkpoint_sha256=job['checkpoint_sha256'],
            maximum_cuda_memory_bytes=torch.cuda.max_memory_allocated(device))
    except Exception as error:
        manifest.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        manifest.update(native_forward_calls=calls,qualification_forward_calls=qualification_calls,
                        elapsed_seconds=time.monotonic()-started)
        write_json(target/'manifest.json',manifest)
    print(json.dumps({'run_id':job['run_id'],'seconds':manifest['elapsed_seconds'],'calls':calls}),flush=True)


def run(study):
    p=admit(study)
    for path,digest in p['input_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Changed frozen input '+path)
    logs=study/'logs';logs.mkdir(exist_ok=True)
    def queue(device,indices):
        for index in indices:
            job=p['jobs'][index];target=study/'runs'/job['run_id']
            if target.exists():
                m=load_json_strict((target/'manifest.json').read_text())
                if m['status']!='complete' or m['protocol_sha256']!=sha256(study/'protocol.json'):
                    raise ValueError('Retained incomplete/foreign observation')
                if sha256(target/'observations.npz')!=m['observations_sha256']:raise ValueError('Changed observation')
                continue
            with (logs/(job['run_id']+'.log')).open('x') as log:
                dispatch_worker([sys.executable,__file__,'worker','--study',str(study),'--index',str(index),'--device',device],
                    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2'),
                    stdout=log,stderr=subprocess.STDOUT,check=True)
            print('COMPLETE '+job['run_id'],flush=True)
    # The first checkpoint provides the repeated native qualification and timing.
    queue('cuda:0',[0])
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(queue,f'cuda:{d}',list(range(1,len(p['jobs'])))[d::2]) for d in range(2)]
        for future in futures:future.result()


if __name__ == '__main__':
    validate_worker_cli(__file__)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','worker'])
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--start',type=int);parser.add_argument('--contexts',type=int,default=32)
    parser.add_argument('--sizes',type=int,nargs='+',default=[128,512,2048,8192,16384])
    parser.add_argument('--index',type=int);parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--exclude-study',type=Path,action='append',default=[])
    parser.add_argument('--reproduce-panel',type=Path)
    args=parser.parse_args();study=args.study.resolve()
    if not study.is_relative_to(ROOT) or study==ROOT:raise ValueError('Use the authorized experiment root')
    if args.action=='prepare':prepare(study,args.start,args.contexts,args.sizes,args.exclude_study,args.reproduce_panel)
    elif args.action=='run':run(study)
    else:worker(study,args.index,args.device)
