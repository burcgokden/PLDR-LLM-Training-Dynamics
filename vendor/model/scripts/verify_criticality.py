#!/usr/bin/env python
"""Verify the executed criticality evidence graph and independent numerical identities."""
from companion_paths import legacy_path
import argparse
import json
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from formal_compatibility import check_formal_inventory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(args.root);study=root/'criticality-study-20260905';repo=Path(__file__).resolve().parents[1]
    cache={}
    def digest(path):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha256(path)
        return cache[path]
    def bound(path,expected):
        if digest(path)!=expected:raise AssertionError(f'Hash mismatch: {path}')
    def binding(folder):
        record=json.loads((folder/'binding.json').read_text())
        for name,value in record['inputs'].items():bound(name,value)
        for name,value in record['source_files'].items():bound(folder/'source'/name,value)
        return digest(folder/'binding.json')
    formal_dir=study/'formal-native-limits'
    formal=json.loads((formal_dir/'verification.json').read_text())
    formal_inventory=check_formal_inventory(repo,formal['module_sources'])
    bound(repo/'scripts/formal/Gate.lean',formal['gate_sha256']);bound(repo/'lake-manifest.json',formal['lake_manifest_sha256'])
    for check in formal['checks']:
        bound(formal_dir/(check['name']+'.log'),check['log_sha256'])
        if not check['expectation_met']:raise AssertionError('Formal gate failed')
    current_formal=Path(args.output).with_suffix('').with_name(Path(args.output).stem+'-formal')
    current_formal.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ELAN_HOME=legacy_path('/pldr-tools/elan'),
             PATH=legacy_path('/pldr-tools/elan/bin')+os.pathsep+os.environ['PATH'])
    for label,command in [('build',['lake','build']),('gate',['lake','env','lean','scripts/formal/Gate.lean'])]:
        with (current_formal/(label+'.log')).open('w') as log:
            subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    formal_inventory['current_gate_log_sha256']=digest(current_formal/'gate.log')
    formal_inventory['current_gate_sha256']=digest(repo/'scripts/formal/Gate.lean')
    print('Formal compatibility and current axiom gate passed; beginning complete data verification.',flush=True)
    protected=json.loads((study/'input-integrity.json').read_text())
    for name,value in protected.items():bound(name,value)
    launchers={name:json.loads((study/f'launcher-{name}.json').read_text()) for name in ['scan','variance','restoration','horizon','width-holdout','fine','attention-regimes','analysis','fine-study','gain-precision','training-replay']}
    if any(l['status']!='complete' for l in launchers.values()):raise AssertionError('An executed prerequisite is incomplete')
    expected={j[0] for name in ['scan','variance','restoration','horizon','width-holdout','fine'] for j in launchers[name]['jobs']}
    if len(expected)!=208:raise AssertionError('Declared scan and restoration inventory changed')
    if any(r['returncode']!=0 for l in launchers.values() for r in l['records']):raise AssertionError('Unresolved execution error')
    tokens=np.load(root/'data/refinedweb-4608/tokens.npy')
    initial={};shared={};streams={};inventory=[];complete_training=[]
    for folder in sorted((study/'runs').iterdir()):
        path=folder/'manifest.json'
        item=dict(run_id=folder.name,required=folder.name in expected,status='incomplete_attempt')
        if (folder/'binding.json').exists():item['binding_sha256']=binding(folder)
        if not path.exists():
            if folder.name in expected:raise AssertionError(f'Missing scientific outcome: {folder.name}')
            inventory.append(item);continue
        meta=json.loads(path.read_text());item.update(status=meta['status'],manifest_sha256=digest(path))
        bound(folder/'binding.json',meta['binding_sha256']);bound(folder/'measurements.npz',meta['raw_sha256'])
        bound(folder/'sampling.npz',meta['sampling_sha256'])
        raw=np.load(folder/'measurements.npz');sampling=np.load(folder/'sampling.npz')
        for name in raw.files:
            if not np.isfinite(raw[name]).all():raise AssertionError(f'Nonfinite raw result: {folder.name}/{name}')
        if meta['status']!='complete':
            if folder.name in expected:raise AssertionError('A numerical failure requires explicit scientific treatment')
            inventory.append(item);continue
        if meta['schema']=='criticality-training-v1':
            a=meta['arguments'];steps=meta['completed_step'];h=a['heads']
            bound(folder/'final-training-state.pt',meta['checkpoint_sha256'])
            checkpoint=torch.load(folder/'final-training-state.pt',map_location='cpu',mmap=True,weights_only=True)
            optimizer=checkpoint['optimizer'];states=list(optimizer['state'].values())
            if sum(len(g['params']) for g in optimizer['param_groups'])!=len(states):
                raise AssertionError('A trainable tensor lacks a recorded Adam state')
            if {float(s['step']) for s in states}!={float(steps)}:
                raise AssertionError('Per-parameter Adam counters differ from the declared training horizon')
            rates=[3e-4*a['multiplier'],3e-4*2/h]
            for group,rate in zip(optimizer['param_groups'],rates,strict=True):
                if group['lr']!=rate or group['weight_decay']!=.01 or group['eps']!=1e-8 or tuple(group['betas'])!=(.9,.95):
                    raise AssertionError('The recorded optimizer differs from the analyzed clipped-decay family')
            del checkpoint,optimizer,states
            if raw['mean_fields'].shape[1]!=36 or len(meta['feature_names'])!=36:raise AssertionError('Common coordinate mismatch')
            rows,offsets=sampling['rows'],sampling['offsets']
            if rows.shape!=(a['steps'],32) or offsets.shape!=rows.shape:raise AssertionError('Training sampler shape')
            if rows.min()<0 or rows.max()>=3072 or offsets.min()<0 or offsets.max()>448:raise AssertionError('Training sampler bounds')
            targets=tokens[rows[:steps],offsets[:steps]+64]
            counts=np.bincount(targets.ravel(),minlength=32000)
            np.testing.assert_array_equal(raw['supervised_target_counts'],counts)
            for t in meta['milestones']:
                hf=raw[f'heads_{t}'];f=raw[f'fields_{t}'];means=hf.mean(2)
                if hf.shape!=(512,5,h,4):raise AssertionError('Head-coordinate shape')
                np.testing.assert_allclose(means[:,:,0]*np.log(64),f[:,:5],rtol=3e-6,atol=2e-6)
                np.testing.assert_allclose(means[:,:,1],f[:,5:10],rtol=3e-6,atol=2e-6)
                np.testing.assert_allclose(means[:,:,2],f[:,26:31],rtol=3e-6,atol=2e-6)
                np.testing.assert_allclose(means[:,:,3],f[:,31:36],rtol=3e-6,atol=2e-6)
                if hf[:,:,:,0].min() < -1e-7 or hf[:,:,:,0].max()>1+1e-6:raise AssertionError('Entropy outside fixed units')
                if hf[:,:,:,2].min() < -1e-7 or hf[:,:,:,2].max()>1+1e-6:raise AssertionError('Row energy outside fixed units')
            logits=raw['source_baseline_logits'].astype(float)
            p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
            for field,key in [('source_secant','source_curvature'),('source_half_secant','source_half_curvature')]:
                derivative=raw[field].astype(float)
                np.testing.assert_allclose((p*derivative**2).sum(1),raw[key],rtol=2e-6,atol=1e-12)
            if raw['source_kl'].min() < -1e-10:raise AssertionError('Negative source KL beyond numerical resolution')
            if folder.name in expected:
                complete_training.append(folder.name)
                normalization='reference' if h==2 else a.get('normalization','fan_in')
                key=(normalization,h,a['seed'],a['stream_seed'],a['shared_seed'])
                if key in initial and initial[key]!=meta['initial_parameter_sha256']:raise AssertionError('Generator-rate pairs change initialization')
                initial[key]=meta['initial_parameter_sha256']
                if a['shared_seed'] in shared and shared[a['shared_seed']]!=meta['shared_generator_sha256']:raise AssertionError('Shared environment changed')
                shared[a['shared_seed']]=meta['shared_generator_sha256']
                stream_key=(a['stream_seed'],a['steps'])
                signature=(rows.tobytes(),offsets.tobytes())
                if stream_key in streams and streams[stream_key]!=signature:raise AssertionError('Common minibatch history changed')
                streams[stream_key]=signature
        elif meta['schema']=='training-restoration-v1':
            if len(meta['branches'])!=8 or not all(b['status']=='complete' for b in meta['branches']):raise AssertionError('Restoration branch coverage')
            np.testing.assert_array_equal(raw['base_fields'][0],raw['frozen_base_fields'][0])
            for branch in ['plus_small','minus_small','plus','minus','frozen_base','frozen_plus','frozen_minus']:
                np.testing.assert_array_equal(raw[branch+'_heads'][0,:,0,:,2],raw['base_heads'][0,:,0,:,2])
            for branch in ['plus','minus']:
                np.testing.assert_array_equal(raw[branch+'_fields'][0],raw['frozen_'+branch+'_fields'][0])
            for name,equivalence in meta['pulse_equivalence'].items():
                magnitude=float(raw[name+'_kl'][0].mean())
                if equivalence['mean_kl']>1e-4*magnitude+1e-9:raise AssertionError('Physical pulse and inference source disagree')
            parent=json.loads((study/'runs'/meta['arguments']['parent']/'manifest.json').read_text())
            if meta['condition']!=parent['arguments']:raise AssertionError('Restoration condition changed')
        else:raise AssertionError('Unknown scientific schema')
        inventory.append(item)
    if expected-{i['run_id'] for i in inventory}:raise AssertionError('Missing required run directory')
    for (seed,steps),(rows,offsets) in streams.items():
        for (long_seed,long_steps),(long_rows,long_offsets) in streams.items():
            if seed==long_seed and long_steps>=steps:
                if not long_rows.startswith(rows) or not long_offsets.startswith(offsets):
                    raise AssertionError('Continuation changed the prefix of its training history')
    attention=[]
    for name in sorted(n for n in complete_training if n.startswith(('scan-','variance-'))):
        folder=study/'attention-regimes'/name;meta=json.loads((folder/'manifest.json').read_text())
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        bound(folder/'measurements.npz',meta['raw_sha256']);raw=np.load(folder/'measurements.npz')
        pmax=raw['probability_max'];trace=raw['jacobian_trace'];frobenius=raw['jacobian_frobenius']
        if (trace < -1e-12).any() or (trace > 2*(1-pmax)+1e-12).any():raise AssertionError('Saturation bound violated')
        # Cancellation in the analytic Frobenius formula is resolved only to sqrt(machine epsilon).
        if (frobenius > trace+3e-8).any():raise AssertionError('Categorical Jacobian norm bound violated')
        attention.append(dict(parent=name,manifest_sha256=digest(folder/'manifest.json')))
    precision=[]
    precision_protocol=json.loads((study/'protocols/source-precision.json').read_text())
    for parent in precision_protocol['parents']:
        folder=study/'source-precision'/parent;meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('A selected precision control is incomplete')
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        bound(folder/'measurements.npz',meta['raw_sha256']);raw=np.load(folder/'measurements.npz')
        source=np.load(study/'runs'/parent/'measurements.npz')
        selected=np.array(sorted(set([0,16,32,48]) | set(np.argsort(source['source_half_curvature'])[-4:].tolist())))
        np.testing.assert_array_equal(raw['selected_source_indices'],selected)
        np.testing.assert_array_equal(raw['native_gpu_logits'],source['source_baseline_logits'][selected])
        np.testing.assert_array_equal(raw['native_gpu_half_curvature'],source['source_half_curvature'][selected])
        for name in raw.files:
            if not np.isfinite(raw[name]).all():raise AssertionError('Nonfinite precision measurement')
        z=raw['smooth64_logits'];p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        v=raw['smooth64_jvp'];centered=v-(p*v).sum(1,keepdims=True)
        np.testing.assert_allclose(centered,raw['smooth64_centered_jvp'],rtol=1e-10,atol=1e-10)
        q=(p*centered**2).sum(1)
        np.testing.assert_allclose(q,raw['smooth64_curvature'],rtol=1e-10,atol=1e-12)
        secants=(raw['positive_logits']-raw['negative_logits'])/(2*raw['amplitudes'][:,None,None])
        secants-=(p[None]*secants).sum(2,keepdims=True)
        np.testing.assert_allclose(secants,raw['smooth64_secants'],rtol=1e-9,atol=1e-9)
        errors=np.sqrt((p[None]*(secants-centered[None])**2).sum(2))
        np.testing.assert_allclose(errors,raw['absolute_errors'],rtol=1e-7,atol=1e-10)
        precision.append(dict(parent=parent,manifest_sha256=digest(folder/'manifest.json'),selected_contexts=len(selected)))
    precision_attempts=[]
    for attempt in sorted((study/'source-precision-attempts').glob('*/attempt.json')):
        entry=dict(attempt=str(attempt.relative_to(study)),record_sha256=digest(attempt))
        for record in attempt.parent.glob('*/binding.json'):entry['binding_sha256']=binding(record.parent)
        entry['logs']={p.name:digest(p) for p in attempt.parent.glob('*.log')}
        precision_attempts.append(entry)
    replays=[]
    replay_protocol=json.loads((study/'protocols/training-replay.json').read_text())
    for parent in replay_protocol['parents']:
        suffix=parent.split('-',1)[1];folder=study/'training-replays'/('replay-'+suffix)
        meta=json.loads((folder/'manifest.json').read_text())
        if meta['schema']!='training-replay-v1' or meta['status']!='complete':raise AssertionError('A native replay is incomplete')
        if len(meta['branches'])!=2 or any(b['amplitude']!=0 or b['freeze_generator'] or b['completed_updates']!=512 or b['status']!='complete' for b in meta['branches']):
            raise AssertionError('The native zero-pulse control changed its intervention')
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        bound(folder/'measurements.npz',meta['raw_sha256']);bound(folder/'sampling.npz',meta['sampling_sha256'])
        raw=np.load(folder/'measurements.npz');sampling=np.load(folder/'sampling.npz')
        original=study/'runs'/('restoration-'+suffix);oldmeta=json.loads((original/'manifest.json').read_text())
        oldsampling=np.load(original/'sampling.npz');oldraw=np.load(original/'measurements.npz')
        if meta['condition']!=oldmeta['condition']:raise AssertionError('Replay parent condition changed')
        for name in ['rows','offsets']:np.testing.assert_array_equal(sampling[name],oldsampling[name])
        np.testing.assert_array_equal(raw['steps'],oldraw['steps'])
        for name in raw.files:
            if not np.isfinite(raw[name]).all():raise AssertionError('Nonfinite replay measurement')
        for key in ['losses','heads']:
            np.testing.assert_array_equal(raw['base_'+key],raw['replay_'+key])
            np.testing.assert_array_equal(raw['base_'+key],oldraw['base_'+key])
        np.testing.assert_array_equal(raw['replay_kl'],np.zeros_like(raw['replay_kl']))
        replays.append(dict(parent=parent,manifest_sha256=digest(folder/'manifest.json'),branches=2))
    # Recompute the initial-law comparison from arrays, including the normalized input boundary.
    initial_dir=study/'analysis/initialization-family'
    initial_meta=json.loads((initial_dir/'results.json').read_text())
    initial_raw=np.load(initial_dir/'measurements.npz')
    expected_initial={(f,h,s) for f in ['native','fan_in','variance'] for h in [2,4,8,14,24] for s in range(640301,640305)}
    if {(r['family'],r['heads'],r['seed']) for r in initial_meta['rows']}!=expected_initial:
        raise AssertionError('The sixty-model initialization inventory changed')
    for family,heads,seed in expected_initial:
        prefix=f'{family}-h{heads}-s{seed}'
        fields=initial_raw[prefix+'_heads']
        if fields.shape!=(16,5,heads,4) or not np.isfinite(fields).all():
            raise AssertionError('Invalid initialization head fields')
        for index in [0,2]:
            if fields[...,index].min() < -1e-7 or fields[...,index].max()>1+1e-6:
                raise AssertionError('Initial bounded field outside its declared units')
    limit_dir=study/'analysis/initialization-limit'
    comparison_dir=study/'analysis/initialization-kernel-comparison'
    limit=json.loads((limit_dir/'results.json').read_text())
    comparison=json.loads((comparison_dir/'results.json').read_text())
    kernel_raw=np.load(limit_dir/'measurements.npz')
    comparison_raw=np.load(comparison_dir/'measurements.npz')
    protocol=json.loads((study/'protocols/initialization-limit.json').read_text())
    rows=np.arange(512,528)
    short=root/'controlled-study-20260905/data/short'
    short_tokens=np.load(short/'tokens.npy');short_offsets=np.load(short/'offsets.npy')
    crops=short_tokens[rows[:,None],short_offsets[rows,None]+np.arange(64)]
    if (crops==0).any():raise AssertionError('Padding changes the initial embedding law')
    for raw in [kernel_raw,comparison_raw]:
        np.testing.assert_array_equal(raw['cohort'],rows)
        np.testing.assert_array_equal(raw['crops'],crops)
        for key in raw.files:
            if not np.isfinite(raw[key]).all():raise AssertionError('Nonfinite initial-kernel measurement')
    if protocol['rows']!=rows.tolist() or limit['monte_carlo_seeds']!=protocol['monte_carlo_seeds']:
        raise AssertionError('The kernel protocol changed')
    if limit['arguments']['samples']!=512 or limit['arguments']['replicas']!=4:
        raise AssertionError('The kernel integration inventory changed')
    kernels=kernel_raw['kernels'];heads_raw=kernel_raw['head_fields']
    if kernels.shape!=(4,16,6,64,64) or heads_raw.shape!=(4,16,5,512,2):
        raise AssertionError('Invalid native-kernel array shape')
    if heads_raw.min() < -1e-7 or heads_raw.max()>1+1e-6:
        raise AssertionError('Kernel head field outside fixed units')
    def check_kernel(array):
        np.testing.assert_allclose(array,array.swapaxes(-1,-2),rtol=0,atol=3e-10)
        if np.linalg.eigvalsh(array).min() < -1e-8:
            raise AssertionError('Residual kernel is not positive semidefinite at numerical tolerance')
        if np.diagonal(array,axis1=-2,axis2=-1).max()>1+1e-5:
            raise AssertionError('Kernel observation is outside the normalized residual boundary')
    check_kernel(kernels)
    first=(crops[:,:,None]==crops[:,None,:]).astype(float)
    np.testing.assert_array_equal(kernels[:,:,0],np.broadcast_to(first,(4,16,64,64)))
    predicted=heads_raw.mean(3)
    np.testing.assert_allclose(limit['predicted_means'],predicted.mean((0,1)),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(limit['prediction_by_context'],predicted.mean(0),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(limit['predicted_means_by_replica'],predicted.mean(1),rtol=1e-13,atol=1e-15)
    logit_variance=(256/32128)*np.diagonal(kernels[:,:,-1],axis1=-2,axis2=-1)[:,:,-1]
    np.testing.assert_allclose(limit['predicted_logit_mean_square'],logit_variance.mean(),rtol=1e-13)
    if kernel_raw['quadrature_errors'].shape!=(80,):raise AssertionError('Quadrature check coverage')
    np.testing.assert_equal(limit['quadrature_max_difference'],kernel_raw['quadrature_errors'].max())
    np.testing.assert_array_equal(comparison_raw['predicted_kernels'],kernels.mean(0))
    records={(r['heads'],r['seed']):r for r in comparison['records']}
    if set(records)!={(h,s) for h in [2,4,8,14,24] for s in range(640301,640305)}:
        raise AssertionError('The repeated initialization inventory changed')
    for heads in [2,4,8,14,24]:
        errors=[];measured=[]
        for seed in range(640301,640305):
            prefix=f'variance-h{heads}-s{seed}'
            k=comparison_raw[prefix+'_kernels'];f=comparison_raw[prefix+'_fields']
            if k.shape!=(16,6,64,64) or f.shape!=(16,5,heads,2):
                raise AssertionError('Repeated kernel observation shape changed')
            check_kernel(k)
            original=initial_raw[prefix+'_heads'][...,[0,2]]
            np.testing.assert_array_equal(f,original)
            if records[heads,seed]['identity_error']!=0:raise AssertionError('Repeated head-field identity changed')
            error=np.sqrt(np.mean((k-kernels.mean(0))**2,axis=(0,2,3)))
            np.testing.assert_allclose(records[heads,seed]['rmse_by_layer'],error,rtol=1e-13,atol=1e-15)
            errors.append(error);measured.append(original.mean(2))
        c=next(r for r in comparison['comparison'] if r['heads']==heads)
        np.testing.assert_allclose(c['rmse_by_layer'],np.sqrt(np.mean(np.array(errors)**2,axis=0)),rtol=1e-13,atol=1e-15)
        empirical=np.array(measured);residual=empirical-predicted.mean(0)[None]
        c=next(r for r in limit['comparison'] if r['heads']==heads)
        np.testing.assert_allclose(c['rmse'],np.sqrt(np.mean(residual**2,axis=(0,1,2))),rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(c['rmse_by_seed'],np.sqrt(np.mean(residual**2,axis=(1,2))),rtol=1e-12,atol=1e-15)
        np.testing.assert_allclose(c['empirical_means'],empirical.mean((0,1)),rtol=1e-6,atol=1e-7)
    mc_se=kernels.std(0,ddof=1)/2
    field_se=predicted.std(0,ddof=1)/2
    np.testing.assert_allclose(comparison['monte_carlo_kernel_se_rms_by_layer'],np.sqrt(np.mean(mc_se**2,axis=(0,2,3))),rtol=1e-13,atol=1e-15)
    np.testing.assert_allclose(comparison['monte_carlo_field_se_rms'],np.sqrt(np.mean(field_se**2,axis=(0,1))),rtol=1e-13,atol=1e-15)
    kernel_attempts=[]
    for attempt in sorted((study/'initialization-kernel-attempts').glob('*/attempt.json')):
        folder=attempt.parent;meta=json.loads((folder/'manifest.json').read_text())
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        bound(folder/'measurements.npz',meta['raw_sha256']);bound(folder/'results.json',meta['results_sha256'])
        kernel_attempts.append(dict(attempt=str(attempt.relative_to(study)),record_sha256=digest(attempt),
            manifest_sha256=digest(folder/'manifest.json'),logs={p.name:digest(p) for p in folder.glob('*.log')}))
    analyses=[]
    for name in ['initialization-family','scan','variance','restoration','attention-regimes','horizon','width-prediction','width-holdout','fine','variance-resolved','source-precision-checked','training-replay','initialization-limit','initialization-kernel-comparison']:
        folder=study/'analysis'/name;meta=json.loads((folder/'manifest.json').read_text())
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        bound(folder/'results.json',meta['results_sha256'])
        if 'raw_sha256' in meta:bound(folder/'measurements.npz',meta['raw_sha256'])
        for filename,value in meta.get('figures',{}).items():bound(folder/filename,value)
        analyses.append(dict(name=name,results_sha256=meta['results_sha256']))
    previous=json.loads((study/'supporting-evidence-verification.json').read_text())
    if previous['status']!='passed':raise AssertionError('Supporting evidence verification failed')
    report=dict(schema='criticality-release-verification-v1',status='passed',verifier_sha256=digest(__file__),protected_release_files=len(protected),
        required_scientific_runs=previous['required_scientific_runs']+len(expected),new_training_runs=len(complete_training),
        fresh_training_runs=192,continued_training_runs=8,restoration_parents=8,restoration_branches=64,initialization_diagnostic_models=60,initial_kernel_monte_carlo_replicas=4,repeated_initial_kernel_models=20,initial_kernel_execution_attempts=kernel_attempts,
        attention_diagnostic_checkpoints=len(attention),source_precision_checkpoints=len(precision),source_precision=precision,numerical_replay_parents=len(replays),numerical_replay_branches=2*len(replays),numerical_replays=replays,precision_execution_attempts=precision_attempts,inventory=inventory,analyses=analyses,attention_regimes=attention,
        formal_compatibility=formal_inventory,formal_explicit_theorems=formal['explicit_theorems'],formal_verification_sha256=digest(formal_dir/'verification.json'),hashed_files=len(cache),
        supporting_evidence_sha256=digest(study/'supporting-evidence-verification.json'))
    if check_formal_inventory(repo,formal['module_sources'])['current_modules']!=formal_inventory['current_modules']:
        raise AssertionError('Formal sources changed during verification')
    write_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['inventory','analyses','attention_regimes']},indent=2))


if __name__=='__main__':main()
