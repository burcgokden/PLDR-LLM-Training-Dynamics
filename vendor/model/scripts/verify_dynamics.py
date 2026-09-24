#!/usr/bin/env python
"""Verify the completed width-time study, interventions and formal source graph."""
from companion_paths import legacy_path
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import os
import subprocess
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256, write_json
from formal_compatibility import check_formal_inventory
from verify_row_dynamics import verify_rows, pair_covariance
from verify_row_adjoint import verify_adjoint
from verify_row_gradient_projection import verify_gradient_projection
from verify_metric_collectives import verify_metric_collectives


def verify_paired_state_distance(reference, other, reported):
    names=list(reference['model'])
    if names!=list(other['model']):raise AssertionError('Return parameter identities changed')
    is_generator=lambda n:any(p in n for p in ['reslayerAs','plgatt_layer','layernormA'])
    ordered=[n for n in names if is_generator(n)]+[n for n in names if not is_generator(n)]
    ids=[i for group in reference['optimizer']['param_groups'] for i in group['params']]
    other_ids=[i for group in other['optimizer']['param_groups'] for i in group['params']]
    if ids!=other_ids or len(ids)!=len(ordered):raise AssertionError('Return optimizer mapping changed')
    totals={g:{k:[0,0.,0.] for k in ['weights','first_moment','second_moment']} for g in ['shared_metric','head_power','remaining']}
    for name,index in zip(ordered,ids,strict=True):
        group='shared_metric' if 'reslayerAs' in name else ('head_power' if is_generator(name) else 'remaining')
        for key,moment in [('weights',None),('first_moment','exp_avg'),('second_moment','exp_avg_sq')]:
            a=(reference['model'][name] if moment is None else reference['optimizer']['state'][index][moment]).double()
            b=(other['model'][name] if moment is None else other['optimizer']['state'][index][moment]).double()
            if a.shape!=b.shape:raise AssertionError('Return state shape changed')
            record=totals[group][key];record[0]+=a.numel()
            record[1]+=float(torch.linalg.vector_norm(a))**2;record[2]+=float(torch.linalg.vector_norm(b-a))**2
    for group,fields in totals.items():
        for key,(n,reference_norm2,difference_norm2) in fields.items():
            r=reported[group][key]
            if n!=r['coordinates']:raise AssertionError('Return coordinate count changed')
            np.testing.assert_allclose([reference_norm2,difference_norm2,np.sqrt(difference_norm2/n),np.sqrt(reference_norm2/n)],
                [r[k] for k in ['reference_squared_norm','squared_distance','distance_rms','reference_rms']],rtol=1e-10,atol=1e-18)
            if reference_norm2>0:np.testing.assert_allclose(np.sqrt(difference_norm2/reference_norm2),r['relative_distance'],rtol=1e-10,atol=1e-15)
            elif r['relative_distance'] is not None:raise AssertionError('A zero state reference needs explicit handling')



def verify_return_trajectories(raw, reported, expected_steps, heads, contexts, stream_seed, branch_names):
    """Reconstruct complete paired norms and the declared native observation design."""
    expected_steps=np.asarray(expected_steps)
    np.testing.assert_array_equal(raw['probe_steps'],expected_steps)
    np.testing.assert_array_equal(reported['steps'],expected_steps)
    end=int(expected_steps[-1]);updates=end-int(expected_steps[0]);probes=len(expected_steps)
    expected_rows=np.random.default_rng(stream_seed+1000).integers(0,3072,size=(end,32))
    expected_offsets=np.random.default_rng(stream_seed+1001000).integers(0,449,size=(end,32))
    np.testing.assert_array_equal(raw['rows'],expected_rows)
    np.testing.assert_array_equal(raw['offsets'],expected_offsets)
    if set(reported['branches'])!=set(branch_names):raise AssertionError('Return trajectory branch identities changed')
    for branch in branch_names:
        h=raw[branch+'_heads'];f=raw[branch+'_fields'];summary=reported['branches'][branch]
        if h.shape!=(probes,contexts,5,heads,4) or f.shape!=(probes,contexts,36):raise AssertionError('Return trajectory observation dimensions changed')
        if h.dtype!=np.float32 or f.dtype!=np.float32:raise AssertionError('Return native observation precision changed')
        if raw[branch+'_losses'].shape!=(updates,) or raw[branch+'_gradient_norms'].shape!=(updates,):raise AssertionError('Return training trace length changed')
        if (raw[branch+'_gradient_norms']<0).any():raise AssertionError('Invalid return gradient norm')
        for name,index in [('entropy',0),('row_energy',2)]:
            # Head means are the declared native NumPy32 observation coordinate.
            # Torch64 independently reconstructs the norm over contexts and layers.
            q=h[...,index].mean(-1)
            delta=q-raw['base_heads'][...,index].mean(-1)
            tq=torch.from_numpy(q.copy()).double();td=torch.from_numpy(delta.copy()).double()
            norms=(torch.linalg.vector_norm(td.reshape(probes,-1),dim=-1)/np.sqrt(contexts*5)).numpy()
            record=summary[name]
            np.testing.assert_allclose(norms,record['paired_rms'],rtol=1e-11,atol=0)
            np.testing.assert_allclose(tq.mean((1,2)).numpy(),record['mean'],rtol=4e-6,atol=1e-30)
            if norms[0]>0:
                if record['relative_paired_rms'] is None:raise AssertionError('A nonzero initial displacement lost its return ratio')
                np.testing.assert_allclose(norms/norms[0],record['relative_paired_rms'],rtol=1e-11,atol=0)
            elif record['relative_paired_rms'] is not None:raise AssertionError('A zero initial displacement cannot define a return ratio')
        losses=torch.from_numpy(f[:,:,25].copy()).double()
        difference=torch.from_numpy((f[:,:,25]-raw['base_fields'][:,:,25]).copy()).double()
        np.testing.assert_allclose(losses.mean(1).numpy(),summary['nll'],rtol=4e-6,atol=1e-10)
        # An absolute accumulation envelope covers cancellation in the native mean.
        unit=np.finfo(np.float32).eps;gamma=contexts*unit/(1-contexts*unit)
        tolerance=gamma*difference.abs().mean(1).numpy()+1e-30
        if np.any(finite_greater(np.abs(difference.mean(1).numpy()-np.asarray(summary['paired_nll'])), tolerance, 'scripts/verify_dynamics.py:83')):raise AssertionError('Paired return loss curve changed')
    return len(branch_names)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--trajectory-family',choices=['pilot','replication','long-horizon'],help='Check one complete trajectory family without producing a release gate')
    args=parser.parse_args();root=Path(args.root);study=root/'criticality-dynamics-20260906';repo=Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    verifier_sources={name:sha256(repo/name) for name in ['scripts/verify_dynamics.py','scripts/formal_compatibility.py','scripts/verify_row_dynamics.py','scripts/verify_row_adjoint.py','scripts/verify_row_gradient_projection.py','scripts/verify_metric_collectives.py']}
    def stable_verifier_sources():
        if any(sha256(repo/name)!=signature for name,signature in verifier_sources.items()):raise AssertionError('Verifier source changed during execution')
    cache={}
    def digest(path):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha256(path)
        return cache[path]
    def bound(path,signature):
        if digest(path)!=signature:raise AssertionError('Hash mismatch: '+str(path))
    def binding(folder):
        record=load_json_strict((folder/'binding.json').read_text())
        for p,h in record['inputs'].items():bound(p,h)
        for p,h in record['source_files'].items():bound(folder/'source'/p,h)
        return digest(folder/'binding.json')
    formal_dir=study/'formal';formal=load_json_strict((formal_dir/'verification.json').read_text())
    if formal['status']!='passed':raise AssertionError('Archived formal gate failed')
    formal_inventory=check_formal_inventory(repo,formal['module_sources'])
    bound(repo/'scripts/formal/Gate.lean',formal['gate_sha256']);bound(repo/'lake-manifest.json',formal['lake_manifest_sha256'])
    for check in formal['checks']:
        bound(formal_dir/(check['name']+'.log'),check['log_sha256'])
        if not check['expectation_met']:raise AssertionError('A formal mutation gate failed')
    current_formal=Path(args.output).with_suffix('').with_name(Path(args.output).stem+'-formal')
    current_formal.mkdir(parents=True,exist_ok=False)
    env=dict(os.environ,ELAN_HOME=legacy_path('/pldr-tools/elan'),
        PATH=legacy_path('/pldr-tools/elan/bin')+os.pathsep+os.environ['PATH'])
    for label,command in [('build',['lake','build']),('gate',['lake','env','lean','scripts/formal/Gate.lean'])]:
        with (current_formal/(label+'.log')).open('w') as log:
            subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    formal_inventory['current_gate_log_sha256']=digest(current_formal/'gate.log')
    print('Formal compatibility and current gate passed; beginning full dynamics verification.',flush=True)
    for p,h in load_json_strict((study/'input-integrity.json').read_text()).items():bound(p,h)
    original=load_json_strict((root/'criticality-study-20260905/verification.json').read_text())
    inherited_path=study/'qa/base-reverification.json';inherited=load_json_strict(inherited_path.read_text())
    if inherited!=original or inherited['status']!='passed' or inherited['required_scientific_runs']!=272:
        raise AssertionError('Independent inherited-evidence verification differs from its bound completed record')
    old_formal=root/'criticality-study-20260905/formal-native-limits/verification.json'
    for name,h in load_json_strict(old_formal.read_text())['module_sources'].items():bound(repo/name,h)
    tokens=np.load(root/'data/refinedweb-4608/tokens.npy');expected={};protocols={}
    for name,count in [('pilot',48),('replication',60),('long-horizon',20)]:
        if args.trajectory_family and name!=args.trajectory_family:continue
        protocol=study/'protocols'/(name+'.json');spec=load_json_strict(protocol.read_text());protocols[name]=spec
        if len(spec['jobs'])!=count:raise AssertionError('Training protocol inventory changed')
        bound(repo/'internal/dynamics-protocols'/(name+'.json'),digest(protocol))
        ledger=load_json_strict((study/('launcher-'+name+'.json')).read_text())
        if ledger['status']!='complete' or any(r['returncode'] for r in ledger['records']):raise AssertionError('Unresolved training outcome')
        bound(protocol,ledger['protocol_sha256'])
        if ledger['jobs']!=spec['jobs'] or len(ledger['records'])!=count:raise AssertionError('Launcher does not match its protocol')
        snapshots=[]
        for job in spec['jobs']:
            folder=study/'runs'/job['run_id']
            snapshots.append((folder,load_json_strict((folder/'binding.json').read_text())))
        for path,signature in ledger['source_files'].items():
            archived=next((folder/'source'/path for folder,record in snapshots if record['source_files'].get(path)==signature),None)
            if archived is None:raise AssertionError('Original launcher source version was not preserved: '+path)
            bound(archived,signature)
        for job in spec['jobs']:
            if job['run_id'] in expected:raise AssertionError('Repeated scientific run')
            expected[job['run_id']]=job
    inventory=[];initial={};common_streams={};shared={}
    training_metadata={}
    for name,job in expected.items():
        folder=study/'runs'/name;meta=load_json_strict((folder/'manifest.json').read_text());a=meta['arguments']
        if meta['status']!='complete' or meta['schema']!='criticality-dynamics-training-v1':raise AssertionError('A numerical failure needs explicit scientific treatment')
        for key in ['heads','seed','multiplier','steps','resume']:
            if a[key]!=job.get(key):raise AssertionError('Changed job condition: '+key)
        if a['normalization']!=job.get('normalization','variance'):raise AssertionError('Changed initial normalization')
        if a['shared_seed']!=640011 or a['stream_seed']!=640001:raise AssertionError('Changed conditioned training environment')
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        bound(folder/'measurements.npz',meta['raw_sha256']);bound(folder/'sampling.npz',meta['sampling_sha256'])
        raw=np.load(folder/'measurements.npz');sampling=np.load(folder/'sampling.npz');h=a['heads'];steps=a['steps'];start=meta['start_step']
        training_metadata[name]=meta
        expected_milestones=sorted({t for t in [start,128,512,1024,2048,4096,8192,steps] if start<=t<=steps})
        if meta['milestones']!=expected_milestones:raise AssertionError('Training observation milestones changed')
        for key in raw.files:
            if not np.isfinite(raw[key]).all():raise AssertionError('Nonfinite raw value: '+name+'/'+key)
        rows,offsets=sampling['rows'],sampling['offsets']
        if rows.shape!=(steps,32) or offsets.shape!=rows.shape:raise AssertionError('Invalid sampler shape')
        regenerated_rows=np.random.default_rng(a['stream_seed']+1000).integers(0,3072,size=(steps,32))
        regenerated_offsets=np.random.default_rng(a['stream_seed']+1001000).integers(0,449,size=(steps,32))
        np.testing.assert_array_equal(rows,regenerated_rows);np.testing.assert_array_equal(offsets,regenerated_offsets)
        np.testing.assert_array_equal(sampling['evaluation_rows'],np.arange(512,1024))
        np.testing.assert_array_equal(sampling['dense_rows'],np.arange(512,528))
        counts=np.bincount(tokens[rows,offsets+64].ravel(),minlength=32000)
        np.testing.assert_array_equal(raw['supervised_target_counts'],counts)
        if raw['losses'].shape!=(steps-start,) or raw['gradient_norms'].shape!=(steps-start,3):raise AssertionError('Incomplete update trace')
        if raw['dense_heads'].shape!=(len(raw['steps']),16,5,h,4):raise AssertionError('Dense cohort changed')
        np.testing.assert_allclose(raw['mean_fields'],raw['dense_fields'].mean(1),rtol=1e-6,atol=1e-7)
        if raw['steps'][0]!=start or raw['steps'][-1]!=steps:raise AssertionError('Temporal endpoints changed')
        if not np.all(np.diff(raw['steps'])==32):raise AssertionError('Dense cadence changed')
        def check_fields(heads,fields):
            means=heads.mean(-2)
            np.testing.assert_allclose(means[...,0]*np.log(64),fields[...,:5],rtol=3e-6,atol=2e-6)
            for index,section in [(1,slice(5,10)),(2,slice(26,31)),(3,slice(31,36))]:
                np.testing.assert_allclose(means[...,index],fields[...,section],rtol=3e-6,atol=2e-6)
            for index in [0,2]:
                if heads[...,index].min() < -1e-7 or heads[...,index].max()>1+1e-6:raise AssertionError('Bounded field units changed')
        check_fields(raw['dense_heads'],raw['dense_fields'])
        for t in meta['milestones']:
            if raw[f'heads_{t}'].shape!=(512,5,h,4):raise AssertionError('Population field cohort changed')
            check_fields(raw[f'heads_{t}'],raw[f'fields_{t}'])
            statistics=raw[f'parameter_statistics_{t}']
            if statistics.shape!=(len(meta['parameter_names']),6):raise AssertionError('Optimizer diagnostic coordinates changed')
            if t==0:
                np.testing.assert_array_equal(statistics[:,1:4],0);np.testing.assert_array_equal(statistics[:,4],1)
        if job.get('resume'):
            parent=Path(job['resume']).parent;old=np.load(parent/'measurements.npz')
            for key in ['fields','heads','context_kl','operator_dispersion']:
                np.testing.assert_array_equal(raw[f'{key}_{start}'],old[f'{key}_{start}'])
        for t,record in meta['saved_states'].items():
            checkpoint_path=folder/record['filename'];bound(checkpoint_path,record['sha256'])
            checkpoint=torch.load(checkpoint_path,map_location='cpu',mmap=True,weights_only=True)
            if checkpoint['step']!=int(t):raise AssertionError('State step mismatch')
            optimizer=checkpoint['optimizer'];states=optimizer['state']
            if {float(s['step']) for s in states.values()}!={float(t)}:raise AssertionError('Adam counters changed')
            if sum(len(g['params']) for g in optimizer['param_groups'])!=len(states):raise AssertionError('Missing Adam state')
            for group,rate in zip(optimizer['param_groups'],[3e-4*a['multiplier'],3e-4*2/h],strict=True):
                if group['lr']!=rate or group['eps']!=1e-8 or group['weight_decay']!=.01 or tuple(group['betas'])!=(.9,.95):raise AssertionError('Optimizer family changed')
            for parameter in checkpoint['model'].values():
                if not torch.isfinite(parameter).all():raise AssertionError('Nonfinite saved weight')
            for state in states.values():
                if not torch.isfinite(state['exp_avg']).all() or not torch.isfinite(state['exp_avg_sq']).all() or (state['exp_avg_sq']<0).any():raise AssertionError('Invalid saved moments')
            del checkpoint,optimizer,states
        bound(folder/'final-training-state.pt',meta['checkpoint_sha256'])
        key=(h,a['seed'],a['shared_seed'],a['stream_seed'])
        if key in initial and initial[key]!=meta['initial_parameter_sha256']:raise AssertionError('A paired rate or horizon changed its initial weights')
        initial[key]=meta['initial_parameter_sha256']
        if a['shared_seed'] in shared and shared[a['shared_seed']]!=meta['shared_generator_sha256']:raise AssertionError('Shared initialization changed')
        shared[a['shared_seed']]=meta['shared_generator_sha256']
        p=raw['source_baseline_logits'].astype(float);p=np.exp(p-p.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
        for field,curvature in [('source_secant','source_curvature'),('source_half_secant','source_half_curvature')]:
            np.testing.assert_allclose((p*raw[field].astype(float)**2).sum(1),raw[curvature],rtol=2e-6,atol=1e-12)
        inventory.append(dict(run_id=name,role='training',manifest_sha256=digest(folder/'manifest.json')))
        print('Verified trajectory',name,flush=True)
    if args.trajectory_family:
        stable_verifier_sources()
        write_json(args.output,dict(schema='dynamics-trajectory-family-verification-v1',status='passed',
            trajectory_family=args.trajectory_family,trajectories=len(inventory),inventory=inventory,
            verifier_sha256=digest(__file__),hashed_files=len(cache),
            interpretation='Intermediate completed-family check; this is not the full scientific release verification.'))
        print('Verified complete trajectory family',args.trajectory_family,len(inventory));return
    return_protocol=study/'protocols/collective-return.json';spec=load_json_strict(return_protocol.read_text())
    emission_protocol=study/'protocols/collective-return-emission.json'
    bound(repo/'internal/dynamics-protocols/collective-return-emission.json',digest(emission_protocol))
    bound(return_protocol,load_json_strict(emission_protocol.read_text())['training_protocol_sha256'])
    return_results={r['name']:r for r in load_json_strict((study/'analysis/controls/results.json').read_text())['collective_returns']}
    if set(return_results)!={c['name'] for c in spec['cases']}:raise AssertionError('Return summary parent identities changed')
    probe_tokens=np.load(root/'controlled-study-20260905/data/short/tokens.npy');probe_offsets=np.load(root/'controlled-study-20260905/data/short/offsets.npy')
    bound(repo/'internal/dynamics-protocols/collective-return.json',digest(return_protocol))
    ledger=load_json_strict((study/'collective-return.json').read_text())
    if ledger['status']!='complete' or len(ledger['records'])!=4:raise AssertionError('Component return inventory incomplete')
    return_trajectory_checks=0
    for case in spec['cases']:
        folder=study/'runs'/case['name'];meta=load_json_strict((folder/'manifest.json').read_text())
        if meta['status']!='complete' or len(meta['branches'])!=4:raise AssertionError('A return branch needs explicit outcome treatment')
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder);bound(folder/'measurements.npz',meta['raw_sha256'])
        raw=np.load(folder/'measurements.npz');base=np.load(Path(case['parent']).parent/'measurements.npz')
        if meta['schema']!='learned-collective-return-v2':raise AssertionError('Endpoint emissions are required')
        if meta['case']!=case or meta['start_step']!=8192 or meta['end_step']!=12288:raise AssertionError('Return parent or horizon changed')
        if [branch['branch'] for branch in meta['branches']]!=spec['branches']:raise AssertionError('Return execution branch inventory changed')
        summary=return_results[case['name']]
        if summary['seed']!=case['seed']:raise AssertionError('Return analysis parent identity changed')
        return_trajectory_checks+=verify_return_trajectories(raw,summary,np.arange(8192,12289,spec['probe_every']),case['heads'],64,640001,spec['branches'])
        np.testing.assert_array_equal(raw['cohort'],np.arange(512,576))
        targets=probe_tokens[raw['cohort'],probe_offsets[raw['cohort']]+64]
        base_checkpoint=torch.load(folder/'base-training-state.pt',map_location='cpu',mmap=True,weights_only=True)
        for key in raw.files:
            if not np.isfinite(raw[key]).all():raise AssertionError('Nonfinite return array')
        np.testing.assert_array_equal(raw['base_heads'][0],base['heads_8192'][:64])
        np.testing.assert_array_equal(raw['base_fields'][0],base['fields_8192'][:64])
        for branch in ['early_shared_weights_moments','frozen_early_shared_weights']:
            np.testing.assert_array_equal(raw[branch+'_heads'][0],raw['early_shared_weights_heads'][0])
            np.testing.assert_array_equal(raw[branch+'_initial_logits'],raw['early_shared_weights_initial_logits'])
        donor=torch.load(case['donor'],map_location='cpu',mmap=True,weights_only=True)
        for branch in meta['branches']:
            if branch['status']!='complete' or branch['updates']!=4096:raise AssertionError('Return update trace incomplete')
            bound(folder/branch['checkpoint'],branch['checkpoint_sha256'])
            checkpoint=torch.load(folder/branch['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
            if checkpoint['step']!=12288 or {float(s['step']) for s in checkpoint['optimizer']['state'].values()}!={12288.}:raise AssertionError('Return counter changed')
            label=branch['branch'];summary=return_results[case['name']]['branches'][label]
            verify_paired_state_distance(base_checkpoint,checkpoint,summary['augmented_endpoint_distance'])
            for stage,index in [('initial',0),('final',-1)]:
                logits=raw[label+'_'+stage+'_logits']
                if logits.shape!=(64,32000) or logits.dtype!=np.float32:raise AssertionError('Return emission coordinates changed')
                lp=torch.from_numpy(logits).double().log_softmax(-1)
                ref=torch.from_numpy(raw['base_'+stage+'_logits']).double().log_softmax(-1)
                kl=(ref.exp()*(ref-lp)).sum(-1).numpy()
                np.testing.assert_allclose(kl,summary['predictive_emission'][stage]['kl_by_context'],rtol=1e-9,atol=1e-12)
                np.testing.assert_allclose([kl.mean(),kl.max()],[summary['predictive_emission'][stage]['mean_kl'],summary['predictive_emission'][stage]['maximum_kl']],rtol=1e-9,atol=1e-12)
                np.testing.assert_allclose(-lp[np.arange(64),targets].numpy(),raw[label+'_fields'][index,:,25],rtol=2e-6,atol=2e-6)
            if branch['branch']=='frozen_early_shared_weights':
                for name,value in checkpoint['model'].items():
                    if 'reslayerAs' in name:torch.testing.assert_close(value,donor['model'][name],rtol=0,atol=0)
            del checkpoint
        del donor,base_checkpoint
        raw.close();base.close()
        inventory.append(dict(run_id=case['name'],role='component_return_parent',manifest_sha256=digest(folder/'manifest.json')))
    # Each diagnostic keeps its independently bound source and raw object.
    diagnostics=[]
    for folder in sorted((study/'analysis').iterdir()):
        path=folder/'manifest.json'
        if not path.exists():raise AssertionError('Unresolved analysis directory: '+folder.name)
        meta=load_json_strict(path.read_text())
        if meta['status']!='complete':raise AssertionError('Unresolved diagnostic outcome')
        bound(folder/'binding.json',meta['binding_sha256']);binding(folder)
        if 'results_sha256' in meta:bound(folder/'results.json',meta['results_sha256'])
        if 'raw_sha256' in meta:bound(folder/('bootstrap.npz' if (folder/'bootstrap.npz').exists() else 'measurements.npz'),meta['raw_sha256'])
        for filename,signature in meta.get('figures',{}).items():bound(folder/filename,signature)
        if folder.name.startswith('shared-crossing-'):
            r=load_json_strict((folder/'results.json').read_text());raw=np.load(folder/'measurements.npz')
            for name,index in [('entropy',0),('row_energy',2)]:
                q=raw['head_fields'][...,index].astype(float).mean(-1);grand=q.mean((0,1));a=q.mean(1)-grand;b=q.mean(0)-grand
                interaction=q-grand[None,None]-a[:,None]-b[None,:]
                pieces=np.array([np.mean(a*a),np.mean(b*b),np.mean(interaction*interaction)])
                total=np.mean((q-grand[None,None])**2)
                np.testing.assert_allclose(total,pieces.sum(),rtol=1e-12,atol=1e-15)
                np.testing.assert_allclose(pieces/total,r['summaries'][name]['fractions'],rtol=3e-6,atol=1e-10)
            np.testing.assert_array_equal(np.diagonal(raw['predictive_kl'],axis1=0,axis2=1),0)
        if folder.name.startswith('tangent-') and (folder/'measurements.npz').exists():
            r=load_json_strict((folder/'results.json').read_text());raw=np.load(folder/'measurements.npz')
            z=raw['updated_logits'];p=np.exp(z-z.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
            tangent=raw['updated_jvp'];tangent=tangent-(p*tangent).sum(-1,keepdims=True)
            norms=np.sqrt((p*tangent*tangent).sum(-1))
            for i,record in enumerate(r['records']):
                secant=(raw['positive_logits'][i]-raw['negative_logits'][i])/(2*record['amplitude'])
                secant-=(p*secant).sum(-1,keepdims=True)
                absolute=np.sqrt((p*(secant-tangent)**2).sum(-1));relative=absolute/np.maximum(norms,1e-14)
                np.testing.assert_allclose(relative.max(),record['maximum_predictive_relative_error'],rtol=1e-7,atol=1e-9)
            if finite_greater(r.get('native_optimizer_max_parameter_error',0), 1e-12, 'scripts/verify_dynamics.py:320') or finite_greater(r.get('native_optimizer_max_moment_error',0), 1e-12, 'scripts/verify_dynamics.py:320'):raise AssertionError('Native Adam primal formula failed')
        diagnostics.append(dict(name=folder.name,manifest_sha256=digest(path)))
    for name in ['tangent-replication','transport-arithmetic','tangent-resolution','row-transport','row-transport-transfer','row-projection','row-adjoint','row-adjoint-source-control','row-gradient-projection']:
        ledger=load_json_strict((study/(name+'.json')).read_text())
        if ledger['status']!='complete' or any(r['returncode'] for r in ledger['records']):raise AssertionError('Incomplete numerical-control ledger')
    for name in ['pilot','replication','long','controls','row-transport-summary','row-projection-summary','optimizer-scales','row-adjoint-summary','row-gradient-projection-summary','metric-collectives']:
        if not (study/'analysis'/name/'manifest.json').exists():raise AssertionError('Missing final analysis: '+name)
    # Recompute each population covariance from pair differences, independently
    # of the seed-centering implementation used by the scientific analyzers.
    covariance_checks=0
    for analysis_name in ['pilot','replication','long']:
        folder=study/'analysis'/analysis_name;result=load_json_strict((folder/'results.json').read_text())
        condition_paths={(c['heads'],c['multiplier']):c['runs'] for c in result.get('conditions',[])}
        protocol_name='long-horizon' if analysis_name=='long' else analysis_name
        expected_groups={}
        members=[(job['run_id'],training_metadata[job['run_id']]) for job in protocols[protocol_name]['jobs']]
        if analysis_name=='replication':
            members.extend((Path(name).name,load_json_strict((Path(name)/'manifest.json').read_text())) for name in protocols['replication']['additional_bound_runs'])
        for name,meta in members:
            a=meta['arguments'];expected_groups.setdefault((a['heads'],a['multiplier']),[]).append((a['seed'],name,meta))
        expected_rows={}
        for key,group in expected_groups.items():
            group.sort(key=lambda item:item[0]);seeds=[item[0] for item in group];names=[item[1] for item in group]
            expected_count=4 if analysis_name!='replication' or key[0]==24 else 16
            if seeds!=list(range(640101,640101+expected_count)):raise AssertionError('Analysis seed population changed')
            if analysis_name=='replication':
                times=[2048,4096,8192]
            else:
                times=group[0][2]['milestones']
                if any(item[2]['milestones']!=times for item in group):raise AssertionError('A group lost common milestones')
                if condition_paths.get(key)!=names:raise AssertionError('Analysis condition omitted or duplicated an explicit trajectory')
            for step in times:expected_rows[key+(step,)]=(seeds,names)
        actual_rows={(row['heads'],row['multiplier'],row['step']):row for row in result['rows']}
        if set(actual_rows)!=set(expected_rows) or len(actual_rows)!=len(result['rows']):raise AssertionError('Analysis omitted or duplicated a width-rate-time condition')
        if analysis_name!='replication':
            if set(condition_paths)!=set(expected_groups) or len(condition_paths)!=len(result['conditions']):raise AssertionError('Analysis condition inventory changed')
            dense_keys=[(row['heads'],row['multiplier']) for row in result['dense_paths']]
            if len(dense_keys)!=len(expected_groups) or set(dense_keys)!=set(expected_groups):raise AssertionError('Dense trajectories omit a condition')
            observed_inventory={row['run_id']:row['status'] for row in result['inventory']}
            if observed_inventory!={name:'complete' for name,_ in members} or len(observed_inventory)!=len(result['inventory']):raise AssertionError('Analysis training inventory changed')
        else:
            drift_keys=[(row['heads'],row['start'],row['end']) for row in result['paired_drifts']]
            if len(drift_keys)!=10 or set(drift_keys)!={(n,a,b) for n in [2,4,8,14,24] for a,b in [(2048,4096),(4096,8192)]}:raise AssertionError('Paired drift inventory changed')
        for row in result['rows']:
            names=row.get('runs',condition_paths.get((row['heads'],row['multiplier'])))
            if names is None:raise AssertionError('Analysis model identities missing')
            expected_seeds,expected_names=expected_rows[row['heads'],row['multiplier'],row['step']]
            if names!=expected_names or row['seeds']!=expected_seeds:raise AssertionError('Analysis row uses a different initialization population')
            paths=[study/'runs'/name if (study/'runs'/name).exists() else root/'criticality-study-20260905/runs'/name for name in names]
            values=np.stack([np.load(path/'measurements.npz')[f"heads_{row['step']}"] for path in paths])
            n,contexts,layers,heads,_=values.shape
            for label,index in [('entropy',0),('row_energy',2)]:
                q=values[...,index].astype(float).mean(-1)
                tq=torch.from_numpy(values[...,index].copy()).double().mean(-1)
                np.testing.assert_allclose(tq.mean((1,2)).numpy(),row[label]['mean_by_seed'],rtol=1e-11,atol=1e-18)
                np.testing.assert_allclose(float(tq.mean()),row[label]['mean'],rtol=1e-11,atol=1e-18)
                tq=tq-tq.mean(0,keepdim=True)
                for power,field in [(2,'centered_second_moment'),(4,'centered_fourth_moment')]:
                    np.testing.assert_allclose(tq.pow(power).mean(0).numpy(),row[label][field],rtol=1e-9,atol=1e-24)
                differences=np.array([q[i]-q[j] for i in range(n) for j in range(i)])
                covariance=heads*np.einsum('pxl,pxm->lm',differences,differences)/(n*(n-1)*contexts)
                np.testing.assert_allclose(covariance,row[label]['susceptibility'],rtol=1e-9,atol=1e-18)
                np.testing.assert_allclose(np.trace(covariance)/layers,row[label]['trace'],rtol=1e-9,atol=1e-18)
                if analysis_name=='replication':
                    boot=np.load(folder/'bootstrap.npz');counts=boot[f'bootstrap_counts_N{heads}']
                    expected_counts=np.random.default_rng(640811).multinomial(n,np.full(n,1/n),size=5000)
                    np.testing.assert_array_equal(counts,expected_counts)
                    traces=boot[f"chi_{label}_N{heads}_t{row['step']}"]
                    for b in [0,23,997,1331,4999]:
                        sampled=np.repeat(np.arange(n),counts[b]);sample=q[sampled]
                        centered=sample-sample.mean(0,keepdims=True)
                        trace=heads*np.mean(np.sum(centered**2,axis=0)/(n-1))
                        np.testing.assert_allclose(trace,traces[b],rtol=1e-9,atol=1e-12)
                    np.testing.assert_allclose(np.quantile(traces,[.025,.975]),row[label]['whole_seed_bootstrap']['percentile_95'],rtol=1e-12,atol=1e-15)
                covariance_checks+=1
    gradient_projection_checks=verify_gradient_projection(study,bound)
    force_summary=load_json_strict((study/'analysis/row-gradient-projection-summary/results.json').read_text())
    force_sources={c['name']:load_json_strict((study/'analysis'/c['name']/'results.json').read_text()) for c in load_json_strict((study/'protocols/row-gradient-projection.json').read_text())['cases']}
    if len(force_summary['cases'])!=48 or {r['case']['name'] for r in force_summary['cases']}!=set(force_sources):raise AssertionError('Shared-force summary omitted a source')
    for record in force_summary['cases']:
        if record!=force_sources[record['case']['name']]:raise AssertionError('Shared-force summary differs from the checked source')
    force_groups={}
    for source in force_sources.values():
        c=source['case']
        for record in source['records']:force_groups.setdefault((c['heads'],c['step'],record['variant']),[]).append((c,record))
    reported_keys=[(r['heads'],r['step'],r['variant']) for r in force_summary['groups']]
    if len(reported_keys)!=24 or set(reported_keys)!=set(force_groups):raise AssertionError('Shared-force summary condition inventory changed')
    for group in force_summary['groups']:
        members=sorted(force_groups[group['heads'],group['step'],group['variant']],key=lambda item:(item[0]['seed'],item[0]['batch_step']))
        expected_records=[dict(seed=c['seed'],batch_step=c['batch_step'],**r) for c,r in members]
        if expected_records!=group['records']:raise AssertionError('Shared-force paired source records changed')
        values=[max(r['mean_kl'] for _,r in members),max(r['maximum_kl'] for _,r in members),
                max(r['unclipped']['relative_error'] for _,r in members),max(r['clipped']['relative_error'] for _,r in members),
                np.median([r['clipped']['relative_error'] for _,r in members]),min(r['clipped']['cosine'] for _,r in members),
                max(np.sqrt(r['clipped']['squared_difference']) for _,r in members),min(np.sqrt(r['clipped']['reference_squared_norm']) for _,r in members)]
        fields=['maximum_mean_kl','maximum_context_kl','maximum_unclipped_relative_error','maximum_clipped_relative_error',
                'median_clipped_relative_error','minimum_clipped_cosine','maximum_clipped_absolute_error','minimum_clipped_reference_norm']
        np.testing.assert_allclose(values,[group[key] for key in fields],rtol=1e-12,atol=1e-20)
    metric_collective_checks=verify_metric_collectives(study,study/'analysis/metric-collectives',bound)
    row_checks=verify_rows(study,bound)
    adjoint_checks=verify_adjoint(study,bound,include_source_control=True)
    adjoint_analysis=load_json_strict((study/'analysis/row-adjoint-summary/results.json').read_text())
    if not adjoint_analysis['include_source_control'] or len(adjoint_analysis['cases'])!=48 or len(adjoint_analysis['groups'])!=12:raise AssertionError('Incomplete balanced adjoint summary')
    adjoint_lookup={}
    for row in adjoint_analysis['cases']:
        native=load_json_strict((study/'analysis'/row['case']['name']/'results.json').read_text())
        if native['case']!=row['case']:raise AssertionError('Adjoint analysis checkpoint identity changed')
        energies=[]
        for unit in range(8):
            selected=[r for r in native['records'] if r['unit'].endswith('_U'+str(unit))]
            vin=sum(r['input_adjoint_energy'] for r in selected);vout=sum(r['output_adjoint_energy'] for r in selected)
            norm=np.sqrt(sum(r['parameter_gradient_norm']**2 for r in selected))
            np.testing.assert_allclose([vin,vout,np.sqrt(vin/vout),norm,norm/native['total_gradient_norm']],
                [row['units'][unit][k] for k in ['input_adjoint_energy','output_adjoint_energy','aggregate_adjoint_gain','parameter_gradient_norm','parameter_gradient_fraction']],rtol=1e-11,atol=1e-14)
            energies.append((vin,vout))
        np.testing.assert_allclose(np.sqrt(energies[0][0]/energies[7][1]),row['block_adjoint_gain'],rtol=1e-11,atol=1e-14)
        adjoint_lookup[row['case']['heads'],row['case']['seed'],row['case']['step'],row['batch_step']]=row
    if len(adjoint_lookup)!=48 or len(adjoint_analysis['paired_state_comparisons'])!=6:raise AssertionError('Unbalanced adjoint analysis design')
    for group in adjoint_analysis['paired_state_comparisons']:
        for record in group['by_seed']:
            early=adjoint_lookup[group['heads'],record['seed'],2048,group['batch_step']]
            late=adjoint_lookup[group['heads'],record['seed'],8192,group['batch_step']]
            np.testing.assert_allclose(record['block_gain_ratio'],late['block_adjoint_gain']/early['block_adjoint_gain'],rtol=1e-11,atol=1e-14)
            np.testing.assert_allclose(record['unit_gain_ratios'],[b['aggregate_adjoint_gain']/a['aggregate_adjoint_gain'] for a,b in zip(early['units'],late['units'],strict=True)],rtol=1e-11,atol=1e-14)
    projection_analysis=load_json_strict((study/'analysis/row-projection-summary/results.json').read_text())
    for row in projection_analysis['rows']:
        data=[np.load(study/'analysis'/name/'measurements.npz') for name in row['cases']]
        n=len(data)
        for label,index in [('entropy',0),('row_energy',2)]:
            values=np.stack([d[row['variant']+'_heads'][...,index] for d in data]).astype(float)
            base=np.stack([d['base_heads'][...,index] for d in data]).astype(float)
            chi=np.trace(pair_covariance(values))/5;reference=np.trace(pair_covariance(base))/5
            error=values.mean(-1)-base.mean(-1);epsilon2=np.mean(error**2)
            penalty=row['heads']*n/(n-1)*epsilon2
            expected_values=[chi,reference,epsilon2,abs(chi-reference),2*np.sqrt(reference*penalty)+penalty]
            keys=['susceptibility','base_susceptibility','collective_squared_error','absolute_susceptibility_error','susceptibility_error_bound']
            np.testing.assert_allclose(expected_values,[row[label][k] for k in keys],rtol=1e-10,atol=1e-13)
            covariance_checks+=2
        for d in data:d.close()
    optimizer_result=load_json_strict((study/'analysis/optimizer-scales/results.json').read_text())
    optimizer_protocol=study/'protocols/optimizer-scales.json';optimizer_spec=load_json_strict(optimizer_protocol.read_text())
    bound(repo/'internal/dynamics-protocols/optimizer-scales.json',digest(optimizer_protocol))
    if optimizer_result['pilot_only'] or len(optimizer_result['cases'])!=64:raise AssertionError('Incomplete optimizer-scale inventory')
    if [r['case'] for r in optimizer_result['cases']]!=optimizer_spec['cases']:raise AssertionError('Optimizer-scale identities changed')
    for row in optimizer_result['cases']:
        state=torch.load(row['case']['checkpoint'],map_location='cpu',mmap=True,weights_only=True)
        names=list(state['model']);groups={label:[n for n in names if (any(p in n for p in ['reslayerAs','plgatt_layer','layernormA']))==label] for label in [True,False]}
        totals={label:dict(coordinates=0,active_coordinates=0,active_below_epsilon=0,adaptive_ratio_squared_sum=0.,decay_squared_sum=0.,active_adaptive_below_decay=0) for label in ['shared_metric','head_power','remaining']}
        for label,group in zip([True,False],state['optimizer']['param_groups'],strict=True):
            if len(groups[label])!=len(group['params']):raise AssertionError('Optimizer-scale parameter mapping changed')
            for name,index in zip(groups[label],group['params'],strict=True):
                moments=state['optimizer']['state'][index];second=moments['exp_avg_sq'].numpy()
                category='shared_metric' if 'reslayerAs' in name else ('head_power' if label else 'remaining')
                selected=totals[category];active=second>0
                selected['coordinates']+=second.size;selected['active_coordinates']+=int(np.count_nonzero(active))
                cutoff=group['eps']**2*(1-group['betas'][1]**int(moments['step']))
                selected['active_below_epsilon']+=int(np.count_nonzero(active & (second.astype(float)<cutoff)))
                step=int(moments['step']);first=moments['exp_avg'].numpy().astype(float)
                scale=np.sqrt(second.astype(float))/np.sqrt(1-group['betas'][1]**step)
                coefficient=first/((1-group['betas'][0]**step)*(scale+group['eps']))
                decay=state['model'][name].numpy().astype(float)*group['weight_decay']
                selected['adaptive_ratio_squared_sum']+=float(np.sum(coefficient*coefficient))
                selected['decay_squared_sum']+=float(np.sum(decay*decay))
                selected['active_adaptive_below_decay']+=int(np.count_nonzero(active & (np.abs(coefficient)<np.abs(decay))))
        for label,counts in totals.items():
            for key,value in counts.items():
                if key.endswith('_sum'):np.testing.assert_allclose(value,row['groups'][label][key],rtol=1e-10,atol=1e-15)
                elif row['groups'][label][key]!=value:raise AssertionError('Optimizer-scale coordinate count changed')
            np.testing.assert_allclose(np.sqrt(counts['adaptive_ratio_squared_sum']/counts['decay_squared_sum']),row['groups'][label]['adaptive_decay_norm_ratio'],rtol=1e-10,atol=1e-12)
        del state
    emission_smoke_path=study/'qa/collective-emission-smoke-verification.json'
    emission_smoke=load_json_strict(emission_smoke_path.read_text())
    if emission_smoke['status']!='passed' or emission_smoke['branches']!=4 or emission_smoke['updates_per_branch']!=8:raise AssertionError('Endpoint recording regression is incomplete')
    for name,signature in emission_smoke['checked_sources'].items():bound(repo/name,signature)
    for name,signature in emission_smoke['inputs'].items():bound(name,signature)
    regression_path=study/'qa/dynamics-regression.json';regression=load_json_strict(regression_path.read_text())
    if regression['status']!='passed' or regression['tests_passed']!=25:raise AssertionError('Numerical regression is incomplete')
    bound(study/'qa/dynamics-regression.log',regression['log_sha256'])
    for p,h in regression['tested_sources'].items():bound(repo/p,h)
    stable_verifier_sources()
    report=dict(schema='criticality-dynamics-release-verification-v1',status='passed',verifier_sha256=digest(__file__),
        required_scientific_runs=272+len(inventory),added_training_trajectories=len(expected),added_component_return_parents=4,added_component_return_branches=16,
        prerequisite_verification_sha256=digest(inherited_path),formal_compatibility=formal_inventory,formal_explicit_theorems=formal['explicit_theorems'],formal_verification_sha256=digest(formal_dir/'verification.json'),
        inventory=inventory,diagnostics=diagnostics,independent_covariance_checks=covariance_checks,row_checks=row_checks,adjoint_checks=adjoint_checks,gradient_projection_checks=gradient_projection_checks,metric_collective_checks=metric_collective_checks,
        numerical_regression_sha256=digest(regression_path),numerical_tests_passed=25,optimizer_scale_states=64,
        collective_emission_smoke_sha256=digest(emission_smoke_path),paired_return_endpoint_emissions=32,verified_return_trajectories=return_trajectory_checks,
        verifier_sources={name:signature for name,signature in verifier_sources.items() if name!='scripts/verify_dynamics.py'},hashed_files=len(cache),protected_files=len(load_json_strict((study/'input-integrity.json').read_text())))
    if report['required_scientific_runs']!=404:raise AssertionError('Scientific inventory count changed')
    write_json(args.output,report);print('Verified',report['required_scientific_runs'],'scientific artifacts and',len(diagnostics),'new diagnostic/analysis objects')


if __name__=='__main__':main()
