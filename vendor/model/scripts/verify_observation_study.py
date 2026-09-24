#!/usr/bin/env python
"""Independently reconstruct the completed observation study and its release gate.

This verifier does not import the statistical producer or its analysis helpers.
Bootstrap spot checks use direct resampled variances, independent of the stored
Gram calculation; all four-seed resamples are checked directly.
"""
import argparse
import itertools
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path
import numpy as np
import torch
from model_rg.provenance import sha256, write_json


def variance(q, heads):
    q = torch.as_tensor(np.asarray(q).copy(), dtype=torch.float64)
    return float(heads * torch.var(q, dim=0, correction=1).mean())


def close(x, y, atol=0, rtol=2e-10):
    np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def check_transfer(x, y, heads, report):
    chi, other, error = variance(x, heads), variance(y, heads), variance(y-x, heads)
    upper = heads * len(x)/(len(x)-1) * float(np.mean((y-x)**2))
    bound = 2*np.sqrt(chi*error)+error
    for key, value in dict(reference=chi, observed=other, signed_difference=other-chi,
                           error_susceptibility=error, uncentered_error_bound=upper,
                           absolute_bound=bound, relative_difference=(other-chi)/chi,
                           relative_bound=bound/chi).items():
        close(value, report[key], atol=1e-29 if key=='signed_difference' else 0)
    if finite_greater(error, upper*(1+1e-12), 'scripts/verify_observation_study.py:35') or finite_greater(abs(other-chi), bound*(1+1e-10), 'scripts/verify_observation_study.py:35'):
        raise AssertionError('Observation norm inequality failed')


def check_screen(field, heads, report):
    threshold=report['threshold']; mask=field<threshold
    low=np.where(mask,field,0).mean(-1); high=np.where(mask,0,field).mean(-1)
    whole=variance(low+high,heads); a=variance(low,heads); b=variance(high,heads)
    # The signed cross term is reconstructed from centered paired fields.
    cross=2*heads/(len(field)-1)*np.mean(np.sum((low-low.mean(0))*(high-high.mean(0)),axis=0))
    close([whole,a,b],[report[k] for k in ['susceptibility','below','above']])
    close(cross,report['cross'],atol=4e-15*whole,rtol=1e-6)
    close([mask.mean(),np.median(field),field.max()],
          [report[k] for k in ['fraction','median','maximum']])
    close(abs(whole-b)/whole,report['removal_relative'],atol=4e-15)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='observation-closure-20260906');p.add_argument('--output',required=True)
    p.add_argument('--criticality-verification',help='Explicit fresh inherited criticality verification')
    p.add_argument('--dynamics-verification',help='Explicit fresh inherited dynamics verification')
    p.add_argument('--numerical-verification',help='Explicit final numerical regression record')
    p.add_argument('--statement-verification',help='Explicit maintained statement correspondence verification')
    p.add_argument('--arithmetic-only',action='store_true',help='Intermediate numerical check, never a release gate')
    p.add_argument('--expected-statements',type=int,default=48,help='Exact current maintained statement count; the observation scope requires at least 48')
    p.add_argument('--expected-tests',type=int,default=32,help='Exact current numerical test count; the observation scope requires at least 32')
    a=p.parse_args();root=Path(a.root);study=root/a.study;repo=Path(__file__).resolve().parents[1]
    if Path(a.output).exists():raise FileExistsError('Preserve the existing verification record: '+a.output)
    if a.expected_statements<48 or a.expected_tests<32:raise ValueError('The verified supporting scope cannot be reduced')
    torch.set_num_threads(4)
    source_names=['scripts/verify_observation_study.py','scripts/verify_criticality.py',
                  'scripts/verify_dynamics.py','scripts/formal_compatibility.py',
                  'scripts/check_statement_contracts.py','scripts/formal/statement-registry.json',
                  'scripts/analyze_observation_study.py','scripts/measure_arithmetic_observation.py',
                  'scripts/train_dynamics.py','src/model_rg/observation.py',
                  'scripts/analyze_row_projection.py']
    sources={n:sha256(repo/n) for n in source_names};cache={}
    def digest(path):
        path=Path(path).resolve()
        if path not in cache:cache[path]=sha256(path)
        return cache[path]
    def bound(path,signature):
        if digest(path)!=signature:raise AssertionError('Changed bound artifact: '+str(path))
    def binding(folder,meta):
        bound(folder/'binding.json',meta['binding_sha256'])
        record=load_json_strict((folder/'binding.json').read_text())
        for name,signature in record['inputs'].items():bound(name,signature)
        for name,signature in record['source_files'].items():bound(folder/'source'/name,signature)
        return record
    arithmetic_path=study/'protocols/arithmetic.json';protocol=load_json_strict(arithmetic_path.read_text())
    ledger=load_json_strict((study/'launcher-arithmetic.json').read_text())
    if ledger['status']!='complete' or len(ledger['records'])!=13 or len(protocol['cells'])!=13:
        raise AssertionError('Incomplete arithmetic design')
    bound(arithmetic_path,ledger['protocol_sha256'])
    if protocol['seeds']!=list(range(640101,640105)) or protocol['rows']!=list(range(512,1024)):
        raise AssertionError('Arithmetic pairing changed')
    if (protocol['primary_screen'],protocol['relative_tolerance'],protocol['absolute_tolerance'])!=(1e-13,.01,1e-8):
        raise AssertionError('Declared observation criterion changed')
    probe=root/'controlled-study-20260905/data/short'
    tokens=np.load(probe/'tokens.npy',mmap_mode='r');offsets=np.load(probe/'offsets.npy');rows=np.arange(512,1024)
    expected_crops=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    arithmetic=[];arithmetic_by_id={};energy_by_id={}
    for cell,entry in zip(protocol['cells'],ledger['records'],strict=True):
        name=f"arithmetic-h{cell['heads']}-g{cell['multiplier']}-t{cell['step']}"
        if name!=entry['run_id'] or entry['status']!='complete':raise AssertionError('Arithmetic order or outcome changed')
        folder=study/'runs'/name;bound(folder/'manifest.json',entry['manifest_sha256'])
        meta=load_json_strict((folder/'manifest.json').read_text());binding(folder,meta)
        bound(folder/'measurements.npz',meta['raw_sha256']);bound(folder/'results.json',meta['results_sha256'])
        result=load_json_strict((folder/'results.json').read_text());arithmetic_by_id[name]=result
        if result['cell']!=cell or result['status']!='complete' or result['contexts']!=512 or result['independent_seeds']!=protocol['seeds']:
            raise AssertionError('Arithmetic cohort changed')
        bound(arithmetic_path,result['protocol_sha256']);fields={k:[] for k in ['Rarchive','R32','R64']}
        with np.load(folder/'measurements.npz') as raw:
            energy=dict(cell=cell)
            for key in ['E32','T32','R32','E64','T64','R64','Rarchive']:
                values=np.stack([raw[f's{seed}_{key}'] for seed in protocol['seeds']])
                energy['mean_'+key]=float(torch.as_tensor(values).mean())
            total=np.stack([raw[f's{seed}_T64'] for seed in protocol['seeds']])
            energy['minimum_T64']=float(total.min());energy['fraction_T64_below_floor']=float(np.mean(total<1e-30))
            energy_by_id[name]=energy
            np.testing.assert_array_equal(raw['rows'],rows);np.testing.assert_array_equal(raw['crops'],expected_crops)
            for seed,record in zip(protocol['seeds'],result['seeds'],strict=True):
                if seed!=record['seed']:raise AssertionError('Arithmetic seed identity changed')
                for k in raw.files:
                    if not np.isfinite(raw[k]).all():raise AssertionError('Nonfinite arithmetic observation')
                for precision in ['32','64']:
                    e,t,r=(raw[f's{seed}_{k}{precision}'] for k in ['E','T','R'])
                    if e.shape!=(512,5,cell['heads']) or (e<0).any() or (t<0).any() or (e>t*(1+1e-12)).any():
                        raise AssertionError('Invalid absolute row energies')
                    close(e/np.maximum(t,1e-30),r)
                parent=root/cell['parent_study']/'runs'/cell['parent_pattern'].format(seed=seed)
                with np.load(parent/'measurements.npz') as archived:
                    np.testing.assert_array_equal(raw[f's{seed}_Rarchive'],archived[f"heads_{cell['step']}"][...,2])
                kl=raw[f's{seed}_KL']
                if kl.shape!=(512,) or kl.min() < -1e-12:raise AssertionError('Invalid paired predictive KL')
                close([kl.mean(),kl.max()],[record['mean_kl'],record['max_kl']])
                for k in fields:fields[k].append(raw[f's{seed}_{k}'].copy())
        fields={k:np.stack(v) for k,v in fields.items()};h=cell['heads']
        check_transfer(fields['Rarchive'].mean(-1),fields['R64'].mean(-1),h,result['transfer'])
        check_transfer(fields['Rarchive'].mean(-1),fields['R32'].mean(-1),h,result['statistic_transfer'])
        for k in fields:
            if [r['threshold'] for r in result['screens'][k]]!=protocol['screen_levels']:raise AssertionError('Resolution screens omitted')
            for screen in result['screens'][k]:check_screen(fields[k],h,screen)
        t=result['transfer'];agree=t['relative_bound']<=.01 and t['absolute_bound']<=1e-8
        expected='paired arithmetic agreement' if agree else 'insufficient arithmetic resolution at declared tolerance'
        if result['precision_status']!=expected:raise AssertionError('Outcome-dependent precision criterion')
        arithmetic.append(dict(run_id=name,manifest_sha256=digest(folder/'manifest.json'),precision_status=expected))
        print('Verified arithmetic cell',name,flush=True)
    if a.arithmetic_only:
        if any(sha256(repo/n)!=h for n,h in sources.items()):raise AssertionError('Verifier dependencies changed')
        write_json(a.output,dict(status='passed',scope='Intermediate arithmetic reconstruction; not a release gate',
                               arithmetic=arithmetic,verifier_sources=sources,hashed_files=len(cache)))
        return
    inherited={}
    for name,count in [('criticality',272),('dynamics',404)]:
        chosen=getattr(a,name+'_verification')
        path=Path(chosen) if chosen else repo/'docs'/f'observation-{name}-verification.json'
        record=load_json_strict(path.read_text())
        if record['status']!='passed' or record['required_scientific_runs']!=count:
            raise AssertionError('A complete fresh inherited verification is required')
        bound(repo/'scripts'/f'verify_{name}.py',record['verifier_sha256'])
        for filename,signature in record.get('verifier_sources',{}).items():bound(repo/filename,signature)
        inherited[name]=dict(path=str(path),sha256=digest(path),required_scientific_runs=count)
    protocol_path=study/'protocols/paired-horizon.json';paired=load_json_strict(protocol_path.read_text())
    launcher=load_json_strict((study/'launcher-paired-horizon.json').read_text())
    if launcher['status']!='complete' or len(paired['jobs'])!=12 or len(launcher['records'])!=12:
        raise AssertionError('Every declared training continuation must be complete')
    bound(protocol_path,launcher['protocol_sha256'])
    if launcher['jobs']!=paired['jobs'] or any(r['returncode']!=0 for r in launcher['records']):
        raise AssertionError('Unresolved paired continuation')
    if [j['seed'] for j in paired['jobs']]!=list(range(640105,640117)):
        raise AssertionError('The additional seed panel changed')
    train_tokens=np.load(root/'data/refinedweb-4608/tokens.npy',mmap_mode='r')
    expected_rows=np.random.default_rng(641001).integers(0,3072,size=(16384,32))
    expected_offsets=np.random.default_rng(1641001).integers(0,449,size=(16384,32))
    target_counts=np.bincount(train_tokens[expected_rows,expected_offsets+64].ravel(),minlength=32000)
    training=[];loaded={};snapshots=[]
    for job in paired['jobs']:
        folder=study/'runs'/job['run_id'];meta=load_json_strict((folder/'manifest.json').read_text());args=meta['arguments']
        if meta['status']!='complete' or meta['start_step']!=8192 or meta['completed_step']!=16384:
            raise AssertionError('Invalid completed horizon')
        for k in ['heads','seed','multiplier','normalization','steps','resume']:
            if args[k]!=job[k]:raise AssertionError('Continuation condition changed: '+k)
        if (args['shared_seed'],args['stream_seed'],args['probe_every'])!=(640011,640001,32):
            raise AssertionError('Conditional training environment changed')
        snapshot=binding(folder,meta);snapshots.append((folder,snapshot))
        bound(job['resume'],job['parent_sha256'])
        for name,key in [('measurements.npz','raw_sha256'),('sampling.npz','sampling_sha256'),('final-training-state.pt','checkpoint_sha256')]:
            bound(folder/name,meta[key])
        parent=Path(job['resume']).parent;parent_meta=load_json_strict((parent/'manifest.json').read_text())
        for k in ['initial_parameter_sha256','shared_generator_sha256','parameter_names']:
            if meta[k]!=parent_meta[k]:raise AssertionError('Continuation initial identity changed')
        with np.load(folder/'sampling.npz') as sampling:
            np.testing.assert_array_equal(sampling['rows'],expected_rows);np.testing.assert_array_equal(sampling['offsets'],expected_offsets)
            np.testing.assert_array_equal(sampling['evaluation_rows'],np.arange(512,1024))
        with np.load(folder/'measurements.npz') as raw, np.load(parent/'measurements.npz') as old:
            for k in raw.files:
                if not np.isfinite(raw[k]).all():raise AssertionError('Nonfinite training observation')
            for k in ['heads','fields','context_kl','operator_dispersion']:
                np.testing.assert_array_equal(raw[f'{k}_8192'],old[f'{k}_8192'])
            np.testing.assert_array_equal(raw['steps'],np.arange(8192,16385,32))
            np.testing.assert_array_equal(raw['supervised_target_counts'],target_counts)
            if raw['losses'].shape!=(8192,) or raw['gradient_norms'].shape!=(8192,3) or raw['dense_heads'].shape!=(257,16,5,14,4):
                raise AssertionError('Missing continuation updates or probes')
            for t in [8192,16384]:
                if raw[f'heads_{t}'].shape!=(512,5,14,4):raise AssertionError('Endpoint cohort changed')
            loaded[14,job['seed']]={k:raw[k].astype(float) for t in [8192,16384] for k in [f'heads_{t}',f'fields_{t}',f'context_kl_{t}']}
        state=torch.load(folder/'final-training-state.pt',map_location='cpu',mmap=True,weights_only=True)
        if state['step']!=16384 or {float(v['step']) for v in state['optimizer']['state'].values()}!={16384.}:
            raise AssertionError('Optimizer time was reset')
        for group,rate in zip(state['optimizer']['param_groups'],[3e-4,3e-4*2/14],strict=True):
            if group['lr']!=rate or group['eps']!=1e-8 or group['weight_decay']!=.01 or tuple(group['betas'])!=(.9,.95):
                raise AssertionError('Optimizer law changed')
        for value in state['model'].values():
            if not torch.isfinite(value).all():raise AssertionError('Nonfinite final weight')
        for value in state['optimizer']['state'].values():
            if not torch.isfinite(value['exp_avg']).all() or not torch.isfinite(value['exp_avg_sq']).all() or (value['exp_avg_sq']<0).any():
                raise AssertionError('Invalid final optimizer moment')
        del state
        training.append(dict(run_id=job['run_id'],role='training_continuation',manifest_sha256=digest(folder/'manifest.json')))
        print('Verified continuation',job['run_id'],flush=True)
    for name,signature in launcher['source_files'].items():
        archived=next((folder/'source'/name for folder,b in snapshots if b['source_files'].get(name)==signature),None)
        if archived is None:raise AssertionError('Launcher source was not preserved')
        bound(archived,signature)
    for h in [2,4,8,14,24]:
        for seed in range(640101,640105):
            path=root/'criticality-dynamics-20260906/runs'/f'long-h{h}-g1-s{seed}'
            meta=load_json_strict((path/'manifest.json').read_text());bound(path/'measurements.npz',meta['raw_sha256'])
            with np.load(path/'measurements.npz') as raw:
                loaded[h,seed]={k:raw[k].astype(float) for t in [8192,16384] for k in [f'heads_{t}',f'fields_{t}',f'context_kl_{t}']}
    analysis=study/'analysis/complete';meta=load_json_strict((analysis/'manifest.json').read_text());binding(analysis,meta)
    bound(analysis/'results.json',meta['results_sha256']);bound(analysis/'measurements.npz',meta['raw_sha256'])
    report=load_json_strict((analysis/'results.json').read_text())
    if len(report['arithmetic'])!=13 or len(report['horizons'])!=7:raise AssertionError('Analysis omitted a declared condition')
    for r in report['arithmetic']:
        c=r['cell'];name=f"arithmetic-h{c['heads']}-g{c['multiplier']}-t{c['step']}"
        if r!=arithmetic_by_id[name]:raise AssertionError('Arithmetic summary was altered')
    if len(report['arithmetic_energies'])!=13:raise AssertionError('Absolute energy summaries omitted')
    for energy in report['arithmetic_energies']:
        c=energy['cell'];name=f"arithmetic-h{c['heads']}-g{c['multiplier']}-t{c['step']}"
        for k,v in energy.items():
            if k!='cell':close(v,energy_by_id[name][k])
    groups=[(f'four-N{h}',h,list(range(640101,640105))) for h in [2,4,8,14,24]]
    groups += [('all-sixteen-N14',14,list(range(640101,640117))),('additional-twelve-N14',14,list(range(640105,640117)))]
    with np.load(analysis/'measurements.npz') as arrays:
        for (name,h,seeds),r in zip(groups,report['horizons'],strict=True):
            if r['group']!=name or r['seeds']!=seeds or r['independent_seeds']!=len(seeds):raise AssertionError('Paired analysis identities changed')
            x,y=(np.stack([loaded[h,s][f'heads_{t}'][...,2].mean(-1) for s in seeds]) for t in [8192,16384])
            np.testing.assert_array_equal(x,arrays[name+'-before']);np.testing.assert_array_equal(y,arrays[name+'-after'])
            chi,other=variance(x,h),variance(y,h);delta=other-chi;s=len(seeds)
            close([chi,other,delta,x.mean(),y.mean()],[r[k] for k in ['before','after','difference','mean_before','mean_after']],atol=2e-14)
            leave=np.array([variance(np.delete(y,i,0),h)-variance(np.delete(x,i,0),h) for i in range(s)])
            close(leave,r['leave_one_out'],atol=2e-14);close([leave.min(),leave.max()],r['leave_one_out_range'],atol=2e-14)
            close(np.sqrt((s-1)/s*((leave-leave.mean())**2).sum()),r['jackknife_se'],atol=2e-14)
            close((y-x).reshape(s,-1).mean(1),r['seed_mean_changes'])
            indices=(np.array(list(itertools.product(range(s),repeat=s))) if s==4 else
                     np.random.default_rng(paired['analysis']['bootstrap_seed']).integers(0,s,(paired['analysis']['resamples'],s)))
            np.testing.assert_array_equal(indices,arrays[name+'-indices'])
            weights=np.array([np.bincount(i,minlength=s) for i in indices]);np.testing.assert_array_equal(weights,arrays[name+'-weights'])
            # Independent seed-pair distance identity for all resamples.
            distances=np.array([[np.mean((y[i]-y[j])**2)-np.mean((x[i]-x[j])**2) for j in range(s)] for i in range(s)])
            samples=h/(2*s*(s-1))*np.einsum('bi,ij,bj->b',weights,distances,weights)
            close(samples,arrays[name+'-differences'],atol=2e-13)
            for k in (range(256) if s==4 else np.linspace(0,len(indices)-1,48,dtype=int)):
                direct=variance(y[indices[k]],h)-variance(x[indices[k]],h)
                close(direct,samples[k],atol=2e-13)
            close(np.quantile(samples,[.025,.975]),r['percentiles'],atol=2e-13)
            if r['resamples']!=len(indices):raise AssertionError('Bootstrap count changed')
            for t,suffix in [(8192,'before'),(16384,'after')]:
                close([loaded[h,seed][f'fields_{t}'][:,25].mean() for seed in seeds],r['nll_'+suffix])
                close([loaded[h,seed][f'context_kl_{t}'].mean() for seed in seeds],r['context_kl_'+suffix])
            print('Verified paired uncertainty',name,flush=True)
    if len(report['dense_entry_screens'])!=12 or len(report['replication_entry_screens'])!=15:
        raise AssertionError('Precision panels were omitted')
    for r in report['dense_entry_screens']:
        dense=[];end=[]
        for seed in range(640101,640105):
            path=root/'criticality-dynamics-20260906/runs'/f"pilot-h{r['heads']}-g{r['multiplier']}-s{seed}"
            with np.load(path/'measurements.npz') as z:
                np.testing.assert_array_equal(z['steps'],r['steps']);dense.append(z['dense_heads'][...,2]);end.append(z['heads_8192'][...,2])
        close(np.mean(np.stack(dense)<1e-13,axis=(0,2,3,4)),r['fraction'])
        close(np.mean(np.stack(end)<1e-13),r['endpoint_fraction'])
    replication=load_json_strict((root/'criticality-dynamics-20260906/protocols/replication.json').read_text())
    parents=[root/'criticality-dynamics-20260906/runs'/j['run_id'] for j in replication['jobs']]+[Path(v) for v in replication['additional_bound_runs']]
    for r in report['replication_entry_screens']:
        fields=[]
        for parent in parents:
            m=load_json_strict((parent/'manifest.json').read_text())
            if m['arguments']['heads']==r['heads']:
                with np.load(parent/'measurements.npz') as z:fields.append(z[f"heads_{r['step']}"][...,2])
        if len(fields)!=r['seeds']:raise AssertionError('Replication denominator changed')
        close(np.mean(np.stack(fields)<1e-13),r['fraction'])
    projection=study/'analysis/row-projection';m=load_json_strict((projection/'manifest.json').read_text());binding(projection,m)
    bound(projection/'results.json',m['results_sha256'])
    if report['released']!=load_json_strict((projection/'results.json').read_text())['released']:raise AssertionError('Released-row summary changed')
    formal_path=Path(a.statement_verification) if a.statement_verification else repo/'docs/observation-statements/verification.json'
    formal=load_json_strict(formal_path.read_text())
    if formal['status']!='passed' or formal['exports']!=a.expected_statements:raise AssertionError('Maintained statement contracts are required')
    for name,signature in {**formal['modules'],**formal['support_sources']}.items():bound(repo/name,signature)
    bound(repo/'scripts/check_statement_contracts.py',formal['checker_sha256'])
    bound(repo/'scripts/formal/statement-registry.json',formal['registry_sha256'])
    for check in formal['checks']:
        bound(formal_path.parent/(check['name']+'.log'),check['log_sha256'])
    if len(formal['mutations'])!=5 or not all(m['rejected'] for m in formal['mutations']):
        raise AssertionError('Statement content mutation checks are required')
    numerical_path=Path(a.numerical_verification) if a.numerical_verification else repo/'docs/observation-numerical-verification.json'
    numerical=load_json_strict(numerical_path.read_text())
    if numerical['status']!='passed' or numerical['tests_passed']!=a.expected_tests or numerical['returncode']!=0:
        raise AssertionError('The final numerical regression is required')
    bound(numerical['log'],numerical['log_sha256'])
    for name,signature in numerical['tested_sources'].items():bound(repo/name,signature)
    if any(sha256(repo/n)!=h for n,h in sources.items()):raise AssertionError('A verification dependency changed')
    result=dict(schema='observation-complete-evidence-v1',status='passed',required_scientific_runs=416,
        inventory_definition='404 inherited required scientific artifacts plus 12 new training continuations; 13 arithmetic measurement conditions and seven paired summaries are separately enumerated.',
        inherited_verifications=inherited,new_training=training,arithmetic=arithmetic,paired_groups=[g[0] for g in groups],
        observation_analysis=str(analysis/'results.json'),observation_analysis_sha256=digest(analysis/'results.json'),
        protocol_sha256={p.name:digest(p) for p in [protocol_path,arithmetic_path]},
        numerical_verification=dict(path=str(numerical_path),sha256=digest(numerical_path),tests_passed=a.expected_tests),
        formal_statement_verification=dict(path=str(formal_path),sha256=digest(formal_path),registered_statements=a.expected_statements),
        verifier_sha256=digest(__file__),verifier_sources=sources,hashed_files=len(cache),
        statistical_scope='Conditional fixed-context whole-seed paired empirical uncertainty; no calibrated coverage or thermodynamic criticality claim.')
    write_json(a.output,result);print('Complete observation evidence passed:',416,'required scientific artifacts and',len(arithmetic),'arithmetic conditions',flush=True)


if __name__=='__main__':main()
