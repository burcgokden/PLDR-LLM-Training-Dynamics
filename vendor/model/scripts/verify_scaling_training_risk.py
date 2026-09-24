#!/usr/bin/env python
"""Independently reconstruct risk crops, logit losses and paired statistics."""
import argparse
import itertools
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def close(left,right):
    np.testing.assert_allclose(left,right,rtol=2e-10,atol=1e-18)


def same_bytes(a,b):
    return a.dtype == b.dtype and a.shape == b.shape and (
        np.ascontiguousarray(a).tobytes() == np.ascontiguousarray(b).tobytes())


def check_moments(record,x,n,draws):
    flat=x.reshape(4,-1);centered=flat-flat.mean(0)
    second=np.mean(centered**2);fourth=np.mean(centered**4)
    cohort=flat.mean(1);d=cohort-cohort.mean();m2=np.mean(d*d);m4=np.mean(d**4)
    leave=np.array([n*np.var(np.delete(flat,i,axis=0),axis=0,ddof=1).mean() for i in range(4)])
    bootstrap=n*np.var(flat[draws],axis=1,ddof=1).mean(-1)
    values=dict(mean=flat.mean(),susceptibility=n*np.var(flat,axis=0,ddof=1).mean(),
        central_second=second,central_fourth=fourth,fourth_ratio=fourth/second**2 if second>0 else None,
        independent_seeds=4,contexts=x.shape[1],empirical_percentiles=np.quantile(bootstrap,[.025,.975]),
        jackknife_se=np.sqrt(3/4*np.sum((leave-leave.mean())**2)),leave_one_out=leave,
        bootstrap_samples=256,bootstrap_zero_fraction=np.mean(bootstrap==0),
        cohort_means=cohort,cohort_susceptibility=n*np.var(cohort,ddof=1),
        cohort_fourth_ratio=m4/m2**2 if m2>0 else None,cohort_binder=1-m4/(3*m2**2) if m2>0 else None,
        finite_seed_gaussian_binder_mean=.4)
    if set(record)!=set(values):raise AssertionError('The risk moment schema changed')
    for key,want in values.items():
        if want is None:
            if record[key] is not None:raise AssertionError('Undefined risk moment changed')
        else:close(record[key],want)
    return bootstrap


def main():
    p = argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--analysis',help='Complete risk analysis directory; defaults to the selected study analysis')
    a = p.parse_args();root = Path(a.root);study = root/a.study;repo = Path(__file__).resolve().parents[1]
    output = Path(a.output)
    if output.exists():raise FileExistsError(output)
    checked = {};cache = {}
    def check(path,signature=None):
        path = Path(path).resolve();st = path.stat();key=(str(path),st.st_size,st.st_mtime_ns)
        if key not in cache:cache[key]=sha256(path)
        got=cache[key]
        if signature is not None and got != signature:raise AssertionError('Changed risk evidence: '+str(path))
        checked[str(path)]=got;return got
    def artifact(folder):
        folder=Path(folder);meta=load_json_strict((folder/'manifest.json').read_text())
        if meta['status']!='complete':raise AssertionError('Incomplete risk artifact')
        check(folder/'manifest.json');check(folder/'binding.json',meta['binding_sha256'])
        binding=load_json_strict((folder/'binding.json').read_text())
        for name,signature in binding['source_files'].items():check(folder/'source'/name,signature)
        for name,signature in binding['inputs'].items():check(name,signature)
        for name,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256')]:check(folder/name,meta[key])
        return meta,binding
    protocol=study/'protocols/training-risk-selection.json';check(protocol)
    spec=load_json_strict(protocol.read_text());check(spec['cohort'],spec['cohort_sha256']);check(spec['cohort_manifest'])
    cm=load_json_strict(Path(spec['cohort_manifest']).read_text());check(cm['source'],cm['source_sha256'])
    if (spec['cohort_seed'],spec['batch_size'],spec['precision'],spec['online_window']) != (650581,32,'float32',8192):
        raise AssertionError('The selected risk observation program changed')
    if cm['status']!='complete' or cm['raw_sha256']!=spec['cohort_sha256'] or cm['seed']!=spec['cohort_seed']:
        raise AssertionError('The frozen risk cohort record changed')
    expected={(n,t,seed) for n in [4,14] for t in [32768,65536,98304,131072] for seed in range(640101,640105)}
    if len(spec['cases'])!=32 or {(c['heads'],c['step'],c['seed']) for c in spec['cases']}!=expected:
        raise AssertionError('The risk design is not the selected balanced panel')
    if any((c['multiplier'],c['shared_seed'],c['stream_seed'])!=(1,640011,640001) for c in spec['cases']):
        raise AssertionError('The conditioning law changed')
    with np.load(spec['cohort']) as z:cohort={k:z[k] for k in z.files}
    rows=np.random.default_rng(spec['cohort_seed']+1000).integers(0,3072,size=(32,32)).ravel()
    offsets=np.random.default_rng(spec['cohort_seed']+1001000).integers(0,449,size=(32,32)).ravel()
    tokens=np.load(cm['source'],mmap_mode='r');crops=tokens[rows[:,None],offsets[:,None]+np.arange(65)]
    for name,want in [('rows',rows),('offsets',offsets),('crops',crops)]:
        np.testing.assert_array_equal(cohort[name],want)
    probe=root/'controlled-study-20260905/data/short';pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy')
    held=pt[np.arange(512,1024)[:,None],po[512:1024,None]+np.arange(65)]
    fields=['training_nll','heldout_nll','training_entropy','heldout_entropy'];values={};logit_elements=0;max_logit_error=0.;state_records=[]
    for case in spec['cases']:
        folder=study/'measurements'/case['name'];meta,binding=artifact(folder)
        if meta['case']!=case:raise AssertionError('The selected risk case changed')
        result=load_json_strict((folder/'results.json').read_text())
        if result['case']!=case:raise AssertionError('The risk result case changed')
        parent=load_json_strict(Path(case['parent_manifest']).read_text());ref=load_json_strict(Path(case['reference_observation']).read_text())
        check(case['state'],parent['saved_states'][str(case['step'])]['sha256'])
        check(case['state'],ref['case']['state_sha256'])
        if parent['status']!='complete' or ref['status']!='complete':raise AssertionError('Incomplete risk parent')
        for key in ['heads','seed','multiplier','shared_seed','stream_seed']:
            if ref['case'][key]!=case[key] or parent['arguments'][key]!=case[key]:
                raise AssertionError('The risk checkpoint condition changed')
        if ref['case']['step']!=case['step']:raise AssertionError('The reference risk horizon changed')
        check(Path(case['parent_manifest']).parent/'measurements.npz',parent['raw_sha256'])
        check(Path(case['reference_observation']).parent/'measurements.npz',ref['raw_sha256'])
        with np.load(folder/'measurements.npz') as z:q={k:z[k] for k in z.files}
        for label,x in [('training',crops),('heldout',held)]:
            if q[label+'_nll'].shape!=(len(x),) or q[label+'_entropy'].shape!=(len(x),):raise AssertionError('Risk context count changed')
            np.testing.assert_array_equal(q[label+'_first_targets'],x[:32,64])
            if q[label+'_first_logits'].dtype!=np.float32 or q[label+'_first_logits'].shape[0]!=32:
                raise AssertionError('The retained first-batch logits changed their observation program')
            if any(q[label+'_'+name].dtype!=np.float32 for name in ['nll','entropy']):
                raise AssertionError('Risk precision changed')
            logits=q[label+'_first_logits'].astype(float);maximum=logits.max(1,keepdims=True)
            logp=logits-maximum-np.log(np.exp(logits-maximum).sum(1,keepdims=True))
            nll=-logp[np.arange(32),x[:32,64]];entropy=-(np.exp(logp)*logp).sum(1)
            for name,want in [('nll',nll),('entropy',entropy)]:
                error=float(np.max(np.abs(q[label+'_'+name][:32]-want)))
                max_logit_error=max(max_logit_error,error)
                np.testing.assert_allclose(q[label+'_'+name][:32],want,rtol=2e-6,atol=3e-5)
                if not np.isfinite(q[label+'_'+name]).all():raise AssertionError('Nonfinite risk array')
                close(result['cohorts'][label][name],q[label+'_'+name].astype(float).mean())
            logit_elements+=logits.size
        with np.load(Path(case['reference_observation']).parent/'measurements.npz') as z:rf=z['float32_fields']
        error=max(float(np.max(np.abs(q['heldout_'+name]-rf[:,column]))) for name,column in [('nll',25),('entropy',24)])
        close(error,result['heldout_reference_maximum_error'])
        if finite_greater(error, 1e-5, 'scripts/verify_scaling_training_risk.py:124'):raise AssertionError('The held-out reference differs materially')
        bits=all(same_bytes(q['heldout_'+name],rf[:,column]) for name,column in [('nll',25),('entropy',24)])
        if bits!=result['heldout_reference_bitwise_equal']:raise AssertionError('Incorrect reference byte comparison')
        with np.load(Path(case['parent_manifest']).parent/'measurements.npz') as z:online=z['losses']
        end=case['step']-parent['start_step'];window=online[end-spec['online_window']:end]
        if window.shape!=(8192,) or not same_bytes(window,q['preceding_online_losses']):
            raise AssertionError('The preceding online loss window changed')
        close(result['online_window']['preupdate_mean_loss'],window.mean())
        if (result['online_window']['begin'],result['online_window']['end'])!=(case['step']-8192,case['step']):
            raise AssertionError('The online window endpoints changed')
        state_records.append(dict(case=case,manifest_sha256=check(folder/'manifest.json'),
            heldout_reference_maximum_error=error,heldout_reference_bitwise_equal=bits))
        values[case['heads'],case['step'],case['seed']]={name:q[name].astype(float) for name in fields}
        values[case['heads'],case['step'],case['seed']]['online_mean_loss']=float(window.mean())
    analysis=Path(a.analysis) if a.analysis else study/'analysis/training-risk';artifact(analysis);report=load_json_strict((analysis/'results.json').read_text())
    if len(values)!=32 or report['states']!=state_records:raise AssertionError('Incomplete risk state report')
    close(report['maximum_reference_error'],max(x['heldout_reference_maximum_error'] for x in state_records))
    if report['reference_bitwise_equal_states']!=sum(x['heldout_reference_bitwise_equal'] for x in state_records):
        raise AssertionError('The risk reference byte count changed')
    if report['cohort_contexts']!=dict(training=1024,heldout=512) or report['online_window']!=8192:
        raise AssertionError('Risk cohort or online window size changed')
    if report['selection_scope']!=spec['scope']:raise AssertionError('Risk selection history was changed')
    if len(report['conditions'])!=8 or {(x['heads'],x['step']) for x in report['conditions']}!={(n,t) for n,t,s in expected}:
        raise AssertionError('A selected risk condition is missing')
    if len(report['paired_horizons'])!=6 or {(x['heads'],x['early'],x['late']) for x in report['paired_horizons']}!={(n,t,u) for n in [4,14] for t,u in zip([32768,65536,98304],[65536,98304,131072])}:
        raise AssertionError('A selected risk horizon pair is missing')
    draws=np.array(list(itertools.product(range(4),repeat=4)))
    moments={};data={};moment_checks=0;paired_checks=0
    with np.load(analysis/'measurements.npz') as z:raw={k:z[k] for k in z.files}
    for row in report['conditions']:
        n,t=row['heads'],row['step'];seeds=list(range(640101,640105))
        if row['seeds']!=seeds:raise AssertionError('Risk initialization identities changed')
        data[n,t]={name:np.stack([values[n,t,s][name] for s in seeds]) for name in fields}
        for name,x in data[n,t].items():
            m=row['observables'][name];flat=x.reshape(4,-1)
            bootstrap=check_moments(m,x,n,draws)
            close(m['mean'],flat.mean());close(m['susceptibility'],n*np.var(flat,axis=0,ddof=1).mean())
            close(m['cohort_means'],flat.mean(1));close(m['empirical_percentiles'],np.quantile(bootstrap,[.025,.975]))
            close(raw[f'h{n}_t{t}_{name}_bootstrap'],bootstrap);moments[n,t,name]=bootstrap;moment_checks+=1
        gap=data[n,t]['heldout_nll'].mean(1)-data[n,t]['training_nll'].mean(1)
        close(row['heldout_minus_training']['mean'],gap.mean())
        close(row['heldout_minus_training']['cohort_means'],gap)
        close(row['heldout_minus_training']['empirical_percentiles'],np.quantile(gap[draws].mean(1),[.025,.975]))
        close(row['online_mean_loss'],np.mean([values[n,t,s]['online_mean_loss'] for s in seeds]))
    for row in report['paired_horizons']:
        n,early,late=row['heads'],row['early'],row['late']
        for name in fields:
            before,after=data[n,early][name],data[n,late][name];r=row['observables'][name]
            check_moments(r['before'],before,n,draws);check_moments(r['after'],after,n,draws)
            means=(after-before).mean(1);delta=moments[n,late,name]-moments[n,early,name]
            leave=np.array([n*(np.var(np.delete(after,i,axis=0),axis=0,ddof=1).mean()-
                np.var(np.delete(before,i,axis=0),axis=0,ddof=1).mean()) for i in range(4)])
            close(r['susceptibility_change_jackknife_se'],np.sqrt(3/4*np.sum((leave-leave.mean())**2)))
            close(r['paired_field_rms'],np.sqrt(np.mean((after-before)**2)))
            close(r['finite_relative_susceptibility_change'],np.var(after,axis=0,ddof=1).mean()/np.var(before,axis=0,ddof=1).mean()-1)
            close(r['mean_change'],means.mean());close(r['per_initialization_mean_change'],means)
            close(r['mean_change_empirical_percentiles'],np.quantile(means[draws].mean(1),[.025,.975]))
            close(r['susceptibility_change'],n*(np.var(after,axis=0,ddof=1).mean()-np.var(before,axis=0,ddof=1).mean()))
            close(r['susceptibility_change_empirical_percentiles'],np.quantile(delta,[.025,.975]))
            close(raw[f'h{n}_t{early}_{late}_{name}_paired_bootstrap'],delta);paired_checks+=1
    if moment_checks!=32 or paired_checks!=24:raise AssertionError('A selected risk contrast is missing')
    sources={name:sha256(repo/name) for name in ['scripts/verify_scaling_training_risk.py','src/model_rg/provenance.py']}
    write_json(output,dict(schema='training-risk-reconstruction-v1',status='passed',selected_states=32,
        moment_fields=moment_checks,paired_fields=paired_checks,logit_elements=logit_elements,
        protocol_sha256=check(protocol),analysis=str((analysis/'results.json').resolve()),
        analysis_sha256=check(analysis/'results.json'),states=state_records,
        maximum_first_batch_logit_reconstruction_error=max_logit_error,verified_files=checked,verifier_sources=sources,
        scope='Independent crop/target reconstruction, completed checkpoint hashes, full-cohort risk statistics, byte comparison of online windows and held-out reference fields, all paired empirical contrasts, and float64 softmax reconstruction from the retained first32 logits in each cohort. The remaining forward emissions are source-bound observations, not independent model replays. Empirical ranges have no guaranteed population coverage.'))
    print('Training-risk reconstruction passed:',len(values),'states;',moment_checks,'moment fields;',paired_checks,'paired fields',flush=True)


if __name__ == '__main__':main()
