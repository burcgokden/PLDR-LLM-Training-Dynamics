#!/usr/bin/env python
"""Reconstruct scheduled observations with exact native risk-reduction replay."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path

import numpy as np
import torch

from model_rg.provenance import sha256,write_json
from verify_scaling_raw import BoundFiles,bitwise_equal,verify_measurement


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=3e-10,atol=1e-14)


def log_prob(z):
    z=z.astype(np.float64);z-=z.max(-1,keepdims=True)
    return z-np.log(np.exp(z).sum(-1,keepdims=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908');p.add_argument('--case',required=True)
    p.add_argument('--selection',default='onepass-observation-selection.json')
    p.add_argument('--output',help='Optional immutable check destination for observer qualification');a=p.parse_args()
    root=Path(a.root).resolve();study=root/a.study;selection=study/'protocols'/a.selection
    output=Path(a.output).resolve() if a.output else study/'verification/observations'/(a.case+'.json')
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4);files=BoundFiles();files.digest(selection)
    spec=load_json_strict(selection.read_text());cases=[c for c in spec['cases'] if c['name']==a.case]
    if len(cases)!=1:raise AssertionError('A unique frozen observation is required')
    case=cases[0]
    spec=dict(spec,**{k:case[k] for k in ['training_cohort','training_cohort_sha256','training_cohort_manifest']})
    parent=Path(case['parent_manifest']);pm=load_json_strict(parent.read_text())
    if pm['status']!='complete':raise AssertionError('Incomplete scheduled parent')
    files.digest(parent);files.check(case['state'],pm['saved_states'][str(case['step'])]['sha256'])
    for key in ['heads','seed','recipe','shared_seed','stream_seed']:
        if pm['arguments'][key]!=case[key]:raise AssertionError('Scheduled condition changed')
    cpu=study/'measurements'/case['name'];collective=verify_measurement(root,cpu/'manifest.json',files)
    cpu_meta=load_json_strict((cpu/'manifest.json').read_text())
    for key in case:
        if cpu_meta['case'][key]!=case[key]:raise AssertionError('CPU observation differs from the frozen selection')
    if collective['precisions']!=['float32','float64']:raise AssertionError('The paired arithmetic observation is missing')

    def artifact(folder):
        meta=load_json_strict((folder/'manifest.json').read_text());files.digest(folder/'manifest.json')
        if meta['status']!='complete' or meta['case']!=case:raise AssertionError('Incomplete or incorrect scheduled observation')
        files.binding(folder,meta)
        files.check(folder/'measurements.npz',meta['raw_sha256']);files.check(folder/'results.json',meta['results_sha256'])
        result=load_json_strict((folder/'results.json').read_text())
        if result['case']!=case:raise AssertionError('Scheduled result condition changed')
        with np.load(folder/'measurements.npz') as z:raw={k:z[k] for k in z.files}
        return result,raw

    files.check(spec['training_cohort'],spec['training_cohort_sha256'])
    cm=load_json_strict(Path(spec['training_cohort_manifest']).read_text());files.digest(spec['training_cohort_manifest'])
    files.check(cm['source'],cm['source_sha256'])
    rt=np.load(cm['source'],mmap_mode='r')
    if rt.shape!=(524288,513) or cm['step']!=case['step']: raise AssertionError('The observed consumed-data population changed')
    blocks=np.arange(4194304,dtype=np.int64)
    np.random.default_rng(case['stream_seed']+1000).shuffle(blocks)
    positions=np.random.default_rng(651581).choice(32*case['step'],size=1024,replace=False)
    selected=blocks[positions]
    rr=selected//8;ro=64*(selected%8)
    if len(np.unique(selected))!=1024: raise AssertionError('Repeated training-risk block')
    training=rt[rr[:,None],ro[:,None]+np.arange(65)]
    with np.load(spec['training_cohort']) as z:
        for name,value in [('rows',rr),('offsets',ro),('crops',training),('consumed_stream_positions',positions)]:np.testing.assert_array_equal(z[name],value)
    probe=root/'controlled-study-20260905/data/short'
    tokens=np.load(probe/'tokens.npy',mmap_mode='r');offsets=np.load(probe/'offsets.npy')
    rows=np.arange(512,1024);held=tokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    risk,q=artifact(study/'measurements'/('risk-'+case['name'].removeprefix('cpu-')))
    logit_elements=0;maximum_error=0.;native_reduction_values=0;paired_reductions=[]
    for label,crops in [('training',training),('heldout',held)]:
        mask=crops[:,1:]!=0;np.testing.assert_array_equal(q[label+'_all_mask'],mask)
        for kind in ['last','all']:
            logits=q[label+'_first_'+kind+'_logits'];count=32 if kind=='last' else 2
            targets=crops[:count,64] if kind=='last' else crops[:count,1:]
            np.testing.assert_array_equal(q[label+'_first_'+kind+'_targets'],targets)
            if logits.dtype!=np.float32:raise AssertionError('Risk logit arithmetic changed')
            lp=log_prob(logits);nll=-np.take_along_axis(lp,targets[...,None],axis=-1).squeeze(-1)
            entropy=-(np.exp(lp)*lp).sum(-1)
            # Replay the declared float32 reduction on the exact retained
            # logits. The all-position CE kernel reduces a transposed vocabulary
            # axis; it need not agree with last-axis log_softmax to a fixed
            # absolute tolerance. NumPy float64 remains a separately reported
            # mathematical reduction, never substituted for the recorded risk.
            native_logits=torch.from_numpy(logits)
            native_targets=torch.as_tensor(targets,dtype=torch.long)
            native_lp=native_logits.log_softmax(-1)
            native_nll=(-native_lp.gather(-1,native_targets[...,None]).squeeze(-1) if kind=='last' else
                torch.nn.functional.cross_entropy(native_logits.transpose(1,2),native_targets,reduction='none'))
            native_entropy=-(native_lp.exp()*native_lp).sum(-1)
            for field,want,replayed in [('nll',nll,native_nll.numpy()),('entropy',entropy,native_entropy.numpy())]:
                value=q[label+'_'+kind+'_'+field]
                expected_shape=(len(crops),) if kind=='last' else (len(crops),64)
                if value.dtype!=np.float32 or value.shape!=expected_shape or not np.isfinite(value).all():
                    raise AssertionError('Risk observations have invalid shape, precision or values')
                if not bitwise_equal(value[:count],replayed):
                    raise AssertionError('A retained native float32 risk reduction did not replay bytewise')
                native_reduction_values+=replayed.size
                difference=value[:count].astype(float)-want
                maximum_error=max(maximum_error,float(np.max(np.abs(difference))))
                paired_reductions.append(dict(cohort=label,kind=kind,field=field,values=replayed.size,
                    native_replay_byte_equal=True,maximum_float64_difference=float(np.max(np.abs(difference))),
                    mean_float64_difference=float(difference.mean()),rms_float64_difference=float(np.sqrt(np.mean(difference*difference)))))
                if not (kind=='all' and field=='nll'):
                    np.testing.assert_allclose(value[:count],want,rtol=2e-6,atol=3e-5)
            logit_elements+=logits.size
        expected=dict(contexts=len(crops),valid_all_targets=int(mask.sum()),
            last_nll=float(q[label+'_last_nll'].astype(float).mean()),last_entropy=float(q[label+'_last_entropy'].astype(float).mean()),
            all_token_weighted_nll=float(np.sum(q[label+'_all_nll'].astype(float)*mask)/mask.sum()),
            all_token_weighted_entropy=float(np.sum(q[label+'_all_entropy'].astype(float)*mask)/mask.sum()),
            native_last_batch_mean_loss=float(q[label+'_native_last_batch_loss'].mean()),
            native_all_batch_mean_loss=float(q[label+'_native_all_batch_loss'].mean()))
        if set(expected)!=set(risk['cohorts'][label]):raise AssertionError('Risk summary schema changed')
        for key,value in expected.items():close(risk['cohorts'][label][key],value)
        np.testing.assert_allclose(q[label+'_native_last_batch_loss'],q[label+'_last_nll'].reshape(-1,32).mean(-1),rtol=2e-6,atol=3e-5)
        all_batch=(q[label+'_all_nll'].astype(float)*mask).reshape(-1,32*64).sum(-1)/mask.reshape(-1,32*64).sum(-1)
        np.testing.assert_allclose(q[label+'_native_all_batch_loss'],all_batch,rtol=2e-6,atol=3e-5)
        discrepancy=q[label+'_last_projection_discrepancy']
        if discrepancy.shape!=(len(crops)//32,2) or (discrepancy<0).any():raise AssertionError('Invalid projection comparison')
        delta=q[label+'_first_all_logits'][:,-1].astype(float)-q[label+'_first_last_logits'][:2].astype(float)
        if finite_greater(np.max(np.abs(delta)), discrepancy[0,0]+1e-12, 'scripts/verify_onepass_observation.py:126'):raise AssertionError('Retained logit difference exceeds the reported batch maximum')
    with np.load(cpu/'measurements.npz') as z:fields=z['float32_fields']
    errors=[q['heldout_last_'+name]-fields[:,column] for name,column in [('nll',25),('entropy',24)]]
    error=max(float(np.max(np.abs(x))) for x in errors);close(error,risk['heldout_reference_maximum_error'])
    if finite_greater(error, 1e-5, 'scripts/verify_onepass_observation.py:130'):raise AssertionError('Scheduled risk and common CPU fields disagree')
    byte_equal=all(bitwise_equal(q['heldout_last_'+name],fields[:,column]) for name,column in [('nll',25),('entropy',24)])
    if byte_equal!=risk['heldout_reference_bitwise_equal']:raise AssertionError('Incorrect risk byte claim')
    start=max(pm['start_step'],case['step']-spec['risk']['online_window']);stop=case['step']
    with np.load(parent.parent/'measurements.npz') as z:window=z['losses'][start-pm['start_step']:stop-pm['start_step']]
    if not bitwise_equal(window,q['preceding_online_losses']):raise AssertionError('The native online risk window changed')
    if (risk['online_window']['begin'],risk['online_window']['end'])!=(start,stop):raise AssertionError('Incorrect online window endpoints')
    close(risk['online_window']['preupdate_mean_loss'],window.mean())
    inference=None
    if case['inference_stability']:
        result,q=artifact(study/'measurements'/('inference-'+case['name'].removeprefix('cpu-')))
        files.check(spec['inference_cohort'],spec['inference_cohort_sha256'])
        with np.load(spec['inference_cohort']) as z:uniforms=z['uniforms'];generation_rows=z['rows']
        np.testing.assert_array_equal(uniforms,np.random.default_rng(650701).random((2,100,32)))
        np.testing.assert_array_equal(generation_rows,np.arange(512,612))
        for key,value in [('uniforms',uniforms),('rows',generation_rows),('prompts',tokens[generation_rows[:,None],offsets[generation_rows,None]+np.arange(64)])]:
            np.testing.assert_array_equal(q[key],value)
        lengths=q['generated_lengths'];generated=q['generated_tokens'];logits=q['generation_logits'];used=0
        if lengths.shape!=(2,100) or generated.shape!=(2,100,32) or logits.shape[:3]!=(2,100,32) or logits.dtype!=np.float32:
            raise AssertionError('Inference generation shape changed')
        for run in range(2):
            for index in range(100):
                length=int(lengths[run,index])
                if not 1<=length<=32:raise AssertionError('Invalid generated length')
                if (generated[run,index,length:]!=-1).any() or (logits[run,index,length:]!=0).any():raise AssertionError('Unused generation slots were populated')
                for step in range(length):
                    z=logits[run,index,step].astype(float);prob=np.exp(z-z.max());prob/=prob.sum()
                    order=np.argsort(-prob,kind='stable');ordered=prob[order];cdf=np.cumsum(ordered)
                    keep=cdf-ordered<.8;retained=ordered*keep;retained/=retained.sum()
                    cdf=np.cumsum(retained);cdf[-1]=1
                    chosen=int(order[np.searchsorted(cdf,uniforms[run,index,step],side='left')])
                    if chosen!=generated[run,index,step]:raise AssertionError('Independent nucleus sampling differs')
                    if chosen==3 and step!=length-1:raise AssertionError('Generation continued past EOS')
                    used+=1
                if length<32 and generated[run,index,length-1]!=3:raise AssertionError('Early stop without EOS')
        stops=[int(np.sum(generated[r,np.arange(100),lengths[r]-1]==3)) for r in range(2)]
        if result['generated_tokens']!=lengths.sum(1).tolist() or result['eos_stops']!=stops:raise AssertionError('Inference stop counts changed')
        for name in ['A','A_LM','A_P','G_LM']:
            for label,left,right,reference in [('run1_run2','run1','run2','run1'),('run1_prompt','run1','prompt','prompt'),('run2_prompt','run2','prompt','prompt')]:
                x=q[left+'_'+name].astype(float);y=q[right+'_'+name].astype(float);ref=q[reference+'_'+name].astype(float)
                if x.shape!=(100,5,case['heads'],64,64):raise AssertionError('Deductive tensor shape changed')
                if not all(np.isfinite(v).all() for v in [x,y,ref]):raise AssertionError('Nonfinite deductive tensor')
                rmse=float(np.sqrt(np.mean((x-y)**2)));mean=float(ref.mean());rms=float(np.sqrt(np.mean(ref*ref)))
                expected=dict(rmse=rmse,reference_mean=mean,reference_rms=rms,mean_normalized_rmse=rmse/abs(mean) if mean else None,
                    rms_normalized_rmse=rmse/rms if rms else None,mean_magnitude_to_rms=abs(mean)/rms if rms else None,
                    per_prompt_rmse=np.sqrt(np.mean((x-y)**2,axis=(1,2,3,4))).tolist())
                got=result['comparisons'][name][label]
                if set(expected)!=set(got):raise AssertionError('Inference statistic schema changed')
                for key,want in expected.items():
                    if want is None:
                        if got[key] is not None:raise AssertionError('Undefined inference ratio changed')
                    else:close(got[key],want)
        inference=dict(sampled_tokens_independently_reconstructed=used,stop_counts=stops,comparisons=12)
    write_json(output,dict(status='passed',case=case,selection_sha256=sha256(selection),collective=collective,
        risk=dict(retained_logit_elements=logit_elements,maximum_numpy_logit_reduction_error=maximum_error,
            native_reduction_values_replayed_bytewise=native_reduction_values,paired_reductions=paired_reductions,
            heldout_reference_maximum_error=error,heldout_reference_bitwise_equal=byte_equal),inference=inference,
        checked_sha256={key[0]:value for key,value in files.cache.items()},
        verifier_sources={name:sha256(Path(__file__).resolve().parents[1]/name) for name in
            ['scripts/verify_onepass_observation.py','scripts/verify_scaling_raw.py','src/model_rg/provenance.py']},
        scope='Raw collective identities, exact selection/cohort/source bindings, bytewise replay of all retained native float32 risk reductions with separately reported NumPy float64 discrepancies, online-window equality, independent sampling reconstruction from every used generation logit, and all retained deductive-output summary statistics. The all-position transposed CE has its own reduction order; agreement with exact real arithmetic is not certified by its bytewise replay. Full native forward replay is a separate qualification; it is not claimed for every observation here.'))
    print(case['name'],'observation verification passed',flush=True)


if __name__=='__main__':main()
