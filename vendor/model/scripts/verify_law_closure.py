#!/usr/bin/env python
"""Independently reconstruct all selected branch sampling, losses and coarse bounds."""
from companion_paths import required_input, acquisition_identity
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(a.root).resolve();study=Path(a.study).resolve();repo=Path(__file__).resolve().parents[1]
    checked={}
    def check(path,expected=None):
        path=Path(path).resolve()
        if str(path) not in checked:checked[str(path)]=sha256(path)
        if expected is not None and checked[str(path)]!=expected:raise AssertionError('Changed input: '+str(path))
        return checked[str(path)]
    def load(path):check(path);return json.loads(Path(path).read_text())
    spec=load(study/'protocol.json');fit=load(study/'frozen-fit.json');innov=load(study/'frozen-innovations.json')
    analysis=load(study/'analysis/results.json');noise=load(study/'analysis/innovations.json')
    assert (analysis['scientific_branches'],analysis['scientific_updates'],len(analysis['records']))==(256,16384,128)
    assert len(spec['cases'])==16 and len(noise['records'])==32
    assert {(c['heads'],c['seed'],c['step']) for c in spec['cases']} == {
        (h,z,t) for h in [4,14] for z in range(640101,640105) for t in [8192,32768]}
    for record in [fit,innov]:
        assert set(record['calibration_inputs'])=={c['name'] for c in spec['cases'] if c['role']=='calibration'}
        for name,digest in record['calibration_inputs'].items():check(study/'runs'/name/'results.json',digest)
    for n,digest in spec['producer_sources'].items():check(repo/n,digest)
    for n,digest in spec['inputs_sha256'].items():check(n,digest)
    for record,script in [(fit,'analyze_law_closure.py'),(analysis,'analyze_law_closure.py'),
                          (innov,'analyze_law_innovations.py'),(noise,'analyze_law_innovations.py')]:
        check(repo/'scripts'/script,record['analyzer_sha256'])
    check(study/'frozen-fit.json',analysis['frozen_fit_sha256'])
    check(study/'frozen-innovations.json',noise['frozen_fit_sha256'])
    for role in ['calibration','validation']:
        launch=load(study/('launcher-'+role+'.json'))
        assert launch['status']=='complete' and len(launch['records'])==8
        for record in launch['records']:check(record['results'],record['sha256'])
    order=np.random.default_rng(641001).permutation(4194304)
    position=np.empty_like(order);position[order]=np.arange(len(order))
    all_q={};logit_panels=0
    for case in spec['cases']:
        folder=study/'runs'/case['name'];r=load(folder/'results.json');meta=load(folder/'manifest.json')
        assert r['case']==case and r['branches']==16 and r['horizons']==[0,1,4,16,64]
        assert r['scientific_updates']==1024 and r['restored_parent_bitwise']
        check(case['parent'],case['parent_sha256']);check(case['parent_verification'],case['parent_verification_sha256'])
        check(folder/'binding.json',meta['binding_sha256']);binding=load(folder/'binding.json')
        for n,d in binding['inputs'].items():check(n,d)
        for n,d in binding['source_files'].items():check(folder/'source'/n,d)
        check(folder/'results.json',meta['results_sha256']);check(folder/'sampling.npz',r['sampling_sha256'])
        with np.load(folder/'sampling.npz') as sampling:
            blocks=sampling['block_ids'];crops=sampling['evaluation_crops']
            assert blocks.shape==(16,64,32) and np.all(position[blocks]>=32*case['step'])
            assert all(len(np.unique(b))==2048 for b in blocks)
            assert sampling['cohort'].tolist()==spec['cohort']
        rows=[]
        for entry in r['records']:
            check(folder/entry['raw'],entry['sha256'])
            assert meta['branch_files'][entry['raw']]==entry['sha256']
            with np.load(folder/entry['raw']) as raw:
                assert raw['losses'].shape==(64,) and np.all(np.isfinite(raw['gradient_norms']))
                np.testing.assert_allclose(raw['q'][:,0],raw['nll'].mean(1),rtol=0,atol=2e-13)
                np.testing.assert_allclose(raw['q'][:,1],raw['entropy'].mean(1),rtol=0,atol=2e-13)
                np.testing.assert_array_equal(raw['q'][:,10:25],raw['layers'].reshape(5,15))
                if entry['branch']==0:
                    assert raw['logits'].shape==(5,32,32000)
                    for k,z in enumerate(raw['logits']):
                        z=z.astype(np.longdouble);maximum=z.max(1,keepdims=True)
                        logp=z-maximum-np.log(np.exp(z-maximum).sum(1,keepdims=True))
                        nll=-logp[np.arange(32),crops[:,64]]
                        np.testing.assert_allclose(nll,raw['nll'][k],rtol=0,atol=2e-12)
                        logit_panels+=1
                rows.append(raw['q'])
        all_q[case['name']]=np.stack(rows)
        assert np.array_equal(all_q[case['name']][:,0],np.broadcast_to(all_q[case['name']][0,0],(16,29)))
        if case['role']=='validation':
            assert datetime.fromisoformat(r['started_at'])>max(datetime.fromisoformat(fit['frozen_at']),datetime.fromisoformat(innov['frozen_at']))
        print('Reconstructed branch inputs and vocabulary risks',case['name'],flush=True)
    # Reconstruct each finite bound as an explicit chronological product/sum.
    for row in analysis['records']:
        m=fit['maps'][f"h{row['heads']}-t{row['step']}-{row['variant']}"]
        q=all_q[row['case']];d=m['dimension'];center=np.array(m['center']);scale=np.array(m['scale'])
        u=(q[:,:,:d]-center)/scale
        h=spec['horizons'].index(row['horizon']);matrices=[np.array(x) for x in m['matrices']]
        offsets=np.array(m['offsets']);bound=np.zeros(16);pred=u[:,0].copy()
        for j in range(h):pred=pred@matrices[j]+offsets[j]
        for j in range(h):
            cost=np.linalg.norm(u[:,j+1]-u[:,j]@matrices[j]-offsets[j],axis=1)
            bound+=cost*np.prod(m['lipschitz'][j+1:h])
        risk=pred[:,0]*scale[0]+center[0];err=risk-q[:,h,0]
        values=dict(mean_risk_error=err.mean(),risk_rmse=np.sqrt(np.mean(err**2)),
                    telescoping_bound_mean=bound.mean(),risk_bound_mean=scale[0]*bound.mean(),
                    risk_wasserstein_empirical=np.abs(err).mean())
        for key,value in values.items():np.testing.assert_allclose(value,row[key],rtol=2e-10,atol=2e-11)
        assert np.max(np.linalg.norm(pred-u[:,h],axis=1)-bound)<1e-9
    for row in noise['records']:
        q=all_q[row['case']][:,:,0];h=spec['horizons'].index(row['horizon'])
        increments=q[:,1:h+1]-q[:,:h];centered=increments-increments.mean(0)
        covariance=centered.T@centered/16
        np.testing.assert_allclose(covariance.sum(),row['actual_risk_variance'],rtol=2e-10,atol=1e-14)
        np.testing.assert_allclose(np.trace(covariance),row['diagonal_increment_variance'],rtol=2e-10,atol=1e-14)
        model=innov['models'][f"h{row['heads']}-t{row['step']}"]
        histories=np.array(model['centered_histories'])[:,:h,0]
        drift=np.array(model['drift'])[:h,0]
        centered_history=histories-histories.mean(0)
        predicted_covariance=centered_history.T@centered_history/len(histories)
        np.testing.assert_allclose(predicted_covariance.sum(),row['history_predicted_variance'],rtol=2e-10,atol=1e-14)
        np.testing.assert_allclose(np.trace(predicted_covariance),row['white_predicted_variance'],rtol=2e-10,atol=1e-14)
        np.testing.assert_allclose(q[0,0]+drift.sum()-q[:,h].mean(),row['mean_risk_error'],rtol=2e-10,atol=1e-13)
    # Strict current qualification is separate from preserved executing sources.
    numerical=load(repo/'docs/law-closure-numerical-final/verification.json')
    assert numerical['status']=='passed' and numerical['tests_passed']==52
    check(repo/'scripts/check_numerical.py',numerical['checker_sha256'])
    for n,d in numerical['tested_sources'].items():check(repo/n,d)
    formal=load(repo/'docs/law-closure-statements/verification.json')
    assert formal['status']=='passed' and formal['exports']==54
    for n,d in {**formal['modules'],**formal['support_sources']}.items():check(repo/n,d)
    check(repo/'scripts/formal/statement-registry.json',formal['registry_sha256'])
    check(repo/'scripts/check_statement_contracts.py',formal['checker_sha256'])
    axioms=load(repo/'docs/law-closure-formal/verification.json')
    assert axioms['status']=='passed' and axioms['explicit_theorems']==54
    for n,d in axioms['module_sources'].items():check(repo/n,d)
    for kind in ['isolation','lifetime','single']:
        record=load(repo/f'docs/law-closure-native-{kind}/evidence/candidate-{kind}.json')
        assert record['status']=='passed'
        for n,d in record['sources'].items():check(repo/n,d)
    math=load(repo/'docs/law-closure-math.json')
    assert (math['status'],math['randomized_inhomogeneous_examples'],math['sharp_examples'])==('passed',120,8)
    check(repo/'scripts/check_law_closure_math.py',math['checker_sha256'])
    mutation=load(repo/'docs/law-closure-statement-mutations/verification.json')
    assert mutation['status']=='passed' and len(mutation['mutations'])==2 and all(x['rejected'] for x in mutation['mutations'])
    check(repo/'scripts/check_law_statement_mutations.py',mutation['checker_sha256'])
    check(repo/'ModelRG/Scaling.lean',mutation['module_sha256'])
    original=root/required_input('verify-law-closure-input-1')
    current=repo/'docs/law-closure-native-single/candidate-single.npz'
    check(original);check(current)
    with np.load(original) as old,np.load(current) as new:
        assert set(old.files)==set(new.files) and len(old.files)==9
        for key in old.files:
            assert old[key].dtype==new[key].dtype and old[key].shape==new[key].shape
            assert old[key].tobytes()==new[key].tobytes()
    for heads in [4,14]:
        record=load(study/f'replay-h{heads}.json')
        assert record['status']=='passed' and record['native_updates']==64 and record['full_state_bitwise']
        for n,d in record['sources'].items():check(repo/n,d)
        for n,d in record['inputs'].items():check(n,d)
    output=dict(status='passed',schema='conditional-law-closure-verification-v1',
        scientific_branches=256,scientific_updates=16384,incoming_states=16,
        heldout_initializations=2,calibration_initializations=2,logit_panels=logit_panels,
        reset_qualification_updates=24,independent_replay_updates=128,
        numerical_tests=52,formal_statements=54,verified_prediction_cells=128,innovation_cells=32,
        verifier_sha256=sha256(__file__),checked_sha256=checked,
        scope='Independent saved-array reconstruction of every selected branch and prediction cell; '
            '80 full-vocabulary horizon panels; two full 64-update native replays at the producing '
            'GPU/precision. Four training initializations, two held out, conditional on one corpus '
            'and consumed-set law. No thermodynamic inference or uniform native closure certificate.')
    if Path(a.output).exists():raise FileExistsError(a.output)
    write_json(a.output,output);print('Complete conditional-law evidence passed',flush=True)


if __name__=='__main__':main()
