#!/usr/bin/env python
"""Independent reconstruction of fresh conditional path calibration and coverage."""
import argparse
from datetime import datetime
from itertools import permutations, product
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    study=Path(a.study).resolve();repo=Path(__file__).resolve().parents[1];checked={}
    def check(path,expected=None):
        path=Path(path).resolve()
        if str(path) not in checked:checked[str(path)]=sha256(path)
        if expected is not None and checked[str(path)]!=expected:raise AssertionError('Changed conditional evidence: '+str(path))
        return checked[str(path)]
    def load(path):check(path);return json.loads(Path(path).read_text())
    design=load(study/'state-law-design.json');tubes=load(study/'state-law-tubes.json');summary=load(study/'state-law-results.json')
    for record in [design,tubes,summary]:check(repo/'scripts/state_conditioned_law.py',record['producer_sha256'])
    check(study/'state-law-design.json',tubes['design_sha256']);check(study/'state-law-tubes.json',summary['tubes_sha256'])
    for path,digest in design['inputs_sha256'].items():check(path,digest)
    for path,digest in tubes['calibration_inputs'].items():check(path,digest)
    expected={(h,z,t) for h in [4,14] for z in [640103,640104] for t in [8192,32768]}
    assert {(c['heads'],c['seed'],c['step']) for c in design['cases']}==expected
    actual={};raw_panels=0
    order=np.random.default_rng(641001).permutation(4194304);position=np.empty_like(order);position[order]=np.arange(len(order))
    for stage,count,seed in [('state-law-calibration',8,973109),('state-law-validation',16,974113)]:
        selection=load(study/stage/'protocol.json');launch=load(study/stage/'launcher.json')
        assert launch['status']=='complete' and len(launch['records'])==8
        assert selection['branches']==count and selection['branch_seed']==seed
        for name,digest in selection['producer_sources'].items():check(repo/name,digest)
        for name,digest in selection['inputs_sha256'].items():check(name,digest)
        for case in design['cases']:
            folder=study/stage/'runs'/case['name'];r=load(folder/'results.json');meta=load(folder/'manifest.json')
            assert r['case']==case and r['status']=='complete' and r['branches']==count and r['scientific_updates']==64*count
            check(folder/'results.json',meta['results_sha256']);check(folder/'sampling.npz',meta['sampling_sha256'])
            check(folder/'binding.json',meta['binding_sha256']);binding=load(folder/'binding.json')
            for name,digest in binding['inputs'].items():check(name,digest)
            for name,digest in binding['source_files'].items():check(folder/'source'/name,digest)
            if stage=='state-law-validation':assert datetime.fromisoformat(r['started_at'])>datetime.fromisoformat(tubes['frozen_at'])
            else:assert datetime.fromisoformat(r['started_at'])>datetime.fromisoformat(design['frozen_at'])
            assert r['initial_state_sha256']==design['fits'][case['name']]['native_incoming_state_sha256']
            with np.load(folder/'sampling.npz') as sampling:
                blocks=sampling['block_ids'];crops=sampling['evaluation_crops']
                assert blocks.shape==(count,64,32) and np.all(position[blocks]>=32*case['step'])
                assert all(len(np.unique(path))==2048 for path in blocks)
                remaining=order[32*case['step']:]
                rng=np.random.default_rng(seed+case['seed']*37+case['heads']*101+case['step'])
                reconstruction=np.stack([rng.choice(remaining,2048,replace=False).reshape(64,32) for _ in range(count)])
                np.testing.assert_array_equal(blocks,reconstruction)
            rows=[]
            for entry in r['records']:
                path=folder/entry['raw'];check(path,entry['sha256'])
                assert meta['branch_files'][entry['raw']]==entry['sha256']
                with np.load(path) as raw:
                    rows.append(np.mean(raw['nll'],axis=1))
                    np.testing.assert_allclose(rows[-1],raw['q'][:,0],rtol=0,atol=3e-13)
                    if entry['branch']==0:
                        for k,z in enumerate(raw['logits']):
                            z=z.astype(np.longdouble);maximum=z.max(1)
                            loss=maximum+np.log(np.exp(z-maximum[:,None]).sum(1))-z[np.arange(32),crops[:,64]]
                            np.testing.assert_allclose(loss,raw['nll'][k],rtol=0,atol=3e-12);raw_panels+=1
            actual[stage,case['name']]=np.array(rows)
            print('Verified fresh conditional paths',stage,case['name'],flush=True)
    records=[]
    for case in design['cases']:
        f=design['fits'][case['name']];center=np.array(f['center']);scale=np.array(f['scale'])
        # The development fit is reconstructed independently from its existing paths.
        base=study/'runs'/case['name'];original=load(base/'results.json');risk=[]
        for entry in original['records']:
            path=base/entry['raw'];check(path,entry['sha256'])
            with np.load(path) as raw:risk.append(raw['nll'].mean(1)[1:])
        risk=np.array(risk);np.testing.assert_array_equal(center,risk.mean(0))
        np.testing.assert_array_equal(scale,np.maximum(risk.std(0,ddof=1),.01))
        calibration=actual['state-law-calibration',case['name']][:,1:]
        threshold=max(max(abs((path-center)/scale)) for path in calibration)
        tube=tubes['tubes'][case['name']];assert threshold==tube['radius']
        lower=center-threshold*scale;upper=center+threshold*scale
        np.testing.assert_array_equal(lower,tube['lower']);np.testing.assert_array_equal(upper,tube['upper'])
        validation=actual['state-law-validation',case['name']]
        inside=np.all((validation[:,1:]>=lower)&(validation[:,1:]<=upper),axis=1)
        row=next(r for r in summary['records'] if r['case']==case['name'])
        assert int(inside.sum())==row['covered'] and len(inside)==row['branches']==16
        differences=np.diff(validation,axis=1);differences-=differences.mean(0)
        covariance=differences.T@differences/16
        np.testing.assert_allclose(covariance,row['interval_covariance'],rtol=2e-12,atol=1e-14)
        np.testing.assert_allclose(covariance.sum(),row['risk_variance'],rtol=2e-12,atol=1e-14)
        records.append(dict(case=case['name'],covered=int(inside.sum()),branches=16))
    assert sum(r['covered'] for r in records)==summary['covered_branches']
    # Exhaustive finite checks of the rank proof, including tied scores.
    distinct_counts=np.zeros(9,dtype=np.int64)
    for scores in permutations(range(9)):distinct_counts[scores.index(8)]+=1
    assert np.all(distinct_counts==40320)
    tied_counts=np.zeros(9,dtype=np.int64);tied_cases=0
    for scores in product(range(3),repeat=9):
        events=[all(scores[i]>scores[j] for j in range(9) if j!=i) for i in range(9)]
        assert sum(events)<=1;tied_counts+=events;tied_cases+=1
    assert np.all(tied_counts==tied_counts[0]) and tied_counts.sum()<=tied_cases
    output=dict(status='passed',schema='state-conditioned-law-verification-v1',records=records,
        coverage_count=sum(r['covered'] for r in records),fresh_calibration_branches=64,validation_branches=128,
        native_updates=12288,incoming_states=8,new_full_vocabulary_panels=raw_panels,
        exact_rank_checks=dict(distinct_permutations=362880,distinct_failures_per_coordinate=40320,
            tied_tuples=tied_cases,tied_failures_per_coordinate=tied_counts.tolist()),
        verifier_sha256=sha256(__file__),checked_sha256=checked,
        scope='Every selected new branch, sampling path and risk-path coverage decision; independent development fit and calibration reconstruction; 80 full-vocabulary panels; complete finite rank counting. No extra native replay or population-conditional-on-calibration claim.')
    if Path(a.output).exists():raise FileExistsError(a.output)
    write_json(a.output,output);print('Passed complete-state conditional path verification',flush=True)


if __name__=='__main__':main()
