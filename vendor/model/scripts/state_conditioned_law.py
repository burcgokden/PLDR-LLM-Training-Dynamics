#!/usr/bin/env python
"""Fresh split calibration and validation of complete-state conditional risk paths."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from model_rg.provenance import sha256, write_json


def paths(study,case):
    folder=study/'runs'/case['name'];result=json.loads((folder/'results.json').read_text())
    values=[]
    for entry in result['records']:
        path=folder/entry['raw']
        if sha256(path)!=entry['sha256']:raise AssertionError('Changed native branch')
        with np.load(path) as raw:values.append(raw['nll'].mean(1))
    return np.stack(values),result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--phase',choices=['prepare','calibrate','freeze','validate','analyze'],required=True)
    a=p.parse_args();study=Path(a.study).resolve();repo=Path(__file__).resolve().parents[1]
    preparation=study/'state-law-design.json'
    if a.phase=='prepare':
        if preparation.exists():raise FileExistsError(preparation)
        original=json.loads((study/'protocol.json').read_text())
        cases=[c for c in original['cases'] if c['role']=='validation']
        fits={};inputs={str(study/'protocol.json'):sha256(study/'protocol.json')}
        for case in cases:
            risk,result=paths(study,case)
            fits[case['name']]=dict(center=risk[:,1:].mean(0).tolist(),
                scale=np.maximum(risk[:,1:].std(axis=0,ddof=1),.01).tolist(),
                incoming_risk=float(risk[0,0]),development_branches=16,
                native_incoming_state_sha256=result['initial_state_sha256'])
            inputs[str(study/'runs'/case['name']/'results.json')]=sha256(study/'runs'/case['name']/'results.json')
        for name,branches,seed in [('state-law-calibration',8,973109),('state-law-validation',16,974113)]:
            out=study/name;out.mkdir(exist_ok=False)
            spec=dict(original);spec.update(cases=cases,branches=branches,branch_seed=seed,
                frozen_at=datetime.now(timezone.utc).isoformat(),stage=name,
                producer_sources={**original['producer_sources'],'scripts/state_conditioned_law.py':sha256(__file__)})
            spec['inputs_sha256']={**original['inputs_sha256'],**inputs}
            spec['analysis']=dict(conditioning='Complete incoming native model, Adam state, drive and remaining corpus.',
                fits=fits,score='Maximum absolute standardized external-target risk error over horizons 1,4,16,64.',
                calibration='Eight fresh branches, independent of all development paths. Threshold is their maximum score.',
                validation='Sixteen additional fresh branches per incoming state. A branch is covered only if all four risks lie inside the frozen tube.',
                guarantee='At least 8/9 marginal coverage for one new full risk path, conditional on incoming state and development fit, averaged over fresh calibration and validation draws. Not conditional-on-calibration or simultaneous-across-validation coverage.',
                no_selection='Every one of the eight incoming states and sixteen validation branches is retained.')
            write_json(out/'protocol.json',spec)
        write_json(preparation,dict(status='frozen',schema='state-conditioned-risk-path-design-v1',
            frozen_at=datetime.now(timezone.utc).isoformat(),producer_sha256=sha256(__file__),fits=fits,
            inputs_sha256=inputs,calibration_branches_per_state=8,validation_branches_per_state=16,
            cases=cases,source_protocol_sha256=sha256(study/'protocol.json'),
            scope='State-dependent stochastic kernels with finite conditional memory. New calibration is independent of the development sample and all new validation outcomes. No cross-initialization coefficient transfer is assumed.'))
        print('Frozen complete-state conditional design with independent fresh calibration',flush=True)
        return
    design=json.loads(preparation.read_text())
    if design['producer_sha256']!=sha256(__file__):raise AssertionError('Frozen state-law producer changed')
    for name,digest in design['inputs_sha256'].items():
        if sha256(name)!=digest:raise AssertionError('Changed development input')
    if a.phase in ['calibrate','validate']:
        name='state-law-calibration' if a.phase=='calibrate' else 'state-law-validation'
        selected=study/name;protocol=json.loads((selected/'protocol.json').read_text())
        if (selected/'launcher.json').exists():raise FileExistsError(selected/'launcher.json')
        if a.phase=='validate':
            frozen=json.loads((study/'state-law-tubes.json').read_text())
            if frozen['status']!='frozen' or frozen['producer_sha256']!=sha256(__file__):
                raise AssertionError('Freeze all risk tubes before new validation')
        def lane(heads,device):
            rows=[]
            for case in design['cases']:
                if case['heads']!=heads:continue
                command=[sys.executable,str(repo/'scripts/measure_law_closure.py'),'--root',a.root,
                    '--study',str(selected),'--case',case['name'],'--device',device]
                env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
                with (selected/(case['name']+'.log')).open('x') as stream:
                    subprocess.run(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
                path=selected/'runs'/case['name']/'results.json';r=json.loads(path.read_text())
                assert r['status']=='complete' and r['branches']==protocol['branches']
                rows.append(dict(case=case['name'],results=str(path),sha256=sha256(path),seconds=r['seconds']))
                print('Completed',name,case['name'],'seconds',round(r['seconds'],1),flush=True)
            return rows
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(lane,4,'cuda:0'),pool.submit(lane,14,'cuda:1')]
            records=[row for future in futures for row in future.result()]
        write_json(selected/'launcher.json',dict(status='complete',records=records,
            scientific_updates=8*protocol['branches']*64,producer_sha256=sha256(__file__)))
        return
    if a.phase=='freeze':
        out=study/'state-law-tubes.json'
        if out.exists():raise FileExistsError(out)
        if (study/'state-law-validation/runs').exists():raise AssertionError('Validation has already started')
        completed=json.loads((study/'state-law-calibration/launcher.json').read_text())
        assert completed['status']=='complete' and len(completed['records'])==8
        tubes={};inputs={}
        for case in design['cases']:
            risk,result=paths(study/'state-law-calibration',case);f=design['fits'][case['name']]
            center=np.array(f['center']);scale=np.array(f['scale'])
            scores=np.max(np.abs(risk[:,1:]-center)/scale,axis=1)
            radius=float(scores.max());tubes[case['name']]=dict(center=center.tolist(),scale=scale.tolist(),
                calibration_scores=scores.tolist(),radius=radius,lower=(center-radius*scale).tolist(),
                upper=(center+radius*scale).tolist())
            path=study/'state-law-calibration/runs'/case['name']/'results.json';inputs[str(path)]=sha256(path)
        write_json(out,dict(status='frozen',schema='complete-state-risk-tubes-v1',
            frozen_at=datetime.now(timezone.utc).isoformat(),producer_sha256=sha256(__file__),
            design_sha256=sha256(preparation),calibration_inputs=inputs,tubes=tubes,
            marginal_path_coverage_lower_bound='8/9'))
        print('Frozen all eight risk tubes before validation',flush=True);return
    tubes=json.loads((study/'state-law-tubes.json').read_text());rows=[]
    completed=json.loads((study/'state-law-validation/launcher.json').read_text())
    assert completed['status']=='complete' and len(completed['records'])==8
    for case in design['cases']:
        risk,result=paths(study/'state-law-validation',case);tube=tubes['tubes'][case['name']]
        assert datetime.fromisoformat(result['started_at'])>datetime.fromisoformat(tubes['frozen_at'])
        score=np.max(np.abs(risk[:,1:]-np.array(tube['center']))/np.array(tube['scale']),axis=1)
        covered=score<=tube['radius']
        increment=np.diff(risk,axis=1);centered=increment-increment.mean(0)
        covariance=centered.T@centered/len(risk)
        variance=float(risk[:,-1].var(ddof=0));diagonal=float(np.trace(covariance))
        rows.append(dict(case=case['name'],heads=case['heads'],step=case['step'],seed=case['seed'],
            covered=int(covered.sum()),branches=16,coverage=float(covered.mean()),scores=score.tolist(),
            tube=tube,mean_risks=risk[:,1:].mean(0).tolist(),
            mean_risk_errors=(np.array(tube['center'])-risk[:,1:].mean(0)).tolist(),
            interval_covariance=covariance.tolist(),risk_variance=variance,
            diagonal_increment_variance=diagonal,cross_interval_variance=variance-diagonal,
            cross_fraction=(variance-diagonal)/variance if variance>0 else None))
    out=study/'state-law-results.json'
    if out.exists():raise FileExistsError(out)
    write_json(out,dict(status='complete',schema='state-conditioned-risk-path-validation-v1',records=rows,
        covered_branches=sum(r['covered'] for r in rows),validation_branches=128,
        fresh_calibration_branches=64,new_native_updates=12288,horizons=[1,4,16,64],
        producer_sha256=sha256(__file__),tubes_sha256=sha256(study/'state-law-tubes.json'),
        scope='Full risk-path coverage over four nested horizons; all selected incoming states and branches. '
            'The exchangeability coverage statement averages over calibration as well as the next validation '
            'branch, conditional on complete incoming state and development fit. It is not a population '
            'guarantee conditional on the realized calibration sample, nor across all validation branches.'))
    print('Covered',sum(r['covered'] for r in rows),'of 128 fresh complete risk paths',flush=True)


if __name__=='__main__':main()
