#!/usr/bin/env python
"""Resume only the unexecuted full data-law comparison after observer recovery."""
from companion_paths import child_pythonpath
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.provenance import sha256, write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--recovery',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path(a.root).resolve();study=root/a.study
    recovery=Path(a.recovery).resolve();marker=study/'launcher-repetition-analysis.json'
    prior=recovery/'original-launcher-repetition-analysis.json'
    if sha256(prior)!=sha256(marker):raise AssertionError('The preserved comparison controller changed')
    state=json.loads(marker.read_text())
    if state['status']!='waiting' or [r['panel'] for r in state['records']]!=['reference_schedule','constant']:
        raise AssertionError('Only the complete comparison remains eligible for recovery')
    for row in state['records']:
        proof=study/'verification'/('repetition-'+row['panel']+'.json')
        if row['status']!='complete' or sha256(proof)!=row['verification_sha256']:
            raise AssertionError('A complete comparison changed')
    selection=study/'protocols/repetition-analysis-selection.json';spec=json.loads(selection.read_text())
    if state['selection_sha256']!=sha256(selection):raise AssertionError('Comparison selection changed')
    sources={n:sha256(repo/n) for n in ['scripts/resume_onepass_repetition_analysis.py',
        'scripts/analyze_repetition_comparison.py','scripts/verify_repetition_comparison.py',
        'scripts/run_repetition_analysis.py','src/model_rg/provenance.py']}
    protocol=recovery/'repetition-resumption.json'
    if protocol.exists():raise FileExistsError(protocol)
    write_json(protocol,dict(schema='onepass-repetition-resumption-v1',
        at=datetime.now(timezone.utc).isoformat(),sources=sources,
        original_controller=str(prior),original_controller_sha256=sha256(prior),
        selection_sha256=sha256(selection),retained_panels=['reference_schedule','constant'],
        resumed_panel='complete',scope='Resume the existing complete 40-pair comparison after the observer stopped. No selected pair, outcome, estimator, source or native update changes.'))
    state['resumption']=dict(protocol=str(protocol),protocol_sha256=sha256(protocol))
    write_json(marker,state);os.nice(8)
    required={q[side]['verification'] for q in spec['pairs'] for side in ['old','new']}
    required.update(q['new']['native_verification'] for q in spec['pairs'])
    required.update(q[side]['risk_verification'] for q in spec['pairs'] for side in ['old','new'] if q[side].get('risk_verification'))
    required.add(str(study/'launcher-onepass-observations.json'))
    while True:
        ready=True
        for name in required:
            path=Path(name)
            if not path.exists():ready=False;continue
            status=json.loads(path.read_text())['status']
            if 'fail' in status:
                state['status']='blocked_on_observation_failure';write_json(marker,state)
                raise RuntimeError('A further observation failure is preserved: '+name)
            if status not in ['passed','complete']:ready=False
        if ready:break
        time.sleep(30)
    for name,digest in sources.items():
        if sha256(repo/name)!=digest:raise AssertionError('The selected comparison implementation changed')
    if sha256(selection)!=state['selection_sha256']:raise AssertionError('The selected comparison changed')
    state['status']='analyzing_complete';write_json(marker,state)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4')
    row=dict(panel='complete',commands=[]);before=time.time()
    for script in ['analyze_repetition_comparison.py','verify_repetition_comparison.py']:
        command=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--study',a.study,'--panel','complete']
        log=study/'logs/repetition-analysis'/('complete-'+script+'.log')
        with log.open('x') as output:r=subprocess.run(command,cwd=repo,env=env,stdout=output,stderr=subprocess.STDOUT)
        row['commands'].append(dict(command=command,returncode=r.returncode,log_sha256=sha256(log)))
        if r.returncode:
            row['status']='failed';state['records'].append(row);state['status']='analysis_failed';write_json(marker,state)
            raise RuntimeError('The complete comparison failure was preserved')
    proof=study/'verification/repetition-complete.json'
    if json.loads(proof.read_text())['status']!='passed':raise AssertionError('The complete comparison did not verify')
    row.update(status='complete',seconds=time.time()-before,verification_sha256=sha256(proof))
    state['records'].append(row);state['status']='complete';write_json(marker,state)
    print('The complete original data-law comparison is independently verified',flush=True)


if __name__=='__main__':main()
