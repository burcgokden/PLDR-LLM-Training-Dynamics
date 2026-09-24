#!/usr/bin/env python
"""Schedule completed-study measurements with a per-process execution ledger."""
from companion_paths import child_pythonpath
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import time
from model_rg.controlled import device_name
from model_rg.provenance import write_json, source_manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--devices',default='0,1')
    ap.add_argument('--parts',default='long,quality,drift');ap.add_argument('--label',default='measurements')
    ap.add_argument('--heads',default='2,4,8,14');ap.add_argument('--seeds',default='630101,630102,630103');ap.add_argument('--family-prefix',default='family')
    a=ap.parse_args();root=Path(a.root);study=root/'controlled-study-20260905';repo=Path(__file__).resolve().parents[1]
    jobs=[];family=[f'{a.family_prefix}-h{h}-s{s}' for h in map(int,a.heads.split(',')) for s in map(int,a.seeds.split(','))]
    for part in a.parts.split(','):
        if part=='long':
            jobs += [(f'long-soc{i}','collect_long_segments.py',['--checkpoint',str(i)]) for i in [1,2]]
        elif part=='quality':
            jobs += [('quality-'+r,'measure_quality_controls.py',['--run-id',r]) for r in family+(['soc1','soc2'] if a.family_prefix=='family' else [])]
        elif part=='drift':
            jobs += [('drift-smooth-'+r,'measure_conditional_drift.py',['--run-id',r]) for r in family]
        else:raise ValueError(part)
    devices=[device_name(d) for d in a.devices.split(',')];ledger=study/f'launcher-{a.label}.json'
    if ledger.exists() or any((study/'runs'/name).exists() for name,_,_ in jobs):raise FileExistsError('Existing measurement destination')
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='3')
    launcher_sources=source_manifest()
    write_json(ledger,dict(status='running',arguments=vars(a),jobs=jobs,source_files=launcher_sources))
    def worker(index):
        records=[]
        for name,script,extra in jobs[index::len(devices)]:
            cmd=[sys.executable,str(repo/'scripts'/script),'--root',str(root),'--device',devices[index],*extra]
            start=time.time();print('Starting',name,flush=True)
            with (study/'logs'/(name+'.log')).open('x') as f:
                r=subprocess.run(cmd,cwd=repo,env=env,stdout=f,stderr=subprocess.STDOUT)
            record=dict(run_id=name,command=cmd,returncode=r.returncode,seconds=time.time()-start)
            write_json(study/('ledger-'+name+'.json'),record);records.append(record)
            print('Finished',name,r.returncode,flush=True)
            if r.returncode:break
        return records
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:records=sum(list(pool.map(worker,range(len(devices)))),[])
    ok=len(records)==len(jobs) and all(r['returncode']==0 for r in records)
    write_json(ledger,dict(status='complete' if ok else 'failed',arguments=vars(a),source_files=launcher_sources,records=records))
    if not ok:raise RuntimeError('Incomplete measurement set; retained ledger and logs')


if __name__=='__main__':main()
