#!/usr/bin/env python3
"""Queue bounded mechanistic follow-ups after both initial GPU workers finish."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
from pathlib import Path
import json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(legacy_path('/pldr-data/model/potential-avalanche-20260913'))
REPO=Path(__file__).resolve().parents[1]

def worker(device,cases):
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    for script,name,directory in cases:
        if (ROOT/directory/name/'manifest.json').exists():continue
        with (ROOT/'logs'/f'{directory}-{name}.log').open('w') as log:
            r=subprocess.run([sys.executable,str(REPO/'scripts'/script),'--case',name,'--device',f'cuda:{device}'],
                cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
        print(script,name,r.returncode,flush=True)
        if r.returncode:raise RuntimeError(name+' failed; see retained log')

if __name__=='__main__':
    spec=json.loads((ROOT/'protocols/training.json').read_text())
    while not all((ROOT/'runs'/j['run_id']/'manifest.json').exists() for j in spec['jobs']):time.sleep(10)
    for j in spec['jobs']:
        if json.loads((ROOT/'runs'/j['run_id']/'manifest.json').read_text())['status']!='complete':raise ValueError('Initial path failed')
    paths=[('continue_potential_avalanches.py',j['name'],'continuations') for j in json.loads((ROOT/'protocols/continuations.json').read_text())['cases']]
    paths += [('measure_potential_relaxation.py',j['name'],'relaxation') for j in json.loads((ROOT/'protocols/relaxation.json').read_text())['cases']]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(worker,i,paths[i::2]) for i in range(2)]
        for f in futures:f.result()
