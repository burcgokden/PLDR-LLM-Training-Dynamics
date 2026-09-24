"""Reproduce the executed readout study and its current execution qualification."""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import os

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))


def main(base):
    base=Path(base).resolve()
    if not base.is_relative_to(ROOT):raise ValueError('Authorized experiment destination required')
    base.mkdir(exist_ok=False)
    def run(log,script,*args):
        command=[sys.executable,str(REPO/'scripts'/script),*map(str,args)]
        with (base/log).open('x') as stream:
            subprocess.run(command,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True,
                           env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',PYTHONPATH=child_pythonpath("model")))
    fine=base/'finetuning';fine.mkdir();(fine/'data').symlink_to(ROOT/'finetuning-sectors-20260912/data',target_is_directory=True)
    run('fine-prepare.log','finetuning_study.py','prepare','--study',fine/'qualification','--stage','qualification')
    run('fine-run.log','finetuning_study.py','run','--study',fine/'qualification')
    run('fine-verify.log','verify_finetuning.py','--study',fine/'qualification','--output',fine/'qualification/verification.json')
    run('fine-baseline.log','finetuning_study.py','prepare','--study',fine/'admission-baseline','--stage','assessment',
        '--qualification',fine/'qualification/verification.json')
    run('fine-admit.log','check_primary_design_admission.py','--study',fine/'admission-baseline',
        '--qualification',fine/'qualification','--output',fine/'entry-checks')
    physical=base/'physical';physical.mkdir()
    for name in ['training-data','development-data']:(physical/name).symlink_to(ROOT/'known-universality-20260912'/name,target_is_directory=True)
    def parity(item):
        h,i=item;run(f'physical-parity-h{h}.log','qualify_physical_native.py','--study',physical,'--heads',h,'--device',f'cuda:{i}')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(parity,[(4,0),(8,1)]))
    run('physical-prepare.log','physical_study.py','prepare','--study',physical,'--stage','qualification','--steps',12)
    def physical_worker(item):
        h,i=item;run(f'physical-worker-h{h}.log','run_physical_queue.py','--study',physical/'qualification','--heads',h,'--device',f'cuda:{i}')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(physical_worker,[(4,0),(8,1)]))
    run('physical-verify.log','verify_physical_execution.py','--study',physical/'qualification','--output',physical/'qualification/verification.json')
    run('physical-baseline.log','physical_study.py','prepare','--study',physical,'--stage','confirmation','--steps',16384)
    run('physical-admit.log','check_physical_admission.py','--study',physical/'confirmation','--output',physical/'entry-checks')
    # These complete-length protocols are admission fixtures; no workers are launched.
    fresh=base/'fresh-readout'
    def profile(item):
        h,i=item;run(f'profile-h{h}.log','fresh_readout_study.py','profile','--study',fresh,'--heads',h,'--device',f'cuda:{i}')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(profile,[(4,0),(8,1)]))
    run('fresh-prepare.log','fresh_readout_study.py','prepare','--study',fresh)
    def observe(item):
        h,i=item;run(f'fresh-h{h}.log','fresh_readout_study.py','worker','--study',fresh,'--heads',h,'--device',f'cuda:{i}')
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(observe,[(4,0),(8,1)]))
    run('fresh-analyze.log','analyze_fresh_readout.py','--study',fresh,'--output',base/'fresh-analysis')
    run('fresh-verify.log','verify_fresh_readout.py','--study',fresh,'--analysis',base/'fresh-analysis','--output',base/'fresh-analysis/verification.json')
    run('readout.log','readout_budget.py','--study',ROOT/'known-universality-20260912','--output',base/'readout')
    run('reference-corrections.log','analyze_reference_corrections.py','--study',ROOT/'known-universality-20260912','--output',base/'reference-corrections')
    print(base,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
