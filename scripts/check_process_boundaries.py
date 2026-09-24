#!/usr/bin/env python3
"""Real bounded child dispatch and row input parsing from a relocated checkout."""
from concurrent.futures import ThreadPoolExecutor
import hashlib,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PARENTS=['run_cache_state_transfer','run_critical_onepass','run_row_path_confirmation','run_matched_clock','run_operator_cache_study','run_potential_avalanches','run_matched_onepass','run_potential_factorial','run_cache_risk_study','run_equal_time_flux','run_collective_row_clock','run_collective_row_quadrature','run_critical_shared_paths','context_categorical_study']
ROWS=['stage_causal_row_map_confirmation','stage_contrast_energy_confirmation','stage_comprehensive_gate_shape_confirmation','stage_row_map_confirmation']

def main():
    records=[]
    with tempfile.TemporaryDirectory(prefix='pldr-process-boundaries-') as tmp:
        base=Path(tmp);code=base/'nested/code';outside=base/'unrelated';outside.mkdir()
        shutil.copytree(ROOT,code,ignore=shutil.ignore_patterns('.git','.lake','build','dist','__pycache__','.pytest_cache','monograph.pdf'))
        guard=base/'guard';guard.mkdir()
        blocked=[str(ROOT),'/pldr-data','/pldr-assets/refinedweb']
        (guard/'sitecustomize.py').write_text('import os,sys\nB='+repr(blocked)+'\n'
          'def audit(event,args):\n'
          ' if event in {"open","os.listdir","os.scandir","os.chdir"} and args and isinstance(args[0],(str,bytes,os.PathLike)):\n'
          '  p=os.path.abspath(os.fsdecode(args[0]))\n'
          '  if any(p==b or p.startswith(b+os.sep) for b in B): raise PermissionError("Blocked original workspace: "+p)\n'
          'sys.addaudithook(audit)\n'
          'with open(os.environ["PLDR_GUARD_LOG"],"a") as f: f.write(str(os.getpid())+"\\n")\n')
        jobs=[('model',n,['--validate-worker'],g) for n in PARENTS for g in [False,True]]
        jobs += [('row',n,option,True) for n in ROWS for option in [['--help'],['--dry-run'],['--dry-run','--tokens',str(base/'explicit-tokens.bin')],['--tokens',str(base/'missing-required-tokens.bin')]]]
        def run(job):
            family,name,args,guarded=job
            env={k:v for k,v in os.environ.items() if k not in ['PYTHONPATH','PLDR_READ_GUARD','PLDR_DATA_ROOT','MODEL_RG_DATA_ROOT','PLDR_ROW_DATA_ROOT','PLDR_RG_DATA_ROOT','PLDR_INPUT_MANIFEST']}
            env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PLDR_DATA_ROOT=str(base/'missing-data'))
            guard_log=base/('guard-'+name+'-'+str(len(args))+'-'+args[0].replace('--','')+'.jsonl')
            env['PLDR_ROW_TOKENS']=str(base/'environment-tokens.bin')
            if guarded:env.update(PYTHONPATH=str(guard),PLDR_READ_GUARD=str(guard),PLDR_GUARD_LOG=str(guard_log))
            cmd=[sys.executable,'-B',str(code/'scripts/run_source.py'),family,'scripts/'+name+'.py',*args]
            start=time.monotonic();r=subprocess.run(cmd,cwd=outside,env=env,capture_output=True,text=True,timeout=65)
            output=r.stdout+r.stderr
            expected='"status": "passed"' if args==['--validate-worker'] else '"status": "dry-run"' if args[0]=='--dry-run' else 'Missing tokens' if args[0]=='--tokens' else 'usage:'
            expected_code=2 if args[0]=='--tokens' else 0
            guard_processes=len(set(guard_log.read_text().splitlines())) if guarded and guard_log.exists() else 0
            good=r.returncode==expected_code and expected in output
            if guarded:good=good and guard_processes >= (3 if args==['--validate-worker'] else 2)
            if args[0]=='--dry-run' and len(args)>1:good=good and str(base/'explicit-tokens.bin') in output and str(base/'environment-tokens.bin') not in output
            return dict(family=family,program=name,arguments=args,guarded=guarded,guard_processes=guard_processes,seconds=time.monotonic()-start,returncode=r.returncode,status='passed' if good else 'failed',output=output.replace(str(base),'<relocated>'))
        with ThreadPoolExecutor(max_workers=2) as pool:
            for rec in pool.map(run,jobs):
                records.append(rec);print(rec['program'],rec['arguments'],rec['guarded'],rec['status'],flush=True)
        unexpected=[str(p.relative_to(code)) for p in code.rglob('*') if p.name in ['__pycache__','logs','runs'] and p.is_dir() and str(p.relative_to(code)).startswith('build/')]
        if (base/'missing-data').exists():raise ValueError('Bounded route created an experiment directory')
    report=dict(status='passed' if all(x['status']=='passed' for x in records) else 'failed',checks=records,worker_parents=len(PARENTS),worker_dispatches=2*len(PARENTS),row_entry_points=len(ROWS),scientific_updates=0,acquisition_calls=0,scope='Real production dispatcher to child imports/help; no native acquisition or complete raw replay claim.')
    (ROOT/'validation/process-boundaries.json').write_text(json.dumps(report,indent=2)+'\n')
    if report['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
