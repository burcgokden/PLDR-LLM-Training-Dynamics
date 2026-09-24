#!/usr/bin/env python3
"""Check the scientific companion and numerical dataset after relocation.

Only explicitly manifested payloads are copied. Reads from both original
locations and optional caller-specified roots are blocked in all Python children.
No raw acquisition input or native forward pass is used.
"""
import argparse,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-repo',type=Path,required=True);p.add_argument('--blocked-root',type=Path,action='append',default=[]);a=p.parse_args();data=a.data_repo.resolve();records=[]
    with tempfile.TemporaryDirectory(prefix='pldr-scientific-relocation-') as tmp:
        base=Path(tmp);code=base/'code';evidence=base/'data';guard=base/'guard';guard.mkdir()
        for source,target,manifest in [(ROOT,code,'scientific-manifest.json'),(data,evidence,'manifest.json')]:
            names=list(json.loads((source/manifest).read_text())['files'])+[manifest]
            for name in names:
                dest=target/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,dest)
        blocked=[str(ROOT),str(data),*[str(x.resolve()) for x in a.blocked_root]]
        (guard/'sitecustomize.py').write_text('import os,sys\nB='+repr(blocked)+'\n'
          'def audit(event,args):\n'
          ' if event in {"open","os.listdir","os.scandir","os.chdir"} and args and isinstance(args[0],(str,bytes,os.PathLike)):\n'
          '  p=os.path.abspath(os.fsdecode(args[0]))\n'
          '  if any(p==b or p.startswith(b+os.sep) for b in B): raise PermissionError("Original workspace access blocked")\n'
          'sys.addaudithook(audit)\n')
        env={k:v for k,v in os.environ.items() if k not in {'PYTHONPATH','PLDR_DATA_ROOT','MODEL_RG_DATA_ROOT','PLDR_RG_DATA_ROOT','PLDR_ROW_DATA_ROOT','PLDR_INPUT_MANIFEST'}}
        env.update(PYTHONPATH=str(guard),PLDR_READ_GUARD=str(guard),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        jobs=[('guard',[sys.executable,'-c','from pathlib import Path\ntry: Path('+repr(str(ROOT/'README.md'))+').read_bytes()\nexcept PermissionError: print("passed")\nelse: raise SystemExit(1)']),
          ('code-integrity',[sys.executable,str(code/'scripts/verify_scientific_manifest.py')]),
          ('formal-map',[sys.executable,str(code/'scripts/check_formal_manifest.py')]),
          ('data-integrity',[sys.executable,str(code/'scripts/verify_evidence.py'),'--data-repo',str(evidence)]),
          ('row-cocycle',[sys.executable,str(code/'scripts/run_source.py'),'rg','scripts/run_synthetic_rg.py','--output',str(base/'synthetic.json')]),
          ('consuming-law',[sys.executable,str(code/'scripts/run_source.py'),'model','scripts/check_consuming_bridge.py','--output',str(base/'consuming.json')]),
          ('physical-clock',[sys.executable,str(code/'scripts/run_source.py'),'model','scripts/check_physical_clock.py','--output',str(base/'clock.json')]),
          ('worker-dispatch',[sys.executable,str(code/'scripts/run_source.py'),'model','scripts/run_cache_state_transfer.py','--validate-worker'])]
        for name,cmd in jobs:
            r=subprocess.run(cmd,cwd=base,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
            records.append({'check':name,'returncode':r.returncode,'output':r.stdout.replace(str(base),'<relocated>')})
            print(name,r.returncode,flush=True)
            if r.returncode:print(r.stdout[-3000:]);break
    result={'status':'passed' if len(records)==len(jobs) and all(x['returncode']==0 for x in records) else 'failed','checks':records,'code_payload_sha256':json.loads((ROOT/'scientific-manifest.json').read_text())['payload_sha256'],'data_payload_sha256':json.loads((data/'manifest.json').read_text())['payload_sha256'],'original_reads':'blocked with successful negative control','scope':'Relocated manifest, evidence, formal-map, finite numerical and real worker-dispatch checks; no native acquisition.'}
    out=ROOT/'validation';out.mkdir(exist_ok=True);(out/'scientific-standalone.json').write_text(json.dumps(result,indent=2)+'\n');raise SystemExit(0 if result['status']=='passed' else 1)
