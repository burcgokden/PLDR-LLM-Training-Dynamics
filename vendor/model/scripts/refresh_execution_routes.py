#!/usr/bin/env python3
"""Fresh whole-package fine-tuning qualification and final executable admission.

The broad package policy is deliberate: every src/model_rg Python file belongs
to the primary inventory. A package change requires actual requalification.
"""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))
DEFAULT=ROOT/'execution-current-20260913'


def run(base,label,script,*args):
    cmd=[sys.executable,str(REPO/'scripts'/script),*map(str,args)];start=time.perf_counter()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=child_pythonpath("model"),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    with (base/(label+'.log')).open('x') as log:subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    record=dict(label=label,command=cmd,seconds=time.perf_counter()-start,log_sha256=sha256(base/(label+'.log')))
    write_json(base/(label+'-command.json'),record);print(label,round(record['seconds'],1),flush=True);return record


def qualify(base):
    base.mkdir(parents=True,exist_ok=False);fine=base/'finetuning';fine.mkdir()
    (fine/'data').symlink_to(ROOT/'finetuning-sectors-20260912/data',target_is_directory=True)
    run(base,'primary-prepare','finetuning_study.py','prepare','--study',fine/'qualification','--stage','qualification')
    run(base,'primary-native','finetuning_study.py','run','--study',fine/'qualification')
    run(base,'primary-verify','verify_finetuning.py','--study',fine/'qualification','--output',fine/'qualification/verification.json')
    run(base,'primary-baseline','finetuning_study.py','prepare','--study',fine/'admission-baseline','--stage','assessment','--qualification',fine/'qualification/verification.json')


def admit(base,out,factorial_study=None):
    factorial_study=Path(factorial_study) if factorial_study else ROOT/'potential-factorial-disjoint-20260914'
    out.mkdir(parents=True,exist_ok=False);fine=base/'finetuning'
    definitions=[
        ('primary','check_primary_design_admission.py',['--study',fine/'admission-baseline','--qualification',fine/'qualification'],fine/'qualification/verification.json','Six declared scripts and every package module'),
        ('physical','check_physical_admission.py',['--study',ROOT/'readout-execution-20260913/physical/confirmation'],ROOT/'readout-execution-20260913/physical/qualification/verification.json','Explicit physical source inventory and native assets'),
        ('continuations','check_execution_admission.py',['--qualification-root',ROOT/'execution-qualification-20260912/final'],ROOT/'execution-qualification-20260912/final','Recursive local-source resolution and native assets'),
        ('optimizer','check_optimizer_admission.py',['--study',ROOT/'optimizer-transport-20260912/assessment'],ROOT/'optimizer-transport-20260912/native-qualification/verification.json','Explicit optimizer transport source inventory'),
        ('released','check_released_current.py',['--study',ROOT/'conditional-transport-20260913/single-pass'],ROOT/'conditional-transport-20260913/single-pass/execution-qualification','Recursive full/factor source resolution, pinned assets and data'),
        ('factorial','check_factorial_admission.py',['--study',factorial_study],factorial_study/'profile/reference1/manifest.json','Three declared scripts, every package module, native assets and frozen corpus/selection')]
    routes=[]
    for label,script,args,qualification,policy in definitions:
        target=out/(label+'.json') if label=='released' else out/label
        command=run(out,label,script,*args,'--output',target)
        vf=target if label=='released' else target/'verification.json';v=json.loads(vf.read_text())
        if v.get('status')!='passed':raise ValueError('Admission incomplete '+label)
        if label=='optimizer':
            qualification=Path(json.loads((ROOT/'optimizer-transport-20260912/assessment/protocol.json').read_text())['qualification']['path'])
        routes.append(dict(name=label,inventory_policy=policy,qualification=str(qualification),
            verification=str(vf),verification_sha256=sha256(vf),command=command,
            checks=len(v.get('checks',v.get('contracts',[]))),native_updates=0,
            coverage='Full/factor current contracts and base-assessment identity; ordinary Python' if label=='released' else 'Positive baselines and adverse entry cases in ordinary and optimized Python'))
    sources={str(p.relative_to(REPO)):sha256(p) for folder in ['src','scripts'] for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts}
    result=dict(schema='current-execution-routes-v2',status='passed',routes=routes,source_identity=sources,
        primary_inventory_policy='Whole package plus six declared scripts; adding analysis modules requires fresh native qualification.',
        distinction='Admission fixtures do not contain scientific trajectories. Source identity, native qualification, and archived scientific reconstruction are separately checked.',
        checker_sha256=sha256(__file__))
    write_json(out/'routes.json',result);return result


def reindex(base,out):
    """Repair reporting metadata while preserving executed admission evidence.

    Every pre-existing Python dependency except this reporting driver must
    still match the executed snapshot. No native qualification, suite result,
    command, log or executing-checker hash is rewritten.
    """
    source=base/'admission/routes.json';result=json.loads(source.read_text())
    if out.exists():raise ValueError('A fresh route-index destination is required')
    if result.get('status')!='passed':raise ValueError('Completed admission required')
    driver=str(Path(__file__).resolve().relative_to(REPO))
    for name,digest in result['source_identity'].items():
        if name!=driver and sha256(REPO/name)!=digest:
            raise ValueError('Executing snapshot changed: '+name)
    old_driver=result['checker_sha256']
    for route in result['routes']:
        vf=Path(route['verification']);v=json.loads(vf.read_text())
        if sha256(vf)!=route['verification_sha256'] or v.get('status')!='passed':
            raise ValueError('Admission evidence changed')
        command=route['command'];cmd=command['command']
        if sha256(Path(cmd[1]))!=v['checker_sha256']:
            raise ValueError('Admission checker changed')
        if sha256(source.parent/(command['label']+'.log'))!=command['log_sha256']:
            raise ValueError('Executed admission log changed')
        if route['name']=='optimizer':
            study=Path(cmd[cmd.index('--study')+1])
            qualification=json.loads((study/'protocol.json').read_text())['qualification']
            if sha256(qualification['path'])!=qualification['sha256']:
                raise ValueError('Bound optimizer qualification changed')
            route['qualification']=qualification['path']
        if not Path(route['qualification']).exists():raise ValueError('Missing qualification')
    result['metadata_reconstruction']=dict(source=str(source),source_sha256=sha256(source),
        executed_driver_sha256=old_driver,
        reason='Resolve the informational optimizer qualification path from its unchanged executed protocol; retain every suite result and command.')
    result['checker_sha256']=sha256(__file__)
    result['source_identity'][driver]=sha256(__file__)
    pending=out.with_suffix('.pending.json')
    if pending.exists():raise ValueError('Pending route index already exists')
    write_json(pending,result)
    from verify_execution_routes import verify
    verified=verify(pending)
    pending.rename(out)
    print(verified['suite_counts'],flush=True)



def assemble(base,out,retained_routes,factorial_study):
    """Reuse unchanged adverse suites and revalidate every live positive route.

    Primary/physical records are freshly completed in base. The factorial
    suite runs on the current qualified source. Other suites retain their
    exact records, checker identities and per-route dependency contracts.
    """
    import copy
    old=json.loads(Path(retained_routes).read_text())
    if old['status']!='passed' or out.exists():raise ValueError('Passed inputs and a fresh index are required')
    target=base/'factorial'
    command=run(base,'factorial','check_factorial_admission.py','--study',factorial_study,'--output',target)
    routes=[]
    for original in old['routes']:
        route=copy.deepcopy(original);name=route['name']
        if name in ['primary','physical','factorial']:
            vf=base/name/'verification.json'
            if name=='factorial':
                route['command']=command
                route['qualification']=str(Path(factorial_study)/'profile/reference1/manifest.json')
            else:
                route['command']=json.loads((base/(name+'-command.json')).read_text())
                if name=='primary':
                    cmd=route['command']['command']
                    route['qualification']=str(Path(cmd[cmd.index('--qualification')+1])/'verification.json')
            v=json.loads(vf.read_text());route['verification']=str(vf.resolve())
            route['verification_sha256']=sha256(vf)
            route['checks']=len(v.get('checks',v.get('contracts',[])))
            route['admission_record_scope']='Freshly executed adverse and positive entry suite'
        else:
            vf=Path(route['verification']);v=json.loads(vf.read_text())
            if sha256(vf)!=route['verification_sha256']:raise ValueError('Retained suite changed')
            route['admission_record_scope']='Immutable completed adverse suite; unchanged checker and native dependencies; live positives revalidated'
        checker=REPO/'scripts'/Path(route['command']['command'][1]).name
        if v['status']!='passed' or v['checker_sha256']!=sha256(checker):raise ValueError('Changed admission checker '+name)
        routes.append(route)
    sources={str(p.relative_to(REPO)):sha256(p) for folder in ['src','scripts'] for p in (REPO/folder).rglob('*.py') if '__pycache__' not in p.parts}
    result=dict(schema='current-execution-routes-v2',status='passed',routes=routes,source_identity=sources,
        primary_inventory_policy=old['primary_inventory_policy'],checker_sha256=sha256(__file__),
        distinction='Fresh primary/physical/factorial suites and retained unchanged continuation/optimizer/released suites. Every live positive contract is revalidated; no scientific update count is inferred.',
        retained_suite_index=str(Path(retained_routes).resolve()),retained_suite_index_sha256=sha256(retained_routes))
    pending=out.with_suffix('.pending.json')
    if pending.exists():raise FileExistsError(pending)
    write_json(pending,result)
    from verify_execution_routes import verify
    verification=verify(pending)
    pending.rename(out)
    print(verification['suite_counts'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['qualify','admit','reindex','assemble']);p.add_argument('--base',type=Path,default=DEFAULT);p.add_argument('--output',type=Path);p.add_argument('--factorial-study',type=Path);p.add_argument('--retained-routes',type=Path);a=p.parse_args()
    if a.action=='qualify':qualify(a.base)
    elif a.action=='admit':admit(a.base,a.output or a.base/'admission',a.factorial_study)
    elif a.action=='assemble':assemble(a.base,a.output or a.base/'routes-final.json',a.retained_routes,a.factorial_study)
    else:reindex(a.base,a.output or a.base/'admission/routes-indexed.json')
