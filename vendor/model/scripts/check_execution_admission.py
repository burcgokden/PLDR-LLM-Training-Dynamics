#!/usr/bin/env python
"""Exercise public P1/P3 entry points, including Python -O, with fault injection.

Positive baselines are completed real native qualifications. A subprocess-level
artifact-read overlay supplies isolated corrupt bytes without changing live data.
Producer/dependency faults use a real separate source-tree copy. Scientific
prepare paths run normally; launch/worker smoke fixtures stop before an update.
"""
from companion_paths import child_pythonpath
from companion_paths import legacy_path
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from model_rg.provenance import sha256, write_json


def probe(args):
    repo=Path(args.repo).resolve(); sys.path[:0]=[str(repo/'src'),str(repo/'scripts')]
    import model_rg.qualification as q
    import importlib
    stage=args.stage
    module=importlib.import_module('refresh_study' if stage=='P1' else 'directional_study')
    study=Path(args.study).resolve(); root=Path(args.root).resolve()
    if args.fault:
        fault=json.loads(Path(args.fault).read_text())
        original_sha=q.sha256; original_read=q.read; original_runtime=q.runtime
        if fault['kind']=='artifact':
            target=Path(fault['target']).resolve(); shadow=Path(fault['shadow'])
            q.sha256=lambda path: original_sha(shadow if Path(path).resolve()==target else path)
        elif fault['kind']=='runtime':
            def changed():
                value=original_runtime(); value['numerical_policy']['tf32']=True; return value
            q.runtime=changed
        elif fault['kind']=='record':
            target=Path(fault['target']).resolve()
            def changed(path):
                return json.loads(Path(fault['shadow']).read_text()) if Path(path).resolve()==target else original_read(path)
            q.read=changed
        else: raise ValueError('Unknown test fault')
    class BeforeNativeUpdate(RuntimeError): pass
    if args.action in ['worker','run'] and args.smoke:
        # Actual public worker/launcher code runs through admission. This sentinel
        # substitutes only the first model construction / child process launch.
        def stop(*a,**k): raise BeforeNativeUpdate('ADMISSION_REACHED_NATIVE_BOUNDARY')
        if args.action=='worker':module.TrainingModel=stop
        else:module.subprocess.run=stop
    if stage=='P1':
        sys.argv=['refresh_study.py',args.action,'--root',str(root),'--study',str(study)]
    else:
        sys.argv=['directional_study.py',args.action,'--root',str(root),'--study',str(study),'--kind','development']
    if args.action=='worker': sys.argv+=['--case','h4-c0-i0','--device','cuda:1']
    try: module.main()
    except BeforeNativeUpdate:
        print('ADMISSION_REACHED_NATIVE_BOUNDARY'); return
    if args.smoke: raise RuntimeError('Expected worker/launch boundary was not reached')


def main():
    p=argparse.ArgumentParser();p.add_argument('--qualification-root');p.add_argument('--output')
    p.add_argument('--repo',default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--root',default=legacy_path('/pldr-data/model'))
    p.add_argument('--probe',action='store_true');p.add_argument('--stage');p.add_argument('--action')
    p.add_argument('--study');p.add_argument('--fault');p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.probe:probe(a);return
    repo=Path(a.repo).resolve();base=Path(a.qualification_root).resolve();out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repo/'src'))
    from model_rg.qualification import validate_qualification, REQUIRED
    records=[];environment=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
    real={}
    for stage,folder in [('P1','refresh'),('P3','directional')]:
        protocol=base/folder/'qualification/protocol.json'
        real[stage]=validate_qualification(protocol,repo,stage).record()
    def execute(label, stage, action, study, optimized, fault=None, smoke=False, repository=repo, passes=False):
        command=[sys.executable]+(['-O'] if optimized else [])+[str(repo/'scripts/check_execution_admission.py'),'--probe',
            '--repo',str(repository),'--root',a.root,'--stage',stage,'--action',action,'--study',str(study)]
        if fault: command+=['--fault',str(fault)]
        if smoke: command+=['--smoke']
        result=subprocess.run(command,env=environment,cwd=repo,capture_output=True,text=True)
        text=result.stdout+result.stderr
        if (result.returncode==0)!=passes: raise RuntimeError(label+': unexpected admission\n'+text)
        if not passes and 'QualificationError' not in text:raise RuntimeError(label+': failure was not the guard\n'+text)
        if passes and smoke and 'ADMISSION_REACHED_NATIVE_BOUNDARY' not in text:raise RuntimeError(label+': missed native boundary')
        log=out/(label+'.log');log.write_text(text)
        records.append(dict(name=label,stage=stage,entry=action,optimized=optimized,expected_pass=passes,
                            returncode=result.returncode,log_sha256=sha256(log)))
    # These preparation outputs are intentionally unexecuted test artifacts.
    # They demonstrate the explicitly permitted full-design inputs and new draws.
    for stage,folder in [('P1','refresh'),('P3','directional')]:
        qprotocol=Path(real[stage]['protocol']);qstudy=qprotocol.parent
        for optimized in [False,True]:
            suffix='optimized' if optimized else 'ordinary'
            area=out/(folder+'-'+suffix);area.mkdir()
            (area/'qualification').symlink_to(qstudy,target_is_directory=True)
            science=area/'science'
            execute(stage+'-prepare-'+suffix,stage,'prepare',science,optimized,passes=True)
            for action in ['run','worker']:
                execute(stage+'-'+action+'-smoke-'+suffix,stage,action,science,optimized,smoke=True,passes=True)
            # Remove only empty smoke-created directories; no native result was written.
            if (science/'runs').exists():shutil.rmtree(science/'runs')
            spec=json.loads(qprotocol.read_text())
            first_case=spec['cases'][0]['name']
            artifacts={
                'producer': repo/REQUIRED[stage][0],
                'verifier': repo/('scripts/verify_refresh.py' if stage=='P1' else 'scripts/verify_directional.py'),
                'transitive_dependency':repo/'src/model_rg/training.py',
                'native_source':Path(a.root)/'assets/PLDR-LLM-v51-SOC-110M-1/modeling_pldrllm.py',
                'incoming_state':Path(a.root)/'outer-transfer-20260911/runs'/first_case/'incoming-state.pt',
                'qualification_protocol':qprotocol,
                'raw_output':qstudy/'runs'/first_case/('parent/branch-000.npz' if stage=='P1' else 'source0-zero.npz'),
                'stale_analysis':qstudy/'analysis.json',
            }
            faults={}
            for key,target in artifacts.items():
                shadow=out/(stage+'-'+suffix+'-'+key+'.corrupt');shadow.write_bytes(b'isolated altered artifact\n')
                faults[key]=dict(kind='artifact',target=str(target),shadow=str(shadow))
            faults['runtime']=dict(kind='runtime')
            for key,record in [('one_field_status',{'status':'passed'}),('wrong_stage',{'schema':'unknown','status':'frozen'})]:
                shadow=out/(stage+'-'+suffix+'-'+key+'.json');shadow.write_text(json.dumps(record))
                faults[key]=dict(kind='record',target=str(qprotocol),shadow=str(shadow))
            terminal=qstudy/('summary-verification.json' if stage=='P1' else 'parity-analysis.json')
            shadow=out/(stage+'-'+suffix+'-missing-secondary.json');shadow.write_text('{}')
            faults['missing_secondary']=dict(kind='record',target=str(terminal),shadow=str(shadow))
            for key,fault in faults.items():
                f=out/(stage+'-'+suffix+'-'+key+'-fault.json');f.write_text(json.dumps(fault))
                for action in ['prepare','run','worker']:
                    target=area/('rejected-'+key) if action=='prepare' else science
                    execute(stage+'-'+suffix+'-'+key+'-'+action,stage,action,target,optimized,fault=f)
                    if action=='prepare' and target.exists():raise RuntimeError('Rejected prepare wrote scientific output')
            # A real isolated transitive-source modification, without a reader overlay.
            clone=out/(stage+'-'+suffix+'-source-copy');clone.mkdir()
            for name in spec['sources']:
                target=clone/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,target)
            (clone/'src/model_rg/training.py').write_text((clone/'src/model_rg/training.py').read_text()+'\n# isolated changed dependency\n')
            execute(stage+'-'+suffix+'-actual-copy-change',stage,'prepare',area/'copy-rejected',optimized,repository=clone)
    write_json(out/'verification.json',dict(status='passed',schema='native-entry-admission-check-v1',checks=records,
        real_qualifications=real,checker_sha256=sha256(__file__),guard_sha256=sha256(repo/'src/model_rg/qualification.py'),
        native_updates=0,scope='Real public preparation/launch/worker paths in ordinary and optimized Python. Complete real short native qualifications supply positive baselines. Isolated artifact-read overlays inject corrupt qualification evidence and assets; copied source trees test actual dependency edits. Smoke stops at model construction or child launch. Prepared full designs are unexecuted test artifacts.'))
    print('Passed',len(records),'entry-point checks; zero scientific updates')

if __name__=='__main__':main()
