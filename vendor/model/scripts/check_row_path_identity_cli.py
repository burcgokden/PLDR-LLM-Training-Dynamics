#!/usr/bin/env python3
"""Actual normal/optimized CLI regressions for row labels, summaries and certificates."""
from companion_paths import child_pythonpath
import argparse,copy,json,os,subprocess,sys
from pathlib import Path
from model_rg.provenance import sha256,write_json


def mutations(base):
    def endpoint(k,v):return lambda a:a['endpoints'][0].__setitem__(k,v)
    def cell(k,v):return lambda a:a['cells'][0].__setitem__(k,v)
    def swap(a):
        for k in ['heads','control','seed']:a['cells'][0][k],a['cells'][4][k]=a['cells'][4][k],a['cells'][0][k]
    def reverse(a):a['endpoints'].reverse();a['cells'].reverse()
    return {
      'valid':lambda a:None,'valid-reordering':reverse,
      'endpoint-seed':endpoint('seed',9163403),'cell-label-swap':swap,
      'increment':endpoint('increment',base['endpoints'][0]['increment']+.1),
      'duplicate-endpoint':lambda a:a['endpoints'].__setitem__(0,copy.deepcopy(a['endpoints'][1])),
      'duplicate-cell':lambda a:a['cells'].__setitem__(0,copy.deepcopy(a['cells'][1])),
      'missing-endpoint':lambda a:a['endpoints'].pop(),'missing-cell':lambda a:a['cells'].pop(),
      'endpoint-heads':endpoint('heads',24),'endpoint-control':endpoint('control',2.),
      'endpoint-steps':endpoint('steps',513),'cell-steps':cell('steps',513),
      'endpoint-foreign-id':endpoint('run_id','foreign'),'endpoint-seed-string':endpoint('seed','9163402'),
      'endpoint-applicability':endpoint('one_step_bound_applicable',1279),'endpoint-pairs':endpoint('one_step_pairs',1279),
      'cell-pairs':cell('pairs',1279),'cell-block-bool':cell('block',True),
      'cell-residual':cell('identity_error',1e-6),'cell-composition':cell('composition_error',1e-6),
      'residual-summary':lambda a:a['maximum_errors'].__setitem__('finite_identity',1e-6),
      'path-count':lambda a:a.__setitem__('paths',17),'identity-count':lambda a:a.__setitem__('initialization_identities',3),
      'schema':lambda a:a.__setitem__('schema','foreign'),'scope':lambda a:a.__setitem__('scope','Independent models'),
      'extra-field':lambda a:a.__setitem__('independent_models',16),
      'raw-binding':lambda a:a['checked_sha256'].__setitem__('protocol.json','0'*64)}


def main(study,analysis,output):
    output.mkdir(parents=True,exist_ok=False);repo=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    base=json.loads(analysis.read_text());records=[]
    for optimized in [False,True]:
        prefix=[sys.executable,'-B']+(['-O'] if optimized else [])
        valid_cert=None
        for name,change in mutations(base).items():
            folder=output/f'{int(optimized)}-{name}';folder.mkdir();a=copy.deepcopy(base);change(a)
            ap=folder/'analysis.json';write_json(ap,a);vp=folder/'verification.json';want=name.startswith('valid')
            cmd=prefix+[str(repo/'scripts/verify_row_path_confirmation.py'),'--study',str(study),'--analysis',str(ap),'--output',str(vp)]
            p=subprocess.run(cmd,env=env,capture_output=True,text=True);(folder/'verify.log').write_text(p.stdout+p.stderr)
            if (p.returncode==0)!=want or (not want and vp.exists()):raise ValueError('Wrong verifier outcome '+name)
            if want:
                rp=subprocess.run(prefix+[str(repo/'scripts/render_row_path_confirmation.py'),'--analysis',str(ap),'--verification',str(vp),'--output',str(folder/'tables')],env=env,capture_output=True,text=True)
                (folder/'render.log').write_text(rp.stdout+rp.stderr)
                if rp.returncode:raise ValueError('Valid render rejected')
                if name=='valid':valid_cert=vp
            records.append(dict(name=name,optimized=optimized,passed=True,expected_acceptance=want,command=cmd))
        for name in ['altered-analysis','stale-schema','stale-contract','stale-verifier']:
            folder=output/f'{int(optimized)}-renderer-{name}';folder.mkdir();ap=folder/'analysis.json';write_json(ap,base)
            cert=json.loads(valid_cert.read_text())
            if name=='altered-analysis':z=copy.deepcopy(base);z['endpoints'][0]['seed']+=1;write_json(ap,z)
            elif name=='stale-schema':cert['schema']='row-path-verification-v1'
            elif name=='stale-contract':cert['contract_sha256']='0'*64
            else:cert['verifier_sha256']='0'*64
            vp=folder/'verification.json';write_json(vp,cert)
            cmd=prefix+[str(repo/'scripts/render_row_path_confirmation.py'),'--analysis',str(ap),'--verification',str(vp),'--output',str(folder/'tables')]
            proc=subprocess.run(cmd,env=env,capture_output=True,text=True);(folder/'render.log').write_text(proc.stdout+proc.stderr)
            if proc.returncode==0 or (folder/'tables').exists():raise ValueError('Renderer accepted '+name)
            records.append(dict(name='renderer-'+name,optimized=optimized,passed=True,expected_acceptance=False,command=cmd))
    sources=['check_row_path_identity_cli.py','verify_row_path_confirmation.py','render_row_path_confirmation.py','row_path_identity_contract.py']
    write_json(output/'verification.json',dict(status='passed',cases=records,case_count=len(records),
      rejected=sum(not c['expected_acceptance'] for c in records),accepted=sum(c['expected_acceptance'] for c in records),
      tested_sources={'scripts/'+n:sha256(repo/'scripts'/n) for n in sources},analysis_sha256=sha256(analysis),
      scope='Intentionally altered fixtures only; all scientific raw inputs read-only. Zero native updates or forwards.'))
    print('Passed',len(records),'ordinary/optimized actual CLI cases',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['study','analysis','output']:p.add_argument('--'+k,type=Path,required=True)
    main(**vars(p.parse_args()))
