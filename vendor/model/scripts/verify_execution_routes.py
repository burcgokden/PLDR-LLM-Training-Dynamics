#!/usr/bin/env python3
"""Publication gate for actual current positive and adverse admission evidence.

The route suites execute before publication. This gate checks their precise
source/mode bindings and calls the live native qualification validators again.
Incidental analysis/build scripts are not execution dependencies; each route
keeps the inventory policy enforced by its native qualification contract.
"""
from companion_paths import child_pythonpath
from companion_paths import configured_path
import argparse,json,os,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.provenance import sha256,write_json


def verify(routes_file):
    p=Path(routes_file);r=json.loads(p.read_text())
    if r.get('status')!='passed' or r.get('schema')!='current-execution-routes-v2':raise ValueError('Missing complete current execution routes')
    if {v['name'] for v in r['routes']}!={'primary','physical','continuations','optimizer','released','factorial'}:raise ValueError('Missing advertised route')
    if r['checker_sha256']!=sha256(REPO/'scripts/refresh_execution_routes.py'):raise ValueError('Route suite driver changed')
    # This complete package inventory is intentionally conservative for primary.
    for source in (REPO/'src/model_rg').glob('*.py'):
        if r['source_identity'].get(str(source.relative_to(REPO)))!=sha256(source):raise ValueError('Package changed after admission')
    bound={str(p.resolve()):sha256(p)};counts={};qualifications=[]
    for route in r['routes']:
        vf=Path(route['verification']);v=json.loads(vf.read_text())
        if sha256(vf)!=route['verification_sha256'] or v.get('status')!='passed':raise ValueError('Changed admission result')
        cmd=route['command']['command'];script=Path(cmd[1]);current=REPO/'scripts'/script.name
        if v['checker_sha256']!=sha256(current):raise ValueError('Admission checker changed')
        if r['source_identity'].get(str(current.relative_to(REPO)))!=sha256(current):raise ValueError('Checker differs from tested source')
        bound[str(vf)]=sha256(vf);counts[route['name']]=len(v.get('checks',v.get('contracts',[])))
        for filename,h in v.get('files',{}).items():
            if sha256(vf.parent/filename)!=h:raise ValueError('Admission mode evidence changed')
            bound[str(vf.parent/filename)]=h
        if route['name'] in ['primary','physical','factorial']:
            combined=[]
            for mode,optimized in [('ordinary',False),('optimized',True)]:
                modefile=vf.parent/(mode+'.json');m=json.loads(modefile.read_text())
                if m.get('optimized') is not optimized or m.get('status')!='passed' or not m.get('checks'):raise ValueError('Missing Python mode')
                if not all(c['admitted']==c['expected_admit'] for c in m['checks']):raise ValueError('Wrong admission outcome')
                combined+=m['checks']
            if combined!=v['checks']:raise ValueError('Incomplete combined admission suite')
        args=cmd[2:]
        arg=lambda name:Path(args[args.index(name)+1])
        if route['name']=='primary':
            import finetuning_study as worker
            study=arg('--study');short=arg('--qualification')
            worker.validate(short);worker.validate(study)
            qualifications.append(str(short/'verification.json'))
        elif route['name']=='physical':
            from physical_design import admit
            study=arg('--study');admit(study)
            qualifications += [x['path'] for x in json.loads((study/'protocol.json').read_text())['qualifications'].values()]
        elif route['name']=='continuations':
            from model_rg.qualification import validate_qualification
            base=arg('--qualification-root')
            if {c['optimized'] for c in v['checks']}!={False,True}:raise ValueError('Continuation Python mode missing')
            for stage,folder in [('P1','refresh'),('P3','directional')]:
                qp=base/folder/'qualification/protocol.json';validate_qualification(qp,REPO,stage);qualifications.append(str(qp))
        elif route['name']=='optimizer':
            from optimizer_transport_study import validate
            study=arg('--study');validate(study)
            if not all(any(c['name'].startswith(mode) and c['expected_pass'] for c in v['checks']) for mode in ['ordinary','optimized']):raise ValueError('Optimizer positive mode missing')
            qualifications.append(json.loads((study/'protocol.json').read_text())['qualification']['path'])
        elif route['name']=='factorial':
            from factorial_admission import preflight,NAMES
            study=arg('--study')
            if v['protocol_sha256']!=sha256(study/'protocol.json') or v['profile_sha256']!=sha256(study/'profile/reference1/manifest.json'):raise ValueError('Changed factorial admission baseline')
            for name in NAMES:
                checked=preflight(study,name,False,REPO,Path(configured_path('data:model')))
                bound.update(checked[-1]);del checked
            qualifications.append(str(study/'profile/reference1/manifest.json'))
        else:
            for flags in [[],['-O']]:
                result=subprocess.run([sys.executable,*flags,str(REPO/'scripts/check_released_current.py'),'--study',str(arg('--study'))],cwd=REPO,capture_output=True,text=True,
                    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4'))
                if result.returncode:raise ValueError('Released current positive baseline failed: '+result.stderr)
                q=json.loads(result.stdout);qualifications += [c['qualification'] for c in q['contracts']]
    for q in set(qualifications):bound[q]=sha256(q)
    return dict(status='passed',schema='current-execution-publication-v2',suite_counts=counts,checked_sha256=bound,
        current_positive_baselines_revalidated=True,both_primary_and_physical_python_modes=True,both_factorial_python_modes=True,covered_routes=sorted(counts),count_units={name:('branch contracts' if name=='released' else 'admission cases') for name in counts},
        verifier_sha256=sha256(__file__),routes_sha256=sha256(p),
        scope='Executed current admission suites with checked mode evidence and live native qualification validators. No scientific updates or passed flags are inferred from file presence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--routes',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=verify(a.routes);write_json(a.output,result);print(result['suite_counts'])
