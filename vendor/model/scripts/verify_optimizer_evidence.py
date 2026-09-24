#!/usr/bin/env python3
"""Bind completed native evidence, current proofs and the publication sources."""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def require(x,message):
    if not x:raise ValueError(message)


def verify(retained,study,output):
    retained=Path(retained).resolve();study=Path(study).resolve();output=Path(output).resolve()
    require(not output.exists(),'Fresh evidence destination required')
    checked={}
    def check(p,h=None):
        p=Path(p).resolve(); actual=sha(p)
        require(h is None or actual==h,'Changed bound evidence: '+str(p));checked[str(p)]=actual;return actual
    def read(p):check(p);return json.loads(Path(p).read_text())
    def record(p):return dict(path=str(Path(p).resolve()),sha256=check(p))
    inventory=set()
    for line in (retained/'MANIFEST.sha256').read_text().splitlines():
        h,n=line.split('  ',1);p=(retained/n).resolve();require(p.is_relative_to(retained),'Invalid retained path')
        check(p,h);inventory.add(n)
    require(inventory=={str(p.relative_to(retained)) for p in retained.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'},'Retained release inventory differs')
    old=read(retained/'evidence-index.json')
    require(old['status']=='passed' and old['schema']=='onepass-numerical-response-complete-evidence-v1','Incorrect retained science')
    preserved={}
    for p in (retained/'arxiv-source/generated').rglob('*'):
        if p.is_file():
            n=p.relative_to(retained/'arxiv-source/generated').as_posix()
            preserved[n]=check(REPO/'manuscript/generated'/n,sha(p))
    native=read(study/'verification.json');analysis=read(study/'analysis.json');spec=read(study/'protocol.json')
    require(native['status']=='passed' and native['stage']=='assessment','Unpassed native assessment')
    require((native['native_updates'],native['arithmetic_updates'],native['replay_updates'],native['raw_archives'])==(288,288,32,76),'Incomplete native inventory')
    require(native['verifier_sha256']==sha(REPO/'scripts/verify_optimizer_transport.py'),'Native verifier changed')
    for p,h in native['verified_files'].items():check(p,h)
    qualification=read(spec['qualification']['path']);check(spec['qualification']['path'],spec['qualification']['sha256'])
    require(qualification['status']=='passed' and qualification['stage']=='qualification','Missing qualification')
    for p,h in qualification['verified_files'].items():check(p,h)
    from optimizer_transport_study import validate
    validate(study)
    from analyze_optimizer_transport import analyze
    require(analyze(study)==analysis,'Stored analysis differs from complete raw reduction')
    from verify_optimizer_compact import verify as compact
    compact_result=compact(REPO/'manuscript')
    rendered=read(REPO/'manuscript/generated/optimizer-transport-render-manifest.json')
    require(rendered['renderer_sha256']==sha(REPO/'scripts/render_optimizer_transport.py'),'Renderer changed')
    for n,h in rendered['generated'].items():check(REPO/'manuscript/generated'/n,h)
    counts=read(REPO/'manuscript/generated/qualification-current.json')
    for name,recorded in counts['records'].items():
        data=read(recorded['path']);check(recorded['path'],recorded['sha256'])
        require(data['status']=='passed','Current qualification incomplete')
        for key in ['tested_sources','module_sources','modules','support_sources']:
            for n,h in data.get(key,{}).items():check(REPO/n,h)
        if name=='statements':check(REPO/'scripts/formal/statement-registry.json',data['registry_sha256'])
    admission=read(REPO/'docs/optimizer-transport-admission/verification.json')
    require(admission['status']=='passed' and admission['native_updates']==0,'Admission check failed')
    require(admission['producer_sha256']==sha(REPO/'scripts/optimizer_transport_study.py'),'Admission source changed')
    require(admission['checker_sha256']==sha(REPO/'scripts/check_optimizer_admission.py'),'Admission checker changed')
    for row in admission['checks']:
        require((row['returncode']==0)==row['expected_pass'],'Admission expectation failed')
        check(REPO/'docs/optimizer-transport-admission'/(row['name']+'.log'),row['log_sha256'])
    ledger=read(REPO/'manuscript/generated/execution-ledger.json')
    require(sum(r['updates'] for r in ledger['rows'] if r['role']=='native_scientific')==2799008,'Scientific ledger does not reconcile')
    require(sum(r['updates'] for r in ledger['rows'] if r['role']=='arithmetic_control')==864,'Arithmetic-control ledger does not reconcile')
    sources={}
    for directory in ['scripts','src','tests','ModelRG','manuscript']:
        for p in (REPO/directory).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh','.lean','.tex','.bib']:
                sources[str(p.relative_to(REPO))]=check(p)
    for n in ['ModelRG.lean','lean-toolchain','lakefile.toml','lake-manifest.json','scripts/formal/statement-registry.json',
              'docs/OPTIMIZER_TRANSPORT_REPRODUCTION.md','docs/RESPONSE_REPRODUCTION.md','docs/response-inputs.json','README.md']:
        sources[n]=check(REPO/n)
    count=counts['counts']
    proof=dict(schema='onepass-optimizer-transport-complete-evidence-v1',status='passed',required_scientific_runs=566,
        retained_release=str(retained),retained_evidence=record(retained/'evidence-index.json'),
        retained_manifest=record(retained/'MANIFEST.sha256'),preserved_generated=preserved,
        fresh_native_paths=36,fresh_native_updates=288,arithmetic_control_paths=36,arithmetic_control_updates=288,
        total_native_scientific_updates=2799008,total_including_arithmetic_controls=2799872,
        current_numerical_tests=count['numerical_tests'],current_formal_statements=count['selected_statements'],
        current_formal_modules=count['formal_modules'],content_mutations=count['rejected_mutations'],
        experiment=record(study/'verification.json'),qualification=record(spec['qualification']['path']),
        admission=record(REPO/'docs/optimizer-transport-admission/verification.json'),compact=compact_result,
        current_sources=sources,checked_sha256=checked,verifier_sha256=sha(__file__),
        scope='Fresh full-vocabulary and sampled-coordinate reduction with current qualification; retained publication assets rehashed, and their earlier raw scientific evidence preserved by explicit binding. No claim to reexecute all retained training or to establish native critical limits.')
    output.write_text(json.dumps(proof,indent=2)+'\n');print('Complete evidence:',output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--retained-release',required=True);p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.retained_release,a.study,a.output)
