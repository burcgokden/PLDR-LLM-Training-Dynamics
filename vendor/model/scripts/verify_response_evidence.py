#!/usr/bin/env python
"""Compose preserved publication evidence, fresh raw reductions and current checks."""
import argparse,json
from pathlib import Path
from model_rg.provenance import sha256,write_json
from model_rg.qualification import validate_qualification
from confirmation_status import validate,render

def require(flag,message):
    if not flag:raise ValueError(message)

def main():
    p=argparse.ArgumentParser();p.add_argument('--retained-release',required=True);p.add_argument('--study',required=True)
    p.add_argument('--qualifications',required=True);p.add_argument('--entry-checks',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];retained=Path(a.retained_release).resolve();study=Path(a.study).resolve();checked={}
    require(not Path(a.output).exists(),'Fresh evidence output required')
    def check(path,digest=None):
        path=Path(path).resolve();value=sha256(path)
        require(digest is None or value==digest,'Changed evidence: '+str(path));checked[str(path)]=value;return value
    def read(path):check(path);return json.loads(Path(path).read_text())
    def record(path):return dict(path=str(Path(path).resolve()),sha256=check(path))
    inventory={}
    for line in (retained/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);path=(retained/name).resolve();require(path.is_relative_to(retained),'Invalid retained path')
        check(path,digest);inventory[name]=digest
    require(set(inventory)=={str(p.relative_to(retained)) for p in retained.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'},'Retained inventory differs')
    old=read(retained/'evidence-index.json');require(old['status']=='passed' and old['schema']=='onepass-directional-complete-evidence-v1','Wrong retained evidence')
    preserved={}
    for path in (retained/'arxiv-source/generated').rglob('*'):
        if path.is_file():
            rel=path.relative_to(retained/'arxiv-source/generated').as_posix();preserved[rel]=check(repo/'manuscript/generated'/rel,sha256(path))
    fresh={}
    for name in ['assessment','invisible-sector']:
        path=study/name/'verification.json';data=read(path);require(data['status']=='passed','Incomplete fresh reduction')
        for file,digest in data['checked_sha256'].items():check(file,digest)
        fresh[name]=record(path)
    response=read(study/'assessment/verification.json');null=read(study/'invisible-sector/verification.json')
    require((response['summary']['native_updates'],response['summary']['arithmetic_control_updates'],response['summary']['raw_files'],len(response['primary']))==(576,576,148,32),'Wrong response inventory')
    require(null['native_updates']==64 and len(null['rows'])==4 and all(r['bitwise_unaffected'] and r['bitwise_row_forecast'] for r in null['rows']),'Incomplete mechanism test')
    support_path=study/'invisible-sector/input-support-verification.json'
    support=read(support_path)
    require(support['status']=='passed' and support['corpus_occurrences']==support['evaluation_occurrences']==0,'Finite input support failed')
    require((support['corpus_input_positions'],support['evaluation_input_positions'])==(33554432,2048),'Wrong support scope')
    check(repo/'scripts/verify_input_support.py',support['verifier_sha256'])
    for file,digest in support['checked_sha256'].items():check(file,digest)
    check(repo/'manuscript/generated/invisible-sector-support.json',sha256(support_path))
    qualifications={}
    for stage,name in [('P1','refresh'),('P3','directional')]:
        item=validate_qualification(Path(a.qualifications)/name/'qualification/protocol.json',repo,stage)
        qualifications[stage]=item.record()
        check(item.protocol,item.protocol_sha256)
        for file,digest in item.terminal.items():check(file,digest)
    admission=read(a.entry_checks)
    require(admission['status']=='passed' and admission['native_updates']==0,'Admission checks incomplete')
    check(repo/'scripts/check_execution_admission.py',admission['checker_sha256']);check(repo/'src/model_rg/qualification.py',admission['guard_sha256'])
    require(admission['real_qualifications']==qualifications,'Admission test baseline changed')
    cases=admission.get('checks',[])
    require(len(cases)==160 and len({r['name'] for r in cases})==160,'Incomplete admission test matrix')
    require({(r['stage'],r['entry'],r['optimized']) for r in cases}==
            {(stage,entry,optimized) for stage in ['P1','P3'] for entry in ['prepare','run','worker'] for optimized in [False,True]},
            'Missing public entry or Python mode')
    for case in cases:
        require((case['returncode']==0)==case['expected_pass'],'Unexpected admission outcome')
        check(Path(a.entry_checks).parent/(case['name']+'.log'),case['log_sha256'])

    checks={}
    for name,folder in [('numerical','numerical-response-numerical'),('formal','numerical-response-formal'),('statements','numerical-response-statements-final')]:
        file=repo/'docs'/folder/'verification.json';data=read(file);require(data['status']=='passed','Unpassed code check')
        for key in ['tested_sources','module_sources','modules','support_sources']:
            for filename,digest in data.get(key,{}).items():check(repo/filename,digest)
        if name=='numerical':require(data['tests_passed']==79,'Numerical suite inventory changed')
        if name=='formal':require(data['explicit_theorems']==69 and len(data['module_sources'])==16,'Formal scope changed')
        if name=='statements':require(data['exports']==69 and len(data['mutations'])==21 and all(x['rejected'] for x in data['mutations']),'Statement scope changed')
        checks[name]=record(file)
    status=read(repo/'docs/confirmation-status.json');validate(status)
    require((repo/'docs/CONFIRMATION_STATUS.md').read_text()==render(status),'Status text stale')
    statuscheck=read(repo/'docs/response-status-verification.json');require(statuscheck['status']=='passed','Status regression failed')
    check(repo/'scripts/check_confirmation_status.py',statuscheck['runner_sha256'])
    check(repo/'docs/confirmation-status.json',statuscheck['input_status_sha256'])
    inputs=read(repo/'docs/response-input-verification.json');require(inputs['status']=='passed','Input mapping failed')
    check(repo/'docs/response-inputs.json',inputs['manifest_sha256'])
    check(repo/'scripts/verify_input_identities.py',inputs['verifier_sha256'])
    compact=read(repo/'manuscript/generated/numerical-response-render-manifest.json')
    for name,digest in compact['generated'].items():check(repo/'manuscript/generated'/name,digest)
    check(repo/'scripts/render_numerical_response.py',compact['renderer_sha256'])
    sources={str(path.relative_to(repo)):check(path) for directory in ['scripts','src','tests','ModelRG','manuscript']
        for path in (repo/directory).rglob('*') if path.is_file() and path.suffix in ['.py','.sh','.lean','.tex','.bib']}
    for name in ['ModelRG.lean','lean-toolchain','lakefile.toml','lake-manifest.json','docs/RESPONSE_REPRODUCTION.md',
                 'docs/response-inputs.json','docs/confirmation-status.json','docs/CONFIRMATION_STATUS.md',
                 'README.md','docs/numerical-response-environment/environment.json',
                 'docs/numerical-response-environment/requirements-observed.txt']:
        sources[name]=check(repo/name)
    proof=dict(old)
    proof.update(schema='onepass-numerical-response-complete-evidence-v1',status='passed',
        fresh_native_paths=80,fresh_native_updates=640,arithmetic_control_paths=72,arithmetic_control_updates=576,
        fresh_replay_updates=48,response_primary_cells=32,invisible_sector_comparisons=4,
        total_conditional_branches=3312,total_conditional_updates=225920,total_singlepass_scientific_updates=2798720,
        total_including_precision_control=2799296,total_including_directional_development=2807936,current_numerical_tests=79,current_formal_statements=69,current_formal_modules=16,content_mutations=21,
        current_qualification_updates=1128,current_qualification_replay_updates=32,
        input_support=record(support_path),current_qualifications=qualifications,checks=checks,admission=record(a.entry_checks),measurements=fresh,
        retained_evidence=record(retained/'evidence-index.json'),preserved_generated=preserved,
        current_sources=sources,checked_sha256=checked,verifier_sha256=sha256(__file__),
        scope='Fresh raw-vocabulary reductions for all new pulse and null-sector paths, current complete native qualifications and entry-point fault checks, and retained publication-asset integrity. Historical training was not rerun. Float64 is a separate arithmetic law. Negative or unresolved cells of every reported comparison are retained.')
    write_json(a.output,proof);print('Passed complete numerical-response publication evidence',flush=True)

if __name__=='__main__':main()
