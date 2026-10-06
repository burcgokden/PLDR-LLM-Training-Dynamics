#!/usr/bin/env python
"""Bind immutable scientific evidence, current design qualification and projection checks."""
from companion_paths import required_input
import argparse
import json
from pathlib import Path
from finetuning_verification_design import sha

REPO=Path(__file__).resolve().parents[1]


def verify(retained,experiment,qualification,admission,output):
    retained,experiment,qualification,admission,output=map(lambda p:Path(p).resolve(),
        [retained,experiment,qualification,admission,output])
    if output.exists():raise FileExistsError('Fresh evidence destination required')
    checked={}
    def check(path,expected=None):
        p=Path(path).resolve();key=str(p)
        digest=checked.get(key)
        if digest is None:digest=sha(p);checked[key]=digest
        if expected is not None and digest!=expected:raise ValueError('Changed bound evidence: '+key)
        return digest
    def read(path):check(path);return json.loads(Path(path).read_text())
    def record(path):return dict(path=str(Path(path).resolve()),sha256=check(path))
    inventory=set()
    for line in (retained/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);p=(retained/name).resolve()
        if not p.is_relative_to(retained):raise ValueError('Invalid retained release path')
        check(p,digest);inventory.add(name)
    actual={str(p.relative_to(retained)) for p in retained.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
    if actual!=inventory:raise ValueError('Retained publication inventory differs')
    old=read(retained/'evidence-index.json')
    if old['status']!='passed' or old['schema']!='onepass-finetuning-complete-evidence-v1':raise ValueError('Wrong retained publication')
    generated=Path(required_input('generated-evidence'));preserved={}
    for p in (retained/'arxiv-source/generated').rglob('*'):
        if p.is_file() and p.name not in ['qualification-current.json','qualification-current.tex']:
            name=str(p.relative_to(retained/'arxiv-source/generated'))
            preserved[name]=check(generated/name,sha(p))
    check(generated/'finetuning-qualification.json',sha(retained/'arxiv-source/generated/qualification-current.json'))
    primary=read(experiment/'primary-verification.json')
    original=read(retained/'arxiv-source/generated/finetuning-verification.json')
    if primary['status']!='passed' or primary['stage']!='assessment' or primary['protocol_sha256']!=original['protocol_sha256']:
        raise ValueError('Scientific evidence identity differs')
    if primary['update_counts']!=original['update_counts'] or primary['complete_outcomes']!=original['complete_outcomes']:
        raise ValueError('Complete scientific outcomes differ')
    check(REPO/'scripts/verify_finetuning.py',primary['verifier_sha256'])
    check(REPO/'scripts/finetuning_verification_design.py',primary['design_verifier_sha256'])
    check(retained/'arxiv-source/code/scripts/verify_finetuning.py',original['verifier_sha256'])
    print('Checking complete primary archive bindings',flush=True)
    for index,(p,digest) in enumerate(primary['verified_files'].items(),1):
        check(p,digest)
        if index%100==0:print('Bound primary files',index,'of',len(primary['verified_files']),flush=True)
    print('Checking current native qualification',flush=True)
    q=read(qualification/'verification.json');qs=read(qualification/'protocol.json')
    if q['status']!='passed' or q['stage']!='qualification' or sum(q['update_counts'].values())!=102:
        raise ValueError('Incomplete canonical native qualification')
    check(REPO/'scripts/verify_finetuning.py',q['verifier_sha256'])
    check(REPO/'scripts/finetuning_verification_design.py',q['design_verifier_sha256'])
    check(qualification/'protocol.json',q['protocol_sha256'])
    from finetuning_study import validate,source_files,runtime
    from finetuning_design import check_design
    check_design(qs)
    if qs['sources']!=source_files() or qs['runtime']!=runtime():raise ValueError('Qualification source/runtime differs')
    for p,digest in q['verified_files'].items():check(p,digest)
    entry=read(admission/'verification.json')
    if entry['status']!='passed' or entry['native_updates']!=0 or not entry['checks'] or not all(c['admitted']==c['expected_admit'] for c in entry['checks']):
        raise ValueError('Admission checks failed')
    check(REPO/'scripts/check_primary_design_admission.py',entry['checker_sha256'])
    check(REPO/'scripts/finetuning_study.py',entry['producer_sha256'])
    check(qualification/'verification.json',entry['qualification_sha256'])
    for name,digest in entry['files'].items():check(admission/name,digest)
    # The frozen full design is exercised only for admission, not counted as an unrun scientific result.
    current_study=qualification.parent/'admission-assessment'
    check(current_study/'protocol.json',entry['protocol_sha256']);validate(current_study)
    print('Checking all-state projection and compact evidence',flush=True)
    raw=read(experiment/'verification.json');analysis=read(experiment/'analysis.json')
    if raw['status']!='passed' or analysis['status']!='passed' or raw['native_updates']!=0:raise ValueError('Projection not verified')
    check(REPO/'scripts/analyze_projection_transport.py',analysis['analyzer_sha256'])
    check(REPO/'scripts/verify_projection_transport.py',raw['verifier_sha256'])
    check(analysis['source'],analysis['source_sha256'])
    for p,digest in raw['input_sha256'].items():check(p,digest)
    from verify_projection_compact import verify as compact_verify
    compact=compact_verify(Path(required_input('publication-source')))
    from verify_finetuning_compact import verify as source_verify
    source_compact=source_verify(Path(required_input('publication-source')))
    rendered=read(generated/'projection-render-manifest.json')
    check(REPO/'scripts/render_projection_transport.py',rendered['renderer_sha256'])
    for p,digest in rendered['generated'].items():check(generated/p,digest)
    counts=read(generated/'qualification-current.json')
    for name,binding in counts['records'].items():
        item=read(binding['path']);check(binding['path'],binding['sha256'])
        if item['status']!='passed':raise ValueError('Current software checks failed')
        for key in ['tested_sources','module_sources','modules','support_sources']:
            for n,digest in item.get(key,{}).items():check(REPO/n,digest)
        if name=='statements':check(REPO/'scripts/formal/statement-registry.json',item['registry_sha256'])
        if name=='formal':check(REPO/'scripts/formal/Gate.lean',item['gate_sha256'])
    sources={}
    for folder in ['scripts','src','tests','ModelRG','manuscript']:
        for p in (REPO/folder).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh','.lean','.tex','.bib']:
                sources[str(p.relative_to(REPO))]=check(p)
    for n in ['ModelRG.lean','lean-toolchain','lakefile.toml','lake-manifest.json','pyproject.toml',
              'scripts/formal/statement-registry.json','docs/PROJECTION_REPRODUCTION.md','README.md']:
        sources[n]=check(REPO/n)
    c=counts['counts']
    proof=dict(schema='onepass-projection-complete-evidence-v1',status='passed',
        retained_release=str(retained),retained_evidence=record(retained/'evidence-index.json'),
        retained_manifest=record(retained/'MANIFEST.sha256'),preserved_generated=preserved,
        required_scientific_runs=old['required_scientific_runs'],
        total_native_scientific_updates=old['total_native_scientific_updates'],
        total_including_arithmetic_controls=old['total_including_arithmetic_controls'],
        fresh_native_scientific_updates=0,qualification_updates=102,projection_cells=24,
        current_numerical_tests=c['numerical_tests'],current_formal_statements=c['selected_statements'],
        current_formal_modules=c['formal_modules'],content_mutations=c['rejected_mutations'],
        primary_verification=record(experiment/'primary-verification.json'),
        qualification=record(qualification/'verification.json'),admission=record(admission/'verification.json'),
        projection_analysis=record(experiment/'analysis.json'),projection_verification=record(experiment/'verification.json'),
        compact=compact,source_compact=source_compact,current_sources=sources,checked_sha256=checked,
        verifier_sha256=sha(__file__),scope='Completed single-pass evidence with independently reconstructed canonical primary design; current native qualification is separate from scientific training. Complete fixed-panel projection diagnostics do not establish out-of-sample closure or native critical exponents.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(proof,indent=2)+'\n')
    print('Passed complete projection release evidence',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--retained-release',required=True);p.add_argument('--experiment',required=True)
    p.add_argument('--qualification',required=True);p.add_argument('--admission',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();verify(a.retained_release,a.experiment,a.qualification,a.admission,a.output)
