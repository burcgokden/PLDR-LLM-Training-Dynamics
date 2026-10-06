#!/usr/bin/env python
"""Bind the completed physical study, retained scientific evidence and current sources."""
from companion_paths import required_input
import argparse
import json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--retained-release',required=True)
    ap.add_argument('--output',required=True);a=ap.parse_args()
    base=Path(a.study).resolve();retained=Path(a.retained_release).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError(out)
    checked={}
    def check(path,expected=None):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=sha256(path)
        if expected is not None and checked[key]!=expected:raise ValueError('Changed bound evidence: '+key)
        return checked[key]
    def read(path):check(path);return json.loads(Path(path).read_text())
    inventory=set()
    for line in (retained/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);p=(retained/name).resolve()
        if not p.is_relative_to(retained):raise ValueError('Invalid retained path')
        check(p,digest);inventory.add(name)
    actual={str(p.relative_to(retained)) for p in retained.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
    if actual!=inventory:raise ValueError('Retained release inventory changed')
    old=read(retained/'evidence-index.json')
    if old['status']!='passed' or old['schema']!='onepass-projection-complete-evidence-v1':raise ValueError('Wrong retained evidence')
    generated=Path(required_input('generated-evidence'));preserved={}
    for p in (retained/'arxiv-source/generated').rglob('*'):
        if p.is_file() and p.name not in ['qualification-current.json','qualification-current.tex']:
            name=str(p.relative_to(retained/'arxiv-source/generated'));preserved[name]=check(generated/name,sha256(p))
    native_manifest=read(REPO/'docs/response-inputs.json')
    check(REPO/'docs/response-inputs.json',sha256(retained/'input-identities.json'))
    native_inputs={}
    for artifact in native_manifest['artifacts']:
        if artifact['role'].startswith('native/'):
            if artifact['root']!='data':raise ValueError('Unexpected native asset root')
            path=(base.parent/artifact['relative_path']).resolve()
            if not path.is_relative_to(base.parent):raise ValueError('Invalid native asset path')
            native_inputs[artifact['role']]=dict(path=str(path),sha256=check(path,artifact['sha256']))
    if len(native_inputs)!=8:raise ValueError('Incomplete native implementation pins')
    new=read(base/'verification.json')
    if new['status']!='passed' or new['scientific_updates']!=132352 or new['generated_cells']!=560:raise ValueError('Physical study incomplete')
    check(REPO/'scripts/verify_physical_study.py',new['verifier_sha256'])
    for p,digest in new['checked_sha256'].items():check(p,digest)
    print('Bound complete physical and retained scientific evidence',flush=True)
    rendered=read(generated/'physical-render-manifest.json')
    check(REPO/'scripts/render_physical_results.py',rendered['renderer_sha256'])
    check(base/'verification.json',rendered['study_verification_sha256'])
    for n,h in rendered['generated'].items():check(generated/n,h)
    qualification=read(generated/'qualification-current.json');counts=qualification['counts']
    if counts!={'numerical_tests':136,'selected_statements':87,'formal_modules':20,'formal_fixtures':8,'rejected_mutations':38}:
        raise ValueError('Unexpected current qualification inventory')
    for name,binding in qualification['records'].items():
        check(binding['path'],binding['sha256']);item=read(binding['path'])
        if item['status']!='passed':raise ValueError('Unpassed software checks')
        for key in ['tested_sources','module_sources','modules','support_sources']:
            for n,h in item.get(key,{}).items():check(REPO/n,h)
        if name=='statements':check(REPO/'scripts/formal/statement-registry.json',item['registry_sha256'])
        if name=='formal':check(REPO/'scripts/formal/Gate.lean',item['gate_sha256'])
    # Preserve and independently reconstruct the existing compact source and
    # fixed-Fisher projection outcomes as well as the new physical ones.
    from verify_finetuning_compact import verify as fine_verify
    from verify_projection_compact import verify as projection_verify
    fine=fine_verify(Path(required_input('publication-source')));projection=projection_verify(Path(required_input('publication-source')))
    sources={}
    for folder in ['scripts','src','tests','ModelRG','manuscript']:
        for p in (REPO/folder).rglob('*'):
            if p.is_file() and p.suffix in ['.py','.sh','.lean','.cpp','.tex','.bib']:
                sources[str(p.relative_to(REPO))]=check(p)
    for n in ['ModelRG.lean','lean-toolchain','lakefile.toml','lake-manifest.json','pyproject.toml',
              'scripts/formal/statement-registry.json','docs/PHYSICAL_REPRODUCTION.md','README.md']:
        sources[n]=check(REPO/n)
    proof=dict(status='passed',schema='physical-complete-release-v1',retained_release=str(retained),
        retained_manifest_sha256=check(retained/'MANIFEST.sha256'),retained_evidence_sha256=check(retained/'evidence-index.json'),
        preserved_generated=preserved,required_scientific_runs=old['required_scientific_runs']+18,
        retained_native_scientific_updates=old['total_native_scientific_updates'],
        fresh_native_scientific_updates=new['scientific_updates'],
        total_native_scientific_updates=old['total_native_scientific_updates']+new['scientific_updates'],
        total_including_arithmetic_controls=old['total_including_arithmetic_controls']+new['scientific_updates'],
        physical_development_updates=4096,physical_native_qualification_updates=24,physical_replay_updates=2048,
        current_numerical_tests=136,current_formal_statements=87,current_formal_modules=20,content_mutations=38,
        native_repository=native_manifest['native_repository'],native_revision=native_manifest['native_revision'],native_inputs=native_inputs,
        physical_study=str(base),physical_verification_sha256=check(base/'verification.json'),
        source_compact=fine,projection_compact=projection,current_sources=sources,checked_sha256=checked,
        verifier_sha256=sha256(__file__),scope='Completed RefinedWeb-conditioned theory and physical-law calibration. Source and native thermodynamic limits are distinct; native critical exponents and self-organization remain unestablished.')
    out.parent.mkdir(parents=True,exist_ok=True);write_json(out,proof)
    print('passed complete physical manuscript evidence',flush=True)


if __name__=='__main__':main()
