"""Publication gate separating scientific evidence and current executable admission."""
from companion_paths import required_input
from companion_paths import child_pythonpath
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def source_inventory():
    files=[]
    for folder in ['src','scripts','tests','ModelRG','manuscript']:
        files.extend(p for p in (REPO/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.lean','.cpp','.sh','.tex','.bib'])
    files += [REPO/n for n in ['ModelRG.lean','lakefile.toml','lake-manifest.json','lean-toolchain',
                             'pyproject.toml','scripts/formal/statement-registry.json']]
    return {str(p.relative_to(REPO)):sha256(p) for p in sorted(files)}


def main(study,retained,output):
    study=Path(study).resolve();retained=Path(retained).resolve();output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False);checked={};records={};source=source_inventory()
    def check(p,h=None):
        p=Path(p).resolve();key=str(p)
        if key not in checked:checked[key]=sha256(p)
        if h is not None and checked[key]!=h:raise ValueError('Evidence identity: '+key)
        return checked[key]
    def read(p):check(p);return json.loads(Path(p).read_text())
    # Validate the exact immutable scientific release, including file inventory.
    expected=set();check(retained/'MANIFEST.sha256')
    for line in (retained/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1);p=(retained/name).resolve()
        if not p.is_relative_to(retained):raise ValueError('Retained manifest path')
        check(p,digest);expected.add(name)
    actual={str(p.relative_to(retained)) for p in retained.rglob('*') if p.is_file() and p.name!='MANIFEST.sha256'}
    if actual!=expected:raise ValueError('Retained release inventory')
    old=read(retained/'evidence-index.json')
    if old['status']!='passed':raise ValueError('Retained scientific evidence')
    records['retained_scientific_release']=dict(path=str(retained),manifest_sha256=check(retained/'MANIFEST.sha256'))
    for key,relative in [
        ('fresh_scientific_observation','fresh-analysis/verification.json'),
        ('archived_physical_design','physical/archived-correspondence.json'),
        ('current_physical_branches','physical/qualification/verification.json'),
        ('current_finetuning','finetuning/qualification/verification.json')]:
        p=study/relative;v=read(p)
        if v['status']!='passed':raise ValueError('Incomplete '+key)
        check(REPO/('scripts/verify_fresh_readout.py' if key=='fresh_scientific_observation' else
                    'scripts/verify_finetuning.py' if key=='current_finetuning' else 'scripts/verify_physical_execution.py'),v['verifier_sha256'])
        for name,digest in v.get('checked_sha256',v.get('verified_files',{})).items():check(name,digest)
        records[key]=dict(path=str(p),sha256=check(p),role=v.get('role',v.get('scope')))
    if read(study/'fresh-analysis/verification.json')['cells']!=648:raise ValueError('Readout cell inventory')
    for name in ['physical/branch-h4.json','physical/branch-h8.json','physical/parity-h4.json','physical/parity-h8.json',
                 'fresh-analysis/analysis.json','reference-corrections/analysis.json','readout/readout-budget.json']:
        p=study/name;v=read(p)
        if v['status'] not in ('passed','complete'):raise ValueError('Incomplete experiment record')
        records[name]=dict(path=str(p),sha256=check(p))
    # The publication gate invokes the actual shipped baseline/adverse checkers.
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='4')
    commands=[('fine-admission',['check_primary_design_admission.py','--study',str(study/'finetuning/admission-baseline'),
        '--qualification',str(study/'finetuning/qualification')]),
        ('physical-admission',['check_physical_admission.py','--study',str(study/'physical/confirmation')])]
    for name,args in commands:
        command=[sys.executable,str(REPO/'scripts'/args[0]),*args[1:],'--output',str(output/name)]
        with (output/(name+'.log')).open('x') as log:
            subprocess.run(command,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        v=read(output/name/'verification.json')
        if v['status']!='passed' or v['native_updates']!=0:raise ValueError('Admission gate failed')
        records[name]=dict(path=str(output/name/'verification.json'),sha256=check(output/name/'verification.json'),
                           checks=len(v['checks']))
    for name in ['numerical','formal','statements']:
        p=REPO/f'docs/readout-{name}/verification.json';v=read(p)
        if v['status']!='passed':raise ValueError('Software gate')
        for field in ['tested_sources','module_sources','modules','support_sources']:
            for n,h in v.get(field,{}).items():check(REPO/n,h)
        records[name]=dict(path=str(p),sha256=check(p))
    counts=read(Path(required_input('generated-evidence')) / 'qualification-current.json')['counts']
    if counts!={'numerical_tests':141,'selected_statements':90,'formal_modules':21,'formal_fixtures':8,'rejected_mutations':42}:
        raise ValueError('Current completed counts')
    rendered=read(Path(required_input('generated-evidence')) / 'readout-render-manifest.json')
    check(REPO/'scripts/render_fresh_readout.py',rendered['renderer_sha256'])
    for n,h in rendered['generated'].items():check(Path(required_input('generated-evidence'))/n,h)
    # Bind every generated source asset that will enter the standalone build.
    for p in (Path(required_input('generated-evidence'))).rglob('*'):
        if p.is_file():check(p)
    for name in ['README.md','docs/READOUT_REPRODUCTION.md','docs/PHYSICAL_REPRODUCTION.md',
                 'docs/FINETUNING_REPRODUCTION.md','docs/PROJECTION_REPRODUCTION.md']:
        check(REPO/name)
    if source_inventory()!=source:raise ValueError('Sources changed during release gate')
    write_json(output/'verification.json',dict(status='passed',schema='readout-complete-release-v1',
        study=str(study),retained_release=str(retained),verifier_sha256=sha256(__file__),
        current_sources=source,checked_sha256=checked,records=records,counts=counts,
        inventory=dict(native_scientific_updates=3002528,arithmetic_control_updates=864,
            new_scientific_optimizer_updates=0,new_qualification_and_replay_updates=198,
            new_source_configurations=2048,new_source_chains=256,new_native_observations=36864,
            readout_cells=648,block_map_cells=144,reference_fit_models=24)))
    print('Passed scientific, current-source admission, numerical and formal release gates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--retained-release',required=True)
    p.add_argument('--output',required=True);a=p.parse_args();main(a.study,a.retained_release,a.output)
