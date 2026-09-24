#!/usr/bin/env python
"""Discover all production Lean modules and run an axiom gate plus mutations."""
from companion_paths import legacy_path
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from model_rg.provenance import sha256, write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True)
    ap.add_argument('--elan-bin',default=legacy_path('/pldr-tools/elan/bin'))
    a=ap.parse_args();repo=Path(__file__).resolve().parents[1];out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PATH=a.elan_bin+os.pathsep+os.environ['PATH'],
             ELAN_HOME=legacy_path('/pldr-tools/elan'))
    modules=sorted(repo.glob('ModelRG/**/*.lean'))
    imports='\n'.join('import '+'.'.join(p.relative_to(repo).with_suffix('').parts) for p in modules)+'\n'
    # Discovery closes the hole left by an environment-only check of one import.
    if (repo/'ModelRG.lean').read_text()!=imports:
        (repo/'ModelRG.lean').write_text(imports)
    records=[]
    def run(name,command,expected,cwd=repo):
        r=subprocess.run(command,cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/(name+'.log')).write_text(r.stdout)
        passed=r.returncode==0
        records.append(dict(name=name,returncode=r.returncode,expected_pass=expected,
                            expectation_met=passed==expected,log_sha256=sha256(out/(name+'.log'))))
        if passed != expected:raise RuntimeError(f'Unexpected formal outcome: {name}; see {out}')
    run('build',['lake','build'],True)
    gate=(repo/'scripts/formal/Gate.lean').read_text()
    with tempfile.TemporaryDirectory(prefix='modelrg-formal-',dir='/tmp') as tmp:
        tmp=Path(tmp)
        fixtures={
            'baseline':('',True),
            'sorry':('namespace ModelRG\ntheorem unfinished : True := by sorry\nend ModelRG\n',False),
            'unused_axiom':('namespace ModelRG\naxiom unused_bad : False\nend ModelRG\n',False),
            'false_identity':('namespace ModelRG\ntheorem wrong (x : ℝ) : x+1=x := by ring\nend ModelRG\n',False)}
        for name,(fixture,expected) in fixtures.items():
            p=tmp/(name+'.lean');p.write_text(gate.replace('#audit_modelrg\n',fixture+'\n#audit_modelrg\n'))
            run(name,['lake','env','lean',str(p)],expected)
        # Discover a production-owned module with an axiom outside its namespace.
        scratch=tmp/'owned-project'
        shutil.copytree(repo/'ModelRG',scratch/'ModelRG')
        for filename in ['ModelRG.lean','lakefile.toml','lake-manifest.json','lean-toolchain']:
            shutil.copy2(repo/filename,scratch/filename)
        (scratch/'.lake').mkdir()
        (scratch/'.lake/packages').symlink_to(repo/'.lake/packages',target_is_directory=True)
        shutil.copytree(repo/'.lake/build',scratch/'.lake/build')
        mutation=scratch/'ModelRG/OwnedGateMutation.lean'
        if mutation.exists():raise FileExistsError(mutation)
        try:
            mutation.write_text('axiom outside_namespace_owned_mutation : False\n')
            run('owned_module_build',['lake','env','lean','-o',str(tmp/'OwnedGateMutation.olean'),str(mutation)],True,cwd=scratch)
            # Build the owned module with Lake to place it on the import path.
            run('owned_lake_build',['lake','build','ModelRG.OwnedGateMutation'],True,cwd=scratch)
            p=tmp/'owned.lean';p.write_text('import ModelRG.OwnedGateMutation\n'+gate)
            run('owned_unused_axiom',['lake','env','lean',str(p)],False,cwd=scratch)
        finally:
            mutation.unlink()
    explicit=sum(len(re.findall(r'^theorem\s',p.read_text(),re.M)) for p in modules)
    write_json(out/'verification.json',dict(status='passed',explicit_theorems=explicit,checker_sha256=sha256(__file__),
        ownership_mutation_scope='Isolated temporary project; production source files are not mutated by the ownership fixture.',
        module_sources={str(p.relative_to(repo)):sha256(p) for p in modules},
        gate_sha256=sha256(repo/'scripts/formal/Gate.lean'),toolchain=(repo/'lean-toolchain').read_text().strip(),
        lake_manifest_sha256=sha256(repo/'lake-manifest.json'),checks=records))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
