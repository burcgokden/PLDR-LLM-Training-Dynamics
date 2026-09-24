#!/usr/bin/env python3
"""Freshly compile every owned module and audit its transitive axioms.

The optional dependency cache is read only. All owned outputs are rebuilt
in this repository, with no use of source-project owned .olean files.
For a portable build, first run `lake build` and omit --dependency-cache.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ['PldrTrainingDynamics','PldrLlmCurvatureSandpile','RowRGMap','ModelRG']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dependency-cache',type=Path,default=ROOT/'.lake/packages')
    ap.add_argument('--lean',default='lean')
    ap.add_argument('--jobs',type=int,default=4)
    a=ap.parse_args()
    lean=shutil.which(a.lean) or a.lean
    (ROOT/'validation').mkdir(exist_ok=True)
    out=ROOT/'build/lean'
    out.mkdir(parents=True,exist_ok=True)
    libs=[p for p in a.dependency_cache.glob('*/.lake/build/lib/lean') if p.is_dir()]
    if not any((p/'Mathlib.olean').exists() for p in libs):
        raise SystemExit('A compiled pinned mathlib dependency cache is required; use lake build first.')
    env=dict(os.environ,LEAN_PATH=os.pathsep.join(map(str,[out,*libs])))
    modules={}
    for base,prefix in [(ROOT,PREFIXES[0]),(ROOT/'vendor/row',PREFIXES[1]),
                        (ROOT/'vendor/rg',PREFIXES[2]),(ROOT/'vendor/model',PREFIXES[3])]:
        for p in [base/(prefix+'.lean'),*sorted((base/prefix).rglob('*.lean'))]:
            module='.'.join(p.relative_to(base).with_suffix('').parts)
            modules[module]=(p,base)
    # Every owned module is compiled, including files not imported by an old root.
    dep={}
    for name,(p,base) in modules.items():
        imports=re.findall(r'^import\s+([\w.]+)',p.read_text(),re.M)
        dep[name]=set(imports)&set(modules)
    records=[]; done=set(); start=time.monotonic()
    def compile_one(name):
        p,base=modules[name]
        target=out/Path(*name.split('.')).with_suffix('.olean')
        target.parent.mkdir(parents=True,exist_ok=True)
        result=subprocess.run([lean,'-o',str(target),str(p)],cwd=base,env=env,
                              text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(name+'\n'+result.stdout)
        return {'module':name,'source':str(p.relative_to(ROOT)),
                'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                'output':result.stdout,'returncode':result.returncode}
    while len(done)<len(modules):
        ready=sorted(n for n in modules if n not in done and dep[n]<=done)
        if not ready: raise RuntimeError('Cyclic or unresolved owned imports')
        with ThreadPoolExecutor(max_workers=a.jobs) as executor:
            for future in as_completed({executor.submit(compile_one,n):n for n in ready}):
                record=future.result();records.append(record);done.add(record['module'])
        print(f'Compiled {len(done)}/{len(modules)} owned modules',flush=True)
    imports='\n'.join('import '+n for n in sorted(modules))
    ownership=' || '.join('(`'+p+').isPrefixOf n' for p in PREFIXES)
    gate=imports+'\n'+r'''
import Lean.Util.CollectAxioms
open Lean Elab Command
elab "#audit_monograph" : command => do
  let env ← getEnv
  let mut count : Nat := 0
  let mut theoremCount : Nat := 0
  let mut names : Array Name := #[]
  for i in [:env.header.moduleNames.size] do
    let n := env.header.moduleNames[i]!
    if OWNED then
      names := names ++ env.header.moduleData[i]!.constNames
  let locals := env.checked.get.constants.foldStage2 (fun ns n _ =>
    if OWNED then ns.push n else ns) (#[] : Array Name)
  names := (names ++ locals).toList.eraseDups.toArray
  for name in names do
    if let some info := env.find? name then
      count := count + 1
      match info with
      | .axiomInfo _ => throwError "Untrusted owned axiom: {name}"
      | .thmInfo _ => theoremCount := theoremCount + 1
      | _ => pure ()
      for ax in (← Lean.collectAxioms name) do
        unless ax == ``propext || ax == ``Classical.choice || ax == ``Quot.sound do
          throwError "Declaration {name} depends on untrusted axiom {ax}"
  if count == 0 then throwError "Empty declaration inventory"
  logInfo m!"AXIOM_GATE_PASS declarations={count} theorem_declarations={theoremCount}"
#audit_monograph
'''.replace('OWNED',ownership)
    gate_path=out/'Audit.lean';gate_path.write_text(gate)
    result=subprocess.run([lean,str(gate_path)],cwd=ROOT,env=env,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/'validation/lean-axioms.log').write_text(result.stdout)
    if result.returncode: raise RuntimeError(result.stdout)
    # Prove the gate rejects an unfinished proof and an owned unused axiom.
    controls=[]
    for name,fixture in [('sorry','namespace PldrTrainingDynamics\ntheorem unfinished : True := by sorry\nend PldrTrainingDynamics\n'),
                         ('unused_axiom','namespace PldrTrainingDynamics\naxiom unsupported : False\nend PldrTrainingDynamics\n')]:
        p=out/(name+'.lean');p.write_text(gate.replace('\n#audit_monograph\n','\n'+fixture+'\n#audit_monograph\n'))
        r=subprocess.run([lean,str(p)],cwd=ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        controls.append({'name':name,'rejected':r.returncode!=0,'output':r.stdout})
        if r.returncode==0: raise RuntimeError('Gate accepted '+name)
    correspondence=json.loads((ROOT/'provenance/statement-manifest.json').read_text())['formal_mappings']
    declarations=sorted({name for row in correspondence for name in row.get('declarations',[])} | {name for row in correspondence for clause in row.get('clauses') or [] for name in clause.get('declarations',[])})
    check_path=out/'Correspondence.lean'
    check_path.write_text(imports+'\nset_option pp.universes true\nset_option pp.explicit true\n'+
                          '\n'.join('#check @'+name for name in declarations)+'\n')
    resolved=subprocess.run([lean,str(check_path)],cwd=ROOT,env=env,text=True,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (ROOT/'validation/lean-correspondence.log').write_text(resolved.stdout)
    if resolved.returncode:raise RuntimeError(resolved.stdout)
    report={'status':'passed','fresh_owned_modules':len(modules),'elapsed_seconds':time.monotonic()-start,
            'toolchain':subprocess.check_output([lean,'--version'],text=True).strip(),
            'dependency_mode':'read-only compiled pinned dependency cache',
            'axiom_gate':result.stdout,'resolved_correspondence_declarations':declarations,
            'correspondence_types_sha256':hashlib.sha256(resolved.stdout.encode()).hexdigest(),
            'correspondence_manifest_sha256':hashlib.sha256((ROOT/'provenance/statement-manifest.json').read_bytes()).hexdigest(),
            'negative_controls':controls,'modules':sorted(records,key=lambda r:r['module'])}
    (ROOT/'validation/lean.json').write_text(json.dumps(report,indent=2)+'\n')
    print(result.stdout,flush=True)

if __name__=='__main__':main()
