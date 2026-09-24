#!/usr/bin/env python3
"""Register only the finite algebra supporting the consuming-population theorem."""
from companion_paths import legacy_path
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from model_rg.provenance import sha256

ROOT=Path(__file__).resolve().parents[1]
registry=ROOT/'scripts/formal/statement-registry.json';r=json.loads(registry.read_text());known={x['name'] for x in r['exports']}
for name,coverage in [
 ('centered_offdiagonal','Finite scalar centered-population ordered-pair identity. The permutation sampling law and vector covariance interpretation remain in the stand-alone written proof.'),
 ('batch_covariance_collect','Finite real coefficient collection with explicit nonzero batch and population denominators. No probability or physical-clock limit is formalized.')]:
 full='ModelRG.FinitePopulation.'+name
 if full not in known:r['exports'].append(dict(name=full,module='ModelRG/FinitePopulation.lean',paper_labels=['prop:consuming-bridge'],coverage=coverage))
for row in r['exports']:
 if row['name']=='ModelRG.closureDefect_eq':
  row['paper_labels']=['prop:closure-error'];row['coverage']='Finite kernel-matrix telescoping defect only. General metric transport and economical native successor closure are outside this export.'
env=dict(os.environ,ELAN_HOME=legacy_path('/pldr-tools/elan'),PATH=legacy_path('/pldr-tools/elan/bin')+os.pathsep+os.environ['PATH'])
with tempfile.TemporaryDirectory(prefix='population-types-',dir='/tmp') as tmp:
 p=Path(tmp)/'Export.lean';p.write_text((ROOT/'scripts/formal/ExportStatements.lean').read_text()+'\n'+'\n'.join('#export_modelrg '+x['name'] for x in r['exports'])+'\n')
 result=subprocess.run(['lake','env','lean',str(p)],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
actual={a['name']:a for line in result.stdout.splitlines() if line.startswith('STATEMENT_JSON ') for a in [json.loads(line[len('STATEMENT_JSON '):])]}
for row in r['exports']:
 text=actual[row['name']]['type'];h=hashlib.sha256(text.encode()).hexdigest()
 if 'type_sha256' in row and row['type_sha256']!=h:raise ValueError('Retained theorem type changed '+row['name'])
 row.update(elaborated_type=text,type_sha256=h,source_sha256=sha256(ROOT/row['module']))
registry.write_text(json.dumps(r,indent=2)+'\n');print('Registered',len(r['exports']),'exact types')
