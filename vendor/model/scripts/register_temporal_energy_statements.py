#!/usr/bin/env python3
"""Register selected finite temporal-energy types without altering any existing theorem type."""
from companion_paths import configured_path
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
 ('energy_identity','Finite real inner-product norm expansion with ordered pairs.'),
 ('diagonal_append','Additivity of unnormalized squared step energies.'),
 ('cross_append','Cross-energy merge law for two adjacent lists.'),
 ('cross_lower_bound','Unnormalized lower bound from squared-norm positivity.'),
 ('normalized_energy_identity','Algebraic normalized identity; the physical domain has positive energy.'),
 ('normalized_lower_bound','Normalized lower bound with explicit positive energy.')]:
 full='ModelRG.TemporalEnergy.'+name
 if full not in known:r['exports'].append(dict(name=full,module='ModelRG/TemporalEnergy.lean',paper_labels=['prop:temporal-energy'],coverage=coverage))
env=dict(os.environ,ELAN_HOME=configured_path('tools:elan'),PATH=configured_path('tools:elan/bin')+os.pathsep+os.environ['PATH'])
with tempfile.TemporaryDirectory(prefix='population-types-',dir='/tmp') as tmp:
 p=Path(tmp)/'Export.lean';p.write_text((ROOT/'scripts/formal/ExportStatements.lean').read_text()+'\n'+'\n'.join('#export_modelrg '+x['name'] for x in r['exports'])+'\n')
 result=subprocess.run(['lake','env','lean',str(p)],cwd=ROOT,env=env,capture_output=True,text=True,check=True)
actual={a['name']:a for line in result.stdout.splitlines() if line.startswith('STATEMENT_JSON ') for a in [json.loads(line[len('STATEMENT_JSON '):])]}
for row in r['exports']:
 text=actual[row['name']]['type'];h=hashlib.sha256(text.encode()).hexdigest()
 if 'type_sha256' in row and row['type_sha256']!=h:raise ValueError('Retained theorem type changed '+row['name'])
 row.update(elaborated_type=text,type_sha256=h,source_sha256=sha256(ROOT/row['module']))
registry.write_text(json.dumps(r,indent=2)+'\n');print('Registered',len(r['exports']),'exact types')
