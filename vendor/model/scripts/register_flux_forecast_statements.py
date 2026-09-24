#!/usr/bin/env python3
"""Register reviewed finite-flux types without altering any existing theorem type."""
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
for name,coverage,label in [
 ('error_identity','Rational finite-flux subtraction with both nonzero denominators. The absolute-error estimate is written analysis.','prop:finite-flux-forecast'),
 ('energy_composition','Finite scalar energy-coordinate composition with nonzero incoming and intermediate energies.','prop:energy-cocycle'),
 ('energy_associativity','Associative ordered affine energy-coordinate product.','prop:energy-cocycle'),
 ('row_action','Rational row-fraction action of two finite energy maps with nonzero ratios.','prop:energy-cocycle')]:
 full='ModelRG.FluxForecast.'+name
 if full not in known:r['exports'].append(dict(name=full,module='ModelRG/FluxForecast.lean',paper_labels=[label],coverage=coverage))
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
