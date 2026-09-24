#!/usr/bin/env python3
"""Register exact finite row-energy statements and manuscript correspondence."""
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
 ('row_ratio_identity','Scalar rational energy identity with both nonzero denominators.','prop:collective-clock'),
 ('derivative_defect_identity','Rational normalization remainder only; no derivative or norm estimate is formalized.','lem:finite-defect'),
 ('center_add','Actual finite row-centering coordinate map is linear.','prop:collective-clock'),
 ('two_increment_cross','Finite matrix energy retains the cross-increment coordinate sum.','prop:collective-clock'),
 ('matrix_row_identity','Finite centered-matrix ratio using actual row means, Frobenius energies and cross sums.','prop:collective-clock'),
 ('weighted_increment','Fixed-weight finite increment identity; bound and positivity assumptions remain written analysis.','prop:collective-clock'),
 ('finite_telescope','Finite chronological row-increment telescope with its initial and final endpoints.','prop:collective-clock')]:
 full='ModelRG.RowFlux.'+name
 if full not in known:r['exports'].append(dict(name=full,module='ModelRG/RowFlux.lean',paper_labels=[label],coverage=coverage))
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
