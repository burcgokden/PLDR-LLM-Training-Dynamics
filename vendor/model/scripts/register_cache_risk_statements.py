#!/usr/bin/env python3
"""Admit the three cache-risk extension types while preserving every retained type."""
from companion_paths import configured_path
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from model_rg.provenance import sha256
ROOT = Path(__file__).resolve().parents[1]
registry = ROOT/'scripts/formal/statement-registry.json'
r = json.loads(registry.read_text())
new = [
 ('ModelRG.CacheRisk.state_displacement_euclidean', 'ModelRG/CacheRisk.lean', ['prop:cache-state-transport'],
  'Finite coordinate-sum displacement with the signed cross term. No population independence or emission sensitivity is formalized.'),
 ('ModelRG.CacheRisk.empirical_state_transport_euclidean', 'ModelRG/CacheRisk.lean', ['prop:cache-state-transport'],
  'Finite Euclidean risk transport with explicit normalized weights and actual coordinatewise means. Statistical sampling and conditional analytic limits remain written results.'),
 ('ModelRG.CacheRisk.empirical_budget_euclidean', 'ModelRG/CacheRisk.lean', ['prop:cache-risk'],
  'Finite Euclidean cache budget with coordinatewise weighted means. Does not formalize conditional Hilbert expectation.'),
 ('ModelRG.CacheRisk.state_displacement', 'ModelRG/CacheRisk.lean', ['prop:cache-state-transport'],
  'Scalar squared-displacement expansion. Summing coordinates proves the finite operator statement; empirical acquisition remains numerical.'),
 ('ModelRG.OptimizerMemory.bias_product_bound', 'ModelRG/OptimizerMemory.lean', ['prop:matched-memory-force'],
  'Only the bias-product inequality. Moment recurrences, Cauchy-Schwarz, exponential limit and uniform envelope remain standalone written analysis.'),
 ('ModelRG.CacheRisk.empirical_budget', 'ModelRG/CacheRisk.lean', ['prop:cache-risk'],
  'Finite weighted scalar cache-risk identity with actual normalized mean. A coordinate-sum bridge covers finite Euclidean operators; independence, Hilbert-space expectation and emission sensitivity remain standalone written results.'),
 ('ModelRG.CacheRisk.cache_sign', 'ModelRG/CacheRisk.lean', ['prop:equivariant-cache'],
  'Fixed weighted scalar averaging commutes with a scalar sign. Native network and optimizer equivariance remain standalone arguments and separate execution checks.'),
 ('ModelRG.constant_stability_bound', 'ModelRG/Reduction.lean', ['prop:constant-stability'],
  'Inductive finite geometric-sum bound for a real error sequence with nonnegative stability and zero initial error. No coupling, native stability estimate or little-o theorem is formalized.')]
known = {x['name'] for x in r['exports']}
items = r['exports']+[dict(name=n,module=m,paper_labels=l,coverage=c) for n,m,l,c in new if n not in known]
env = dict(os.environ, ELAN_HOME=configured_path('tools:elan'),
           PATH=configured_path('tools:elan/bin')+os.pathsep+os.environ['PATH'])
with tempfile.TemporaryDirectory(prefix='cache-risk-types-', dir='/tmp') as tmp:
    p = Path(tmp)/'Exports.lean'
    p.write_text((ROOT/'scripts/formal/ExportStatements.lean').read_text()+'\n'+
                 '\n'.join('#export_modelrg '+x['name'] for x in items)+'\n')
    result = subprocess.run(['lake','env','lean',str(p)], cwd=ROOT, env=env,
                            text=True, capture_output=True, check=True)
actual = {a['name']:a for line in result.stdout.splitlines() if line.startswith('STATEMENT_JSON ')
          for a in [json.loads(line[len('STATEMENT_JSON '):])]}
for item in items:
    a = actual[item['name']]; h = hashlib.sha256(a['type'].encode()).hexdigest()
    if 'type_sha256' in item and h != item['type_sha256']:
        raise ValueError('Retained theorem type changed: '+item['name'])
    item.update(elaborated_type=a['type'], type_sha256=h, source_sha256=sha256(ROOT/item['module']))
r['exports'] = items
registry.write_text(json.dumps(r, indent=2)+'\n')
print('Registered', len(items), 'unchanged or new exact types')
