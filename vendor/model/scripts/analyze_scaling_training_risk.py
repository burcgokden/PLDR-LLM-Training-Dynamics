#!/usr/bin/env python
"""Keep frozen training-law risk, held-out risk and online loss distinct."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from analyze_scaling_paths import paired_summary
from analyze_size_time import empirical_weights, whole_seed_statistics
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


FIELDS = ['training_nll', 'heldout_nll', 'training_entropy', 'heldout_entropy']


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args();study = Path(a.root)/a.study
    ledger = study/'launcher-training-risk.json'
    while not (ledger.exists() and json.loads(ledger.read_text())['status'] == 'complete'):
        if ledger.exists() and 'failure' in json.loads(ledger.read_text())['status']:
            raise RuntimeError('Resolve the selected training-risk measurement')
        if not a.wait:raise RuntimeError('All selected training-risk measurements are required')
        time.sleep(30)
    protocol = study/'protocols/training-risk-selection.json';spec = json.loads(protocol.read_text())
    inputs = [ledger, protocol, Path(spec['cohort']), Path(spec['cohort_manifest'])]
    groups = defaultdict(dict);states = []
    for case in spec['cases']:
        folder = study/'measurements'/case['name'];m = json.loads((folder/'manifest.json').read_text())
        r = json.loads((folder/'results.json').read_text())
        if m['status'] != 'complete' or m['case'] != case or r['case'] != case:
            raise AssertionError('A selected risk state is incomplete or changed')
        for name, key in [('measurements.npz','raw_sha256'),('results.json','results_sha256')]:
            if sha256(folder/name) != m[key]:raise AssertionError('Changed risk observations')
        with np.load(folder/'measurements.npz') as z:
            q = {name:z[name].astype(float) for name in FIELDS}
        q['online_mean_loss'] = r['online_window']['preupdate_mean_loss']
        groups[case['heads'], case['step']][case['seed']] = q
        states.append(dict(case=case, manifest_sha256=sha256(folder/'manifest.json'),
            heldout_reference_maximum_error=r['heldout_reference_maximum_error'],
            heldout_reference_bitwise_equal=r['heldout_reference_bitwise_equal']))
        inputs.extend([folder/'manifest.json',folder/'results.json',folder/'measurements.npz'])
    if len(states) != 32 or len(groups) != 8:raise AssertionError('The complete risk design changed')
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False);bind_run(out, inputs, vars(a))
    records, paired, raw, data = [], [], {}, {}
    for (n,t), values in sorted(groups.items()):
        seeds = list(range(640101,640105))
        if sorted(values) != seeds:raise AssertionError('The matched risk identities changed')
        fields = {name:np.stack([values[s][name] for s in seeds]) for name in FIELDS}
        data[n,t] = fields
        row = dict(heads=n, step=t, seeds=seeds, observables={})
        for name,q in fields.items():
            row['observables'][name], raw[f'h{n}_t{t}_{name}_bootstrap'] = whole_seed_statistics(q,n)
        gap = fields['heldout_nll'].mean(1)-fields['training_nll'].mean(1)
        row['heldout_minus_training'] = dict(mean=float(gap.mean()), cohort_means=gap.tolist(),
            empirical_percentiles=np.quantile(empirical_weights(4)@gap/4,[.025,.975]).tolist())
        row['online_mean_loss'] = float(np.mean([values[s]['online_mean_loss'] for s in seeds]))
        records.append(row)
    for n in [4,14]:
        for early,late in zip([32768,65536,98304],[65536,98304,131072]):
            row = dict(heads=n, early=early, late=late, observables={})
            for name in FIELDS:
                row['observables'][name], raw[f'h{n}_t{early}_{late}_{name}_paired_bootstrap'] = paired_summary(data[n,early][name],data[n,late][name],n)
            paired.append(row)
    result = dict(schema='conditional-training-risk-analysis-v1', status='complete', states=states,
        conditions=records, paired_horizons=paired, cohort_contexts=dict(training=1024,heldout=512),
        maximum_reference_error=max(r['heldout_reference_maximum_error'] for r in states),
        reference_bitwise_equal_states=sum(r['heldout_reference_bitwise_equal'] for r in states),
        online_window=spec['online_window'],
        scope='Whole-initialization empirical statistics conditional on the frozen1024 training-law crops and512 held-out crops. The same four multiplicities are retained across horizons and both cohorts. These ranges have no guaranteed population coverage and do not integrate uncertainty in the two context cohorts. Online means average preceding8192 native preupdate minibatch losses on an evolving path; they are not frozen-state risks or independent temporal replicates.',
        selection_scope=spec['scope'])
    write_json(out/'results.json',result);np.savez_compressed(out/'measurements.npz',**raw)
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),states=len(states)))
    print('Complete training-risk analysis:',len(states),'states',flush=True)


if __name__ == '__main__':main()
