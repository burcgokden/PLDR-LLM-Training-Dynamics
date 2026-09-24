#!/usr/bin/env python
"""Freeze a local derivative panel before differentiating native row maps."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908'); a = p.parse_args()
    study = Path(a.root).resolve()/a.study; repo = Path(__file__).resolve().parents[1]
    output = study/'protocols/row-jacobian-selection.json'
    if output.exists(): raise FileExistsError(output)
    selection = study/'protocols/row-input-transport-selection.json'
    verification = study/'verification/row-input-transport.json'
    base = json.loads(selection.read_text()); check = json.loads(verification.read_text())
    if check['status'] != 'passed': raise AssertionError('Verified native row-stage inputs are required')
    inputs = {str(path):sha256(path) for path in [selection, verification]}; cases = []
    for old in base['cases']:
        case = dict(old); case['reference_transport'] = old['name']
        case['name'] = 'row-jacobian-'+old['name'].removeprefix('row-transport-')
        for name in ['manifest.json', 'measurements.npz']:
            path = study/'measurements'/old['name']/name; digest = sha256(path)
            if check['checked_sha256'][str(path)] != digest: raise AssertionError('A verified native row-stage input changed')
            inputs[str(path)] = digest
        cases.append(case)
    sources = ['scripts/measure_row_jacobians.py', 'scripts/prepare_row_jacobians.py',
        'src/model_rg/native.py', 'src/model_rg/training.py', 'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    write_json(output, dict(schema='native-row-jacobian-selection-v1', frozen_at=datetime.now(timezone.utc).isoformat(),
        cases=cases, context_indices=[0,3], head_indices=[0,6,13], row_indices=[0,31,63],
        points_per_layer=18, decoder_layers=5, selected_jacobians=450,
        inputs_sha256=inputs, producer_sources={n:sha256(repo/n) for n in sources},
        selection_scope='All five completed row-stage states; two fixed context indices, three fixed head indices and three fixed row indices. This derivative selection follows the finite secant observations and precedes all local Jacobian outcomes. It is a descriptive mechanism extension, not a held-out prediction of the earlier secants.',
        arithmetic='Exactly lift the saved float32 row-unit parameters and normalized-Gram inputs to float64. Check native float32 full-stage replay separately. Retain all 64 by 64 local Jacobians and singular values.'))
    print('Frozen 450 native row Jacobians', sha256(output), flush=True)


if __name__ == '__main__': main()
