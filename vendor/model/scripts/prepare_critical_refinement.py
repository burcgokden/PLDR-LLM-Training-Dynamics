#!/usr/bin/env python3
"""Freeze an independently selected native refinement and retain every Adam state.

Preparation is completed before qualification or scientific execution. The
native numerical program is the same source-bound single-pass runner. The
selection record identifies the complete search evidence and the hypotheses
selected from it; no field is promoted retrospectively in an executed study.
"""
import argparse
import json
from pathlib import Path
import shutil

from model_rg.provenance import sha256, write_json
from run_critical_onepass import REPO, ROOT, prepare


def prepare_refinement(study, design, selection):
    if not study.is_relative_to(ROOT) or study == ROOT or study.exists():
        raise ValueError('A fresh destination inside the experiment workspace is required')
    decision = json.loads(selection.read_text())
    required = {'search_analyses', 'hypotheses', 'primary_fields', 'held_out_axes', 'decision_rules'}
    if set(decision) != required or any(not decision[key] for key in required):
        raise ValueError('The complete selection and prediction record is required')
    for name, digest in decision['search_analyses'].items():
        path = Path(name)
        data = json.loads(path.read_text())
        if sha256(path) != digest or data.get('status') != 'complete':
            raise ValueError('Selection requires completed, unchanged search analyses')
    declared = json.loads(design.read_text())
    search_designs = [json.loads(Path(name).read_text()).get('design') for name in decision['search_analyses']]
    search_seeds = {seed for d in search_designs if d for seed in d['seeds']}
    if not search_seeds or set(declared['seeds']) & search_seeds:
        raise ValueError('The refinement requires independent initialization seeds')
    prepare(study, design)
    protocol = json.loads((study/'protocol.json').read_text())
    for job in protocol['jobs']:
        job['save_optimizer'] = True
    protocol['checkpoint_policy'] = 'Every final model, full Adam state, random state and consumed block identity. No optimizer state is inferred from a model-only checkpoint.'
    protocol['selection'] = decision
    protocol['selection_record_sha256'] = sha256(selection)
    protocol['source_sha256']['scripts/prepare_critical_refinement.py'] = sha256(__file__)
    write_json(study/'protocol.json', protocol)
    shutil.copy2(design, study/'design.json')
    shutil.copy2(selection, study/'selection-decision.json')
    snapshot = study/'executed-source'
    for name, digest in protocol['source_sha256'].items():
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/name, target)
        if sha256(target) != digest:
            raise ValueError('The preparation source changed while freezing it')
    print(json.dumps({'study': str(study), 'scientific_paths': len(protocol['jobs']),
                      'scientific_updates_planned': sum(j['steps'] for j in protocol['jobs']),
                      'qualification_required': True, 'all_optimizer_states_retained': True}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--design', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    a = p.parse_args()
    prepare_refinement(a.study.resolve(), a.design.resolve(), a.selection.resolve())
