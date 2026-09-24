#!/usr/bin/env python3
"""Freeze a single-pass family with size-proportional time and corpus resources.

Each width is a separately qualified instance of the same native update
program. Its rates, horizon and nested document population are frozen before
execution. This matches nominal clocks and consumption; it does not assume
that the emitted observables already have a thermodynamic limit.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from model_rg.provenance import sha256, write_json
from prepare_critical_refinement import prepare_refinement
from run_critical_onepass import ROOT, REPO, validate_design


def design_at(specification, heads):
    return dict(name=specification['name']+f'-h{heads}', heads=[heads],
                controls=[2*r/heads for r in specification['rate_ratios']],
                seeds=specification['seeds'], environments=specification['environments'],
                steps=specification['steps_per_head']*heads, shared_seed=specification['shared_seed'])


def prepare_family(destination, specification_path, selection):
    if destination.exists() or not destination.is_relative_to(ROOT) or destination == ROOT:
        raise ValueError('A fresh experiment-workspace destination is required')
    spec = json.loads(specification_path.read_text())
    required = {'name', 'heads', 'rate_ratios', 'seeds', 'environments', 'steps_per_head',
                'corpus_documents_per_head', 'shared_seed'}
    if set(spec) != required or type(spec['steps_per_head']) is not int or type(spec['corpus_documents_per_head']) is not int:
        raise ValueError('The complete integer time/resource family is required')
    if not spec['heads'] or len(spec['heads']) != len(set(spec['heads'])):
        raise ValueError('Widths must be distinct and nonempty')
    if spec['steps_per_head'] <= 0 or spec['corpus_documents_per_head'] <= 0:
        raise ValueError('Time and resource coefficients must be positive')
    if 32*spec['steps_per_head'] > 8*spec['corpus_documents_per_head']:
        raise ValueError('The family would repeat token blocks')
    for n in spec['heads']:
        validate_design(design_at(spec, n))
        if n*spec['corpus_documents_per_head'] > 262144:
            raise ValueError('A nested population exceeds its disjoint master-corpus half')
    destination.mkdir(parents=True)
    shutil.copy2(specification_path, destination/'family.json')
    shutil.copy2(selection, destination/'selection-decision.json')
    master_halves = np.random.default_rng(9152501).permutation(524288).reshape(2, -1)
    members = []
    for n in spec['heads']:
        design = design_at(spec, n)
        design_path = destination/f'design-h{n}.json'
        write_json(design_path, design)
        study = destination/f'h{n}'
        prepare_refinement(study, design_path, selection)
        with np.load(study/'selection.npz') as source:
            selections = {key: source[key].copy() for key in ['probe_rows', 'probes']}
        population_documents = n*spec['corpus_documents_per_head']
        for e in spec['environments']:
            population = master_halves[e, :population_documents]
            local = np.random.default_rng(9152502+e).permutation(population_documents*8)[:32*design['steps']]
            blocks = 8*population[local//8]+local % 8
            selections[f'population_{e}'] = population
            selections[f'blocks_{e}'] = blocks.reshape(-1, 32)
        np.savez_compressed(study/'selection.npz', **selections)
        protocol = json.loads((study/'protocol.json').read_text())
        protocol['selection_sha256'] = sha256(study/'selection.npz')
        protocol['source'].update(population_policy='nested-width-prefix-v1',
            population_documents=population_documents, population_blocks=8*population_documents,
            conditioning='Nested width-proportional document subsets of disjoint halves of one fixed RefinedWeb master corpus. The realized order is fixed within each width; corpus populations are not independent master draws.')
        protocol['joint_family'] = dict(specification=spec, specification_sha256=sha256(specification_path),
            generator_multiplier='g_N=2*r/N', other_rate='0.0006/N',
            terminal_other_clock=0.0006*spec['steps_per_head'],
            terminal_generator_clocks=[0.0006*spec['steps_per_head']*r for r in spec['rate_ratios']],
            consumed_fraction=4*spec['steps_per_head']/spec['corpus_documents_per_head'],
            interpretation='The two nominal rate clocks and consumed fraction match across widths. Jacobians, Adam normalization, update-unit memory and retained-source laws remain part of the dynamics.')
        milestones = set(protocol['observations']['milestone_steps'])
        milestones.update([design['steps']//4, design['steps']//2, design['steps']])
        protocol['observations']['milestone_steps'] = sorted(t for t in milestones if t % 64 == 0)
        protocol['source_sha256']['scripts/prepare_critical_clock_family.py'] = sha256(__file__)
        write_json(study/'protocol.json', protocol)
        target = study/'executed-source/scripts/prepare_critical_clock_family.py'
        shutil.copy2(__file__, target)
        members.append(dict(heads=n, study=str(study), protocol_sha256=sha256(study/'protocol.json'),
                            scientific_paths=len(protocol['jobs']), steps_per_path=design['steps'],
                            population_documents=population_documents))
    write_json(destination/'prepared-family.json', dict(schema='critical-clock-family-v1',
        role='prepared', scientific_updates=0, members=members,
        family_sha256=sha256(destination/'family.json'), selection_sha256=sha256(selection),
        preparer_sha256=sha256(__file__)))
    print(json.dumps({'family': str(destination), 'qualified_members': 0, 'members': len(members)}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', type=Path, required=True)
    p.add_argument('--family', type=Path, required=True)
    p.add_argument('--selection', type=Path, required=True)
    a = p.parse_args()
    prepare_family(a.destination.resolve(), a.family.resolve(), a.selection.resolve())
