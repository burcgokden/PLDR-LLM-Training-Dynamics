#!/usr/bin/env python
"""Bind the complete executed size-time, noise, environment and prediction evidence."""
import argparse
import json
from pathlib import Path
import time

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


COMPONENTS = ['size-time', 'collectives', 'paired-paths', 'conditional-noise', 'forcing-geometry',
              'frozen-forecasts', 'forecast-scores', 'frozen-variance', 'variance-scores',
              'environment-factorial', 'shared-updates', 'matched-forcing', 'visible-noise', 'clock-observation-sensitivity', 'source-basis', 'native-step', 'common-geometry', 'training-risk']


def compact_observable(value):
    return {key:value[key] for key in ['mean', 'susceptibility', 'empirical_percentiles', 'jackknife_se',
            'independent_seeds', 'contexts', 'cohort_means'] if key in value}


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--output', required=True)
    p.add_argument('--wait', action='store_true');a = p.parse_args();root = Path(a.root);study = root/a.study
    components = {name:study/'analysis'/name for name in COMPONENTS}
    while not all((folder/'manifest.json').exists() for folder in components.values()):
        if not a.wait:raise RuntimeError('The complete analysis requires every selected component')
        time.sleep(30)
    clarification = study/'protocols/source-basis-interpretation.json'
    inputs, reports, component_bindings = [clarification], {}, {}
    for name, folder in components.items():
        m = json.loads((folder/'manifest.json').read_text())
        if m['status'] != 'complete':raise AssertionError('A selected analysis did not complete: '+name)
        if sha256(folder/'results.json') != m['results_sha256']:raise AssertionError('Changed analysis result')
        reports[name] = json.loads((folder/'results.json').read_text())
        component_bindings[name] = dict(path=str(folder), manifest_sha256=sha256(folder/'manifest.json'), results_sha256=m['results_sha256'])
        inputs.extend([folder/'manifest.json', folder/'results.json'])
    protocols = ['main-size-time-execution', 'clock-holdout', 'diffusive-size-map',
                 'environment-factorial', 'zero-horizon', 'extended-horizon']
    selected = {}
    for name in protocols:
        path = study/'protocols'/f'{name}.json';spec = json.loads(path.read_text());inputs.append(path)
        for job in spec['jobs']:
            if job['run_id'] in selected:raise AssertionError('Duplicate selected trajectory')
            selected[job['run_id']] = name
    selected['long-h4-g1-s640101'] = 'long-horizon'
    inputs.append(study/'protocols/long-horizon.json')
    training = []
    for name, protocol in selected.items():
        path = study/'runs'/name/'manifest.json';m = json.loads(path.read_text());inputs.append(path)
        if m['status'] != 'complete':raise AssertionError('A selected training outcome is incomplete')
        c = m['arguments']
        training.append(dict(run_id=name, protocol=protocol, heads=c['heads'], seed=c['seed'],
            shared_seed=c['shared_seed'], stream_seed=c['stream_seed'], multiplier=c['multiplier'],
            start_step=m['start_step'], completed_step=m['completed_step'], additional_updates=m['completed_step']-m['start_step'],
            manifest_sha256=sha256(path)))
    if len(training) != 96 or sum(r['additional_updates'] for r in training) != 2154496:
        raise AssertionError('Executed training inventory differs from the frozen complete design')
    spectator = []
    spec_path = study/'protocols/spectator-control.json';spec = json.loads(spec_path.read_text());inputs.append(spec_path)
    for case in spec['cases']:
        folder = study/'measurements'/case['name'];m = json.loads((folder/'manifest.json').read_text())
        if m['status'] != 'complete' or sha256(folder/'results.json') != m['results_sha256']:
            raise AssertionError('A selected spectator control is incomplete or changed')
        spectator.append(json.loads((folder/'results.json').read_text()))
        inputs.extend([folder/'manifest.json', folder/'results.json'])
    if len(spectator) != 5 or len(reports['conditional-noise']['states']) != 24:
        raise AssertionError('Conditional diagnostic design is incomplete')
    if len(reports['forecast-scores']['forecasts']) != 54 or len(reports['forecast-scores']['joint_forecasts']) != 3:
        raise AssertionError('A frozen clock prediction was omitted')
    if len(reports['variance-scores']['forecasts']) != 144 or len(reports['shared-updates']['runs']) != 60:
        raise AssertionError('A variance prediction or passive trajectory was omitted')
    if len(reports['source-basis']['states']) != 16:raise AssertionError('Expanded source validation is incomplete')
    if len(reports['native-step']['states']) != 16 or reports['native-step']['conditional_draws'] != 512:
        raise AssertionError('Complete native conditional step selection is incomplete')
    if len(reports['training-risk']['states']) != 32 or len(reports['training-risk']['conditions']) != 8:
        raise AssertionError('Complete frozen training-risk selection is required')
    if len(reports['environment-factorial']['conditions']) != 2:
        raise AssertionError('Finite crossed environments incomplete')
    out = Path(a.output);out.mkdir(parents=True, exist_ok=False);bind_run(out, inputs, vars(a))
    compact = []
    for r in reports['collectives']['conditions']:
        row = {k:v for k,v in r.items() if k not in ['observables', 'sources']}
        row['observables'] = {k:compact_observable(v) for k,v in r['observables'].items()}
        compact.append(row)
    result = dict(schema='complete-native-scaling-evidence-v1', status='complete', components=component_bindings,
        training=training, training_artifacts=len(training), additional_updates=sum(r['additional_updates'] for r in training),
        maximum_completed_horizon=max(r['completed_step'] for r in training),
        cohort=dict(training_rows=3072, training_crop_offsets=449, full_batch=32, input_length=64,
            supervised_targets_per_crop=1, evaluation_contexts=512, calibration_matrix_contexts=64,
            dense_contexts=16, initialization_unit='One nonshared parameter initialization within the specified shared initial state and entire batch history.'),
        collectives=compact, paired_arithmetic=reports['collectives']['paired_arithmetic'],
        paired_horizons=reports['paired-paths']['paired_horizons'],
        observation_batch_shape=reports['paired-paths']['batch_shape_comparisons'],
        clock_fits=reports['frozen-forecasts']['clock_fits'], width_clock_fits=reports['frozen-forecasts']['width_clock_fits'],
        clock_identifiability=reports['forecast-scores']['interval_identifiability'],
        clock_scores=reports['forecast-scores']['forecasts'], joint_clock_scores=reports['forecast-scores']['joint_forecasts'],
        variance_predictions=reports['frozen-variance']['forecasts'], variance_scores=reports['variance-scores']['forecasts'],
        variance_aggregate=reports['variance-scores']['aggregate'],
        conditional_noise=reports['conditional-noise']['conditions'], forcing_geometry=reports['forcing-geometry']['states'],
        matched_forcing=reports['matched-forcing']['states'], visible_noise=reports['visible-noise'],
        source_basis=reports['source-basis'], native_step=reports['native-step'], common_geometry=reports['common-geometry'],
        training_risk=reports['training-risk'],
        interpretation_bindings={str(clarification):sha256(clarification)},
        clock_observation_sensitivity=reports['clock-observation-sensitivity'],
        finite_environment_factorial=reports['environment-factorial']['conditions'], spectator_controls=spectator,
        temporal_paths=reports['shared-updates']['runs'],
        scope='Completed conditional finite-size and finite-time laws, with all frozen predictions and all selected outcomes retained. The inventory counts training artifacts, including continuations, separately from independent initializations and frozen-state diagnostics. No native critical surface, thermodynamic critical exponent or self-organized critical attraction is inferred solely from these finite diagnostics.')
    write_json(out/'results.json', result)
    write_json(out/'manifest.json', dict(status='complete', binding_sha256=sha256(out/'binding.json'),
        results_sha256=sha256(out/'results.json'), training_artifacts=len(training), components=len(components)))
    print('Complete scaling evidence assembled:', len(training), 'training artifacts,', result['additional_updates'], 'additional updates', flush=True)


if __name__ == '__main__':main()
