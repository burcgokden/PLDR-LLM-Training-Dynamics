#!/usr/bin/env python
"""Combine the complete scheduled and inference evidence without selecting outcomes."""
import argparse
import json
from pathlib import Path

from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


COMPONENTS = {
    'collectives': 'collective-statistics.json',
    'predictions': 'prediction-statistics.json',
    'frozen-clocks': 'frozen-clock-statistics.json',
    'clock-scores': 'clock-score-statistics.json',
    'training-paths': 'temporal-statistics.json',
    'initial-inference-mechanism': 'initial-inference-summary.json',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve()/args.study
    inputs = []

    def load(path):
        path = Path(path)
        inputs.append(path)
        return json.loads(path.read_text())

    def check(path, digest):
        path = Path(path)
        if sha256(path) != digest:
            raise AssertionError('Changed scheduled aggregate input: ' + str(path))
        inputs.append(path)

    training_path = study/'verification/training-raw.json'
    training = load(training_path)
    if training['status'] != 'complete' or training['artifacts'] != 40 or training['native_updates'] != 10485760:
        raise AssertionError('All 40 selected native trajectories must complete and pass reconstruction')
    selection_path = study/'protocols/regime-training-selection.json'
    selection = load(selection_path)
    check(selection_path, training['selection_sha256'])
    if {r['run_id'] for r in training['trajectories']} != {j['run_id'] for j in selection['jobs']}:
        raise AssertionError('A scientific trajectory was omitted or replaced')
    components = {}
    reports = {}
    for name, verifier in COMPONENTS.items():
        parent = study/'analysis'/name
        meta = load(parent/'manifest.json')
        proof_path = study/'verification'/verifier
        proof = load(proof_path)
        if meta['status'] != 'complete' or proof['status'] != 'passed':
            raise AssertionError('A selected analytical component lacks complete verification: ' + name)
        check(parent/'results.json', meta['results_sha256'])
        check(parent/'manifest.json', proof['checked_sha256'][str(parent/'manifest.json')])
        check(parent/'results.json', proof['checked_sha256'][str(parent/'results.json')])
        reports[name] = load(parent/'results.json')
        components[name] = dict(path=str(parent), manifest_sha256=sha256(parent/'manifest.json'),
            results_sha256=sha256(parent/'results.json'), verification=str(proof_path), verification_sha256=sha256(proof_path))
    if len(reports['collectives']['conditions']) != 160 or len(reports['collectives']['paired_arithmetic']) != 80:
        raise AssertionError('The full paired-arithmetic collective panel is required')
    if len(reports['predictions']['states']) != 320 or len(reports['predictions']['conditions']) != 80:
        raise AssertionError('The full scheduled prediction panel is required')
    if len(reports['training-paths']['runs']) != 40 or len(reports['initial-inference-mechanism']['states']) != 5:
        raise AssertionError('The complete temporal and initial inference panels are required')
    observation_selection = load(study/'protocols/regime-observation-selection.json')
    prefix_selection = load(study/'protocols/scheduled-prefix-selection.json')
    task_selection = load(study/'protocols/reasoning-mechanism-selection.json')
    diagnostic_checks = {}
    for label, cases, subfolder, expected in [('observations', observation_selection['cases'], 'observations', 320),
            ('prefix', prefix_selection['cases'], 'prefix', 320), ('reasoning', task_selection['cases'], 'reasoning', 85)]:
        if len(cases) != expected or len({c['name'] for c in cases}) != expected:
            raise AssertionError('A diagnostic selection changed: ' + label)
        records = []
        for case in cases:
            path = study/'verification'/subfolder/(case['name']+'.json')
            proof = load(path)
            if proof['status'] != 'passed' or proof['case'] != case:
                raise AssertionError('A selected diagnostic was not reconstructed: ' + case['name'])
            records.append(dict(case=case, verification=str(path), verification_sha256=sha256(path)))
        diagnostic_checks[label] = records
    initial_checks = {}
    for name in ['prefix-mechanism-initial', 'row-input-transport', 'row-jacobians', 'reasoning-cohort', 'early-prefix-reconciliation']:
        path = study/'verification'/(name+'.json')
        proof = load(path)
        if proof['status'] != 'passed':
            raise AssertionError('A selected inference mechanism check is incomplete: ' + name)
        initial_checks[name] = dict(path=str(path), sha256=sha256(path))
    jacobians = load(study/'verification/row-jacobians.json')
    if (jacobians['states'], jacobians['jacobians']) != (5, 450):
        raise AssertionError('All selected local row-map Jacobians are required')
    reconciliation = load(study/'verification/early-prefix-reconciliation.json')
    if reconciliation['states'] != 6 or reconciliation['additional_independent_training_identities'] != 0:
        raise AssertionError('All early copies must reconcile without adding independent identities')
    gpu_path = study/'verification/gpu-qualification.json'
    gpu = load(gpu_path)
    if gpu['status'] != 'complete' or gpu['qualification_artifacts'] != 6 or not gpu['checkpoint_recovery']['byte_equal']:
        raise AssertionError('The scheduled GPU and resume qualification is required')
    observer_path = study/'launcher-scheduled-observations-recovered.json'
    observer = load(observer_path)
    if observer['status'] != 'complete':
        raise AssertionError('The recovered observation selection has not completed')
    qualified = [r for r in observer['records'] if r['role'] == 'qualification']
    scientific = [r for r in observer['records'] if r['role'] == 'scientific']
    if len(qualified) != 4 or len(scientific) != 320 or any(r['status'] != 'complete' for r in observer['records']):
        raise AssertionError('Observation qualification or scientific coverage changed')
    qualification = []
    for record in qualified:
        path = Path(record['verification'])
        check(path, record['verification_sha256'])
        proof = load(path)
        if proof['status'] != 'passed' or proof['risk']['native_reduction_values_replayed_bytewise'] != 640:
            raise AssertionError('Native loss-reduction qualification is incomplete')
        qualification.append(dict(name=record['name'], risk=proof['risk'], inference=proof['inference'],
                                  verification=str(path), verification_sha256=sha256(path)))
    intervention = study/'qualification/inference-interventions'
    intervention_meta = load(intervention/'manifest.json')
    intervention_results = load(intervention/'results.json')
    if intervention_meta['status'] != 'complete' or len(intervention_results['records']) != 5:
        raise AssertionError('All fixed-operator native intervention qualifications are required')
    check(intervention/'results.json', intervention_meta['results_sha256'])
    for name in ['scheduled-analysis-selection.json', 'scheduled-path-analysis-selection.json',
                 'observation-implementation.json', 'observation-recovered-implementation.json',
                 'row-jacobian-selection.json', 'reasoning-mechanism-selection.json']:
        inputs.append(study/'protocols'/name)
    output = study/'analysis/complete'
    output.mkdir(parents=True, exist_ok=False)
    bind_run(output, list(dict.fromkeys(inputs)), vars(args))
    result = dict(status='complete', schema='complete-scheduled-inference-evidence-v1', components=components,
        scientific_trajectories=40, scientific_updates=10485760, maximum_completed_horizon=262144,
        training=training['trajectories'], training_verification=dict(path=str(training_path), sha256=sha256(training_path)),
        counts=dict(cpu_states=320, risk_states=320, generation_states=80, scheduled_prefix_states=320,
                    task_states=85, tasks_per_state=320, task_modes=4, initial_prefix_states=5,
                    initial_row_transport_states=5, local_row_jacobians=450, early_checkpoint_copies=6),
        collective_conditions=reports['collectives']['conditions'], paired_arithmetic=reports['collectives']['paired_arithmetic'],
        prediction_states=reports['predictions']['states'], prediction_conditions=reports['predictions']['conditions'],
        paired_recipe_contrasts=reports['predictions']['paired_recipe_contrasts'],
        frozen_clocks=reports['frozen-clocks'], clock_scores=reports['clock-scores'], temporal_paths=reports['training-paths']['runs'],
        initial_inference=reports['initial-inference-mechanism']['states'], local_row_jacobian_summaries=jacobians['summaries'],
        diagnostic_checks=diagnostic_checks, mechanism_checks=initial_checks,
        gpu_qualification=dict(path=str(gpu_path), sha256=sha256(gpu_path), results=gpu),
        native_reduction_qualification=qualification, native_intervention_qualification=intervention_results['records'],
        observation_controller=dict(path=str(observer_path), sha256=sha256(observer_path)),
        scope='All completed scheduled native training, selected frozen observations, task interventions and local input-transport results are retained. Checkpoints, task items, contexts, local Jacobians and sealed copies do not increase the independent training-identity count.',
        inference='Reference near/subcritical recipe names identify inherited hyperparameter settings, not measured physical phases in the adapted finite data law. Proper predictive competence, row/input contraction, critical training response and endogenous approach to a critical surface are distinct properties. These completed finite observations do not alone establish thermodynamic critical exponents or self-organized criticality.')
    write_json(output/'results.json', result)
    write_json(output/'manifest.json', dict(status='complete', components=len(components), scientific_trajectories=40,
        scientific_updates=10485760, binding_sha256=sha256(output/'binding.json'), results_sha256=sha256(output/'results.json')))
    print('Complete scheduled and inference evidence:', 40, 'scientific trajectories and', 85, 'task states', flush=True)


if __name__ == '__main__':
    main()
