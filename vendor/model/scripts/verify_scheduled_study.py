#!/usr/bin/env python
"""Require all completed constant, scheduled and inference evidence for release."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json
from model_rg.evidence_progress import HashProgress, SourceArchive


COMPONENTS = {
    'collectives': ('collective-statistics.json', 'verify_scheduled_collectives.py'),
    'predictions': ('prediction-statistics.json', 'verify_scheduled_predictions.py'),
    'frozen-clocks': ('frozen-clock-statistics.json', 'verify_scheduled_frozen_clocks.py'),
    'clock-scores': ('clock-score-statistics.json', 'verify_scheduled_clock_scores.py'),
    'training-paths': ('temporal-statistics.json', 'verify_scheduled_paths.py'),
    'initial-inference-mechanism': ('initial-inference-summary.json', 'verify_initial_inference_summary.py'),
}


class Evidence:
    def __init__(self, repo, *, progress=None, progress_path=None, source_archive=None):
        self.repo = Path(repo).resolve(); self.cache = {}; self.checked = {}
        self.progress = progress or (HashProgress(progress_path) if progress_path else None)
        self.archive = SourceArchive(source_archive, self.repo) if source_archive else None

    def check(self, path, expected=None):
        path = Path(path).resolve(); st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if key not in self.cache:
            self.cache[key] = self.progress.digest(path) if self.progress else sha256(path)
        elif self.progress:
            self.progress.cache_hit()
        digest = self.cache[key]
        if expected is not None and digest != expected:
            if self.archive and path.is_relative_to(self.repo):
                target = self.archive.resolve(path, expected)
                # Archive contents are hashed through the same cache and progress path.
                return self.check(target, expected)
            raise AssertionError('Changed complete evidence: '+str(path))
        self.checked[str(path)] = digest
        return digest

    def finish(self, status='passed'):
        if self.progress:self.progress.finish(status)

    def load(self, path):
        self.check(path); return json.loads(Path(path).read_text())

    def record(self, path):
        return dict(path=str(Path(path).resolve()), sha256=self.check(path))

    def proof(self, path, checker=None, status='passed', schema=None):
        if self.progress:self.progress.boundary(path)
        result = self.load(path)
        if result['status'] != status or (schema and result['schema'] != schema):
            raise AssertionError('A complete passing reconstruction is required: '+str(path))
        if checker and result.get('verifier_sha256') != self.check(self.repo/'scripts'/checker):
            raise AssertionError('A reconstruction predates its current verifier: '+str(path))
        for key in ['verifier_sources', 'qualified_sources']:
            for name, digest in result.get(key, {}).items(): self.check(self.repo/name, digest)
        for key in ['checked_sha256', 'verified_files']:
            for name, digest in result.get(key, {}).items(): self.check(name, digest)
        return result

    def artifact(self, folder):
        folder = Path(folder); meta = self.load(folder/'manifest.json')
        if meta['status'] != 'complete': raise AssertionError('Incomplete selected artifact: '+str(folder))
        for name, key in [('binding.json', 'binding_sha256'), ('results.json', 'results_sha256'),
                          ('measurements.npz', 'raw_sha256'), ('prediction-index.json', 'prediction_index_sha256')]:
            if key in meta: self.check(folder/name, meta[key])
        binding = self.load(folder/'binding.json')
        for name, digest in binding['inputs'].items(): self.check(name, digest)
        for name, digest in binding['source_files'].items(): self.check(folder/'source'/name, digest)
        for key in ['figures', 'logit_sidecars']:
            for name, digest in meta.get(key, {}).items(): self.check(folder/name, digest)
        return meta


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True)
    parser.add_argument('--study', default='scheduled-training-20260908')
    parser.add_argument('--scaling-verification', required=True); parser.add_argument('--producer-registry', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]; root = Path(args.root).resolve(); study = root/args.study
    if Path(args.output).exists(): raise FileExistsError(args.output)
    # Refuse an unfinished selected experiment before performing expensive rehashes.
    complete_path = study/'analysis/complete/results.json'
    if not complete_path.exists(): raise AssertionError('The complete forty-trajectory scheduled analysis is still required')
    files = Evidence(repo, progress_path=str(args.output)+'.progress.json')
    sources = {str(p.relative_to(repo)): sha256(p) for base in [repo/'scripts', repo/'src'] for p in base.rglob('*.py')}
    inherited = files.proof(args.scaling_verification, 'verify_scaling_study.py', schema='scaling-complete-evidence-v1')
    if (inherited['required_scientific_runs'], inherited['training_artifacts'], inherited['additional_updates']) != (512, 96, 2154496):
        raise AssertionError('The complete inherited scientific inventory is required')
    for record in inherited['inherited_verifications'].values(): files.check(record['path'], record['sha256'])
    for key in ['raw_verification', 'statistical_verification', 'native_step_verification', 'training_risk_verification',
                'numerical_verification', 'formal_statement_verification', 'formal_axiom_verification']:
        record = inherited[key]; files.check(record['path'], record['sha256'])
    if inherited['numerical_verification']['tests_passed'] != 49 or inherited['formal_statement_verification']['registered_statements'] != 53:
        raise AssertionError('The selected numerical and formal scope changed')
    files.check(inherited['observation_analysis'], inherited['observation_analysis_sha256'])
    files.check(inherited['scaling_analysis'], inherited['scaling_analysis_sha256'])
    registry = files.load(args.producer_registry)
    files.check(args.producer_registry, inherited['producer_registry']['sha256'])
    if registry['status'] != 'complete' or args.study not in registry['studies']:
        raise AssertionError('The producer registry must include the completed scheduled study')
    for name, digest in registry['current_sources'].items(): files.check(repo/name, digest)
    if set(registry['current_sources']) != set(sources): raise AssertionError('The current source inventory is incomplete')
    artifacts = {r['artifact']: r for r in registry['artifacts']}

    def registered(folder):
        folder = Path(folder); name = str(folder.relative_to(root))
        if name not in artifacts: raise AssertionError('A selected artifact lacks a preserved producer: '+name)
        files.check(folder/'binding.json', artifacts[name]['binding_sha256']); files.artifact(folder)

    training_selection = files.load(study/'protocols/regime-training-selection.json')
    jobs = training_selection['jobs']; recipes = {'controlled', 'reference1', 'reference2', 'subcritical1', 'subcritical2'}
    expected = {(recipe, heads, seed) for recipe in recipes for heads in [4, 14] for seed in range(640101, 640105)}
    if len(jobs) != 40 or {(j['recipe'], j['heads'], j['seed']) for j in jobs} != expected:
        raise AssertionError('A selected recipe, size or initialization was omitted')
    training_path = study/'verification/training-raw.json'
    training = files.proof(training_path, status='complete', schema='scheduled-raw-reconstruction-v1')
    files.check(study/'protocols/regime-training-selection.json', training['selection_sha256'])
    if (training['artifacts'], training['native_updates']) != (40, 10485760): raise AssertionError('Scheduled training is incomplete')
    trajectories = {r['run_id']: r for r in training['trajectories']}
    if set(trajectories) != {j['run_id'] for j in jobs}: raise AssertionError('Training identities differ from the selection')
    for job in jobs:
        row = trajectories[job['run_id']]
        if (row['start'], row['stop'], row['updates'], job['steps']) != (0, 262144, 262144, 262144):
            raise AssertionError('A selected native horizon is incomplete')
        for key in ['heads', 'seed', 'recipe', 'shared_seed', 'stream_seed']:
            if row[key] != job[key]: raise AssertionError('A native trajectory changed condition')
        registered(study/'runs'/job['run_id'])
    if len(training['individual_reconstructions']) != 40: raise AssertionError('A per-trajectory reconstruction is missing')
    for record in training['individual_reconstructions']:
        files.check(record['path'], record['sha256']); files.proof(record['path'], status='complete', schema='scheduled-single-path-reconstruction-v1')

    registered(study/'analysis/complete'); complete = files.load(complete_path)
    if complete['schema'] != 'complete-scheduled-inference-evidence-v1' or set(complete['components']) != set(COMPONENTS):
        raise AssertionError('The scheduled aggregate has an incomplete component selection')
    if (complete['scientific_trajectories'], complete['scientific_updates'], complete['maximum_completed_horizon']) != (40, 10485760, 262144):
        raise AssertionError('Aggregate training totals changed')
    if complete['training'] != training['trajectories']: raise AssertionError('The aggregate changed native outcomes')
    components = {}
    for name, (proof_name, checker) in COMPONENTS.items():
        entry = complete['components'][name]; folder = study/'analysis'/name; path = study/'verification'/proof_name
        if Path(entry['path']) != folder or Path(entry['verification']) != path: raise AssertionError('Analytical component identity changed')
        registered(folder); proof = files.proof(path, checker)
        files.check(path, entry['verification_sha256'])
        for name_in, key in [('manifest.json', 'manifest_sha256'), ('results.json', 'results_sha256')]:
            files.check(folder/name_in, entry[key]); files.check(folder/name_in, proof['checked_sha256'][str(folder/name_in)])
        components[name] = files.load(folder/'results.json')
        print('Verified complete scheduled component', name, flush=True)
    mirrors = [('collective_conditions', 'collectives', 'conditions'), ('paired_arithmetic', 'collectives', 'paired_arithmetic'),
        ('prediction_states', 'predictions', 'states'), ('prediction_conditions', 'predictions', 'conditions'),
        ('paired_recipe_contrasts', 'predictions', 'paired_recipe_contrasts'), ('temporal_paths', 'training-paths', 'runs'),
        ('initial_inference', 'initial-inference-mechanism', 'states')]
    for name, component, field in mirrors:
        if complete[name] != components[component][field]: raise AssertionError('A completed analytical outcome was omitted: '+name)
    if complete['frozen_clocks'] != components['frozen-clocks'] or complete['clock_scores'] != components['clock-scores']:
        raise AssertionError('A frozen clock outcome was omitted')
    frozen = datetime.fromisoformat(components['frozen-clocks']['frozen_at'])
    controlled = [j for j in jobs if j['recipe'] == 'controlled']
    if set(components['frozen-clocks']['controlled_target_ids']) != {j['run_id'] for j in controlled}:
        raise AssertionError('Frozen clock targets changed')
    for job in controlled:
        binding = files.load(study/'runs'/job['run_id']/'binding.json')
        if frozen >= datetime.fromisoformat(binding['environment']['utc']): raise AssertionError('A clock prediction used a started target')

    diagnostics = {}
    for label, protocol, subfolder, count in [('observations', 'regime-observation-selection.json', 'observations', 320),
            ('prefix', 'scheduled-prefix-selection.json', 'prefix', 320), ('reasoning', 'reasoning-mechanism-selection.json', 'reasoning', 85),
            ('prompting', 'prompt-conditioning-selection.json', 'reasoning', 5),
            ('benchmarks', 'reference-benchmark-selection.json', 'reasoning', 2)]:
        selected = files.load(study/'protocols'/protocol)['cases']; records = []
        if len(selected) != count or len({c['name'] for c in selected}) != count: raise AssertionError('Incomplete diagnostic selection: '+label)
        for case in selected:
            path = study/'verification'/subfolder/(case['name']+'.json'); proof = files.proof(path)
            if proof['case'] != case: raise AssertionError('A diagnostic proof has the wrong selected state')
            registered(study/'measurements'/case['name'])
            if label == 'observations':
                registered(study/'measurements'/('risk-'+case['name'].removeprefix('cpu-')))
                if case['inference_stability']: registered(study/'measurements'/('inference-'+case['name'].removeprefix('cpu-')))
            records.append(dict(case=case, verification=files.record(path)))
        diagnostics[label] = records
        if label not in ['prompting', 'benchmarks']:
            old = complete['diagnostic_checks'][label]
            if [r['case'] for r in old] != selected: raise AssertionError('The aggregate selected different diagnostic states')
            for r in old: files.check(r['verification'], r['verification_sha256'])
    counts = dict(cpu_states=320, risk_states=320, generation_states=80, scheduled_prefix_states=320, task_states=85,
        tasks_per_state=320, task_modes=4, initial_prefix_states=5, initial_row_transport_states=5, local_row_jacobians=450, early_checkpoint_copies=6)
    if complete['counts'] != counts: raise AssertionError('The observation inventory changed')
    mechanism_checks = {}
    for name, checker in [('prefix-mechanism-initial', None), ('row-input-transport', None),
            ('row-jacobians', 'verify_row_jacobians.py'), ('reasoning-cohort', None),
            ('early-prefix-reconciliation', 'reconcile_early_scheduled_prefix.py')]:
        path = study/'verification'/(name+'.json'); result = files.proof(path, checker)
        files.check(path, complete['mechanism_checks'][name]['sha256']); mechanism_checks[name] = files.record(path)
        if name == 'row-jacobians' and ((result['states'], result['jacobians']) != (5, 450) or result['summaries'] != complete['local_row_jacobian_summaries']):
            raise AssertionError('The selected Jacobian outcomes changed')
        if name == 'early-prefix-reconciliation' and (result['states'], result['additional_independent_training_identities']) != (6, 0):
            raise AssertionError('Early copied states were not reconciled')

    prompt_path = study/'verification/prompt-conditioning-summary.json'
    prompt = files.proof(prompt_path, 'verify_prompt_conditioning.py'); registered(study/'analysis/prompt-conditioning')
    if (prompt['selected_states'], prompt['candidate_margin_comparisons'], prompt['group_summaries'], prompt['demonstration_contrasts']) != (5, 11520, 240, 160):
        raise AssertionError('The complete prompting outcome panel is required')
    registered(study/'data/prompt-conditioning-cohort')
    files.proof(study/'verification/prompt-conditioning-cohort.json')
    prompt_result = study/'analysis/prompt-conditioning/results.json'
    files.check(prompt_result, prompt['checked_sha256'][str(prompt_result)])

    benchmark_path = study/'verification/reference-benchmark-summary.json'
    benchmark = files.proof(benchmark_path, 'verify_reference_benchmark_summary.py')
    if (benchmark['selected_states'], benchmark['questions_per_state'], benchmark['candidate_margin_comparisons'], benchmark['normalized_subset_groups']) != (2, 3548, 113536, 64):
        raise AssertionError('The complete full-ARC benchmark panel is required')
    registered(study/'analysis/reference-benchmarks'); registered(study/'data/reference-benchmark-cohort')
    files.proof(study/'verification/reference-benchmark-cohort.json', 'verify_reference_benchmark_cohort.py')
    benchmark_result = study/'analysis/reference-benchmarks/results.json'
    files.check(benchmark_result, benchmark['checked_sha256'][str(benchmark_result)])
    benchmark_controller = files.load(study/'launcher-reference-benchmarks.json')
    if benchmark_controller['status'] != 'complete' or len(benchmark_controller['records']) != 2 or any(r['status'] != 'complete' for r in benchmark_controller['records']):
        raise AssertionError('A full-ARC measurement or raw reconstruction is missing')
    for name, digest in benchmark_controller['producer_sources'].items(): files.check(repo/name, digest)

    gpu_path = study/'verification/gpu-qualification.json'; gpu = files.proof(gpu_path, status='complete', schema='scheduled-gpu-qualification-v1')
    if (gpu['qualification_artifacts'], gpu['native_updates'], gpu['checkpoint_recovery']['byte_equal']) != (6, 5120, True):
        raise AssertionError('Native GPU and recovery qualification changed')
    analysis_path = study/'verification/analysis-qualification.json'; qualified = files.proof(analysis_path)
    if (qualified['candidate_mode_comparisons'], qualified['native_temporal_arrays'], qualified['changed_summary_rejected']) != (6400, 24, True):
        raise AssertionError('Supplemental analysis arithmetic is unqualified')
    recovery = files.load(study/'protocols/observation-recovered-implementation.json')
    files.check(study/'launcher-scheduled-observations.json', recovery['recovery']['original_controller_sha256'])
    files.check(study/'protocols/observation-implementation.json', recovery['recovery']['original_implementation_sha256'])
    for name, digest in recovery['producer_sources'].items(): files.check(repo/name, digest)
    observer_path = study/'launcher-scheduled-observations-recovered.json'; observer = files.load(observer_path)
    if observer['status'] != 'complete' or len(observer['records']) != 324 or any(r['status'] != 'complete' for r in observer['records']):
        raise AssertionError('Recovered observations are incomplete')
    qa_records = [r for r in observer['records'] if r['role'] == 'qualification']
    if len(qa_records) != 4: raise AssertionError('The four native reduction qualifications are required')
    for record in qa_records:
        proof = files.proof(record['verification']); files.check(record['verification'], record['verification_sha256'])
        if proof['risk']['native_reduction_values_replayed_bytewise'] != 640: raise AssertionError('Native risk replay is incomplete')

    intervention = study/'qualification/inference-interventions'; registered(intervention)
    intervention_result = files.load(intervention/'results.json')
    if len(intervention_result['records']) != 5 or intervention_result['records'] != complete['native_intervention_qualification']:
        raise AssertionError('Native operator intervention qualifications changed')
    # Recheck retained arrays rather than trusting their producer's booleans.
    with np.load(intervention/'measurements.npz') as raw:
        for record in intervention_result['records']:
            name = record['name']; native = raw[name+'_native_logits']; replay = raw[name+'_replay_logits']
            for other in [replay, raw[name+'_restored_logits'], raw[name+'_restored2_logits']]:
                if native.shape != other.shape or native.dtype != other.dtype or native.tobytes() != other.tobytes():
                    raise AssertionError('Native operator replay or restoration changed')
            if replay[:, :32].tobytes() != raw[name+'_suffix_logits'][:, :32].tobytes():
                raise AssertionError('Fixed-operator equal-prefix logits changed')
            projected = raw[name+'_projected_A']
            if not np.array_equal(projected, np.broadcast_to(projected[..., :1, :], projected.shape)):
                raise AssertionError('The projected rows differ')
    for name, digest in sources.items(): files.check(repo/name, digest)
    inherited_keys = ['observation_analysis', 'observation_analysis_sha256', 'scaling_analysis', 'scaling_analysis_sha256',
        'raw_verification', 'statistical_verification', 'native_step_verification', 'training_risk_verification',
        'numerical_verification', 'formal_statement_verification', 'formal_axiom_verification']
    result = dict(schema='scheduled-complete-evidence-v1', status='passed', required_scientific_runs=552,
        inherited_verifications=dict(scaling=files.record(args.scaling_verification)),
        **{key: inherited[key] for key in inherited_keys},
        training_artifacts=136, constant_training_artifacts=96, scheduled_training_artifacts=40,
        additional_updates=12640256, scheduled_updates=10485760, maximum_completed_horizon=262144,
        frozen_observation_states=544, diagnostics=diagnostics, scheduled_counts=counts,
        scheduled_analysis=str(complete_path), scheduled_analysis_sha256=files.check(complete_path),
        prompting_analysis=str(prompt_result), prompting_analysis_sha256=files.check(prompt_result),
        scheduled_raw_verification=files.record(training_path), prompt_verification=files.record(prompt_path),
        benchmark_verification=files.record(benchmark_path), benchmark_analysis=str(benchmark_result),
        benchmark_analysis_sha256=files.check(benchmark_result), full_reference_benchmark_states=2,
        full_reference_questions_per_state=3548, full_reference_normalizations=4,
        scheduled_components=complete['components'], mechanism_checks=mechanism_checks,
        gpu_qualification=files.record(gpu_path), analysis_qualification=files.record(analysis_path),
        observation_controller=files.record(observer_path), producer_registry=files.record(args.producer_registry),
        verifier_sources=sources, verifier_sha256=sha256(__file__), checked_sha256=files.checked,
        inventory_definition='416 inherited scientific artifacts plus 96 constant native artifacts including continuations, '
        'plus 40 complete scheduled training trajectories. Frozen checkpoints, 450 local Jacobians, 92 four-mode task states, '
        'six reconciled checkpoint copies and six GPU qualification artifacts do not add independent scientific training identities.',
        scope='Complete source-bound finite evidence, independently reconstructed statistics and selected proof scope. '
        'Every selected outcome is retained. Completion of this gate does not establish thermodynamic critical exponents, '
        'endogenous self-organization or broad reasoning competence without the corresponding scientific evidence.')
    write_json(args.output, result); print('Complete combined scientific evidence passed: 552 artifacts', flush=True)
    files.finish('passed')


if __name__ == '__main__': main()
