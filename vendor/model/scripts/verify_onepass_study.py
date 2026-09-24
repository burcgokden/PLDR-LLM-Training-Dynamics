#!/usr/bin/env python
"""Require the executed single-pass evidence and its complete comparison scope."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256,write_json
from verify_scheduled_study import Evidence


RENDERERS={
    'inference-render-manifest.json':'render_inference_mechanism.py',
    'prompt-render-manifest.json':'render_prompt_conditioning.py',
    'benchmark-render-manifest.json':'render_reference_benchmarks.py',
    'onepass-source-render-manifest.json':'render_onepass_data_comparison.py',
    'onepass-domain-render-manifest.json':'render_onepass_input_domain.py',
    'onepass-decoder-render-manifest.json':'render_decoder_prefix_interventions.py',
    'onepass-render-manifest.json':'render_onepass.py',
    'onepass-native-step-render-manifest.json':'render_onepass_native_steps.py',
    'onepass-source-reuse-render-manifest.json':'render_source_reuse.py',
    'onepass-schedule-comparison-render-manifest.json':'render_onepass_schedule_comparison.py',
    'law-closure-render-manifest.json':'render_law_closure.py',
}


class CompleteEvidence(Evidence):
    def proof(self,path,checker=None,status='passed',schema=None):
        result=super().proof(path,checker,status,schema)
        for key in ['source_sha256','inputs_sha256','checked_original_inputs_sha256']:
            values=result.get(key,{})
            if isinstance(values,dict):
                for name,digest in values.items():
                    self.check(self.repo/name if key=='source_sha256' else name,digest)
        return result



def verify_source_horizon_identities(files, study):
    """Check the paired source-recipe schedule histories without new training."""
    training = files.load(study/'protocols/onepass-training-selection.json')
    observations = files.load(study/'protocols/onepass-observation-selection.json')
    jobs = {j['run_id']: j for j in training['jobs']}
    cases = {(c['run_id'], c['step']): c for c in observations['cases']}
    records = []
    for recipe in ['reference1', 'subcritical1']:
        names = [f'{role}-{recipe}-h14-s640101' for role in ['data_schedule', 'compact']]
        left, right = [jobs[name] for name in names]
        profile_left, profile_right = [dict(j['profile']) for j in [left, right]]
        if (profile_left.pop('total_steps'), profile_right.pop('total_steps')) != (250000, 32768):
            raise AssertionError('Source-recipe annealing horizons changed')
        if profile_left != profile_right:
            raise AssertionError('A source-horizon comparison changes another optimizer setting')
        for key in ['heads', 'seed', 'shared_seed', 'stream_seed', 'microbatch',
                    'checkpoint_decoders', 'probe_every', 'recipe', 'warmup_override']:
            if left[key] != right[key]:
                raise AssertionError('A source-horizon identity changes: '+key)
        manifests = [files.load(study/'runs'/name/'manifest.json') for name in names]
        for key in ['initial_parameter_sha256', 'shared_generator_sha256', 'effective_batch_size',
                    'metric_units', 'normalization_alias']:
            if manifests[0][key] != manifests[1][key]:
                raise AssertionError('A source-horizon native identity changes: '+key)
        for key in ['corpus', 'manifest_sha256', 'blocks', 'documents', 'stream_seed', 'sampling']:
            if manifests[0]['data_law'][key] != manifests[1]['data_law'][key]:
                raise AssertionError('A source-horizon corpus identity changes: '+key)
        common_steps = sorted(set(map(int, left['save_steps'].split(','))) &
                              set(map(int, right['save_steps'].split(','))))
        warmup = profile_left['warmup_steps']
        if common_steps != [warmup, 8192, 32768]:
            raise AssertionError('Source-horizon paired checkpoint coverage changed')
        cohorts = []
        for step in common_steps:
            pair = [cases[name, step] for name in names]
            if pair[0]['training_cohort_sha256'] != pair[1]['training_cohort_sha256']:
                raise AssertionError('Source-horizon seen-data cohorts differ')
            files.check(pair[0]['training_cohort'], pair[0]['training_cohort_sha256'])
            cohorts.append(dict(step=step, cohort=files.record(pair[0]['training_cohort'])))
        paths = [study/'runs'/name/'sampling.npz' for name in names]
        for path, manifest in zip(paths, manifests, strict=True):
            files.check(path, manifest['sampling_sha256'])
        with np.load(paths[0], allow_pickle=False) as x, np.load(paths[1], allow_pickle=False) as y:
            for key in ['rows', 'offsets']:
                if not np.array_equal(x[key][:32768], y[key][:32768]):
                    raise AssertionError('The source-horizon sampling prefixes differ')
            for key in ['calibration_rows', 'evaluation_rows', 'dense_rows']:
                if not np.array_equal(x[key], y[key]):
                    raise AssertionError('The source-horizon native observation cohorts differ')
        paths = [study/'runs'/name/'measurements.npz' for name in names]
        rates = []
        for path, manifest in zip(paths, manifests, strict=True):
            files.check(path, manifest['raw_sha256'])
            with np.load(path, allow_pickle=False) as raw:
                rates.append(raw['applied_learning_rates'][:32768])
        if not np.array_equal(rates[0][:warmup+1], rates[1][:warmup+1]):
            raise AssertionError('Warmup histories differ in the source-horizon pair')
        if np.array_equal(rates[0][warmup+1:], rates[1][warmup+1:]):
            raise AssertionError('The source-horizon comparison has no changed annealing history')
        records.append(dict(recipe=recipe, source_path=names[0], compact_path=names[1],
            initial_parameter_sha256=manifests[0]['initial_parameter_sha256'],
            matched_checkpoint_cohorts=cohorts, identical_sampling_updates=32768,
            identical_applied_rate_updates=warmup+1,
            source_applied_rate_at_32768=rates[0][-1].tolist(),
            compact_applied_rate_at_32768=rates[1][-1].tolist()))
    return dict(status='passed', trajectory_pairs=2, checkpoint_pairs=6, records=records,
        scope='A secondary comparison of already selected source-recipe paths, conditional on one '
        'shared corpus, stream and initialization. Recorded inputs and warmup rates are matched; '
        'later imposed annealing histories differ. No new native update, independent corpus '
        'replication, warmup checkpoint tensor replay or population uncertainty is asserted.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='scheduled-training-feasible-20260908')
    p.add_argument('--reference-study',default='scheduled-training-20260908')
    p.add_argument('--scaling-verification',required=True);p.add_argument('--producer-registry',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--source-archive',required=True,help='Exact Git-bound executed project sources')
    p.add_argument('--closure-verification',required=True,help='Completed native branch and current-code qualification')
    p.add_argument('--state-law-verification',required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];root=Path(a.root).resolve();study=root/a.study;old=root/a.reference_study
    if Path(a.output).exists():raise FileExistsError(a.output)
    controller_counts={'onepass':61,'onepass-observations':288,'onepass-analyses':3,
        'repetition-analysis':3,'prefix-risk-comparison':21,'decoder-prefix-interventions':16,
        'onepass-input-domain':4,'repeated-mechanisms':4,'onepass-native-steps':6,
        'onepass-schedule-comparison':1}
    # The stopped repeated-corpus queue is deliberately absent. A selected fresh
    # result must exist before the expensive raw-file consistency checks begin.
    for name,count in controller_counts.items():
        path=study/('launcher-'+name+'.json')
        if not path.exists():raise AssertionError('Unfinished selected component: '+name)
        d=json.loads(path.read_text())
        if d['status']!='complete' or len(d['records'])!=count:
            raise AssertionError('Unfinished selected component: '+name)
    files=CompleteEvidence(repo,progress_path=str(a.output)+'.progress.json',source_archive=a.source_archive)
    files.check(Path(a.source_archive)/'manifest.json')
    sources={str(q.relative_to(repo)):sha256(q) for base in [repo/'scripts',repo/'src'] for q in base.rglob('*.py')}
    inherited=files.proof(a.scaling_verification,'verify_scaling_study.py',schema='scaling-complete-evidence-v1')
    if (inherited['required_scientific_runs'],inherited['training_artifacts'],inherited['additional_updates'])!=(512,96,2154496):
        raise AssertionError('Incomplete inherited constant-study evidence')
    if (inherited['numerical_verification']['tests_passed'],
            inherited['formal_statement_verification']['registered_statements'])!=(49,53):
        raise AssertionError('The selected numerical and formal proof scope changed')
    registry=files.load(a.producer_registry)
    if registry['status']!='complete' or registry['producer_sha256']!=files.check(repo/'scripts/build_producer_registry.py'):
        raise AssertionError('Current complete producer inventory is required')
    if registry['current_sources']!=sources:raise AssertionError('Producer inventory predates source changes')
    artifacts={r['artifact']:r for r in registry['artifacts']}
    if len(artifacts)!=registry['artifact_count']:raise AssertionError('Duplicate registered artifact')
    registered_paths=set()
    def registered(folder):
        folder=Path(folder);name=str(folder.relative_to(root))
        if name not in artifacts:raise AssertionError('Missing preserved producer: '+name)
        files.check(folder/'binding.json',artifacts[name]['binding_sha256']);files.artifact(folder)
        registered_paths.add(name)
    def selected_protocol(path):
        spec=files.load(path)
        for key in ['producer_sources','source_files']:
            for name,digest in spec.get(key,{}).items():files.check(repo/name,digest)
        for name,digest in spec.get('inputs_sha256',{}).items():files.check(name,digest)
        return spec
    controllers={}
    observation_recovery=None
    for name,count in controller_counts.items():
        path=study/('launcher-'+name+'.json');d=files.load(path)
        if name=='onepass':
            if any(r['returncode'] or r['scientific_status']!='complete' for r in d['records']):
                raise AssertionError('A selected native path failed')
        elif any(r.get('status','complete')!='complete' for r in d['records']):
            raise AssertionError('A selected diagnostic failed')
        if name=='onepass-observations':
            recovery=d.get('recovery')
            if recovery is None:raise AssertionError('The preserved observation recovery is required')
            files.check(recovery['protocol'],recovery['protocol_sha256'])
            contract=files.load(recovery['protocol'])
            if contract['schema']!='onepass-observation-arithmetic-recovery-v1':
                raise AssertionError('Unknown observation recovery contract')
            for entry,digest in contract['sources'].items():files.check(repo/entry,digest)
            files.check(recovery['original_controller'],recovery['original_controller_sha256'])
            previous=files.load(recovery['original_controller'])
            failures=[r for r in previous['records'] if r['status']!='complete']
            retained=[r for r in previous['records'] if r['status']=='complete']
            if previous['status']!='observation_failure' or len(failures)!=2 or len(retained)!=70:
                raise AssertionError('The original observation execution was not preserved')
            if failures!=d['preserved_failed_attempts'] or failures!=contract['failed_attempts']:
                raise AssertionError('A failed verification attempt was omitted')
            final={r['name']:r for r in d['records']}
            if len(final)!=288 or any(final[r['name']]!=r for r in retained):
                raise AssertionError('A completed observation was replaced during recovery')
            retained_names={r['name'] for r in retained}
            for row in d['records']:
                if row['name'] in retained_names:continue
                if row.get('recovery_protocol_sha256')!=recovery['protocol_sha256']:
                    raise AssertionError('An observation omitted its recovery binding')
                for command in row['commands']:
                    if command.get('returncode',0):raise AssertionError('A recovered stage failed')
                    if command.get('status')=='reused_complete_producer':
                        files.check(command['manifest'],command['manifest_sha256'])
                        for entry,digest in command['checked_sha256'].items():files.check(entry,digest)
            files.check(recovery['qualification'],recovery['qualification_sha256'])
            qualification=files.load(recovery['qualification'])
            if qualification['status']!='passed' or qualification['cases']!=6 or len(qualification['records'])!=6:
                raise AssertionError('The six arithmetic recovery qualifications are required')
            for entry,digest in qualification['sources'].items():files.check(repo/entry,digest)
            for row in qualification['records']:
                if row['returncode']:raise AssertionError('An arithmetic qualification failed')
                files.check(row['proof'],row['proof_sha256']);files.proof(row['proof'])
            observation_recovery=dict(status='passed',qualification_cases=6,
                preserved_failed_attempts=2,previous_complete_scientific_observations=66,
                completed_scientific_observations_under_recovery=218,
                protocol=files.record(recovery['protocol']),
                qualification=files.record(recovery['qualification']),
                original_controller=files.record(recovery['original_controller']))
        if name=='repetition-analysis':
            resumed=d.get('resumption')
            if resumed is None:raise AssertionError('The comparison resumption binding is required')
            files.check(resumed['protocol'],resumed['protocol_sha256'])
            contract=files.load(resumed['protocol'])
            if contract['schema']!='onepass-repetition-resumption-v1':
                raise AssertionError('Unknown comparison resumption contract')
            for entry,digest in contract['sources'].items():files.check(repo/entry,digest)
            files.check(contract['original_controller'],contract['original_controller_sha256'])
            prior=files.load(contract['original_controller'])
            if d['records'][:2]!=prior['records'] or d['records'][2]['panel']!='complete':
                raise AssertionError('A completed data-law comparison was replaced')
            files.check(study/'protocols/repetition-analysis-selection.json',contract['selection_sha256'])
        if name=='onepass-schedule-comparison':
            for entry,digest in d['execution_sources'].items():files.check(repo/'scripts'/entry,digest)
            files.check(study/'protocols/onepass-schedule-comparison-selection.json',d['selection_sha256'])
            files.check(study/'verification/onepass-schedule-comparison.json',d['records'][0]['verification_sha256'])
        controllers[name]=files.record(path)

    selection_path=study/'protocols/onepass-training-selection.json';selection=selected_protocol(selection_path)
    jobs=selection['jobs'];expected_roles={'compact':40,'data_schedule':2,'data_constant':4,'data_constant_short':8}
    if len(jobs)!=54 or dict(Counter(j['role'] for j in jobs))!=expected_roles or sum(j['steps'] for j in jobs)!=2555904:
        raise AssertionError('The complete single-pass training design changed')
    trajectories=[]
    for job in jobs:
        path=study/'verification/training'/(job['run_id']+'.json')
        proof=files.proof(path,status='complete',schema='scheduled-single-path-reconstruction-v1')
        if proof['selection_sha256']!=files.check(selection_path):raise AssertionError('Wrong native selection')
        row=proof['trajectory']
        if (row['run_id'],row['start'],row['stop'],row['updates'])!=(job['run_id'],0,job['steps'],job['steps']):
            raise AssertionError('Incomplete selected native horizon')
        for key in ['heads','seed','recipe','shared_seed','stream_seed']:
            if row[key]!=job[key]:raise AssertionError('A native condition changed')
        registered(study/'runs'/job['run_id']);trajectories.append(dict(job=job,verification=files.record(path)))
    source_horizon_identity=verify_source_horizon_identities(files,study)
    diagnostics={}
    for label,protocol,category,count in [
            ('observations','onepass-observation-selection.json','observations',284),
            ('prefix','scheduled-prefix-selection.json','prefix',284),
            ('reasoning','reasoning-mechanism-selection.json','reasoning',94)]:
        cases=selected_protocol(study/'protocols'/protocol)['cases'];records=[]
        if len(cases)!=count or len({c['name'] for c in cases})!=count:raise AssertionError('Selected checkpoint coverage changed')
        for c in cases:
            path=study/'verification'/category/(c['name']+'.json');proof=files.proof(path)
            if proof['case']!=c:raise AssertionError('Wrong diagnostic checkpoint')
            registered(study/'measurements'/c['name'])
            if label=='observations':
                suffix=c['name'].removeprefix('cpu-');registered(study/'measurements'/('risk-'+suffix))
                if c['inference_stability']:registered(study/'measurements'/('inference-'+suffix))
            records.append(dict(case=c,verification=files.record(path)))
        diagnostics[label]=records
    native_step_checks={}
    for branch,expected_draws,expected_cases in [('',32,6),('conditional-step-qualification',2,2)]:
        local=study/branch
        selection_path=local/'protocols/onepass-native-step-selection.json'
        selected=selected_protocol(selection_path)
        controller_path=local/'launcher-onepass-native-steps.json';controller=files.load(controller_path)
        if (controller['status'],len(controller['records']),selected['batch_replicas'])!=('complete',expected_cases,expected_draws):
            raise AssertionError('Incomplete conditional native-step selection or qualification')
        files.check(selection_path,controller['selection_sha256'])
        choices={c['name']:c for c in selected['cases']}
        if len(choices)!=expected_cases or set(choices)!={r['name'] for r in controller['records']}:
            raise AssertionError('Conditional native-step state coverage changed')
        for record in controller['records']:
            proof_path=Path(record['verification']);proof=files.proof(proof_path,'verify_onepass_native_step.py')
            files.check(proof_path,record['verification_sha256'])
            if record['status']!='complete' or (proof['conditional_native_steps'],proof['contexts'])!=(expected_draws,512):
                raise AssertionError('Incomplete conditional native optimizer reconstruction')
            if any(proof['case'][key]!=value for key,value in choices[record['name']].items()):
                raise AssertionError('A selected conditional incoming state changed')
            registered(local/'measurements'/record['name'])
            native_step_checks[record['name']]=files.record(proof_path)
        if branch:controllers['conditional-native-step-qualification']=files.record(controller_path)
    components={}
    for name,filename,checker in [
            ('onepass-collectives','onepass-collective-statistics.json','verify_onepass_collectives.py'),
            ('onepass-predictions','onepass-prediction-statistics.json','verify_onepass_predictions.py'),
            ('onepass-training-paths','onepass-temporal-statistics.json','verify_onepass_paths.py'),
            ('repetition-reference_schedule','repetition-reference_schedule.json','verify_repetition_comparison.py'),
            ('repetition-constant','repetition-constant.json','verify_repetition_comparison.py'),
            ('repetition-complete','repetition-complete.json','verify_repetition_comparison.py'),
            ('decoder-prefix-interventions','decoder-prefix-intervention-summary.json','verify_decoder_prefix_summary.py'),
            ('source-reuse','source-reuse.json','verify_source_reuse.py'),
            ('onepass-schedule-comparison','onepass-schedule-comparison.json','verify_onepass_schedule_comparison.py')]:
        path=study/'verification'/filename
        # The collective checker records its entire source closure instead of a separate entrypoint field.
        proof=files.proof(path,None if name=='onepass-collectives' else checker)
        registered(study/'analysis'/name)
        if name=='source-reuse':
            selected_protocol(study/'protocols/source-reuse-selection.json')
            if (proof['matched_pairs'],proof['checkpoint_histograms'],proof['exposure_groups'],proof['additional_training_updates'])!=(9,18,7,0):
                raise AssertionError('Incomplete source-position exposure reconstruction')
        if name=='onepass-schedule-comparison':
            spec=selected_protocol(study/'protocols/onepass-schedule-comparison-selection.json')
            for entry,digest in spec['sources'].items():files.check(repo/entry,digest)
            if (proof['conditions'],proof['native_state_pairs'],proof['trajectory_pairs'],proof['precision_conditions'],proof['paired_bootstrap_arrays'],proof['additional_training_updates'])!=(4,16,8,8,24,0):
                raise AssertionError('Incomplete matched single-pass schedule comparison')
            observations=files.load(study/'protocols/onepass-observation-selection.json')['cases']
            observed={(c['run_id'],c['step']):c for c in observations}
            identities=set()
            for group in spec['groups']:
                for pair in group['pairs']:
                    left,right=[observed[pair[k],group['step']] for k in ['scheduled','constant']]
                    if left['training_cohort_sha256']!=right['training_cohort_sha256']:
                        raise AssertionError('A schedule pair uses different seen-training risk cohorts')
                    manifests=[files.load(study/'runs'/pair[k]/'manifest.json') for k in ['scheduled','constant']]
                    if manifests[0]['initial_parameter_sha256']!=manifests[1]['initial_parameter_sha256']:
                        raise AssertionError('A schedule pair has different native initial parameters')
                    identities.add((pair['scheduled'],pair['constant']))
            if len(identities)!=8:raise AssertionError('The schedule pairs do not retain eight trajectory identities')
        components[name]=dict(verification=files.record(path),results=files.record(study/'analysis'/name/'results.json'))
    for category,protocol,count,checker in [
            ('prefix-risk','prefix-risk-comparison.json',21,'verify_prefix_risk_comparison.py'),
            ('decoder-prefix-risk','decoder-prefix-intervention-selection.json',16,'verify_decoder_prefix_intervention.py')]:
        selected=selected_protocol(study/'protocols'/protocol)
        cases=selected['cases']
        if category=='prefix-risk':
            if len(cases)!=20 or selected['qualification']['family']!='qualification':
                raise AssertionError('The twenty causal states and separate execution qualification are required')
            cases=[selected['qualification'],*cases]
        if len(cases)!=count:raise AssertionError('Incomplete selected causal-risk panel')
        for c in cases:
            proof=files.proof(study/'verification'/category/(c['name']+'.json'),checker)
            if proof['case']!=c or proof['native_risks_replayed_bytewise']!=4096:
                raise AssertionError('Incomplete causal-risk reconstruction')
            registered(study/'measurements'/category/c['name'])
    for protocol in ['onepass-input-domain-selection.json','onepass-row-jacobian-selection.json']:
        for c in selected_protocol(study/'protocols'/protocol)['cases']:registered(study/'measurements'/c['name'])
    mechanisms={}
    for name,checker in [('onepass-input-domain','verify_onepass_input_domain.py'),
            ('onepass-row-jacobians','verify_onepass_row_jacobians.py'),
            ('repeated-checkpoint-sealing','prepare_repeated_comparison.py'),
            ('repeated-risk-comparison','verify_repeated_comparison_risk.py')]:
        path=study/'verification'/(name+'.json');proof=files.proof(path,checker);mechanisms[name]=files.record(path)
        if name=='onepass-row-jacobians' and proof['jacobians']!=1440:raise AssertionError('Missing input-domain derivatives')
    repeated=selected_protocol(study/'protocols/repeated-risk-comparison.json')['cases']
    if len(repeated)!=4:raise AssertionError('Incomplete retained data-law comparison')
    for c in repeated:
        registered(study/'repeated-checkpoints'/c['name']/'checkpoint')
        registered(study/'measurements/repeated-risk'/c['name'])
    repeated_study=study/'repeated-observations'
    for filename,category in [('repeated-observation-selection.json','collectives'),
                             ('scheduled-prefix-selection.json','prefix'),('reasoning-mechanism-selection.json','reasoning')]:
        cases=selected_protocol(repeated_study/'protocols'/filename)['cases']
        if len(cases)!=4:raise AssertionError('Incomplete retained-state mechanism panel')
        for c in cases:
            proof=files.proof(repeated_study/'verification'/category/(c['name']+'.json'))
            if proof['case']!=c:raise AssertionError('Wrong retained checkpoint')
            registered(repeated_study/'measurements'/c['name'])

    qualifications={}
    for name,checker in [('expanded-data','verify_expanded_refinedweb.py'),('onepass-data','verify_onepass_refinedweb.py'),
                         ('onepass-cohort-separation','verify_onepass_cohort_separation.py')]:
        path=study/'verification'/(name+'.json');files.proof(path,checker);qualifications[name]=files.record(path)
    if {r['artifact'] for r in registry.get('corpus_records',[])}!= {'data/refinedweb-expanded-307200','data/refinedweb-onepass-524288'}:
        raise AssertionError('The two corpus preparations need explicit producer authority')
    registered(study/'qualification/onepass-stream');stream=files.load(study/'qualification/onepass-stream/results.json')
    if (stream['status'],stream['unique_blocks'],stream['maximum_updates'],stream['scalar_phases'])!=('passed',4194304,131072,1407006):
        raise AssertionError('The complete nonrepetition and schedule qualification is required')
    gpu_path=study/'verification/gpu-qualification.json';gpu=files.proof(gpu_path,status='complete',schema='onepass-gpu-qualification-v1')
    if (gpu['qualification_artifacts'],gpu['native_updates'],gpu['checkpoint_recovery']['byte_equal'])!=(7,6144,True):
        raise AssertionError('Incomplete native GPU or resume qualification')
    qualifications['gpu']=files.record(gpu_path)
    for job in selected_protocol(study/'protocols/onepass-qualification-selection.json')['jobs']:
        registered(study/'runs'/job['run_id'])
    observer_path=study/'verification/observer-qualification.json';observer=files.proof(observer_path)
    if len(observer['records'])!=4:raise AssertionError('The four observer qualifications are required')
    for record in observer['records']:
        if record['status']!='complete' or len(record['verification_sha256'])!=3:
            raise AssertionError('An observer qualification omitted an independent check')
        for path,digest in record['verification_sha256'].items():
            files.proof(path);files.check(path,digest)
    for b in (study/'observer-qualification/measurements').glob('*/binding.json'):registered(b.parent)
    qualifications['observer']=files.record(observer_path)
    for name,path in [('analysis',study/'qualification/onepass-analysis/verification.json'),
                      ('data-law-theory',repo/'docs/data-law-numerical/verification.json'),
                      ('finite-data-response',repo/'docs/finite-data-response/verification.json'),
                      ('corpus-conditioning',repo/'docs/corpus-conditioning/verification.json'),
                      ('paired-schedules',study/'qualification/onepass-schedule-comparison/verification.json')]:
        proof=files.proof(path);qualifications[name]=files.record(path)
        if name=='analysis' and (len(proof['fixtures']),proof['mutations_rejected'],proof['selected_windows'])!=(21,4,1718):
            raise AssertionError('The finite-ensemble analysis qualification changed')
        if name=='paired-schedules':
            if (proof['fixtures'],proof['pairing_distinctions'])!=(5,2):
                raise AssertionError('Missing matched schedule reduction qualifications')
            for entry,digest in proof['sources'].items():files.check(repo/entry,digest)
        if name=='data-law-theory':
            if proof['exact_finite_cases']!=13:raise AssertionError('Missing exact data-law cases')
            files.check(repo/'scripts/check_data_law_theory.py',proof['checker_sha256'])
        if name=='finite-data-response':
            if proof['exact_cases']!=18:raise AssertionError('Missing finite-population response cases')
            files.check(repo/'scripts/check_finite_data_response.py',proof['source_sha256'])
        if name=='corpus-conditioning':
            if (proof['exact_cases'],len(proof['covariance_cases']),len(proof['adaptive_path_cases']),proof['distinctions_detected'])!=(15,12,3,3):
                raise AssertionError('Missing exact corpus-conditioning cases')
            files.check(repo/'scripts/check_corpus_conditioning.py',proof['checker_sha256'])
            files.check(repo/'src/model_rg/provenance.py',proof['helper_sha256'])

    # Completed released-model and fixed-checkpoint studies remain useful
    # inference evidence. No canceled source-schedule training is required here.
    reference={}
    for name,checker in [('initial-inference-summary','verify_initial_inference_summary.py'),
            ('prefix-mechanism-initial',None),('row-input-transport',None),('row-jacobians','verify_row_jacobians.py'),
            ('reasoning-cohort',None),('prompt-conditioning-cohort','verify_prompt_cohort.py'),
            ('prompt-conditioning-summary','verify_prompt_conditioning.py'),
            ('reference-benchmark-cohort','verify_reference_benchmark_cohort.py'),
            ('reference-benchmark-summary','verify_reference_benchmark_summary.py')]:
        path=old/'verification'/(name+'.json');proof=files.proof(path,checker);reference[name]=files.record(path)
        if name=='initial-inference-summary' and (proof['selected_states'],proof['candidate_margin_comparisons'])!=(5,6400):
            raise AssertionError('Missing fixed-checkpoint task outcomes')
        if name=='row-jacobians' and (proof['states'],proof['jacobians'])!=(5,450):
            raise AssertionError('Missing fixed-checkpoint local derivatives')
        if name=='prompt-conditioning-summary' and (proof['selected_states'],proof['candidate_margin_comparisons'])!=(5,11520):
            raise AssertionError('Missing prompting outcomes')
        if name=='reference-benchmark-summary' and (proof['selected_states'],proof['questions_per_state'],proof['candidate_margin_comparisons'])!=(2,3548,113536):
            raise AssertionError('Missing full reference benchmark outcomes')
    for name in ['initial-inference-mechanism','prompt-conditioning','reference-benchmarks']:registered(old/'analysis'/name)
    for name in ['reasoning-cohort','prompt-conditioning-cohort','reference-benchmark-cohort']:registered(old/'data'/name)
    for filename,count in [('prompt-conditioning-selection.json',5),('reference-benchmark-selection.json',2)]:
        cases=selected_protocol(old/'protocols'/filename)['cases']
        if len(cases)!=count:raise AssertionError('Missing complete reference outcome panel')
        for c in cases:
            proof=files.proof(old/'verification/reasoning'/(c['name']+'.json'))
            if proof['case']!=c:raise AssertionError('Reference inference state changed')
            registered(old/'measurements'/c['name'])
    initial=files.load(old/'analysis/initial-inference-mechanism/results.json')
    intervention=old/'qualification/inference-interventions';registered(intervention)
    records=files.load(intervention/'results.json')['records']
    if len(records)!=5:raise AssertionError('Incomplete reference intervention qualification')
    with np.load(intervention/'measurements.npz') as raw:
        for r in records:
            name=r['name'];native=raw[name+'_native_logits']
            for suffix in ['_replay_logits','_restored_logits','_restored2_logits']:
                other=raw[name+suffix]
                if native.shape!=other.shape or native.dtype!=other.dtype or native.tobytes()!=other.tobytes():
                    raise AssertionError('Reference operator replay or restoration changed')
            if raw[name+'_replay_logits'][:,:32].tobytes()!=raw[name+'_suffix_logits'][:,:32].tobytes():
                raise AssertionError('Reference fixed-prefix replay changed')
            projected=raw[name+'_projected_A']
            if not np.array_equal(projected,np.broadcast_to(projected[...,:1,:],projected.shape)):
                raise AssertionError('Reference row projection changed')
    rendered={}
    for manifest,script in RENDERERS.items():
        path=repo/'manuscript/generated'/manifest;value=files.load(path)
        if value['status']!='complete' or value['renderer_sha256']!=files.check(repo/'scripts'/script):
            raise AssertionError('An interpreted result table predates its renderer')
        for name,digest in value['inputs'].items():files.check(name,digest)
        for name,digest in value['generated'].items():files.check(path.parent/name,digest)
        rendered[manifest]=files.record(path)
    closure=files.proof(a.closure_verification,'verify_law_closure.py',schema='conditional-law-closure-verification-v1')
    if (closure['scientific_branches'],closure['scientific_updates'],closure['incoming_states'],
            closure['heldout_initializations'],closure['calibration_initializations'])!=(256,16384,16,2,2):
        raise AssertionError('Incomplete single-pass closure validation')
    if (closure['numerical_tests'],closure['formal_statements'])!=(52,54):
        raise AssertionError('Current native/code/formal qualification is incomplete')
    # Explicit archived resolutions retain their executing-source identity.
    # Fresh qualification above must bind all its sources to current bytes.
    for name,digest in closure['checked_sha256'].items():
        if Path(name).is_relative_to(repo) and not Path(name).is_relative_to(repo/'internal'):
            if sha256(name)!=digest:raise AssertionError('Current qualification changed: '+name)
    closure_record=files.record(a.closure_verification)
    state_law=files.proof(a.state_law_verification,'verify_state_law.py',schema='state-conditioned-law-verification-v1')
    if (state_law['fresh_calibration_branches'],state_law['validation_branches'],state_law['native_updates'],state_law['incoming_states'])!=(64,128,12288,8):
        raise AssertionError('Incomplete state-conditioned stochastic-law validation')
    state_law_record=files.record(a.state_law_verification)
    for name,digest in sources.items():files.check(repo/name,digest)
    inherited_keys=['observation_analysis','observation_analysis_sha256','scaling_analysis','scaling_analysis_sha256',
        'raw_verification','statistical_verification','native_step_verification','training_risk_verification',
        'numerical_verification','formal_statement_verification','formal_axiom_verification']
    result=dict(schema='onepass-complete-evidence-v1',status='passed',required_scientific_runs=566,
        inherited_verifications=dict(scaling=files.record(a.scaling_verification)),
        **{key:inherited[key] for key in inherited_keys},
        training_artifacts=150,constant_training_artifacts=96,onepass_training_artifacts=54,
        additional_updates=4710400,onepass_updates=2555904,maximum_completed_horizon=131072,
        onepass_cpu_states=284,onepass_task_states=94,onepass_prefix_states=284,
        onepass_causal_scientific_states=20,onepass_causal_qualification_states=1,
        conditional_native_steps=192,conditional_native_states=6,conditional_qualification_steps=4,
        native_step_checks=native_step_checks,
        law_closure_verification=closure_record,law_closure_branches=256,law_closure_updates=16384,
        law_closure_states=16,current_numerical_tests=52,current_formal_statements=54,
        state_law_verification=state_law_record,state_law_calibration_branches=64,
        state_law_validation_branches=128,state_law_updates=12288,
        total_new_conditional_branches=448,total_new_conditional_updates=28672,
        executed_source_archive=files.record(Path(a.source_archive)/'manifest.json'),
        executed_source_resolutions=files.archive.resolutions,
        source_reuse_pairs=9,source_reuse_checkpoint_histograms=18,source_reuse_exposure_groups=7,
        paired_schedule_conditions=4,paired_schedule_checkpoint_pairs=16,paired_schedule_trajectory_pairs=8,
        source_horizon_identity=source_horizon_identity,
        trajectories=trajectories,diagnostics=diagnostics,onepass_components=components,
        mechanism_checks=mechanisms,qualifications=qualifications,reference_checks=reference,
        full_reference_benchmark_states=2,full_reference_questions_per_state=3548,
        controllers=controllers,observation_recovery=observation_recovery,
        rendered=rendered,producer_registry=files.record(a.producer_registry),
        registered_selected_artifacts=sorted(registered_paths),
        verifier_sources=sources,verifier_sha256=sha256(__file__),checked_sha256=files.checked,
        inventory_definition='416 inherited scientific artifacts plus 96 constant native artifacts, '
        'including continuations, plus 54 complete single-pass training paths. Copied checkpoints, '
        'CPU observations, local derivatives and operator interventions add no independent training identities. '
        'The 192 scientific conditional CPU steps at six incoming states and four qualification steps '
        'are separate diagnostic branches, not additional complete training paths. '
        'The administratively stopped repeated-corpus source runs are retained as checkpoint comparisons '
        'and are not counted as completed long training paths.',
        scope='Complete finite evidence under the declared data, initialization, schedule and observation laws. '
        'The selected proof correspondence is separate from empirical validation. Completion does not '
        'establish thermodynamic exponents, endogenous self-organization or broad reasoning competence.')
    write_json(a.output,result);files.finish('passed')
    print('Complete single-pass evidence passed: 566 artifacts and 256 conditional closure branches',flush=True)


if __name__=='__main__':main()
