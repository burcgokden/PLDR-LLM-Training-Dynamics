#!/usr/bin/env python
"""Require a complete source-bound scaling study before manuscript publication.

This gate composes separately executed raw and statistical reconstructions. It
also checks the diagnostic inventory, frozen target timing and complete analysis
aggregation. It does not turn a numerical check into a criticality claim.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from model_rg.provenance import sha256, write_json


COMPONENTS = {
    'size-time', 'collectives', 'paired-paths', 'conditional-noise', 'forcing-geometry',
    'frozen-forecasts', 'forecast-scores', 'frozen-variance', 'variance-scores',
    'environment-factorial', 'shared-updates', 'matched-forcing', 'visible-noise',
    'clock-observation-sensitivity', 'source-basis', 'native-step', 'common-geometry', 'training-risk',
}


def read(path):
    return json.loads(Path(path).read_text())


class Evidence:
    def __init__(self):
        self.cache = {}

    def digest(self, path):
        path = Path(path).resolve()
        st = path.stat()
        key = (str(path), st.st_size, st.st_mtime_ns)
        if key not in self.cache:self.cache[key] = sha256(path)
        return self.cache[key]

    def check(self, path, expected):
        if self.digest(path) != expected:
            raise AssertionError('Changed verified evidence: '+str(path))

    def binding(self, path, meta):
        self.check(path/'binding.json',meta['binding_sha256'])
        b = read(path/'binding.json')
        for file,signature in b['source_files'].items():self.check(path/'source'/file,signature)
        for file,signature in b['inputs'].items():self.check(file,signature)
        return b

    def manifest(self, path):
        path = Path(path)
        meta = read(path/'manifest.json')
        if meta['status'] != 'complete':raise AssertionError('Incomplete selected artifact: '+str(path))
        self.binding(path,meta)
        for file,key in [('results.json','results_sha256'),('measurements.npz','raw_sha256'),
            ('generator-velocities.npy','velocities_sha256'),('predictor.npz','predictor_sha256'),
            ('validation-directions.npy','directions_sha256'),('predictor.json','predictor_record_sha256'),
            ('first-step-digests.json','first_step_digests_sha256')]:
            if key in meta:self.check(path/file,meta[key])
        return meta

    def verified_record(self, path, repo, expected_schema=None):
        path = Path(path)
        record = read(path)
        if record['status'] != 'passed':raise AssertionError('A complete passing reconstruction is required')
        if expected_schema and record['schema'] != expected_schema:raise AssertionError('Wrong verification scope')
        for name,signature in record['verifier_sources'].items():self.check(repo/name,signature)
        return record,dict(path=str(path.resolve()),sha256=self.digest(path))


def same(x, y):
    np.testing.assert_allclose(x,y,rtol=2e-10,atol=1e-18)


def diagnostic_inventory(study, protocol, expected, files):
    spec = read(study/'protocols'/protocol)
    names = [c['name'] for c in spec['cases']]
    if len(names) != expected or len(set(names)) != expected:
        raise AssertionError('Selected diagnostic design changed: '+protocol)
    result = []
    for case in spec['cases']:
        path = study/'measurements'/case['name']
        meta = files.manifest(path)
        if meta['case'] != case:raise AssertionError('Diagnostic case differs from its frozen selection')
        result.append(dict(name=case['name'],manifest_sha256=files.digest(path/'manifest.json')))
    return result


def frozen_before_targets(study, frozen, jobs):
    time = datetime.fromisoformat(frozen['frozen_at'])
    for job in jobs:
        binding = read(study/'runs'/job['run_id']/'binding.json')
        if time >= datetime.fromisoformat(binding['environment']['utc']):
            raise AssertionError('A prediction was frozen after a target started')


def main():
    p = argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--study',default='critical-scaling-20260906');p.add_argument('--output',required=True)
    p.add_argument('--observation-verification',required=True);p.add_argument('--raw-verification',required=True)
    p.add_argument('--statistical-verification',required=True);p.add_argument('--producer-registry',required=True)
    p.add_argument('--axiom-verification',default='docs/scaling-formal/verification.json')
    p.add_argument('--numerical-verification',default='docs/scaling-numerical-bytewise/verification.json')
    p.add_argument('--native-step-verification');p.add_argument('--training-risk-verification')
    a = p.parse_args();repo = Path(__file__).resolve().parents[1];root = Path(a.root);study = root/a.study
    if Path(a.output).exists():raise FileExistsError(a.output)
    files = Evidence()
    source_names = ['scripts/verify_scaling_study.py','scripts/verify_scaling_raw.py',
        'scripts/verify_scaling_statistics.py','scripts/verify_observation_study.py',
        'scripts/build_producer_registry.py','scripts/check_formal.py','scripts/check_numerical.py',
        'scripts/verify_scaling_native_step.py','scripts/verify_scaling_training_risk.py','src/model_rg/provenance.py']
    sources = {name:sha256(repo/name) for name in source_names}
    observation, observation_binding = files.verified_record(a.observation_verification,repo,'observation-complete-evidence-v1')
    if observation['required_scientific_runs'] != 416:raise AssertionError('Inherited scientific evidence is incomplete')
    for record in observation['inherited_verifications'].values():files.check(record['path'],record['sha256'])
    files.check(observation['observation_analysis'],observation['observation_analysis_sha256'])
    for key,count_field,expected in [('numerical_verification','tests_passed',49),
                                     ('formal_statement_verification','registered_statements',53)]:
        info = observation[key];files.check(info['path'],info['sha256'])
        if info[count_field] != expected:raise AssertionError('Current supporting scope changed')
    numerical_path = Path(a.numerical_verification).resolve()
    numerical = read(numerical_path)
    if numerical['status'] != 'passed' or numerical['tests_passed'] != 49:
        raise AssertionError('The complete current numerical suite is required')
    numerical_binding = dict(path=str(numerical_path),sha256=files.digest(numerical_path),tests_passed=49)
    files.check(repo/'scripts/check_numerical.py',numerical['checker_sha256'])
    for name,signature in numerical['tested_sources'].items():files.check(repo/name,signature)
    files.check(numerical['log'],numerical['log_sha256'])
    statements = read(observation['formal_statement_verification']['path'])
    for name,signature in {**statements['modules'],**statements['support_sources']}.items():files.check(repo/name,signature)
    axiom_path = Path(a.axiom_verification).resolve()
    axioms = read(axiom_path)
    if axioms['status'] != 'passed' or axioms['explicit_theorems'] != 53:
        raise AssertionError('The complete axiom and ownership gate is required')
    for name,signature in axioms['module_sources'].items():files.check(repo/name,signature)
    files.check(repo/'scripts/check_formal.py',axioms['checker_sha256'])
    files.check(repo/'scripts/formal/Gate.lean',axioms['gate_sha256'])
    files.check(repo/'lake-manifest.json',axioms['lake_manifest_sha256'])
    if (repo/'lean-toolchain').read_text().strip() != axioms['toolchain']:
        raise AssertionError('Lean toolchain changed after verification')
    for check in axioms['checks']:
        if not check['expectation_met']:raise AssertionError('An axiom-gate mutation did not have its required outcome')
        files.check(axiom_path.parent/(check['name']+'.log'),check['log_sha256'])

    sources.update(numerical['tested_sources'])
    sources.update(statements['modules']);sources.update(statements['support_sources'])
    sources.update(axioms['module_sources'])
    for name in ['scripts/formal/Gate.lean','lean-toolchain','lake-manifest.json','scripts/check_statement_contracts.py','scripts/formal/statement-registry.json']:
        sources[name] = sha256(repo/name)

    raw,raw_binding = files.verified_record(a.raw_verification,repo,'native-scaling-raw-verification-v1')
    stats,stats_binding = files.verified_record(a.statistical_verification,repo,'native-scaling-statistical-verification-v1')
    if raw['pending_protocols'] or raw['pending_training'] or raw['arguments']['allow_partial']:
        raise AssertionError('A partial raw snapshot cannot serve as the release gate')
    if stats['arguments']['collective_snapshot']:raise AssertionError('A partial statistical snapshot cannot serve as the release gate')
    if (raw['training_artifacts'],raw['additional_updates'],raw['frozen_observation_states']) != (96,2154496,224):
        raise AssertionError('The selected training or observation inventory changed')
    for row in raw['training']:files.check(study/'runs'/row['run_id']/'manifest.json',row['manifest_sha256'])
    for row in raw['observations']:files.check(study/'measurements'/row['name']/'manifest.json',row['manifest_sha256'])

    complete = study/'analysis/complete'
    files.manifest(complete);report = read(complete/'results.json')
    if set(report['components']) != COMPONENTS:raise AssertionError('A completed analytical component is missing')
    reports = {}
    for name,entry in report['components'].items():
        folder = Path(entry['path'])
        files.check(folder/'manifest.json',entry['manifest_sha256'])
        files.check(folder/'results.json',entry['results_sha256'])
        files.manifest(folder);reports[name] = read(folder/'results.json')
        print('Verified analysis binding',name,flush=True)
    if report['training_artifacts'] != raw['training_artifacts'] or report['additional_updates'] != raw['additional_updates']:
        raise AssertionError('Aggregated training count differs from raw reconstruction')
    if {r['run_id'] for r in report['training']} != {r['run_id'] for r in raw['training']}:
        raise AssertionError('A trained outcome was omitted or counted twice')
    for row in report['training']:
        other = next(r for r in raw['training'] if r['run_id']==row['run_id'])
        if (row['start_step'],row['completed_step'],row['additional_updates']) != (other['start'],other['stop'],other['updates']):
            raise AssertionError('A continuation changed its elapsed-update count')
    if report['maximum_completed_horizon'] != max(r['stop'] for r in raw['training']):
        raise AssertionError('Maximum horizon is not the executed maximum')

    diagnostics = {}
    for label,protocol,expected in [('conditional_noise','conditional-noise-jvp.json',24),
        ('fresh_visible_sources','visible-noise.json',16),('expanded_source_bases','source-basis.json',16),
        ('native_conditional_steps','native-step.json',16),
        ('frozen_training_risk','training-risk-selection.json',32),
        ('spectator_controls','spectator-control.json',5)]:
        diagnostics[label] = diagnostic_inventory(study,protocol,expected,files)
        print('Verified selected diagnostics',label,expected,flush=True)
    if [r['case']['name'] for r in reports['conditional-noise']['states']] != [x['name'] for x in diagnostics['conditional_noise']]:
        raise AssertionError('Conditional noise outcomes were selected after observation')
    if [r['case']['name'] for r in reports['visible-noise']['states']] != [x['name'] for x in diagnostics['fresh_visible_sources']]:
        raise AssertionError('Fresh source outcomes were selected after observation')
    if [r['case']['name'] for r in reports['source-basis']['states']] != [x['name'] for x in diagnostics['expanded_source_bases']]:
        raise AssertionError('An expanded source outcome was omitted')
    if [r['case']['name'] for r in reports['native-step']['states']] != [x['name'] for x in diagnostics['native_conditional_steps']]:
        raise AssertionError('A native conditional step was omitted')
    risk_path = Path(a.training_risk_verification) if a.training_risk_verification else study/'verification/training-risk.json'
    risk,risk_binding = files.verified_record(risk_path,repo,'training-risk-reconstruction-v1')
    if (risk['selected_states'],risk['moment_fields'],risk['paired_fields']) != (32,32,24):
        raise AssertionError('The complete training-risk reconstruction is required')
    for name,signature in risk['verified_files'].items():files.check(name,signature)
    files.check(study/'protocols/training-risk-selection.json',risk['protocol_sha256'])
    risk_analysis=Path(report['components']['training-risk']['path'])/'results.json'
    if Path(risk['analysis']).resolve()!=risk_analysis.resolve():
        raise AssertionError('Risk reconstruction concerns a different selected analysis')
    files.check(risk_analysis,risk['analysis_sha256'])
    if risk['states']!=reports['training-risk']['states']:
        raise AssertionError('The risk reconstruction selection differs from the complete analysis')
    if [r['case']['name'] for r in reports['training-risk']['states']] != [x['name'] for x in diagnostics['frozen_training_risk']]:
        raise AssertionError('A selected frozen training-risk outcome was omitted')
    replay_path = Path(a.native_step_verification) if a.native_step_verification else study/'verification/native-step-bytewise.json'
    replay,replay_binding = files.verified_record(replay_path,repo)
    protocol_path = study/'protocols/native-step.json'
    files.check(protocol_path,replay['protocol_sha256'])
    if replay['selected_states'] != 16 or reports['native-step']['conditional_draws'] != 512:
        raise AssertionError('The complete native step reconstruction is required')
    if [r['case']['name'] for r in replay['states']] != [r['name'] for r in diagnostics['native_conditional_steps']]:
        raise AssertionError('Native replay selection differs from the measured selection')
    for row in replay['states']:
        if not row['first_step_bitwise_replay_passed']:raise AssertionError('A native replay failed')
        folder = study/'measurements'/row['case']['name']
        files.check(folder/'manifest.json',row['manifest_sha256'])
    clarification = study/'protocols/source-basis-interpretation.json'
    if report['interpretation_bindings'] != {str(clarification):files.digest(clarification)}:
        raise AssertionError('Source protocol clarification is missing or changed')
    selected_updates = read(study/'protocols/shared-update-observation.json')['run_ids']
    if len(selected_updates) != 60 or set(selected_updates) != {r['run_id'] for r in reports['shared-updates']['runs']}:
        raise AssertionError('A passively observed trajectory was omitted')
    for name in selected_updates:
        if next(r for r in raw['training'] if r['run_id']==name)['passive_update_check'] is None:
            raise AssertionError('Passive observation lacks its independent raw check')

    frozen = reports['frozen-forecasts']
    jobs = []
    for name in ['clock-holdout.json','diffusive-size-map.json']:
        path = study/'protocols'/name;files.check(path,frozen['protocol_sha256'][name]);jobs += read(path)['jobs']
    frozen_before_targets(study,frozen,jobs)
    if len(reports['forecast-scores']['forecasts']) != 54 or len(reports['forecast-scores']['joint_forecasts']) != 3:
        raise AssertionError('A frozen clock alternative was omitted')
    frozen_before_targets(study,reports['frozen-variance'],reports['frozen-variance']['targets'])
    if len(reports['variance-scores']['forecasts']) != 144:
        raise AssertionError('A frozen variance alternative was omitted')
    # The complete report must reproduce the checked component, not a hand-edited
    # favorable subset. Additional compact fields are checked by their verifier.
    mirrors = [('clock_scores','forecast-scores','forecasts'),('joint_clock_scores','forecast-scores','joint_forecasts'),
        ('variance_scores','variance-scores','forecasts'),('variance_aggregate','variance-scores','aggregate'),
        ('paired_horizons','paired-paths','paired_horizons'),('paired_arithmetic','collectives','paired_arithmetic'),
        ('conditional_noise','conditional-noise','conditions'),('matched_forcing','matched-forcing','states'),
        ('finite_environment_factorial','environment-factorial','conditions'),('temporal_paths','shared-updates','runs')]
    for key,component,field in mirrors:
        if report[key] != reports[component][field]:raise AssertionError('Complete evidence aggregation changed: '+key)
    if report['source_basis'] != reports['source-basis']:raise AssertionError('An expanded source comparison was omitted')
    if report['visible_noise'] != reports['visible-noise']:raise AssertionError('A fresh-source outcome was omitted')
    if report['native_step'] != reports['native-step']:raise AssertionError('A complete native-step source outcome was omitted')
    if report['training_risk'] != reports['training-risk']:raise AssertionError('A training-risk outcome was omitted')
    if report['common_geometry'] != reports['common-geometry']:raise AssertionError('A common-centroid geometry outcome was omitted')
    files.check(reports['common-geometry']['parent_collectives'],reports['common-geometry']['parent_collectives_sha256'])
    if len(reports['common-geometry']['conditions']) != len(reports['collectives']['conditions']):
        raise AssertionError('Common geometry predates the complete collective selection')
    if report['clock_observation_sensitivity'] != reports['clock-observation-sensitivity']:
        raise AssertionError('Clock observation sensitivity differs from its complete component')
    counts = stats['checks']
    if counts['collectives']['conditions'] != len(reports['collectives']['conditions']):
        raise AssertionError('Statistical reconstruction predates the complete observation panel')
    if counts['paired_paths']['paired_groups'] != len(reports['paired-paths']['paired_horizons']):
        raise AssertionError('Statistical reconstruction predates the complete paired panel')

    registry_path = Path(a.producer_registry)
    registry = read(registry_path)
    if registry['status'] != 'complete' or a.study not in registry['studies']:
        raise AssertionError('The complete producer inventory is required')
    for name,signature in registry['current_sources'].items():files.check(repo/name,signature)
    artifacts = {r['artifact']:r for r in registry['artifacts']}
    selected_artifacts = [f'{a.study}/runs/'+r['run_id'] for r in raw['training']]
    selected_artifacts += [f'{a.study}/measurements/'+r['name'] for r in raw['observations']]
    selected_artifacts += [f'{a.study}/measurements/'+r['name'] for rows in diagnostics.values() for r in rows]
    selected_artifacts += [f'{a.study}/analysis/'+name for name in COMPONENTS|{'complete'}]
    for name in selected_artifacts:
        if name not in artifacts:raise AssertionError('A selected artifact has no preserved producer: '+name)
        files.check(root/name/'binding.json',artifacts[name]['binding_sha256'])
    for name,signature in sources.items():files.check(repo/name,signature)
    result = dict(schema='scaling-complete-evidence-v1',status='passed',
        required_scientific_runs=observation['required_scientific_runs']+raw['training_artifacts'],
        inventory_definition='416 verified scientific artifacts plus96 native training artifacts, including continuations. The224 frozen CPU states,24 conditional-noise states,16 fresh-source validations,16 expanded source-basis validations,16 native conditional-step states,32 frozen training-risk states and5 spectator controls are separately enumerated; none is counted as an independent pretraining.',
        inherited_verifications=dict(observation=observation_binding),
        observation_analysis=observation['observation_analysis'],observation_analysis_sha256=observation['observation_analysis_sha256'],
        scaling_analysis=str(complete/'results.json'),scaling_analysis_sha256=files.digest(complete/'results.json'),
        raw_verification=raw_binding,statistical_verification=stats_binding,native_step_verification=replay_binding,
        training_risk_verification=risk_binding,
        producer_registry=dict(path=str(registry_path.resolve()),sha256=files.digest(registry_path)),
        numerical_verification=numerical_binding,
        formal_statement_verification=observation['formal_statement_verification'],
        formal_axiom_verification=dict(path=str(axiom_path),sha256=files.digest(axiom_path),statements=53),
        training_artifacts=raw['training_artifacts'],additional_updates=raw['additional_updates'],
        maximum_completed_horizon=report['maximum_completed_horizon'],frozen_observation_states=raw['frozen_observation_states'],
        diagnostics=diagnostics,components=report['components'],verifier_sources=sources,
        verifier_sha256=sha256(__file__),hashed_file_versions=len(files.cache),
        scope='Complete recorded finite scientific evidence and selected proof/test scope. This gate verifies provenance, inventory and the separate raw/statistical reconstructions; it does not certify native critical exponents or self-organized criticality.')
    write_json(a.output,result)
    print('Complete scaling evidence passed:',result['required_scientific_runs'],'required scientific artifacts',flush=True)


if __name__=='__main__':main()
