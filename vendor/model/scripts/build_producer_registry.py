#!/usr/bin/env python
"""Bind every inventoried artifact to its actual preserved producer and project imports."""
import argparse
import ast
from collections import Counter, defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import re
from model_rg.provenance import sha256, write_json


def producer_for(study,category,name,args,observation_study='observation-closure-20260906'):
    if study.startswith('scheduled-training-feasible-'):
        if category == 'runs': return 'train_onepass_regimes.py'
        if category == 'checkpoint': return 'prepare_repeated_comparison.py'
        if category == 'qualification' and name == 'onepass-stream': return 'qualify_onepass_stream.py'
        if category == 'repeated-risk': return 'measure_repeated_comparison_risk.py'
        if category == 'prefix-risk': return 'measure_prefix_risk_comparison.py'
        if category == 'decoder-prefix-risk': return 'measure_decoder_prefix_intervention.py'
        if category == 'measurements':
            for prefix, producer in [
                    ('cpu-', 'measure_scaling_checkpoints.py'), ('risk-', 'measure_onepass_risk.py'),
                    ('inference-', 'measure_scheduled_inference.py'), ('prefix-', 'measure_scheduled_prefix.py'),
                    ('reasoning-', 'measure_reasoning_mechanism.py'),
                    ('input-domain-', 'measure_onepass_input_domain.py'),
                    ('onepass-row-jacobian-', 'measure_onepass_row_jacobians.py'),
                    ('onepass-native-step-', 'measure_onepass_native_step.py')]:
                if name.startswith(prefix): return producer
        if category == 'analysis':
            if name.startswith('repetition-'): return 'analyze_repetition_comparison.py'
            exact = {'onepass-collectives': 'analyze_onepass_collectives.py',
                'onepass-predictions': 'analyze_onepass_predictions.py',
                'onepass-training-paths': 'analyze_onepass_paths.py',
                'decoder-prefix-interventions': 'analyze_decoder_prefix_interventions.py',
                'source-reuse': 'analyze_source_reuse.py',
                'onepass-schedule-comparison': 'analyze_onepass_schedule_comparison.py'}
            if name in exact: return exact[name]
        raise ValueError('Unmapped single-pass producer: '+str((study, category, name, args)))
    if study.startswith('scheduled-training-'):
        if category == 'runs': return 'train_scheduled_regimes.py'
        if category == 'checkpoint': return 'watch_early_scheduled_prefix.py'
        if category == 'data' and name.startswith('reasoning-cohort'): return 'prepare_reasoning_cohort.py'
        if category == 'data' and name == 'prompt-conditioning-cohort': return 'prepare_prompt_conditioning.py'
        if category == 'data' and name == 'reference-benchmark-cohort': return 'prepare_reference_benchmarks.py'
        if category == 'qualification':
            if name.startswith('reference-cpu'): return 'check_reference_schedule.py'
            if name == 'reference-regimes': return 'check_reference_regimes.py'
            if name == 'inference-interventions': return 'qualify_inference_interventions.py'
        if category == 'measurements':
            for prefix, producer in [('cpu-', 'measure_scaling_checkpoints.py'),
                    ('risk-', 'measure_scheduled_risk.py'), ('inference-', 'measure_scheduled_inference.py'),
                    ('reasoning-', 'measure_reasoning_mechanism.py'), ('prompt-', 'measure_prompt_conditioning.py'),
                    ('benchmark-', 'measure_reference_benchmarks.py'),
                    ('row-transport-', 'measure_row_input_transport.py'),
                    ('row-jacobian-', 'measure_row_jacobians.py'), ('early-prefix-', 'measure_scheduled_prefix.py')]:
                if name.startswith(prefix): return producer
            if name.startswith('prefix-'):
                return 'probe_prefix_mechanism.py' if 'protocol' in args else 'measure_scheduled_prefix.py'
        if category == 'analysis':
            exact = {'initial-inference-mechanism': 'analyze_initial_inference_mechanism.py',
                'collectives': 'analyze_scheduled_collectives.py', 'predictions': 'analyze_scheduled_predictions.py',
                'frozen-clocks': 'freeze_scheduled_clocks.py', 'clock-scores': 'score_scheduled_clocks.py',
                'training-paths': 'analyze_scheduled_paths.py', 'complete': 'analyze_scheduled_study.py',
                'prompt-conditioning': 'analyze_prompt_conditioning.py',
                'reference-benchmarks': 'analyze_reference_benchmarks.py'}
            if name in exact: return exact[name]
    if study.startswith('critical-scaling-'):
        if category=='runs':return 'train_scaling.py'
        if category=='measurements':
            if name.startswith('training-risk-'):return 'measure_scaling_training_risk.py'
            if 'native-step-' in name:return 'measure_scaling_native_step.py'
            if 'source-basis-' in name:return 'measure_scaling_source_basis.py'
            if 'visible-noise-' in name:return 'measure_scaling_visible_noise.py'
            if name.startswith('cpu-'):return 'measure_scaling_checkpoints.py'
            if 'noise-' in name:return 'measure_scaling_noise.py'
            if name.startswith('spectator-'):return 'measure_spectator_control.py'
        if category=='qa':
            if name=='native-step-qualification-analysis':return 'analyze_scaling_native_step.py'
            if name=='shared-update-native-control':return 'prepare_shared_updates.py'
            if name.startswith('native-'):return 'check_scaling_smoke.py'
        if category=='analysis':
            if 'include_current' in args:return 'analyze_size_time.py'
            if 'prefix' in args:return 'analyze_scaling_collectives.py'
            exact={'frozen-forecasts':'freeze_scaling_forecasts.py',
                'frozen-variance':'freeze_scaling_variance.py','variance-scores':'score_scaling_variance.py',
                'conditional-noise':'analyze_scaling_noise.py','forecast-scores':'score_scaling_forecasts.py',
                'forcing-geometry':'analyze_scaling_forcing.py',
                'matched-forcing':'analyze_scaling_matched_forcing.py','visible-noise':'analyze_scaling_visible_noise.py',
                'source-basis':'analyze_scaling_source_basis.py',
                'native-step':'analyze_scaling_native_step.py',
                'training-risk':'analyze_scaling_training_risk.py',
                'common-geometry':'analyze_scaling_common_geometry.py',
                'baseline-common-geometry':'analyze_scaling_common_geometry.py',
                'clock-observation-sensitivity':'analyze_clock_observation_sensitivity.py',
                'shared-updates':'analyze_scaling_updates.py','environment-factorial':'analyze_scaling_environment.py',
                'paired-paths':'analyze_scaling_paths.py',
                'complete':'analyze_scaling_study.py'}
            if name in exact:return exact[name]
    if study==observation_study or study.startswith('observation-closure-'):
        if category=='runs':return 'measure_arithmetic_observation.py' if name.startswith('arithmetic-') else 'train_dynamics.py'
        if category=='analysis':return 'analyze_row_projection.py' if name=='row-projection' else 'analyze_observation_study.py'
    if study=='controlled-study-20260905':
        if category=='analysis':return 'analyze_supporting.py' if name=='supporting' else 'analyze_controlled.py'
        if category=='runs':
            if name.startswith('drift-'):return 'measure_conditional_drift.py'
            if name.startswith('quality-'):return 'measure_quality_controls.py'
            if name.startswith('long-'):return 'collect_long_segments.py'
            if 'heads' in args:return 'train_controlled.py'
    if study=='criticality-study-20260905':
        if category in ['runs','training-replays']:
            return 'measure_training_restoration.py' if 'parent' in args else 'train_criticality.py'
        if category=='attention-regimes':return 'measure_attention_regime.py'
        if category=='source-precision':return 'measure_gain_precision.py'
        if category=='initialization-kernel-attempts':return 'measure_initial_kernel_comparison.py'
        if category=='analysis':
            exact={'attention-regimes':'analyze_attention_regimes.py','horizon':'analyze_horizon.py',
                'initialization-family':'measure_initialization_family.py','initialization-kernel-comparison':'measure_initial_kernel_comparison.py',
                'initialization-limit':'measure_initialization_limit.py','restoration':'analyze_training_restoration.py',
                'source-precision':'analyze_gain_precision.py','source-precision-checked':'analyze_gain_precision.py',
                'training-replay':'analyze_training_replay.py','width-holdout':'analyze_width_holdout.py',
                'width-prediction':'predict_width_holdout.py'}
            return exact.get(name,'analyze_criticality.py')
    if study=='criticality-dynamics-20260906':
        if category=='runs':return 'measure_collective_return.py' if 'case' in args else 'train_dynamics.py'
        if category in ['analysis','qa']:
            if 'allow_partial' in args:return 'analyze_dynamics.py'
            if 'amplitudes' in args:return 'measure_adam_tangent.py'
            if 'reference' in args:return 'measure_native_transport.py'
            if 'case' in args:
                for prefix,producer in [('row-gradient','measure_row_gradient_projection.py'),('row-adjoint','measure_row_adjoint.py'),
                    ('row-projection','measure_row_projection.py'),('row-transport','measure_row_transport.py'),('shared-crossing','measure_shared_crossing.py')]:
                    if name.startswith(prefix):return producer
            for prefix,producer in [('row-gradient','analyze_row_gradient_projection.py'),('row-adjoint','analyze_row_adjoint.py'),
                ('row-projection','analyze_row_projection.py'),('row-transport','analyze_row_transport.py'),
                ('optimizer-','analyze_optimizer_scales.py'),('metric-collectives','analyze_metric_collectives.py'),
                ('replication','analyze_dynamics_replication.py'),('controls','analyze_dynamics_controls.py')]:
                if name.startswith(prefix):return producer
    raise ValueError('Unmapped producer: '+str((study,category,name,args)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--study',default='observation-closure-20260906')
    p.add_argument('--scaling-study',help='Additional completed size-time study, when included in the release')
    p.add_argument('--scheduled-study', help='Additional scheduled training and inference mechanism study')
    p.add_argument('--onepass-study', help='Single-pass training and matched data-law observations')
    a=p.parse_args()
    root=Path(a.root);repo=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    studies=['controlled-study-20260905','criticality-study-20260905','criticality-dynamics-20260906',a.study]
    if a.scaling_study:studies.append(a.scaling_study)
    if a.scheduled_study:studies.append(a.scheduled_study)
    if a.onepass_study:studies.append(a.onepass_study)
    if len(studies) != len(set(studies)): raise AssertionError('Study inventory contains duplicates')
    current={str(p.relative_to(repo)):sha256(p) for base in [repo/'src',repo/'scripts'] for p in base.rglob('*.py')}
    cache={};rows=[];versions={};families=defaultdict(list)
    def payload(folder,sources,name):
        key=(name,sources[name])
        if key not in cache:
            path=folder/'source'/name;data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=sources[name]:raise AssertionError('Changed preserved source: '+str(path))
            cache[key]=data.decode()
        return cache[key]
    def dependencies(folder,sources,producer):
        pending=[producer];found={}
        while pending:
            name=pending.pop()
            if name in found:continue
            text=payload(folder,sources,name);found[name]=sources[name]
            for node in ast.walk(ast.parse(text)):
                mods=[node.module] if isinstance(node,ast.ImportFrom) and node.module else ([v.name for v in node.names] if isinstance(node,ast.Import) else [])
                for mod in mods:
                    if mod=='model_rg':candidate='src/model_rg/__init__.py'
                    elif mod.startswith('model_rg.'):candidate='src/'+mod.replace('.','/')+'.py'
                    else:candidate='scripts/'+mod.replace('.','/')+'.py'
                    if candidate in sources and candidate not in found:pending.append(candidate)
        return found
    for study in studies:
        categories = [c for c in sorted((root/study).iterdir()) if c.is_dir()]
        if study.startswith('scheduled-training-'):
            categories = [c for c in categories if c.name != 'early-checkpoints']
            categories += [c for c in sorted((root/study/'early-checkpoints').glob('*/measurements')) if c.is_dir()]
            categories += [c for c in sorted((root/study/'early-checkpoints').glob('*')) if c.is_dir()]
        if study == a.onepass_study:
            # Explicit nested observations, never a recursive source-tree scan.
            categories += [root/study/'measurements'/name for name in
                           ['repeated-risk', 'prefix-risk', 'decoder-prefix-risk']]
            categories += [root/study/name/'measurements' for name in
                           ['observer-qualification', 'repeated-observations', 'conditional-step-qualification']]
            categories += sorted((root/study/'repeated-checkpoints').glob('*'))
            categories = [c for c in categories if c.is_dir()]
        if len(categories) != len(set(categories)): raise AssertionError('Repeated artifact category')
        for category in categories:
            for binding in sorted(category.glob('*/binding.json')):
                folder=binding.parent;record=json.loads(binding.read_text());args=record['arguments'];sources=record['source_files']
                producer='scripts/'+producer_for(study,'checkpoint' if study.startswith('scheduled-training-') and folder.name=='checkpoint' else category.name,folder.name,args,a.study)
                if producer not in sources:raise AssertionError('Selected producer absent from preserved snapshot: '+str(folder)+' / '+producer)
                deps=dependencies(folder,sources,producer)
                key=hashlib.sha256(json.dumps(deps,sort_keys=True).encode()).hexdigest()
                changed={name:dict(recorded=digest,current=current.get(name)) for name,digest in deps.items() if current.get(name)!=digest}
                differences=[]
                for name,value in changed.items():
                    before=payload(folder,sources,name);after=(repo/name).read_text() if (repo/name).exists() else ''
                    before_funcs={n.name for n in ast.walk(ast.parse(before)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
                    after_funcs={n.name for n in ast.walk(ast.parse(after)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
                    lines=list(difflib.unified_diff(before.splitlines(),after.splitlines(),fromfile='recorded/'+name,tofile='current/'+name,lineterm=''))
                    delta=out/'source-differences'/(value['recorded']+'-'+Path(name).name+'.diff')
                    if not delta.exists():delta.parent.mkdir(exist_ok=True);delta.write_text('\n'.join(lines)+'\n')
                    differences.append(dict(source=name,added_functions=sorted(after_funcs-before_funcs),removed_functions=sorted(before_funcs-after_funcs),
                        diff=str(delta.relative_to(out)),diff_sha256=sha256(delta)))
                # This is an explicit artifact inventory, not a classifier-derived count of scientific runs.
                role='measurement_or_analysis'
                if any(w in folder.name for w in ['interim','progress','smoke','check','prototype','attempt']) or category.name in ['qa','initialization-kernel-attempts']:
                    role='development_or_implementation_record'
                if folder.name=='source-precision-checked':role='measurement_or_analysis'
                if study.startswith('critical-scaling-') and (folder.name.startswith('qa-') or folder.name=='common-sector-initial'):
                    role='development_or_implementation_record'
                if study.startswith('critical-scaling-') and category.name=='runs' and not (folder/'manifest.json').exists():
                    role='incomplete_implementation_or_resource_record'
                if study.startswith('scheduled-training-') and (category.name=='qualification' or folder.name.startswith('qualification-') or 'preparation' in folder.name):
                    role='development_or_implementation_record'
                if study.startswith('scheduled-training-') and category.name == 'runs' and not (folder/'manifest.json').exists():
                    role='incomplete_or_administratively_censored_native_path'
                if study == a.onepass_study and ('observer-qualification' in folder.parts or 'conditional-step-qualification' in folder.parts or 'qualification' in folder.name):
                    role='development_or_implementation_record'
                if study.startswith('scheduled-training-') and folder.name=='checkpoint':
                    role='sealed_copy_of_existing_native_state'
                excluded='Excluded adjoint sampler prototype: offset RNG transcription error.' if folder.name=='row-adjoint-sampler-prototype' else None
                inline=study=='criticality-dynamics-20260906' and producer in ['scripts/measure_row_transport.py','scripts/measure_row_projection.py','scripts/measure_row_adjoint.py']
                entry=dict(artifact=str(folder.relative_to(root)),binding_sha256=sha256(binding),role=role,exclusion=excluded,
                    recorded_commit=record.get('git_head'),dirty_tree=bool(record.get('git_status','')),recorded_git_status=record.get('git_status',''),
                    executed_entrypoint=producer,entrypoint_basis='Recorded artifact arguments and preserved producer; row diagnostics used inline commands.' if inline else 'Recorded artifact arguments, family protocol and preserved producer.',
                    launcher_role='Subsequent equivalent protocol encoding, not an assertion of launcher execution.' if inline else 'Use the bound execution ledger where present.',
                    producer_sha256=sources[producer],project_imports=deps,
                    bound_library_snapshot={n:h for n,h in sources.items() if n.startswith('src/model_rg/')},
                    input_hashes=record['inputs'],arguments=args,source_snapshot=str((folder/'source').relative_to(root)),version=key,
                    differences_from_current=changed,equivalence='Exact producer and statically resolved project-import source identity, conditional on the same arguments, inputs and package environment.' if not changed else 'No numerical equivalence asserted; the preserved source version remains the reproduction authority.',
                    environment=record.get('environment'))
                rows.append(entry);families[study+'/'+producer].append(entry)
                versions[key]=dict(producer=producer,source_hashes=deps,differences=differences)
    if len(rows) != len({r['artifact'] for r in rows}): raise AssertionError('Artifact counted more than once')
    # Corpus protocols have no binding.json snapshot and are reported separately.
    datasets=[]
    if a.onepass_study:
        for name, producer in [('refinedweb-expanded-307200', 'prepare_expanded_refinedweb.py'),
                               ('refinedweb-onepass-524288', 'prepare_onepass_refinedweb.py')]:
            folder=root/'data'/name; path=folder/'manifest.json'; meta=json.loads(path.read_text())
            protocol=Path(meta['protocol'])
            if meta['status'] != 'complete' or sha256(protocol) != meta['protocol_sha256']:
                raise AssertionError('Incomplete or changed training corpus')
            spec=json.loads(protocol.read_text()); deps={}
            for source,digest in meta['source_files'].items():
                source=Path(source)
                if sha256(source) != digest or spec['source_files'][str(source)] != digest:
                    raise AssertionError('A corpus producer changed without a preserved source authority')
                deps[str(source.relative_to(repo))]=digest
            entrypoint='scripts/'+producer
            if entrypoint not in deps: raise AssertionError('Corpus entrypoint is unbound')
            outputs={filename:sha256(folder/filename) for filename in ['tokens.npy','records.json']}
            if outputs != {'tokens.npy':meta['tokens_sha256'], 'records.json':meta['records_sha256']}:
                raise AssertionError('A complete corpus payload changed')
            datasets.append(dict(artifact=str(folder.relative_to(root)),manifest_sha256=sha256(path),
                protocol=str(protocol),protocol_sha256=sha256(protocol),executed_entrypoint=entrypoint,
                project_sources=deps,outputs=outputs,shape=meta['shape'],
                authority='Protocol-bound current producer and complete token/source-record payloads; no historical source snapshot is claimed.'))
    # The first inference study predates binding.json. Keep its source-object
    # authority explicit; absent commit/dirty metadata is not reconstructed.
    legacy_index=root/'provenance/source-index.json'
    legacy=json.loads(legacy_index.read_text())
    if legacy['missing']:raise AssertionError('Legacy source objects are missing')
    grouped=defaultdict(list)
    for row in legacy['source_bindings']:grouped[row['record']].append(row)
    legacy_records=[]
    for record_name,entries in sorted(grouped.items()):
        record=json.loads((root/record_name).read_text());objects={e['source']:e for e in entries}
        if {n:e['sha256'] for n,e in objects.items()}!=record['source_files']:
            raise AssertionError('Legacy source inventory differs from its original manifest')
        name=Path(record_name).parent.name
        if record_name.startswith('analysis/'):
            producer='analyze_training.py' if name.startswith('training') else 'analyze.py'
        else:
            producer=next((v for prefix,v in [('features-','collect_features.py'),('response-','measure_response.py'),
                ('segments-','collect_segments.py'),('training-derivative-','validate_training_derivative.py'),
                ('training-response-','measure_training_response.py'),('training-','train_width_family.py'),
                ('native-validation-','validate_model.py')] if name.startswith(prefix)),None)
        if producer is None:raise AssertionError('Unmapped legacy entrypoint')
        producer='scripts/'+producer;pending=[producer];deps={}
        while pending:
            n=pending.pop()
            if n in deps:continue
            entry=objects[n];source=root/entry['object']
            if sha256(source)!=entry['sha256']:raise AssertionError('Changed legacy source object')
            deps[n]=entry['sha256']
            for node in ast.walk(ast.parse(source.read_text())):
                mods=[node.module] if isinstance(node,ast.ImportFrom) and node.module else ([v.name for v in node.names] if isinstance(node,ast.Import) else [])
                for mod in mods:
                    candidate=('src/model_rg/__init__.py' if mod=='model_rg' else 'src/'+mod.replace('.','/')+'.py' if mod.startswith('model_rg.') else 'scripts/'+mod.replace('.','/')+'.py')
                    if candidate in objects and candidate not in deps:pending.append(candidate)
        differences={n:dict(recorded=h,current=current.get(n)) for n,h in deps.items() if h!=current.get(n)}
        legacy_records.append(dict(record=record_name,record_sha256=sha256(root/record_name),
            executed_entrypoint=producer,producer_sha256=deps[producer],project_imports=deps,
            source_objects={n:objects[n]['object'] for n in deps},arguments=record.get('arguments'),
            environment=record.get('environment'),recorded_commit=None,dirty_tree=None,
            metadata_limit='Commit and dirty-tree status were not recorded in these manifests; source objects are the reproduction authority.',
            input_and_output_hash_fields={k:v for k,v in record.items() if k.endswith('_sha256')},
            differences_from_current=differences,equivalence='No numerical equivalence is inferred from a current script; use the content-addressed original producer and imports.'))
    legacy_record=dict(index_path=str(legacy_index),index_sha256=sha256(legacy_index),
        manifests=len(legacy_records),records=legacy_records)
    family_rows=[]
    for family,items in sorted(families.items()):
        family_rows.append(dict(family=family,artifacts=len(items),roles=dict(Counter(r['role'] for r in items)),
            versions=dict(Counter(r['version'] for r in items)),recorded_commits=dict(Counter(r['recorded_commit'] for r in items)),
            dirty_records=sum(r['dirty_tree'] for r in items),identical_source_records=sum(not r['differences_from_current'] for r in items)))
    write_json(out/'registry.json',dict(schema='artifact-producer-registry-v2',status='complete',
        inventory_definition='Every binding.json exactly two directory levels below each named study, plus explicitly nested sealed checkpoints, observer qualifications, retained-state observations and causal-risk measurements. Corpus protocol records are listed separately. Development and censored records are included with their roles. Artifact counts are separate from independent training identities.',
        dependency_scope='Statically resolved project imports are distinguished from native modules dynamically loaded from bound asset inputs and third-party packages identified by the recorded environment.',
        studies=studies,artifact_count=len(rows),family_count=len(family_rows),source_version_count=len(versions),
        legacy_inference=legacy_record,corpus_records=datasets,families=family_rows,versions=versions,artifacts=rows,producer_sha256=sha256(__file__),current_sources=current,
        exclusions_without_run_binding=[dict(path='criticality-dynamics-20260906/qa/concentrated-statistics-before-fix.log',
            reason='Pre-fix analyzer implementation test, excluded from scientific evidence; the corrected central-moment implementation has separate passing tests.')]))
    lines=['# Artifact producer registry','',f'{len(rows)} explicitly inventoried bound artifacts; {len(family_rows)} producer families; {len(versions)} executed source versions.',
        '', 'The JSON stores commits, dirty status, actual producer and project-import hashes, source snapshots, input hashes, differences and coverage boundaries. Source identity is conditional on unchanged arguments, inputs and external packages. No untested numerical equivalence is inferred from a dirty commit.',
        '', f"The legacy source-object index separately binds {len(legacy_records)} original inference/training manifests. Missing historical commit metadata is explicit.", '', '| Family | Artifacts | Source versions | Identical source |','|---|---:|---:|---:|']
    for r in family_rows:lines.append(f"| {r['family']} | {r['artifacts']} | {len(r['versions'])} | {r['identical_source_records']} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('Registered',len(rows),'artifacts and',len(versions),'source versions',flush=True)


if __name__=='__main__':main()
