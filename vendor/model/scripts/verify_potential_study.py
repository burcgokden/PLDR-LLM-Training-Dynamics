#!/usr/bin/env python3
"""Independently verify the selected potential study and bind publication inputs."""
from companion_paths import legacy_path
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
ROOT=Path(legacy_path('/pldr-data/model/potential-avalanche-20260913'))
REPO=Path(__file__).resolve().parents[1]
PREVIOUS=Path(legacy_path('/pldr-archive/model/paper-outputs/paper-outputs-rev23'))
CURRENT=ROOT.parent/'potential-factorial-20260913'
REPLICATION=ROOT.parent/'potential-factorial-disjoint-20260914'
ANALYSIS=CURRENT/'retained-analysis'

def verify_manifest(path):
    for line in (path/'MANIFEST.sha256').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha256(path/name)!=digest:raise ValueError('Changed retained publication artifact: '+name)


def independent_events(x):
    # Fixed primary assay: postwarm probe0, native update cadence, 90th percentile.
    cut=float(np.quantile(x,.9));events=[];start=None
    for i,value in enumerate(x):
        if value>cut and start is None:start=i
        if value<=cut and start is not None:
            if start>0:events.append((i-start,float(np.sum(x[start:i]-cut))))
            start=None
    # A run continuing to the final sample is right-censored.
    return events


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--routes',required=True);p.add_argument('--metric-verification',required=True);p.add_argument('--emission',required=True);p.add_argument('--factorial-reconstruction',required=True);a=p.parse_args()
    spec=load_json_strict((ROOT/'protocols/training.json').read_text());summary=load_json_strict((ANALYSIS/'summary.json').read_text())
    follow=load_json_strict((ANALYSIS/'followups-summary.json').read_text())
    if summary['status']!='complete' or follow['status']!='complete':raise ValueError('All selected paths must be complete')
    if len(spec['jobs'])!=18 or summary['complete_paths']!=18:raise ValueError('Changed family')
    if len(follow['continuations'])!=4 or len(follow['relaxation'])!=4:raise ValueError('Changed follow-up family')
    data_root=ROOT.parent/'data/refinedweb-onepass-524288'
    data_meta=load_json_strict((data_root/'manifest.json').read_text())
    if sha256(data_root/'tokens.npy')!=data_meta['tokens_sha256']:raise ValueError('Changed corpus token array')
    if sha256(data_root/'records.json')!=data_meta['records_sha256']:raise ValueError('Changed corpus records')
    selected=np.load(ROOT/'selection.npz')
    probe=ROOT.parent/'controlled-study-20260905/data/short'
    pt=np.load(probe/'tokens.npy',mmap_mode='r');po=np.load(probe/'offsets.npy');pr=selected['probe_rows']
    if not np.array_equal(selected['probes'],pt[pr[:,None],po[pr,None]+np.arange(65)]):raise ValueError('Changed fixed probes')
    analyses={r['run_id']:r for r in summary['run_results']};checks=[]
    for job in spec['jobs']:
        path=ROOT/'runs'/job['run_id'];m=load_json_strict((path/'manifest.json').read_text());r=analyses[job['run_id']]
        if m['status']!='complete' or m['job']!=job or r['manifest_sha256']!=sha256(path/'manifest.json'):raise ValueError('Changed run binding')
        for f,h in m['artifacts'].items():
            if sha256(path/f)!=h:raise ValueError('Changed run artifact')
        for f,h in m['producer_sources'].items():
            if sha256(REPO/f)!=h:raise ValueError('Changed scientific producer '+f)
        z=np.load(path/'activity.npz')
        expected=(2049,2,5,job['heads'],14)
        if z['fields'].shape!=expected:raise ValueError('Changed time/probe/head axes')
        for k in z.files:
            if not np.isfinite(z[k]).all():raise ValueError('Nonfinite raw field '+k)
        if not np.array_equal(z['rows'],selected[f'rows_{job["seed"]}']) or not np.array_equal(z['offsets'],selected[f'offsets_{job["seed"]}']):raise ValueError('Selected stream changed')
        if np.unique(z['rows']*8+z['offsets']//64).size!=65536:raise ValueError('Block repetition')
        f=z['fields'][1:];energy=np.mean(f[...,0]**2,axis=(2,3))
        recovered=np.mean(f[...,1]**2+f[...,2]**2+2*f[...,3],axis=(2,3))
        np.testing.assert_allclose(energy,recovered,atol=1e-13,rtol=1e-11)
        x=np.sqrt(energy[256:,0])
        for primary in r['primary']:
            xx=x if primary['clock']=='update' else x/z['lr'][256:]
            es=independent_events(xx)
            if len(es)!=primary['events']:raise ValueError('Independent event count mismatch')
            if es and not np.isclose(np.mean([e[0] for e in es]),primary['mean_duration'],rtol=1e-13):raise ValueError('Duration mismatch')
            fit=primary['tail_diagnostics']
            if fit['status']=='fitted':
                tail=np.sort([e[1] for e in es if e[1]>=fit['xmin']]);alpha=1+len(tail)/np.log(tail/fit['xmin']).sum()
                if len(tail)!=fit['n_tail'] or not np.isclose(alpha,fit['alpha'],rtol=1e-12):raise ValueError('Independent tail fit mismatch')
        checks.append(dict(run_id=path.name,unique_blocks=65536,events=r['primary'][0]['events'],identity_max_error=float(np.max(np.abs(energy-recovered)))))
    for item in follow['continuations']:
        path=ROOT/'continuations'/item['case']['name'];m=load_json_strict((path/'manifest.json').read_text())
        if not m['no_repetition_including_parent'] or m['unique_prefix_blocks']!=67584*32:raise ValueError('Wrong remaining-corpus continuation')
        for f,h in m['artifacts'].items():
            if sha256(path/f)!=h:raise ValueError('Changed continuation artifact')
        for f,h in m['producer_sources'].items():
            if sha256(REPO/f)!=h:raise ValueError('Changed continuation producer')
        z=np.load(path/'activity.npz')
        if z['tensor_statistics'].shape!=(2049,2,5,14,5,5):raise ValueError('Changed native deductive tensor axes')
        if z['tensor_samples'].shape!=(2049,2,5,14,5,128):raise ValueError('Changed native entry panel')
        if len(np.unique(z['rows']*8+z['offsets']//64))!=65536:raise ValueError('Repeated continuation block')
        initial=np.mean(z['fields'][0,...,8],axis=(1,2));final=np.mean(z['fields'][-1,...,8],axis=(1,2))
        np.testing.assert_allclose(initial,item['initial_row_ratio'],rtol=1e-12);np.testing.assert_allclose(final,item['final_row_ratio'],rtol=1e-12)
        for k in z.files:
            if not np.isfinite(z[k]).all():raise ValueError('Nonfinite continuation '+k)
    for item in follow['relaxation']:
        for name,branch in item['branches'].items():
            p=ROOT/'relaxation'/item['case']['name']/(name+'.npz')
            if sha256(p)!=branch['artifact_sha256']:raise ValueError('Changed intervention')
            z=np.load(p);measured=float(np.sqrt(np.mean(z['fields'][1:,0,...,0]**2,axis=(1,2))).sum())
            if not np.isclose(measured,branch['summed_logpotential_activity'],rtol=1e-12):raise ValueError('Wrong relaxation statistic')
            if name=='frozen' and np.max(z['tensor_statistics'][1:,...,1])!=0:raise ValueError('Frozen tensors moved')
            if name!='frozen' and finite_greater(np.max(z['predicted_exponent_step_max_error']), 1e-6, 'scripts/verify_potential_study.py:98'):raise ValueError('Passive moment recurrence mismatch')
    for name in ['formal','clean','statement-mutations']:
        v=load_json_strict((REPO/'internal/revision24'/name/'verification.json').read_text())
        if v['status']!='passed':raise ValueError('Formal check failed '+name)
        for group in ['module_sources','modules','support_sources']:
            for f,h in v.get(group,{}).items():
                if sha256(REPO/f)!=h:raise ValueError('Changed formal source '+f)
        if 'registry_sha256' in v and sha256(REPO/'scripts/formal/statement-registry.json')!=v['registry_sha256']:raise ValueError('Changed formal registry')
    v=load_json_strict((REPO/'internal/revision24/statement-mutations/verification.json').read_text())
    if v['exports']!=116 or len(v['mutations'])!=65:raise ValueError('Unexpected formal inventory')
    numerical=load_json_strict((REPO/'internal/revision24/numerical/verification.json').read_text())
    if numerical['status']!='passed' or numerical['tests_passed']<232:raise ValueError('Numerical suite not complete')
    for f,h in numerical['tested_sources'].items():
        if sha256(REPO/f)!=h:raise ValueError('Numerical test dependency changed '+f)
    factorial=load_json_strict((REPLICATION/'retained-verification-final.json').read_text())
    if factorial['status']!='passed' or factorial['reconstruction_mode']!='retained-executed-source' or factorial['execution_permission'] is not False or factorial['verifier_sha256']!=sha256(REPO/'scripts/verify_potential_factorial.py'):raise ValueError('Missing current factorial reduction')
    for f,h in factorial['checked_sha256'].items():
        if sha256(f)!=h:raise ValueError('Changed factorial verification input '+f)
    additional={}
    for name,script in [('verification-final.json','verify_potential_factorial.py'),('area-verification-final.json','verify_potential_area.py')]:
        path=Path(a.factorial_reconstruction) if name=='verification-final.json' else REPLICATION/name;record=load_json_strict(path.read_text())
        if record['status']!='passed' or record['verifier_sha256']!=sha256(REPO/'scripts'/script):raise ValueError('Missing current reconstruction '+name)
        for f,h in record['checked_sha256'].items():
            if sha256(f)!=h:raise ValueError('Changed reconstruction input '+f)
        additional.update(record['checked_sha256']);additional[str(path)]=sha256(path)
        if name=='verification-final.json' and (record['scientific_updates']!=2048 or record['reconstruction_mode']!='current-schema'):raise ValueError('Missing current-schema scientific positive path')
        if name=='area-verification-final.json' and record['scale_cells']!=326:raise ValueError('Incomplete signed-area family')
    if sha256(a.factorial_reconstruction)!=sha256(REPO/'manuscript/generated/potential-replication-evidence/current-reconstruction.json'):raise ValueError('Stale executed-source reconstruction')
    emission=load_json_strict(Path(a.emission).read_text())
    if emission['status']!='passed' or emission['comparisons']!=16 or emission['step_comparisons']!=2048 or emission['analyzer_sha256']!=sha256(REPO/'scripts/analyze_finite_emission_transport.py'):raise ValueError('Missing finite categorical transport')
    for f,h in emission['checked_sha256'].items():
        if sha256(f)!=h:raise ValueError('Changed finite-emission input')
    additional.update(emission['checked_sha256'])
    if sha256(a.emission)!=sha256(REPO/'manuscript/generated/potential-replication-evidence/finite-emission-transport.json'):raise ValueError('Stale explicit finite-emission publication')
    additional[str(Path(a.emission).resolve())]=sha256(a.emission)
    replication=load_json_strict((REPLICATION/'analysis/replication-summary.json').read_text())
    if replication['status']!='passed' or replication['source_overlap_blocks']!=0 or not all(replication['confirmation'].values()):raise ValueError('Unsupported source-replication claim')
    for f,h in replication['analysis_sources'].items():
        if sha256(REPO/f)!=h:raise ValueError('Changed replication analysis source')
    for f,h in replication['inputs'].items():
        if sha256(f)!=h:raise ValueError('Changed replication input')
    for path in (REPLICATION/'analysis').glob('*.json'):
        if path.name=='finite-emission-transport.json':continue
        if sha256(path)!=sha256(REPO/'manuscript/generated/potential-replication-evidence'/path.name):raise ValueError('Stale compact replication evidence')
    area=load_json_strict((CURRENT/'analysis/block-area.json').read_text())
    if area['status']!='passed' or area['scale_cells']!=198 or area['analysis_sha256']!=sha256(REPO/'scripts/analyze_potential_block_area.py'):raise ValueError('Missing signed-area reconstruction')
    from verify_execution_routes import verify as verify_routes
    execution=verify_routes(a.routes)
    write_json(REPO/'internal/revision24/execution-publication.json',execution)

    verify_manifest(PREVIOUS)
    # Existing numerical evidence is retained byte-for-byte except the explicitly
    # regenerated current software-qualification table.
    for p in (PREVIOUS/'arxiv-source/generated').rglob('*'):
        if p.is_file() and 'potential-evidence' not in p.parts and p.name not in ['finite-emission-transport.json','formal-correspondence.tex','formal-correspondence.json','qualification-current.tex','qualification-current.json','current-execution-routes.tex','current-execution-routes.json','execution-ledger.tex','execution-ledger.json','potential-early.tex','potential-relaxation.tex','deductive-predictive.tex','potential_activity.pdf','potential_excursions.pdf','potential_continuations.pdf','deductive_comparison.pdf']:
            if sha256(p)!=sha256(REPO/'manuscript/generated'/p.relative_to(PREVIOUS/'arxiv-source/generated')):raise ValueError('Changed retained numerical evidence '+p.name)
    for name in ['summary.json','followups-summary.json','base-sensitivity.json','base-null.json']:
        if sha256(ANALYSIS/name)!=sha256(REPO/'manuscript/generated/potential-evidence'/name):raise ValueError('Stale compact activity evidence '+name)
    for name in ['factorial-summary.json','endpoint-panels.json','block-area.json']:
        if sha256(CURRENT/'analysis'/name)!=sha256(REPO/'manuscript/generated/potential-factorial-evidence'/name):raise ValueError('Stale compact factorial evidence '+name)
    baseline_ledger=load_json_strict((PREVIOUS/'arxiv-source/generated/execution-ledger.json').read_text())
    current_ledger=load_json_strict((REPO/'manuscript/generated/execution-ledger.json').read_text())
    if current_ledger['rows'][:len(baseline_ledger['rows'])]!=baseline_ledger['rows']:raise ValueError('Changed retained execution rows')
    if current_ledger['native_scientific_updates']-baseline_ledger['native_scientific_updates']!=24576:raise ValueError('Wrong new training inventory')
    if current_ledger['optimizer_only_control_updates']!=512:raise ValueError('Wrong optimizer-only inventory')
    qualification=load_json_strict((REPO/'manuscript/generated/qualification-current.json').read_text())
    if qualification['counts']!=dict(numerical_tests=numerical['tests_passed'],selected_statements=116,formal_modules=26,formal_fixtures=8,rejected_mutations=65):raise ValueError('Qualification table mismatch')
    metric_check=load_json_strict(Path(a.metric_verification).read_text())
    if metric_check['status']!='passed' or metric_check['verifier_sha256']!=sha256(REPO/'scripts/verify_metric_studies.py') or metric_check['endpoint_spans']!=28416 or metric_check['scientific_updates']!=24576 or metric_check['one_percent_refined_pairs']!=192:raise ValueError('Missing completed metric study verification')
    for f,h in metric_check['checked_sha256'].items():
        if sha256(f)!=h:raise ValueError('Changed metric study input '+f)
    for source,dest in [(Path(metric_check['metric_root'])/'metric-budget.json','finite-metric-budget.json'),(Path(metric_check['metric_root'])/'metric-refinement-final.json','finite-metric-refinement.json'),(Path(metric_check['matched_analysis']),'matched-onepass-analysis.json'),(Path(metric_check['matched_refinement']),'matched-onepass-refinement.json')]:
        if sha256(source)!=sha256(REPO/'manuscript/generated'/dest):raise ValueError('Stale metric publication '+dest)
    bound=dict(execution['checked_sha256']);bound.update(factorial['checked_sha256']);bound.update(additional)
    bound.update(metric_check['checked_sha256']);bound[str(Path(a.metric_verification).resolve())]=sha256(a.metric_verification)
    for folder in [REPO/'manuscript',REPO/'src',REPO/'scripts',REPO/'tests',REPO/'ModelRG',REPO/'internal/revision24',ROOT/'protocols',ANALYSIS,ROOT/'qualification',CURRENT/'analysis',REPLICATION/'analysis',REPO/'evidence/executed/potential-factorial']:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.lean','.json','.tex','.bib','.pdf','.png','.npz','.md'] and '__pycache__' not in p.parts and 'preview' not in p.parts and 'release-build' not in p.parts:
                if p.name in ['main.pdf'] or p.resolve()==Path(a.output).resolve():continue
                bound[str(p)]=sha256(p)
    for folder in ['runs','continuations','relaxation']:
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and (p.name in ['manifest.json','binding.json','activity.npz','final-state.pt'] or (folder=='relaxation' and p.suffix=='.npz')):
                bound[str(p)]=sha256(p)
    for name in ['docs/POTENTIAL_ACTIVITY_ASSESSMENT.md','docs/POTENTIAL_ACTIVITY_REPRODUCTION.md','README.md','docs/CURRENT_EXECUTION.md','docs/POTENTIAL_FACTORIAL_REPRODUCTION.md','docs/ASSET_ACCESS.md','docs/ASSET_ACCESS.json','docs/FINITE_METRIC_REPRODUCTION.md','docs/MATCHED_ONEPASS_REPRODUCTION.md','internal/revision24/numerical/verification.json']:
        bound[str(REPO/name)]=sha256(REPO/name)
    bound[str(PREVIOUS/'MANIFEST.sha256')]=sha256(PREVIOUS/'MANIFEST.sha256')
    write_json(a.output,dict(schema='potential-study-publication-v4',status='passed',verifier_sha256=sha256(__file__),
        early_paths=18,continuation_paths=4,zero_gradient_states=4,factorial_paths=32,total_new_training_updates=24576,new_scientific_paths=24,disjoint_source_replication=True,finite_emission_comparisons=16,
        execution_routes=execution['suite_counts'],current_positive_admission=True,factorial_verification_sha256=sha256(REPLICATION/'retained-verification-final.json'),current_factorial_verification_sha256=sha256(a.factorial_reconstruction),
        independent_event_and_tail_checks=checks,checked_sha256=bound,
        retained_publication=str(PREVIOUS),retained_manifest_verified=True,
        scope='Finite conditional activity, signed-area blocking and native paired intervention evidence. Current positive/adverse admission, raw scientific reconstruction, formal/numerical correspondence and retained-source integrity have separate checked identities.'))
    print('Verified 18 initial paths, four continuations, four intervention states, and retained publication evidence',flush=True)
if __name__=='__main__':main()
