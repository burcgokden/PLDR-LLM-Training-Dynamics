#!/usr/bin/env python3
"""Prepare the complete observed-data layer with relative paths and exact hashes."""
from companion_paths import legacy_path
import argparse
import json
from pathlib import Path
import shutil
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))


def package(output, nested_analysis=None, context_analyses=(), context_risk=()):
    if output.exists() or not output.is_relative_to(ROOT):raise ValueError('Use a fresh authorized package directory')
    output.mkdir(parents=True);files={};observations=[];source_bytes=0
    def copy(source,name,expected=None):
        nonlocal source_bytes
        h=sha256(source)
        if expected is not None and h!=expected:raise ValueError('Changed original observation '+str(source))
        dest=output/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
        if sha256(dest)!=h:raise ValueError('Copied data differ')
        files[name]=h;source_bytes+=dest.stat().st_size
    nested_analysis=nested_analysis or ROOT/'nested-categorical-20260915'
    context_analyses=context_analyses or [ROOT/'context-categorical-20260915/analysis']
    families=[('coarse',ROOT/'critical-onepass-coarse-20260914'),('refinement',ROOT/'critical-onepass-refinement-20260914')]
    analysis_families=[dict(family='refinement',times=[2048,3072,4096],reference='per_context',expected='refinement-expected.json')]
    for i,analysis in enumerate(context_analyses):
        plan=json.loads((analysis/'protocol.json').read_text())
        family='context' if i==0 else 'context'+str(i+1)
        families.append((family,Path(plan['study'])))
        analysis_families.append(dict(family=family,times=[4096],reference='shared',expected=family+'-expected.json'))
    for family,study in families:
        p=json.loads((study/'protocol.json').read_text());copy(study/'protocol.json','protocols/'+family+'.json')
        for job in p['jobs']:
            folder=study/'runs'/job['run_id'];m=json.loads((folder/'manifest.json').read_text())
            if m['status']!='complete':raise ValueError('Incomplete observation')
            name='observations/'+family+'/'+job['run_id']+'.npz'
            copy(folder/'observations.npz',name,m['observations_sha256'] if family.startswith('context') else m['artifacts']['observations.npz'])
            observations.append(dict(family=family,path=name,**job))
    copy(ROOT/'categorical-visibility-20260915/reference.npz','reference.npz')
    copy(nested_analysis/'analysis.json','refinement-expected.json')
    for item,analysis in zip(analysis_families[1:],context_analyses):copy(analysis/'analysis.json',item['expected'])
    for name in ['reproduce_categorical_observations.py','analyze_categorical_scales.py','numerical_validation.py','observed_package_contract.py','categorical_summary_contract.py']:
        copy(REPO/'scripts'/name,'code/'+name)
    for name in ['__init__.py','provenance.py']:copy(REPO/'src/model_rg'/name,'code/model_rg/'+name)
    risk_inventory=[]
    for family,record_name in context_risk:
        if family not in {f for f,_ in families if f.startswith('context')}:raise ValueError('Unknown context-risk family')
        record=Path(record_name);risk=json.loads(record.read_text())
        if risk['status']!='passed' or risk['source_sha256']!=sha256(REPO/'scripts/analyze_context_risk.py'):raise ValueError('Unverified context risk')
        counts={r['contexts'] for r in risk['rows']}
        if len(counts)!=1:raise ValueError('Context count differs')
        name=family+'-risk-expected.json';copy(record,name)
        risk_inventory.append(dict(family=family,expected=name,contexts=counts.pop()))
    if risk_inventory:
        for name in ['reproduce_context_risk_observations.py','analyze_context_risk.py']:
            copy(REPO/'scripts'/name,'code/'+name)
    expected_cells=sum(json.loads((output/item['expected']).read_text())['scale_cells'] for item in analysis_families)
    expected_chains=sum(json.loads((output/item['expected']).read_text())['path_context_kl_chains'] for item in analysis_families)
    readme=f"""# Complete categorical observed-data package

This complete local package contains {len(observations)} original observation files,
frozen protocols and dictionary, expected reductions, and portable CPU reducers.
From any directory, with Python and NumPy installed, set PACKAGE_ROOT to this
complete directory and OUTPUT_JSON to a fresh output filename:

    OPENBLAS_NUM_THREADS=1 python "$PACKAGE_ROOT/code/reproduce_categorical_observations.py" --package "$PACKAGE_ROOT" --output "$OUTPUT_JSON"

The reducer checks every indexed file, reconstructs the development reference,
and reproduces {expected_cells} scale cells and {expected_chains} KL chains.
Contexts and time slices share six complete training replicas per cell.
It performs zero native forwards and zero training updates. Native training
replay separately requires the original tokens and complete optimizer states.
The package is prepared locally; no public persistent identifier is assigned.
A source archive may carry metadata for this directory without these data.
"""
    if risk_inventory:
        readme+='\nContext-resolved reproduction uses the same complete directory:\n\n    OPENBLAS_NUM_THREADS=1 python "$PACKAGE_ROOT/code/reproduce_context_risk_observations.py" --package "$PACKAGE_ROOT" --output "$OUTPUT_JSON"\n\nIt reconstructs every per-context variance and target exceedance from logits.\n'
    (output/'README.md').write_text(readme);files['README.md']=sha256(output/'README.md')
    write_json(output/'INDEX.json',dict(schema='categorical-observations-package-v2',
        access_status='local_prepared_deposit',persistent_identifier=None,files=files,observations=observations,
        analysis_families=analysis_families,context_risk=risk_inventory,expected_scale_cells=expected_cells,expected_kl_chains=expected_chains,
        sizes=[128,512,2048,8192,16384],file_bytes=source_bytes+(output/'README.md').stat().st_size,
        original_training_paths=384,new_observed_checkpoints=sum(len(json.loads((study/'protocol.json').read_text())['jobs']) for family,study in families if family.startswith('context')),new_training_updates=0,
        packager_sha256=sha256(__file__)))
    print(json.dumps({'files':len(files),'observation_files':len(observations),'bytes':source_bytes}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--nested-analysis',type=Path);p.add_argument('--context-analysis',type=Path,action='append',default=[])
    p.add_argument('--context-risk',nargs=2,action='append',default=[],metavar=('FAMILY','RECORD'))
    a=p.parse_args();package(a.output.resolve(),a.nested_analysis,a.context_analysis,a.context_risk)
