"""End-to-end finite admission, including adversarial CLI execution under -O."""
from companion_paths import child_pythonpath
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'scripts'))
sys.path.insert(0,str(REPO/'tests'))
from cache_risk_fixture import make_fixture
from analyze_cache_risk_study import analyze
from verify_cache_risk_study import verify
from model_rg.provenance import sha256, write_json


@pytest.fixture(scope='module')
def fixture(tmp_path_factory):
    root=tmp_path_factory.mktemp('cache-risk'); study=root/'study'; p=make_fixture(study)
    analysis=root/'analysis.json'; analyze(study,analysis)
    verification=root/'verification.json';verify(study,analysis,verification)
    baseline=root/'baseline';baseline.mkdir()
    write_json(baseline/'execution-ledger.json',dict(rows=[],input_sha256={}))
    write_json(baseline/'study-lineage.json',dict(records=[],checked_sha256={}))
    for name in ['execution-ledger.tex','study-lineage.tex']:(baseline/name).write_text('\\bottomrule\n')
    return root,study,p,analysis,verification,baseline




CASES=[(key,tag) for key in ['mean','powers','scatter','risk','bias','cache0','cache1','cache2'] for tag in ['nan','inf','-inf']]
CASES += [(key,'shape') for key in ['mean','powers','scatter','risk','bias','cache0']]
CASES += [('powers',kind) for kind in ['bool','complex','string','object','missing','negative','disagree']]
CASES += [('mean','overflow'),('cache2','overflow')]


@pytest.mark.parametrize('key,kind',CASES)
def test_raw_corruption_rejects_primary_and_verifier(fixture,tmp_path,key,kind):
    root,study,p,analysis,verification,baseline=fixture
    folder=study/'runs'/p['jobs'][0]['run_id']; cache=key.startswith('cache')
    path=folder/('caches.npz' if cache else 'operators-L32.npz');manifest=folder/'manifest.json'
    original=path.read_bytes();original_manifest=manifest.read_bytes()
    with np.load(path) as a: values={k:a[k].copy() for k in a.files}
    member='C32-P0-M16' if cache else 'L32-C32-P0-M16-'+key if key in ['risk','bias'] else key
    try:
        if kind=='shape':values[member]=values[member][...,None]
        elif kind in ['bool','complex','string','object']:values[member]=values[member].astype({'bool':bool,'complex':complex,'string':'U24','object':object}[kind])
        elif kind=='missing':del values[member]
        else:
            value={'nan':np.nan,'inf':np.inf,'-inf':-np.inf,'negative':-1.,'disagree':2.,'overflow':1e308}[kind]
            values[member]=values[member].astype(float)
            if cache:values[member][0,int(key[-1]),0,0,0,0]=value
            else:values[member].flat[0]=value
        np.savez_compressed(path,**values)
        m=json.loads(original_manifest);m['artifacts'][path.name]=sha256(path);write_json(manifest,m)
        a=json.loads(analysis.read_text());a['checked_sha256'][str(path.relative_to(study))]=sha256(path);a['checked_sha256'][str(manifest.relative_to(study))]=sha256(manifest)
        modified=tmp_path/'analysis.json';write_json(modified,a)
        with pytest.raises((ValueError,TypeError,OverflowError)):analyze(study,tmp_path/'primary.json')
        with pytest.raises((ValueError,TypeError,OverflowError)):verify(study,modified,tmp_path/'certificate.json')
        assert not (tmp_path/'primary.json').exists() and not (tmp_path/'certificate.json').exists()
    finally:path.write_bytes(original);manifest.write_bytes(original_manifest)


@pytest.mark.parametrize('optimized',[False,True])
@pytest.mark.parametrize('case',['positive','nan','overflow','summary','stale-source'])
def test_cli_boundaries(fixture,tmp_path,optimized,case):
    root,study,p,analysis,verification,baseline=fixture
    a=json.loads(analysis.read_text())
    if case=='nan':a['cells'][0]['mean_operator_risk']=float('nan')
    elif case=='overflow':a['cells'][0]['mean_operator_risk']=float('inf')
    elif case=='summary':a['cells'][0]['operator_risk_by_seed_layer']=[[1.]]
    elif case=='stale-source':a['analysis_sources_sha256']['scripts/cache_risk_validation.py']='0'*64
    changed=tmp_path/'analysis.json';changed.write_text(json.dumps(a))
    output=tmp_path/'certificate.json'
    command=[sys.executable]+(['-O'] if optimized else [])+[str(REPO/'scripts/verify_cache_risk_study.py'),'--study',str(study),'--analysis',str(changed),'--output',str(output)]
    result=subprocess.run(command,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
    assert (result.returncode==0)==(case=='positive'),result.stderr
    assert output.exists()==(case=='positive')




@pytest.mark.parametrize('optimized',[False,True])
@pytest.mark.parametrize('corruption',['nan-power','cache-overflow'])
def test_raw_cli_rejection_in_both_modes(fixture,tmp_path,optimized,corruption):
    root,study,p,analysis,verification,baseline=fixture
    folder=study/'runs'/p['jobs'][0]['run_id'];path=folder/('operators-L32.npz' if corruption=='nan-power' else 'caches.npz')
    manifest=folder/'manifest.json';original=path.read_bytes();saved_manifest=manifest.read_bytes()
    try:
        with np.load(path) as z:values={k:z[k].astype(float) for k in z.files}
        if corruption=='nan-power':values['powers'][0,0]=np.nan
        else:values['C32-P0-M16'][0,2,0,0,0,0]=1e308
        np.savez_compressed(path,**values)
        m=json.loads(saved_manifest);m['artifacts'][path.name]=sha256(path);write_json(manifest,m)
        a=json.loads(analysis.read_text());a['checked_sha256'][str(path.relative_to(study))]=sha256(path);a['checked_sha256'][str(manifest.relative_to(study))]=sha256(manifest)
        changed=tmp_path/'analysis.json';write_json(changed,a)
        prefix=[sys.executable]+(['-O'] if optimized else [])
        for script,args in [('analyze_cache_risk_study.py',['--study',str(study)]),('verify_cache_risk_study.py',['--study',str(study),'--analysis',str(changed)])]:
            output=tmp_path/(script+'.json')
            run=subprocess.run(prefix+[str(REPO/'scripts'/script),*args,'--output',str(output)],env=dict(os.environ,PYTHONPATH=child_pythonpath("model"),OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True)
            assert run.returncode!=0 and not output.exists(),run.stdout+run.stderr
    finally:path.write_bytes(original);manifest.write_bytes(saved_manifest)
