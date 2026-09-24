#!/usr/bin/env python3
"""Discover shipped scientific tests and report exclusions separately."""
import argparse,json,os,shutil,subprocess,sys,tempfile,time,xml.etree.ElementTree as ET
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from companion_paths import child_environment
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--family',choices=['root','row','rg','model']);a=p.parse_args();out=ROOT/'validation';out.mkdir(exist_ok=True);records=[]
    for family,suite in [('root','tests'),('model','tests'),('row','experiments/tests'),('rg','tests')]:
        if a.family and a.family!=family:continue
        cwd=ROOT if family=='root' else ROOT/'vendor'/family
        fixture=None
        if family=='rg':
            # Provenance tests need committed source bytes, not the clone's base HEAD.
            # Create a disposable exact source fixture; never stage or commit ROOT.
            fixture=tempfile.TemporaryDirectory(prefix='pldr-scientific-fixture-')
            fixture_root=Path(fixture.name)/'code'
            shutil.copytree(ROOT,fixture_root,ignore=shutil.ignore_patterns('.git','.lake','build','validation','__pycache__','.pytest_cache'))
            subprocess.run(['git','init','-q'],cwd=fixture_root,check=True)
            subprocess.run(['git','add','.'],cwd=fixture_root,check=True)
            subprocess.run(['git','-c','user.name=Scientific fixture','-c','user.email=fixture@example.invalid','commit','-qm','Disposable scientific provenance fixture'],cwd=fixture_root,check=True)
            cwd=fixture_root/'vendor/rg'
        xml=out/('scientific-'+family+'.xml');start=time.monotonic()
        env=child_environment('model' if family=='root' else family,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        if fixture is not None:env['PYTHONPATH']=os.pathsep.join([str(fixture_root),str(cwd/'src'),str(cwd/'scripts'),str(cwd)])
        r=subprocess.run([sys.executable,'-B','-m','pytest','-q','-p','no:cacheprovider','-p','public_pytest',suite,'--tb=short','-rs','--junitxml='+str(xml)],cwd=cwd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (out/('scientific-'+family+'.log')).write_text(r.stdout)
        cases=ET.parse(xml).getroot().findall('.//testcase') if xml.exists() else []
        records.append({'family':family,'returncode':r.returncode,'seconds':time.monotonic()-start,'cases':len(cases),'skips':[dict(test=c.attrib,reason=c.find('skipped').get('message')) for c in cases if c.find('skipped') is not None],'failures':[dict(test=c.attrib,detail=ET.tostring(c,encoding='unicode')) for c in cases if c.find('error') is not None or c.find('failure') is not None]})
        print(family,r.returncode,r.stdout[-1500:],flush=True)
        if fixture is not None:fixture.cleanup()
    result={'status':'passed' if all(r['returncode']==0 for r in records) else 'failed','suites':records,'exclusion_inventory':'provenance/export-disposition.json','scope':'Discovery of shipped scientific tests; excluded publication checks are not passes. Resource and campaign checks overlap these suites.'}
    (out/('scientific-'+(a.family or 'all')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    raise SystemExit(0 if result['status']=='passed' else 1)
