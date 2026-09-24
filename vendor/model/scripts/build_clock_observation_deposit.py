#!/usr/bin/env python3
"""Versioned, CPU-reconstructible observation deposit without copying large payloads."""
from companion_paths import legacy_path
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from model_rg.provenance import sha256,write_json
from matched_clock_coverage import read,validate_certificate

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))


def build(clock,panel,records,output):
    if output.exists() or not output.resolve().is_relative_to(ROOT):raise ValueError('Fresh authorized deposit required')
    validate_certificate(read(records/'matched-clock-verification.json'),read(records/'matched-clock-analysis.json'),records/'matched-clock-analysis.json')
    analyses={'matched-clock':json.loads((records/'matched-clock-analysis.json').read_text()),'fixed-cache':json.loads((records/'fixed-cache-panel.json').read_text())}
    if any(a['status']!='passed' for a in analyses.values()):raise ValueError('Completed observations required')
    output.mkdir(parents=True);inventory={}
    for key,study in [('matched-clock',clock),('fixed-cache',panel)]:
        for name,digest in analyses[key]['checked_sha256'].items():
            src=study/name
            if sha256(src)!=digest:raise ValueError('Changed raw observation '+name)
            dst=output/'observations'/key/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
            inventory[str(dst.relative_to(output))]=dict(sha256=digest,bytes=dst.stat().st_size)
        protocol=json.loads((study/'protocol.json').read_text())
        for name,digest in protocol.get('source_sha256',{}).items():
            src=study/'executed-source'/name
            if sha256(src)!=digest:raise ValueError('Changed acquisition source '+name)
            dst=output/'observations'/key/'executed-source'/name;dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
            inventory[str(dst.relative_to(output))]=dict(sha256=digest,bytes=dst.stat().st_size)
        expected_path=output/(key+'-expected.json');write_json(expected_path,analyses[key])
        inventory[expected_path.name]=dict(sha256=sha256(expected_path),bytes=expected_path.stat().st_size)
    code=['scripts/analyze_matched_clock.py','scripts/verify_matched_clock.py','scripts/analyze_fixed_cache_panel.py','scripts/verify_fixed_cache_panel.py','scripts/numerical_validation.py','scripts/matched_clock_coverage.py','scripts/contracts/matched-clock-observation-v1.json','src/model_rg/provenance.py','src/model_rg/__init__.py']
    for name in code:
        dst=output/'code'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dst)
        inventory[str(dst.relative_to(output))]=dict(sha256=sha256(dst),bytes=dst.stat().st_size)
    version=sha256(clock/'protocol.json')[:16]+'-'+sha256(panel/'protocol.json')[:16]+'-coverage-'+sha256(REPO/'scripts/matched_clock_coverage.py')[:12]
    write_json(output/'manifest.json',dict(schema='clock-observation-deposit-v1',version=version,members=inventory,
        payload_scope='All raw observations used by the 64 matched-clock field cells, 26 age/cache cells and ten fixed-cache transfer cells. Native replay checkpoints, training corpora and unrelated manuscript tables remain separate.',
        transfer='Directory with hard-linked local observation files; no duplicate large archive and no public retrieval claim.',public_retrieval=None))
    (output/'README.md').write_text('''# Conditioned-clock observations

The manifest identifies every required observation and code member. These data reconstruct the complete matched-clock and fixed-cache tables on CPU, without model weights. Other manuscript tables have separate evidence coverage. Large replay states and corpora are not included. The versioned admitted protocol and full condition, seed/run and observation census accompany the reductions; older count-only certificates are not accepted.

From any working directory, with NumPy and SciPy installed:

```sh
PYTHONPATH=/path/to/deposit/code/src OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python3 /path/to/deposit/code/scripts/analyze_matched_clock.py --study /path/to/deposit/observations/matched-clock --output /tmp/clock-new.json
PYTHONPATH=/path/to/deposit/code/src OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= python3 /path/to/deposit/code/scripts/analyze_fixed_cache_panel.py --study /path/to/deposit/observations/fixed-cache --output /tmp/cache-new.json
```

The included `verify_matched_clock.py` and `verify_fixed_cache_panel.py` independently reconstruct these same results; each accepts `--study`, `--analysis` and `--output`. `verification.json` records the clean local executions of both reducers and both independent verifiers. The native acquisition snapshot preserves provenance; it is separate from the portable CPU code and requires additional native-replay assets to execute.

Use fresh output filenames. This is a locally prepared versioned deposit. No external download, persistent identifier or public retrieval test is asserted.
''')
    with tempfile.TemporaryDirectory(prefix='clock-cpu-',dir='/tmp') as name:
        tmp=Path(name);env=dict(os.environ,PYTHONPATH=str(output/'code/src'),OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
        comparisons={}
        for key,script in [('matched-clock','analyze_matched_clock.py'),('fixed-cache','analyze_fixed_cache_panel.py')]:
            dest=tmp/(key+'.json')
            with (output/(key+'-reconstruction.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',str(output/'code/scripts'/script),'--study',str(output/'observations'/key),'--output',str(dest)],cwd=tmp,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            actual=json.loads(dest.read_text())
            if actual!=analyses[key]:raise ValueError('Clean CPU reconstruction differs '+key)
            verifier='verify_matched_clock.py' if key=='matched-clock' else 'verify_fixed_cache_panel.py'
            verified=tmp/(key+'-verification.json')
            with (output/(key+'-independent.log')).open('x') as log:
                subprocess.run([sys.executable,'-B',str(output/'code/scripts'/verifier),'--study',str(output/'observations'/key),'--analysis',str(dest),'--output',str(verified)],cwd=tmp,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            checked=json.loads(verified.read_text())
            expected_verification=json.loads((records/('matched-clock-verification.json' if key=='matched-clock' else 'fixed-cache-verification.json')).read_text())
            if checked!=expected_verification:raise ValueError('Clean independent reconstruction differs '+key)
            comparisons[key]=dict(exact_result_match=True,independent_result_match=True,result_sha256=sha256(dest),verification_sha256=sha256(verified))
    receipt=dict(status='passed',schema='clock-deposit-verification-v1',version=version,manifest_sha256=sha256(output/'manifest.json'),
        comparisons=comparisons,external_retrieval_tested=False,local_clean_cpu_reconstruction=True,
        member_bytes=sum(x['bytes'] for x in inventory.values()),large_payload_storage_duplicated=False,builder_sha256=sha256(__file__))
    write_json(output/'verification.json',receipt);write_json(records/'observation-deposit.json',dict(**receipt,local_root=str(output)))
    print(receipt,flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser()
    for name in ['clock','panel','records','output']:a.add_argument('--'+name,type=Path,required=True)
    build(**vars(a.parse_args()))
