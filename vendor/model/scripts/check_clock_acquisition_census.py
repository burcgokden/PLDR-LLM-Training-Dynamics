#!/usr/bin/env python3
"""Reject incomplete raw observation inventories before analysis publication."""
from companion_paths import child_pythonpath
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from model_rg.provenance import sha256, write_json
from matched_clock_coverage import admitted_protocol, read

REPO = Path(__file__).resolve().parents[1]


def check(study, output):
    p = admitted_protocol(study)
    output.mkdir(parents=True, exist_ok=False)
    cases = []
    names = ['missing_run','extra_run','missing_observation','extra_observation',
             'missing_manifest_role','boolean_manifest_identity','missing_qualification','changed_protocol']
    env = dict(os.environ, PYTHONPATH=child_pythonpath("model"), PYTHONDONTWRITEBYTECODE='1',
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    for optimized in [False, True]:
        for name in names:
            with tempfile.TemporaryDirectory(prefix='clock-census-',dir='/tmp') as temp:
                root=Path(temp)
                for item in ['protocol.json','selection.npz','reservation.json','qualification','executed-source']:
                    (root/item).symlink_to(study/item, target_is_directory=(study/item).is_dir())
                (root/'runs').mkdir()
                first=p['jobs'][0]['run_id']
                for j in p['jobs']:
                    dest=root/'runs'/j['run_id']
                    if j['run_id'] == first:
                        dest.mkdir()
                        for f in (study/'runs'/first).iterdir():
                            (dest/f.name).symlink_to(f)
                    else:dest.symlink_to(study/'runs'/j['run_id'],target_is_directory=True)
                folder=root/'runs'/first
                if name=='missing_run':
                    (root/'runs'/p['jobs'][-1]['run_id']).unlink()
                elif name=='extra_run':
                    (root/'runs'/'foreign').mkdir()
                elif name=='missing_observation':
                    (folder/'age-0.npz').unlink()
                elif name=='extra_observation':
                    (folder/'unplanned.npz').write_bytes(b'invalid fixture')
                elif name=='missing_manifest_role':
                    m=read(folder/'manifest.json');m['artifacts'].pop('age-0.npz')
                    (folder/'manifest.json').unlink();write_json(folder/'manifest.json',m)
                elif name=='boolean_manifest_identity':
                    m=read(folder/'manifest.json');m['job']['control']=False
                    (folder/'manifest.json').unlink();write_json(folder/'manifest.json',m)
                elif name=='missing_qualification':
                    (root/'qualification').unlink();(root/'qualification').mkdir()
                else:
                    q=dict(p);q['contexts']=63
                    (root/'protocol.json').unlink();write_json(root/'protocol.json',q)
                result_path=root/'analysis.json'
                command=[sys.executable,*(['-O'] if optimized else []),'-B',str(REPO/'scripts/analyze_matched_clock.py'),
                         '--study',str(root),'--output',str(result_path)]
                result=subprocess.run(command,cwd=REPO,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                log=output/(name+('-O' if optimized else '-N')+'.log');log.write_text(result.stdout)
                passed=result.returncode!=0 and not result_path.exists() and ('Matched-clock coverage:' in result.stdout or 'FileNotFoundError:' in result.stdout)
                cases.append(dict(name=name,optimized=optimized,passed=passed,no_output=not result_path.exists(),returncode=result.returncode,command=command,log_sha256=sha256(log)))
                if not passed:raise RuntimeError('Raw census accepted '+name)
    write_json(output/'verification.json',dict(status='passed',cases=cases,case_count=len(cases),
        checker_sha256=sha256(__file__),protocol_sha256=sha256(study/'protocol.json'),
        tested_sources={n:sha256(REPO/n) for n in ['scripts/analyze_matched_clock.py','scripts/matched_clock_coverage.py']},
        scope='Actual analysis CLI with symlinked read-only observations and isolated invalid fixtures; no native calls.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    check(**{k:v.resolve() for k,v in vars(p.parse_args()).items()})
