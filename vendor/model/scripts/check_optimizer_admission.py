#!/usr/bin/env python3
"""Exercise the real admission guard without executing scientific updates."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[1]


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(study, output):
    study=Path(study).resolve(); output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    spec=json.loads((study/'protocol.json').read_text()); rows=[]
    if spec['stage']!='assessment': raise ValueError('Use the prepared admitted assessment')
    def call(name,command,expected):
        result=subprocess.run(command,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=output/(name+'.log');log.write_text(result.stdout)
        if (result.returncode==0)!=expected: raise ValueError('Unexpected admission result: '+name)
        rows.append(dict(name=name,command=command,returncode=result.returncode,expected_pass=expected,log_sha256=sha(log)))
    script=str(REPO/'scripts/optimizer_transport_study.py')
    for optimized in [False,True]:
        py=[sys.executable]+(['-O'] if optimized else [])
        tag='optimized' if optimized else 'ordinary'
        call(tag+'-valid',py+[script,'validate','--study',str(study)],True)
        with tempfile.TemporaryDirectory(prefix='optimizer-admission-',dir='/tmp') as tmp:
            temp=Path(tmp)
            for mutation in ['source','input','sample','qualification']:
                changed=json.loads(json.dumps(spec)); bad=temp/mutation;bad.mkdir()
                for case in spec['cases']:shutil.copy2(study/(case['name']+'-blocks.npy'),bad/(case['name']+'-blocks.npy'))
                if mutation=='qualification':changed['qualification']['sha256']='0'*64
                else:
                    group={'source':'sources','input':'inputs','sample':'sampling'}[mutation]
                    changed[group][next(iter(changed[group]))]='0'*64
                (bad/'protocol.json').write_text(json.dumps(changed))
                entries=['validate','run','worker'] if mutation=='source' else ['validate']
                for entry in entries:
                    command=py+[script,entry,'--study',str(bad)]
                    if entry=='worker':command+=['--heads','4','--device','cuda:0']
                    call(tag+'-'+mutation+'-'+entry,command,False)
                if any((bad/c['name']).exists() for c in spec['cases']):raise ValueError('Rejected entry created a native worker destination')
            # Preparation must refuse a stale qualification before creating its destination.
            q=Path(spec['qualification']['path']).parent; badq=temp/'bad-qualification';badq.mkdir()
            qs=json.loads((q/'protocol.json').read_text());qs['sources'][next(iter(qs['sources']))]='0'*64
            (badq/'protocol.json').write_text(json.dumps(qs));shutil.copy2(q/'verification.json',badq/'verification.json')
            destination=temp/'never-created'
            call(tag+'-prepare-stale',py+[script,'prepare','--stage','assessment','--study',str(destination),
                '--data-root',str(temp),'--qualification',str(badq/'verification.json')],False)
            if destination.exists():raise ValueError('Rejected preparation created its destination')
    report=dict(status='passed',native_updates=0,checker_sha256=sha(__file__),producer_sha256=sha(script),
        assessment_protocol_sha256=sha(study/'protocol.json'),checks=rows)
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',len(rows),'admission checks; zero native updates')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();check(a.study,a.output)
