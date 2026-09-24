#!/usr/bin/env python
"""Check real fine-tuning admission entrypoints without native updates."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();study=Path(a.study).resolve();out=Path(a.output).resolve()
    out.mkdir(parents=True,exist_ok=False);spec=json.loads((study/'protocol.json').read_text())
    if spec['stage']!='assessment':raise ValueError('Assessment required')
    script=REPO/'scripts/finetuning_reference.py';checks=[]
    def run(name,command,success):
        p=subprocess.run(command,cwd=REPO,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=out/(name+'.log');log.write_text(p.stdout)
        if (p.returncode==0)!=success:raise ValueError('Unexpected admission outcome: '+name)
        checks.append(dict(name=name,returncode=p.returncode,expected_pass=success,log_sha256=sha(log)))
    for optimize in [False,True]:
        py=[sys.executable]+(['-O'] if optimize else []);tag='optimized' if optimize else 'ordinary'
        run(tag+'-valid',py+[str(script),'validate','--study',str(study)],True)
        with tempfile.TemporaryDirectory(prefix='finetuning-admission-',dir='/tmp') as tmp:
            for mutation in ['source','input','qualification','runtime','horizon','arms','optimizer_origin','rate']:
                s=json.loads(json.dumps(spec));folder=Path(tmp)/mutation;folder.mkdir()
                if mutation=='source':s['sources'][next(iter(s['sources']))]='0'*64
                if mutation=='input':s['inputs'][next(iter(s['inputs']))]='0'*64
                if mutation=='qualification':s['qualification']['sha256']='0'*64
                if mutation=='runtime':s['runtime']['dtype']='float16'
                if mutation=='horizon':s['horizon']+=1
                if mutation=='arms':s['arms']=s['arms'][:-1]
                if mutation=='optimizer_origin':s['optimizer_origin']='retained'
                if mutation=='rate':s['rate']*=2
                (folder/'protocol.json').write_text(json.dumps(s))
                for entry in ['validate','run','worker'] if mutation=='source' else ['validate']:
                    command=py+[str(script),entry,'--study',str(folder)]
                    if entry=='worker':command+=['--arm','general','--device','cuda:0']
                    run(tag+'-'+mutation+'-'+entry,command,False)
                if any((folder/c['name']).exists() for c in spec['arms']):raise ValueError('Rejected entry created a native worker')
    report=dict(status='passed',native_updates=0,producer_sha256=sha(script),checker_sha256=sha(__file__),protocol_sha256=sha(study/'protocol.json'),checks=checks,
                scope='Real validate, launch and direct-worker rejection of changed source; input, qualification and platform bindings checked in ordinary and optimized Python. Reference producer also rejects changed horizon, source arms, optimizer origin and rate.')
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print('Passed',len(checks),'admission checks; zero native updates')


if __name__=='__main__':main()
