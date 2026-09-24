#!/usr/bin/env python
"""Exercise current primary entrypoints in normal and optimized Python.

No worker is allowed to start. Temporary negative verification records are
created next to the new qualification and removed before this checker exits.
Original scientific protocols and evidence are never modified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPO=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def run_checks(study,short,output):
    sys.path.insert(0,str(REPO/'scripts'))
    import finetuning_study as producer
    from finetuning_design import check_correspondence
    spec=json.loads((study/'protocol.json').read_text())
    qs=json.loads((short/'protocol.json').read_text())
    verification=json.loads((short/'verification.json').read_text())
    checks=[]
    def record(name,operation,admit=False):
        error=None
        try:operation()
        except (ValueError,KeyError,TypeError,FileNotFoundError,StopIteration) as exc:error=str(exc)
        if (error is None)!=admit:raise RuntimeError((name,'unexpected admission/refusal',error))
        checks.append(dict(name=name,admitted=error is None,expected_admit=admit,reason=error))
    record('valid-qualification',lambda:producer.validate(short),True)
    record('valid-assessment',lambda:producer.validate(study),True)
    record('valid-correspondence',lambda:check_correspondence(spec,qs,verification),True)
    mutations={
        'horizon_times':lambda s:s.update(horizon=1,times=[0,1]),
        'missing_case':lambda s:s.update(cases=s['cases'][:-1]),
        'batch64':lambda s:s.update(batch_size=64),
        'unknown_stage':lambda s:(s.update(stage='unknown'),s.pop('qualification')),
        'duplicated_case':lambda s:s['cases'].append(copy.deepcopy(s['cases'][0])),
        'substituted_state':lambda s:s['cases'][0].update(state=s['cases'][1]['state']),
        'profile':lambda s:s['cases'][0]['profile'].update(epsilon=1e-8),
        'arm_role':lambda s:s['arms'][0].update(role='replay'),
        'arm_reset':lambda s:s['arms'][0].update(reset=True),
        'arm_mixture':lambda s:s['arms'][0].update(rho=.5),
        'omitted_qualification':lambda s:s.pop('qualification'),
        'source':lambda s:s['sources'].update({'scripts/finetuning_design.py':'0'*64}),
        'runtime':lambda s:s['runtime'].update(dtype='float16'),
        'input_inventory':lambda s:s['inputs'].pop(next(iter(s['inputs']))),
    }
    with tempfile.TemporaryDirectory(prefix='primary-design-',dir='/tmp') as temp:
        for label,edit in mutations.items():
            folder=Path(temp)/label;folder.mkdir();changed=copy.deepcopy(spec);edit(changed)
            (folder/'protocol.json').write_text(json.dumps(changed))
            for entry,operation in [('validate',lambda:producer.validate(folder)),
                                    ('run',lambda:producer.run(folder)),
                                    ('worker',lambda:producer.worker(folder,spec['cases'][0]['name'],'cuda:0'))]:
                record(label+'/'+entry,operation)
            if list(folder.iterdir())!=[folder/'protocol.json']:
                raise RuntimeError('Refusal created worker output: '+label)
        for label,edit in {
            'short_stage':lambda q:q.update(stage='assessment'),
            'short_horizon':lambda q:q.update(horizon=3),
            'short_grid':lambda q:q.update(times=[0,2]),
            'short_cases':lambda q:q.update(cases=q['cases'][:-1]),
            'short_runtime':lambda q:q['runtime'].update(dtype='float16'),
            'short_sources':lambda q:q['sources'].update(extra='0'*64),
            'short_law':lambda q:q.update(law='different source law'),
        }.items():
            altered=copy.deepcopy(qs);edit(altered)
            record(label,lambda:check_correspondence(spec,altered,verification))
        for label,edit in {
            'verification_stage':lambda v:v.update(stage='assessment'),
            'verification_schema':lambda v:v.update(schema='detached-status'),
            'verification_status':lambda v:v.update(status='failed'),
            'verification_verifier':lambda v:v.update(verifier_sha256='0'*64),
            'verification_design_verifier':lambda v:v.update(design_verifier_sha256='0'*64),
            'terminal_inventory_missing':lambda v:v['verified_files'].pop(str(short/qs['cases'][0]['name']/'results.json')),
            'terminal_record_hash':lambda v:v['verified_files'].update({str(short/qs['cases'][0]['name']/'results.json'):'0'*64}),
            'detached_pass':lambda v:v.update(verified_files={}),
        }.items():
            altered=copy.deepcopy(verification);edit(altered)
            with tempfile.NamedTemporaryFile(mode='w',prefix='admission-fixture-',suffix='.json',dir=short) as bound:
                json.dump(altered,bound);bound.flush()
                changed=copy.deepcopy(spec);changed['qualification']=dict(path=bound.name,sha256=sha(bound.name))
                record(label,lambda:producer.validate_spec(changed))
        # Mutate the actual assembled preparation through its arm factory.
        invalid=study.parent/('invalid-assembled-'+str(not __debug__))
        original=producer.arms
        producer.arms=lambda:[]
        try:record('prepare-invalid-assembled',lambda:producer.prepare(invalid,spec['root'],'qualification',None))
        finally:producer.arms=original
        if invalid.exists():raise RuntimeError('Invalid preparation created a study')
        record('prepare-unknown-stage',lambda:producer.prepare(invalid,spec['root'],'unknown',None))
    output.write_text(json.dumps(dict(status='passed',optimized=not __debug__,checks=checks,
        native_updates=0,producer_sha256=sha(REPO/'scripts/finetuning_study.py'),checker_sha256=sha(__file__),
        protocol_sha256=sha(study/'protocol.json'),qualification_sha256=sha(short/'verification.json')),indent=2)+'\n')
    print('Passed',len(checks),'entrypoint/design checks; native updates 0',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--study',required=True);parser.add_argument('--qualification',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--child',action='store_true');a=parser.parse_args()
    study,short,out=Path(a.study).resolve(),Path(a.qualification).resolve(),Path(a.output).resolve()
    if a.child:run_checks(study,short,out);return
    out.mkdir(parents=True,exist_ok=False);checks=[]
    for mode,flags in [('ordinary',[]),('optimized',['-O'])]:
        path=out/(mode+'.json');log=out/(mode+'.log')
        command=[sys.executable,*flags,str(Path(__file__).resolve()),'--child','--study',str(study),
                 '--qualification',str(short),'--output',str(path)]
        with log.open('w') as stream:subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        checks.extend(json.loads(path.read_text())['checks'])
    report=dict(status='passed',native_updates=0,checks=checks,producer_sha256=sha(REPO/'scripts/finetuning_study.py'),
        checker_sha256=sha(__file__),protocol_sha256=sha(study/'protocol.json'),
        qualification_sha256=sha(short/'verification.json'),files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Passed',len(checks),'ordinary/optimized checks; zero native updates')


if __name__=='__main__':main()
