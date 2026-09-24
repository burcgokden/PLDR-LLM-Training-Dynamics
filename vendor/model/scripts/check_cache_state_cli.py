#!/usr/bin/env python3
"""Run ordinary and optimized CLI contracts against complete native observations.

Fixtures hard-link immutable inputs. Every mutation replaces a directory entry
atomically, so neither scientific observations nor historical snapshots change.
All fixture destinations must lie in the authorized experiment subtree.
"""
from companion_paths import child_pythonpath
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from cache_state_contract import ROOT, REPO, read, SOURCES, CONTRACT
from context_reservations import reserve
from model_rg.provenance import sha256, write_json

ENTRY_PROBE = r"""
import runpy, sys
from unittest.mock import patch
import run_cache_state_transfer as module
def boundary(*args, **kwargs):
    print('ADMITTED_BOUNDARY', flush=True)
    raise SystemExit(77)
with patch('run_cache_state_transfer.TrainingModel', boundary), \
     patch('subprocess.run', boundary), \
     patch('torch.cuda.set_device'), patch('torch.cuda.reset_peak_memory_stats'), \
     patch('torch.cuda.mem_get_info', return_value=(24*1024**3,24*1024**3)), \
     patch('torch.cuda.max_memory_allocated', return_value=0):
    # Execute the actual CLI module; imported TrainingModel is patched at its
    # original module, since runpy creates a separate __main__ namespace.
    with patch('model_rg.training.TrainingModel', boundary):
        sys.argv=['run_cache_state_transfer.py']+sys.argv[1:]
        runpy.run_path(str(module.REPO/'scripts/run_cache_state_transfer.py'), run_name='__main__')
"""


def check(study, analysis, verification, initial, memory, output):
    study=study.resolve();output=output.resolve()
    if output.exists() or output==ROOT or not output.is_relative_to(ROOT):
        raise ValueError('Fresh authorized check directory required')
    output.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH=child_pythonpath("model")+os.pathsep+str(REPO/'scripts'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    results=[]
    def call(label, mode, command, expected, forbidden=()):
        log=output/(label+('-optimized' if mode else '-ordinary')+'.log')
        with log.open('x') as f:
            r=subprocess.run([sys.executable,*(['-O'] if mode else []),*command],
                             cwd=REPO,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
        valid = r.returncode==0 if expected=='pass' else r.returncode==77 if expected=='boundary' else r.returncode not in [0,77]
        if not valid or any(p.exists() for p in forbidden):
            raise RuntimeError('Unexpected CLI result '+label+'; see '+str(log))
        if expected=='boundary' and 'ADMITTED_BOUNDARY' not in log.read_text():
            raise ValueError('Missing intercepted boundary')
        results.append(dict(label=label,optimized=mode,expected=expected,returncode=r.returncode,
                            log_sha256=sha256(log),no_forbidden_output=not any(p.exists() for p in forbidden)))
    def replace_json(path, value):
        write_json(path.with_name(path.name+'.replacement'),value)
        path.with_name(path.name+'.replacement').replace(path)
    def clone(source,target):
        shutil.copytree(source,target,copy_function=os.link,ignore=shutil.ignore_patterns('__pycache__'))
    # Both CLI baselines execute all primary and independent numerical algorithms.
    for mode in [False,True]:
        stem='opt' if mode else 'ordinary'
        a=output/(stem+'-analysis.json');v=output/(stem+'-verification.json')
        call('baseline-analysis',mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(study),'--output',str(a)],'pass')
        call('baseline-verification',mode,[str(REPO/'scripts/verify_cache_state_transfer.py'),'--study',str(study),'--analysis',str(a),'--output',str(v)],'pass')
        rendered=output/(stem+'-rendered')
        render_args=[str(REPO/'scripts/render_cache_state_transfer.py'),'--study',str(study),
                     '--analysis',str(a),'--verification',str(v),'--initial',str(initial),
                     '--memory',str(memory),'--output',str(rendered)]
        call('baseline-rendering',mode,render_args,'pass')
        base=read(a);cert=read(v)
        for field in ['analysis_sources_sha256','checked_sha256']:
            for kind in ['empty','partial','unexpected']:
                bad=copy.deepcopy(base)
                if kind=='empty':bad[field]={}
                elif kind=='partial':bad[field].pop(next(iter(bad[field])))
                else:bad[field]['unexpected.py' if 'sources' in field else 'unexpected.npy']='0'*64
                name=stem+'-'+field+'-'+kind;ap=output/(name+'.json');vp=output/(name+'-certificate.json')
                write_json(ap,bad)
                call(name+'-verify',mode,[str(REPO/'scripts/verify_cache_state_transfer.py'),'--study',str(study),'--analysis',str(ap),'--output',str(vp)],'reject',[vp])
                vc=copy.deepcopy(cert);vc['analysis_sha256']=sha256(ap);cp=output/(name+'-claimed.json');write_json(cp,vc)
                rp=output/(name+'-rendered')
                args=render_args.copy();args[args.index('--analysis')+1]=str(ap);args[args.index('--verification')+1]=str(cp);args[-1]=str(rp)
                call(name+'-render',mode,args,'reject',[rp])
        for label, mutate in [
                ('stale-analysis',lambda v:v.update(analysis_sha256='0'*64)),
                ('empty-certificate-coverage',lambda v:v.update(checked_sha256={})),
                ('stale-coverage',lambda v:v.update(coverage_sha256='0'*64)),
                ('wrong-certificate-schema',lambda v:v.update(schema='foreign')),
                ('missing-verifier-role',lambda v:v['verification_sources_sha256'].pop(next(iter(v['verification_sources_sha256']))))]:
            vc=copy.deepcopy(cert);mutate(vc);cp=output/(stem+'-'+label+'.json');write_json(cp,vc)
            rp=output/(stem+'-'+label+'-rendered');args=render_args.copy();args[args.index('--verification')+1]=str(cp);args[-1]=str(rp)
            call(label+'-render',mode,args,'reject',[rp])
        # A common logit shift preserves the softmax but must fail acquisition
        # hashes. The local hard link is replaced before writing changed bytes.
        fixture=output/(stem+'-changed');clone(study,fixture)
        job=read(study/'protocol.json')['jobs'][0]['run_id']
        path=fixture/'runs'/job/'initial-L32-native.npy'
        import numpy as np
        shifted=np.load(path,allow_pickle=False).astype(np.float64)+8
        temp=path.with_name('replacement.npy');np.save(temp,shifted);temp.replace(path)
        for route in ['analyze','verify']:
            result=output/(stem+'-shift-'+route+'.json')
            args=[str(REPO/'scripts'/('analyze_cache_state_transfer.py' if route=='analyze' else 'verify_cache_state_transfer.py')),'--study',str(fixture),'--output',str(result)]
            if route=='verify':args += ['--analysis',str(a)]
            call('common-logit-shift-'+route,mode,args,'reject',[result])
        # Missing and malformed scientific members are independently exercised.
        path.unlink();os.link(study/'runs'/job/'initial-L32-native.npy',path)
        mpath=fixture/'runs'/job/'manifest.json';manifest=read(mpath)
        for label,name in [('missing-cache','caches.npz'),('missing-operator','initial-L32-operators.npz'),('missing-logit','initial-L32-native.npy')]:
            item=fixture/'runs'/job/name;item.unlink()
            result=output/(stem+'-'+label+'.json')
            call(label,mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(fixture),'--output',str(result)],'reject',[result])
            os.link(study/'runs'/job/name,item)
        # Duplicate JSON keys, foreign job identities and manifest inventories.
        raw=mpath.read_text()
        for label,mutate in [
                ('empty-artifact-map',lambda m:m.update(artifacts={})),
                ('partial-artifact-map',lambda m:m['artifacts'].pop('caches.npz')),
                ('unexpected-artifact',lambda m:m['artifacts'].update({'extra.npy':'0'*64})),
                ('wrong-manifest-job',lambda m:m['job'].update(seed=99)),
                ('wrong-manifest-protocol',lambda m:m.update(protocol_sha256='0'*64)),
                ('boolean-call-count',lambda m:m['calls'].update(calibration=True))]:
            m=copy.deepcopy(manifest);mutate(m);replace_json(mpath,m)
            result=output/(stem+'-'+label+'.json')
            call(label,mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(fixture),'--output',str(result)],'reject',[result])
        replacement=mpath.with_name('manifest-replacement.json')
        replacement.write_text(raw.replace('{','{"status":"complete",',1));replacement.replace(mpath)
        result=output/(stem+'-duplicate-keys.json')
        call('duplicate-manifest-key',mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(fixture),'--output',str(result)],'reject',[result])
        replace_json(mpath,manifest)
        # Corrupted arrays carry updated manifest hashes so finite geometry,
        # rather than a digest mismatch, is the rejecting boundary.
        target=fixture/'runs'/job/'initial-L32-native.npy'
        original=np.load(study/'runs'/job/'initial-L32-native.npy',allow_pickle=False)
        for label,value in [('wrong-logit-shape',original[:,:-1]),
                            ('complex-logits',original.astype(np.complex64)),
                            ('boolean-logits',np.zeros(original.shape,dtype=bool)),
                            ('nonfinite-logits',np.full(original.shape,np.nan,dtype=np.float32))]:
            temp=target.with_name('replacement.npy');np.save(temp,value);temp.replace(target)
            altered=copy.deepcopy(manifest);altered['artifacts'][target.name]=sha256(target)
            replace_json(mpath,altered)
            result=output/(stem+'-'+label+'.json')
            call(label,mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(fixture),'--output',str(result)],'reject',[result])
        target.unlink();os.link(study/'runs'/job/target.name,target);replace_json(mpath,manifest)
        for label,name in [('missing-protocol','protocol.json'),('missing-input','inputs.npz'),
                           ('missing-manifest','runs/'+job+'/manifest.json')]:
            target=fixture/name;target.unlink();result=output/(stem+'-'+label+'.json')
            call(label,mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(fixture),'--output',str(result)],'reject',[result])
            os.link(study/name,target)
        target=fixture/'runs'/job/'initial-L32-native.npy';target.unlink()
        target.symlink_to(study/'runs'/job/target.name)
        result=output/(stem+'-symlink-escape.json')
        call('symlink-escape',mode,[str(REPO/'scripts/analyze_cache_state_transfer.py'),'--study',str(fixture),'--output',str(result)],'reject',[result])
        target.unlink();os.link(study/'runs'/job/target.name,target)
        # Admission: a genuine current protocol reaches the intercepted native
        # construction/launch boundary; malformed variants create neither runs
        # nor logs. None of these probes is a scientific acquisition.
        incoming=output/(stem+'-entry')
        receipt=reserve(incoming,80,reproduce=study)
        incoming.mkdir()
        os.link(study/'inputs.npz',incoming/'inputs.npz')
        protocol=read(study/'protocol.json')
        protocol.update(schema='cache-state-transfer-v3',implementation=CONTRACT,
            excluded_context_protocols=sorted(receipt['imported_protocols']))
        protocol['input_sha256'].update(receipt['imported_protocols'])
        write_json(incoming/'reservation.json',receipt)
        protocol['reservation_sha256']=sha256(incoming/'reservation.json')
        protocol['source_sha256']={name:sha256(REPO/name) for name in SOURCES}
        for name in SOURCES:
            dest=incoming/'executed-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
        write_json(incoming/'protocol.json',protocol)
        for route in ['worker','run']:
            args=['-c',ENTRY_PROBE,route,'--study',str(incoming)]
            if route=='worker':args += ['--index','0','--device','cuda:0']
            call('valid-'+route,mode,args,'boundary')
            for name in ['runs','logs']:
                if (incoming/name).exists():shutil.rmtree(incoming/name)
            variants=[
                ('wrong-native-code',lambda p:p['input_sha256'].update({str(ROOT/'assets/PLDR-LLM-v51-SOC-110M-1/modeling_pldrllm.py'):'0'*64})),
                ('empty-sources',lambda p:p.update(source_sha256={})),
                ('partial-sources',lambda p:p['source_sha256'].pop(next(iter(p['source_sha256'])))),
                ('empty-external',lambda p:p.update(input_sha256={})),
                ('partial-external',lambda p:p['input_sha256'].pop(next(iter(p['input_sha256'])))),
                ('altered-prefix',lambda p:p.update(prefix_lengths=[32,64])),
                ('duplicate-job',lambda p:p['jobs'].__setitem__(1,copy.deepcopy(p['jobs'][0]))),
                ('foreign-parent',lambda p:p['jobs'][0]['endpoints']['g0'].update(run_id='foreign')),
                ('wrong-token-panel',lambda p:p.update(selection_sha256='0'*64)),
                ('changed-source',lambda p:p['source_sha256'].update({'scripts/run_cache_state_transfer.py':'0'*64}))]
            for label,mutate in variants:
                p=copy.deepcopy(protocol);mutate(p);replace_json(incoming/'protocol.json',p)
                call(route+'-'+label,mode,args,'reject',[incoming/'runs',incoming/'logs'])
            replace_json(incoming/'protocol.json',protocol)
            if route=='worker':
                for index in ['-1','12']:
                    bad=args.copy();bad[bad.index('--index')+1]=index
                    call('worker-index-'+index,mode,bad,'reject',[incoming/'runs',incoming/'logs'])
    write_json(output/'verification.json',dict(status='passed',schema='cache-state-cli-v2',
        checker_sha256=sha256(__file__),cases=results,case_count=len(results),
        scientific_training_updates=0,scientific_forward_calls=0,
        input_sha256={str(p):sha256(p) for p in [study/'protocol.json',analysis,verification,initial,memory]},
        tested_sources={str(p.relative_to(REPO)):sha256(p) for p in (REPO/'scripts').glob('*cache_state*.py')}))
    print('Passed',len(results),'ordinary/optimized CLI cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['study','analysis','verification','initial','memory','output']:p.add_argument('--'+key,type=Path,required=True)
    check(**vars(p.parse_args()))
