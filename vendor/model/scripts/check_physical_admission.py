"""Exercise real physical entry points under ordinary and optimized Python."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


class Constructed(Exception):pass


def child(study, output):
    import physical_study as worker
    import physical_design as design
    import run_physical_queue as queue
    from physical_verification_design import check as independent
    study=Path(study);spec=json.loads((study/'protocol.json').read_text())
    rows=[]
    def record(name, fn, passes=False, sentinel=False):
        error=None;constructed=False
        try:fn()
        except Constructed:constructed=True
        except (ValueError,KeyError,TypeError,FileNotFoundError,StopIteration) as exc:error=str(exc)
        if sentinel:
            if not constructed:raise RuntimeError((name,'baseline did not reach construction',error))
        elif (error is None)!=passes:raise RuntimeError((name,'wrong admission',error))
        rows.append(dict(name=name,expected_admit=passes or sentinel,admitted=error is None,reason=error))
    record('genuine-design',lambda:design.admit(study),True)
    record('independent-canonical-design',lambda:independent(spec),True)
    mutations={
        'stage':lambda s:s.update(stage='unknown'),
        'horizon':lambda s:s.update(steps=1,checkpoints=[0,1]),
        'qualification-missing':lambda s:s.pop('qualifications'),
        'qualification-digest':lambda s:s['qualifications']['4'].update(sha256='0'*64),
        'batch':lambda s:s.update(batch_size=64),
        'epsilon':lambda s:s['optimizer'].update(epsilon=1e-6),
        'decay':lambda s:s['optimizer'].update(weight_decay=.2),
        'betas':lambda s:s['optimizer'].update(betas=[.9,.999]),
        'dtype':lambda s:s['runtime'].update(dtype='float16'),
        'rate':lambda s:s.update(learning_rate=.1),
        'unknown-arm':lambda s:s['cases'][0].update(arm='unknown'),
        'schema':lambda s:s.update(schema='unknown'),
        'duplicate-case':lambda s:s['cases'].append(copy.deepcopy(s['cases'][0])),
        'missing-case':lambda s:s.update(cases=s['cases'][1:]),
        'checkpoint-order':lambda s:s.update(checkpoints=[0,2048,512,16384]),
        'schedule-clock':lambda s:s['schedule'].update(clock='after optimizer'),
        'optimizer-extra-field':lambda s:s['optimizer'].update(other=0),
        'nonfinite-rate':lambda s:s.update(learning_rate=float('nan')),
        'boolean-batch':lambda s:s.update(batch_size=True),
        'tokenizer-identity':lambda s:s['native_assets'].update({next(n for n in s['native_assets'] if n.endswith('tokenizer.model')):'0'*64}),
        'native-identity':lambda s:s['native_assets'].update({next(n for n in s['native_assets'] if n.endswith('modeling_pldrllm.py')):'0'*64}),
        'observation-manifest':lambda s:s.update(development_manifest_sha256='0'*64),
    }
    with tempfile.TemporaryDirectory(prefix='physical-admission-',dir='/tmp') as td:
        root=Path(td)
        def setup(name,changed,draw_edit=None):
            folder=root/name;folder.mkdir()
            if draw_edit:
                with np.load(study/'draws.npz') as z:d={k:z[k].copy() for k in z.files}
                draw_edit(d);np.savez(folder/'draws.npz',**d);changed['draws_sha256']=sha256(folder/'draws.npz')
            else:(folder/'draws.npz').symlink_to(study/'draws.npz')
            write_json(folder/'protocol.json',changed)
            return folder
        for label,edit in mutations.items():
            changed=copy.deepcopy(spec);edit(changed)
            # Nonfinite JSON is deliberately an invalid input file.
            if label=='nonfinite-rate':
                folder=root/label;folder.mkdir();(folder/'draws.npz').symlink_to(study/'draws.npz')
                (folder/'protocol.json').write_text(json.dumps(changed))
            else:folder=setup(label,changed)
            for entry,fn in [('validate',lambda:design.admit(folder)),
                             ('queue',lambda:queue.run(folder,4,'cuda:0')),
                             ('worker',lambda:worker.worker(folder,spec['cases'][0]['name'],'cuda:0'))]:
                record(label+'/'+entry,fn)
            if {p.name for p in folder.iterdir()}!={'protocol.json','draws.npz'}:raise RuntimeError('Refusal wrote output')
        for label,edit in {
            'draw-shape':lambda d:d.update(index=d['index'][:,:31]),
            'draw-dtype':lambda d:d.update(site=d['site'].astype(float)),
            'draw-extra':lambda d:d.update(extra=np.array([1])),
            'draw-cell':lambda d:d['cell'].__setitem__(0,100),
            'draw-site':lambda d:d['site'].__setitem__(0,10000),
            'draw-sample':lambda d:d['index'].__setitem__((0,0),1000000),
            'draw-repeat':lambda d:d['index'].__setitem__((0,0),d['index'][0,1]),
        }.items():
            folder=setup(label,copy.deepcopy(spec),edit)
            for entry,fn in [('validate',lambda:design.admit(folder)),('queue',lambda:queue.run(folder,4,'cuda:0')),
                             ('worker',lambda:worker.worker(folder,spec['cases'][0]['name'],'cuda:0'))]:
                record(label+'/'+entry,fn)
            if {p.name for p in folder.iterdir()}!={'protocol.json','draws.npz'}:raise RuntimeError('Draw refusal wrote output')
        for label,edit in {
            'qualification-width':lambda q:q.update(heads=8),
            'qualification-status':lambda q:q.update(status='failed'),
            'qualification-source':lambda q:q['sources'].update({'scripts/physical_study.py':'0'*64}),
            'qualification-native':lambda q:q['native_assets'].update(extra='0'*64),
            'qualification-branches':lambda q:q.update(branches=['full']),
            'qualification-detached':lambda q:q.update(verified_files={}),
            'qualification-count':lambda q:q.update(qualification_updates=1),
        }.items():
            q=json.loads(Path(spec['qualifications']['4']['path']).read_text());edit(q)
            qp=root/(label+'.json');write_json(qp,q)
            changed=copy.deepcopy(spec);changed['qualifications']['4']=dict(path=str(qp),sha256=sha256(qp))
            folder=setup(label,changed)
            for entry,fn in [('validate',lambda:design.admit(folder)),('queue',lambda:queue.run(folder,4,'cuda:0')),
                             ('worker',lambda:worker.worker(folder,spec['cases'][0]['name'],'cuda:0'))]:record(label+'/'+entry,fn)
            if {p.name for p in folder.iterdir()}!={'protocol.json','draws.npz'}:raise RuntimeError('Qualification refusal wrote output')
        folder=setup('genuine-worker',copy.deepcopy(spec))
        original=worker.load_model;original_cuda=worker.torch.cuda.set_device
        worker.load_model=lambda *args:(_ for _ in ()).throw(Constructed())
        worker.torch.cuda.set_device=lambda *args:None
        try:record('genuine-worker-construction',lambda:worker.worker(folder,spec['cases'][0]['name'],'cuda:0'),sentinel=True)
        finally:worker.load_model=original;worker.torch.cuda.set_device=original_cuda
        for stage,steps,lr in [('unknown',12,.0003),('confirmation',1,.0003),('confirmation',16384,.2)]:
            invalid=design.ROOT/'readout-execution-20260913/physical/invalid-preparation'
            record('prepare-'+stage+'-'+str(steps),lambda:worker.prepare(invalid,stage,steps,lr))
            if invalid.exists():raise RuntimeError('Invalid preparation created destination')
        # A resume record with matching update count still must bind the protocol.
        folder=setup('resume',copy.deepcopy(spec));case=spec['cases'][0];out=folder/case['name'];out.mkdir()
        write_json(out/'manifest.json',dict(status='complete',completed_updates=spec['steps'],case=case,
            protocol_sha256='0'*64,files={},checkpoints=[]))
        record('resume-protocol',lambda:queue.run(folder,4,'cuda:0'))
        record('unknown-worker-case',lambda:worker.worker(study,'h4-unknown','cuda:0'))
    write_json(output,dict(status='passed',optimized=not __debug__,native_updates=0,checks=rows,
        producer_sha256=sha256(REPO/'scripts/physical_study.py'),checker_sha256=sha256(__file__),
        protocol_sha256=sha256(study/'protocol.json'),qualifications=spec['qualifications']))
    print('passed',len(rows),'physical entrypoint checks; zero native updates',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:child(a.study,Path(a.output))
    else:
        out=Path(a.output);out.mkdir(exist_ok=False);rows=[]
        for name,flags in [('ordinary',[]),('optimized',['-O'])]:
            with (out/(name+'.log')).open('w') as log:
                subprocess.run([sys.executable,*flags,__file__,'--child','--study',a.study,
                    '--output',str(out/(name+'.json'))],stdout=log,stderr=subprocess.STDOUT,check=True)
            rows+=json.loads((out/(name+'.json')).read_text())['checks']
        write_json(out/'verification.json',dict(status='passed',native_updates=0,checks=rows,
            checker_sha256=sha256(__file__),files={p.name:sha256(p) for p in out.iterdir() if p.is_file()}))
        print('passed',len(rows),'ordinary/optimized physical admission checks',flush=True)
