#!/usr/bin/env python3
"""Analyze every frozen matched single-pass job; write only the requested JSON."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.special import logsumexp
if __package__:
    from .analyze_finite_metric_budget import budget, digest
else:
    from analyze_finite_metric_budget import budget, digest


def analyze(study, output):
    study=Path(study).resolve();output=Path(output).resolve()
    if output.exists(): raise FileExistsError(output)
    start=time.monotonic()
    protocol=study/'protocol.json';spec=json.loads(protocol.read_text())
    if spec['schema']!='matched-onepass-v1' or spec['role']!='scientific':
        raise ValueError('Expected the frozen matched single-pass family')
    if len(spec['jobs'])!=24: raise ValueError('Incomplete design')
    checked={str(protocol):digest(protocol),str(study/'selection.npz'):digest(study/'selection.npz')}
    if checked[str(study/'selection.npz')]!=spec['selection_sha256']:
        raise ValueError('Changed selection')
    for name,h in spec['input_sha256'].items():
        if digest(name)!=h: raise ValueError('Changed frozen input: '+name)
        checked[name]=h
    repo=Path(__file__).resolve().parents[1]
    for name,h in spec['producer_sources'].items():
        if digest(repo/name)!=h: raise ValueError('Changed native producer: '+name)
        checked[str(repo/name)]=h
    selected=np.load(study/'selection.npz',allow_pickle=False)
    # Verify the complete document separation using actual content identities.
    corpus=next(Path(p).parent for p in spec['input_sha256'] if p.endswith('/refinedweb-onepass-524288/records.json'))
    probes=next(Path(p).parent for p in spec['input_sha256'] if p.endswith('/data/short/manifest.json'))
    pm=json.loads((probes/'manifest.json').read_text())
    if digest(probes/'records.json')!=pm['records_sha256']: raise ValueError('Changed probe records')
    checked[str(probes/'records.json')]=pm['records_sha256']
    train_hashes={r['content_sha256'] for r in json.loads((corpus/'records.json').read_text())}
    probe_records=json.loads((probes/'records.json').read_text())
    probe_hashes={probe_records[int(i)]['content_sha256'] for i in selected['probe_rows']}
    if train_hashes & probe_hashes: raise ValueError('Evaluation document leakage')
    pt=np.load(probes/'tokens.npy',mmap_mode='r');po=np.load(probes/'offsets.npy');pr=selected['probe_rows']
    if not np.array_equal(selected['probes'],pt[pr[:,None],po[pr,None]+np.arange(65)]):
        raise ValueError('Probe construction mismatch')
    records=[];cells=[];prefixes={};maximum_violation=0.;max_identity=0.
    for job in spec['jobs']:
        run=study/'runs'/job['run_id'];mp=run/'manifest.json';m=json.loads(mp.read_text())
        if m['status']!='complete' or m['job']!=job or m['completed_steps']!=job['steps'] or m['role']!='scientific':
            raise ValueError('Incomplete or mismatched scientific job')
        if m['protocol_sha256']!=checked[str(protocol)] or m['producer_sources']!=spec['producer_sources']:
            raise ValueError('Changed run binding')
        checked[str(mp)]=digest(mp)
        for name,h in m['artifacts'].items():
            if digest(run/name)!=h: raise ValueError('Changed native artifact')
            checked[str(run/name)]=h
        with np.load(run/'observations.npz',allow_pickle=False) as z:
            arrays={k:z[k] for k in z.files}
        if not all(np.isfinite(v).all() for v in arrays.values()): raise ValueError('Nonfinite native record')
        steps=arrays['steps'];logits=arrays['logits'].astype(np.float64)
        if not np.array_equal(steps,np.arange(0,1025,32)) or logits.shape!=(33,8,32000):
            raise ValueError('Changed observation design')
        if not np.array_equal(arrays['blocks'],selected[f'blocks_{job["seed"]}']):
            raise ValueError('Changed native stream')
        if len(np.unique(arrays['blocks']))!=32768: raise ValueError('Repeated supervised block')
        x=np.arange(1,1025)
        expected=np.full(1024,8e-4)
        if job['schedule']=='cosine':
            expected=8e-4*(.1+.45*(1+np.cos(np.pi*(x-128)/(1024-128))))
        expected[:128]=8e-4*x[:128]/128
        np.testing.assert_allclose(arrays['lr'],expected,rtol=2e-15,atol=1e-18)
        key=(job['heads'],job['seed'])
        prefix={k:arrays[k][:5].copy() for k in ['logits','row_ratio','logbase','power']}
        if key in prefixes:
            for name in prefix:
                if not np.array_equal(prefix[name],prefixes[key][name]):
                    raise ValueError('Matched warmup diverged: '+name)
        else: prefixes[key]=prefix
        p=arrays['power'][:,None];b=arrays['logbase'];q=p*b
        dp=np.diff(p,axis=0);db=np.diff(b,axis=0)
        u=dp*(b[:-1]+b[1:])/2;v=(p[:-1]+p[1:])*db/2
        defect=float(np.max(np.abs(np.diff(q,axis=0)-u-v)))
        max_identity=max(max_identity,defect)
        if defect>1e-11: raise ValueError('Finite product identity failed')
        energy=np.mean(np.diff(q,axis=0)**2,axis=(2,3,4))
        activity=np.sqrt(energy)
        exposure=np.add.reduceat(arrays['lr'],np.arange(0,1024,32))
        # Each risk evaluation uses the same 128 external targets.
        risks={k:float(np.mean(vals)) for k,vals in m['evaluation_nll'].items()}
        for step,vals in m['evaluation_nll'].items():
            a=logits[list(steps).index(int(step))]
            lp=a-logsumexp(a,axis=-1,keepdims=True)
            nll=-lp[np.arange(8),selected['probes'][:8,64]]
            np.testing.assert_allclose(nll,np.asarray(vals)[:8],atol=2e-6,rtol=1e-7)
        record=dict(job=job,parameter_count=m['parameter_count'],evaluation_nll=risks,
            final_nll_per_probe=m['evaluation_nll']['1024'],
            row_ratio_initial=float(arrays['row_ratio'][0].mean()),
            row_ratio_final=float(arrays['row_ratio'][-1].mean()),
            row_ratio_by_probe=arrays['row_ratio'][-1].mean((1,2)).tolist(),
            late_block_activity=float(activity[-8:].mean()),
            late_clock_velocity=float((activity[-8:]/exposure[-8:,None]).mean()),
            observed_steps=steps.tolist(),nll_path=(-logits[np.arange(33)[:,None],np.arange(8)[None,:],selected['probes'][None,:8,64]]+logsumexp(logits,axis=-1)).mean(1).tolist(),
            row_path=arrays['row_ratio'].mean((1,2,3)).tolist(),
            block_activity_path=activity.mean(1).tolist(),
            unique_blocks=32768,consumed_fraction=1/128,
            runtime_seconds=m['runtime_seconds'],max_cuda_memory_bytes=m['max_cuda_memory_bytes'])
        records.append(record)
        for probe in range(8):
            a=logits[:,probe]
            for block in [1,2,4,8,16,32]:
                r=budget(a[:-block:block],a[block::block])
                violation=np.maximum(r['lower']-r['kl'],r['kl']-r['upper'])
                maximum_violation=max(maximum_violation,float(violation.max()))
                if not all(np.isfinite(v).all() for v in r.values()) or np.any(violation>2e-12*(1+np.abs(r['kl']))):
                    raise ValueError('Metric budget violation')
                cells.append(dict(run_id=job['run_id'],probe=probe+1,block=block,
                                  count=len(r['kl']),**{k:v.tolist() for k,v in r.items()}))
        print(job['run_id'],'checked',flush=True)
    contrasts=[]
    for heads in [2,4,8,14]:
        for seed in [914241,914242,914243]:
            pair={r['job']['schedule']:r for r in records if r['job']['heads']==heads and r['job']['seed']==seed}
            a,b=pair['plateau'],pair['cosine']
            contrasts.append(dict(heads=heads,seed=seed,
                delta_nll=b['evaluation_nll']['1024']-a['evaluation_nll']['1024'],
                delta_row_ratio=b['row_ratio_final']-a['row_ratio_final'],
                activity_ratio=b['late_block_activity']/a['late_block_activity'],
                clock_velocity_ratio=b['late_clock_velocity']/a['late_clock_velocity']))
    summaries=[]
    for block in [1,2,4,8,16,32]:
        rows=[r for r in cells if r['block']==block]
        cat=lambda k:np.concatenate([np.asarray(r[k]) for r in rows])
        kl,q,lo,hi=map(cat,['kl','quadratic','lower','upper']);active=q>1e-12
        error=np.abs(kl[active]/q[active]-1)
        summaries.append(dict(block_updates=32*block,spans=len(kl),
            median_relative_error=float(np.median(error)),max_relative_error=float(error.max()),
            maximum_oscillation=float(cat('oscillation').max()),
            envelope_one_percent=int(np.sum(np.maximum(hi/q-1,1-lo/q)<=.01))))
    result=dict(status='passed',schema='matched-onepass-analysis-v1',
        scientific_paths=len(records),scientific_updates=sum(r['job']['steps'] for r in records),
        independent_seed_order_pairs=3,paired_schedule_contrasts=contrasts,
        evaluation_documents=len(probe_hashes),evaluation_training_overlap=0,
        observations_per_path=33,fixed_geometry_probes=8,scale_cells=len(cells),
        endpoint_spans=sum(c['count'] for c in cells),maximum_bound_violation=max(0.,maximum_violation),
        maximum_product_identity_defect=max_identity,records=records,metric_cells=cells,
        metric_summaries=summaries,checked_sha256=checked,analyzer_sha256=digest(__file__),
        metric_analyzer_sha256=digest(Path(__file__).with_name('analyze_finite_metric_budget.py')),
        runtime_seconds=time.monotonic()-start,
        scientific_gpu_seconds=sum(r['runtime_seconds'] for r in records),
        maximum_cuda_memory_bytes=max(r['max_cuda_memory_bytes'] for r in records),
        scope='Fixed realized corpus and fixed evaluation documents. Seeds jointly vary initialization and source ordering. All comparisons are finite consuming transients; no population critical exponents or autonomous forecasts.')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f: json.dump(result,f,indent=2);f.write('\n')
    print('PASS',result['scientific_paths'],'paths,',result['endpoint_spans'],'spans',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();analyze(a.study,a.output)
