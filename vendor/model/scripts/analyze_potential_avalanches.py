#!/usr/bin/env python3
"""Reconstruct finite potential activity, threshold excursions, and matched nulls."""
from companion_paths import legacy_path
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from model_rg.provenance import sha256,write_json
from model_rg.potential_avalanches import (excursions,complete_events,pareto_fit,tail_diagnostics,
                                           surrogate_diagnostics,event_summary)
ROOT=Path(legacy_path('/pldr-data/model/potential-avalanche-20260913'))
REPO=Path(__file__).resolve().parents[1]
PHASES={'whole':(0,2048),'warmup':(0,256),'early':(256,768),
        'middle':(768,1280),'late':(1280,2048),'postwarm':(256,2048)}

def finite_corr(x,y):
    return float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None


def analyze(path,replicates):
    meta=json.loads((path/'manifest.json').read_text());job=meta['job']
    if meta['status']!='complete':raise ValueError('Incomplete fixed condition')
    for f,digest in meta['artifacts'].items():
        if sha256(path/f)!=digest:raise ValueError('Artifact changed: '+str(path/f))
    z=np.load(path/'activity.npz')
    if len(z['loss'])!=job['steps']:raise ValueError('Incomplete record')
    if not all(np.isfinite(z[k]).all() for k in z.files):raise ValueError('Nonfinite raw value')
    # [time, probe, layer, head, observable]; first observation precedes training.
    f=z['fields'][1:];lr=z['lr'];activity=np.sqrt(np.mean(f[...,0]**2,axis=(2,3)))
    fullbase=np.mean(f[...,2]**2,axis=(2,3));fullpower=np.mean(f[...,1]**2,axis=(2,3))
    cross=np.mean(f[...,3],axis=(2,3))
    identity=np.max(np.abs(activity**2-(fullbase+fullpower+2*cross)))
    if identity>1e-12:raise ValueError('Squared-increment decomposition failed')
    result=dict(run_id=path.name,job=job,manifest_sha256=sha256(path/'manifest.json'),
        parameter_count=meta['parameter_count'],runtime_seconds=meta['runtime_seconds'],
        max_cuda_memory_bytes=meta['max_cuda_memory_bytes'],
        examples=job['steps']*32,exposed_input_tokens=job['steps']*32*64,
        initial_nll=float(np.mean(meta['evaluation_nll']['0'])),
        final_nll=float(np.mean(meta['evaluation_nll'][str(job['steps'])])),
        squared_increment_identity_max_error=float(identity),
        log_identity_max_error=float(f[...,12].max()),
        phase_summaries=[],excursion_settings=[],primary=[],frozen_thresholds=[])
    rng=np.random.default_rng(914000+job['seed']+job['heads'])
    for probe in [0,1]:
        for phase,(lo,hi) in PHASES.items():
            a=activity[lo:hi,probe];be=fullbase[lo:hi,probe];pe=fullpower[lo:hi,probe];cr=cross[lo:hi,probe]
            ps=dict(probe=probe,phase=phase,activity_median=float(np.median(a)),activity_q95=float(np.quantile(a,.95)),
                rms_velocity_median=float(np.median(a/lr[lo:hi])),lr_spearman=finite_corr(a,lr[lo:hi]),
                base_energy_fraction=float(be.sum()/(be.sum()+pe.sum())),
                cross_fraction=float(2*cr.sum()/(be.sum()+pe.sum())),
                row_ratio_median=float(np.median(f[lo:hi,probe,:,:,8])),
                lag1_activity_correlation=finite_corr(a[1:],a[:-1]))
            result['phase_summaries'].append(ps)
            for clock,x in [('update',a),('lr_normalized',a/lr[lo:hi])]:
                for bins in [1,2,4]:
                    b=x[:len(x)//bins*bins].reshape(-1,bins).mean(1)
                    for q in [.8,.9,.95]:
                        threshold=float(np.quantile(b,q));ee=excursions(b,threshold,bins);events=complete_events(ee)
                        sizes=np.array([e['size'] for e in events]);durations=np.array([e['duration'] for e in events])
                        record=dict(probe=probe,phase=phase,clock=clock,bin_width=bins,quantile=q,
                            threshold=threshold,events=len(events),censored=len(ee)-len(events),
                            mean_duration=float(durations.mean()) if len(durations) else None,
                            max_duration=int(durations.max()) if len(durations) else None,
                            size_fit=pareto_fit(sizes),
                            unique_durations=int(len(np.unique(durations))))
                        result['excursion_settings'].append(record)
                        if probe==0 and phase=='postwarm' and bins==1 and q==.9:
                            primary=dict(record,events_raw=ee,
                                tail_diagnostics=tail_diagnostics(sizes,rng,bootstraps=replicates),
                                surrogate_diagnostics=surrogate_diagnostics(b,rng,q=q,repeats=replicates))
                            result['primary'].append(primary)
        for clock,x in [('update',activity[:,probe]),('lr_normalized',activity[:,probe]/lr)]:
            for q in [.8,.9,.95]:
                threshold=float(np.quantile(x[256:512],q))
                for phase,(lo,hi) in {'middle':(768,1280),'late':(1280,2048)}.items():
                    ee=excursions(x[lo:hi],threshold);es=complete_events(ee)
                    result['frozen_thresholds'].append(dict(probe=probe,clock=clock,quantile=q,phase=phase,
                        threshold=threshold,active_fraction=float(np.mean(x[lo:hi]>threshold)),events=len(es),censored=len(ee)-len(es)))
    # The selected entry panel diagnoses whether undriven-style empty bins arise
    # under a coordinate threshold. Coordinates are observations, never replicates.
    logbase=z['logbase'].astype(np.float64);power=z['power'].astype(np.float64)[:,None]
    potential=logbase*power;delta=np.diff(potential,axis=0)
    sitelevel=np.quantile(np.abs(delta[256:512]),.95,axis=0)
    site_active=np.abs(delta)>sitelevel
    fraction=site_active.reshape(len(delta),2,-1).mean(-1)
    sitecounts=site_active.reshape(len(delta),2,-1).sum(-1)
    result['site_activity']=dict(coordinates_per_probe=int(np.prod(delta.shape[2:])),
        late_empty_bin_fraction=np.mean(sitecounts[1280:]==0,axis=0).tolist(),
        late_active_fraction_median=np.median(fraction[1280:],axis=0).tolist(),
        primary_probe_agreement=float(np.corrcoef(activity.T)[0,1]))
    arrays=dict(activity=activity,lr=lr,base_energy=fullbase,power_energy=fullpower,cross_energy=cross,
        loss=z['loss'],site_active_fraction=fraction,probe_nll=z['probe_nll'],
        curvature_activity=np.sqrt(np.mean(f[...,6]**2,axis=(2,3))),
        row_ratio=np.mean(f[...,8],axis=(2,3)))
    return result,arrays


def figures(results,arrays,dest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    colors={'constant':'#43526b','warm_plateau':'#168a80','warm_cosine':'#bb5c27'}
    fig,axes=plt.subplots(3,3,figsize=(10,8),sharex=True)
    for col,h in enumerate([2,4,8]):
        for sc in colors:
            rr=[r for r in results if r['job']['heads']==h and r['job']['schedule']==sc]
            if len(rr)!=2:continue
            aa=[arrays[r['run_id']] for r in rr]
            for row,vals in enumerate([
                np.array([a['activity'][:,0] for a in aa]),
                np.array([a['activity'][:,0]/a['lr'] for a in aa]),
                np.array([a['base_energy'][:,0]/(a['base_energy'][:,0]+a['power_energy'][:,0]) for a in aa])]):
                # Nonoverlapping 32-update means for display only; events use raw cadence.
                smooth=vals.reshape(2,-1,32).mean(-1)
                t=np.arange(smooth.shape[1])*32+16
                axes[row,col].plot(t,smooth.mean(0),label=sc.replace('_',' '),color=colors[sc],lw=1)
                axes[row,col].fill_between(t,smooth.min(0),smooth.max(0),color=colors[sc],alpha=.14)
            axes[0,col].set_title(f'{h} heads per layer')
        for row in range(3):
            axes[row,col].axvline(256,color='0.5',ls=':',lw=.8)
            if row<2:axes[row,col].set_yscale('log')
            else:axes[row,col].set_ylim(0,1.02)
        axes[2,col].set_xlabel('Optimizer update')
    for ax,label in zip(axes[:,0],['Log-potential activity','Activity / learning rate','Base contribution fraction']):ax.set_ylabel(label)
    axes[0,0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/'potential_activity.pdf');fig.savefig(dest/'potential_activity.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10,3.2))
    for ax,h in zip(axes,[2,4,8]):
        for sc in colors:
            rr=[r for r in results if r['job']['heads']==h and r['job']['schedule']==sc]
            # Each path is drawn separately. No pooled event fit across seeds.
            for j,r in enumerate(rr):
                e=[e for e in r['primary'][0]['events_raw'] if not e['left_censored'] and not e['right_censored']]
                s=np.sort([v['size'] for v in e])
                if len(s):ax.step(s,1-np.arange(len(s))/len(s),where='post',color=colors[sc],ls='-' if j==0 else ':',label=sc.replace('_',' ') if j==0 else None)
        ax.set(xscale='log',yscale='log',xlabel='Integrated excess log-potential activity',title=f'{h} heads per layer')
    axes[0].set_ylabel('Empirical survival probability');axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(dest/'potential_excursions.pdf');fig.savefig(dest/'potential_excursions.png',dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=199);p.add_argument('--partial',action='store_true');p.add_argument('--output',type=Path,default=ROOT/'analysis');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((ROOT/'protocols/training.json').read_text())
    signature={str(Path(__file__).relative_to(REPO)):sha256(__file__),
        'src/model_rg/potential_avalanches.py':sha256(REPO/'src/model_rg/potential_avalanches.py')}
    results=[];arrays={};missing=[]
    for j in spec['jobs']:
        path=ROOT/'runs'/j['run_id'];cache=out/(j['run_id']+'.json');arrayfile=out/(j['run_id']+'.npz')
        if not (path/'manifest.json').exists():missing.append(j['run_id']);continue
        result=None
        if cache.exists():
            old=json.loads(cache.read_text())
            if old.get('analysis_sources')==signature and old.get('replicates')==a.replicates:
                if old['manifest_sha256']!=sha256(path/'manifest.json'):raise ValueError('Changed input after analysis')
                result=old;arr=dict(np.load(arrayfile))
        if result is None:
            result,arr=analyze(path,a.replicates);result.update(analysis_sources=signature,replicates=a.replicates)
            np.savez_compressed(arrayfile,**arr);write_json(cache,result)
        results.append(result);arrays[j['run_id']]=arr
        print(j['run_id'],result['primary'][0]['events'],result['primary'][0]['tail_diagnostics']['status'],flush=True)
    if missing and not a.partial:raise ValueError('Missing selected paths: '+str(missing))
    summary=dict(status='partial' if missing else 'complete',protocol_sha256=sha256(ROOT/'protocols/training.json'),
        analysis_sources=signature,selected_paths=len(spec['jobs']),complete_paths=len(results),missing=missing,
        total_updates=sum(r['job']['steps'] for r in results),total_gpu_seconds=sum(r['runtime_seconds'] for r in results),
        run_results=results)
    write_json(out/'summary.json',summary)
    if not missing:figures(results,arrays,out)

if __name__=='__main__':main()
