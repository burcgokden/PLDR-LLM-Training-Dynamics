#!/usr/bin/env python3
"""Compare all deductive tensors, late single-pass branches, and passive relaxation."""
from companion_paths import configured_path
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from model_rg.provenance import sha256,write_json
from model_rg.potential_avalanches import excursions,complete_events,pareto_fit,tail_diagnostics,surrogate_diagnostics
ROOT=Path(configured_path('data:model/potential-avalanche-20260913'))
REPO=Path(__file__).resolve().parents[1]

def corr(x,y):
    return float(spearmanr(x,y).statistic) if np.ptp(x)>0 and np.ptp(y)>0 else None

def describe_signal(x,lr,rng,replicates,full_null=False):
    result=[]
    for phase,(lo,hi) in {'whole':(0,len(x)),'first_half':(0,len(x)//2),'second_half':(len(x)//2,len(x))}.items():
        for clock,y in [('update',x[lo:hi]),('lr_normalized',x[lo:hi]/lr[lo:hi])]:
            for q in [.8,.9,.95]:
                es=complete_events(excursions(y,float(np.quantile(y,q))))
                sizes=np.array([e['size'] for e in es]);dur=np.array([e['duration'] for e in es])
                item=dict(phase=phase,clock=clock,quantile=q,events=len(es),
                    mean_duration=float(dur.mean()) if len(dur) else None,
                    max_duration=int(dur.max()) if len(dur) else None,
                    fit=pareto_fit(sizes))
                if phase=='whole' and q==.9:
                    item['fit']=tail_diagnostics(sizes,rng,bootstraps=replicates)
                    if full_null:item['surrogates']=surrogate_diagnostics(y,rng,q=.9,repeats=replicates)
                result.append(item)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--replicates',type=int,default=199);p.add_argument('--output',type=Path,default=ROOT/'analysis');a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True);records=[];missing=[]
    signature={str(Path(__file__).relative_to(REPO)):sha256(__file__),
        'src/model_rg/potential_avalanches.py':sha256(REPO/'src/model_rg/potential_avalanches.py')}
    spec=json.loads((ROOT/'protocols/continuations.json').read_text())
    allarrays={}
    for index,case in enumerate(spec['cases']):
        path=ROOT/'continuations'/case['name'];mf=path/'manifest.json';cache=out/('continuation-'+case['name']+'.json');arrayfile=cache.with_suffix('.npz')
        if not mf.exists():missing.append(case['name']);continue
        if cache.exists():
            r=json.loads(cache.read_text())
            if r.get('analysis_sources')==signature and r.get('replicates')==a.replicates and r['manifest_sha256']==sha256(mf):
                records.append(r);allarrays[case['name']]=dict(np.load(arrayfile));continue
        m=json.loads(mf.read_text())
        if m['status']!='complete':raise ValueError('Unfinished continuation')
        for f,h in m['artifacts'].items():
            if sha256(path/f)!=h:raise ValueError('Changed continuation artifact')
        z=np.load(path/'activity.npz');f=z['fields'][1:];ts=z['tensor_statistics'];lr=z['lr'][:,0]
        activity=np.sqrt(np.mean(f[...,0]**2,axis=(2,3)))
        base=np.mean(f[...,2]**2,axis=(2,3));power=np.mean(f[...,1]**2,axis=(2,3));cross=np.mean(f[...,3],axis=(2,3))
        tensors=np.sqrt(np.mean(ts[1:,...,1]**2,axis=(2,3))) # [time, probe, kind]
        levels=np.sqrt(np.mean(ts[... ,0]**2,axis=(2,3)))
        if np.max(np.abs(activity**2-base-power-2*cross))>1e-12:raise ValueError('Activity identity failed')
        rr=dict(case=case,manifest_sha256=sha256(mf),analysis_sources=signature,replicates=a.replicates,
            no_repetition_including_parent=m['no_repetition_including_parent'],runtime_seconds=m['runtime_seconds'],
            max_cuda_memory_bytes=m['max_cuda_memory_bytes'],parent_objective=m['parent_objective'],
            base_energy_fraction=(base.sum(0)/(base.sum(0)+power.sum(0))).tolist(),
            cross_fraction=(2*cross.sum(0)/(base.sum(0)+power.sum(0))).tolist(),
            initial_row_ratio=np.mean(z['fields'][0,...,8],axis=(1,2)).tolist(),
            final_row_ratio=np.mean(z['fields'][-1,...,8],axis=(1,2)).tolist(),
            probe_activity_correlation=corr(activity[:,0],activity[:,1]),
            initial_probe_nll=z['probe_nll'][0].tolist(),final_probe_nll=z['probe_nll'][-1].tolist(),
            signals={})
        rng=np.random.default_rng(915001+index)
        signals={'log_potential':activity}
        signals.update({name:tensors[:,:,j] for j,name in enumerate(m['tensor_names'])})
        for name,x in signals.items():
            rr['signals'][name]=dict(activity_median=np.median(x,axis=0).tolist(),
                activity_q95=np.quantile(x,.95,axis=0).tolist(),
                probe_spearman=corr(x[:,0],x[:,1]),
                predictive_kl_spearman=corr(x[:,0],z['predictive_step_kl'][1:,0]),
                rate_normalized_predictive_kl_spearman=corr(x[:,0]/lr,z['predictive_step_kl'][1:,0]/lr**2),
                logpotential_spearman=corr(x[:,0],activity[:,0]),
                settings=describe_signal(x[:,0],lr,rng,a.replicates,full_null=(name in ['log_potential','curvature_G'])))
        arr=dict(activity=activity,lr=lr,tensor_activity=tensors,tensor_levels=levels,
             base_energy=base,power_energy=power,cross_energy=cross,
             predictive_step_kl=z['predictive_step_kl'][1:],row_ratio=np.mean(z['fields'][...,8],axis=(2,3)))
        np.savez_compressed(arrayfile,**arr);write_json(cache,rr);records.append(rr);allarrays[case['name']]=arr
        print(case['name'],rr['base_energy_fraction'],flush=True)
    relaxation=[]
    for case in json.loads((ROOT/'protocols/relaxation.json').read_text())['cases']:
        path=ROOT/'relaxation'/case['name'];mf=path/'manifest.json'
        if not mf.exists():missing.append('relaxation-'+case['name']);continue
        m=json.loads(mf.read_text());rr=dict(case=case,manifest_sha256=sha256(mf),branches={})
        for name,b in m['branches'].items():
            f=path/(name+'.npz')
            if sha256(f)!=b['artifact_sha256']:raise ValueError('Changed relaxation artifact')
            z=np.load(f);t=z['tensor_statistics'];act=np.sqrt(np.mean(z['fields'][1:,0,...,0]**2,axis=(1,2)))
            rr['branches'][name]=dict(b,first_activity=float(act[0]),last_activity=float(act[-1]),
                max_activity=float(act.max()),peak_step=int(act.argmax()+1),
                integrated_tensor_activity=np.sqrt(np.mean(t[1:,0,...,1]**2,axis=(1,2))).sum(0).tolist(),
                peak_predictive_kl=float(z['predictive_step_kl'][:,0].max()))
        rr['memory_to_reset_activity_ratio']=rr['branches']['zero_gradient']['summed_logpotential_activity']/max(rr['branches']['zero_gradient_reset_moments']['summed_logpotential_activity'],1e-30)
        relaxation.append(rr)
    # Existing early paths already retain full-coordinate exponent and curvature
    # increments. The base-log series is reconstructed from the fixed entry panel.
    early=[]
    for path in sorted((ROOT/'runs').glob('*')):
        if not (path/'manifest.json').exists():continue
        cache=out/('other-early-'+path.name+'.json')
        if cache.exists():
            item=json.loads(cache.read_text())
            if item.get('analysis_sources')==signature:early.append(item);continue
        z=np.load(path/'activity.npz');f=z['fields'][1:,0];lr=z['lr']
        logbase=np.diff(z['logbase'][:,0].astype(float),axis=0)
        signals={'exponent_P':np.sqrt(np.mean(f[...,13]**2,axis=(1,2))),
            'curvature_G':np.sqrt(np.mean(f[...,6]**2,axis=(1,2))),
            'sampled_logbase':np.sqrt(np.mean(logbase**2,axis=(1,2,3)))}
        item=dict(run_id=path.name,analysis_sources=signature,signals={})
        for n,x in signals.items():
            vals=[]
            for clock,y in [('update',x[256:]),('lr_normalized',x[256:]/lr[256:])]:
                es=complete_events(excursions(y,float(np.quantile(y,.9))))
                vals.append(dict(clock=clock,events=len(es),fit=pareto_fit([e['size'] for e in es])))
            item['signals'][n]=vals
        write_json(cache,item);early.append(item)
    if missing and not a.partial:raise ValueError('Missing fixed follow-ups: '+str(missing))
    write_json(out/'followups-summary.json',dict(status='partial' if missing else 'complete',analysis_sources=signature,
        continuations=records,relaxation=relaxation,early_other_tensors=early,missing=missing))
    if not missing:figures(records,allarrays,relaxation,out)


def figures(records,arrays,relaxation,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,6.2),sharex=True)
    for col,recipe in enumerate(['reference1','subcritical1']):
        for drive,color in [('native','#43526b'),('anneal','#bb5c27')]:
            name=recipe+'-'+drive;x=arrays[name]
            for row,values in enumerate([x['activity'][:,0],x['activity'][:,0]/x['lr']]):
                smooth=values.reshape(-1,32).mean(1);t=65536+16+32*np.arange(len(smooth))
                axes[row,col].plot(t,smooth,color=color,label=drive,lw=1)
                axes[row,col].set_yscale('log')
            axes[0,col].set_title('Strong row collapse' if col==0 else 'Partial row collapse')
        axes[1,col].set_xlabel('Cumulative optimizer update');axes[0,col].legend()
    axes[0,0].set_ylabel('Log-potential activity');axes[1,0].set_ylabel('Activity / learning rate')
    fig.tight_layout();fig.savefig(out/'potential_continuations.pdf');fig.savefig(out/'potential_continuations.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.3))
    kinds=['A','M','P','V','G']
    for j,r in enumerate(records):
        x=arrays[r['case']['name']]
        relative=np.median(x['tensor_activity'][:,0]/x['tensor_levels'][0,0].clip(1e-30),axis=0)
        label=r['case']['name'].replace('reference1-','Strong: ').replace('subcritical1-','Partial: ')
        axes[0].plot(kinds,relative,marker='o',label=label,lw=1)
    axes[0].set(yscale='log',ylabel='Median increment / incoming tensor RMS');axes[0].legend(fontsize=7)
    for r in relaxation:
        label={'reference1':'Strong','subcritical1':'Partial','early-h2':'Early 2h','early-h8':'Early 8h'}[r['case']['name']]
        axes[1].scatter(label,r['memory_to_reset_activity_ratio'])
    axes[1].axhline(1,color='0.5',ls=':');axes[1].set(yscale='log',ylabel='Zero-gradient activity / reset-moment activity')
    axes[1].tick_params(axis='x',labelrotation=20)
    fig.tight_layout();fig.savefig(out/'deductive_comparison.pdf');fig.savefig(out/'deductive_comparison.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
