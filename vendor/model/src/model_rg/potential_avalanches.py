"""Excursion statistics with explicit censoring and noncritical surrogate controls.

A compatible Pareto tail is a distributional diagnostic, not a criticality test.
The iid bootstrap is not interpreted as calibrated inference for a correlated
training trajectory. Nulls preserve different parts of the nonstationary signal.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr


def excursions(activity, threshold, bin_width=1):
    x=np.asarray(activity,dtype=np.float64)
    if x.ndim!=1 or len(x)<1 or not np.isfinite(x).all():raise ValueError('Finite nonempty scalar path required')
    if not np.isfinite(threshold) or bin_width<1:raise ValueError('Invalid threshold or bin width')
    above=x>threshold
    starts=np.flatnonzero(above & ~np.r_[False,above[:-1]])
    ends=np.flatnonzero(above & ~np.r_[above[1:],False])+1
    return [dict(start=int(a*bin_width),stop=int(b*bin_width),
        duration=int((b-a)*bin_width),size=float((x[a:b]-threshold).sum()*bin_width),
        total_activity=float(x[a:b].sum()*bin_width),peak=float(x[a:b].max()),
        left_censored=bool(a==0),right_censored=bool(b==len(x))) for a,b in zip(starts,ends)]


def complete_events(events):
    return [e for e in events if not e['left_censored'] and not e['right_censored']]


def pareto_fit(data,min_tail=50):
    """KS-selected continuous tail on inclusive, distinct-value cutoffs.

    Eligible cutoffs retain at least ``min_tail`` positive finite observations.
    At most 128 cutoffs are searched, equally spaced by rank in the eligible
    distinct-value list (including its endpoints). Ties are never split.
    The cap is a deterministic approximation to the complete cutoff search.
    """
    x=np.sort(np.asarray(data,dtype=float))
    x=x[np.isfinite(x)&(x>0)]
    if len(x)<min_tail:return dict(status='insufficient_events',n=int(len(x)))
    logs=np.log(x);best=None
    _,starts=np.unique(x,return_index=True)
    eligible=starts[starts<=len(x)-min_tail]
    candidate_ranks=np.unique(np.linspace(0,len(eligible)-1,min(128,len(eligible))).astype(int))
    candidates=eligible[candidate_ranks]
    for start in candidates:
        tail=x[start:];den=(logs[start:]-logs[start]).sum()
        if den<=0:continue
        alpha=1+len(tail)/den
        cdf=-np.expm1((1-alpha)*(logs[start:]-logs[start]))
        n=len(tail)
        ks=max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
        item=dict(status='fitted',n=int(len(x)),n_tail=int(n),xmin=float(tail[0]),
            alpha=float(alpha),ks=float(ks),tail_decades=float(np.log10(tail[-1]/tail[0])))
        if best is None or ks<best['ks']:best=item
    return best or dict(status='degenerate',n=int(len(x)))


def tail_diagnostics(data,rng,bootstraps=199,min_tail=50):
    x=np.asarray(data,dtype=float);x=x[np.isfinite(x)&(x>0)]
    fit=pareto_fit(x,min_tail)
    if fit['status']!='fitted':return fit
    xmin=fit['xmin'];alpha=fit['alpha'];tail=x[x>=xmin];body=x[x<xmin]
    synthetic_ks=[];alphas=[]
    for _ in range(bootstraps):
        nt=int(rng.binomial(len(x),len(tail)/len(x)))
        nb=len(x)-nt
        synth=xmin*np.exp(rng.exponential(1/(alpha-1),nt))
        if nb:
            synth=np.r_[synth,rng.choice(body,nb,replace=True)] if len(body) else np.r_[synth,xmin*np.exp(rng.exponential(1/(alpha-1),nb))]
        sf=pareto_fit(synth,min_tail)
        if sf['status']=='fitted':synthetic_ks.append(sf['ks']);alphas.append(sf['alpha'])
    fit['bootstrap_replicates']=len(synthetic_ks)
    fit['iid_bootstrap_ks_p']=(1+int(np.sum(np.asarray(synthetic_ks)>=fit['ks'])))/(1+len(synthetic_ks))
    fit['iid_alpha_bootstrap_interval']=np.quantile(alphas,[.025,.975]).tolist() if alphas else None
    # Alternatives condition on the same selected tail support. AIC includes
    # the same cutoff selection for every model, not an independent search.
    y=tail/xmin;logy=np.log(y);n=len(y)
    llp=np.log(alpha-1)-alpha*logy
    rate=1/np.mean(y-1)
    lle=np.log(rate)-rate*(y-1)
    def objective(v):
        mu,logsig=v;sig=np.exp(logsig)
        ll=-logy-logsig-.5*np.log(2*np.pi)-.5*((logy-mu)/sig)**2-log_ndtr(mu/sig)
        return -ll.sum()
    candidates=[minimize(objective,[float(logy.mean()),float(np.log(max(logy.std(),.05)))],
                   method='L-BFGS-B',bounds=[(-100,100),(-7,5)]),
                minimize(objective,[-5.,1.],method='L-BFGS-B',bounds=[(-100,100),(-7,5)])]
    opt=min(candidates,key=lambda v:v.fun)
    fit.update(log_likelihood_pareto=float(llp.sum()),
        aic_pareto=float(2-2*llp.sum()),aic_exponential=float(2-2*lle.sum()),
        aic_lognormal=float(4+2*opt.fun),lognormal_mu=float(opt.x[0]),lognormal_sigma=float(np.exp(opt.x[1])),
        lognormal_optimizer_success=bool(opt.success),
        pareto_minus_exponential_loglik=float((llp-lle).sum()),
        caveat='Conditional iid tail-fit diagnostic. Correlated events and cutoff selection limit inferential interpretation; no SOC classification follows.')
    return fit


def iaaft(x,rng,iterations=100):
    """Rank and Fourier alternating projections: exact amplitudes, approximate spectrum."""
    x=np.asarray(x,dtype=float);ordered=np.sort(x);target=np.abs(np.fft.rfft(x))
    y=rng.permutation(x)
    for _ in range(iterations):
        phase=np.angle(np.fft.rfft(y))
        z=np.fft.irfft(target*np.exp(1j*phase),n=len(x))
        rank=np.argsort(np.argsort(z,kind='stable'),kind='stable')
        yy=ordered[rank]
        if np.array_equal(y,yy):break
        y=yy
    err=np.linalg.norm(np.abs(np.fft.rfft(y-y.mean()))-np.abs(np.fft.rfft(x-x.mean())))/max(np.linalg.norm(np.fft.rfft(x-x.mean())),1e-30)
    return y,float(err)


def event_summary(x,q=.9):
    t=float(np.quantile(x,q));all_events=excursions(x,t);events=complete_events(all_events)
    if not events:return dict(events=0,censored=len(all_events),mean_duration=None,max_duration=None,size_moment_ratio=None)
    d=np.array([e['duration'] for e in events]);s=np.array([e['size'] for e in events])
    return dict(events=len(events),censored=len(all_events)-len(events),mean_duration=float(d.mean()),
                max_duration=int(d.max()),size_moment_ratio=float(np.sum(s*s)/s.sum()))


def surrogate_diagnostics(x,rng,q=.9,repeats=199):
    x=np.asarray(x,dtype=float);obs=event_summary(x,q);result=dict(observed=obs,nulls={})
    for kind in ['permutation','local_permutation','iaaft']:
        stats=[];errors=[]
        for _ in range(repeats):
            if kind=='permutation':y=rng.permutation(x)
            elif kind=='local_permutation':y=np.concatenate([rng.permutation(x[i:i+64]) for i in range(0,len(x),64)])
            else:y,e=iaaft(x,rng);errors.append(e)
            stats.append(event_summary(y,q))
        result['nulls'][kind]={}
        for field in ['events','mean_duration','max_duration','size_moment_ratio']:
            vals=np.asarray([s[field] for s in stats if s[field] is not None])
            result['nulls'][kind][field]=dict(median=float(np.median(vals)),interval=np.quantile(vals,[.025,.975]).tolist(),
                upper_tail_rank_p=float((1+np.sum(vals>=obs[field]))/(1+len(vals)))) if obs[field] is not None and len(vals) else None
        if errors:result['nulls'][kind]['relative_spectral_error_median']=float(np.median(errors))
    return result
