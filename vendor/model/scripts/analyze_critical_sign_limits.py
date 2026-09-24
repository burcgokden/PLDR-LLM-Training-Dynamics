#!/usr/bin/env python3
"""Check exact sign-orbit cumulants and conditional Gaussian approximations.

Orbit representatives have identical predictions in the mathematical native
quotient. They are not independent training realizations. The observations
below concern fixed projection panels, not a critical exponent fit.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json

REPO=Path(__file__).resolve().parents[1]


def orbit_projection_statistics(x,frequencies=(.25,.5,1.)):
    """Last axis is head; all other axes remain fixed observation coordinates."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim<1 or x.shape[-1]<1 or not np.isfinite(x).all():
        raise ValueError('Expected finite signed head projections')
    n=x.shape[-1];power=np.sum(x*x,axis=-1);fourth=np.sum(x**4,axis=-1)
    active=power>0
    normalized=np.divide(x,np.sqrt(power)[...,None],out=np.zeros_like(x),where=active[...,None])
    inverse_effective=np.sum(normalized**4,axis=-1)
    effective=np.divide(1.,inverse_effective,out=np.zeros_like(power),where=active)
    variance=power/n
    cumulant=-2*fourth/n**2
    gaussian_fourth=3*variance**2
    moments=dict(variance=variance,fourth_moment=gaussian_fourth+cumulant,
        fourth_cumulant=cumulant,effective_heads=effective,active=active,
        standardized_orbit_excess=-2*inverse_effective)
    comparisons=[]
    for frequency in frequencies:
        u=float(frequency)*normalized
        exact=np.prod(np.cos(u),axis=-1)
        normal=np.exp(-.5*float(frequency)**2*active)
        error=np.abs(exact-normal)
        bound=np.sum(u**4,axis=-1)/3
        valid=np.max(np.abs(u),axis=-1)<=.5
        violation=np.max(np.maximum(error-bound,0)[valid],initial=0.)
        if violation>256*np.finfo(float).eps:
            raise ValueError('Conditional characteristic-function bound failed')
        comparisons.append(dict(frequency=float(frequency),exact=exact,gaussian=normal,
            absolute_error=error,bound=bound,bound_domain=valid,maximum_bound_violation=float(violation)))
    return moments,comparisons


def mixture_fourth_budget(x):
    """First axis is a finite uniform mixture of orbits, with divisor S."""
    x=np.asarray(x,dtype=np.float64)
    if x.ndim<2 or len(x)<2:raise ValueError('Expected multiple complete saved orbits')
    moments,_=orbit_projection_statistics(x)
    variance=moments['variance'];mean_variance=variance.mean(0)
    conditional= moments['fourth_cumulant'].mean(0)
    mixing=3*np.mean((variance-mean_variance)**2,axis=0)
    fourth=moments['fourth_moment'].mean(0)
    direct=fourth-3*mean_variance**2
    scale=np.maximum(np.abs(conditional)+mixing+fourth,1e-300)
    residual=float(np.max(np.abs(direct-conditional-mixing)/scale))
    if residual>2e-12:raise ValueError('Fourth-moment mixture budget differs')
    return dict(variance=mean_variance,fourth_moment=fourth,fourth_cumulant=direct,
        conditional_sign_contribution=conditional,variance_mixture_contribution=mixing,
        maximum_relative_budget_residual=residual)


def exact_first_panel_check(x):
    """Enumerate every sign for one predeclared panel coordinate, through N=14."""
    x=np.asarray(x,dtype=float);n=x.shape[-1]
    if n>14:return dict(status='outside_enumeration_range',scope='Exact product and moment formulas are still evaluated.')
    signs=2*((np.arange(1<<n,dtype=np.uint64)[:,None]>>np.arange(n,dtype=np.uint64))&1).astype(float)-1
    power=np.sum(x*x,axis=-1);active=power>0
    normalized=np.divide(x,np.sqrt(power)[:,None],out=np.zeros_like(x),where=active[:,None])
    y=signs@normalized.T
    fourth=np.mean(y**4,axis=0);expected=3*active-2*np.sum(normalized**4,axis=-1)
    error4=float(np.max(np.abs(fourth-expected)))
    error2=float(np.max(np.abs(np.mean(y*y,axis=0)-active)))
    char_error=0.
    for t in [.25,.5,1.]:
        char_error=max(char_error,float(np.max(np.abs(np.cos(t*y).mean(0)-np.prod(np.cos(t*normalized),axis=-1)))))
    if max(error4,error2,char_error)>3e-11:raise ValueError('Complete sign enumeration disagrees')
    return dict(status='passed',orbits=len(x),signs_per_orbit=1<<n,projection='First context, first decoder, first fixed operator projection.',
        maximum_second_moment_error=error2,maximum_fourth_moment_error=error4,maximum_characteristic_error=char_error,
        scientific_realizations_added=0)


def summarize(values):
    x=np.asarray(values,dtype=float).reshape(-1)
    if not len(x):return None
    if not np.isfinite(x).all():raise ValueError('Nonfinite projected summary')
    return dict(mean=float(x.mean()),minimum=float(x.min()),maximum=float(x.max()),
        panel_percentiles_05_50_95=np.quantile(x,[.05,.5,.95]).tolist())


def analyze(study,qualification,output):
    if output.exists():raise FileExistsError(output)
    checked={}
    def read(path,status=None):
        d=json.loads(path.read_text());checked[str(path)]=sha256(path)
        if status and d.get('status')!=status:raise ValueError('Incomplete input '+str(path))
        return d
    p=read(study/'protocol.json');verification=read(study/'verification.json','passed')
    native=read(qualification,'passed')
    if verification['schema']!='critical-collective-verification-v1' or verification['training_updates']!=0:
        raise ValueError('Expected independently verified zero-training observations')
    if Path(verification['study']).resolve()!=study:
        raise ValueError('Collective verification belongs to another study')
    if native['schema']!='critical-head-sign-verification-v1' or native['scientific_updates']!=0:
        raise ValueError('Expected separate native head-sign qualification')
    parent=Path(p['study']);training=read(parent/'protocol.json')
    seeds=sorted(training['design']['seeds']);groups=defaultdict(list);basis=None
    for job in p['jobs']:
        folder=study/'runs'/job['run_id'];m=read(folder/'manifest.json','complete')
        if m['job']!=job or m['protocol_sha256']!=sha256(study/'protocol.json') or m['training_updates']!=0:
            raise ValueError('Unbound endpoint observation')
        path=folder/'collectives.npz';digest=sha256(path)
        if digest!=m['observations_sha256']:raise ValueError('Changed operator acquisition')
        checked[str(path)]=digest
        with np.load(path) as arrays:
            current_basis=arrays['operator_basis']
            if basis is None:basis=current_basis
            np.testing.assert_array_equal(current_basis,basis)
            # [context, decoder, projection, head]; projections of G/d.
            x=np.moveaxis(arrays['operator_projections']/64.,-2,-1)
        if x.shape!=(64,5,8,job['heads']):raise ValueError('Changed fixed projection panel')
        groups[(job['environment'],job['heads'],job['control'])].append((job['seed'],x))
    cells=[];unresolved=[]
    for (e,n,g),rows in sorted(groups.items()):
        rows.sort(key=lambda r:r[0])
        if [r[0] for r in rows]!=seeds:
            unresolved.append(dict(environment=e,heads=n,control=g,reason='Incomplete initialization ensemble'));continue
        x=np.stack([r[1] for r in rows]);moments,comparisons=orbit_projection_statistics(x)
        active=moments['active'];mixture=mixture_fourth_budget(x)
        signal=mixture['variance']>0
        standardized_mixture=np.divide(mixture['fourth_cumulant'],mixture['variance']**2,
            out=np.zeros_like(mixture['variance']),where=signal)
        standardized_conditional=np.divide(mixture['conditional_sign_contribution'],mixture['variance']**2,
            out=np.zeros_like(mixture['variance']),where=signal)
        standardized_mixing=np.divide(mixture['variance_mixture_contribution'],mixture['variance']**2,
            out=np.zeros_like(mixture['variance']),where=signal)
        characteristic=[]
        for r in comparisons:
            valid=r['bound_domain']
            characteristic.append(dict(frequency=r['frequency'],absolute_error=summarize(r['absolute_error']),
                bound_domain_fraction=float(valid.mean()),valid_bound=summarize(r['bound'][valid]),
                maximum_bound_violation=r['maximum_bound_violation']))
        cells.append(dict(environment=e,heads=n,control=g,seed_ids=seeds,
            fixed_panel_shape=list(x.shape[1:-1]),zero_projection_fraction=float((~active).mean()),
            effective_heads=summarize(moments['effective_heads'][active]),
            effective_head_fraction=summarize(moments['effective_heads'][active]/n),
            standardized_orbit_excess=summarize(moments['standardized_orbit_excess'][active]),
            characteristic_comparisons=characteristic,
            orbit_variance=summarize(moments['variance']),
            mixture_excess=summarize(standardized_mixture[signal]),
            conditional_sign_excess=summarize(standardized_conditional[signal]),
            variance_mixing_excess=summarize(standardized_mixing[signal]),
            maximum_relative_fourth_budget_residual=mixture['maximum_relative_budget_residual'],
            independent_enumeration=exact_first_panel_check(x[:,0,0,0,:])))
    source_names=['scripts/analyze_critical_sign_limits.py','src/model_rg/provenance.py']
    result=dict(schema='critical-head-sign-limit-v1',status='complete',study=str(study),parent_study=str(parent),
        cells=cells,unresolved_cells=unresolved,additional_training_updates=0,additional_scientific_realizations=0,
        checked_sha256=checked,source_sha256={name:sha256(REPO/name) for name in source_names},
        projection_units='Eight fixed orthonormal projections of G/64, conditional on each fixed context and decoder.',
        characteristic_units='Each scalar orbit is standardized by its own positive variance; zero projections remain degenerate.',
        panel_scope='Panel percentiles describe the fixed coordinates, not independent training replicas or calibrated population confidence intervals.',
        mixture_scope='Uniform mixture of the six saved sign orbits at each fixed panel entry. Population divisors are used for orbit and mixture moments; no sign representative is an extra training realization.',
        interpretation='Exact finite sign-orbit cumulants and a conditional Gaussian characteristic-function bound. The regular square-root averaging normalization is not a critical exponent, and these observations do not identify a unique thermodynamic covariance law.')
    write_json(output,result)
    print(json.dumps(dict(status='complete',cells=len(cells),unresolved=len(unresolved),additional_training_updates=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['study','qualification','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analyze(a.study.resolve(),a.qualification.resolve(),a.output.resolve())
