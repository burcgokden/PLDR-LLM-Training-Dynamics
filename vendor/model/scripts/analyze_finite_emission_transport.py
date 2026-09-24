#!/usr/bin/env python3
"""Full-vocabulary finite KL transport on all paired native offset branches.

Quadrature integrates categorical Fisher covariance along straight logit
segments, with orders 8 and 16 checked against direct endpoint KL. These
segments are observation geometry, not counterfactual native training paths.
Only --output is written. Publication rendering is an explicit separate step
using render_finite_emission_transport.py.
"""
from companion_paths import legacy_path
import argparse,json,time
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from model_rg.provenance import sha256,write_json
ROOT=Path(legacy_path('/pldr-data/model'))
REPO=Path(__file__).resolve().parents[1]

def integrate(a,b,order):
    d=np.diff(a,axis=0);D=np.diff(b,axis=0);e=D-d
    nodes,weights=leggauss(order);result=np.zeros((len(d),5))
    for s,w in zip((nodes+1)/2,weights/2):
        logits=a[:-1]+s*d;p=np.exp(logits-logsumexp(logits,axis=-1,keepdims=True))
        other=b[:-1]+s*D;pp=np.exp(other-logsumexp(other,axis=-1,keepdims=True))
        md=(p*d).sum(-1);me=(p*e).sum(-1);mD=(p*D).sum(-1);mDD=(pp*D).sum(-1)
        vd=(p*d*d).sum(-1)-md*md
        ve=(p*e*e).sum(-1)-me*me
        cov=(p*d*e).sum(-1)-md*me
        vD=(p*D*D).sum(-1)-mD*mD
        vDD=(pp*D*D).sum(-1)-mDD*mDD
        result+=w*(1-s)*np.stack([vd,vDD,2*cov,ve,vDD-vD],axis=-1)
    return result

def direct(a):
    logp=a-logsumexp(a,axis=-1,keepdims=True)
    return (np.exp(logp[:-1])*(logp[:-1]-logp[1:])).sum(-1)

def main(out):
    if out.exists():raise FileExistsError(out)
    start=time.monotonic();records=[];bound={};maximum=0.;largest_order=0
    for label,study in [('A',ROOT/'potential-factorial-20260913'),('B',ROOT/'potential-factorial-disjoint-20260914')]:
        protocol=study/'protocol.json';spec=json.loads(protocol.read_text());bound[str(protocol)]=sha256(protocol)
        for case in spec['cases']:
            mf=study/'runs'/case['name']/'manifest.json';manifest=json.loads(mf.read_text());bound[str(mf)]=sha256(mf)
            for moment in ['keep','reset']:
                arrays=[]
                for offset in ['native','raised']:
                    key=offset+'_'+moment;raw=mf.parent/(key+'.npz');digest=sha256(raw)
                    if digest!=manifest['branches'][key]['artifact_sha256']:raise ValueError('Changed branch')
                    bound[str(raw)]=digest
                    with np.load(raw) as z:arrays.append(z['step_logits'][:,0].astype(np.float64))
                a,b=arrays;ka,kb=direct(a),direct(b);low=integrate(a,b,8);high=integrate(a,b,16);order=16
                err=max(float(np.max(np.abs(high[:,:2]-np.stack([ka,kb],axis=-1)))),float(np.max(np.abs(high-low))))
                if err>2e-11:
                    low=high;high=integrate(a,b,32);order=32
                    err=max(float(np.max(np.abs(high[:,:2]-np.stack([ka,kb],axis=-1)))),float(np.max(np.abs(high-low))))
                if err>2e-11:raise ValueError('Finite categorical quadrature requires refinement')
                residual=float(np.max(np.abs((kb-ka)-high[:,2:].sum(-1))))
                if residual>4e-11:raise ValueError('Signed finite KL transport mismatch')
                totals=high.sum(0);maximum=max(maximum,err,residual);largest_order=max(largest_order,order)
                record=dict(suffix=label,case=case['name'],moment=moment,steps=len(ka),vocabulary=a.shape[-1],
                    native_kl=float(ka.sum()),raised_kl=float(kb.sum()),delta_kl=float((kb-ka).sum()),
                    cross_increment=float(totals[2]),squared_increment=float(totals[3]),metric_change=float(totals[4]),
                    quadrature_order=order,maximum_absolute_error=err,maximum_signed_identity_residual=residual)
                records.append(record);print(label,case['name'],moment,record,flush=True)
    result=dict(status='passed',schema='finite-categorical-transport-v1',records=records,comparisons=len(records),
        step_comparisons=sum(r['steps'] for r in records),maximum_absolute_error=maximum,maximum_quadrature_order=largest_order,
        checked_sha256=bound,analyzer_sha256=sha256(__file__),seconds=time.monotonic()-start,
        scope='Fixed first probe, every paired moment arm and source suffix, full categorical vocabulary. Exact finite observation geometry checked by quadrature and endpoint KL, not causal mediation or autonomous training closure.')
    write_json(out,result)
    print('PASS',len(records),'paired comparisons; maximum error',maximum,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'potential-factorial-disjoint-20260914/analysis/finite-emission-transport.json');a=p.parse_args();main(a.output)
