#!/usr/bin/env python
"""Describe both finite pulse sectors and independently check their covariance sum.

This is an algebraic decomposition of all recorded arms, not a new fitted or
selected confirmation endpoint. The amplitude-halving endpoint remains primary.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    study=Path(a.study).resolve();output=study/'parity-analysis.json'
    if output.exists():raise FileExistsError(output)
    spec=json.loads((study/'protocol.json').read_text());checked=json.loads((study/'verification.json').read_text())
    if checked.get('status') != 'passed' or spec.get('kind') not in ['confirmation', 'qualification']:
        raise ValueError('Passed native qualification or confirmation required')
    rows=[];terms=[];inputs={};max_error=0.
    def trace_cov(x,y):
        x=x-x.mean(0);y=y-y.mean(0)
        return float(sum(float(np.dot(u,v)) for u,v in zip(x,y))/(len(x)-1))
    for case in spec['cases']:
        arrays={}
        for arm in spec['arms']:
            values=[]
            for source in range(spec['source_replicates']):
                path=study/'runs'/case['name']/f"source{source}-{arm['name']}.npz"
                with np.load(path) as f:values.append(f['q'])
                inputs[str(path)]=sha256(path)
                if inputs[str(path)] != checked['checked_sha256'][str(path)]:
                    raise ValueError('Raw pulse archive changed')
            arrays[arm['name']]=np.array(values)
        x=arrays['zero']
        for family in ['generator','body']:
            for radius,prefix in [('full',''),('half','half')]:
                plus=arrays[family+'-'+prefix+'plus'];minus=arrays[family+'-'+prefix+'minus']
                odd=(plus-minus)/2;even=(plus+minus)/2-x
                assert np.max(np.abs(plus-(x+even+odd)))<1e-12
                assert np.max(np.abs(minus-(x+even-odd)))<1e-12
                for source in range(spec['source_replicates']):
                    a=float(np.sqrt(np.mean(odd[source]**2)));b=float(np.sqrt(np.mean(even[source]**2)))
                    rows.append(dict(case=case['name'],family=family,radius=radius,source=source,
                        odd_path_rms=a,even_path_rms=b,even_over_odd=b/max(a,1e-30)))
                xx,ee,oo=x[:,-1],even[:,-1],odd[:,-1]
                common=trace_cov(xx,xx)+trace_cov(ee,ee)+trace_cov(oo,oo)+2*trace_cov(xx,ee)
                signed=2*trace_cov(xx,oo)+2*trace_cov(ee,oo)
                for sign,y in [(1,plus),(-1,minus)]:
                    actual=trace_cov(y[:,-1],y[:,-1]);reconstructed=common+sign*signed
                    err=abs(actual-reconstructed);assert err<1e-12;max_error=max(max_error,err)
                    terms.append(dict(case=case['name'],family=family,radius=radius,sign=sign,
                        base=trace_cov(xx,xx),even=trace_cov(ee,ee),odd=trace_cov(oo,oo),
                        base_even=2*trace_cov(xx,ee),base_odd=sign*2*trace_cov(xx,oo),
                        even_odd=sign*2*trace_cov(ee,oo),actual=actual,reconstructed=reconstructed,error=err))
    summary=[]
    for case in spec['cases']:
        r=dict(case=case['name'])
        for family in ['generator','body']:
            values=[v['even_over_odd'] for v in rows if v['case']==case['name'] and v['family']==family and v['radius']=='half']
            r[family]=dict(min=min(values),median=float(np.median(values)),max=max(values))
        summary.append(r)
    write_json(output,dict(status='passed',schema='finite-pulse-parity-v1',analyzer_sha256=sha256(__file__),
        protocol_sha256=sha256(study/'protocol.json'),native_verification_sha256=sha256(study/'verification.json'),
        rows=rows,covariance_terms=terms,summary=summary,max_identity_error=max_error,inputs_sha256=inputs,
        input_binding='All native archives are rehashed against the passed independent verifier.',
        scope='Descriptive exact decomposition of all native finite-amplitude arms. No additional fitted predictor, primary hypothesis, derivative certificate or independent model sample.'))
    print('Retained',len(rows),'parity path decompositions and',len(terms),'signed covariance sums',flush=True)

if __name__=='__main__':main()
