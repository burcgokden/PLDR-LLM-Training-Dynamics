#!/usr/bin/env python
"""Reduce every prespecified native pulse, retaining signed temporal covariance."""
import argparse
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json


def covariance(x, y=None):
    x=np.asarray(x,dtype=np.float64);y=x if y is None else np.asarray(y,dtype=np.float64)
    return (x-x.mean(0)).T@(y-y.mean(0))/(len(x)-1) if len(x)>1 else None


def analyze(study):
    study=Path(study);spec=json.loads((study/'protocol.json').read_text())
    status=json.loads((study/'run-status.json').read_text());assert status['status']=='complete'
    records=[];transport=[];chronological=[];curves=[];inputs={}
    for case in spec['cases']:
        folder=study/'runs'/case['name'];arrays={}
        for arm in spec['arms']:
            x=[]
            for s in range(spec['source_replicates']):
                file=folder/f"source{s}-{arm['name']}.npz"
                with np.load(file) as f:x.append(f['q'])
                inputs[str(file)]=sha256(file)
            arrays[arm['name']]=np.array(x)
        for family in ['generator','body']:
            full=(arrays[family+'-plus']-arrays[family+'-minus'])/2
            half=(arrays[family+'-halfplus']-arrays[family+'-halfminus'])/2
            for s in range(spec['source_replicates']):
                signal=float(np.sqrt(np.mean(half[s]**2)))
                discrepancy=float(np.linalg.norm(full[s]-2*half[s])/max(np.linalg.norm(2*half[s]),1e-30))
                gate=spec['gate']; resolved=signal>gate['signal_multiple']*gate['absolute_floor']
                records.append(dict(case=case['name'],heads=case['heads'],family=family,source=s,
                    discrepancy=discrepancy,half_signal_rms=signal,resolved=bool(resolved),
                    passes=bool(resolved and discrepancy<=gate['relative_tolerance'])))
                for k,t in enumerate(spec['times']):
                    small=float(np.sqrt(np.mean(half[s,k]**2)))
                    relative=float(np.linalg.norm(full[s,k]-2*half[s,k])/max(np.linalg.norm(2*half[s,k]),1e-30))
                    curves.append(dict(case=case['name'],family=family,source=s,time=t,
                        half_signal_rms=small,discrepancy=relative,
                        full_secant=(full[s,k]/spec['amplitude']).tolist(),
                        half_secant=(2*half[s,k]/spec['amplitude']).tolist()))
        if spec['source_replicates']>1:
            x=arrays['zero'][:,-1]
            for arm in spec['arms'][1:]:
                y=arrays[arm['name']][:,-1];e=y-x
                c0=covariance(x);cy=covariance(y);ce=covariance(e);cross=covariance(x,e)
                delta=cy-c0;rhs=cross+cross.T+ce
                bound=2*np.sqrt(np.trace(c0)*np.trace(ce))+np.trace(ce)
                transport.append(dict(case=case['name'],arm=arm['name'],
                    identity_max_error=float(np.max(np.abs(delta-rhs))),
                    covariance_change_trace=float(np.trace(delta)),
                    signed_cross_trace=float(2*np.trace(cross)),error_trace=float(np.trace(ce)),
                    operator_change=float(np.linalg.norm(delta,2)),bound=float(bound)))
            for arm in spec['arms']:
                x=arrays[arm['name']][:,:,0]
                increments=np.diff(x,axis=1);c=covariance(increments)
                # Initial level is deterministic across source replicates at a fixed pulse.
                assert np.array_equal(x[:,0],np.repeat(x[0,0],len(x)))
                endpoint=float(np.var(x[:,-1],ddof=1));diagonal=float(np.trace(c));complete=float(c.sum())
                chronological.append(dict(case=case['name'],arm=arm['name'],endpoint_variance=endpoint,
                    complete_variance=complete,diagonal_variance=diagonal,
                    signed_offdiagonal=complete-diagonal,absolute_error=abs(complete-endpoint),
                    diagonal_ratio=diagonal/max(endpoint,1e-30)))
    summary=[]
    for case in spec['cases']:
        row=dict(case=case['name'],heads=case['heads'])
        for family in ['generator','body']:
            r=[v for v in records if v['case']==case['name'] and v['family']==family]
            row[family]=dict(passed=sum(v['passes'] for v in r),total=len(r),
                discrepancy_min=min(v['discrepancy'] for v in r),discrepancy_max=max(v['discrepancy'] for v in r),
                discrepancy_median=float(np.median([v['discrepancy'] for v in r])))
        summary.append(row)
    return dict(schema='onepass-directional-results-v1',status='complete',kind=spec['kind'],
        protocol_sha256=sha256(study/'protocol.json'),analyzer_sha256=sha256(__file__),
        primary=records,summary=summary,time_resolved=curves,signed_transport=transport,
        chronological=chronological,inputs_sha256=inputs,
        statistical_unit='Source sequences condition on complete incoming states. Signed arms share sources; the two corpora and four nested initialization identities are inherited. No independent-model confidence interval or critical exponent is estimated.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args()
    output=Path(a.study)/'analysis.json'
    if output.exists():raise FileExistsError(output)
    result=analyze(a.study);write_json(output,result)
    print('All',len(result['primary']),'primary cells retained;',sum(v['passes'] for v in result['primary']),'pass the fixed response domain',flush=True)

if __name__=='__main__':main()
