"""Independently check all compact projection matrices and rendered cell coverage."""
import json
from pathlib import Path
import numpy as np


def verify(source):
    g=Path(source)/'generated'
    a=json.loads((g/'projection-analysis.json').read_text())
    ret=json.loads((g/'finetuning-return-analysis.json').read_text())
    raw=json.loads((g/'projection-verification.json').read_text())
    selected=[r for r in ret['rows'] if r['time']==256 and r['cohort']=='heldout' and r['calibration_cohort']=='calibration']
    expected={(f'h{h}-s{s}',n) for h in [4,8,14] for s in range(640101,640105) for n in [2,7]}
    if len(a['rows'])!=24 or {(r['case'],r['columns']) for r in a['rows']}!=expected:raise ValueError('Projection inventory differs')
    comparisons=0
    def equal(x,y):
        nonlocal comparisons
        if not np.allclose(x,y,rtol=3e-10,atol=3e-12):raise ValueError('Compact projection changed')
        comparisons+=1
    for row in a['rows']:
        s=next(x for x in selected if x['case']==row['case']);d=s if row['columns']==2 else s['expanded']
        gram=np.array(d['incoming_gram']);cross=np.array(d['cross_gram']);target=np.array(s['current_gram'])
        # Cholesky solves provide a separate computation of the projection coefficients.
        root=np.linalg.cholesky(gram);oracle=np.linalg.solve(root.T,np.linalg.solve(root,cross))
        coefficient=np.array(d['matrix']);residual=target-cross.T@oracle
        excess=(coefficient-oracle).T@gram@(coefficient-oracle)
        equal(oracle,row['oracle_matrix']);equal(residual,row['oracle_residual_gram']);equal(excess,row['coefficient_excess_gram'])
        equal(residual+excess,d['residual_gram'])
        equal(np.sqrt(np.trace(residual)/np.trace(target)),row['oracle_relative_residual'])
        equal(np.trace(residual)/np.trace(np.array(d['residual_gram'])),row['orthogonal_fraction'])
        equal(d['relative_residual'],row['calibrated_relative_residual'])
    text=(g/'projection-complete.tex').read_text()
    for case in {r[0] for r in expected}:
        if text.count(case+' & ')!=2:raise ValueError('A selected projection cell is omitted from the paper')
    if raw['status']!='passed' or raw['cases']!=12 or len(raw['outcomes'])!=24 or len(raw['checks'])!=312:
        raise ValueError('Raw projection check scope differs')
    return dict(status='passed',dictionary_cells=24,compact_comparisons=comparisons,
                raw_comparisons=312,raw_cases=12,oracle_is_forecast=False)
