#!/usr/bin/env python3
"""Check the analytically derived noninteracting small-base tangent observation."""
from companion_paths import configured_path
from pathlib import Path
import numpy as np
from model_rg.provenance import write_json,sha256

def main():
    rng=np.random.default_rng(916001);z=rng.normal(size=1000000);c=.2
    cutoffs=np.array([10.,20.,40.,80.,160.]);tail=np.array([np.mean(2*c/np.abs(z)>u) for u in cutoffs]);pred=4*c/np.sqrt(2*np.pi)/cutoffs
    rows=[]
    for epsilon in [1e-9,1e-8,1e-6]:
        x=np.sqrt(2*epsilon);value=c*x/(epsilon+x*x/2);expected=c/np.sqrt(2*epsilon)
        if not np.isclose(value,expected,rtol=1e-15):raise ValueError('Wrong regularized tangent maximum')
        rows.append(dict(offset=epsilon,maximum=value,formula=expected))
    write_json(Path(configured_path('data:model/potential-avalanche-20260913/analysis/base-null.json')),dict(
        status='complete',producer_sha256=sha256(__file__),seed=916001,samples=len(z),
        law='iid standard normal preactivations; fixed c=0.2; tangent observation only',
        cutoffs=cutoffs.tolist(),empirical_survival=tail.tolist(),asymptotic_survival=pred.tolist(),maxima=rows,
        scope='Checks a reduced noninteracting observation law; no native critical exponent is estimated.'))
if __name__=='__main__':main()
