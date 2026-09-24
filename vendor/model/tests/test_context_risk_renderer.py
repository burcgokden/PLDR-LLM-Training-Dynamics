"""Strict finite admission of every displayed context-variance scalar."""
import copy
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pytest
from model_rg.provenance import sha256
from analyze_context_risk import context_metrics

@pytest.fixture
def record():
    rows=[]
    for n in [8,24]:
        for g in [0,1.5]:
            for k in [128,512,2048,8192,16384]:
                native=np.arange(1.,65.);residual=native*.04;m=context_metrics(native,residual,.25)
                rows.append(dict(heads=n,control=g,retained_tokens=k,contexts=64,
                    native_variance=native.tolist(),residual_variance=residual.tolist(),
                    native_variance_mean=float(native.mean()),residual_variance_mean=float(residual.mean()),
                    per_context_rms_min=min(m['per_context_rms']),per_context_rms_median=float(np.median(m['per_context_rms'])),
                    per_context_rms_max=max(m['per_context_rms']),**m))
    return dict(status='passed',schema='context-disaggregation-v1',resolution_cells=20,rows=rows,
        source_sha256=sha256(Path(__file__).resolve().parents[1]/'scripts/analyze_context_risk.py'),
        checked_sha256={},support_sha256={})


