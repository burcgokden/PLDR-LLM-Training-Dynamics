#!/usr/bin/env python3
"""Verify the signed-area completion of symmetric potential blocking."""
from companion_paths import legacy_path
import argparse,json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json
REPO=Path(__file__).resolve().parents[1]
ROOT=Path(legacy_path('/pldr-data/model'))


def reduce_path(b,p,width):
    steps=len(b)-1
    if steps%width:raise ValueError('Aligned complete blocks are required')
    delta_b=np.diff(b,axis=0);delta_p=np.diff(p,axis=0)
    blocks=steps//width;shape=(blocks,width)+b.shape[1:]
    db=delta_b.reshape(shape);dp=delta_p.reshape(shape)
    # Independent prefix-sum evaluation of all strictly ordered pairs.
    area=.5*np.sum(dp*(np.cumsum(db,axis=1)-db)-db*(np.cumsum(dp,axis=1)-dp),axis=1)
    fine_u=(delta_p*(b[1:]+b[:-1])/2).reshape(shape).sum(1)
    fine_v=(delta_b*(p[1:]+p[:-1])/2).reshape(shape).sum(1)
    coarse_u=(p[width::width]-p[:-width:width])*(b[width::width]+b[:-width:width])/2
    coarse_v=(b[width::width]-b[:-width:width])*(p[width::width]+p[:-width:width])/2
    d=p[width::width]*b[width::width]-p[:-width:width]*b[:-width:width]
    err=max(float(np.max(np.abs(fine_u-coarse_u-area))),float(np.max(np.abs(fine_v-coarse_v+area))))
    scale=max(1.,float(np.max(np.abs(fine_u))),float(np.max(np.abs(fine_v))))
    if err>3e-12*scale:raise ValueError('Signed-area identity failed')
    if width>1:
        half=width//2;a=reduce_arrays(db[:,:half],dp[:,:half]);c=reduce_arrays(db[:,half:],dp[:,half:])
        composed=a+c+.5*(dp[:,half:].sum(1)*db[:,:half].sum(1)-dp[:,:half].sum(1)*db[:,half:].sum(1))
        np.testing.assert_allclose(composed,area,atol=3e-16,rtol=3e-11)
    return dict(width=width,blocks=blocks,area_relative_increment_norm=float(np.linalg.norm(area)/np.linalg.norm(d)),
        identity_max_abs=err,coarse_base_fraction=float(np.sum(coarse_v**2)/(np.sum(coarse_u**2)+np.sum(coarse_v**2))),
        accumulated_base_fraction=float(np.sum(fine_v**2)/(np.sum(fine_u**2)+np.sum(fine_v**2))))


def reduce_arrays(db,dp):
    return .5*np.sum(dp*(np.cumsum(db,axis=1)-db)-db*(np.cumsum(dp,axis=1)-dp),axis=1)




