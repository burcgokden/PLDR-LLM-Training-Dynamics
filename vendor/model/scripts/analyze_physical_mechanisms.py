#!/usr/bin/env python
"""Held-out physical readouts, block-map closure and finite pulse linearity."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def coordinates(x,calibration,dimension=8):
    x=np.asarray(x,dtype=float).reshape(len(x),-1)
    mean=x[calibration].mean(0);center=x-mean
    _,s,v=np.linalg.svd(center[calibration],full_matrices=False)
    rank=min(dimension,int(np.sum(s>max(s[0],1e-300)*1e-10)))
    if rank<1:raise ValueError('No visible representation variance')
    return center@v[:rank].T,rank


def regression(x,y,calibration):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    mx=x[calibration].mean(0);my=y[calibration].mean(0)
    xc=x[calibration]-mx;yc=y[calibration]-my
    scale=max(float(np.trace(xc.T@xc))/x.shape[1],1e-30)
    weights=np.linalg.solve(xc.T@xc+1e-3*scale*np.eye(x.shape[1]),xc.T@yc)
    return weights,my-mx@weights


def apply(fit,x):return x@fit[0]+fit[1]


def relative_mse(predicted,truth,calibration,holdout):
    reference=truth[calibration].mean(0)
    numerator=np.mean((predicted[holdout]-truth[holdout])**2,axis=0)
    denominator=np.mean((truth[holdout]-reference)**2,axis=0)
    return numerator/np.maximum(denominator,1e-30)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    base=Path(a.study);study=base/'mechanisms';out=base/'mechanism-analysis';out.mkdir(exist_ok=False)
    spec=json.loads((study/'protocol.json').read_text());checked={str(study/'protocol.json'):sha256(study/'protocol.json')}
    readouts=[];closures=[];responses=[]
    for case in spec['cases']:
        parent=study/case['name'];m=json.loads((parent/'manifest.json').read_text())
        if m['status']!='complete':raise ValueError('Mechanism experiment incomplete')
        checked[str(parent/'manifest.json')]=sha256(parent/'manifest.json')
        for q in [2,3]:
            for L in [8,16]:
                path=parent/f'blocking-q{q}-L{L}.npz'
                if sha256(path)!=m['files'][path.name]:raise ValueError('Changed blocking pair')
                checked[str(path)]=sha256(path)
                z=np.load(path);cal=z['chain']<8;test=~cal
                targets=[z[f'targets-{d}'] for d in range(3)]
                for family in ['hidden','A_common','G_common']:
                    arrays=[z[f'{family}-{d}'] for d in range(3)]
                    if family=='hidden':arrays=[x[:,-1] for x in arrays]
                    codes=[];readout=[]
                    for depth,x in enumerate(arrays):
                        code,rank=coordinates(x,cal);codes.append(code)
                        fit=regression(code,targets[depth],cal);readout.append(fit)
                        mse=relative_mse(apply(fit,code),targets[depth],cal,test)
                        readouts.append(dict(case=case['name'],q=q,L=L,depth=depth,family=family,rank=rank,
                            physical_targets=['color_field','m2','energy_per_site'],relative_mse=mse.tolist(),
                            heldout_skill=(1-mse).tolist(),training_chains=8,heldout_chains=8))
                    block=[]
                    for depth in [0,1]:
                        fit=regression(codes[depth],codes[depth+1],cal);block.append(fit)
                        predicted=apply(fit,codes[depth])
                        numerator=np.sum((predicted[test]-codes[depth+1][test])**2)
                        denominator=np.sum((codes[depth+1][test]-codes[depth+1][cal].mean(0))**2)
                        physical=regression(targets[depth],targets[depth+1],cal)
                        route_internal=apply(readout[depth+1],predicted)
                        route_physical=apply(physical,apply(readout[depth],codes[depth]))
                        scale=np.mean((targets[depth+1][test]-targets[depth+1][cal].mean(0))**2,axis=0)
                        commutator=np.mean((route_internal[test]-route_physical[test])**2,axis=0)/np.maximum(scale,1e-30)
                        physical_closure=relative_mse(apply(physical,targets[depth]),targets[depth+1],cal,test)
                        closures.append(dict(case=case['name'],q=q,L=L,family=family,depth=depth,
                            internal_relative_mse=float(numerator/max(denominator,1e-30)),
                            physical_relative_mse=physical_closure.tolist(),
                            route_discrepancy_relative_mse=commutator.tolist()))
                    direct=regression(codes[0],codes[2],cal)
                    one=apply(direct,codes[0]);two=apply(block[1],apply(block[0],codes[0]))
                    denominator=np.sum((codes[2][test]-codes[2][cal].mean(0))**2)
                    closures.append(dict(case=case['name'],q=q,L=L,family=family,depth='two-step',
                        direct_relative_mse=float(np.sum((one[test]-codes[2][test])**2)/max(denominator,1e-30)),
                        composed_relative_mse=float(np.sum((two[test]-codes[2][test])**2)/max(denominator,1e-30)),
                        composition_discrepancy=float(np.sqrt(np.sum((two[test]-one[test])**2)/max(denominator,1e-30)))))
        path=parent/'response-logits.npz'
        if sha256(path)!=m['files'][path.name]:raise ValueError('Changed response logits')
        checked[str(path)]=sha256(path);z=np.load(path)
        def vector(branch,age):
            arrays=[z[f'b{branch}-t{age}-q{q}-L{L}'].astype(float) for q in [2,3] for L in [8,16]]
            return np.concatenate([(x-x.mean(1,keepdims=True)).ravel() for x in arrays])
        initial=None
        for age in spec['times']:
            baseline=vector(0,age)
            coarse=(vector(1,age)-vector(2,age))/.002
            fine=(vector(3,age)-vector(4,age))/.001
            norm=np.linalg.norm(fine)
            if initial is None:initial=norm
            error=np.linalg.norm(coarse-fine)/max(norm,1e-30)
            even=np.linalg.norm((vector(1,age)+vector(2,age))/2-baseline)/max(.001*norm,1e-30)
            responses.append(dict(case=case['name'],time=age,sensitivity_norm=float(norm),
                amplification=float(norm/max(initial,1e-30)),
                amplitude_halving_relative_difference=float(error),even_remainder_relative_to_linear=float(even)))
    write_json(out/'analysis.json',dict(status='complete',schema='physical-mechanism-analysis-v1',
        analyzer_sha256=sha256(__file__),checked_sha256=checked,readouts=readouts,closures=closures,responses=responses,
        method='First eight independent chains calibrate centering, up to eight PCA coordinates and ridge coefficients (penalty 0.001 times mean Gram eigenvalue); the other eight chains are held out. No hyperparameter is selected from held-out outcomes. Relative mean-square error uses the calibration-mean predictor as denominator.',
        scope='Finite physical visibility, empirical block-map closure, and matched optimizer directional response. These are not RG eigenvalues or native thermodynamic exponents.'))
    print('complete',len(readouts),'readouts',len(closures),'closure cells',len(responses),'response times',flush=True)


if __name__=='__main__':main()
