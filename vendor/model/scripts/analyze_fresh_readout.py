"""All-cell reduction of the frozen fresh-size readout study."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json
from readout_budget import budget


def simplex(x):
    ordered=np.sort(x,axis=1)[:,::-1]
    theta=(ordered.cumsum(1)-1)/np.arange(1,x.shape[1]+1)
    rho=(ordered>theta).sum(1)-1
    return np.maximum(x-theta[np.arange(len(x)),rho,None],0)


def coordinates(x,cal):
    x=x.astype(float).reshape(len(x),-1);mean=x[cal].mean(0)
    centered=x-mean;_,s,v=np.linalg.svd(centered[cal],full_matrices=False)
    rank=min(8,int((s>s[0]*1e-10).sum()))
    return centered@v[:rank].T,mean,v[:rank]


def ridge(x,y,indices):
    xc=x[indices];yc=y[indices];mx=xc.mean(0);my=yc.mean(0)
    xc=xc-mx;yc=yc-my;gram=xc.T@xc
    scale=max(float(np.trace(gram))/x.shape[1],1e-30)
    w=np.linalg.solve(gram+.001*scale*np.eye(x.shape[1]),xc.T@yc)
    return w,my-mx@w


def vector(f):return np.sqrt(f.shape[1]/(f.shape[1]-1))*(f-1/f.shape[1])


def error_bootstrap(code,truth,cal,test,seed=218300):
    rng=np.random.default_rng(seed);values=[];diff=[]
    target=(vector(truth)**2).sum(1)
    # Eight configurations per chain, with disjoint sets of 16 chains.
    calidx=np.flatnonzero(cal).reshape(-1,8);testidx=np.flatnonzero(test).reshape(-1,8)
    for _ in range(1000):
        ci=calidx[rng.integers(len(calidx),size=len(calidx))].ravel()
        ti=testidx[rng.integers(len(testidx),size=len(testidx))].ravel()
        w,b=ridge(code,truth,ci);pred=simplex(code[ti]@w+b)
        report=budget(vector(truth[ti]),vector(pred))
        values.append([report['relative_rms'],report['connected_relative_rms'],report['absolute_relative_moment_error']])
        wd,bd=ridge(code,target[:,None],ci);direct=(code[ti]@wd+bd)[:,0]
        inv=(vector(pred)**2).sum(1)
        diff.append(float(np.mean((inv-target[ti])**2-(direct-target[ti])**2)))
    return dict(bootstrap95=np.quantile(values,[.025,.975],axis=0).tolist(),
        paired_mse_difference95=np.quantile(diff,[.025,.975]).tolist(),
        # 48 common-row model/class/size cells. Exploratory simultaneous band.
        paired_mse_difference_bonferroni=np.quantile(diff,[.025/48,1-.025/48]).tolist(),
        uncertainty='Calibration coefficients and assessment chains resampled; PCA conditioned on its frozen calibration acquisition.')


def gram_readout(z,depth,q):
    C=z[f'gram-coefficients-{depth}'];offset=z[f'gram-offset-{depth}']
    L=z[f'configurations-{depth}'].shape[-1];site=L*L-1
    t=(z[f'gram-traces-{depth}']-offset)/site
    U=np.linalg.qr(np.eye(q)[:,:q-1]-np.eye(q)[:,-1,None])[0]
    D=C@U;singular=np.linalg.svd(D,compute_uv=False)
    centered=t-C@np.full(q,1/q)
    recovered=np.full((len(t),q),1/q)+np.linalg.lstsq(D,centered.T,rcond=None)[0].T@U.T
    pred=simplex(recovered);prefix=z[f'prefix-fractions-{depth}']
    residual=t-prefix@C.T
    arithmetic=np.linalg.norm(residual,axis=1)/singular[-1]
    bound=np.sqrt(q/(q-1))*(np.sqrt(2)/L**2+arithmetic)
    errors=np.linalg.norm(vector(pred)-vector(z[f'fractions-{depth}']),axis=1)
    if np.any(errors>bound+2e-10):raise ValueError('Prefix/arithmetic vector bound')
    return pred,dict(sigma_min=float(singular[-1]),condition=float(singular[0]/singular[-1]),
        maximum_prefix_fraction_error=float(abs(pred-prefix).max()),
        maximum_arithmetic_error=float(arithmetic.max()),
        rms_vector_bound=float(np.sqrt(np.mean(bound**2))),
        maximum_vector_bound_residual=float(np.max(errors-bound)))


def main(study,output):
    study=Path(study);output=Path(output);output.mkdir(exist_ok=False)
    spec=json.loads((study/'protocol.json').read_text());rows=[];blocks=[];inputs={}
    inputs[str(study/'protocol.json')]=sha256(study/'protocol.json')
    for case in spec['cases']:
        folder=study/case['name'];mp=folder/'manifest.json';manifest=json.loads(mp.read_text())
        if manifest['status']!='complete' or manifest['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Incomplete observations')
        inputs[str(mp)]=sha256(mp);cache={}
        dest=output/case['name'];dest.mkdir()
        for c in spec['cells']:
            path=folder/f'q{c["q"]}-L{c["L"]}.npz'
            if sha256(path)!=manifest['files'][path.name]:raise ValueError('Observation identity')
            inputs[str(path)]=sha256(path)
            with np.load(path) as raw:z={k:raw[k] for k in raw.files}
            q=c['q'];chain=z['chain'];cal=chain<16;test=~cal
            cache[(q,c['L'])]=z;derived={}
            for depth in range(3):
                truth=z[f'fractions-{depth}'];target=(vector(truth)**2).sum(1)
                pred,details=gram_readout(z,depth,q)
                for family in ['gram','hidden','A_common','G_common']:
                    if family!='gram':
                        code,mean,basis=coordinates(z[f'{family}-{depth}'],cal)
                        w,b=ridge(code,truth,cal);pred=simplex(code@w+b)
                        wd,bd=ridge(code,target[:,None],cal);direct=(code@wd+bd)[:,0]
                        derived[f'{family}-code-{depth}']=code
                        derived[f'{family}-weights-{depth}']=w;derived[f'{family}-intercept-{depth}']=b
                        detail=dict(rank=len(basis))
                    else:
                        pred,detail=gram_readout(z,depth,q);direct=None
                    estimate=(vector(pred)**2).sum(1)
                    result=budget(vector(truth[test]),vector(pred[test]))
                    denominator=np.mean((target[test]-target[cal].mean())**2)
                    result.update(invariant_skill=float(1-np.mean((estimate[test]-target[test])**2)/denominator))
                    if direct is not None:
                        result['linear_skill']=float(1-np.mean((direct[test]-target[test])**2)/denominator)
                    if family=='A_common' and depth==0:result.update(error_bootstrap(code,truth,cal,test))
                    rows.append(dict(case=case['name'],heads=case['heads'],origin=case['origin'],q=q,L=c['L'],
                        depth=depth,family=family,mode='within-size' if family!='gram' else 'analytic',**detail,**result))
                    derived[f'{family}-predicted-fractions-{depth}']=pred
            # One-block and direct two-block maps from each retained common-row code.
            for family in ['A_common','G_common','hidden']:
                codes=[derived[f'{family}-code-{d}'] for d in range(3)]
                w01,b01=ridge(codes[0],codes[1],cal);w12,b12=ridge(codes[1],codes[2],cal)
                w02,b02=ridge(codes[0],codes[2],cal)
                via=(codes[0]@w01+b01)@w12+b12;direct=codes[0]@w02+b02
                norm=np.sqrt(np.mean(np.sum((codes[2][test]-codes[2][cal].mean(0))**2,axis=1)))
                blocks.append(dict(case=case['name'],q=q,L=c['L'],family=family,
                    direct_relative_rms=float(np.sqrt(np.mean(np.sum((direct[test]-codes[2][test])**2,axis=1)))/norm),
                    composition_relative_rms=float(np.sqrt(np.mean(np.sum((via[test]-direct[test])**2,axis=1)))/norm)))
            np.savez_compressed(dest/path.name,**derived)
        # True size transport: fit only the L=8,16 calibration chains.
        for q in [2,3]:
            for family in ['A_common','G_common','hidden']:
                zz=[cache[(q,L)] for L in [8,16,12,20]]
                xx=np.concatenate([z[f'{family}-0'] for z in zz])
                truth=np.concatenate([z['fractions-0'] for z in zz])
                cal=np.concatenate([z['chain']<16 if i<2 else np.zeros(256,dtype=bool) for i,z in enumerate(zz)])
                code,mean,basis=coordinates(xx,cal);w,b=ridge(code,truth,cal);pred=simplex(code@w+b)
                for i,L in enumerate([12,20],start=2):
                    test=np.zeros(len(xx),dtype=bool);test[256*i:256*(i+1)]=zz[i]['chain']>=16
                    result=budget(vector(truth[test]),vector(pred[test]))
                    if family=='A_common':result.update(error_bootstrap(code,truth,cal,test))
                    rows.append(dict(case=case['name'],heads=case['heads'],origin=case['origin'],q=q,L=L,
                        depth=0,family=family,mode='size-transfer',rank=len(basis),**result))
                np.savez_compressed(dest/f'transfer-q{q}-{family}.npz',predicted_fractions=pred,
                                    truth=truth,calibration=cal,code=code,weights=w,intercept=b)
        print('analyzed',case['name'],flush=True)
    report=dict(status='complete',schema='fresh-readout-analysis-v1',rows=rows,blocking=blocks,
        inputs=inputs,analyzer_sha256=sha256(__file__),budget_sha256=sha256(REPO/'scripts/readout_budget.py'),
        scientific_optimizer_updates=0,source_configurations=2048,source_chains=256,
        model_configuration_evaluations=36864,all_cells_retained=True)
    write_json(output/'analysis.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.study,a.output)
