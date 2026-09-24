"""Independent raw-array reconstruction of fresh readout claims and fit scope."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def close(a,b,name):
    if not np.allclose(a,b,rtol=2e-8,atol=2e-10):raise ValueError(name)


def project(x):
    s=np.sort(x,axis=-1)[:,::-1];t=(np.cumsum(s,axis=-1)-1)/np.arange(1,x.shape[1]+1)
    k=np.sum(s>t,axis=-1)-1
    return np.clip(x-t[np.arange(len(x)),k,None],0,None)


def check_fit(raw,code,w,b,pred,target,cal):
    X=raw.reshape(len(raw),-1).astype(float);center=X[cal].mean(0);xc=X[cal]-center
    cc=code[cal];close(cc.mean(0),0,'PCA calibration mean')
    # Recover the principal subspace by SVD, avoiding division by tiny
    # squared singular values in a normal-equation reconstruction.
    _,singular,right=np.linalg.svd(xc,full_matrices=False)
    native=(X-center)@right[:code.shape[1]].T
    left,_,rotation=np.linalg.svd(native[cal].T@cc,full_matrices=False)
    aligned=native@(left@rotation)
    close(aligned,code,'Out-of-sample PCA transform')
    close(np.linalg.svd(cc,compute_uv=False),singular[:code.shape[1]],'Leading calibration singular values')
    y=target[cal];yc=y-y.mean(0);gram=cc.T@cc
    penalty=.001*max(np.trace(gram)/code.shape[1],1e-30)
    close((gram+penalty*np.eye(code.shape[1]))@w,cc.T@yc,'Calibration-only ridge equation')
    close(b,y.mean(0)-cc.mean(0)@w,'Ridge intercept')
    close(project(code@w+b),pred,'Readout prediction')


def check_budget(truth,pred,row):
    q=truth.shape[1];v=np.sqrt(q/(q-1))*(truth-1/q);vh=np.sqrt(q/(q-1))*(pred-1/q)
    e=vh-v;mu=np.square(v).sum(1).mean();d2=np.square(e).sum(1).mean()
    estimate=np.square(vh).sum(1).mean();cross=2*(v*e).sum(1).mean()
    ec=e-e.mean(0);vc=v-v.mean(0);dc2=np.square(ec).sum(1).mean();kappa=np.square(vc).sum(1).mean()
    kappa_h=np.square(vh-vh.mean(0)).sum(1).mean()
    close(estimate-mu,cross+d2,'Signed second-moment identity')
    close([row['relative_rms'],row['connected_relative_rms'],row['absolute_relative_moment_error']],
          [np.sqrt(d2/mu),np.sqrt(dc2/kappa),abs(estimate/mu-1)],'Displayed readout budgets')
    if finite_greater(abs(estimate-mu), min(2*np.sqrt(d2),2*np.sqrt(mu*d2)+d2)+2e-12, 'scripts/verify_fresh_readout.py:48'):raise ValueError('Second-moment inequality')
    if finite_greater(abs(kappa_h-kappa), 2*np.sqrt(kappa*dc2)+dc2+2e-12, 'scripts/verify_fresh_readout.py:49'):raise ValueError('Connected inequality')


def main(study,analysis,output):
    study=Path(study);analysis=Path(analysis);output=Path(output)
    if output.exists():raise FileExistsError(output)
    spec=load_json_strict((study/'protocol.json').read_text());report=load_json_strict((analysis/'analysis.json').read_text())
    checked={};count=0;fits=0;gram_checks=0;source_rows=[]
    def bind(p,h=None):
        p=Path(p);digest=sha256(p)
        if h is not None and h!=digest:raise ValueError('Input identity: '+str(p))
        checked[str(p)]=digest
    bind(analysis/'analysis.json')
    for p,h in report['inputs'].items():bind(p,h)
    for n,h in spec['sources'].items():bind(REPO/n,h);bind(study/'executed-source'/n,h)
    for p,h in spec['native_assets'].items():bind(p,h)
    for c in spec['cells']:
        bind(c['path'],c['sha256']);x=np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape']))
        fractions=np.stack([(x==a).mean((2,3)) for a in range(c['q'])],-1)
        m2=c['q']/(c['q']-1)*np.square(fractions-1/c['q']).sum(-1)
        means=m2.mean(1);ordered=means[::2];random=means[1::2]
        se=np.sqrt(ordered.var(ddof=1)/len(ordered)+random.var(ddof=1)/len(random))
        centered=m2-m2.mean(1,keepdims=True)
        lag=np.mean(centered[:,:-1]*centered[:,1:])/max(np.mean(centered**2),1e-30)
        source_rows.append(dict(q=c['q'],L=c['L'],m2=float(m2.mean()),
            ordered_random_start_standardized_difference=float((ordered.mean()-random.mean())/se),
            within_chain_lag1=float(lag)))
    rows={(r['case'],r['q'],r['L'],r['depth'],r['family'],r['mode']):r for r in report['rows']}
    if len(rows)!=648:raise ValueError('Incomplete all-cell inventory')
    for case in spec['cases']:
        cache={}
        for c in spec['cells']:
            q,L=c['q'],c['L'];filename=f'q{q}-L{L}.npz'
            with np.load(study/case['name']/filename) as f:z={k:f[k] for k in f.files}
            with np.load(analysis/case['name']/filename) as f:a={k:f[k] for k in f.files}
            bind(analysis/case['name']/filename);cache[(q,L)]=z
            if not np.array_equal(z['chain'],np.repeat(np.arange(32),8)):raise ValueError('Chain roles')
            cal=z['chain']<16;test=~cal
            for depth in range(3):
                x=z[f'configurations-{depth}'];full=np.stack([(x==color).mean((1,2)) for color in range(q)],1)
                close(full,z[f'fractions-{depth}'],'Source fractions')
                for family in ['gram','hidden','A_common','G_common']:
                    pred=a[f'{family}-predicted-fractions-{depth}']
                    mode='analytic' if family=='gram' else 'within-size'
                    row=rows[(case['name'],q,L,depth,family,mode)]
                    check_budget(full[test],pred[test],row);count+=1
                    if family=='gram':
                        j=x.shape[-1]**2-1;flat=x.reshape(len(x),-1)
                        prefix=np.stack([(flat[:,:j]==color).mean(1) for color in range(q)],1)
                        C=z[f'gram-coefficients-{depth}'];offset=z[f'gram-offset-{depth}']
                        t=(z[f'gram-traces-{depth}']-offset)/j
                        delta=t-prefix@C.T;U=np.linalg.qr(np.eye(q)[:,:q-1]-np.eye(q)[:,-1,None])[0]
                        sigma=np.linalg.svd(C@U,compute_uv=False)[-1]
                        recovered=np.full_like(prefix,1/q)+np.linalg.lstsq(C@U,(t-C@np.full(q,1/q)).T,rcond=None)[0].T@U.T
                        close(project(recovered),pred,'Analytic Gram readout')
                        err=np.sqrt(q/(q-1))*np.linalg.norm(pred-full,axis=1)
                        bound=np.sqrt(q/(q-1))*(np.sqrt(2)/(j+1)+np.linalg.norm(delta,axis=1)/sigma)
                        if np.any(finite_greater(err, bound+2e-10, 'scripts/verify_fresh_readout.py:106')):raise ValueError('Omitted-site theorem')
                        gram_checks+=len(x)
                    else:
                        check_fit(z[f'{family}-{depth}'],a[f'{family}-code-{depth}'],
                            a[f'{family}-weights-{depth}'],a[f'{family}-intercept-{depth}'],pred,full,cal);fits+=1
            for family in ['A_common','G_common','hidden']:
                codes=[a[f'{family}-code-{d}'] for d in range(3)]
                def learned(x,y):
                    mx=x[cal].mean(0);my=y[cal].mean(0);xc=x[cal]-mx;yc=y[cal]-my
                    g=xc.T@xc;reg=.001*max(float(np.trace(g))/x.shape[1],1e-30)
                    w=np.linalg.solve(g+reg*np.eye(x.shape[1]),xc.T@yc)
                    return w,my-mx@w
                w01,b01=learned(codes[0],codes[1]);w12,b12=learned(codes[1],codes[2]);w02,b02=learned(codes[0],codes[2])
                via=(codes[0]@w01+b01)@w12+b12;direct=codes[0]@w02+b02
                denom=np.sqrt(np.mean(np.sum((codes[2][test]-codes[2][cal].mean(0))**2,axis=1)))
                block=next(r for r in report['blocking'] if r['case']==case['name'] and r['q']==q and r['L']==L and r['family']==family)
                close([block['direct_relative_rms'],block['composition_relative_rms']],
                      [np.sqrt(np.mean(np.sum((direct[test]-codes[2][test])**2,axis=1)))/denom,
                       np.sqrt(np.mean(np.sum((via[test]-direct[test])**2,axis=1)))/denom],'Block-map reconstruction')
        for q in [2,3]:
            for family in ['hidden','A_common','G_common']:
                zz=[cache[(q,L)] for L in [8,16,12,20]]
                p=analysis/case['name']/f'transfer-q{q}-{family}.npz';bind(p)
                with np.load(p) as f:a={k:f[k] for k in f.files}
                target=np.concatenate([z['fractions-0'] for z in zz]);close(target,a['truth'],'Transfer targets')
                cal=np.concatenate([z['chain']<16 if i<2 else np.zeros(256,dtype=bool) for i,z in enumerate(zz)])
                close(cal,a['calibration'],'Transfer calibration masks')
                check_fit(np.concatenate([z[f'{family}-0'] for z in zz]),a['code'],a['weights'],
                          a['intercept'],a['predicted_fractions'],target,cal);fits+=1
                for i,L in enumerate([12,20],start=2):
                    idx=np.arange(256*i+128,256*(i+1))
                    check_budget(target[idx],a['predicted_fractions'][idx],rows[(case['name'],q,L,0,family,'size-transfer')]);count+=1
    write_json(output,dict(status='passed',schema='fresh-readout-independent-v1',cells=count,
        calibration_only_fits=fits,pointwise_prefix_bounds=gram_checks,block_map_cells=len(report['blocking']),source_checks=source_rows,
        verifier_sha256=sha256(__file__),checked_sha256=checked,native_updates=0,
        scope='Raw moment, Gram, simplex, PCA and ridge reconstruction. Bootstrap intervals retain the stated conditional fit scope; no infinite-size limit is verified.'))
    print('passed',count,'readout cells,',fits,'calibration-only fits,',gram_checks,'prefix bounds',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.study,a.analysis,a.output)
