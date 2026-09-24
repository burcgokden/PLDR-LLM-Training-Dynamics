"""Independent empirical magnetic error budgets for fixed native readouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def budget(v, vh):
    error = vh - v
    mu = np.mean(np.sum(v*v, axis=1))
    delta2 = np.mean(np.sum(error*error, axis=1))
    cross = 2*np.mean(np.sum(v*error, axis=1))
    estimate = np.mean(np.sum(vh*vh, axis=1))
    delta = np.sqrt(delta2)
    sharp = 2*np.sqrt(mu)*delta+delta2
    old = 2*delta
    vc = v-v.mean(axis=0)
    ec = error-error.mean(axis=0)
    variance = np.mean(np.sum(vc*vc, axis=1))
    centered_delta2 = np.mean(np.sum(ec*ec, axis=1))
    estimated_variance = np.mean(np.sum((vh-vh.mean(axis=0))**2, axis=1))
    connected_bound = 2*np.sqrt(variance*centered_delta2)+centered_delta2
    identity_error = abs(estimate-mu-cross-delta2)
    checks = [identity_error < 2e-12,
              abs(estimate-mu) <= sharp+2e-12,
              abs(estimate-mu) <= old+2e-12,
              abs(estimated_variance-variance) <= connected_bound+2e-12]
    if not all(checks):
        raise ValueError('Moment/centered-variance identity or bound failed')
    return dict(connected_relative_rms=np.sqrt(centered_delta2/variance) if variance>0 else None,mu=mu,estimated_mu=estimate,delta=delta,relative_rms=delta/np.sqrt(mu),
                signed_cross=cross,squared_error=delta2,identity_error=identity_error,
                absolute_relative_moment_error=abs(estimate/mu-1),
                sharp_relative_bound=sharp/mu,unit_relative_bound=old/mu,
                sharp_to_unit_bound=sharp/old if old else 1.,
                combined_relative_bound=min(sharp,old)/mu,
                connected_variance=variance,estimated_connected_variance=estimated_variance,
                connected_delta=np.sqrt(centered_delta2),connected_bound=connected_bound,
                mean_sector=float(np.sum(v.mean(axis=0)**2)))



def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();BASE=Path(args.study);OUT=Path(args.output)
    
    
    rows=[]
    inputs={}
    for origin in ['adapted','random','pretrained']:
        for heads in [4,8]:
            folder=(BASE/f'collective-validation/h{heads}-s640101-full' if origin=='adapted'
                    else BASE/f'collective-baseline/h{heads}-{origin}')
            verification=folder/'verification.json'
            inputs[str(verification)]=hashlib.sha256(verification.read_bytes()).hexdigest()
            spec=json.loads(verification.read_text())
            for filename in spec['files']:
                path=folder/filename
                inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                if isinstance(spec['files'],dict) and inputs[str(path)]!=spec['files'][filename]:
                    raise ValueError('Saved native array hash differs from its verification record')
                with np.load(path) as z:
                    chain=z['chain'];test=chain>=8
                    chain_ids=np.unique(chain[test])
                    if len(chain_ids)!=8:
                        raise ValueError('Expected eight held-out chains')
                    L=z['configurations-0'].shape[-1]
                    for depth in [0,1]:
                        f=z[f'fractions-{depth}'][test]
                        q=f.shape[1];v=np.sqrt(q/(q-1))*(f-1/q)
                        for family in ['hidden','A_common','G_common']:
                            fh=z[f'{family}-predicted-fractions-{depth}'][test]
                            if np.min(fh)<-1e-9 or not np.allclose(fh.sum(axis=1),1,atol=1e-9):
                                raise ValueError('Readout is not in the simplex')
                            vh=np.sqrt(q/(q-1))*(fh-1/q)
                            result=budget(v,vh)
                            result.update(origin=origin,heads=heads,L=L,q=q,depth=depth,family=family,
                                          source=str(path),test_configurations=int(test.sum()),test_chains=8)
                            if depth==0 and family=='A_common':
                                rng=np.random.default_rng(9182601)
                                ids=[np.flatnonzero(chain[test]==c) for c in chain_ids]
                                samples=[]
                                for _ in range(2000):
                                    idx=np.concatenate([ids[i] for i in rng.integers(0,8,size=8)])
                                    b=budget(v[idx],vh[idx])
                                    samples.append([b['relative_rms'],b['absolute_relative_moment_error']])
                                interval=np.quantile(samples,[.025,.975],axis=0)
                                result['chain_bootstrap95_relative_rms']=interval[:,0].tolist()
                                result['chain_bootstrap95_absolute_relative_moment_error']=interval[:,1].tolist()
                            rows.append(result)
    
    primary=[r for r in rows if r['origin']=='adapted' and r['depth']==0 and r['family']=='A_common']
    summary=dict(status='passed',native_updates=0,new_native_observations=0,
        method='Independent float64 reduction of saved native predictions; fixed calibration fit; chain-level descriptive bootstrap.',
        rows=rows,inputs=inputs,primary_common_rows=primary,
        counts=dict(cells=len(rows),bounds_checked=3*len(rows),identities_checked=len(rows),input_arrays=len(inputs)//2),
        maximum_identity_error=max(r['identity_error'] for r in rows),
        bound_improved_cells=int(sum(r['sharp_to_unit_bound']<1 for r in rows)),
        relative_rms_range=[min(r['relative_rms'] for r in rows),max(r['relative_rms'] for r in rows)],
        sharp_to_unit_bound_range=[min(r['sharp_to_unit_bound'] for r in rows),max(r['sharp_to_unit_bound'] for r in rows)])
    summary['counts']['input_arrays']=sum(p.endswith('.npz') for p in inputs)
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'readout-budget.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','inputs']},indent=2))


if __name__=='__main__':main()
