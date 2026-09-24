"""Independent all-state vocabulary-space check of fixed-dictionary projection.

All twelve terminal states, both dictionaries and every selected comparison.
No model/producer/analyzer imports and no native training updates.
"""
from pathlib import Path
import argparse, hashlib, json, time
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
parser.add_argument('--diagnostic',required=True)
parser.add_argument('--output',required=True)
config=parser.parse_args()
BASE=Path(config.base).resolve(); OUT=Path(config.output).resolve()
if OUT.exists():raise FileExistsError('Fresh reconstruction output required')
OUT.parent.mkdir(parents=True,exist_ok=True)
diagnostic=json.loads(Path(config.diagnostic).read_text())
primary=json.loads((BASE/'assessment/analysis.json').read_text())
returned=json.loads((BASE/'return-assessment/analysis.json').read_text())
with np.load(BASE/'data/evaluation.npz',allow_pickle=False) as f:
    split=f['split'].copy(); targets=f['crops'][:,64].copy()
cal=np.flatnonzero(split==0); test=np.flatnonzero(split==1)
files={}; checks=[]; outcomes=[]; started=time.time()
def read(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    files[str(path)]=h.hexdigest()
    with np.load(path,allow_pickle=False) as f:
        ids=f['indices']
        if not np.array_equal(ids,np.arange(480)):raise ValueError('Noncanonical terminal panel')
        return f['logits64'].astype(np.float64)
def compare(name,actual,expected):
    actual=np.asarray(actual); expected=np.asarray(expected)
    err=float(np.max(np.abs(actual-expected)))
    if not np.allclose(actual,expected,rtol=3e-10,atol=3e-12):raise ValueError((name,err))
    checks.append(dict(name=name,max_absolute_error=err))
def weighted_columns(arrays,prob,ids):
    columns=[]
    for arr in arrays:
        z=arr[ids].copy(); p=prob[ids]
        z-=np.sum(p*z,axis=1,keepdims=True)
        columns.append((z*np.sqrt(p/len(ids))).ravel())
    return np.column_stack(columns)
names=['mix0000','mix1000','mix0125','mix0250','mix0500','mix0750','mix0875']
for case in [f'h{h}-s{s}' for h in [4,8,14] for s in range(640101,640105)]:
    print('START',case,flush=True)
    folder=BASE/'assessment'/case
    z0=read(folder/'initial-observation.npz'); z0-=z0.max(1,keepdims=True)
    p=np.exp(z0); p/=p.sum(1,keepdims=True)
    general=read(folder/'general/observation-0512.npz')
    contrasts=[read(folder/n/'observation-0512.npz')-general for n in names]
    dictionary_test=weighted_columns(contrasts,p,test)
    gram=dictionary_test[:,:2].T@dictionary_test[:,:2]
    record=next(r for r in primary['responses'] if r['case']==case and r['time']==512 and r['cohort']=='heldout')
    compare(case+'/source_gram',gram,record['predictive']['gram'])
    eig=np.linalg.eigvalsh(gram)[::-1]
    compare(case+'/source_eigenvalues',eig,record['predictive']['eigenvalues'])
    for rho in [.125,.25,.75,.875]:
        key=f'mix{int(rho*1000):04d}'; actual=contrasts[names.index(key)]
        affine=(1-rho)*contrasts[0]+rho*contrasts[1]
        quadratic=affine+4*rho*(1-rho)*(contrasts[4]-.5*(contrasts[0]+contrasts[1]))
        residuals=weighted_columns([actual,actual-affine,actual-quadratic],p,test)
        norms=np.sqrt(np.sum(residuals*residuals,axis=0))
        row=next(r for r in primary['mixture_forecasts'] if r['case']==case and r['time']==512 and r['cohort']=='heldout' and r['rho']==rho)
        compare(case+f'/mixture_{rho}',norms,[row['signal_rms'],row['affine_rms'],row['quadratic_rms']])
        del residuals
    dictionary_cal=weighted_columns(contrasts,p,cal)
    rfolder=BASE/'return-assessment'/case
    rgeneral=read(rfolder/'general/observation-0256.npz')
    rcontrasts=[read(rfolder/n/'observation-0256.npz')-rgeneral for n in names[:2]]
    target_cal=weighted_columns(rcontrasts,p,cal)
    target_test=weighted_columns(rcontrasts,p,test)
    row=next(r for r in returned['rows'] if r['case']==case and r['time']==256 and r['cohort']=='heldout' and r['calibration_cohort']=='calibration')
    for width in [2,7]:
        a=dictionary_cal[:,:width]; b=dictionary_test[:,:width]
        # Only calibration documents enter this fit.
        matrix=np.linalg.solve(a.T@a,a.T@target_cal)
        residual=target_test-b@matrix
        direct=residual.T@residual
        target=row if width==2 else row['expanded']
        compare(case+f'/return{width}_matrix',matrix,target['matrix'])
        compare(case+f'/return{width}_residual',direct,target['residual_gram'])
        rel=float(np.linalg.norm(residual)/np.linalg.norm(target_test))
        compare(case+f'/return{width}_relative',rel,target['relative_residual'])
        oracle=np.linalg.solve(b.T@b,b.T@target_test)
        orthogonal=target_test-b@oracle
        excess=b@(oracle-matrix)
        expected=next(x for x in diagnostic['rows'] if x['case']==case and x['columns']==width)
        compare(case+f'/oracle{width}_relative',np.linalg.norm(orthogonal)/np.linalg.norm(target_test),expected['oracle_relative_residual'])
        compare(case+f'/oracle{width}_fraction',np.sum(orthogonal**2)/np.sum(residual**2),expected['orthogonal_fraction'])
        compare(case+f'/oracle{width}_split',direct,orthogonal.T@orthogonal+excess.T@excess)
        compare(case+f'/oracle{width}_matrix',oracle,expected['oracle_matrix'])
        compare(case+f'/oracle{width}_gram',orthogonal.T@orthogonal,expected['oracle_residual_gram'])
        compare(case+f'/excess{width}_gram',excess.T@excess,expected['coefficient_excess_gram'])
        del orthogonal,excess

        outcomes.append(dict(case=case,columns=width,relative_residual=rel,
                             weak_incoming_eigenvalue=float(eig[-1])))
    # Direct proper-prefix risk, with the target outside the supplied 64 tokens.
    for arm in ['general','mix0500']:
        z=general if arm=='general' else general+contrasts[4]
        shift=z[test]-z[test].max(1,keepdims=True)
        risk=float(np.mean(np.log(np.exp(shift).sum(1))-shift[np.arange(len(test)),targets[test]]))
        row=next(r for r in primary['outcomes'] if r['case']==case and r['time']==512 and r['cohort']=='heldout' and r['arm']==arm)
        compare(case+'/'+arm+'_risk',risk,row['risk64'])
    del dictionary_cal,dictionary_test,target_cal,target_test,contrasts,rcontrasts,residual,a,b
    print('END',case,flush=True)
for path in [Path(config.diagnostic),BASE/'assessment/analysis.json',BASE/'return-assessment/analysis.json',BASE/'data/evaluation.npz']:
    files[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
OUT.write_text(json.dumps(dict(status='passed',seconds=time.time()-started,
    verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_updates=0,cases=12,terminal_documents_per_case=480,heldout_per_case=len(test),
    vocabulary=32000,checks=checks,outcomes=outcomes,input_sha256=files,
    scope='Independent terminal raw-logit reconstruction for all twelve states and both dictionaries; no project imports, no native updates. The assessment-panel oracle is descriptive, not a forecast.'),indent=2)+'\n')
print('Passed',len(checks),'independent terminal geometry comparisons',flush=True)
