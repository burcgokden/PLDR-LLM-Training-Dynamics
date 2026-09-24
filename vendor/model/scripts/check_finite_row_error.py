"""Independent matrix algebra controls; no author mathematical helper imports."""
from pathlib import Path
import argparse, json, numpy as np
from model_rg.provenance import sha256
from numerical_validation import load_json_strict
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--study',type=Path,required=True)
ap.add_argument('--output',type=Path,required=True)
args=ap.parse_args()
if args.output.exists():raise FileExistsError(args.output)
rng=np.random.default_rng(9162635)
maxerr={k:0. for k in ['exact_identity','derivative_defect','derivative_norm','blocking']}
ratios=[]
def check(A,D):
    PA=A-A.mean(-2,keepdims=True);PD=D-D.mean(-2,keepdims=True)
    E=np.sum(A*A,axis=(-2,-1));Ep=np.sum((A+D)**2,axis=(-2,-1))
    u=np.sum(PA*PA,axis=(-2,-1))/E
    up=np.sum((PA+PD)**2,axis=(-2,-1))/Ep
    L=2*np.sum(PA*PD,axis=(-2,-1))-2*u*np.sum(A*D,axis=(-2,-1))
    Q=np.sum(PD*PD,axis=(-2,-1))-u*np.sum(D*D,axis=(-2,-1))
    deriv=L/E;delta=np.sqrt(np.sum(D*D,axis=(-2,-1))/E)
    maxerr['exact_identity']=max(maxerr['exact_identity'],float(np.max(np.abs(up-u-(L+Q)/Ep))))
    defect=up-u-deriv;identity=Q/Ep-(Ep-E)/Ep*deriv
    maxerr['derivative_defect']=max(maxerr['derivative_defect'],float(np.max(np.abs(defect-identity))))
    grad=2*(PA-u[...,None,None]*A)/E[...,None,None]
    maxerr['derivative_norm']=max(maxerr['derivative_norm'],float(np.max(np.abs(np.sum(grad*grad,axis=(-2,-1))-4*u*(1-u)/E))))
    mask=delta<1
    if np.any(mask):
        bound=(3*delta[mask]**2+delta[mask]**3)/(1-delta[mask])**2
        if np.any(np.abs(defect[mask])>bound+5e-14):raise ValueError('Remainder bound failed')
        ratios.extend((np.abs(defect[mask])/np.maximum(bound,1e-300)).ravel().tolist())
    return dict(count=int(E.size),delta_max=float(delta.max()),bounded=int(np.count_nonzero(mask)))
for k in range(1000):
    d=int(rng.integers(2,17));A=rng.normal(size=(d,d));D=rng.normal(size=(d,d));D*=10**rng.uniform(-5,np.log10(.8))*np.linalg.norm(A)/np.linalg.norm(D)
    check(A,D)
    D2=.07*rng.normal(size=(d,d))
    row=lambda a:np.sum((a-a.mean(0))**2)/np.sum(a*a)
    e=(row(A+D)-row(A))+(row(A+D+D2)-row(A+D))-(row(A+D+D2)-row(A))
    maxerr['blocking']=max(maxerr['blocking'],abs(float(e)))
study=args.study
p=load_json_strict((study/'protocol.json').read_text());native=[]
checked={'protocol.json':sha256(study/'protocol.json')}
expected={(n,g,9163401) for n in [4,8,14,24] for g in [1.5,2.]}
if len(p['jobs'])!=8 or {(j['heads'],j['control'],j['seed']) for j in p['jobs']}!=expected:raise ValueError('Incomplete finite native grid')
for j in p['jobs']:
    path=study/'runs'/j['run_id']/'observations.npz'
    manifest=load_json_strict((path.parent/'manifest.json').read_text())
    if manifest['artifacts']['observations.npz']!=sha256(path):raise ValueError('Changed finite matrices')
    checked[str(path.relative_to(study))]=sha256(path)
    with np.load(path) as z:
        A=z['A'].astype(float)
    native.append(dict(heads=j['heads'],control=j['control'],**check(A[0],A[3]-A[0])))
# Counterexample: a finite cross term has the outgoing energy denominator.
A=np.array([[1.,0.],[0.,0.]]);D=np.array([[0.,0.],[1.,0.]])
P=lambda a:a-a.mean(0)
E=np.sum(A*A);Ep=np.sum((A+D)**2);u=np.sum(P(A)**2)/E
L=2*np.sum(P(A)*P(D))-2*u*np.sum(A*D)
example=dict(u_before=float(u),u_after=float(np.sum(P(A+D)**2)/Ep),finite_cross=float(L/Ep),matrix_directional_derivative=float(L/E))
if max(maxerr.values())>1e-10:raise ValueError(maxerr)
out=dict(status='passed',source_sha256=sha256(__file__),checked_sha256=checked,random_transitions=1000,native_cases=native,native_pairs=sum(c['count'] for c in native),maximum_errors=maxerr,maximum_fraction_of_bound=max(ratios),finite_cross_counterexample=example,scope='Finite exact matrix identities and conditional remainder bound only; native fields reused, zero new forwards or optimizer updates.')
args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
