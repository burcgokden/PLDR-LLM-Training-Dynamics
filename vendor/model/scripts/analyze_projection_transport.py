"""Post hoc finite-panel diagnostic; test-panel fits are oracles, not forecasts.

No project imports and no native execution. Complete selected terminal vector.
"""
from pathlib import Path
import argparse,hashlib,json,time
import numpy as np
parser=argparse.ArgumentParser()
parser.add_argument('--base',required=True)
parser.add_argument('--output',required=True)
a=parser.parse_args()
BASE=Path(a.base).resolve(); OUT=Path(a.output).resolve()
if OUT.exists():raise FileExistsError('Fresh projection output required')
OUT.parent.mkdir(parents=True,exist_ok=True)
source=BASE/'return-assessment/analysis.json'
started=time.time()
analysis=json.loads(source.read_text())
selected=[r for r in analysis['rows'] if r['time']==256 and r['cohort']=='heldout' and r['calibration_cohort']=='calibration']
if len(selected)!=12:raise ValueError('Expected all twelve terminal states')
rows=[]
for row in selected:
    for n in [2,7]:
        d=row if n==2 else row['expanded']
        G=np.array(d['incoming_gram']); C=np.array(d['cross_gram']); H=np.array(row['current_gram']); A=np.array(d['matrix'])
        ev=np.linalg.eigvalsh(G)
        if ev[0]<=1e-12*ev[-1]:raise ValueError('Unresolved dictionary')
        oracle=np.linalg.solve(G,C)
        residual=H-C.T@oracle
        residual=(residual+residual.T)/2
        mismatch=(A-oracle).T@G@(A-oracle)
        reported=np.array(d['residual_gram'])
        error=float(np.max(np.abs(reported-residual-mismatch)))
        if error>2e-11:raise ValueError('Pythagorean decomposition failed')
        if np.linalg.eigvalsh(residual)[0]<-2e-11:raise ValueError('Invalid projection Gram')
        energy=float(np.trace(reported)); irreducible=float(np.trace(residual)); excess=float(np.trace(mismatch))
        rows.append(dict(case=row['case'],heads=row['heads'],columns=n,documents=row['documents'],
            oracle_relative_residual=float(np.sqrt(max(0,irreducible)/np.trace(H))),
            calibrated_relative_residual=d['relative_residual'],oracle_residual_energy=irreducible,
            calibration_excess_energy=excess,orthogonal_fraction=irreducible/energy,
            max_identity_error=error,condition=float(ev[-1]/ev[0]),oracle_matrix=oracle.tolist(),
            oracle_residual_gram=residual.tolist(),coefficient_excess_gram=mismatch.tolist()))
summary=[]
for n in [2,7]:
    for h in [4,8,14]:
        rr=[r for r in rows if r['heads']==h and r['columns']==n]
        summary.append(dict(heads=h,columns=n,oracle_relative_range=[min(r['oracle_relative_residual'] for r in rr),max(r['oracle_relative_residual'] for r in rr)],orthogonal_fraction_range=[min(r['orthogonal_fraction'] for r in rr),max(r['orthogonal_fraction'] for r in rr)]))
result=dict(status='passed',analysis_kind='post hoc complete finite-panel diagnostic',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),native_updates=0,seconds=time.time()-started,rows=rows,summary=summary,scope='Test-panel oracle lower bounds for two fixed dictionaries. All 12 states, 24 dictionary cells. No future prediction or new population replication. Independent all-state raw-vocabulary reconstruction is recorded separately.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(summary,indent=2))
