"""Test a finite Binder/thermal-contrast error budget on all 18 exact laws.
The bounds use complete probabilities, not empirical NLL as a proxy for KL.
"""
from companion_paths import legacy_path
import hashlib
import json
from pathlib import Path
import numpy as np

STUDY=Path(legacy_path('/pldr-data/model/released-adaptation-20260913'))

def main():
    panels={};rows=[];inputs={}
    for name in ['base5','physical-s0','physical-s1']:
        root=STUDY/'assessment'/name/'exact'
        record=json.loads((root/'result.json').read_text())
        for cell in record['rows']:
            path=root/cell['raw'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
            if digest!=cell['sha256']:raise ValueError('Changed exact-law input')
            inputs[str(path)]=digest
            with np.load(path) as z:
                x=z['configurations'];P=np.exp(z['source_logp']);Q=np.exp(z['model_logp'])
            q=cell['q'];f=np.stack([(x==a).mean((1,2)) for a in range(q)],axis=1)
            m2=q/(q-1)*np.square(f-1/q).sum(1)
            a=float(P@m2);b=float(P@(m2*m2));c=float(Q@m2);d=float(Q@(m2*m2))
            if min(a,b,c,d)<=0:raise ValueError('Nonpositive exact moment')
            RP=b/a**2;RQ=d/c**2
            bound=abs(d-b)/c**2+b*abs(c-a)*(c+a)/(c**2*a**2)
            if abs(RQ-RP)>bound+1e-12:raise ValueError('Exact ratio budget failed')
            e2=abs(c-a)/a;e4=abs(d-b)/b
            relative=RP*(e4+2*e2+e2**2)/(1-e2)**2 if e2<1 else None
            if relative is not None and abs(RQ-RP)>relative+1e-12:
                raise ValueError('Relative ratio budget failed')
            r=dict(name=name,q=q,L=2,ratio=cell['ratio'],source_m2=a,model_m2=c,
                source_binder=RP,model_binder=RQ,error=abs(RQ-RP),exact_ratio_bound=bound,
                relative_second_error=e2,relative_fourth_error=e4,relative_bound=relative)
            rows.append(r);panels[name,q,cell['ratio']]=r
    contrasts=[]
    for name in ['base5','physical-s0','physical-s1']:
        for q in [2,3]:
            lo=panels[name,q,.94];hi=panels[name,q,1.06];h=.06
            observed=((hi['model_binder']-lo['model_binder'])-(hi['source_binder']-lo['source_binder']))/(2*h)
            bound=(hi['exact_ratio_bound']+lo['exact_ratio_bound'])/(2*h)
            if abs(observed)>bound+1e-12:raise ValueError('Thermal budget failed')
            contrasts.append(dict(name=name,q=q,L=2,half_width=h,error=abs(observed),bound=bound))
    # Two equally weighted deterministic binary targets: better log score can
    # coexist with worse top-1 decisions. This is a mathematical example.
    before=np.array([.51,.01]);after=np.array([.49,.49])
    example=dict(before_nll=float(-np.log(before).mean()),after_nll=float(-np.log(after).mean()),
        before_accuracy=float(np.mean(before>.5)),after_accuracy=float(np.mean(after>.5)))
    result=dict(status='passed',scope=__doc__,moments=rows,contrasts=contrasts,inputs=inputs,
        exact_laws=len(rows),thermal_comparisons=len(contrasts),relative_budget_eligible=sum(r['relative_bound'] is not None for r in rows),
        nll_accuracy_example=example,producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print({key:result[key] for key in ['status','exact_laws','thermal_comparisons','relative_budget_eligible','nll_accuracy_example']})

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--study',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();STUDY=args.study.resolve();OUT=args.output.resolve();main()
