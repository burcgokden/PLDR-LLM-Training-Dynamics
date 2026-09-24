#!/usr/bin/env python3
"""Exact arithmetic for the illustrative row-to-prediction chain, not acquired data."""
import argparse,json
from fractions import Fraction as Q
from pathlib import Path

def check():
    a,b=Q(1),Q(-1,2)
    incoming=2*a*a;successor=2*(a+b)**2
    work=4*a*b;charge=2*b*b
    assert successor-incoming==work+charge
    alpha=-b/a;gain=(1-alpha)**2
    assert successor/incoming==gain==Q(1,4)
    alternative=2*(a-b)**2
    assert alternative==Q(9,2) and alternative!=successor
    # F(x,y)=(x+y,y); the prescribed slice (z,0) misses the realized y=1.
    state=(Q(1),Q(1));z0=state[0];transported_defect=Q(0)
    for t in range(5):
        x,y=state;actual=x+y;fixed=x;adapted=x+y
        defect=actual-fixed;transported_defect+=defect
        assert defect==1 and actual==adapted
        assert actual==z0+transported_defect
        # H(x,y)=y has a separate unit observation mismatch on the fixed slice.
        assert y-Q(0)==1
        state=(actual,y)
    # With K=L=1 and delta=0, the conditional bound is E/4.
    kl_bound=successor/4
    assert kl_bound==Q(1,8)
    values=dict(incoming_energy=incoming,successor_energy=successor,work=work,
                finite_step_charge=charge,radial_coefficient=alpha,exact_gain=gain,
                alternative_successor_energy=alternative,illustrative_kl_bound=kl_bound)
    return {'status':'passed','exact_rational_values':{k:str(v) for k,v in values.items()},
            'fixed_and_adapted_steps_checked':5,'new_experiments':0,
            'scope':'Illustrative exact arithmetic and explicit slice counterexample. The Lipschitz and logit sensitivity constants are hypotheses, not estimates for a trained model.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);a=p.parse_args();result=check();text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print(text,end='')
