#!/usr/bin/env python3
"""Finite control-window and width diagnostics; no universal exponent is presumed."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from model_rg.provenance import sha256,write_json


def downward_crossings(g,y,level):
    result=[]
    for a,b,u,v in zip(g[:-1],g[1:],y[:-1],y[1:]):
        if u>level>=v:
            result.append(float(a+(level-u)*(b-a)/(v-u)))
    return result


def width_models(n,chi):
    n=np.asarray(n,dtype=float);chi=np.asarray(chi,dtype=float)
    if len(n)<3 or np.any(chi<=0):return dict(status='unresolved')
    logn=np.log(n);logchi=np.log(chi)
    slope,intercept=np.polyfit(logn,logchi,1)
    # Fit on every smaller width, then predict the withheld largest one.
    k,c=np.polyfit(logn[:-1],logchi[:-1],1)
    out=dict(status='descriptive',effective_log_size_slope=float(slope),
        power_fit=(np.exp(intercept)*n**slope).tolist(),
        withheld_largest_power_prediction=float(np.exp(c)*n[-1]**k),
        actual_largest=float(chi[-1]))
    for label,X in [('regular',np.column_stack([1-n.min()/n,n.min()/n])),
                    ('shared_mode',np.column_stack([np.ones_like(n),n])),
                    ('shared_with_signed_correction',np.column_stack([n-n.min(),np.full_like(n,n.min())]))]:
        fit=lsq_linear(X,chi,bounds=(0,np.inf))
        held=lsq_linear(X[:-1],chi[:-1],bounds=(0,np.inf))
        bases={'regular':'[1-Nmin/N,Nmin/N]', 'shared_mode':'[1,N]',
               'shared_with_signed_correction':'[N-Nmin,Nmin]'}
        out[label]=dict(coefficients=fit.x.tolist(),basis=bases[label],
                        fitted=(X@fit.x).tolist(),
                        withheld_largest_prediction=float(X[-1]@held.x))
        if label=='shared_with_signed_correction':
            out[label]['asymptotic_intensive_variance']=float(fit.x[0])
            out[label]['susceptibility_intercept']=float(n.min()*(fit.x[1]-fit.x[0]))
            out[label]['scope']='A*N+B with A>=0 and positivity for every N>=Nmin; the finite correction B can have either sign.'
    out['scope']='A finite width slope and model comparisons, not a thermodynamic or universal exponent.'
    return out


def analyze(paths,output):
    if output.exists():raise FileExistsError(output)
    records=[json.loads(p.read_text()) for p in paths]
    if any(d['status']!='complete' for d in records):raise ValueError('All grids must be complete')
    crossings=[];widths=[];horizon_fits=[];checked={str(p):sha256(p) for p in paths}
    for d in records:
        grid=defaultdict(list)
        for row in d['cells']:
            if row['field']=='row':grid[(row['environment'],row['heads'],row['step'])].append(row)
        for (e,n,t),rows in sorted(grid.items()):
            rows.sort(key=lambda r:r['control'])
            if rows[0]['control']!=0 or len(rows)!=len(d['design']['controls']):continue
            g=np.array([r['control'] for r in rows]);mean=np.array([r['mean'] for r in rows])
            base=mean[0]
            if base<=0:continue
            levels={str(v):downward_crossings(g,mean/base,v) for v in [.1,.5,.9]}
            brackets={str(v):[[float(a),float(b)] for a,b,u,w in zip(g[:-1],g[1:],(mean/base)[:-1],(mean/base)[1:])
                              if u>v>=w] for v in [.1,.5,.9]}
            crossings.append(dict(study=d['study'],environment=e,heads=n,step=t,zero_control_mean=float(base),
                                  crossings=levels,brackets=brackets,nonmonotone_mean=bool(np.any(np.diff(mean)>0.02))))
        peak_groups=defaultdict(list)
        for row in d['peaks']:
            if row['step'] in [256,512,1024,2048,4096,8192,16384,32768]:
                peak_groups[(row['environment'],row['step'],row['field'])].append(row)
        for (e,t,field),rows in sorted(peak_groups.items()):
            rows.sort(key=lambda r:r['heads'])
            if len(rows)<3:continue
            result=width_models([r['heads'] for r in rows],[r['sampled_maximum'] for r in rows])
            result.update(study=d['study'],environment=e,step=t,field=field,
                          heads=[r['heads'] for r in rows],peak_controls=[r['sampled_maximum_control'] for r in rows],
                          all_peaks_interior=all(r['interior'] for r in rows),
                          half_widths=[r['half_width'] for r in rows])
            widths.append(result)
        by_head=defaultdict(list)
        for r in crossings:
            if r['study']==d['study'] and r['step']>=256 and len(r['crossings']['0.5'])==1 and r['brackets']['0.5'][0][0]>0:
                by_head[(r['environment'],r['heads'])].append(r)
        for (e,n),rows in sorted(by_head.items()):
            times=np.array([r['step'] for r in rows],dtype=float)
            controls=np.array([r['crossings']['0.5'][0] for r in rows])
            if len(rows)<4 or np.any(controls<=0):continue
            # This is a kinetic diagnostic, not a chosen critical limit.
            slope,offset=np.polyfit(np.log(times),np.log(controls),1)
            prediction=np.exp(offset)*times**slope
            horizon_fits.append(dict(study=d['study'],environment=e,heads=n,points=len(rows),
                first_step=int(times.min()),last_step=int(times.max()),
                log_control_vs_log_time_slope=float(slope),coefficient=float(np.exp(offset)),
                relative_rmse=float(np.sqrt(np.mean(((prediction-controls)/controls)**2))),
                scope='Finite-time crossing drift on the sampled grid; neither a critical-point estimate nor a universal exponent.'))
    result=dict(schema='critical-onepass-scaling-v1',status='complete',crossings=crossings,width_diagnostics=widths,
        horizon_crossing_fits=horizon_fits,checked_sha256=checked,
        source_sha256=sha256(__file__),
        conclusion='Descriptive finite-size and finite-time diagnostics. Native criticality requires independent control, response and limit evidence.')
    write_json(output,result)
    print(json.dumps({'status':'complete','crossings':len(crossings),'width_diagnostics':len(widths),'horizon_fits':len(horizon_fits)}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    analyze([p.resolve() for p in a.input],a.output.resolve())

if __name__=='__main__':main()
