#!/usr/bin/env python3
"""Score frozen coarse predictions against the complete independent refinement."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256,write_json


def score_prediction(prediction,observed,bracket):
    if not np.isfinite([prediction,observed,*bracket]).all() or observed<=0 or bracket[0]>bracket[1]:
        raise ValueError('Invalid positive crossing or interpolation bracket')
    return dict(prediction=float(prediction),observed=float(observed),bracket=list(bracket),
        signed_error=float(prediction-observed),relative_error=float((prediction-observed)/observed),
        absolute_distance_outside_bracket=float(max(bracket[0]-prediction,prediction-bracket[1],0.)))



def peak_segments(cells, peak):
    """Retain the control-grid segments behind each interpolated half width."""
    rows=sorted([r for r in cells if r['field']==peak['field'] and r['heads']==peak['heads']
        and r['step']==peak['step'] and r['environment']==peak['environment']],key=lambda r:r['control'])
    g=np.array([r['control'] for r in rows]);y=np.array([r['susceptibility'] for r in rows])
    k=int(np.argmax(y));level=float(y[k]/2);left=right=None
    if g[k]!=peak['sampled_maximum_control'] or y[k]!=peak['sampled_maximum']:
        raise ValueError('Peak profile differs from its complete native cells')
    for i in range(k-1,-1,-1):
        if y[i]<=level<y[i+1]:left=[float(g[i]),float(g[i+1])];break
    for i in range(k,len(g)-1):
        if y[i]>level>=y[i+1]:right=[float(g[i]),float(g[i+1])];break
    for bracket,point in [(left,peak['half_left']),(right,peak['half_right'])]:
        if (bracket is None)!=(point is None) or bracket is not None and not bracket[0]<=point<=bracket[1]:
            raise ValueError('Half-height interpolation lies outside its observed grid segment')
    return dict(heads=peak['heads'],step=peak['step'],environment=peak['environment'],
        all_tied_global_maximum_controls=g[y==y[k]].tolist(),
        local_maximum_controls=[float(g[i]) for i in range(1,len(g)-1) if y[i]>=y[i-1] and y[i]>=y[i+1]],
        half_height=level,left_segment=left,right_segment=right,
        half_left=peak['half_left'],half_right=peak['half_right'],interpolated_width=peak['half_width'],
        width_range_within_grid_segments=[max(0.,right[0]-left[1]),right[1]-left[0]] if left and right else None,
        scope='Observed segments for the piecewise-linear profile; the segment range measures grid resolution, not statistical uncertainty or a continuum critical window.')


def analyze(coarse,refinement,output):
    if output.exists():raise FileExistsError(output)
    checked={}
    def read(path,status=None):
        d=json.loads(path.read_text());checked[str(path)]=sha256(path)
        if status and d.get('status')!=status:raise ValueError('Incomplete prediction input '+str(path))
        return d
    old=read(coarse/'analysis.json','complete');new=read(refinement/'analysis.json','complete')
    p=read(refinement/'protocol.json');selection=read(refinement/'selection-decision.json')
    if selection!=p['selection'] or sha256(refinement/'selection-decision.json')!=p['selection_record_sha256']:
        raise ValueError('Frozen selection changed')
    for name,digest in selection['search_analyses'].items():
        if sha256(name)!=digest:raise ValueError('Changed preselection evidence')
        checked[name]=digest
    if selection['search_analyses'].get(str(coarse/'analysis.json'))!=sha256(coarse/'analysis.json'):
        raise ValueError('Foreign search study')
    if set(new['design']['seeds'])&set(old['design']['seeds']):raise ValueError('Confirmation initializations overlap search')
    if set(new['design']['environments'])&set(old['design']['environments']):raise ValueError('Confirmation source partition overlaps search')
    for study in [coarse,refinement]:
        verification=read(study/'verification.json','passed')
        if verification['protocol_sha256']!=sha256(study/'protocol.json'):raise ValueError('Unbound study verification')
    scaling=read(refinement/'scaling.json','complete')
    joint=read(refinement/'joint-analysis.json','complete')
    predictions=selection['hypotheses']['frozen_half_crossing_predictions'];comparisons=[];unresolved=[]
    for row in scaling['crossings']:
        n=row['heads'];t=row['step']
        if t<256:continue
        if len(row['crossings']['0.5'])!=1 or row['brackets']['0.5'][0][0]<=0:
            unresolved.append(dict(heads=n,step=t,crossings=row['crossings']['0.5'],brackets=row['brackets']['0.5']));continue
        for name,model in predictions[str(n)]['models'].items():
            value=model['coefficient']*t**model['control_time_exponent']
            comparisons.append(dict(heads=n,step=t,model=name,
                prediction_role='new_width_saturation_reference' if n==24 else 'independent_source_and_initialization',
                horizon_role='extended_horizon' if t>old['design']['steps'] else 'within_search_horizon',
                **score_prediction(value,row['crossings']['0.5'][0],row['brackets']['0.5'][0])))
    aggregate=defaultdict(list)
    for row in comparisons:aggregate[(row['heads'],row['model'],row['horizon_role'])].append(row)
    errors=[]
    for (n,model,horizon),rows in sorted(aggregate.items()):
        errors.append(dict(heads=n,model=model,horizon_role=horizon,observation_times=len(rows),
            first_step=min(r['step'] for r in rows),last_step=max(r['step'] for r in rows),
            relative_rmse=float(np.sqrt(np.mean([r['relative_error']**2 for r in rows]))),
            mean_absolute_control_error=float(np.mean([abs(r['signed_error']) for r in rows])),
            fraction_predictions_inside_observed_bracket=float(np.mean([r['absolute_distance_outside_bracket']==0 for r in rows])),
            scope='Descriptive error across correlated horizons, not independent test replicates or a calibrated significance test.'))
    width_windows=[];regions=[]
    for t in [2048,4096]:
        rows=sorted([r for r in new['peaks'] if r['field']=='row' and r['step']==t],key=lambda r:r['heads'])
        regions.extend(peak_segments(new['cells'],r) for r in rows)
        entry=dict(step=t,heads=[r['heads'] for r in rows],peak_controls=[r['sampled_maximum_control'] for r in rows],
            peak_susceptibilities=[r['sampled_maximum'] for r in rows],half_widths=[r['half_width'] for r in rows],
            all_interior=all(r['interior'] for r in rows),
            scope='Resolved finite grid widths and slopes; head count is a size coordinate, not a spatial length.')
        if all(r['half_width'] is not None and r['half_width']>0 for r in rows) and len(rows)>=3:
            n=np.array(entry['heads'],float);w=np.array(entry['half_widths'],float)
            slope,offset=np.polyfit(np.log(n),np.log(w),1)
            held_slope,held_offset=np.polyfit(np.log(n[:-1]),np.log(w[:-1]),1)
            entry.update(half_width_log_size_slope=float(slope),withheld_largest_width_prediction=float(np.exp(held_offset)*n[-1]**held_slope))
        width_windows.append(entry)
    result=dict(schema='critical-independent-confirmation-v1',status='complete',coarse=str(coarse),refinement=str(refinement),
        checked_sha256=checked,source_sha256={'scripts/analyze_critical_confirmation.py':sha256(__file__)},
        frozen_prediction_comparisons=comparisons,frozen_prediction_error_summaries=errors,unresolved_crossings=unresolved,
        primary_width_windows=width_windows,primary_peak_regions=regions,refinement_width_model_comparisons=scaling['width_diagnostics'],
        descriptive_milestone_drifts=joint['crossing_drifts'],
        endpoint_crossing_predictions=[r for r in comparisons if r['step'] in [2048,4096]],
        scientific_updates=new['scientific_updates'],additional_training_updates=0,
        interpretation='Prediction coefficients are unchanged from the pre-execution selection. Finite-time replication and kinetic predictions do not establish a positive limiting critical control, a stationary response gap or universal exponents.')
    write_json(output,result)
    print(json.dumps({'status':'complete','frozen_predictions_scored':len(comparisons),'unresolved_crossings':len(unresolved)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['coarse','refinement','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();analyze(a.coarse.resolve(),a.refinement.resolve(),a.output.resolve())
