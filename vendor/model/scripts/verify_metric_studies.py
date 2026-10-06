#!/usr/bin/env python3
"""Independent reconstruction and source checks for finite metric studies."""
from companion_paths import required_input, acquisition_identity
from companion_paths import configured_path
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict, discrepancy, array_discrepancy, finite_array, finite_scalar
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from model_rg.provenance import sha256, write_json

REPO=Path(__file__).resolve().parents[1]
ROOT=Path(configured_path('data:model'))


def verify(metric_root, matched_analysis, matched_refinement, output, source_root=None):
    if output.exists(): raise FileExistsError(output)
    bound={};results={};source_aliases={}
    def resolve_input(path,digest):
        actual=Path(path)
        if sha256(actual)==digest:return str(actual)
        if source_root is not None and actual.is_relative_to(REPO) and actual.suffix=='.py':
            retained=source_root/actual.relative_to(REPO)
            if retained.is_file() and sha256(retained)==digest:
                source_aliases[str(actual)]=str(retained.resolve())
                return str(retained.resolve())
        raise ValueError('Changed evidence input '+str(actual))
    paths={'metric':metric_root/'metric-budget.json','refinement':metric_root/'metric-refinement-final.json','matched':matched_analysis,'fresh_refinement':matched_refinement}
    scripts={'metric':'analyze_finite_metric_budget.py','refinement':'refine_finite_metric_budget.py','matched':'analyze_matched_onepass.py','fresh_refinement':'refine_matched_metric.py'}
    for name,path in paths.items():
        r=load_json_strict(path.read_text());results[name]=r
        if r['status']!='passed': raise ValueError('Incomplete study')
        source=r.get('script_sha256',r.get('analyzer_sha256'))
        if source!=sha256(REPO/'scripts'/scripts[name]): raise ValueError('Changed analyzer')
        for p,h in r['checked_sha256'].items():
            checked_path=resolve_input(p,h)
            bound[checked_path]=h
        bound[str(path)]=sha256(path)
    metric,refined,matched=[results[k] for k in ['metric','refinement','matched']]
    if metric['scale_cells']!=512 or metric['endpoint_spans']!=16320 or len(refined['records'])!=8:
        raise ValueError('Incomplete retained metric family')
    if matched['scientific_paths']!=24 or matched['scientific_updates']!=24576 or matched['endpoint_spans']!=12096 or matched['scale_cells']!=1152:
        raise ValueError('Incomplete fresh metric family')
    # Independent KL, covariance, and oscillation calculation. No project metric
    # calculator, native producer, or rendering code is imported.
    def endpoint(a,b):
        pa=np.exp(a-logsumexp(a,axis=-1,keepdims=True))
        d=b-a
        kl=logsumexp(b,axis=-1)-logsumexp(a,axis=-1)-np.sum(pa*d,axis=-1)
        mean=np.sum(pa*d,axis=-1,keepdims=True)
        q=np.sum(pa*(d-mean)**2,axis=-1)/2
        return kl,q,np.max(d,axis=-1)-np.min(d,axis=-1)
    maxerror=0.;checked_spans=0
    def cells_check(logits,cells):
        nonlocal maxerror,checked_spans
        for cell in cells:
            probe=cell['probe']-1;b=cell['block'];a=logits[:,probe].astype(np.float64)
            kl,q,r=endpoint(a[:-b:b],a[b::b])
            for key,actual in [('kl',kl),('quadratic',q),('oscillation',r)]:
                difference=array_discrepancy(actual,cell[key],'metric '+key,atol=2e-12,rtol=2e-12);maxerror=max(maxerror,difference)
                if not np.allclose(actual,cell[key],atol=2e-12,rtol=2e-12): raise ValueError('Independent endpoint mismatch')
            finite_array(kl, 'metric KL');finite_array(cell['lower'], 'metric lower');finite_array(cell['upper'], 'metric upper')
            if np.any(kl<np.asarray(cell['lower'])-2e-12*(1+np.abs(kl))) or np.any(finite_greater(kl, np.asarray(cell['upper'])+2e-12*(1+np.abs(kl)), 'scripts/verify_metric_studies.py:51')):
                raise ValueError('Independent bound violation')
            checked_spans+=len(kl)
    partial={}
    for suffix,folder in [('A','potential-factorial-20260913'),('B','potential-factorial-disjoint-20260914')]:
        spec=load_json_strict((ROOT/folder/'protocol.json').read_text())
        for case in spec['cases']:
            for branch in ['native_keep','raised_keep','native_reset','raised_reset']:
                with np.load(ROOT/folder/'runs'/case['name']/(branch+'.npz'),allow_pickle=False) as z: logits=z['step_logits']
                cells=[c for c in metric['cells'] if c['suffix']==suffix and c['case']==case['name'] and c['branch']==branch]
                if len(cells)!=16: raise ValueError('Missing retained cell')
                cells_check(logits,cells)
                if case['name']=='subcritical1': partial[suffix,branch]=logits.astype(np.float64)
    resolved=0
    for row in refined['records']:
        arrays=[partial[row['suffix'],name+'_'+row['moment']][:,row['probe']-1] for name in ['native','raised']]
        totals=[endpoint(x[:-1],x[1:])[0].sum() for x in arrays];delta=float(totals[1]-totals[0])
        discrepancy(delta,row['delta_kl'],'refined contrast',atol=1e-10)
        previous=(-float('inf'),float('inf'))
        for level in row['levels']:
            lo,hi=level['lower'],level['upper']
            finite_scalar(lo,'refined lower');finite_scalar(hi,'refined upper')
            if lo>hi:raise ValueError('Reversed interval')
            if lo>delta+1e-10 or hi<delta-1e-10 or lo<previous[0]-1e-12 or hi>previous[1]+1e-12:
                raise ValueError('Invalid refined interval')
            previous=lo,hi
        resolved+=int(previous[0]>1e-8 or previous[1]<-1e-8)
    protocol_paths=[Path(p) for p in matched['checked_sha256'] if p.endswith('/protocol.json') and load_json_strict(Path(p).read_text()).get('schema')=='matched-onepass-v1']
    if len(protocol_paths)!=1: raise ValueError('Missing matched protocol')
    study=protocol_paths[0].parent
    qpath=study/'qualification/verification.json';q=load_json_strict(qpath.read_text())
    if q['status']!='passed' or q['scientific_updates']!=0 or q['native_updates']!=64 or q['protocol_sha256']!=sha256(protocol_paths[0]):
        raise ValueError('Missing widest replay qualification')
    for p,h in q['checked_sha256'].items():
        checked_path=resolve_input(p,h)
        bound[checked_path]=h
    bound[str(qpath)]=sha256(qpath)
    for record in matched['records']:
        job=record['job']
        with np.load(study/'runs'/job['run_id']/'observations.npz',allow_pickle=False) as z:
            cells=[c for c in matched['metric_cells'] if c['run_id']==job['run_id']]
            if len(cells)!=48: raise ValueError('Missing matched cell')
            cells_check(z['logits'],cells)
    fresh=results['fresh_refinement']
    if fresh['endpoint_pairs']!=192 or len(fresh['records'])!=192 or fresh['tolerance']!=.01:
        raise ValueError('Incomplete one-percent refinement')
    cells={(c['run_id'],c['probe']):c for c in matched['metric_cells'] if c['block']==32}
    seen=set()
    for row in fresh['records']:
        key=row['run_id'],row['probe']
        if key in seen or key not in cells:raise ValueError('Duplicate or unknown refined endpoint')
        seen.add(key);c=cells[key];kl=c['kl'][0];radius=c['oscillation'][0]
        for key in ['kl','oscillation','midpoint','lower','upper','intervals']:finite_scalar(row[key],'refinement '+key)
        if type(row['intervals']) is not int or row['intervals']<=0:raise ValueError('Invalid segment count')
        envelope=np.exp(radius/(2*row['intervals']))
        finite_scalar(envelope,'refinement envelope');finite_scalar(kl,'reconstructed KL');finite_scalar(radius,'reconstructed radius')
        if envelope>1.01+1e-14 or finite_greater(abs(kl-row['kl']), 2e-12, 'scripts/verify_metric_studies.py:102') or finite_greater(abs(radius-row['oscillation']), 2e-12, 'scripts/verify_metric_studies.py:102'):
            raise ValueError('Wrong refinement domain or segment count')
        np.testing.assert_allclose([row['lower'],row['upper']],[row['midpoint']/envelope,row['midpoint']*envelope],rtol=1e-13,atol=1e-13)
        if not row['lower']-2e-12<=kl<=row['upper']+2e-12 or finite_greater(abs(row['midpoint']/kl-1), .01+1e-12, 'scripts/verify_metric_studies.py:105'):
            raise ValueError('Refined one-percent budget failed')
    if seen!=set(cells):raise ValueError('Missing full-horizon endpoint')
    if checked_spans!=28416 or resolved!=8: raise ValueError('Wrong complete metric inventory')
    # CLI checks have separate roles and are explicitly tested under -O as well.
    for name,optimized in [('ordinary',False),('optimized',True)]:
        p=REPO/required_input('verify-metric-studies-input-1')/('cli-'+name+'.json');r=load_json_strict(p.read_text())
        if r['status']!='passed' or r['checked_cases']!=12 or r['python_optimized']!=optimized or r['native_updates']!=0 or r['script_sha256']!=sha256(REPO/'scripts/run_potential_factorial.py'):
            raise ValueError('Missing current CLI role check')
        bound[str(p)]=sha256(p)
    write_json(output,dict(status='passed',schema='finite-metric-studies-verification-v1',
        endpoint_spans=checked_spans,resolved_partial_state_contrasts=resolved,
        independent_endpoint_max_error=maxerror,scientific_updates=24576,
        qualification_updates=64,one_percent_refined_pairs=192,matched_refinement=str(matched_refinement),checked_sha256=bound,verifier_sha256=sha256(__file__),
        metric_root=str(metric_root),matched_analysis=str(matched_analysis),retained_source_aliases=source_aliases,
        scope='Independent finite observation reconstruction and input identities; no rigorous floating-point intervals, native critical exponents, or autonomous forecasts.'))
    print('PASS',checked_spans,'independently reconstructed spans',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--metric-root',type=Path,required=True);p.add_argument('--matched-analysis',type=Path,required=True);p.add_argument('--matched-refinement',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-root',type=Path)
    a=p.parse_args();verify(a.metric_root,a.matched_analysis,a.matched_refinement,a.output,a.source_root)
