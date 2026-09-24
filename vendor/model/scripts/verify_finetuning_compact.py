"""Reconstruct spectra, finite source transport and size secants in the source ZIP."""
import json
from pathlib import Path
import numpy as np


def verify(source):
    g=Path(source)/'generated';data=json.loads((g/'finetuning-analysis.json').read_text());ret=json.loads((g/'finetuning-return-analysis.json').read_text())
    spec=json.loads((g/'finetuning-protocol.json').read_text());rspec=json.loads((g/'finetuning-return-protocol.json').read_text())
    if data['status']!='complete' or ret['status']!='complete':raise ValueError('Incomplete compact fine-tuning evidence')
    if (data['primary_paths'],data['control_paths'],data['scientific_updates'],ret['scientific_paths'],ret['scientific_updates'])!=(96,21,59904,36,9216):raise ValueError('Incomplete scientific path inventory')
    cohorts={'all','heldout','calibration','technical','narrative','unassigned','technical-heldout','narrative-heldout','unassigned-heldout'}
    cohorts|={x+'-fixed' for x in cohorts}
    expected={(c['name'],t,cohort) for c in spec['cases'] for t in [16,64,128,256,512] for cohort in cohorts}
    actual={(r['case'],r['time'],r['cohort']) for r in data['responses']}
    if len(data['responses'])!=len(expected) or actual!=expected:raise ValueError('Incomplete response grid')
    def eq(x,y):
        if x is None or y is None:
            if x is not None or y is not None:raise ValueError('Changed unresolved statistic')
        elif not np.allclose(x,y,atol=5e-11,rtol=5e-9):raise ValueError('Changed compact reduction')
    for r in data['responses']:
        for key in ['predictive','common']:
            q=r[key];m=np.array(q['gram']);eig=np.maximum(0.,np.linalg.eigvalsh(m)[::-1])
            eq(q['eigenvalues'],eig);eq(q['second_to_first'],eig[1]/eig[0] if eig[0]>0 else None);eq(q['cosine'],m[0,1]/np.sqrt(m[0,0]*m[1,1]) if m[0,0]*m[1,1]>0 else None)
            if q['second_above_reporting_floor']!=bool(np.sqrt(eig[1])>5e-7):raise ValueError('Changed reporting-floor decision')
    forecasts={(r['case'],r['time'],r['cohort'],r['rho']) for r in data['mixture_forecasts']}
    if forecasts!={(*k,rho) for k in expected for rho in [.125,.25,.75,.875]} or len(forecasts)!=len(data['mixture_forecasts']):raise ValueError('Incomplete untouched-mixture grid')
    for r in data['mixture_forecasts']:
        eq(r['affine_relative'],r['affine_rms']/r['signal_rms']);eq(r['quadratic_relative'],r['quadratic_rms']/r['signal_rms'])
        if r['improvement']!=(r['quadratic_rms']<r['affine_rms']):raise ValueError('Changed forecast decision')
    def angles(g,h,c):
        def inverse(x):
            w,v=np.linalg.eigh(x)
            if w[0]<=1e-13*w[-1]:return None
            return (v/np.sqrt(w))@v.T
        a=inverse(g);b=inverse(h)
        return None if a is None or b is None else np.minimum(np.linalg.svd(a@c@b,compute_uv=False),1.)
    for r in data['time_alignment']:
        result=angles(np.array(r['early_gram']),np.array(r['final_gram']),np.array(r['cross_gram']))
        if result is None:
            if r['canonical_cosines'] is not None:raise ValueError('Unresolved time span treated as resolved')
        else:eq(r['canonical_cosines'],result)
    for r in data['susceptibilities']:
        m=np.array(r['layer_covariance']);eq(r['common_chi'],r['heads']*np.trace(m)/5)
        eq(r['common_intensive'],np.trace(m)/5);eq(r['layer_covariance_eigenvalues'],np.linalg.eigvalsh(m)[::-1])
    for r in data['size_diagnostics']:
        vals=np.array(r['values']);sizes=[4,8,14]
        adj=[np.log(vals[i+1]/vals[i])/np.log(sizes[i+1]/sizes[i]) if vals[i]>0 and vals[i+1]>0 else None for i in range(2)]
        k=np.log(vals[2]/vals[0])/np.log(14/4) if vals[2]>0 and vals[0]>0 else None
        pred=vals[0]*2**k if k is not None else None
        for x,y in zip(r['adjacent_slopes'],adj):eq(x,y)
        eq(r['endpoint_slope'],k);eq(r['middle_prediction'],pred);eq(r['middle_relative_error'],abs(pred-vals[1])/vals[1] if pred is not None and vals[1]>0 else None)
    expected_return={(c['name'],t,cohort) for c in rspec['cases'] for t in [16,64,128,256] for cohort in cohorts}
    if {(r['case'],r['time'],r['cohort']) for r in ret['rows']}!=expected_return or len(ret['rows'])!=len(expected_return):raise ValueError('Incomplete return grid')
    for r in ret['rows']:
        expected_cal=48 if r['cohort'].endswith('-fixed') or r['time']!=256 else 240
        if r['calibration_documents']!=expected_cal:raise ValueError('Time-comparison calibration budget changed')
        gc=np.array(r['calibration_gram']);cc=np.array(r['calibration_cross']);m=np.linalg.solve(gc,cc)
        g0=np.array(r['incoming_gram']);gt=np.array(r['current_gram']);c=np.array(r['cross_gram'])
        error=gt-m.T@c-c.T@m+m.T@g0@m;error=(error+error.T)/2
        eps=np.sqrt(max(0.,np.trace(error)));sv=np.linalg.svd(m,compute_uv=False);eig=np.linalg.eigvalsh(g0)[::-1]
        lower=np.maximum(0.,sv[-1]*np.sqrt(np.maximum(eig,0))-eps);upper=sv[0]*np.sqrt(np.maximum(eig,0))+eps
        canonical=angles(g0,gt,c)
        if canonical is not None:eq(r['canonical_cosines'],canonical)
        elif r['canonical_cosines'] is not None:raise ValueError('Unresolved return span treated as resolved')
        eq(r['matrix'],m);eq(r['residual_gram'],error);eq(r['residual_rms'],eps);eq(r['singular_values'],sv)
        eq(r['relative_residual'],eps/np.sqrt(np.trace(gt)));eq(r['residual_to_weak_incoming'],eps/np.sqrt(eig[-1]))
        eq(r['lower_singular_bounds'],lower);eq(r['upper_singular_bounds'],upper)
        if r['weak_lower_bound_positive']!=bool(lower[-1]>0):raise ValueError('Changed finite span certificate')
        q=r['expanded'];dg=np.array(q['incoming_gram']);dgc=np.array(q['calibration_gram']);dc=np.array(q['cross_gram']);dcc=np.array(q['calibration_cross'])
        dm=np.linalg.solve(dgc,dcc);de=gt-dm.T@dc-dc.T@dm+dm.T@dg@dm;de=(de+de.T)/2
        deps=np.sqrt(max(0.,np.trace(de)));dvals=np.linalg.eigvalsh(dgc)[::-1]
        eq(q['matrix'],dm);eq(q['residual_gram'],de);eq(q['residual_rms'],deps);eq(q['relative_residual'],deps/np.sqrt(np.trace(gt)))
        eq(q['calibration_eigenvalues'],dvals);eq(q['calibration_condition'],dvals[0]/dvals[-1])

    return dict(primary_paths=96,controls=21,return_paths=36,response_cells=len(expected),untouched_mixture_cells=len(forecasts),return_cells=len(expected_return),
                source_scope='Spectra, source interpolation errors, size secants and fitted transport reconstructed from compact moments. Full-vocabulary origins are bound to the separate independent raw geometry check.')
