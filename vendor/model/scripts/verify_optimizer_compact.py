"""Reconstruct all published optimizer-response coordinates without raw models."""
import json
from numerical_validation import load_json_strict, discrepancy, array_discrepancy, finite_array, finite_scalar
from pathlib import Path
import numpy as np


def verify(source):
    g=Path(source)/'generated'
    result=load_json_strict((g/'optimizer-transport-analysis.json').read_text())
    spec=load_json_strict((g/'optimizer-transport-protocol.json').read_text())
    paths=result['compact']['paths']
    expected={c['name']+'/'+m+'/'+str(s)+'/'+a['name'] for c in spec['cases'] for m in spec['modes']
              for s in range(2) for a in spec['arms']}
    expected|={c['name']+'/'+m+'/0/zero-replay' for c in spec['cases'] for m in spec['modes']}
    if set(paths)!=expected or result['compact']['times']!=[0,1,2,4,8]: raise ValueError('Incomplete compact optimizer paths')
    expectedrows={(c['name'],m,s,f) for c in spec['cases'] for m in spec['modes'] for s in range(2) for f in ['first','logsecond']}
    if {(r['case'],r['mode'],r['source'],r['family']) for r in result['rows']}!=expectedrows or len(result['rows'])!=16:
        raise ValueError('Missing optimizer outcome cell')
    for key, value in paths.items():
        finite_array(value, 'optimizer path '+key)
    norm=lambda x:float(np.sqrt(np.mean(np.asarray(x)**2)))
    for r in result['rows']:
        prefix=r['case']+'/'+r['mode']+'/'+str(r['source'])+'/'
        x=np.array(paths[prefix+'zero']);p=np.array(paths[prefix+r['family']+'-plus']);m=np.array(paths[prefix+r['family']+'-minus'])
        odd=(p-m)/2;even=(p+m)/2-x
        half=(np.array(paths[prefix+r['family']+'-halfplus'])-np.array(paths[prefix+r['family']+'-halfminus']))/2
        values=dict(incoming_odd_rms=norm(odd[0]),incoming_even_rms=norm(even[0]),
            first_odd_rms=norm(odd[1]),first_even_rms=norm(even[1]),endpoint_odd_rms=norm(odd[-1]),
            endpoint_even_rms=norm(even[-1]),path_odd_rms=norm(odd),path_even_rms=norm(even),
            half_signal=norm(half),halving_discrepancy=norm(odd-2*half)/norm(2*half))
        for key, value in values.items():
            discrepancy(value, r[key], 'optimizer '+key)
        if r['resolved']!=(norm(half)>spec['signal_floor']): raise ValueError('Changed signal decision')
        risks=[float(np.array(paths[prefix+r['family']+'-'+tag])[-1,0]-x[-1,0]) for tag in ['plus','minus','halfplus','halfminus']]
        array_discrepancy(risks, r['signed_endpoint_risk'], 'signed risk')
        if r['source']==0 and not np.array_equal(x,np.array(paths[prefix+'zero-replay'])):raise ValueError('Compact replay differs')
    return dict(paths=72,replays=4,complete_cells=16,all_signs_and_amplitudes_retained=True)
