"""Independent reconstruction of the portable finite-response evidence."""
import json
from pathlib import Path
import numpy as np

def require(flag,message):
    if not flag:raise ValueError(message)

def verify(source):
    source=Path(source);g=source/'generated'
    spec=json.loads((g/'numerical-response-protocol.json').read_text())
    result=json.loads((g/'numerical-response-results.json').read_text())
    require(spec['times']==[0,1,2,4,8] and spec['horizon']==8 and spec['source_replicates']==4,'Changed observation contract')
    with np.load(g/'numerical-response-paths.npz',allow_pickle=False) as f:
        expected={'times'}|{c['name']+'__'+mode+'__'+arm['name'] for c in spec['cases'] for mode in spec['modes'] for arm in spec['arms']}
        require(set(f.files)==expected and np.array_equal(f['times'],spec['times']),'Incomplete compact pulse inventory')
        for key in expected - {'times'}:
            require(f[key].shape == (4, 5, 12), 'Wrong source/time/coordinate shape')
        expected_cells={(case['name'],mode,family,source,time)
                        for case in spec['cases'] for mode in spec['modes']
                        for family in ['generator','body'] for source in range(4)
                        for time in spec['times']}
        require({(r['case'],r['mode'],r['family'],r['source'],r['horizon'])
                 for r in result['prefixes']}==expected_cells,'Incomplete prefix design')
        for r in result['prefixes']:
            key=r['case']+'__'+r['mode']+'__'+r['family']+'-3e-07-';index=spec['times'].index(r['horizon'])+1
            arrays={tag:f[key+tag][r['source'],:index] for tag in ['plus','minus','halfplus','halfminus']}
            full=(arrays['plus']-arrays['minus'])/2;half=(arrays['halfplus']-arrays['halfminus'])/2
            signal=np.sqrt(np.mean(half**2));discrepancy=np.linalg.norm(full-2*half)/np.linalg.norm(2*half)
            require(abs(discrepancy-r['discrepancy'])<1e-12 and abs(signal-r['half_signal_rms'])<1e-12,'Changed pulse outcome')
            require(r['passes']==bool(signal>5e-7 and discrepancy<=.1),'Wrong pulse disposition')
        require(len(result['prefixes'])==160 and len(result['primary'])==32,'Incomplete response cells')
        require([r for r in result['prefixes'] if r['horizon']==8]==result['primary'],'Primary differs from full path')
    require({(r['case'],r['source']) for r in result['invisible']}==
            {(c['name'],i) for c in spec['cases'] for i in range(2)}
            and len(result['invisible'])==4,'Incomplete null-sector design')
    support=json.loads((g/'invisible-sector-support.json').read_text())
    require(support['status']=='passed' and support['token']==0 and
            support['corpus_occurrences']==support['evaluation_occurrences']==0,
            'Wrong finite input-support result')
    require((support['corpus_input_positions'],support['evaluation_input_positions'])==
            (33554432,2048),'Wrong finite support scope')
    with np.load(g/'invisible-sector-paths.npz',allow_pickle=False) as f:
        for r in result['invisible']:
            key=r['case']+'__'+str(r['source'])+'__'
            require(np.array_equal(f[key+'zero__q'],f[key+'pulse__q']),'Changed null-sector observations')
            for arm in ['zero','pulse']:
                rows=f[key+arm+'__embedding_rows'];pred=f[key+arm+'__predicted_rows']
                require(np.array_equal(rows,pred),'Changed pre-update forecast')
                require(all(np.array_equal(rows[t]*np.float32(r['decay_factor']),rows[t+1]) for t in range(8)),'Changed null-row update law')
            initial=np.linalg.norm(f[key+'pulse__embedding_rows'][0]-f[key+'zero__embedding_rows'][0])
            require(initial>0 and abs(float(initial)-r['initial_displacement_norm'])<1e-12,'Missing nonzero null pulse')
    return dict(response_primary_cells=32,response_prefix_cells=160,invisible_comparisons=4)
