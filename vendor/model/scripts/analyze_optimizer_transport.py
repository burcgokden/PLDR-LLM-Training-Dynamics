#!/usr/bin/env python3
"""Summarize every frozen optimizer intervention; no selection on outcomes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
from model_rg.finite_optimizer import pulse_parts


def rms(x): return float(np.sqrt(np.mean(np.asarray(x)**2)))


def analyze(study):
    study = Path(study); spec = json.loads((study/'protocol.json').read_text())
    rows = []; predictions = []; costs = []; compact = {}
    for case in spec['cases']:
        result = json.loads((study/case['name']/'results.json').read_text())
        costs.append(dict(case=case['name'], seconds=result['seconds']))
        for rec in result['records']:
            predictions.append(dict(case=case['name'], mode=rec['mode'], source=rec['source'], arm=rec['arm'], **rec['full_prediction']))
            with np.load(rec['file']) as f:
                compact[case['name']+'/'+rec['mode']+'/'+str(rec['source'])+'/'+rec['arm']] = f['q'].tolist()
        for mode in spec['modes']:
            for source in range(spec['replicates']):
                key = case['name']+'/'+mode+'/'+str(source)+'/'
                zero = np.array(compact[key+'zero'])
                for family in ['first', 'logsecond']:
                    even, odd = pulse_parts(zero, compact[key+family+'-plus'], compact[key+family+'-minus'])
                    eh, oh = pulse_parts(zero, compact[key+family+'-halfplus'], compact[key+family+'-halfminus'])
                    signal = rms(oh)
                    d = rms(odd-2*oh)/rms(2*oh) if rms(2*oh) else None
                    rows.append(dict(case=case['name'], heads=case['heads'], mode=mode, source=source, family=family,
                        incoming_odd_rms=rms(odd[0]), incoming_even_rms=rms(even[0]),
                        first_odd_rms=rms(odd[1]), first_even_rms=rms(even[1]),
                        endpoint_odd_rms=rms(odd[-1]), endpoint_even_rms=rms(even[-1]),
                        path_odd_rms=rms(odd), path_even_rms=rms(even), half_signal=signal,
                        halving_discrepancy=d, resolved=signal>spec['signal_floor'],
                        signed_endpoint_risk=[float(np.array(compact[key+family+'-'+tag])[-1,0]-zero[-1,0])
                                              for tag in ['plus','minus','halfplus','halfminus']]))
    npaths = len(spec['cases'])*spec['replicates']*len(spec['arms'])
    return dict(schema='optimizer-transport-results-v1', stage=spec['stage'], rows=rows,
        predictions=predictions, costs=costs, native_paths=npaths, native_updates=npaths*spec['horizon'],
        arithmetic_paths=npaths, arithmetic_updates=npaths*spec['horizon'],
        replay_updates=len(spec['cases'])*len(spec['modes'])*spec['horizon'],
        max_full_prediction_units=np.max([r['max_roundoff_units'] for r in predictions],axis=0).tolist(),
        compact=dict(times=spec['times'], paths=compact),
        scope='Two paired widths at one corpus/full-initialization identity; conditional source sequences, coupled signs and arithmetic controls. Full prediction residuals are producer reductions; independent saved-coordinate checks are separately recorded.')


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--study',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); output=Path(a.output)
    if output.exists(): raise FileExistsError(output)
    output.write_text(json.dumps(analyze(a.study),indent=2,allow_nan=False)+'\n')
