#!/usr/bin/env python
"""Analyze paired restoration without equating a recovery time with criticality."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def response_curve(plus, minus, amplitude):
    secant=(plus-minus)/(2*amplitude)
    initial=secant[0]
    energy=float(np.mean(initial**2))
    absolute=np.sqrt(np.mean(secant**2,axis=tuple(range(1,secant.ndim))))
    resolved=energy>1e-14
    projection=np.mean(secant*initial[None],axis=tuple(range(1,secant.ndim)))/energy if resolved else None
    relative=absolute/np.sqrt(energy) if resolved else None
    return dict(initial_rms=np.sqrt(energy),absolute_rms=absolute.tolist(),resolved=resolved,
                projection=projection.tolist() if resolved else None,
                relative_norm=relative.tolist() if resolved else None),secant


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    paths=sorted((study/'runs').glob('restoration-*/manifest.json'))
    if len(paths)!=8:raise RuntimeError('All eight declared parent realizations are required')
    bind_run(out,[p for m in paths for p in [m,m.parent/'measurements.npz',m.parent/'sampling.npz']],vars(args))
    results=[]
    for path in paths:
        meta=json.loads(path.read_text())
        if meta['status']!='complete':raise RuntimeError('A numerical failure must be assessed before paired analysis')
        if sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Raw hash mismatch')
        raw=np.load(path.parent/'measurements.npz')
        if not np.array_equal(raw['base_fields'][0],raw['frozen_base_fields'][0]):
            raise AssertionError('Baseline initial states differ')
        fields={}
        for label,index in [('attention_entropy',0),('row_energy',2)]:
            get=lambda name:raw[name+'_heads'][...,index].mean(-1).astype(float)
            small,ds=response_curve(get('plus_small'),get('minus_small'),.05)
            normal,dn=response_curve(get('plus'),get('minus'),.1)
            frozen,df=response_curve(get('frozen_plus'),get('frozen_minus'),.1)
            disagreement=np.sqrt(np.mean((ds-dn)**2,axis=(1,2)))
            normal['amplitude_disagreement_rms']=disagreement.tolist()
            normal['amplitude_disagreement_relative']=(disagreement/np.maximum(np.sqrt(np.mean(ds**2,axis=(1,2))),1e-14)).tolist()
            fields[label]=dict(normal=normal,small=small,frozen=frozen)
        results.append(dict(run_id=path.parent.name,heads=meta['condition']['heads'],seed=meta['condition']['seed'],
                            steps=raw['steps'].tolist(),fields=fields,pulse_equivalence=meta['pulse_equivalence'],
                            paired_kl={name:raw[name+'_kl'].mean(1).tolist() for name in ['plus','minus','frozen_plus','frozen_minus']},
                            nll={name:raw[name+'_fields'][:,:,25].mean(1).tolist() for name in ['base','plus','minus','frozen_base','frozen_plus','frozen_minus']},
                            seconds=meta['seconds'],peak_cuda_gb=meta['peak_cuda_gb']))
    figure,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
    for col,heads in enumerate([2,14]):
        rows=[r for r in results if r['heads']==heads]
        if len(rows)!=4:raise AssertionError('Missing independent initializations')
        steps=np.array(rows[0]['steps'])
        for row,field in enumerate(['attention_entropy','row_energy']):
            ax=axes[row,col]
            for mode,color in [('normal','C0'),('frozen','C1')]:
                curves=[r['fields'][field][mode]['relative_norm'] for r in rows if r['fields'][field][mode]['resolved']]
                if curves:
                    curves=np.array(curves)
                    ax.plot(steps,np.median(curves,axis=0),color=color,label=mode)
                    ax.fill_between(steps,curves.min(0),curves.max(0),color=color,alpha=.15)
            ax.axhline(1,color='.6',lw=.5);ax.set_xscale('symlog',linthresh=4)
            ax.set_yscale('symlog',linthresh=1e-3);ax.set_xlabel('Updates after pulse');ax.set_ylabel('Norm of signed response / initial norm')
            ax.set_title(f'{field.replace("_"," ")}, width {64*heads}');ax.grid(alpha=.2)
    axes[0,0].legend()
    figure.savefig(out/'training-restoration.pdf');plt.close(figure)
    write_json(out/'results.json',dict(schema='restoration-analysis-v1',runs=results,
        definition='Signed response (U_plus-U_minus)/(2*amplitude), averaged over heads at each fixed context/layer; no recentering at later times.',
        normalization='Initial RMS on exactly the same context/layer coordinates; initial RMS <=1e-7 is unresolved.',
        uncertainty='Four independent initialization seeds. Figure band is the observed seed range, not a confidence interval.',
        interpretation='Finite pulse restoration and generator dependence; no fitted critical relaxation exponent or automatic identification of a critical manifold.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json'),figures={p.name:sha256(p) for p in out.glob('*.pdf')}))


if __name__=='__main__':
    main()
