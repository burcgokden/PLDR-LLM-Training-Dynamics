#!/usr/bin/env python
"""Analyze selected source-resolution controls without a population-tail claim."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-study-20260905';out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    protocol=study/'protocols/source-precision.json'
    parents=json.loads(protocol.read_text())['parents']
    paths=[study/'source-precision'/name/'manifest.json' for name in parents]
    bind_run(out,[protocol,*[p for m in paths for p in [m,m.parent/'measurements.npz']]],vars(args))
    results=[];figure,axes=plt.subplots(2,2,figsize=(9,7),constrained_layout=True)
    for path,ax in zip(paths,axes.ravel(),strict=True):
        meta=json.loads(path.read_text());raw=np.load(path.parent/'measurements.npz')
        if meta['status']!='complete' or sha256(path.parent/'measurements.npz')!=meta['raw_sha256']:
            raise AssertionError('Precision evidence is incomplete or changed')
        eps=raw['amplitudes'];errors=raw['relative_errors'];radius=None
        resolved=raw['smooth64_curvature']>1e-28
        for index in np.argsort(eps):
            if not resolved.any() or np.max(errors[index,resolved])>.05:break
            radius=float(eps[index])
        original=int(np.flatnonzero(eps==.005)[0])
        result=dict(parent=meta['parent'],condition=meta['condition'],selected=raw['selected_source_indices'].tolist(),
            resolved_contexts=int(resolved.sum()),smooth64_curvature=raw['smooth64_curvature'].tolist(),native_gpu_curvature=raw['native_gpu_half_curvature'].tolist(),
            amplitudes=eps.tolist(),relative_errors=errors.tolist(),absolute_errors=raw['absolute_errors'].tolist(),
            original_radius_max_relative_error=float(errors[original,resolved].max()) if resolved.any() else None,grid_radius_5pct=radius,
            dtype_reference_max_logit_error=meta['dtype_reference_max_logit_error'],jvp_primal_max_logit_error=meta['jvp_primal_max_logit_error'],
            native_cpu_gpu_kl=raw['native_cpu_gpu_kl'].tolist(),smooth_native_cpu_kl=raw['smooth_native_cpu_kl'].tolist())
        results.append(result)
        order=np.argsort(eps)
        if resolved.any():
            plotted=errors[:,resolved]
            ax.plot(eps[order],np.median(plotted,axis=1)[order],'o-',label='Selected-context median')
            ax.fill_between(eps[order],plotted.min(1)[order],plotted.max(1)[order],alpha=.15,label='Selected-context range')
        else:
            ax.text(.5,.5,'No resolved source norm',ha='center',transform=ax.transAxes)
        ax.set_xlim(eps.min(),eps.max())
        ax.axhline(.05,color='.5',ls='--',lw=.7);ax.axvline(.005,color='.7',ls=':',lw=.7)
        ax.set_xscale('log');ax.set_yscale('symlog',linthresh=1e-12);ax.grid(alpha=.2)
        ax.set_xlabel('Source amplitude');ax.set_ylabel('Secant error / directional-derivative norm')
        ax.set_title(f"g = {meta['condition']['multiplier']:g}, seed {meta['condition']['seed']}")
    axes[0,0].legend(fontsize=7)
    figure.savefig(out/'gain-precision.pdf');plt.close(figure)
    write_json(out/'results.json',dict(schema='gain-precision-analysis-v1',runs=results,
        interpretation='Selected tails and fixed context controls; not a population tail-frequency estimate. The radius is the largest tested amplitude passing 5% error for every selected context with directional-derivative Fisher norm above 1e-14 and every smaller tested amplitude, not a bound over a continuous interval.'))
    write_json(out/'manifest.json',dict(results_sha256=sha256(out/'results.json'),binding_sha256=sha256(out/'binding.json'),figures={p.name:sha256(p) for p in out.glob('*.pdf')}))


if __name__=='__main__':main()
