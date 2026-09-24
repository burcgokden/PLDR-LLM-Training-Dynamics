#!/usr/bin/env python
"""Resolve common-row and contrast collectives in the recorded native metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from model_rg.controlled import bind_run
from model_rg.provenance import sha256, write_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    study=Path(args.root)/'criticality-dynamics-20260906';protocol=study/'protocols/metric-collectives.json';spec=json.loads(protocol.read_text())
    gradient_protocol=study/'protocols/row-gradient-projection.json';ledger=study/'row-gradient-projection.json';execution=json.loads(ledger.read_text())
    if execution['status']!='complete' or sha256(gradient_protocol)!=spec['source_protocol_sha256']:raise AssertionError('Incomplete or changed metric sources')
    groups=defaultdict(list);inputs=[protocol,gradient_protocol,ledger]
    for case in spec['cases']:
        folder=study/'analysis'/case['name'];meta=json.loads((folder/'manifest.json').read_text())
        if meta['status']!='complete' or sha256(folder/'measurements.npz')!=meta['raw_sha256']:raise AssertionError('Changed native metric input')
        inputs.extend([folder/'manifest.json',folder/'measurements.npz']);groups[case['heads'],case['step']].append(case)
    if set(groups)!={(n,t) for n in [4,8,14] for t in [2048,8192]}:raise AssertionError('Metric condition inventory changed')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);bind_run(out,inputs,vars(args));results=[];arrays={}
    for (heads,step),cases in sorted(groups.items()):
        lookup={(c['seed'],c['batch_step']):c for c in cases};seeds=list(range(640101,640105));batches=[2048,8192]
        if len(cases)!=8 or set(lookup)!={(s,b) for s in seeds for b in batches}:raise AssertionError('Unbalanced metric input law')
        matrices=[];energies=[];centroid_squares=[];matrix_squares=[];cohort=None;case_names=[]
        for seed in seeds:
            by_batch=[];energy_batch=[];centroid_batch=[];square_batch=[];crops=[]
            for batch in batches:
                case=lookup[seed,batch];case_names.append(case['name']);raw=np.load(study/'analysis'/case['name']/'measurements.npz')
                a=np.stack([raw[f'base_L{layer}_terminal_rows'] for layer in range(5)],axis=1).astype(float)
                if a.shape!=(32,5,heads,64,64):raise AssertionError('Metric row coordinates changed')
                mu=a.mean(-2,keepdims=True);contrast=a-mu
                by_batch.append(a.mean(2));energy_batch.append((contrast*contrast).mean((-1,-2)))
                centroid_batch.append((mu*mu).mean((-1,-2)));square_batch.append((a*a).mean((-1,-2)))
                crops.append(raw['crops']);raw.close()
            matrices.append(np.concatenate(by_batch));energies.append(np.concatenate(energy_batch));centroid_squares.append(np.concatenate(centroid_batch));matrix_squares.append(np.concatenate(square_batch))
            current=np.concatenate(crops)
            if cohort is None:cohort=current
            else:np.testing.assert_array_equal(current,cohort)
        a=np.stack(matrices);e=np.stack(energies);c2=np.stack(centroid_squares);a2=np.stack(matrix_squares)
        np.testing.assert_allclose(a2,e+c2,rtol=1e-12,atol=1e-15)
        delta=a-a.mean(0,keepdims=True);common=delta.mean(-2,keepdims=True);normal=delta-common
        factor=heads/(3*5*4096)
        total_by_context=factor*(delta*delta).sum((0,2,3,4))
        normal_by_context=factor*(normal*normal).sum((0,2,3,4))
        common_by_context=factor*64*(common*common).sum((0,2,3,4))
        np.testing.assert_allclose(total_by_context,normal_by_context+common_by_context,rtol=1e-12,atol=1e-18)
        mu=a.mean(-2);flat=(mu-mu.mean(0,keepdims=True)).reshape(4*64,320)
        covariance=heads*(flat.T@flat)/(3*64*320);spectrum=np.linalg.eigvalsh(covariance)
        total=float(total_by_context.mean());common_trace=float(common_by_context.mean());normal_trace=float(normal_by_context.mean())
        np.testing.assert_allclose(np.trace(covariance),common_trace,rtol=1e-12,atol=1e-18)
        qe=e.mean(-1);de=qe-qe.mean(0,keepdims=True);chi_e=heads*np.mean(np.sum(de*de,axis=0)/3)
        key=f'N{heads}_t{step}'
        for label,value in dict(mean_head_matrix=a,head_row_energy=e,head_centroid_square=c2,head_matrix_square=a2,crops=cohort,
                                common_covariance=covariance,total_by_context=total_by_context,common_by_context=common_by_context,normal_by_context=normal_by_context).items():arrays[key+'_'+label]=value
        result=dict(heads=heads,step=step,seeds=seeds,batch_steps=batches,source_cases=case_names,
            mean_absolute_row_energy=float(e.mean()*4096),mean_row_energy_per_entry=float(e.mean()),mean_centroid_square=float(c2.mean()),mean_matrix_square=float(a2.mean()),
            metric_susceptibility=total,common_row_susceptibility=common_trace,contrast_susceptibility=normal_trace,common_fraction=common_trace/total if total>0 else None,
            row_energy_susceptibility=float(chi_e),common_covariance_eigenvalues=spectrum.tolist(),
            common_effective_rank=float(common_trace**2/np.sum(covariance*covariance)) if common_trace>0 else None,
            common_leading_fraction=float(spectrum[-1]/common_trace) if common_trace>0 else None)
        results.append(result);print('Metric collective',heads,step,'common fraction',result['common_fraction'],'chi',common_trace,flush=True)
    np.savez_compressed(out/'measurements.npz',**arrays)
    write_json(out/'results.json',dict(schema='native-metric-collectives-v1',conditions=results,normalization=spec['normalization'],interpretation=spec['interpretation']))
    write_json(out/'manifest.json',dict(status='complete',binding_sha256=sha256(out/'binding.json'),raw_sha256=sha256(out/'measurements.npz'),results_sha256=sha256(out/'results.json')))


if __name__=='__main__':main()
