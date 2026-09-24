#!/usr/bin/env python
"""Measure fixed-coordinate collective laws at explicitly listed frozen states."""
import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from model_rg.controlled import bind_run
from model_rg.criticality import predictive_kl
from model_rg.precision import preserve_response_dtype
from model_rg.provenance import sha256, write_json
from model_rg.scaling import observations
from model_rg.training import TrainingModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906')
    p.add_argument('--protocol', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    root = Path(a.root)
    study = root/a.study
    protocol = study/'protocols'/a.protocol
    spec = json.loads(protocol.read_text())
    label = Path(a.protocol).stem
    ledger = study/('launcher-'+label+'.json')
    if ledger.exists():
        raise FileExistsError(ledger)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1'
    probe = root/'controlled-study-20260905/data/short'
    rows = np.array(spec['rows'], dtype=int)
    calibration_rows = np.arange(64)
    pt = np.load(probe/'tokens.npy', mmap_mode='r')
    po = np.load(probe/'offsets.npy')
    crops = np.asarray(pt[rows[:, None], po[rows, None]+np.arange(65)])
    calibration = np.asarray(pt[calibration_rows[:, None], po[calibration_rows, None]+np.arange(65)])
    pairs = np.random.default_rng(640071).permutation(len(rows)).reshape(-1, 2)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    records = []
    started = time.time()
    write_json(ledger, dict(status='running', protocol_sha256=sha256(protocol), records=records))
    for case in spec['cases']:
        state = Path(case['state'])
        if not state.exists():
            raise FileNotFoundError(state)
        out = study/'measurements'/case['name']
        out.mkdir(parents=True, exist_ok=False)
        inputs = [protocol, state, probe/'tokens.npy', probe/'offsets.npy',
                  source/'modeling_pldrllm.py', source/'configuration_pldrllm.py']
        if case.get('parent_manifest'):
            parent_manifest = Path(case['parent_manifest'])
            meta = json.loads(parent_manifest.read_text())
            if meta['status'] != 'complete':
                raise AssertionError('The frozen parent is incomplete')
            inputs.append(parent_manifest)
        binding = bind_run(out, inputs, dict(**vars(a), case=case))
        if case.get('state_sha256') and binding['inputs'][str(state.resolve())] != case['state_sha256']:
            raise AssertionError('Frozen state hash changed')
        before = time.time()
        saved = torch.load(state, map_location='cpu', mmap=True, weights_only=True)
        condition = saved['arguments']
        for name in ['heads', 'seed', 'multiplier', 'shared_seed', 'stream_seed']:
            if name in case and condition[name] != case[name]:
                raise AssertionError('Frozen condition changed: '+name)
        if saved['step'] != case['step']:
            raise AssertionError('Frozen horizon changed')
        model = TrainingModel(source, condition['heads'], condition['seed'], a.device)
        model.model.load_state_dict(saved['model'])
        del saved
        model.model.eval().requires_grad_(False)
        raw = dict(rows=rows, crops=crops, calibration_rows=calibration_rows,
                   calibration_crops=calibration, context_pairs=pairs)
        precision_records = []
        for precision in spec.get('precisions', ['float32']):
            if precision == 'float64':
                model.model.double()
                preserve_response_dtype(model)
                if hasattr(model, '_common_logit_projection'):
                    model._common_logit_projection = model._common_logit_projection.double()
            elif next(model.model.parameters()).dtype != torch.float32:
                raise ValueError('Float32 must precede float64 in a paired arithmetic protocol')
            fields, heads, centroids, energies, logits, mean_matrices = [], [], [], [], [], []
            evaluation_mean_matrices = []
            with torch.no_grad():
                for begin in range(0, len(rows), spec.get('batch_size', 32)):
                    batch = torch.tensor(crops[begin:begin+spec.get('batch_size',32)], dtype=torch.long, device=a.device)
                    f, h, z, _, extra = observations(model, batch[:, :64], batch[:, 64], matrices=spec.get('evaluation_matrices', False))
                    if spec.get('evaluation_matrices', False):
                        evaluation_mean_matrices.append(extra['mean_matrices'].cpu().numpy())
                    for value in [f,h,z,*extra.values()]:
                        if not torch.isfinite(value).all():
                            raise FloatingPointError('Nonfinite frozen observation')
                    fields.append(f.cpu().numpy())
                    heads.append(h.cpu().numpy())
                    centroids.append(extra['centroids'].cpu().numpy())
                    energies.append(extra['energies'].cpu().numpy())
                    logits.append(z.double().cpu())
                for begin in range(0, 64, spec.get('batch_size',32)):
                    batch = torch.tensor(calibration[begin:begin+spec.get('batch_size',32)], dtype=torch.long, device=a.device)
                    _, _, _, _, extra = observations(model, batch[:,:64], batch[:,64], matrices=True)
                    mean_matrices.append(extra['mean_matrices'].cpu().numpy())
            z = torch.cat(logits)
            left,right = pairs.T
            kl = .5*(predictive_kl(z[left], z[right])+predictive_kl(z[right], z[left]))
            for name,value in dict(fields=np.concatenate(fields),heads=np.concatenate(heads),
                centroids=np.concatenate(centroids),energies=np.concatenate(energies),
                mean_metric=np.concatenate(mean_matrices),context_kl=kl.numpy()).items():
                raw[precision+'_'+name] = value
            if evaluation_mean_matrices:
                raw[precision+'_evaluation_mean_metric'] = np.concatenate(evaluation_mean_matrices)
            precision_records.append(dict(precision=precision, mean_context_kl=float(kl.mean())))
            if len(spec.get('precisions', ['float32'])) > 1:
                if precision == 'float32':
                    reference_logits = z
                else:
                    raw['paired_predictive_kl'] = predictive_kl(reference_logits,z).numpy()
            del logits,z
        np.savez_compressed(out/'measurements.npz', **raw)
        record = dict(schema='frozen-size-time-observations-v1', status='complete', case=case,
                      condition=condition, precision_records=precision_records, seconds=time.time()-before,
                      device=a.device, batch_size=spec.get('batch_size',32),
                      evaluation_matrix_contexts=len(rows) if spec.get('evaluation_matrices', False) else 0,
                      accumulation='float64 matrix reductions', binding_sha256=sha256(out/'binding.json'),
                      raw_sha256=sha256(out/'measurements.npz'))
        write_json(out/'manifest.json',record)
        records.append(dict(name=case['name'],status='complete',manifest_sha256=sha256(out/'manifest.json')))
        write_json(ledger,dict(status='running',protocol_sha256=sha256(protocol),records=records))
        print(case['name'],'complete',round(record['seconds'],1),flush=True)
        del model,raw
        gc.collect()
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
    write_json(ledger,dict(status='complete',protocol_sha256=sha256(protocol),records=records,seconds=time.time()-started))


if __name__ == '__main__':
    main()
