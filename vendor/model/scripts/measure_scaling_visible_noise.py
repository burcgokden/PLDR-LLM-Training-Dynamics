#!/usr/bin/env python
"""Freeze Fisher source subspaces, then validate their adjoint predictions on fresh batches."""
import argparse
import copy
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp
import torch

from measure_scaling_noise import analytical_emission
from model_rg.controlled import bind_run
from model_rg.criticality import generator_parameter, optimizer_for, predictive_kl, sample_batches
from model_rg.precision import preserve_response_dtype, expand_smooth_response
from model_rg.provenance import sha256, write_json
from model_rg.training import TrainingModel
from model_rg.visible_noise import fisher_image, fisher_adjoint, visible_basis, projection_statistics


def main():
    p = argparse.ArgumentParser();p.add_argument('--root', required=True)
    p.add_argument('--study', default='critical-scaling-20260906');p.add_argument('--protocol', required=True)
    p.add_argument('--case', required=True);p.add_argument('--threads', type=int, default=4)
    a = p.parse_args();root = Path(a.root);study = root/a.study
    protocol = study/'protocols'/a.protocol;spec = json.loads(protocol.read_text())
    case = next(c for c in spec['cases'] if c['name'] == a.case)
    calibration = Path(case['calibration']);meta = json.loads((calibration/'manifest.json').read_text())
    out = study/'measurements'/case['name'];out.mkdir(parents=True, exist_ok=False)
    source = root/'assets/PLDR-LLM-v51-SOC-110M-1';probe = root/'controlled-study-20260905/data/short'
    inputs = [protocol, calibration/'manifest.json', calibration/'results.json', calibration/'measurements.npz',
              Path(case['parent']), Path(case['parent_manifest']), root/'data/refinedweb-4608/tokens.npy',
              probe/'tokens.npy', probe/'offsets.npy', source/'modeling_pldrllm.py', source/'configuration_pldrllm.py']
    bind_run(out, inputs, vars(a))
    if meta['status'] != 'complete' or sha256(calibration/'measurements.npz') != meta['raw_sha256']:
        raise AssertionError('Frozen calibration changed')
    if sha256(case['parent']) != case['parent_sha256']:
        raise AssertionError('Incoming optimizer state changed')
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False;torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    parent = torch.load(case['parent'], map_location='cpu', mmap=True, weights_only=True)
    condition = parent['arguments']
    if (condition['heads'], condition['seed'], parent['step']) != (case['heads'], case['seed'], case['step']):
        raise AssertionError('Conditional source identity changed')
    with np.load(calibration/'measurements.npz') as z:
        base = z['base_logits']
        probability = np.exp(base-logsumexp(base, axis=-1, keepdims=True))
        images = np.stack([fisher_image(z[f'b{k}_jvp_logits'], probability) for k in range(8)])
    basis, singular = visible_basis(images, spec['basis_singular_tolerance'])
    if not len(basis):raise AssertionError('A nonzero calibrated predictive response is required')
    tokens, offsets = np.load(probe/'tokens.npy'), np.load(probe/'offsets.npy')
    selected = np.arange(512, 528)
    crops = tokens[selected[:, None], offsets[selected, None]+np.arange(65)]
    response = TrainingModel(source, case['heads'], case['seed'], 'cpu')
    response.model.load_state_dict(parent['model']);response.model.double()
    preserve_response_dtype(response);expand_smooth_response(response)
    response.model.eval().requires_grad_(False)
    named_response = [(name, p) for name, p in response.model.named_parameters() if generator_parameter(name)]
    for _, parameter in named_response:parameter.requires_grad_(True)
    logits = response.logits(torch.tensor(crops[:, :64], dtype=torch.long))
    np.testing.assert_allclose(logits.detach().numpy(), base, rtol=1e-10, atol=1e-10)
    adjoints = []
    for k, direction in enumerate(basis):
        weight = torch.from_numpy(fisher_adjoint(direction.reshape(base.shape), probability))
        scalar = (logits*weight).sum()
        gradient = torch.autograd.grad(scalar, [p for _, p in named_response], retain_graph=k+1 < len(basis))
        adjoints.append(np.concatenate([g.detach().numpy().reshape(-1) for g in gradient]))
    adjoints = np.stack(adjoints)
    del logits, scalar, gradient, weight
    response.model.requires_grad_(False)
    names = [name for name, _ in named_response]
    slices = [];begin = 0
    for _, value in named_response:
        slices.append([begin, begin+value.numel()]);begin += value.numel()
    np.savez_compressed(out/'predictor.npz', basis=basis, adjoints=adjoints, probability=probability,
                        base_logits=base, calibration_singular_values=singular)
    predictor = dict(status='frozen_before_validation_batches', frozen_at=datetime.now(timezone.utc).isoformat(),
        predictor_sha256=sha256(out/'predictor.npz'), parameter_names=names, parameter_slices=slices,
        basis_dimension=len(basis), calibration_batches=8, requested_dimensions=spec['dimensions'],
        construction='Uncentered right singular vectors of the eight fixed full-vocabulary Fisher tangents. Each source coordinate is the full-graph Euclidean adjoint of its fixed Fisher direction. No target batch or target tangent enters this predictor.')
    write_json(out/'predictor.json', predictor)
    gc.collect()
    # The validation sample is generated only after the complete predictor is fixed.
    rows, offsets, batches = sample_batches(np.load(root/'data/refinedweb-4608/tokens.npy'),
                                           spec['validation_batch_seed'], spec['validation_batches'])
    native = TrainingModel(source, case['heads'], case['seed'], 'cpu')
    native.model.load_state_dict(parent['model'])
    optimizer = optimizer_for(native, condition['multiplier'])
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
    named = [(name, p) for name, p in native.model.named_parameters() if generator_parameter(name)]
    if [name for name, _ in named] != names:raise AssertionError('Source parameter order changed')
    rate = optimizer.param_groups[0]['lr']
    bp = {name:parent['model'][name].double() for name in names}
    raw = dict(rows=rows, offsets=offsets, cohort=selected, base_logits=base)
    velocities = np.lib.format.open_memmap(out/'validation-directions.npy', mode='w+', dtype=np.float64,
                                          shape=(spec['validation_batches'], begin))
    teacher, coefficients, numerical, losses, norms = [], [], [], [], []
    native_control = None
    zero, one = torch.tensor(0., dtype=torch.float64), torch.tensor(1., dtype=torch.float64)
    for k, batch in enumerate(batches):
        native.model.train();optimizer.zero_grad(set_to_none=True)
        batch = torch.tensor(batch, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(native.logits(batch[:, :64]), batch[:, 64]);loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(native.model.parameters(), 1., error_if_nonfinite=True)
        losses.append(float(loss.detach()));norms.append(float(norm))
        directions = {}
        for (name, parameter), (lo, hi) in zip(named, slices, strict=True):
            state = optimizer.state[parameter];counter = int(state['step'])+1
            g = parameter.grad.detach().double()
            m = .9*state['exp_avg'].double()+.1*g
            v = .95*state['exp_avg_sq'].double()+.05*g*g
            direction = rate*(-.01*parameter.detach().double()-(m/(1-.9**counter))/(torch.sqrt(v/(1-.95**counter))+1e-8))
            directions[name] = direction;velocities[k, lo:hi] = direction.numpy().reshape(-1)
        if k == 0:
            expected = [p.detach().double()+directions[name] for name, p in named]
            optimizer.step()
            numerator = sum(float((p.detach().double()-x).square().sum()) for (_, p), x in zip(named, expected, strict=True))
            denominator = sum(float(p.detach().double().square().sum()) for _, p in named)
            native_control = dict(relative_parameter_error=float(np.sqrt(numerator/denominator)))
            native.model.load_state_dict(parent['model']);optimizer.load_state_dict(copy.deepcopy(parent['optimizer']))
            del expected
        def emission(alpha):return analytical_emission(response, crops, bp, directions, alpha)[2]
        with torch.no_grad():
            primal, tangent = torch.func.jvp(emission, (zero,), (one,))
        np.testing.assert_allclose(primal.numpy(), base, rtol=1e-10, atol=1e-10)
        weighted = fisher_image(tangent.numpy(), probability).reshape(-1)
        prediction_coefficients = adjoints@np.asarray(velocities[k])
        direct_coefficients = basis@weighted
        dual_error = float(np.linalg.norm(prediction_coefficients-direct_coefficients))
        dual_scale = float(np.linalg.norm(direct_coefficients))
        passed = dual_error <= spec['duality_absolute_tolerance']+spec['duality_relative_tolerance']*dual_scale
        record = dict(batch=k, duality_error=dual_error, duality_reference_norm=dual_scale, duality_passed=bool(passed), finite_responses=[])
        if not passed:raise AssertionError('The frozen full-graph adjoint failed a fresh direction')
        for amplitude in spec['amplitudes']:
            with torch.no_grad():
                plus = emission(torch.tensor(amplitude, dtype=torch.float64))
                minus = emission(torch.tensor(-amplitude, dtype=torch.float64))
            secant = fisher_image(((plus-minus)/(2*amplitude)).numpy(), probability).reshape(-1)
            kl = float((predictive_kl(primal, plus)+predictive_kl(primal, minus)).mean()/amplitude**2)
            curvature = float(weighted@weighted)
            first_error = float(np.linalg.norm(secant-weighted)/max(np.linalg.norm(weighted), 1e-300))
            curvature_error = abs(kl-curvature)/max(curvature, 1e-300)
            record['finite_responses'].append(dict(amplitude=amplitude, first_relative_error=first_error,
                fisher_relative_error=curvature_error, local_diagnostic=bool(first_error<=.01 and curvature_error<=.05)))
            raw[f'b{k}_a{amplitude}_secant'] = secant
            raw[f'b{k}_a{amplitude}_symmetric_kl'] = np.array(kl)
        teacher.append(weighted);coefficients.append(prediction_coefficients);numerical.append(record)
        print(case['name'], 'fresh batch', k, 'duality error', dual_error, flush=True)
    velocities.flush();del velocities
    teacher, coefficients = np.stack(teacher), np.stack(coefficients)
    raw.update(teacher=teacher, source_coefficients=coefficients, losses=np.array(losses), gradient_norms=np.array(norms))
    if any(not np.isfinite(value).all() for value in raw.values()):raise AssertionError('Nonfinite validation record')
    projections = [projection_statistics(teacher, coefficients, basis, k) for k in spec['dimensions']]
    for r in projections:
        # Fixed orthogonal emission subspaces split the held-out covariance trace.
        np.testing.assert_allclose(r['covariance_trace'], r['predicted_covariance_trace']+r['residual_covariance_trace'], rtol=2e-6, atol=1e-15)
    result = dict(schema='frozen-visible-noise-validation-v1', status='complete', case=case,
        predictor=predictor, validation_batch_seed=spec['validation_batch_seed'], validation_batches=spec['validation_batches'],
        projections=projections, numerical=numerical, native_update_control=native_control,
        seconds=time.time()-started,
        scope='Fresh-batch validation conditional on the entire frozen incoming state and the 16 fixed contexts. The generator-only local predictor uses frozen full-graph adjoint coordinates. It is not an autonomous reduced training law or a population rank estimate. All declared dimensions and numerical magnitudes are retained.')
    np.savez_compressed(out/'measurements.npz', **raw);write_json(out/'results.json', result)
    write_json(out/'manifest.json', dict(schema=result['schema'], status='complete', case=case,
        binding_sha256=sha256(out/'binding.json'), predictor_sha256=sha256(out/'predictor.npz'),
        predictor_record_sha256=sha256(out/'predictor.json'), raw_sha256=sha256(out/'measurements.npz'),
        directions_sha256=sha256(out/'validation-directions.npy'), results_sha256=sha256(out/'results.json')))


if __name__ == '__main__':main()
