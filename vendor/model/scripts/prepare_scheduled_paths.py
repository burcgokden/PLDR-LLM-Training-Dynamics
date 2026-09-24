#!/usr/bin/env python
"""Record all scheduled temporal windows and the information available at selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from model_rg.provenance import sha256, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--study', default='scheduled-training-20260908')
    args = p.parse_args()
    study = Path(args.root).resolve()/args.study
    repo = Path(__file__).resolve().parents[1]
    output = study/'protocols/scheduled-path-analysis-selection.json'
    if output.exists():
        raise FileExistsError(output)
    selection = study/'protocols/regime-training-selection.json'
    passive = study/'protocols/regime-shared-update-observation.json'
    jobs = json.loads(selection.read_text())['jobs']
    information = []
    for job in jobs:
        parent = study/'runs'/job['run_id']
        if parent.exists():
            record = dict(run_id=job['run_id'], completed=(parent/'manifest.json').exists())
            for name in ['binding.json', 'manifest.json']:
                if (parent/name).exists():
                    record[name+'_sha256'] = sha256(parent/name)
            log = study/'logs'/(job['run_id']+'.log')
            if log.exists():
                # The immutable snapshot describes what had been observed when
                # this analysis choice was made, not the eventual full log.
                record['available_log_tail'] = log.read_text().splitlines()[-5:]
            information.append(record)
    early = {}
    for path in sorted((study/'early-checkpoints').glob('*/measurements/*/manifest.json')):
        early[str(path)] = sha256(path)
    sources = ['scripts/analyze_scheduled_paths.py', 'scripts/analyze_scaling_updates.py',
               'src/model_rg/controlled.py', 'src/model_rg/provenance.py']
    write_json(output, dict(schema='scheduled-path-analysis-selection-v1', recorded_at=datetime.now(timezone.utc).isoformat(),
        run_ids=[j['run_id'] for j in jobs], horizon=262144,
        warmup_updates=dict(controlled=2000, reference1=2000, reference2=1000, subcritical1=6000, subcritical2=2000),
        floor_start=250000, phase_local_block_widths=[2048, 8192, 32768], regular_probe_spacing=64,
        phase_rule='Warmup [0,W), annealing [W,250000), and fixed floor [250000,262144). Each phase receives its complete finite-window summary and every full phase-start-aligned block at each declared width. A block never crosses a learning-rate phase boundary.',
        probe_rule='Use only the immutable regular absolute 64-update mesh for temporal covariances. Extra saved or observed phase endpoints remain in the whole-path crossing records but do not perturb the uniform temporal sample grid. Each window reports its actual first and last probe and sample count.',
        floor_rule='The entire 12144-update fixed-floor interval also has three equal, disjoint 4048-update subwindows. They retain all floor updates, including the remainder not covered by complete power-of-two windows.',
        fields=['shared_update_projection', 'shared_clipped_gradient_projection', 'common_centroid',
                'absolute_row_energy', 'total_metric_energy', 'row', 'attention', 'operator_rms',
                'prediction_entropy', 'nll', 'logit_projection'], thresholds=[.1, .01, .001],
        statistics='Finite window mean-square energy, mean-centered variance, linear trend, detrended variance, endpoint displacement and covariance at zero and power-of-two lags through one quarter of each window. Native projected gradients and updates use every update; emission coordinates use the regular probe mesh. Undefined zero-variance correlations retain null.',
        uncertainty='Each temporal window is one path observation; its time points are not independent training replicas. Four initialization identities remain the only random units for between-model comparisons at a conditioned recipe and head count.',
        development_scope='This temporal analysis was specified during the initial two scheduled trajectories. Their available log tails and every completed early-prefix result are recorded below. It is a complete diagnostic selection, not a claim that all choices preceded scheduled execution.',
        available_scientific_information=information, available_early_observation_sha256=early,
        inference='Prescribed warmup and cosine drift do not establish endogenous feedback. A finite fixed-floor window does not by itself establish stationarity, a diverging relaxation time, critical exponents or an avalanche process. All declared windows and thresholds are retained regardless of their measured behavior.',
        inputs_sha256={str(path): sha256(path) for path in [selection, passive]},
        producer_sources={name: sha256(repo/name) for name in sources}))
    print('Recorded all scheduled phase and fixed-floor temporal diagnostics', sha256(output), flush=True)


if __name__ == '__main__':
    main()
