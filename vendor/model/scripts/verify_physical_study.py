#!/usr/bin/env python
"""Verify complete native physical experiments and independently recompute outcomes."""
import argparse
import json
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
from pathlib import Path
import sys
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json
from verify_physical_data import statistics
from physical_verification_design import check as check_canonical_physical_design


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',required=True);a=ap.parse_args()
    base=Path(a.study);output=base/'verification.json'
    if output.exists():raise FileExistsError(output)
    checked={}
    def check(path,expected=None):
        path=Path(path).resolve();key=str(path)
        if key not in checked:checked[key]=sha256(path)
        if expected is not None and checked[key]!=expected:raise ValueError('Changed physical evidence: '+key)
        return checked[key]
    def read(path):check(path);return load_json_strict(Path(path).read_text())
    def sources(spec,folder):
        for n,h in spec['sources'].items():check(REPO/n,h);check(folder/'executed-source'/n,h)
    data=read(base/'data-verification.json')
    if data['status']!='passed':raise ValueError('Physical data verification missing')
    check(REPO/'scripts/verify_physical_data.py',data['verifier_sha256'])
    for p,h in data['checked_sha256'].items():check(p,h)
    primary=read(base/'confirmation/protocol.json')
    check_canonical_physical_design(primary,archived=primary.get('schema')=='physical-native-study-v1')
    sources(primary,base/'confirmation')
    assessment=read(base/'assessment/protocol.json');sources(assessment,base/'assessment')
    mechanisms=read(base/'mechanisms/protocol.json');sources(mechanisms,base/'mechanisms')
    collective=read(base/'collective-validation/protocol.json');sources(collective,base/'collective-validation')
    for h in [4,8]:
        qualification=read(base/f'qualification-h{h}.json')
        if qualification['status']!='passed':raise ValueError('Native prefix qualification failed')
        for n,digest in qualification['sources'].items():check(REPO/n,digest)
    draws=np.load(base/'confirmation/draws.npz');check(base/'confirmation/draws.npz',primary['draws_sha256'])
    replay=read(base/'replay/verification.json')
    if replay['status']!='passed' or replay['updates']!=2048 or len(replay['results'])!=4:raise ValueError('Native replay incomplete')
    check(REPO/'scripts/verify_physical_replay.py',replay['verifier_sha256'])
    analysis=read(base/'model-analysis/analysis.json')
    if analysis['status']!='complete':raise ValueError('Model analysis incomplete')
    check(REPO/'scripts/analyze_physical_models.py',analysis['analyzer_sha256'])
    for p,h in analysis['checked_sha256'].items():check(p,h)
    exact_cells=0;generated_cells=0;generation_count=0;scientific_updates=0;max_summary_error=0.
    summaries={(r['case'],r['step'],r['q'],r['L'],r['ratio']):r for r in analysis['rows']}
    for case in primary['cases']:
        check(case['state'],case['state_sha256'])
        folder=base/'confirmation'/case['name'];manifest=read(folder/'manifest.json')
        if manifest['status']!='complete' or manifest['completed_updates']!=16384 or manifest['consumed_configurations']!=524288:
            raise ValueError('Incomplete primary trajectory')
        check(base/'confirmation/protocol.json',manifest['protocol_sha256'])
        for n,h in manifest['files'].items():check(folder/n,h)
        trace=np.load(folder/'training-trace.npy')
        if trace.shape!=(16384,6) or not np.isfinite(trace).all():raise ValueError('Invalid primary trace')
        if not np.array_equal(trace[:,0],np.arange(1,16385)) or not np.array_equal(trace[:,1],draws['cell']) or not np.array_equal(trace[:,2],draws['site']):
            raise ValueError('Training trace differs from frozen identities')
        scientific_updates+=16384
        results=read(base/'assessment'/case['name']/'manifest.json')
        if results['status']!='complete' or [c['step'] for c in results['checkpoints']]!=[512,2048,16384]:raise ValueError('Assessment incomplete')
        for checkpoint in results['checkpoints']:
            check(checkpoint['results'],checkpoint['sha256']);r=read(checkpoint['results'])
            expected_cells=50 if checkpoint['step']==16384 else 10
            if len(r['generated'])!=expected_cells or len(r['observations'])!=expected_cells:raise ValueError('Missing assessment cell')
            for c in r['generated']:
                check(c['path'],c['sha256']);x=np.load(c['path']);o=statistics(x,c['q'])
                z=summaries[(case['name'],checkpoint['step'],c['q'],c['L'],c['ratio'])]
                independent=[o['m'].mean(),c['L']**2*o['m2'].mean(),o['m4'].mean()/o['m2'].mean()**2]
                displayed=[z['m'],z['chi'],z['binder']]
                difference=float(np.max(np.abs(np.asarray(independent)-displayed)));max_summary_error=max(max_summary_error,difference)
                if not np.allclose(independent,displayed,rtol=1e-11,atol=1e-11):raise ValueError('Generated physical summary differs')
                fractions=np.stack([(x==color).mean((1,2)) for color in range(c['q'])],1)
                conn=c['L']**2*c['q']/(c['q']-1)*np.square(fractions-fractions.mean(0)).sum(1).mean()
                bias=c['L']**2*c['q']/(c['q']-1)*np.square(fractions.mean(0)-1/c['q']).sum()
                if finite_greater(abs(conn-z['connected_chi']), 1e-10, 'scripts/verify_physical_study.py:79') or finite_greater(abs(conn+bias-z['chi']), 1e-10, 'scripts/verify_physical_study.py:79'):raise ValueError('Connected susceptibility decomposition differs')
                if len(x)!=c['count']:raise ValueError('Generation count differs')
                generated_cells+=1;generation_count+=len(x)
        expected_interventions=16 if case['seed']==640101 and case['arm']=='full' else 0
        if len(results['interventions'])!=expected_interventions:raise ValueError('Incomplete inference intervention panel')
        for intervention in results['interventions']:
            check(intervention['path'],intervention['sha256'])
            x=np.load(intervention['path']);o=statistics(x,intervention['q']);summary=intervention['summary']
            if len(x)!=256 or finite_greater(abs(o['m'].mean()-summary['m']), 1e-10, 'scripts/verify_physical_study.py:87') or finite_greater(abs(intervention['L']**2*o['m2'].mean()-summary['chi']), 1e-10, 'scripts/verify_physical_study.py:87'):
                raise ValueError('Intervention generated-moment reconstruction differs')
        exact=read(base/'exact-laws'/case['name']/'verification.json')
        if exact['status']!='passed' or len(exact['rows'])!=6:raise ValueError('Exact finite law incomplete')
        check(REPO/'scripts/physical_exact_law.py',exact['producer_sha256'])
        for row in exact['rows']:
            path=base/'exact-laws'/case['name']/row['raw'];check(path,row['raw_sha256']);z=np.load(path)
            x=z['configurations'];flat=x.reshape(len(x),-1);conditionals=np.exp(z['model_log_conditionals'])
            qprob=np.ones(len(x))
            for j in range(4):
                if not np.allclose(conditionals[:,j].sum(1),1,rtol=1e-12,atol=1e-12):raise ValueError('Conditional normalization failed')
                qprob*=conditionals[np.arange(len(x)),j,flat[:,j]]
                for prefix in {tuple(r[:j]) for r in flat}:
                    select=np.all(flat[:,:j]==np.array(prefix,dtype='uint8'),axis=1)
                    if np.max(np.ptp(conditionals[select,j],axis=0))>2e-6:raise ValueError('Conditional depends on future configuration')
            o=statistics(x,row['q']);temperature=row['ratio']/np.log1p(np.sqrt(row['q']))
            p=np.exp(-(o['energy']-o['energy'].min())/temperature);p/=p.sum()
            kl=float(np.sum(p*np.log(p/qprob)));tv=float(np.abs(p-qprob).sum()/2)
            if finite_greater(abs(qprob.sum()-1), 2e-6, 'scripts/verify_physical_study.py:105') or finite_greater(abs(kl-row['kl']), 2e-6, 'scripts/verify_physical_study.py:105') or finite_greater(abs(tv-row['tv']), 2e-6, 'scripts/verify_physical_study.py:105'):raise ValueError('Exact law reconstruction failed')
            if tv>min(1,np.sqrt(max(kl,0)/2))+2e-6:raise ValueError('Finite law bound failed')
            terms=[]
            for j in range(4):
                term=0.
                for prefix in {tuple(r[:j]) for r in flat}:
                    select=np.all(flat[:,:j]==np.array(prefix,dtype='uint8'),axis=1)
                    mass=p[select].sum()
                    conditional=np.array([p[select & (flat[:,j]==color)].sum()/mass for color in range(row['q'])])
                    model_conditional=conditionals[np.flatnonzero(select)[0],j]
                    term+=mass*np.sum(conditional*np.log(conditional/model_conditional))
                terms.append(float(term))
            if not np.allclose(terms,row['prefix_kl'],rtol=2e-6,atol=2e-6) or finite_greater(abs(sum(terms)-kl), 2e-6, 'scripts/verify_physical_study.py:117'):
                raise ValueError('Independent prefix KL chain reconstruction failed')
            exact_cells+=1
    mechanism_analysis=read(base/'mechanism-analysis/analysis.json')
    if mechanism_analysis['status']!='complete':raise ValueError('Mechanism analysis missing')
    check(REPO/'scripts/analyze_physical_mechanisms.py',mechanism_analysis['analyzer_sha256'])
    for p,h in mechanism_analysis['checked_sha256'].items():check(p,h)
    invariant_checks=0
    for case in mechanisms['cases']:
        folder=base/'mechanisms'/case['name'];m=read(folder/'manifest.json')
        if m['status']!='complete' or m['completed_updates']!=640:raise ValueError('Finite response incomplete')
        for n,h in m['files'].items():check(folder/n,h)
        scientific_updates+=640
        folder=base/'collective-validation'/case['name'];v=read(folder/'verification.json')
        if v['status']!='passed' or len(v['rows'])!=24:raise ValueError('Fresh collective validation incomplete')
        for n,h in v['files'].items():
            check(folder/n,h);z=np.load(folder/n)
            for depth in [0,1]:
                truth=z[f'fractions-{depth}'];q=truth.shape[1];m2=q/(q-1)*np.square(truth-1/q).sum(1)
                test=z['chain']>=8;cal=~test
                for family in ['hidden','A_common','G_common']:
                    pred=z[f'{family}-predicted-fractions-{depth}']
                    if np.min(pred)<-1e-14 or not np.allclose(pred.sum(1),1,atol=1e-13):raise ValueError('Invalid projected magnetic vector')
                    invariant=q/(q-1)*np.square(pred-1/q).sum(1)
                    if not np.allclose(invariant,z[f'{family}-invariant-{depth}'],rtol=1e-13,atol=1e-13):raise ValueError('Quadratic invariant differs')
                    bound=2*np.sqrt(q/(q-1))*np.linalg.norm(pred-truth,axis=1)
                    if np.any(finite_greater(np.abs(invariant-m2), bound+1e-12, 'scripts/verify_physical_study.py:143')):raise ValueError('Magnetic invariant inequality failed')
                    L=z['configurations-0'].shape[-1]
                    row=next(r for r in v['rows'] if r['q']==q and r['L']==L and r['family']==family and r['depth']==depth)
                    skill=1-np.mean((invariant[test]-m2[test])**2)/np.mean((m2[test]-m2[cal].mean())**2)
                    if finite_greater(abs(skill-row['invariant_skill']), 1e-10, 'scripts/verify_physical_study.py:147'):raise ValueError('Held-out invariant score differs')
                    invariant_checks+=len(truth)
    origins=read(base/'collective-baseline/protocol.json');sources(origins,base/'collective-baseline')
    for h in [4,8]:
        for origin in ['random','pretrained']:
            folder=base/'collective-baseline'/f'h{h}-{origin}'
            v=read(folder/'verification.json')
            if v['status']!='passed' or v['native_updates']!=0 or len(v['rows'])!=24:raise ValueError('Origin control incomplete')
            for n,digest in v['files'].items():
                check(folder/n,digest);z=np.load(folder/n)
                for depth in [0,1]:
                    f=z[f'fractions-{depth}'];q=f.shape[1];truth=q/(q-1)*np.square(f-1/q).sum(1);cal=z['chain']<8;test=~cal
                    for family in ['hidden','A_common','G_common']:
                        predicted=z[f'{family}-invariant-{depth}']
                        score=1-np.mean((predicted[test]-truth[test])**2)/np.mean((truth[test]-truth[cal].mean())**2)
                        row=next(r for r in v['rows'] if r['q']==q and r['L']==z['configurations-0'].shape[-1] and r['depth']==depth and r['family']==family)
                        if finite_greater(abs(score-row['invariant_skill']), 1e-10, 'scripts/verify_physical_study.py:163'):raise ValueError('Origin readout score differs')
    gram=read(base/'gram-identity/verification.json')
    if gram['status']!='passed' or len(gram['rows'])!=72 or gram['native_updates']!=0:raise ValueError('Native Gram identity incomplete')
    for n,digest in gram['sources'].items():check(REPO/n,digest)
    check(base/'gram-identity/raw.npz',gram['raw_sha256']);g=np.load(base/'gram-identity/raw.npz')
    for row in gram['rows']:
        key=f'h{row["heads"]}-{row["origin"]}-q{row["q"]}-L{row["L"]}-j{row["site"]}'
        traces=g[key+'-traces'];counts=g[key+'-counts'];coef=g[key+'-coefficients'];offset=g[key+'-metadata']
        expected=counts@coef.T+offset
        error=float(np.max(np.abs(traces-expected)/np.maximum(np.abs(expected),1e-30)))
        if finite_greater(abs(error-row['trace_relative_error']), 1e-12, 'scripts/verify_physical_study.py:173') or finite_greater(error, 2e-5, 'scripts/verify_physical_study.py:173'):raise ValueError('Native Gram reconstruction differs')
    transport=read(base/'transport-analysis/analysis.json')
    if transport['status']!='passed' or len(transport['rows'])!=24:raise ValueError('Bounded transport incomplete')
    check(REPO/'scripts/analyze_physical_transport.py',transport['analyzer_sha256'])
    check(base/'transport-analysis/transport.npz',transport['raw_sha256'])
    for p,h in transport['checked_sha256'].items():check(p,h)
    for r in transport['rows']:
        bound=r['lipschitz_bound']*r['fine_readout_rms']+r['physical_closure_rms']
        if finite_greater(abs(bound-r['prediction_bound']), 1e-12, 'scripts/verify_physical_study.py:181') or r['transported_prediction_rms']>bound+1e-12 or finite_greater(r['route_discrepancy_rms'], bound+r['coarse_readout_rms']+1e-12, 'scripts/verify_physical_study.py:181'):
            raise ValueError('Bounded physical transport identity differs')
    if scientific_updates!=132352 or generated_cells!=560 or exact_cells!=48 or len(analysis['finite_slopes'])!=96:
        raise ValueError('Incomplete scientific accounting')
    write_json(output,dict(status='passed',schema='physical-complete-study-v1',verifier_sha256=sha256(__file__),
        checked_sha256=checked,primary_paths=8,primary_updates=131072,finite_response_paths=10,
        finite_response_updates=1280,scientific_updates=scientific_updates,development_updates=4096,
        native_qualification_updates=24,replay_updates=2048,generated_cells=generated_cells,
        generated_configurations=generation_count,exact_joint_laws=exact_cells,
        invariant_inequality_checks=invariant_checks,maximum_generated_summary_difference=max_summary_error,
        scope='Completed physical adaptation, independent configuration-law assessment, fresh collective validation and finite native responses. Conditional finite-size calibration does not establish native thermodynamic criticality.'))
    print('passed complete native physical study',flush=True)


if __name__=='__main__':main()
