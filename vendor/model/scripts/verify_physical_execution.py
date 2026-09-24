"""Independently check complete physical branch execution and archive design."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json
from physical_verification_design import check,expected_rate


def verify(study, output, archived=False):
    study=Path(study);output=Path(output)
    if output.exists():raise FileExistsError(output)
    p=study/'protocol.json';spec=json.loads(p.read_text());n=check(spec,archived)
    checked={str(p):sha256(p)};rows=[]
    def bind(p,h=None):
        p=Path(p);digest=sha256(p)
        if h is not None and h!=digest:raise ValueError('Changed execution file: '+str(p))
        checked[str(p)]=digest
    for name,h in spec['sources'].items():bind(study/'executed-source'/name,h)
    bind(study/'draws.npz',spec['draws_sha256'])
    with np.load(study/'draws.npz',allow_pickle=False) as d:
        cell,site,index=d['cell'],d['site'],d['index']
    if cell.shape!=(n,) or site.shape!=(n,) or index.shape!=(n,32):raise ValueError('Draw shape')
    data=json.loads((Path(spec['training_data'])/'manifest.json').read_text())
    for c in data['cells']:
        selected=cell==c['id'];ids=index[selected].ravel()
        if len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=c['chains']*c['samples_per_chain']):
            raise ValueError('Not distinct legal physical identities')
        if np.any(site[selected]<0) or np.any(site[selected]>=c['L']**2):raise ValueError('Site range')
    for c in spec['cases']:
        folder=study/c['name'];mp=folder/'manifest.json';bind(mp)
        m=json.loads(mp.read_text())
        if m['status']!='complete' or m['completed_updates']!=n or m['consumed_configurations']!=32*n:
            raise ValueError('Incomplete branch')
        if m['case']!=c or m['protocol_sha256']!=sha256(p):raise ValueError('Branch identity')
        if [v['step'] for v in m['checkpoints']]!=spec['checkpoints'][1:]:raise ValueError('Checkpoint inventory')
        for name,h in m['files'].items():bind(folder/name,h)
        t=np.load(folder/'training-trace.npy')
        if t.shape!=(n,6) or not np.isfinite(t).all():raise ValueError('Trace')
        if not np.array_equal(t[:,0],np.arange(1,n+1)) or not np.array_equal(t[:,1],cell) or not np.array_equal(t[:,2],site):
            raise ValueError('Draw/trace correspondence')
        if not np.allclose(t[:,5],[expected_rate(i,n) for i in range(n)],rtol=1e-13,atol=0):raise ValueError('Schedule trace')
        state_path=Path(m['checkpoints'][-1]['path']);bind(state_path,m['checkpoints'][-1]['sha256'])
        state=torch.load(state_path,map_location='cpu',weights_only=False)
        if state['step']!=n or state['case']!=c or state['protocol_sha256']!=sha256(p):raise ValueError('State provenance')
        for g in state['optimizer']['param_groups']:
            if g['betas']!=(.9,.95) or g['eps']!=1e-8 or g['weight_decay']!=.01:raise ValueError('Executed optimizer')
        del state
        if not archived:
            b=m['branch_checks'];frozen=c['arm']=='frozen-generator'
            if (b['initial_generator']==b['final_generator'])!=frozen:raise ValueError('Generator control')
            if b['initial_body']==b['final_body']:raise ValueError('Body did not update')
            if b['shuffle_histograms_checked']!=(n*32 if c['arm']=='shuffled' else 0):raise ValueError('Shuffle checks')
        rows.append(dict(case=c['name'],updates=n,seconds=m['runtime_seconds'],peak_gib=m['peak_gib']))
    result=dict(status='passed',schema='physical-execution-verification-v2',
        role='archived scientific design correspondence' if archived else 'current native branch qualification',
        rows=rows,updates=n*len(rows),protocol_sha256=sha256(p),verifier_sha256=sha256(__file__),
        design_verifier_sha256=sha256(REPO/'scripts/physical_verification_design.py'),verified_files=checked)
    write_json(output,result)
    if not archived and spec['stage']=='qualification':
        for h in [4,8]:
            report=dict(schema='physical-branch-qualification-v2',status='passed',heads=h,
                branches=['full','frozen-generator','shuffled'],qualification_updates=36,
                protocol=str(p),protocol_sha256=sha256(p),sources=spec['sources'],
                native_assets=spec['native_assets'],runtime=spec['runtime'],verified_files=checked,
                verifier_sha256=sha256(__file__),design_verifier_sha256=result['design_verifier_sha256'])
            target=study.parent/f'branch-h{h}.json'
            if target.exists():raise FileExistsError(target)
            write_json(target,report)
    print(json.dumps(dict(status='passed',role=result['role'],updates=result['updates'],rows=rows)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True)
    p.add_argument('--archived',action='store_true');a=p.parse_args();verify(a.study,a.output,a.archived)
