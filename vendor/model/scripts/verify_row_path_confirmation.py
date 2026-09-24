#!/usr/bin/env python3
"""Independent common-row-energy reconstruction of the entire path census."""
import argparse
from pathlib import Path
import numpy as np
from model_rg.provenance import sha256, write_json
from numerical_validation import load_json_strict
from row_path_identity_contract import bind, diagnostics, integer


def verify(study, analysis):
    a=load_json_strict(analysis.read_text());p=load_json_strict((study/'protocol.json').read_text())
    endpoints,cells=bind(a,p,study)
    summary=dict(finite_identity=0.,chronological_composition=0.,path_error_budget=0.)
    expected={(n,g,s,b) for n in [4,8,14,24] for g in [1.5,2.] for s in [9163402,9163403] for b in [1,2,4,8]}
    if a['status']!='passed' or len(a['cells'])!=64 or {(c['heads'],c['control'],c['seed'],c['block']) for c in a['cells']}!=expected or len(a['endpoints'])!=16:
        raise ValueError('Incomplete analysis census')
    if a['analysis_source_sha256']!=sha256(Path(__file__).with_name('analyze_row_path_confirmation.py')) or a['protocol_sha256']!=sha256(study/'protocol.json'):
        raise ValueError('Changed reducer or protocol')
    checked={'protocol.json'}
    for j in p['jobs']:
        checked.update('runs/'+j['run_id']+'/'+n for n in ['manifest.json','observations.npz'])
    if set(a['checked_sha256'])!=checked:raise ValueError('Incomplete raw inventory')
    for n,h in a['checked_sha256'].items():
        if sha256(study/n)!=h:raise ValueError('Changed raw record')
    q=load_json_strict((study/'qualification.json').read_text())
    if q['status']!='passed' or q['protocol_sha256']!=a['protocol_sha256'] or q['manifest_sha256']!=sha256(study/'runs/h24-g2-s9163403/manifest.json'):
        raise ValueError('Invalid largest-workload qualification')
    maximum=0.;updates=0;forwards=0;seconds=0.;peak=0
    def close(x,y):
        nonlocal maximum
        if isinstance(x,bool) or not np.isfinite(np.asarray(x,dtype=float)).all(): raise ValueError('Invalid numerical field')
        error=float(np.max(np.abs(np.asarray(x)-np.asarray(y))))
        if not np.isfinite(error) or error>3e-12:raise ValueError('Independent reconstruction differs')
        maximum=max(maximum,error)
    def reduce(x,y):
        d=y-x;dim=x.shape[-2]
        energy=lambda z:np.einsum('...ij,...ij->...',z,z)
        e=energy(x);en=energy(y);de=energy(d)
        sx=x.sum(-2);sy=y.sum(-2);sd=d.sum(-2)
        common=(sx*sx).sum(-1)/dim;cn=(sy*sy).sum(-1)/dim
        w=common/e
        l=2*(w*np.einsum('...ij,...ij->...',x,d)-(sx*sd).sum(-1)/dim)
        quadratic=(w*de-(sd*sd).sum(-1)/dim)/en
        return (w-cn/en,l/en,quadratic,l/e,np.sqrt(de/e),e,de)
    for job in p['jobs']:
        folder=study/'runs'/job['run_id'];m=load_json_strict((folder/'manifest.json').read_text())
        if m['status']!='complete' or m['job']!=job or m['protocol_sha256']!=a['protocol_sha256'] or (m['scientific_updates'],m['native_forwards'])!=(8,17) or m['artifacts']!={'observations.npz':sha256(folder/'observations.npz')}:
            raise ValueError('Wrong acquisition manifest')
        with np.load(folder/'observations.npz') as z:
            raw=z['A'].astype(float);nll=z['target_nll'].astype(float)
            if not np.array_equal(z['optimizer_steps'],np.arange(job['steps'],job['steps']+9)):
                raise ValueError('Wrong optimizer clock')
            with np.load(Path(p['panel_reuse']['parent'])/'selection.npz') as selection:
                if not np.array_equal(z['document_rows'].ravel(),selection['order'][32*job['steps']:32*(job['steps']+8)]):
                    raise ValueError('Source suffix differs')
        if raw.shape!=(9,8,5,job['heads'],64,64):raise ValueError('Wrong full matrix shape')
        if nll.shape!=(9,8) or not np.isfinite(nll).all() or not np.isfinite(raw).all(): raise ValueError('Invalid observations')
        if np.any(np.square(raw).sum((-2,-1))<=0): raise ValueError('Nonpositive energy')
        integer(m['outgoing_consumed_documents'],32*(job['steps']+8),'source cursor')
        step=reduce(raw[:-1],raw[1:])
        diag_step=diagnostics(raw[:-1],raw[1:])
        endpoint=endpoints[job['run_id']]
        vals=[float(step[i].sum(0).mean()) for i in range(4)]
        for key,v in zip(['increment','accumulated_finite_cross','accumulated_quadratic','accumulated_matrix_derivative'],vals):close(endpoint[key],v)
        close(endpoint['derivative_error'],abs(vals[0]-vals[3]))
        close(endpoint['maximum_inner_derivative_error'],abs((step[0]-step[3]).sum(0)).max())
        close(endpoint['maximum_one_step_delta'],step[4].max())
        close(endpoint['target_nll_before'],nll[0].mean());close(endpoint['target_nll_after'],nll[-1].mean())
        integer(endpoint['one_step_bound_applicable'],int(np.sum(step[4]<1)),'bound applicability')
        if np.all(step[4]<1):
            b=(3*step[4]**2+step[4]**3)/(1-step[4])**2
            close(endpoint['accumulated_bound_mean'],b.sum(0).mean())
            if np.any(abs(np.cumsum(step[0]-step[3],axis=0))>np.cumsum(b,axis=0)+3e-12):
                raise ValueError('Pathwise bound fails')
            summary['path_error_budget']=max(summary['path_error_budget'],float(np.max(abs(np.cumsum(step[0]-step[3],axis=0))-np.cumsum(b,axis=0))))
        elif endpoint['accumulated_bound_mean'] is not None:
            raise ValueError('Undefined bound must be null')
        for block in [1,2,4,8]:
            starts=np.arange(0,8,block);r=reduce(raw[starts],raw[starts+block])
            cell=cells[(job['run_id'],block)]
            dc=diagnostics(raw[starts],raw[starts+block])
            chrono=np.stack([diag_step[1][k:k+block].sum(0)+diag_step[2][k:k+block].sum(0) for k in starts])
            identity=float(np.max(abs(dc[0]-dc[1]-dc[2])))
            composition=float(np.max(abs(dc[0]-chrono)))
            if cell['identity_error']!=identity or cell['composition_error']!=composition:
                raise ValueError('Reported residual differs from defined reduction')
            summary['finite_identity']=max(summary['finite_identity'],identity)
            summary['chronological_composition']=max(summary['chronological_composition'],composition)
            for key,index in [('increment',0),('finite_cross',1),('quadratic',2),('matrix_derivative',3)]:close(cell[key],r[index].mean())
            close(cell['matrix_derivative_absolute_error'],abs(r[0]-r[3]).mean())
            close(cell['relative_displacement_energy'],(r[6]/r[5]).mean())
            close(cell['maximum_delta'],r[4].max())
            # Explicit pair sums independently check the signed cross-time term.
            cross=[]
            for k,e in zip(starts,r[5]):
                increments=np.diff(raw[k:k+block+1],axis=0)
                inner=np.zeros(e.shape)
                for i in range(block):
                    for j in range(i+1,block):inner+=2*np.einsum('...ij,...ij->...',increments[i],increments[j])
                cross.append(inner/e)
            close(cell['signed_cross_time_energy'],np.mean(cross))
            if cell['pairs']!=r[0].size or cell['aligned_blocks']!=8//block:raise ValueError('Wrong inner census')
            close(r[0],r[1]+r[2])
            close(r[0],np.stack([step[0][k:k+block].sum(0) for k in starts]))
        updates+=m['scientific_updates'];forwards+=m['native_forwards'];seconds+=m['elapsed_seconds'];peak=max(peak,m['peak_allocated_bytes'])
    if (updates,forwards,a['scientific_updates'],a['native_forward_calls'],a['one_step_pairs'],a['all_aligned_block_pairs'])!=(128,272,128,272,64000,120000):raise ValueError('Wrong execution or matrix census')
    if a['maximum_errors']!=summary: raise ValueError('Reported maximum residual differs')
    close(a['worker_seconds'],seconds);close(a['peak_allocated_bytes'],peak)
    return dict(status='passed',schema='row-path-verification-v2',identity_binding='complete-frozen-job-keys',
                contract_sha256=sha256(Path(__file__).with_name('row_path_identity_contract.py')),analysis_sha256=sha256(analysis),verifier_sha256=sha256(__file__),
                protocol_sha256=a['protocol_sha256'],qualification_sha256=sha256(study/'qualification.json'),
                paths=16,cells=64,scientific_updates=128,native_forward_calls=272,
                one_step_pairs=64000,all_aligned_block_pairs=120000,
                maximum_independent_discrepancy=maximum,
                method='Common-row energy and explicit cross-time pair products; no imports of producer or analysis mathematics.',
                scope='Saved-matrix numerical reconstruction and deterministic bounds; zero new training or inference during verification.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['study','analysis','output']:parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=verify(args.study,args.analysis);write_json(args.output,result);print(result)
