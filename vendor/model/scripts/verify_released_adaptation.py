"""Verify complete released-model evidence, resource identities and checkpoint selection."""
from companion_paths import required_input, acquisition_identity
from companion_paths import configured_path
from numerical_claims import finite_greater
from numerical_validation import load_json_strict
import argparse,json,math,sys,subprocess
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json
ROOT=Path(configured_path('data:model'))

def main(a):
    producer=Path(a.producer_root).resolve()
    s=Path(a.study).resolve();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=False);bound={}
    def check(p,h=None):
        p=Path(p);actual=sha256(p)
        if h is not None and actual!=h:raise ValueError('Changed bound evidence: '+str(p))
        bound[str(p)]=actual;return actual
    def read(p):check(p);return load_json_strict(Path(p).read_text())
    data=read(s/'data/language.json');check(s/'data/language.npz',data['data_sha256'])
    for p,h in data['inputs'].items():check(p,h)
    with np.load(s/'data/language.npz') as z:
        ids=[];language_labels={}
        for domain in ['technical','narrative','unassigned']:
            for split,count in [('train',4096),('validation',256),('test',512)]:
                v=z[f'{domain}-{split}-ids'];ids.extend(v.tolist())
                if split!='train':language_labels[(domain,split)]=(v.copy(),z[f'{domain}-{split}-tokens'].copy())
                if len(v)!=count or len(set(v))!=count or np.any(z[f'{domain}-{split}-positions']<160000000):raise ValueError('Language sampling contract')
        if len(set(ids))!=14592:raise ValueError('Language document leakage')
        trainids={d:set(z[d+'-train-ids'].tolist()) for d in ['technical','narrative','unassigned']}
    def check_language_raw(path,split):
        with np.load(path) as z:
            for domain in ['technical','narrative','unassigned']:
                ids,tokens=language_labels[(domain,split)]
                for offset in [0,192]:
                    for length in [32,64,128]:
                        key=f'{domain}-o{offset}-p{length}';v=z[key]
                        if v.shape!=(len(ids),6) or not np.isfinite(v).all() or not np.array_equal(z[key+'-document-ids'],ids) or not np.array_equal(v[:,3],tokens[:,offset+length]):raise ValueError('Language target or document mismatch')
                        if not np.array_equal(v[:,4],v[:,2]==v[:,3]) or np.any(v[:,0]<v[:,1]) or not np.isin(v[:,5],[0,1]).all():raise ValueError('Compact language scores disagree')
    phys=read(s/'data/physical.json');cells={c['id']:c for c in phys['cells']}
    if len(cells)!=118 or {split:sum(c['split']==split for c in cells.values()) for split in ['train','validation','test']}!={'train':30,'validation':18,'test':70}:raise ValueError('Incomplete physical design')
    for c in cells.values():check(c['path'],c['sha256'])
    for index in [1,4,5]:
        q=read(s/f'base-qualification/model{index}/verification.json')
        if q['status']!='passed' or q['qualification_updates']!=3 or finite_greater(q['relative_gradient_error'], 2e-5, 'scripts/verify_released_adaptation.py:42'):raise ValueError('Native qualification failed')
        check(Path(q['assets'])/'model.safetensors',q['weights_sha256']);check(Path(q['assets'])/'modeling_pldrllm.py',q['native_sha256']);check(Path(q['assets'])/'tokenizer.model',q['tokenizer_sha256'])
        for filename,field in [('scripts/qualify_released_base.py','producer_sha256'),('src/model_rg/native.py','adapter_sha256'),('src/model_rg/physical_native.py','physical_adapter_sha256')]:check(producer/filename,q[field])
        check(s/'data/language.npz',q['language_data_sha256']);check(s/'data/physical.json',q['physical_manifest_sha256'])
        for n,h in q['files'].items():check(s/f'base-qualification/model{index}'/n,h)
        check_language_raw(s/f'base-qualification/model{index}/language-validation.npz','validation')
    downloaded=read(s/'downloaded-models.json')
    for name,entry in downloaded.items():
        for n,h in entry['files'].items():check(ROOT/'assets'/name/n,h)
    select=read(s/'base-selection.json')
    if select['selected']!=5 or min(select['candidates'],key=lambda v:v['language_nll'])['model']!=5:raise ValueError('Base selection changed')
    check(s/'assessment-plan.json');check(s/'language-adaptation-extension.json');read(s/'lowrank-rate-selection.json')
    expected=['pilot-technical-low','pilot-technical-high','pilot-physical-low','pilot-physical-high','pilot-lowrank-low','pilot-lowrank-high']+[task+suffix for suffix in ['-s0','-lowrank'] for task in ['technical','narrative','mixture','general']]+['physical-s0','physical-s1']
    inventory=[]
    for name in expected:
        root=s/'runs'/name;result=read(root/'result.json');spec=read(root/'protocol.json')
        if result['status']!='complete' or result['name']!=name or result['steps']!=spec['steps']:raise ValueError('Incomplete training')
        check(root/'protocol.json',result['protocol_sha256']);check(root/'draws.npz',result['draws_sha256']);check(root/'trace.npy',result['trace_sha256']);check(root/'best.pt',result['best_sha256']);check(root/'last.pt',result['last_sha256'])
        for n,h in spec['sources'].items():check(root/'executed-source'/n,h);check(producer/n,h)
        obs=result['observations'];best=min(obs,key=lambda v:v['nll'])
        if best['step']!=result['selected_step'] or best['nll']!=result['selected_validation_nll'] or obs[-1]['step']!=result['steps']:raise ValueError('Checkpoint selection contract')
        trace=np.load(root/'trace.npy')
        if trace.shape!=(result['steps'],8) or not np.isfinite(trace).all() or not np.array_equal(trace[:,0],np.arange(1,len(trace)+1)):raise ValueError('Training trace shape or counters')
        if np.any(trace[:,5]<0) or np.any(trace[:,6]<0) or np.any(trace[:,6]>1):raise ValueError('Invalid training scores')
        epochs=spec['epochs'];per=spec['steps_per_epoch']
        if spec['role']=='primary' and (epochs!=3 or result['steps']!=epochs*per):raise ValueError('Incomplete three-epoch path')
        with np.load(root/'draws.npz') as z:
            for e in range(epochs):
                if spec['task']=='physical':
                    seen=set()
                    for cid,ii,j,col,sy in zip(*(z[f'e{e}-{k}'] for k in ['cell','sample','site','colors','symmetry'])):
                        c=cells[int(cid)];q=c['q']
                        if c['split']!='train' or not 0<=j<c['L']**2 or len(ii)!=32 or not np.all(np.sort(col[:,:q],axis=1)==np.arange(q)) or not np.all((sy>=0)&(sy<8)):raise ValueError('Invalid physical training draw')
                        for i in ii:
                            key=(int(cid),int(i))
                            if key in seen or not 0<=i<c['chains']*c['samples_per_chain']:raise ValueError('Repeated or nonexistent within-epoch identity')
                            seen.add(key)
                    if len(seen)!=61440:raise ValueError('Dropped physical configurations')
                else:
                    pairs=z[f'e{e}-examples'].reshape(-1,2);unique=set(map(tuple,pairs));docs=set(pairs[:,0])
                    if len(unique)!=32768 or len(docs)!=4096 or not docs<=set.union(*trainids.values()) or set(pairs[:,1])!=set(range(8)):raise ValueError('Language resource contract')
                    if spec['task'] in trainids and not docs<=trainids[spec['task']]:raise ValueError('Source-arm mismatch')
                    if e==0:first=unique
                    elif unique!=first:raise ValueError('Epoch changed finite language corpus')
        if name.endswith('-lowrank') or name.startswith('pilot-lowrank'):
            if result['trainable_parameters']!=203264 or finite_greater(max(v['max_logit_error'] for v in result['merge_errors']), 1e-3, 'scripts/verify_released_adaptation.py:87'):raise ValueError('Low-rank export mismatch')
        inventory.append(dict(name=name,steps=result['steps'],selected_step=result['selected_step'],selected_epoch=result['selected_epoch'],scientific_updates=result['scientific_updates'],development_updates=result['development_updates'],seconds=result['seconds']))
    language_names=['base5']+[task+suffix for suffix in ['-s0','-lowrank'] for task in ['technical','narrative','mixture','general']]
    for name in language_names:
        root=s/'assessment'/name/'language';r=read(root/'result.json');check(root/'language-test.npz',r['raw_sha256'])
        if r['status']!='complete' or len(r['rows'])!=18 or any(v['documents']!=512 for v in r['rows']):raise ValueError('Incomplete language test')
        check_language_raw(root/'language-test.npz','test')
        for n,h in r['sources'].items():check(producer/n,h)
    for name in ['base5','physical-s0','physical-s1']:
        for kind in ['spin','exact','geometry','critical','thermal']:
            root=s/'assessment'/name/kind;r=read(root/'result.json')
            if r['status']!='complete':raise ValueError('Incomplete physical assessment')
            if kind=='spin':
                if len(r['rows'])!=560:raise ValueError('Incomplete proper-prefix panel')
                check(root/'spin-test.npz',r['raw_sha256'])
                rng=np.random.default_rng(219301);actual={(v['cell'],v['site']):v for v in r['rows']};expected=set()
                if len(actual)!=560:raise ValueError('Duplicated spin cells or target sites')
                with np.load(root/'spin-test.npz') as z:
                    for c in cells.values():
                        if c['split']!='test':continue
                        L,q=c['L'],c['q'];edges=np.linspace(0,L*L,9,dtype=int);sites=[int(rng.integers(edges[i],edges[i+1])) for i in range(8)];weights=np.diff(edges)/(L*L)
                        x=np.asarray(np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape'])))[:,:8].reshape(128,L*L)
                        for site,weight in zip(sites,weights):
                            key=(c['id'],site);expected.add(key);row=actual[key];v=z[row['raw']];hist=z[row['raw']+'-histogram']
                            if (row['q'],row['L'],row['ratio'],row['site_weight'],row['samples'])!=(q,L,c['temperature_ratio'],weight,128) or v.shape!=(128,q+3):raise ValueError('Native spin panel design mismatch')
                            if not np.array_equal(v[:,q],x[:,site]):raise ValueError('Spin target differs from independent source chain')
                            counts=np.stack([(x[:,:site]==a).sum(1) for a in range(q)],1);prob=(counts+.5)/(site+q*.5)
                            baseline=np.stack([-np.log(prob[np.arange(128),x[:,site]]),prob.argmax(1)==x[:,site]],1)
                            np.testing.assert_allclose(hist,baseline,rtol=0,atol=1e-12)
                    if set(actual)!=expected:raise ValueError('Spin target-strata inventory mismatch')
            elif kind=='geometry':
                check(root/'native-geometry.npz',r['raw_sha256'])
                if len(r['rows'])!=6:raise ValueError('Incomplete geometry panels')
                with np.load(root/'native-geometry.npz') as z:
                    for row in r['rows']:
                        key=f"cell{row['cell']}";L=row['L'];mu=z[key+'-common'];en=z[key+'-energy'];rn=z[key+'-row-energy'];da=z[key+'-paired-A-MSE'];dg=z[key+'-paired-G-MSE'];ma=z[key+'-short-A-mean'];mg=z[key+'-short-G-mean']
                        if mu.shape!=(64,5,14,64) or en.shape!=(64,5,14):raise ValueError('Geometry population shape')
                        rf=rn/np.maximum(en,1e-300);between=np.square(mu-mu.mean(0)).mean();total=np.square(mu).mean()
                        reconstructed=dict(row_fraction_median=np.median(rf),row_fraction_max=rf.max(),between_input_common_fraction=between/max(total,1e-300),common_rms=np.sqrt(total),common_input_rms=np.sqrt(between),paired_A_rms=np.sqrt(da.mean()),paired_G_rms=np.sqrt(dg.mean()),short_A_mean=ma.mean(),short_G_mean=mg.mean(),order_A=np.sqrt(da.mean())/abs(ma.mean()),order_G=np.sqrt(dg.mean())/abs(mg.mean()))
                        for field,value in reconstructed.items():np.testing.assert_allclose(value,row[field],rtol=1e-10,atol=1e-28,err_msg=field)
                        A=z[key+f'-j{L*L-1}-A'].astype(np.float64);A0=z[key+f'-j{L*L//2}-A'].astype(np.float64);G=z[key+f'-j{L*L-1}-G'].astype(np.float64);G0=z[key+f'-j{L*L//2}-G'].astype(np.float64)
                        for observed,expected in [(A.mean(-2),mu[:2]),(np.square(A).mean((-1,-2)),en[:2]),(np.square(A-A.mean(-2,keepdims=True)).mean((-1,-2)),rn[:2]),(np.square(A-A0).mean((-1,-2)),da[:2]),(np.square(G-G0).mean((-1,-2)),dg[:2])]:np.testing.assert_allclose(observed,expected,rtol=1e-10,atol=1e-28)

            elif kind=='exact':
                if len(r['rows'])!=6:raise ValueError('Incomplete exact laws')
                for row in r['rows']:
                    check(root/row['raw'],row['sha256'])
                    with np.load(root/row['raw']) as z:
                        x=z['configurations'];q=row['q'];flat=x.reshape(-1,4);lp=z['source_logp'];lq=z['model_logp'];lc=z['model_log_conditionals']
                        if len(x)!=q**4 or len(set(map(tuple,flat)))!=q**4:raise ValueError('Incomplete exact sample space')
                        energy=-sum((x==np.roll(x,1,axis=axis)).sum((1,2)) for axis in [1,2]);pred=-energy*np.log(1+np.sqrt(q))/row['ratio'];pred-=np.logaddexp.reduce(pred)
                        np.testing.assert_allclose(lp,pred,rtol=1e-12,atol=1e-12)
                        np.testing.assert_allclose(np.logaddexp.reduce(lc,axis=-1),0,rtol=0,atol=1e-12)
                        np.testing.assert_allclose(lq,np.take_along_axis(lc,flat[:,:,None],axis=2)[:,:,0].sum(1),rtol=1e-12,atol=1e-12)
                        P=np.exp(lp);Q=np.exp(lq);kl=P@(lp-lq);tv=.5*np.abs(P-Q).sum();terms=[]
                        for j in range(4):
                            term=0.
                            for pref in set(map(tuple,flat[:,:j])):
                                mask=np.all(flat[:,:j]==np.asarray(pref,dtype=np.uint8),axis=1);mass=P[mask].sum();cond=np.array([P[mask&(flat[:,j]==c)].sum()/mass for c in range(q)])
                                np.testing.assert_allclose(lc[mask,j],np.broadcast_to(lc[np.flatnonzero(mask)[0],j],lc[mask,j].shape),rtol=0,atol=2e-6)
                                term+=mass*(cond@(np.log(cond)-lc[np.flatnonzero(mask)[0],j]))
                            terms.append(term)
                        np.testing.assert_allclose([kl,tv,sum(terms)], [row['kl'],row['tv'],kl],rtol=1e-10,atol=2e-6)
                        if finite_greater(abs(Q.sum()-1), 2e-6, 'scripts/verify_released_adaptation.py:150') or tv>min(1,np.sqrt(max(kl,0)/2))+2e-6:raise ValueError('Finite probability-law bound')
                        fractions=np.stack([(flat==c).mean(1) for c in range(q)],1);m2=q/(q-1)*np.square(fractions-1/q).sum(1)
                        for key,ob in [('m',np.sqrt(m2)),('m2',m2),('m4',m2*m2)]:
                            error=abs((P-Q)@ob);np.testing.assert_allclose(error,row['errors'][key],rtol=1e-10,atol=1e-12)
                            if finite_greater(error, tv+2e-6, 'scripts/verify_released_adaptation.py:154'):raise ValueError('Finite moment-law bound')

            else:
                if len(r['rows'])!=(14 if kind=='critical' else 32):raise ValueError('Incomplete generated-law panel')
                for row in r['rows']:
                    check(root/row['path'],row['sha256']);x=np.load(root/row['path'])
                    if x.shape!=(row['count'],row['L'],row['L']) or not np.isin(x,range(row['q'])).all() or row['count']!=(512 if kind=='critical' else 128):raise ValueError('Generated configurations mismatch')
            for n,h in r['sources'].items():check(producer/n,h)
    for kind in ['language','spin','generation']:
        r=read(s/'analysis'/(kind+'.json'))
        if r['status']!='complete' or r['analyzer_sha256']!=sha256(REPO/'scripts/analyze_released_adaptation.py'):raise ValueError('Unbound analysis')
        for p,h in r['checked_sha256'].items():check(p,h)
    spatial=read(s/'analysis/spatial-centering.json')
    if spatial['status']!='complete' or spatial['analyzer_sha256']!=sha256(REPO/'scripts/analyze_released_spatial_centering.py') or len(spatial['cells'])!=138:raise ValueError('Incomplete spatial covariance decomposition')
    for path,h in spatial['checked_sha256'].items():check(path,h)
    for row in spatial['cells']:
        for field in ['model','source']:
            d=row[field];np.testing.assert_allclose(d['chi'],d['connected_chi']+row['L']**2*d['global_mean_sector'],rtol=1e-12,atol=1e-12)
            for r in d['correlations'].values():np.testing.assert_allclose(r['raw'],r['connected']+r['mean_sector'],rtol=1e-12,atol=1e-12)
    diagnostics=read(s/'analysis/source-chains.json')
    if diagnostics['status']!='complete' or diagnostics['producer_sha256']!=sha256(REPO/'scripts/check_released_source_chains.py') or {c['cell'] for c in diagnostics['cells']}!=set(cells):raise ValueError('Incomplete source-chain diagnostics')
    for path,h in diagnostics['checked_sha256'].items():check(path,h)
    subprocess.run([sys.executable,str(REPO/'scripts/reconstruct_released_independent.py'),'--study',str(s),'--output',str(out/'independent-reconstruction.json')],check=True,cwd=REPO)
    raw=read(out/'independent-reconstruction.json')
    if raw['status']!='passed' or raw['counts'].get('spin_logit_coordinates')!=537600:raise ValueError('Independent aggregate reconstruction failed')
    transport=Path(a.transport_study).resolve();single=transport/'single-pass'
    from model_rg.released_qualification import configure,validate,assessment_identity,qualify_path
    configure()
    for branch,entry in [('full','train_released_adaptation'),('factor','train_released_lowrank')]:
        q,path=validate(single,5,branch,entry);check(path)
        for n,h in q['artifacts'].items():check(path.parent/n,h)
        for key in ['assets','data']:
            for p,h in q['contract'][key].items():check(p,h)
        for n,h in q['contract']['sources'].items():check(REPO/n,h)
    for name in ['base5','technical-once-full','technical-once-factor']:assessment_identity(single,name)
    subprocess.run([sys.executable,str(REPO/'scripts/analyze_singlepass_adaptation.py'),'--study',str(single),'--output',str(out/'singlepass-reconstruction.json')],check=True,cwd=REPO)
    reconstructed=read(out/'singlepass-reconstruction.json');reported=read(single/'analysis/singlepass.json')
    if reconstructed!=reported:raise ValueError('Single-pass raw reconstruction differs from published summary')
    for p,h in reported['checked_sha256'].items():check(p,h)
    for script,filename in [('check_thermal_transport.py','thermal-budgets.json'),('analyze_thermal_transport.py','thermal-contrasts.json')]:
        subprocess.run([sys.executable,str(REPO/'scripts'/script),'--study',str(s),'--output',str(out/filename)],check=True,cwd=REPO)
        if read(out/filename)!=read(transport/filename):raise ValueError('Thermal reconstruction differs: '+filename)
    subprocess.run([sys.executable,str(REPO/'scripts/check_released_summary_mutations.py'),'--study',str(s),'--output',str(out/'summary-mutations.json')],check=True,cwd=REPO)
    mutations=read(out/'summary-mutations.json')
    if len(mutations['mutations'])!=3 or not all(r['rejected'] for r in mutations['mutations']):raise ValueError('Independent summary mutation gate failed')
    for name in ['thermal-budgets.json','thermal-contrasts.json']:
        r=read(transport/name)
        for p,h in r['inputs'].items():check(p,h)
    for n in ['docs/TRANSPORT_REPRODUCTION.md','docs/environment-transport.json']:check(REPO/n)
    generated=Path(required_input('generated-evidence'));render=read(generated/'released-render-manifest.json')
    if render['status']!='complete' or render['renderer_sha256']!=sha256(REPO/'scripts/render_released_adaptation.py'):raise ValueError('Changed renderer')
    for kind,h in render['analyses'].items():check(s/'analysis'/(kind+'.json'),h)
    for n,h in render['generated'].items():check(generated/n,h)
    copied=generated/'released-evidence'
    if {str(p.relative_to(copied)) for p in copied.rglob('*') if p.is_file()}!=set(render['evidence_files']):raise ValueError('Copied evidence inventory mismatch')
    for n,h in render['evidence_files'].items():check(copied/n,h);check(s/n,h)
    counts=read(generated/'qualification-current.json')
    registry=read(REPO/'scripts/formal/statement-registry.json')
    if counts['counts']['selected_statements']!=len(registry['exports']) or counts['counts']['formal_modules']!=len(list((REPO/'ModelRG').glob('*.lean'))):raise ValueError('Qualification count table mismatch')
    for entry in counts['records'].values():check(entry['path'],entry['sha256'])
    # Spin reduction uses every q-token logit; language reduction uses stored vocabulary
    # normalizers, target logits and predicted indices, not unretained vocabulary vectors.
    # Selected checkpoint metadata, frozen examples and all omitted-looking cells are checked above.
    software={}
    for name in ['numerical','formal','statements']:
        p=REPO/f'docs/transport-{name}/verification.json';r=read(p)
        if r['status']!='passed':raise ValueError('Software gate failure')
        for field in ['tested_sources','module_sources','modules','support_sources']:
            for n,h in r.get(field,{}).items():check(REPO/n,h)
        software[name]=dict(path=str(p),sha256=sha256(p))
    clean=read(REPO/'docs/transport-clean/verification.json')
    if clean.get('status')!='passed' or clean.get('preexisting_owned_artifacts') is not False:raise ValueError('Missing clean owned formal rebuild')
    for n,h in clean['module_sources'].items():check(REPO/n,h)
    retained=Path(required_input('verify-released-adaptation-input-1'));manifest=retained/'MANIFEST.sha256';check(manifest)
    for line in manifest.read_text().splitlines():
        h,n=line.split('  ',1);check(retained/n,h)
    source={}
    for folder in ['src','scripts','tests','ModelRG']:
        for p in sorted((REPO/folder).rglob('*')):
            if p.is_file() and p.suffix in ['.py','.lean','.cpp','.sh','.json']:source[str(p.relative_to(REPO))]=sha256(p)
    for p in (Path(required_input('publication-source'))).rglob('*'):
        if p.is_file() and p.suffix not in ['.aux','.log','.bbl','.blg','.toc','.out'] and (p.parent!=Path(required_input('publication-source')) or p.suffix in ['.tex','.bib']):check(p)
    for n in ['ModelRG.lean','README.md','docs/RELEASED_ADAPTATION_REPRODUCTION.md',str(Path(required_input('generated-evidence'))/'qualification-current.json')]:check(REPO/n)
    write_json(out/'verification.json',dict(status='passed',schema='released-adaptation-complete-release-v1',study=str(s),retained_release=str(retained),verifier_sha256=sha256(__file__),current_sources=source,archived_producer_root=str(producer),checked_sha256=bound,software=software,transport_study=str(transport),singlepass_updates=2048,current_execution_qualification_updates=6,runs=inventory,counts=dict(scientific_updates=sum(r['scientific_updates'] for r in inventory),development_updates=sum(r['development_updates'] for r in inventory),qualification_updates=9,physical_test_cells=210,generated_cells=138,exact_law_cells=18,language_test_models=9),scope='All completed selected paths and frozen assessment cells, independent q-token logit and compact language-score reductions, proper-prefix and finite-resource invariants, source-bound software checks, immutable retained publication. This does not certify population universality or unavailable pretraining document manifests.'))
    print(out/'verification.json',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output',required=True);p.add_argument('--transport-study',required=True);p.add_argument('--producer-root',required=True,help='Original producer snapshot for archived scientific records');main(p.parse_args())
