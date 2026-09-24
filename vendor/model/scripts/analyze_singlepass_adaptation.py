"""Reconstruct a fixed two-path single-pass study from draws and compact scores."""
import argparse,json,sys
from pathlib import Path
import numpy as np
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
from model_rg.provenance import sha256,write_json


def main(a):
    s=Path(a.study).resolve();bound={}
    def read(p):bound[str(p)]=sha256(p);return json.loads(p.read_text())
    def check(p,h):
        if sha256(p)!=h:raise ValueError('Changed study input: '+str(p))
        bound[str(p)]=h
    plan=read(s/'singlepass-comparison-plan.json');panels={};inventory=[]
    for name in ['base5']+[r['name'] for r in plan['runs']]:
        if name!='base5':
            root=s/'runs'/name;r=read(root/'result.json');p=read(root/'protocol.json')
            if r['status']!='complete' or r['steps']!=1024 or p['epochs']!=1:raise ValueError('Incomplete single-pass path')
            for file,key in [('protocol.json','protocol_sha256'),('draws.npz','draws_sha256'),('trace.npy','trace_sha256'),('best.pt','best_sha256'),('last.pt','last_sha256')]:check(root/file,r[key])
            for n,h in p['sources'].items():check(root/'executed-source'/n,h)
            with np.load(root/'draws.npz') as z:
                if set(z.files)!={'e0-examples'}:raise ValueError('Unexpected repeated epoch')
                ids=z['e0-examples'].reshape(-1,2)
                if len(set(map(tuple,ids)))!=32768 or len(set(ids[:,0]))!=4096:raise ValueError('Single-pass resource failure')
            if name==plan['runs'][0]['name']:reference_draws=ids.copy()
            elif not np.array_equal(ids,reference_draws):raise ValueError('Source coupling changed')
            trace=np.load(root/'trace.npy')
            if trace.shape!=(1024,8) or not np.isfinite(trace).all():raise ValueError('Invalid native trace')
            best=min(r['observations'],key=lambda row:row['nll'])
            if best['step']!=r['selected_step']:raise ValueError('Checkpoint selection mismatch')
            inventory.append(dict(name=name,steps=1024,selected_step=r['selected_step'],validation_nll=r['selected_validation_nll'],seconds=r['seconds'],peak_gib=r['peak_gib'],distinct_examples=32768,distinct_documents=4096))
        root=s/'assessment'/name/'language';r=read(root/'result.json');check(root/'language-test.npz',r['raw_sha256'])
        with np.load(root/'language-test.npz') as z:
            for domain in ['technical','narrative','unassigned']:
                for lengths,label in [([64],'matched64'),([32,64,128],'all')]:
                    arrays=[]
                    for offset in [0,192]:
                        for length in lengths:
                            v=z[f'{domain}-o{offset}-p{length}'].astype(float)
                            if not np.array_equal(v[:,4],v[:,2]==v[:,3]):raise ValueError('Incorrect argmax indicators')
                            arrays.append(np.column_stack([v[:,0]-v[:,1],v[:,4]]))
                    panels[name,domain,label]=np.mean(arrays,0)
    rng=np.random.default_rng(220021);rows=[]
    for run in plan['runs']:
        name=run['name']
        for domain in ['technical','narrative','unassigned']:
            for label in ['matched64','all']:
                scores=panels[name,domain,label];base=panels['base5',domain,label];delta=scores-base
                samples=[]
                for _ in range(32):
                    ix=rng.integers(512,size=(500,512));samples.extend(delta[ix].mean(1))
                ci=np.quantile(samples,[.025,.975],axis=0);adjusted=np.quantile(samples,[.05/24,1-.05/24],axis=0)
                rows.append(dict(name=name,domain=domain,prefix=label,nll=float(scores[:,0].mean()),accuracy=float(scores[:,1].mean()),delta_nll=float(delta[:,0].mean()),delta_accuracy=float(delta[:,1].mean()),nll_interval=ci[:,0].tolist(),accuracy_interval=ci[:,1].tolist(),nll_family12_interval=adjusted[:,0].tolist()))
    dest=Path(a.output) if getattr(a,'output',None) else s/'analysis/singlepass.json';write_json(dest,dict(status='passed',producer_sha256=sha256(__file__),checked_sha256=bound,inventory=inventory,rows=rows,seed=220021,bootstrap_repeats=16000,scientific_updates=2048,scope=plan['scope'],sampling_unit='Paired test documents with the whole offset/prefix panel retained; conditional on fixed checkpoints and corpus.'))
    print(json.dumps(dict(inventory=inventory,matched_technical=[r for r in rows if r['domain']=='technical' and r['prefix']=='matched64']),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--output');main(p.parse_args())
