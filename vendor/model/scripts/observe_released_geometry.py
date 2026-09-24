"""Native A/G input coupling and row concentration at competent spin checkpoints."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from model_rg.physical_native import selected_forward,prefix_batch,spin_tokens,context_tokens
from model_rg.provenance import sha256,write_json
from assess_released_adaptation import load_model

@torch.no_grad()
def main(a):
    from model_rg.released_qualification import admit_assessment,destination
    study=Path(a.study).resolve();out=destination(study,a.name,'assessment')/'geometry'
    admit_assessment(study,a.name,'observe_released_geometry')
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter,identity=load_model(study,a.name,a.device);proc=spm.SentencePieceProcessor(model_file=str(adapter.path/'tokenizer.model'));alphabet=spin_tokens(proc)
    sources={str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'scripts/assess_released_adaptation.py',REPO/'src/model_rg/native.py',REPO/'src/model_rg/physical_native.py']}
    rows=[];raw={};start=time.perf_counter()
    for c in json.loads((study/'data/physical.json').read_text())['cells']:
        if c['split']!='test' or c['temperature_ratio']!=1. or c['L'] not in [8,16,24]:continue
        if sha256(c['path'])!=c['sha256']:raise ValueError('Changed test configurations')
        L,q=c['L'],c['q'];x=np.asarray(np.memmap(c['path'],mode='r',dtype=np.uint8,shape=tuple(c['shape'])))[:,:4].reshape(-1,L,L)
        common=[];energies=[];rowenergies=[];gnorm=[];diffA=[];diffG=[];meansA=[];meansG=[]
        for begin in range(0,len(x),4):
            ops=[]
            for site in [L*L//2,L*L-1]:
                inp=prefix_batch(x[begin:begin+4],site,context_tokens(proc,q,L,1.),alphabet[:q],a.device)
                _,output=selected_forward(adapter,inp,alphabet[:q],capture=True)
                A=torch.stack([att[0] for att in output.pldr_attentions],1).double();G=torch.stack([att[5] for att in output.pldr_attentions],1).double()
                if begin==0:
                    raw[f"cell{c['id']}-j{site}-A"]=A[:2].float().cpu().numpy();raw[f"cell{c['id']}-j{site}-G"]=G[:2].float().cpu().numpy()
                ops.append((A,G))
            A,G=ops[1];A0,G0=ops[0];mu=A.mean(-2);z=A-mu.unsqueeze(-2)
            common.append(mu.cpu().numpy());energies.append(A.square().mean((-1,-2)).cpu().numpy());rowenergies.append(z.square().mean((-1,-2)).cpu().numpy());gnorm.append(G.square().mean((-1,-2)).cpu().numpy())
            diffA.append((A-A0).square().mean((-1,-2)).cpu().numpy());diffG.append((G-G0).square().mean((-1,-2)).cpu().numpy());meansA.append(A0.mean((-1,-2)).cpu().numpy());meansG.append(G0.mean((-1,-2)).cpu().numpy())
        mu=np.concatenate(common);en=np.concatenate(energies);rn=np.concatenate(rowenergies);ga=np.concatenate(gnorm);da=np.concatenate(diffA);dg=np.concatenate(diffG);ma=np.concatenate(meansA);mg=np.concatenate(meansG)
        raw[f"cell{c['id']}-common"]=mu;raw[f"cell{c['id']}-energy"]=en;raw[f"cell{c['id']}-row-energy"]=rn
        raw[f"cell{c['id']}-paired-A-MSE"]=da;raw[f"cell{c['id']}-paired-G-MSE"]=dg;raw[f"cell{c['id']}-short-A-mean"]=ma;raw[f"cell{c['id']}-short-G-mean"]=mg
        rf=rn/np.maximum(en,1e-300);between=np.square(mu-mu.mean(0)).mean();total=np.square(mu).mean()
        rows.append(dict(cell=c['id'],q=q,L=L,samples=len(x),row_fraction_median=float(np.median(rf)),row_fraction_max=float(rf.max()),between_input_common_fraction=float(between/max(total,1e-300)),common_rms=float(np.sqrt(total)),common_input_rms=float(np.sqrt(between)),paired_A_rms=float(np.sqrt(da.mean())),paired_G_rms=float(np.sqrt(dg.mean())),short_A_mean=float(ma.mean()),short_G_mean=float(mg.mean()),order_A=float(np.sqrt(da.mean())/abs(ma.mean())) if ma.mean()!=0 else None,order_G=float(np.sqrt(dg.mean())/abs(mg.mean())) if mg.mean()!=0 else None))
        print(a.name,q,L,rows[-1],flush=True)
    np.savez_compressed(out/'native-geometry.npz',**raw)
    if sources!={n:sha256(REPO/n) for n in sources}:raise ValueError('Changed executing source')
    write_json(out/'result.json',dict(status='complete',identity=identity,sources=sources,rows=rows,raw_sha256=sha256(out/'native-geometry.npz'),seconds=time.perf_counter()-start,scope='Native residual metric A after the generator and native G, all five layers and fourteen heads. Same physical configuration couples half and final proper prefixes. Independent input variation is distinct from within-A row dispersion.'))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--name',required=True);p.add_argument('--device',required=True);main(p.parse_args())
