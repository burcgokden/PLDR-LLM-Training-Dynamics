#!/usr/bin/env python
"""Cross trained shared metric learners and recipient models at fixed observations."""
import argparse
import gc
import json
from pathlib import Path
import time
import numpy as np
import torch
from model_rg.controlled import bind_run
from model_rg.criticality import observations,predictive_kl
from model_rg.provenance import sha256,write_json
from model_rg.training import TrainingModel


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--protocol',required=True);parser.add_argument('--case',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root);protocol_path=Path(args.protocol)
    protocol=json.loads(protocol_path.read_text());case=next(x for x in protocol['cases'] if x['name']==args.case)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    source=root/'assets/PLDR-LLM-v51-SOC-110M-1';probe=root/'controlled-study-20260905/data/short'
    checkpoints=[Path(p) for p in case['checkpoints']]
    bind_run(out,[protocol_path,*checkpoints,source/'modeling_pldrllm.py',source/'configuration_pldrllm.py',probe/'tokens.npy',probe/'offsets.npy'],vars(args))
    torch.set_num_threads(4);ptokens=np.load(probe/'tokens.npy');offsets=np.load(probe/'offsets.npy')
    rows=np.array(protocol['rows']);crops=ptokens[rows[:,None],offsets[rows,None]+np.arange(65)]
    ids=torch.tensor(crops[:,:64],dtype=torch.long);targets=torch.tensor(crops[:,64],dtype=torch.long)
    shared=[];conditions=[]
    for checkpoint in checkpoints:
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
        conditions.append(saved['arguments']);shared.append({n:p.clone() for n,p in saved['model'].items() if 'reslayerAs' in n})
        if saved['step']!=case['step']:raise AssertionError('Crossed model horizon changed')
        del saved
    states={};fields=[];head_fields=[];kls=[];baseline_fields=[];started=time.time()
    for recipient,checkpoint in enumerate(checkpoints):
        condition=conditions[recipient]
        model=TrainingModel(source,condition['heads'],condition['seed'],'cpu')
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=True)
        model.model.load_state_dict(saved['model']);del saved;model.model.eval()
        def evaluate():
            fs=[];hs=[];zs=[]
            with torch.no_grad():
                for start in range(0,len(ids),16):
                    f,h,z,_=observations(model,ids[start:start+16],targets[start:start+16])
                    fs.append(f.numpy());hs.append(h.numpy());zs.append(z.double())
            return np.concatenate(fs),np.concatenate(hs),torch.cat(zs)
        original_f,original_h,original_z=evaluate();baseline_fields.append(original_f)
        donor_f=[];donor_h=[];donor_kl=[]
        for donor in range(len(checkpoints)):
            with torch.no_grad():
                for name,parameter in model.model.named_parameters():
                    if name in shared[donor]:parameter.copy_(shared[donor][name])
            f,h,z=evaluate()
            if donor==recipient:
                np.testing.assert_array_equal(f,original_f);np.testing.assert_array_equal(h,original_h)
                torch.testing.assert_close(z,original_z,rtol=0,atol=0)
            donor_f.append(f);donor_h.append(h);donor_kl.append(predictive_kl(original_z,z).numpy())
            print(case['name'],'recipient',recipient,'donor',donor,'mean_R',float(h[...,2].mean()),'seconds',round(time.time()-started,1),flush=True)
        fields.append(donor_f);head_fields.append(donor_h);kls.append(donor_kl)
        del model;gc.collect()
    fields=np.array(fields);head_fields=np.array(head_fields);kls=np.array(kls)
    raw=dict(cohort=rows,crops=crops,fields=fields,head_fields=head_fields,predictive_kl=kls,baseline_fields=np.array(baseline_fields))
    summaries={}
    for name,index in [('entropy',0),('row_energy',2)]:
        q=head_fields[...,index].mean(-1).astype(float)
        grand=q.mean((0,1));recipient=q.mean(1)-grand;donor=q.mean(0)-grand
        interaction=q-grand[None,None]-recipient[:,None]-donor[None,:]
        total=np.mean((q-grand[None,None])**2)
        pieces=[float(np.mean(recipient**2)),float(np.mean(donor**2)),float(np.mean(interaction**2))]
        np.testing.assert_allclose(total,sum(pieces),rtol=1e-12,atol=1e-15)
        factual=np.stack([q[i,i] for i in range(len(checkpoints))])
        summaries[name]=dict(mean_matrix=q.mean((2,3)).tolist(),total_crossed_variance=float(total),
            recipient_variance=pieces[0],donor_variance=pieces[1],interaction_variance=pieces[2],
            factual_seed_variance=float(np.var(factual,axis=0,ddof=0).mean()),
            fractions=[x/total for x in pieces] if total>0 else None)
    for name,array in raw.items():
        if not np.isfinite(array).all():raise AssertionError('Nonfinite crossed intervention: '+name)
    np.savez_compressed(out/'measurements.npz',**raw)
    result=dict(schema='shared-generator-crossing-v1',case=case,conditions=conditions,summaries=summaries,seconds=time.time()-started,
        mean_predictive_kl=kls.mean(2).tolist(),mean_nll=fields[...,25].mean(2).tolist(),
        interpretation='Full native CPU float32 inference with trained shared metric learner transplanted between recipients. Four recipient and four donor identities give a balanced finite counterfactual product law, not sixteen independent trained models. Exact finite ANOVA separates recipient, donor and interaction variance. This intervention identifies component influence without declaring the learned shared collective noncritical.')
    write_json(out/'results.json',result)
    write_json(out/'manifest.json',dict(status='complete',results_sha256=sha256(out/'results.json'),raw_sha256=sha256(out/'measurements.npz'),binding_sha256=sha256(out/'binding.json')))


if __name__=='__main__':main()
