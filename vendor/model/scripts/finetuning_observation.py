"""Fixed-prefix inference observations of native single-pass fine-tuning."""
import numpy as np
import torch


@torch.no_grad()
def observe(model, crops, indices, raw_indices, batch_size=8):
    """All tensor reductions use float64; raw logits retain their native dtype."""
    model.model.eval(); collected={}; raws=[]; positions=[]
    selected=set(map(int,raw_indices))
    def add(key, value): collected.setdefault(key,[]).append(value.detach().cpu().numpy())
    for begin in range(0,len(indices),batch_size):
        ids=indices[begin:begin+batch_size]
        x=torch.as_tensor(crops[ids],dtype=torch.long,device=model.device)
        short=model.forward(x[:,:32],capture=True)
        short_matrices=[torch.stack([att[0],att[5]],dim=2).detach() for att in short.pldr_attentions]
        z32=short.logits[:,-1].detach(); add('logits32',z32)
        del short
        full=model.forward(x[:,:64],capture=True)
        z64=full.logits[:,-1].detach(); add('logits64',z64)
        common=[]; energies=[]; totals=[]; sse=[]; refs=[]; refsq=[]; attention=[]; raw_layers=[]
        for layer,att in enumerate(full.pldr_attentions):
            a=att[0].double(); g=att[5].double(); center=a.mean(-2)
            common.append(center); energies.append((a-center.unsqueeze(-2)).square().mean((-2,-1)))
            totals.append(a.square().mean((-2,-1)))
            reference=short_matrices[layer].double()
            current=torch.stack([a,g],dim=2)
            sse.append((current-reference).square().sum((-2,-1)))
            refs.append(reference.sum((-2,-1))); refsq.append(reference.square().sum((-2,-1)))
            w=att[-1][:,:,-1].double()
            attention.append(-(w*w.clamp_min(1e-300).log()).sum(-1)/np.log(64))
            raw_layers.append(torch.stack([reference,current],dim=2).float())
        add('common',torch.stack(common,1)); add('energy',torch.stack(energies,1))
        add('total',torch.stack(totals,1)); add('paired_sse',torch.stack(sse,1))
        add('reference_sum',torch.stack(refs,1)); add('reference_sumsq',torch.stack(refsq,1))
        add('attention_entropy',torch.stack(attention,1))
        raw=torch.stack(raw_layers,1)
        for j,index in enumerate(ids):
            if int(index) in selected:
                raws.append(raw[j].cpu().numpy()); positions.append(int(index))
        del full,short_matrices,raw,raw_layers
    answer={k:np.concatenate(v) for k,v in collected.items()}
    answer.update(indices=np.asarray(indices),raw_indices=np.asarray(positions),raw_tensors=np.stack(raws))
    if not all(np.isfinite(v).all() for v in answer.values()): raise FloatingPointError('Nonfinite observation')
    model.capture=False;model.head_outputs={};model.eta=None
    return answer
