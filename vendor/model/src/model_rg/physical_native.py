"""Native PLDR proper-prefix conditional probabilities on spin alphabets."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F


def spin_tokens(processor):
    ids=[int(processor.encode(str(i),out_type=int)[-1]) for i in range(3)]
    if len(set(ids))!=3 or min(ids)<3:
        raise ValueError('Distinct non-special spin tokens required')
    return ids


def context_tokens(processor,q,L,ratio):
    text=f'Potts q={q}, lattice {L}x{L}, temperature ratio {ratio:.3f}: '
    return [int(processor.bos_id())]+processor.encode(text,out_type=int)


def prefix_batch(configurations, site, metadata, alphabet, device):
    flat=np.asarray(configurations).reshape(len(configurations),-1)
    if not 0<=site<flat.shape[1]:raise ValueError('Invalid proper prefix')
    spins=np.asarray(alphabet,dtype=np.int64)[flat[:,:site]]
    meta=np.broadcast_to(np.asarray(metadata),(len(flat),len(metadata)))
    return torch.as_tensor(np.concatenate([meta,spins],1),dtype=torch.long,device=device)


def selected_forward(adapter, inputs, alphabet, capture=False, eta=None):
    adapter.eta=eta;adapter.capture=capture;adapter.head_outputs={}
    output=adapter.model.decoder(input_ids=inputs,use_cache=False,
        output_pldr_attentions=capture,output_hidden_states=capture)
    hidden=output.last_hidden_state[:,-1]
    layer=adapter.model.final_layer
    ids=torch.as_tensor(alphabet,dtype=torch.long,device=inputs.device)
    logits=F.linear(hidden,layer.weight[ids],layer.bias[ids] if layer.bias is not None else None)
    return logits,output


@torch.no_grad()
def generate(adapter, metadata, alphabet, L, count, seed, batch_size=32, eta=None):
    adapter.model.eval();device=adapter.device
    generator=torch.Generator(device=device);generator.manual_seed(seed)
    samples=[]
    ids=torch.as_tensor(alphabet,dtype=torch.long,device=device)
    for begin in range(0,count,batch_size):
        n=min(batch_size,count-begin)
        tokens=torch.as_tensor(metadata,dtype=torch.long,device=device)[None].repeat(n,1)
        result=[]
        for site in range(L*L):
            logits,_=selected_forward(adapter,tokens,alphabet,eta=eta)
            p=logits.softmax(-1)
            draw=torch.multinomial(p,1,generator=generator)
            result.append(draw[:,0]);tokens=torch.cat([tokens,ids[draw]],1)
        samples.append(torch.stack(result,1).cpu().numpy().astype(np.uint8).reshape(n,L,L))
    adapter.eta=None
    return np.concatenate(samples)
