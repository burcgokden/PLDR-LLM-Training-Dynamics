"""Passive all-coordinate activity of native deductive tensors at fixed inputs."""
import numpy as np
import torch
from model_rg.controlled import stable_kl

TENSOR_NAMES=['metric_A','positive_base_M','exponent_P','potential_V','curvature_G']
STAT_NAMES=['rms','increment_rms','mean','increment_mean','row_contrast_ratio']

class DeductiveActivity:
    def __init__(self,model,probes,coordinates):
        self.model=model;self.probes=probes;self.coordinates=coordinates
        self.previous=None;self.previous_logits=None;self.stats=[];self.samples=[];self.predictive=[]

    @torch.no_grad()
    def snapshot(self):
        self.model.model.eval()
        out=self.model.forward(self.probes[:,:64],capture=True)
        layers=[]
        for a,m,p,_,_,g,_ in out.pldr_attentions:
            pp=p[None].expand_as(m)
            layers.append(torch.stack([a,m,pp,torch.pow(m,pp),g],dim=2).double())
        values=torch.stack(layers,dim=1)
        delta=torch.zeros_like(values) if self.previous is None else values-self.previous
        rms=lambda x:x.square().mean((-2,-1)).sqrt()
        row=(values-values.mean(-2,keepdim=True)).square().mean((-2,-1))/values.square().mean((-2,-1)).clamp_min(1e-30)
        stat=torch.stack([rms(values),rms(delta),values.mean((-2,-1)),delta.mean((-2,-1)),row],-1)
        z=out.logits[:,-1]
        kl=torch.zeros(len(z),device=z.device,dtype=torch.float64) if self.previous_logits is None else stable_kl(self.previous_logits,z)
        if not all(torch.isfinite(x).all() for x in [values,stat,kl]):raise FloatingPointError('Nonfinite deductive observation')
        self.stats.append(stat.cpu().numpy())
        self.samples.append(values.flatten(-2).index_select(-1,self.coordinates).float().cpu().numpy())
        self.predictive.append(kl.cpu().numpy())
        self.previous=values;self.previous_logits=z.detach().clone()

    def arrays(self):
        return dict(tensor_statistics=np.asarray(self.stats),tensor_samples=np.asarray(self.samples),
                    predictive_step_kl=np.asarray(self.predictive))
