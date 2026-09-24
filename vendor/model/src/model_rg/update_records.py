"""Read-only optimizer hooks for fixed-unit shared-parameter time series."""
from pathlib import Path
import numpy as np
import torch


class SharedUpdateRecorder:
    def __init__(self, model, optimizer, projection_path, expected_names):
        self.parameters=[p for n,p in model.named_parameters() if 'reslayerAs' in n]
        self.names=[n for n,p in model.named_parameters() if 'reslayerAs' in n]
        if self.names!=expected_names:
            raise AssertionError('Shared parameter coordinate order changed')
        count=sum(p.numel() for p in self.parameters)
        signs=np.load(Path(projection_path))
        if signs.ndim!=2 or signs.shape[0]!=count or not np.isin(signs,[-1,1]).all():
            raise AssertionError('The declared shared projection changed')
        self.projection=torch.as_tensor(signs,device=self.parameters[0].device,dtype=torch.float64)/np.sqrt(count)
        self.before=None
        self.updates=[];self.gradients=[];self.moments=[]
        self.handles=[optimizer.register_step_pre_hook(self._before),optimizer.register_step_post_hook(self._after)]

    @torch.no_grad()
    def _before(self, optimizer, args, kwargs):
        if self.before is not None:
            raise RuntimeError('Unpaired optimizer observation hook')
        self.before=torch.cat([p.detach().reshape(-1) for p in self.parameters]).double()
        gradient=torch.cat([p.grad.detach().reshape(-1) for p in self.parameters]).double()
        self.gradients.append((gradient@self.projection).cpu().numpy())

    @torch.no_grad()
    def _after(self, optimizer, args, kwargs):
        if self.before is None:
            raise RuntimeError('Missing optimizer pre-step observation')
        after=torch.cat([p.detach().reshape(-1) for p in self.parameters]).double()
        delta=after-self.before
        self.updates.append((delta@self.projection).cpu().numpy())
        self.moments.append(torch.stack([delta.square().mean(),self.before.square().mean(),
                                        (delta*self.before).mean()]).cpu().numpy())
        self.before=None

    def arrays(self):
        if self.before is not None or len(self.updates)!=len(self.gradients):
            raise AssertionError('Incomplete update observations')
        return dict(shared_update_projection=np.asarray(self.updates),
                    shared_clipped_gradient_projection=np.asarray(self.gradients),
                    shared_update_moments=np.asarray(self.moments))

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[]
