"""Common observation coordinates and source bindings for controlled studies."""
from pathlib import Path
import subprocess
import numpy as np
import torch
from model_rg.provenance import sha256, source_manifest, environment, write_json


def device_name(value):
    value = str(value).strip()
    return 'cuda:'+value if value.isdigit() else value


def bind_run(out, inputs, arguments):
    repo = Path(__file__).resolve().parents[2]
    sources = source_manifest()
    for name, digest in sources.items():
        target = Path(out)/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (repo/name).read_bytes()
        target.write_bytes(payload)
        if sha256(target) != digest: raise RuntimeError('Source changed during snapshot')
    record = dict(arguments=arguments, inputs={str(Path(p).resolve()):sha256(p) for p in inputs},
        source_files=sources, environment=environment(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True))
    write_json(Path(out)/'binding.json', record)
    return record


def collective_names():
    return ([f'{kind}/layer{l}' for kind in ['attention_entropy','head_rms'] for l in range(5)]
        + [f'hidden_rms/stage{l}' for l in range(6)]
        + [f'logit_projection/{k}' for k in range(8)] + ['entropy','nll']
        + [f'{kind}/layer{l}' for kind in ['row_ratio','g_rms'] for l in range(5)])


def collectives(m, ids, targets, matrices=False):
    """Thirty-six recorded coordinates, including the exported scaled embedding."""
    out = m.forward(ids, capture=True)
    z = out.logits[:, -1]
    logp = z.log_softmax(-1); p = logp.exp()
    att, head, row, grms, gs = [], [], [], [], []
    for l, values in enumerate(out.pldr_attentions):
        a, _, _, _, _, g, weights = values
        last = weights[:, :, -1]
        att.append(-(last*last.clamp_min(1e-38).log()).sum(-1).mean(1))
        head.append(m.head_outputs[l].mean(1))
        ac = a-a.mean(-2,keepdim=True)
        row.append((ac.square().mean((-2,-1))/a.square().mean((-2,-1)).clamp_min(1e-30)).mean(1))
        grms.append(g.square().mean((-2,-1)).sqrt().mean(1))
        if matrices: gs.append(g.detach().double())
    hidden = [h[:, -1].square().mean(-1).sqrt() for h in out.hidden_states]
    # A separate RNG makes the vocabulary projection independent of residual width.
    if not hasattr(m, '_common_logit_projection'):
        rng = np.random.default_rng(630061)
        pr = rng.normal(size=(m.config.vocab_size,8))/np.sqrt(m.config.vocab_size)
        pr -= pr.mean(0,keepdims=True)
        m._common_logit_projection = torch.tensor(pr,device=z.device,dtype=z.dtype)
    nll = -logp.gather(1,targets[:,None]).squeeze(1)
    entropy = -(p*logp).sum(-1)
    fields = torch.cat([torch.stack(att+head+hidden,1), z@m._common_logit_projection,
                        entropy[:,None], nll[:,None],torch.stack(row+grms,1)],1)
    return fields, gs


def stable_kl(z, zz):
    """KL(p(z)||p(zz)) using a centered logit increment in float64."""
    z, zz = z.double(), zz.double()
    p = z.softmax(-1); d = zz-z
    d = d-(p*d).sum(-1,keepdim=True)
    residual=torch.where(d.abs()<1e-3,
        d.square()*(.5+d*(1/6+d*(1/24+d*(1/120+d/720)))),torch.expm1(d)-d)
    mean=(p*d).sum(-1)
    return torch.log1p(mean+(p*residual).sum(-1))-mean
