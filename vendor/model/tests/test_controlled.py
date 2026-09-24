import numpy as np
import torch
from model_rg.correlations import geometric_block, geometric_lag
from model_rg.controlled import stable_kl, device_name


def test_correlated_block_retains_nonsymmetric_tails_and_composes():
    rng=np.random.default_rng(419)
    c0=10*np.eye(3);a=rng.normal(size=(3,2));shared=a@a.T/10
    tails=rng.normal(size=(2,3,3))*.02;poles=np.array([.2,.7]);state=(c0,shared,tails,poles)
    fine=np.block([[geometric_lag(*state,j-i) for j in range(18)] for i in range(18)])
    assert np.linalg.eigvalsh(fine).min()>0
    b=3;H=.37;matrix=np.kron(np.eye(6),np.ones((1,b)))/b**H
    blockmap=np.kron(matrix,np.eye(3));observed=blockmap@fine@blockmap.T
    coarse=geometric_block(*state,b,H)
    predicted=np.block([[geometric_lag(*coarse,j-i) for j in range(6)] for i in range(6)])
    np.testing.assert_allclose(observed,predicted,atol=1e-12)
    twice=geometric_block(*coarse,2,H);direct=geometric_block(*state,6,H)
    for x,y in zip(twice,direct):np.testing.assert_allclose(x,y,atol=1e-12)
    # Imposing a fresh AR relation would lose the retained lag-zero information.
    assert not np.allclose(coarse[2].sum(0),coarse[3].mean()*(coarse[0]-coarse[1]))


def test_stable_kl_resolves_cubic_transport_and_gauge():
    gen=torch.Generator().manual_seed(481)
    z=torch.randn(2,37,generator=gen,dtype=torch.float64)
    v=torch.randn(2,37,generator=gen,dtype=torch.float64)
    w=torch.randn(2,37,generator=gen,dtype=torch.float64)
    p=z.softmax(-1);vc=v-(p*v).sum(-1,keepdim=True);wc=w-(p*w).sum(-1,keepdim=True)
    q=(p*vc**2).sum(-1);c=(p*vc**3).sum(-1)+3*(p*vc*wc).sum(-1)
    e=.0001;zz=z+e*v+.5*e*e*w
    actual=stable_kl(z,zz);expected=.5*e*e*q+e**3*c/6
    torch.testing.assert_close(actual,expected,rtol=2e-7,atol=1e-16)
    torch.testing.assert_close(stable_kl(z+3,zz-7),actual,rtol=1e-8,atol=1e-17)
    assert torch.all(stable_kl(z,z)==0)


def test_adam_bound_handles_sign_changing_gradients():
    rng=np.random.default_rng(433);g=rng.normal(size=(250,20));g[10:30]=0
    b1,b2=.9,.95;m=np.zeros(20);v=np.zeros(20)
    for t,gradient in enumerate(g,1):
        m=b1*m+(1-b1)*gradient;v=b2*v+(1-b2)*gradient**2
        ratio=np.abs(m/(1-b1**t))/(np.sqrt(v/(1-b2**t))+1e-8)
        bound=(1-b1)/np.sqrt(1-b2)*np.sqrt(1-b2**t)/(1-b1**t)*np.sqrt(sum((b1*b1/b2)**k for k in range(t)))
        assert np.all(ratio<=bound+1e-12)


def test_device_argument_normalization():
    assert device_name('0')==device_name('cuda:0')=='cuda:0'
    assert device_name('cpu')=='cpu'
