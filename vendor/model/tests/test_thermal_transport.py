"""Independent finite-law tests of spatial and ratio transport, including cancellation."""
import numpy as np
import pytest

@pytest.mark.parametrize('q',[2,3,5])
def test_nonstationary_color_law_requires_local_means(q):
    rng=np.random.default_rng(91+q);x=rng.integers(q,size=(200,9));x[:150,0]=0;x[:100,1]=0
    for i,j in [(0,1),(0,5),(3,7)]:
        a=np.eye(q)[x[:,i]];b=np.eye(q)[x[:,j]]
        connected=q/(q-1)*np.mean((a-a.mean(0))*(b-b.mean(0)),axis=0).sum()
        profile=q/(q-1)*((a.mean(0)-1/q)*(b.mean(0)-1/q)).sum()
        raw=(q*np.mean(x[:,i]==x[:,j])-1)/(q-1)
        assert abs(raw-connected-profile)<2e-14


def test_ratio_budget_and_signed_cancellation_on_random_laws():
    rng=np.random.default_rng(92);m2=np.linspace(0,1,31)
    for _ in range(300):
        p=rng.dirichlet(np.ones(31));q=rng.dirichlet(np.ones(31))
        a,b,c,d=p@m2,p@m2**2,q@m2,q@m2**2
        signed=(a*a*(d-b)-b*(c-a)*(c+a))/(a*a*c*c)
        delta=d/c**2-b/a**2;bound=abs(d-b)/c**2+b*abs(c-a)*(c+a)/(a*a*c*c)
        assert abs(delta-signed)<1e-12 and abs(delta)<=bound+1e-12
        e2=abs(c-a)/a;e4=abs(d-b)/b
        if e2<1:assert abs(delta)<=(b/a**2)*(e4+2*e2+e2**2)/(1-e2)**2+1e-12


def test_shrinking_step_can_preserve_nonzero_contrast_error():
    # Endpoint errors vanish but their finite-difference error stays one.
    for h in [1e-1,1e-3,1e-5]:
        source=(1.,1.);model=(1.-h,1.+h)
        contrast=((model[1]-model[0])-(source[1]-source[0]))/(2*h)
        assert abs(contrast-1)<1e-10
