import itertools
import numpy as np
from scripts.measure_resource_windows import index_tv


def test_index_distance_by_enumeration():
    for n,k in [(4,0),(4,1),(4,3),(4,4)]:
        paths=list(itertools.product(range(n),repeat=k))
        distinct=[x for x in paths if len(set(x))==len(x)]
        tv=.5*sum(abs(1/len(paths)-(1/len(distinct) if x in distinct else 0)) for x in paths)
        assert abs(tv-index_tv(n,k))<1e-14


def test_anisotropic_correlation_transfer_bound():
    rng=np.random.default_rng(91811)
    for _ in range(20):
        x=rng.normal(size=(5,5));r=x@x.T+np.eye(5);diag=np.sqrt(np.diag(r));r=r/np.outer(diag,diag)
        d=np.diag(np.exp(rng.normal(size=5)*2));e=np.diag(rng.uniform(-.04,.04,5))
        direction=rng.normal(size=(5,5));direction=(direction+direction.T)*.005
        rhat=r+direction;dhat=d@(np.eye(5)+e)
        truth=d@r@d;prediction=dhat@rhat@dhat
        values,vectors=np.linalg.eigh(truth);inv=(vectors/np.sqrt(values))@vectors.T
        actual=np.linalg.norm(inv@(prediction-truth)@inv,2)
        a=np.linalg.norm(e,2);shape=np.linalg.norm(direction,2)
        bound=((2*a+a*a)*np.linalg.norm(r,2)+(1+a)**2*shape)/np.linalg.eigvalsh(r).min()
        assert actual<=bound+1e-10


def test_rank_event_transports_simultaneously_and_enclosures_can_widen():
    # Nine possible labels for the next path enumerate its exchangeable rank.
    paths=np.arange(9.)[:,None]*np.array([[1.,-2.,.5]])
    center=np.array([.2,.1,-.3]);scale=np.array([1.,2.,.5]);scores=np.max(abs(paths-center)/scale,axis=1)
    b=np.array([[1.,-1.,0.],[0.,1.,-1.]])
    covered=0
    for future in range(9):
        radius=np.delete(scores,future).max()
        if scores[future]<=radius:
            covered+=1
            assert np.all(abs(b@(paths[future]-center))<=radius*abs(b)@scale+1e-12)
    assert covered==8
    # Repeated coordinate hulls lose correlations, even for exact cancelling maps.
    b=np.array([[1.],[1.]]);a=np.array([[1.,-1.]])
    assert (abs(a)@abs(b))[0,0]==2 and abs(a@b)[0,0]==0
