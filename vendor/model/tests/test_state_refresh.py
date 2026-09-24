"""Check conditional-law drift bounds in noncommuting finite examples."""
import numpy as np

from model_rg.refresh import fit, calibrate, increments
from model_rg.moment_rg import score, block_maps


def invsqrt(c):
    w,v=np.linalg.eigh(c)
    return (v/np.sqrt(w))@v.T


def test_relative_state_drift_and_all_scale_images():
    rng=np.random.default_rng(11511)
    for d in [4,8]:
        for _ in range(32):
            a=rng.normal(size=(d,d));c0=a@a.T+np.eye(d)
            w=invsqrt(c0);root=np.linalg.inv(w)
            perturb=.003*rng.normal(size=(d,d));b=.003*rng.normal(size=d)
            noise=.002*rng.normal(size=(d,d));transition=np.eye(d)+perturb
            c1=transition@c0@transition.T+noise@noise.T
            rho=np.sqrt(np.sum((w@perturb@root)**2)+np.sum((w@noise)**2)+np.sum((w@b)**2))
            beta=2*rho+rho*rho
            assert beta<1
            forecast_cov=1.01*c0;forecast_mean=root@np.full(d,.005/np.sqrt(d))
            epsilon=(.01+beta)/(1-beta);delta=(.005+rho)/np.sqrt(1-beta)
            for block in block_maps(d//4).values():
                true=block@c1@block.T;white=invsqrt(true)
                cov=block@forecast_cov@block.T;bias=block@(forecast_mean-b)
                assert np.linalg.norm(white@(cov-true)@white,2)<=epsilon+1e-12
                assert np.linalg.norm(white@bias)<=delta+1e-12
                inv=np.linalg.inv(cov);rank=len(block)
                regret=np.linalg.slogdet(cov)[1]-np.linalg.slogdet(true)[1]+np.trace(inv@true)-rank+bias@inv@bias
                bound=rank*epsilon**2/(2*(1-epsilon)**2)+delta**2/(1-epsilon)
                assert -1e-10<=regret<=bound+1e-10


def test_quadratic_drift_term_cannot_be_dropped():
    # Y=(1+rho)X for Var(X)=1 saturates 2 rho + rho^2.
    rho=.1
    covariance_change=(1+rho)**2-1
    assert np.isclose(covariance_change,2*rho+rho*rho)
    assert covariance_change>2*rho


def test_calibration_is_whole_path_and_not_refitted_at_coarse_scale():
    rng=np.random.default_rng(9511)
    paths=np.cumsum(rng.normal(size=(24,5,2)),axis=1)
    paths-=paths[:,0:1]
    archive={t:dict(correlation=np.eye(d).tolist()) for t,d in [('risk',4),('risk_entropy',8)]}
    fitted=fit(paths[:16],archive);tube=calibrate(paths[16:],fitted)
    x=paths[16:,1:,0]-paths[16:,0,None,0]
    assert np.all(np.max(np.abs(x-tube['mean'])/tube['scale'],axis=1)<=tube['radius'])
    for b in block_maps(1).values():
        diff=np.eye(4)-np.eye(4,k=-1);mapped=b@diff
        assert np.all(np.abs((x-tube['mean'])@mapped.T)<=tube['radius']*(abs(mapped)@tube['scale'])+1e-12)
    # General correlated non-Gaussian increments are legal scoring targets.
    values=score(increments(paths[16:],1),fitted['risk']['mean'],np.asarray(fitted['risk']['covariances']['local']))
    assert np.isfinite(values).all()


def test_nonstationary_covariance_retains_initial_and_forcing_cross_terms():
    rng=np.random.default_rng(221)
    n=3;t=4;mix=rng.normal(size=((t+1)*n,7));joint=mix@mix.T
    js=[rng.normal(size=(n,n))/4 for _ in range(t)]
    phi=[None]*(t+1);phi[t]=np.eye(n)
    for k in reversed(range(t)):
        phi[k]=phi[k+1]@js[k]
    transport=np.concatenate([phi[0]]+[phi[k+1] for k in range(t)],axis=1)
    direct=transport@joint@transport.T
    c=phi[0]@joint[:n,:n]@phi[0].T
    for k in range(t):
        h=joint[:n,(k+1)*n:(k+2)*n]
        cross=phi[0]@h@phi[k+1].T;c+=cross+cross.T
        for j in range(t):
            q=joint[(k+1)*n:(k+2)*n,(j+1)*n:(j+2)*n]
            c+=phi[k+1]@q@phi[j+1].T
    np.testing.assert_allclose(c,direct,atol=1e-12)


def test_acquisition_noise_is_a_shared_sector():
    import itertools
    # Exact uniform product law: two fitting draws and two independent futures.
    outcomes=np.array(list(itertools.product([-1.,1.],repeat=4)))
    fitted=outcomes[:,:2].mean(1)
    z=outcomes[:,2:]-fitted[:,None]
    covariance=z.T@z/len(z)
    np.testing.assert_array_equal(covariance,np.array([[1.5,.5],[.5,1.5]]))
    assert np.mean(fitted**2)==.5
    # A common fit error disappears in the difference of the two futures.
    assert np.mean((z[:,0]-z[:,1])**2)==2.
