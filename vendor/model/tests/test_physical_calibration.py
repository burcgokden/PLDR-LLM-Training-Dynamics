import itertools
import numpy as np
import pytest
from model_rg.lattice import (critical_temperature,dyadic_block,exact_small,
                               observables,prefix_tv_bound,slope_bound)


def test_exact_ising_temperature_conversion():
    assert critical_temperature(2)==pytest.approx(1/np.log1p(np.sqrt(2)))
    configurations=np.array(list(itertools.product([0,1],repeat=4))).reshape(-1,2,2)
    e=observables(configurations,2)['energy']
    spin=2*configurations-1
    product=-(spin*np.roll(spin,1,axis=1)+spin*np.roll(spin,1,axis=2)).sum((1,2))
    np.testing.assert_array_equal(e,(product-8)/2)


@pytest.mark.parametrize('q',[2,3])
def test_histogram_shuffle_and_spatial_separation(q):
    rng=np.random.default_rng(218)
    x=np.tile(np.arange(16).reshape(4,4)%q,(17,1,1)).astype('uint8')
    shuffled=x.copy().reshape(17,-1)
    for row in shuffled:rng.shuffle(row)
    shuffled=shuffled.reshape(x.shape)
    a,b=observables(x,q),observables(shuffled,q)
    for k in ['m','m2','m4','field']:np.testing.assert_array_equal(a[k],b[k])
    assert not np.array_equal(a['corr1'],b['corr1'])


def test_color_relabeling_preserves_moments():
    x=np.random.default_rng(90).integers(0,3,(20,8,8))
    for perm in itertools.permutations(range(3)):
        y=np.array(perm)[x]
        for key in ['m','m2','m4','energy','corr1']:
            np.testing.assert_allclose(observables(x,3)[key],observables(y,3)[key],atol=1e-14)


def test_prefix_chain_rule_and_pinsker():
    rng=np.random.default_rng(902)
    p=rng.dirichlet(np.ones(8));q=rng.dirichlet(np.ones(8))
    p=p.reshape(2,2,2);q=q.reshape(2,2,2)
    terms=[]
    for j in range(3):
        axes=tuple(range(j+1,3))
        pj=p.sum(axes) if axes else p;qj=q.sum(axes) if axes else q
        pp=pj.sum(-1,keepdims=True);qp=qj.sum(-1,keepdims=True)
        terms.append(np.sum(pj*np.log((pj/pp)/(qj/qp))))
    joint=np.sum(p*np.log(p/q))
    assert sum(terms)==pytest.approx(joint)
    tv=.5*np.abs(p-q).sum()
    assert tv<=prefix_tv_bound(joint/3,3)


def test_two_size_bound_is_attained():
    eps=.2;b=2.;truth=np.array([3.,12.]);approx=truth*np.array([1-eps,1+eps])
    error=abs(np.log(approx[1]/approx[0])/np.log(b)-np.log(truth[1]/truth[0])/np.log(b))
    assert error==pytest.approx(slope_bound(eps,b))


def test_invisible_mode_cannot_be_identified():
    A=np.diag([2.,3.]);C=np.diag([1.,0.]);B=np.diag([2.,17.])
    np.testing.assert_array_equal(B@C,C@A)
    np.testing.assert_array_equal(C@np.array([0.,1.]),np.zeros(2))
    assert 3. not in np.linalg.eigvals(B)


def test_stochastic_blocking_preserves_ordered_states():
    x=np.stack([np.full((8,8),i,dtype='uint8') for i in range(3)])
    b=dyadic_block(x,3,11)
    np.testing.assert_array_equal(b,np.stack([np.full((4,4),i) for i in range(3)]))


def test_uniform_high_temperature_exact_limit():
    # Finite enumeration gives E[m^2] = 1/V for independent uniform colors.
    for q in [2,3]:
        exact=exact_small(q,2,1e12)
        assert exact['m2']==pytest.approx(.25,abs=1e-11)
        assert exact['field']==pytest.approx(0.,abs=1e-14)
