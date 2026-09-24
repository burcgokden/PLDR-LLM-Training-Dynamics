import copy
import numpy as np
import pytest
import torch
from model_rg.replay_equality import logical_equal, require_replay, require_observations


@pytest.mark.parametrize('make',[torch.tensor,np.array])
def test_signed_zero_dtype_and_shape(make):
    assert not logical_equal(make([0.]),make([-0.]))
    a=make([1.])
    b=a.double() if torch.is_tensor(a) else a.astype(np.float32)
    assert not logical_equal(a,b)
    assert not logical_equal(a,a.reshape(1,1))


@pytest.mark.parametrize('a,b',[(True,1),([1],(1,)),({True:1},{1:1}),(0.,-0.)])
def test_typed_containers_and_scalars(a,b):assert not logical_equal(a,b)


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16,torch.float32,torch.float64,torch.int64,torch.bool,torch.complex64])
def test_dense_scalar_empty_and_noncontiguous(dtype):
    for x in [torch.zeros((),dtype=dtype),torch.empty((0,2),dtype=dtype),torch.ones((2,3),dtype=dtype).T]:
        require_replay(x,x.contiguous())


def test_nan_representation_is_not_successful_state():
    a=np.array([0x7ff8000000000001],dtype=np.uint64).view(np.float64)
    b=np.array([0x7ff8000000000002],dtype=np.uint64).view(np.float64)
    assert logical_equal(a,a.copy()) and not logical_equal(a,b)
    with pytest.raises(ValueError,match='Nonfinite'):require_replay(a,a.copy())


def test_optimizer_mutation_byte_order_and_unsupported():
    a={'state':{0:{'exp_avg':torch.zeros(2)}}};b=copy.deepcopy(a);b['state'][0]['exp_avg'][0]=-0.
    assert not logical_equal(a,b)
    assert not logical_equal(np.array([1.],dtype='<f8'),np.array([1.],dtype='>f8'))
    with pytest.raises(TypeError):logical_equal(np.array([{}],dtype=object),np.array([{}],dtype=object))
    with pytest.raises(TypeError):logical_equal(torch.eye(2).to_sparse(),torch.eye(2).to_sparse())


def test_missing_common_payload_is_not_an_allowed_recorder_addition(tmp_path):
    np.savez(tmp_path/'a.npz',heads=np.ones(2),steps=np.arange(2))
    np.savez(tmp_path/'b.npz',heads=np.ones(2))
    with np.load(tmp_path/'a.npz') as a,np.load(tmp_path/'b.npz') as b:
        with pytest.raises(ValueError,match='fields'):require_observations(a,b)


def test_independent_ieee_bit_pattern_oracle():
    # Expected equality and finiteness come from declared IEEE-754 encodings,
    # independently of the comparator's tensor/array byte conversion.
    words=np.array([0x00000000,0x80000000,0x3f800000,0xbf800000,
                    0x7f800000,0x7fc00001,0x7fc00002],dtype=np.uint32)
    values=words.view(np.float32)
    for i,left in enumerate(words):
        for j,right in enumerate(words):
            expected=int(left)==int(right)
            a=values[i:i+1].copy();b=values[j:j+1].copy()
            assert logical_equal(a,b) is expected
            assert logical_equal(torch.from_numpy(a),torch.from_numpy(b)) is expected
        finite=(int(left)&0x7f800000)!=0x7f800000
        a=torch.from_numpy(values[i:i+1].copy())
        if finite:require_replay(a,a.clone())
        else:
            with pytest.raises(ValueError,match='Nonfinite'):require_replay(a,a.clone())
