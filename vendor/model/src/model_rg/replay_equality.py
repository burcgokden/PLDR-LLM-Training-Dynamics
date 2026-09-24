"""Logical C-order byte equality for the declared native checkpoint schema.

Device, strides, allocator and serialization layout are outside this contract.
Representation equality admits identical NaN payloads; scientific admission
requires the separate finite-state check. Unsupported leaves fail closed.
"""
import struct
from collections import OrderedDict
import numpy as np
import torch
from model_rg.provenance import sha256

SCHEMA = 'native-logical-bytes-v1'


def contract():
    return dict(schema=SCHEMA, comparator_sha256=sha256(__file__),
                order='C logical elements', device_identity=False,
                successful_state_requires_finite=True)


def _tensor_bytes(x):
    if x.layout != torch.strided or x.is_quantized or x.device.type == 'meta':
        raise TypeError('Unsupported tensor representation')
    return x.detach().resolve_conj().resolve_neg().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


def logical_equal(a, b):
    if type(a) is not type(b):
        return False
    if type(a) is torch.Tensor:
        return a.dtype == b.dtype and a.shape == b.shape and _tensor_bytes(a) == _tensor_bytes(b)
    if type(a) is np.ndarray:
        if a.dtype.hasobject or a.dtype.fields is not None:
            raise TypeError('Object and structured arrays have no declared byte contract')
        return a.dtype.str == b.dtype.str and a.shape == b.shape and a.tobytes(order='C') == b.tobytes(order='C')
    if type(a) in (dict, OrderedDict):
        # Typed keys distinguish, for example, integer 1 from boolean True.
        keys = lambda x: {(type(k), k) for k in x}
        for k in (*a.keys(), *b.keys()):
            if type(k) not in (str, int, bool, bytes):
                raise TypeError('Unsupported dictionary key')
        return keys(a) == keys(b) and all(logical_equal(a[k], b[k]) for k in a)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(logical_equal(x, y) for x, y in zip(a, b))
    if type(a) is float:
        return struct.pack('!d', a) == struct.pack('!d', b)
    if type(a) is complex:
        return logical_equal(a.real, b.real) and logical_equal(a.imag, b.imag)
    if type(a) in (str, bytes, bool, int, type(None)):
        return a == b
    if isinstance(a, np.generic) and not a.dtype.hasobject and a.dtype.fields is None:
        return a.dtype.str == b.dtype.str and a.tobytes() == b.tobytes()
    raise TypeError('Unsupported checkpoint leaf: '+type(a).__name__)


def finite_state(value):
    if type(value) is torch.Tensor:
        _tensor_bytes(value)
        return bool(torch.isfinite(value).all())
    if type(value) is np.ndarray or isinstance(value, np.generic):
        if value.dtype.hasobject or value.dtype.fields is not None:
            raise TypeError('Unsupported array')
        return bool(np.isfinite(value).all()) if value.dtype.kind in 'biufc' else True
    if type(value) in (dict, OrderedDict):
        return all(finite_state(v) for v in value.values())
    if type(value) in (list, tuple):
        return all(finite_state(v) for v in value)
    if type(value) in (float, complex):
        return bool(np.isfinite(value))
    if type(value) in (str, bytes, bool, int, type(None)):
        return True
    raise TypeError('Unsupported checkpoint leaf: '+type(value).__name__)


def require_replay(a, b):
    if not logical_equal(a, b):
        raise ValueError('Logical representation replay mismatch')
    if not finite_state(a) or not finite_state(b):
        raise ValueError('Nonfinite replay state is not admissible')


def require_observations(a, b):
    """Compare the complete common NPZ payload; recorder files are separate."""
    if set(a.files) != set(b.files):
        raise ValueError('Missing or additional common observation fields')
    for key in a.files:
        require_replay(a[key], b[key])
