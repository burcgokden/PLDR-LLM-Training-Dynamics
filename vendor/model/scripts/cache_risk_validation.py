"""Primary-reduction admission for the frozen cache-risk native geometry.

The independent verifier implements its own moment reconstruction.
"""
import numpy as np
from numerical_validation import finite_array, array_discrepancy


def real_array(value, label, shape):
    value = finite_array(value, label)
    if value.shape != shape or value.size == 0:
        raise ValueError('Wrong native dimensions: ' + label)
    with np.errstate(over='ignore', invalid='ignore'):
        value = value.astype(np.float64)
    return finite_array(value, 'float64 ' + label)


def validate_operators(study, protocol):
    files = arrays = 0
    keys = {f"C{a['calibration_length']}-P{a['panel']}-M{a['size']}" for a in protocol['arms']}
    for job in protocol['jobs']:
        folder = study/'runs'/job['run_id']
        with np.load(folder/'caches.npz', allow_pickle=False) as archive:
            if set(archive.files) != keys:
                raise ValueError('Wrong cache members')
            caches = {k: real_array(archive[k], k, (5, 3, 1, job['heads'], 64, 64)) for k in keys}
            files += 1
            arrays += len(keys)
        for length in protocol['prefix_lengths']:
            arms = [a for a in protocol['arms'] if a['length'] == length]
            expected = {'mean', 'powers', 'scatter'} | {a['name']+s for a in arms for s in ['-risk', '-bias']}
            with np.load(folder/f'operators-L{length}.npz', allow_pickle=False) as archive:
                if set(archive.files) != expected:
                    raise ValueError('Wrong operator members')
                values = {}
                for k in expected:
                    shape = (5, job['heads'], 64, 64) if k == 'mean' else (5, protocol['contexts']) if k == 'powers' else (5,)
                    values[k] = real_array(archive[k], k, shape)
                    if k != 'mean' and np.any(values[k] < 0):
                        raise ValueError('Negative squared statistic: ' + k)
                files += 1
                arrays += len(expected)
            mean, powers = values['mean'], values['powers']
            with np.errstate(over='ignore', invalid='ignore'):
                scatter = powers.mean(1) - np.mean(mean*mean, axis=(1, 2, 3))
                array_discrepancy(scatter, values['scatter'], 'raw scatter', atol=2e-12)
                for arm in arms:
                    cg = caches[f"C{arm['calibration_length']}-P{arm['panel']}-M{arm['size']}"][:, 2, 0]
                    displacement = np.mean((mean-cg)**2, axis=(1, 2, 3))
                    array_discrepancy(displacement, values[arm['name']+'-bias'], 'raw displacement', atol=2e-12)
                    array_discrepancy(scatter+displacement, values[arm['name']+'-risk'], 'raw risk', atol=2e-12)
    return dict(files=files, arrays=arrays, nonfinite_arrays=0)
