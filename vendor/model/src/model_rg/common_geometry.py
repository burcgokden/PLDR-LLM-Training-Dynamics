"""Exact radial and directional contributions to finite vector susceptibility."""
import numpy as np


def radial_directional(points, size):
    x = np.asarray(points, dtype=np.float64)
    if x.ndim < 2 or len(x) < 2 or x.shape[-1] < 1 or size <= 0:
        raise ValueError('At least two vector realizations and a positive size are required')
    if not np.isfinite(x).all():raise ValueError('Nonfinite vector observation')
    s = len(x)
    radii = np.sqrt(np.mean(x*x, axis=-1))
    direction = np.divide(x, radii[..., None], out=np.zeros_like(x), where=radii[..., None] > 0)
    radial, angular, direct = np.zeros((s, s)), np.zeros((s, s)), np.zeros((s, s))
    cosine, cosine_count = 0., 0
    identity_error, identity_tolerance = 0., 0.
    gamma = 128*np.finfo(np.float64).eps*x.shape[-1]
    for i in range(s):
        for j in range(i+1, s):
            dr = (radii[i]-radii[j])**2
            da = radii[i]*radii[j]*np.mean((direction[i]-direction[j])**2, axis=-1)
            dd = np.mean((x[i]-x[j])**2, axis=-1)
            scale = (radii[i]+radii[j])**2
            tolerance = 2e-12*dd+gamma*np.sqrt(dd*scale)+gamma*gamma*scale
            error = np.abs(dr+da-dd)
            if np.any(error > tolerance):raise AssertionError('Radial/directional identity exceeds its scale-aware arithmetic check')
            identity_error = max(identity_error, float(error.max()))
            identity_tolerance = max(identity_tolerance, float(tolerance.max()))
            radial[i, j] = radial[j, i] = dr.mean()
            angular[i, j] = angular[j, i] = da.mean()
            direct[i, j] = direct[j, i] = dd.mean()
            active = (radii[i] > 0) & (radii[j] > 0)
            inner = np.mean(direction[i]*direction[j], axis=-1)
            cosine += float(inner[active].sum());cosine_count += int(active.sum())
    factor = size/(s*(s-1))
    rad, ang, total = [float(factor*np.triu(matrix, 1).sum()) for matrix in [radial, angular, direct]]
    np.testing.assert_allclose(total, size*np.var(x, axis=0, ddof=1).mean(), rtol=2e-12, atol=1e-25)
    closure_tolerance = size*identity_tolerance/2+2e-12*total
    if abs(rad+ang-total) > closure_tolerance:
        raise AssertionError('Aggregate radial/directional closure exceeds its arithmetic check')
    mean_energy = float(np.mean(radii**2))
    mean_vector_energy = float(np.mean(x.mean(0)**2))
    energy_identity = size*s*(mean_energy-mean_vector_energy)/(s-1)
    energy_tolerance = 2e-10*total+gamma*size*s*(mean_energy+mean_vector_energy)/(s-1)
    if abs(total-energy_identity) > energy_tolerance:
        raise AssertionError('Subtracted-energy identity exceeds its arithmetic check')
    leave = []
    if s > 2:
        for i in range(s):
            numerator = np.triu(angular, 1).sum()-angular[i].sum()
            denominator = np.triu(direct, 1).sum()-direct[i].sum()
            leave.append(float(numerator/denominator) if denominator > 0 else None)
    valid = [v for v in leave if v is not None]
    jackknife = (float(np.sqrt((s-1)*np.sum((np.array(valid)-np.mean(valid))**2)/s))
                 if len(valid) == s else None)
    return dict(independent_seeds=s, vector_dimension=x.shape[-1], coordinate_inner_product='Mean over vector coordinates.',
        radial_susceptibility=rad, directional_susceptibility=ang, total_susceptibility=total,
        directional_fraction=ang/total if total > 0 else None,
        radial_fraction=rad/total if total > 0 else None,
        mean_radius=float(radii.mean()), mean_radius_squared=mean_energy,
        mean_vector_squared=mean_vector_energy,
        pair_identity_absolute_error_maximum=identity_error,
        pair_identity_absolute_tolerance_maximum=identity_tolerance,
        susceptibility_closure_error=abs(rad+ang-total),
        susceptibility_closure_tolerance=closure_tolerance,
        subtracted_energy_identity_error=abs(total-energy_identity),
        arithmetic_scope='Scale-aware float64 consistency checks; not interval arithmetic or an exact-arithmetic certificate. Actual susceptibilities use nonnegative pair distances, not a subtraction of large second moments.',
        across_seed_direction_cosine=cosine/cosine_count if cosine_count else None,
        zero_radius_vectors=int(np.sum(radii == 0)), valid_direction_pairs=cosine_count,
        directional_fraction_leave_one_out=leave, directional_fraction_jackknife_se=jackknife,
        radial_pair_distances=radial.tolist(), directional_pair_distances=angular.tolist(),
        complete_pair_distances=direct.tolist())
