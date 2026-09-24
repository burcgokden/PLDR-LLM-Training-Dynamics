"""Finite endpoint allocations of a log potential; no causal interpretation."""
import numpy as np


def potential_allocations(logbase, power):
    """Aligned time/coordinate arrays, reconstructed in float64."""
    b, p = np.asarray(logbase, dtype=np.float64), np.asarray(power, dtype=np.float64)
    if b.shape != p.shape or b.ndim < 2 or len(b) < 2:
        raise ValueError('Aligned time and coordinate axes are required')
    if not np.isfinite(b).all() or not np.isfinite(p).all():
        raise ValueError('Finite coordinates are required')
    db, dp = np.diff(b, axis=0), np.diff(p, axis=0)
    return dict(increment=np.diff(p*b, axis=0), corner=dp*db,
                forward=(dp*b[:-1], p[1:]*db),
                reverse=(dp*b[1:], p[:-1]*db),
                symmetric=(dp*(b[:-1]+b[1:])/2, (p[:-1]+p[1:])*db/2))


def summarize_allocations(logbase, power):
    a = potential_allocations(logbase, power)
    energy = float(np.sum(a['increment']**2))
    result = dict(increment_energy=energy,
                  corner_relative_norm=float(np.sqrt(np.sum(a['corner']**2)/energy)) if energy else 0.)
    for name in ['forward', 'reverse', 'symmetric']:
        u, v = a[name]
        eu, ev, cross = float(np.sum(u*u)), float(np.sum(v*v)), float(2*np.sum(u*v))
        denominator = eu+ev
        result[name] = dict(exponent_energy=eu, base_energy=ev, signed_cross=cross,
                            base_fraction=ev/denominator if denominator else None,
                            relative_identity_residual=float(np.linalg.norm(a['increment']-u-v)/max(np.sqrt(energy), 1e-300)))
    return result
