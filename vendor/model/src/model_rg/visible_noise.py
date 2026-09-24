"""Fixed-state Fisher source coordinates and finite-sample projection diagnostics."""
import numpy as np


def fisher_image(logit_direction, probability):
    """Whiten a full-vocabulary logit tangent, averaging fixed contexts equally."""
    p = np.asarray(probability, dtype=float)
    v = np.asarray(logit_direction, dtype=float)
    return (v-np.sum(v*p, axis=-1, keepdims=True))*np.sqrt(p/p.shape[0])


def fisher_adjoint(weighted_direction, probability):
    """Euclidean transpose of fisher_image at a fixed probability array."""
    p = np.asarray(probability, dtype=float)
    w = np.asarray(weighted_direction, dtype=float)*np.sqrt(p/p.shape[0])
    return w-p*np.sum(w, axis=-1, keepdims=True)


def visible_basis(images, relative_singular_tolerance=1e-12):
    """Uncentered empirical response basis; the same subspace retains the drift."""
    y = np.asarray(images, dtype=float).reshape(len(images), -1)
    _, singular, right = np.linalg.svd(y, full_matrices=False)
    active = singular > max(float(singular[0]), 1e-300)*relative_singular_tolerance
    return right[active], singular


def projection_statistics(images, coefficients, basis, requested):
    """Total response and centered covariance errors, with a frozen source basis."""
    y = np.asarray(images, dtype=float).reshape(len(images), -1)
    coefficients = np.asarray(coefficients, dtype=float)
    k = min(requested, len(basis))
    predicted = coefficients[:, :k]@basis[:k]
    error = y-predicted
    energy = float(np.mean(np.sum(y*y, axis=1)))
    error_energy = float(np.mean(np.sum(error*error, axis=1)))
    trace = float(np.var(y, axis=0, ddof=1).sum())
    error_trace = float(np.var(error, axis=0, ddof=1).sum())
    predicted_trace = float(np.var(predicted, axis=0, ddof=1).sum())
    return dict(requested_dimension=requested, retained_dimension=k,
        total_response_second_moment=energy, residual_second_moment=error_energy,
        total_response_residual_fraction=error_energy/energy if energy else None,
        covariance_trace=trace, residual_covariance_trace=error_trace,
        predicted_covariance_trace=predicted_trace,
        covariance_residual_fraction=error_trace/trace if trace else None,
        mean_response_squared=float(np.sum(y.mean(0)**2)),
        mean_response_error_squared=float(np.sum(error.mean(0)**2)))
