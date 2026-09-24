"""Finite-domain ordered comparisons at scientific verification boundaries."""
from numerical_validation import finite_array


def finite_greater(actual, bound, label):
    """Validate both domains before a scalar or elementwise upper-bound check."""
    actual=finite_array(actual,'reconstructed comparison '+label)
    bound=finite_array(bound,'comparison bound '+label)
    return actual > bound
