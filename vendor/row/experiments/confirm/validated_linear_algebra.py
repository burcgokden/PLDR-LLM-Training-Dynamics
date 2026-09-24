"""Exact witnesses for matrix bounds and residual-aware Lyapunov gains."""

from __future__ import annotations

from fractions import Fraction

from validated_interval import (
    RationalInterval,
    as_fraction,
    nonnegative_sqrt_upper,
    rational_object,
)


def _matrix(value):
    rows = tuple(tuple(as_fraction(entry) for entry in row) for row in value)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("matrix must be nonempty and rectangular")
    return rows


def _square(value):
    value = _matrix(value)
    if len(value) != len(value[0]):
        raise ValueError("matrix must be square")
    return value


def matrix_object(value):
    return [[rational_object(entry) for entry in row] for row in _matrix(value)]


def transpose(value):
    value = _matrix(value)
    return tuple(tuple(value[i][j] for i in range(len(value)))
                 for j in range(len(value[0])))


def add(left, right):
    left, right = _matrix(left), _matrix(right)
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        raise ValueError("matrix addition dimensions disagree")
    return tuple(tuple(a + b for a, b in zip(lrow, rrow))
                 for lrow, rrow in zip(left, right))


def subtract(left, right):
    left, right = _matrix(left), _matrix(right)
    if (len(left), len(left[0])) != (len(right), len(right[0])):
        raise ValueError("matrix subtraction dimensions disagree")
    return tuple(tuple(a - b for a, b in zip(lrow, rrow))
                 for lrow, rrow in zip(left, right))


def multiply(left, right):
    left, right = _matrix(left), _matrix(right)
    if len(left[0]) != len(right):
        raise ValueError("matrix product dimensions disagree")
    return tuple(tuple(sum(
        (left[i][k] * right[k][j] for k in range(len(right))),
        Fraction(0),
    ) for j in range(len(right[0]))) for i in range(len(left)))


def identity(size):
    return tuple(tuple(Fraction(int(i == j)) for j in range(size))
                 for i in range(size))


def is_symmetric(value):
    value = _square(value)
    return value == transpose(value)


def inverse(value):
    """Gauss-Jordan inverse over exact rationals."""

    value = _square(value)
    size = len(value)
    augmented = [list(row) + list(identity(size)[i])
                 for i, row in enumerate(value)]
    for column in range(size):
        pivot = next((row for row in range(column, size)
                      if augmented[row][column] != 0), None)
        if pivot is None:
            raise ValueError("matrix is singular")
        augmented[column], augmented[pivot] = (
            augmented[pivot], augmented[column])
        divisor = augmented[column][column]
        augmented[column] = [entry / divisor for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor:
                augmented[row] = [
                    entry - factor * pivot_entry
                    for entry, pivot_entry in zip(
                        augmented[row], augmented[column])
                ]
    return tuple(tuple(row[size:]) for row in augmented)


def exact_ldl(value):
    """Return an exact unpivoted LDL transpose witness for a symmetric matrix."""

    value = _square(value)
    if not is_symmetric(value):
        raise ValueError("LDL witness requires a symmetric matrix")
    size = len(value)
    lower = [[Fraction(int(i == j)) for j in range(size)]
             for i in range(size)]
    diagonal = [Fraction(0) for _ in range(size)]
    for j in range(size):
        diagonal[j] = value[j][j] - sum(
            (lower[j][k] ** 2 * diagonal[k] for k in range(j)),
            Fraction(0),
        )
        if diagonal[j] <= 0:
            raise ValueError("matrix is not positive definite under exact LDL")
        for i in range(j + 1, size):
            numerator = value[i][j] - sum(
                (lower[i][k] * lower[j][k] * diagonal[k]
                 for k in range(j)),
                Fraction(0),
            )
            lower[i][j] = numerator / diagonal[j]
    reconstructed = multiply(
        multiply(lower, tuple(tuple(
            diagonal[i] if i == j else Fraction(0)
            for j in range(size)) for i in range(size))),
        transpose(lower),
    )
    if reconstructed != value:
        raise ArithmeticError("exact LDL reconstruction failed")
    return tuple(tuple(row) for row in lower), tuple(diagonal)


def induced_one_norm(value):
    value = _matrix(value)
    return max(sum((abs(value[i][j]) for i in range(len(value))), Fraction(0))
               for j in range(len(value[0])))


def induced_infinity_norm(value):
    value = _matrix(value)
    return max(sum((abs(entry) for entry in row), Fraction(0)) for row in value)


def spectral_norm_upper(value):
    """Checked upper bound ``sqrt(||A||_1 ||A||_infinity)``."""

    value = _matrix(value)
    return nonnegative_sqrt_upper(
        induced_one_norm(value) * induced_infinity_norm(value))


def positive_eigenvalue_bounds(value):
    """Exact positive lower and checked upper eigenvalue bounds.

    Exact LDL proves positive definiteness.  The inverse norm inequality
    gives ``lambda_min(H) >= 1 / ||H^-1||_2`` and the induced-norm inequality
    bounds both spectral norms without trusting a floating eigensolver.
    """

    value = _square(value)
    exact_ldl(value)
    upper = spectral_norm_upper(value)
    inverse_upper = spectral_norm_upper(inverse(value))
    lower = Fraction(1) / inverse_upper
    return lower, upper


def solve_discrete_lyapunov_exact(nominal, dissipation=None):
    """Solve ``H - A.T H A = Q`` over exact rationals.

    This is a certificate constructor, not a floating point Lyapunov solve.
    The full linear operator on matrix entries is inverted over
    :class:`fractions.Fraction`; the returned matrix is then checked for
    symmetry, positive definiteness, and exact satisfaction of the equation.
    ``Q`` defaults to the identity.
    """

    nominal = _square(nominal)
    size = len(nominal)
    dissipation = identity(size) if dissipation is None else _square(dissipation)
    if len(dissipation) != size or not is_symmetric(dissipation):
        raise ValueError("Lyapunov dissipation must be symmetric and dimension matched")
    positive_eigenvalue_bounds(dissipation)

    dimension = size * size
    operator = []
    target = []
    for i in range(size):
        for j in range(size):
            operator.append(tuple(
                Fraction(int(i == k and j == ell))
                - nominal[k][i] * nominal[ell][j]
                for k in range(size)
                for ell in range(size)
            ))
            target.append(dissipation[i][j])
    solution = multiply(inverse(tuple(operator)), tuple((entry,) for entry in target))
    metric = tuple(tuple(solution[i * size + j][0] for j in range(size))
                   for i in range(size))
    if not is_symmetric(metric):
        raise ArithmeticError("exact Lyapunov solution is not symmetric")
    positive_eigenvalue_bounds(metric)
    residual = subtract(
        metric,
        multiply(multiply(transpose(nominal), metric), nominal),
    )
    if residual != dissipation:
        raise ArithmeticError("exact Lyapunov equation verification failed")
    if len(operator) != dimension:
        raise ArithmeticError("Lyapunov operator dimension mismatch")
    return metric


def interval_matrix(value):
    rows = tuple(tuple(RationalInterval.from_object(entry) for entry in row)
                 for row in value)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("interval matrix must be nonempty and rectangular")
    return rows


def interval_matrix_object(value):
    return [[entry.to_object() for entry in row] for row in interval_matrix(value)]


def interval_absolute_radius(intervals, center):
    intervals = interval_matrix(intervals)
    center = _matrix(center)
    if (len(intervals), len(intervals[0])) != (len(center), len(center[0])):
        raise ValueError("interval family and center dimensions disagree")
    return tuple(tuple(max(
        abs(intervals[i][j].lower - center[i][j]),
        abs(intervals[i][j].upper - center[i][j]),
    ) for j in range(len(center[0]))) for i in range(len(center)))


def interval_spectral_norm_upper(intervals):
    """Bound every matrix in an interval family by absolute row/column sums."""

    intervals = interval_matrix(intervals)
    radius = tuple(tuple(entry.maximum_absolute() for entry in row)
                   for row in intervals)
    return spectral_norm_upper(radius)


def centered_interval_spectral_radius(intervals, center):
    return spectral_norm_upper(interval_absolute_radius(intervals, center))


def residual_aware_lyapunov_certificate(nominal, metric, family_intervals):
    """Construct a fully rational common-metric gain certificate.

    The effective dissipation is computed from the supplied metric itself,
    so any residual of an approximate Lyapunov solve is automatically present
    in the strict margin.  The family radius encloses the complete interval
    image, not a list of sampled matrices.
    """

    nominal = _square(nominal)
    metric = _square(metric)
    if len(nominal) != len(metric):
        raise ValueError("nominal matrix and metric dimensions disagree")
    if not is_symmetric(metric):
        raise ValueError("metric must be symmetric")
    metric_lower, metric_upper = positive_eigenvalue_bounds(metric)
    effective = subtract(
        metric,
        multiply(multiply(transpose(nominal), metric), nominal),
    )
    dissipation_lower, dissipation_upper = positive_eigenvalue_bounds(effective)
    q0_squared = Fraction(1) - dissipation_lower / metric_upper
    if not 0 <= q0_squared < 1:
        raise ValueError("residual-adjusted nominal gain is not strictly contractive")
    q0_upper = nonnegative_sqrt_upper(q0_squared)
    euclidean_radius = centered_interval_spectral_radius(
        family_intervals, nominal)
    condition_upper = nonnegative_sqrt_upper(metric_upper / metric_lower)
    metric_radius = condition_upper * euclidean_radius
    robust_q_upper = q0_upper + metric_radius
    if robust_q_upper >= 1:
        raise ValueError("complete interval family does not have strict common gain")
    return {
        "nominal_matrix": matrix_object(nominal),
        "metric": matrix_object(metric),
        "metric_lower": rational_object(metric_lower),
        "metric_upper": rational_object(metric_upper),
        "effective_dissipation": matrix_object(effective),
        "dissipation_lower": rational_object(dissipation_lower),
        "dissipation_upper": rational_object(dissipation_upper),
        "nominal_q_squared_upper": rational_object(q0_squared),
        "nominal_q_upper": rational_object(q0_upper),
        "family_intervals": interval_matrix_object(family_intervals),
        "euclidean_family_radius_upper": rational_object(euclidean_radius),
        "metric_condition_sqrt_upper": rational_object(condition_upper),
        "metric_family_radius_upper": rational_object(metric_radius),
        "robust_q_upper": rational_object(robust_q_upper),
        "derivation": "exact_ldl_inverse_norm_and_induced_family_norm",
    }


def check_residual_aware_lyapunov_certificate(certificate):
    rebuilt = residual_aware_lyapunov_certificate(
        certificate["nominal_matrix"],
        certificate["metric"],
        certificate["family_intervals"],
    )
    if rebuilt != certificate:
        raise ValueError("Lyapunov certificate does not replay exactly")
    return True


def _scale(value, scalar):
    value = _matrix(value)
    scalar = as_fraction(scalar)
    return tuple(tuple(scalar * entry for entry in row) for row in value)


def _strict_vertex_dissipations(vertices, source_metric, target_metric,
                                gain_squared):
    vertices = tuple(_square(vertex) for vertex in vertices)
    if not vertices:
        raise ValueError("a convex matrix family needs at least one vertex")
    source_metric = _square(source_metric)
    target_metric = _square(target_metric)
    dimension = len(source_metric)
    if len(target_metric) != dimension or any(
        len(vertex) != dimension for vertex in vertices
    ):
        raise ValueError("vertex and metric dimensions disagree")
    if not is_symmetric(source_metric) or not is_symmetric(target_metric):
        raise ValueError("path metrics must be symmetric")
    positive_eigenvalue_bounds(source_metric)
    positive_eigenvalue_bounds(target_metric)
    gain_squared = as_fraction(gain_squared)
    if gain_squared < 0:
        raise ValueError("squared vertex gain must be nonnegative")
    dissipations = tuple(
        subtract(
            _scale(source_metric, gain_squared),
            multiply(multiply(transpose(vertex), target_metric), vertex),
        )
        for vertex in vertices
    )
    bounds = tuple(positive_eigenvalue_bounds(value)
                   for value in dissipations)
    return vertices, source_metric, target_metric, dissipations, bounds


def convex_family_lyapunov_certificate(vertices, source_metric,
                                       target_metric=None,
                                       gain_squared=None):
    """Certify a correlated convex matrix family by exact vertex LMIs.

    If A is a convex combination of the supplied vertices, the norm triangle
    inequality transports the vertex gain to A. This keeps the correlations
    induced by primitive optimizer coordinates and does not replace them by
    an entrywise interval matrix.

    When gain_squared is omitted, the constructor searches the dyadic sequence
    1 - 2^-k and retains the first exact strict witness. target_metric may
    differ from source_metric for a graph edge.
    """

    vertices = tuple(_square(vertex) for vertex in vertices)
    source_metric = _square(source_metric)
    target_metric = (
        source_metric if target_metric is None else _square(target_metric)
    )
    if gain_squared is None:
        selected = None
        for exponent in range(1, 161):
            candidate = Fraction(1) - Fraction(1, 2 ** exponent)
            try:
                _strict_vertex_dissipations(
                    vertices, source_metric, target_metric, candidate)
            except ValueError:
                continue
            selected = candidate
            break
        if selected is None:
            raise ValueError(
                "convex vertex family has no strict dyadic gain below one")
        gain_squared = selected
    (
        vertices,
        source_metric,
        target_metric,
        dissipations,
        bounds,
    ) = _strict_vertex_dissipations(
        vertices, source_metric, target_metric, gain_squared)
    gain_squared = as_fraction(gain_squared)
    gain_upper = nonnegative_sqrt_upper(gain_squared)
    return {
        "schema_version": "pldr-convex-family-lyapunov-v1",
        "source_metric": matrix_object(source_metric),
        "target_metric": matrix_object(target_metric),
        "vertices": [matrix_object(vertex) for vertex in vertices],
        "gain_squared_upper": rational_object(gain_squared),
        "gain_upper": rational_object(gain_upper),
        "vertex_dissipations": [
            matrix_object(value) for value in dissipations
        ],
        "vertex_dissipation_lower_bounds": [
            rational_object(lower) for lower, _ in bounds
        ],
        "vertex_dissipation_upper_bounds": [
            rational_object(upper) for _, upper in bounds
        ],
        "strictly_contracting": gain_upper < 1,
        "family_rule": "convex_hull_of_correlated_primitive_box_vertices",
        "derivation": "exact_ldl_on_every_cross_metric_vertex_lmi",
    }


def check_convex_family_lyapunov_certificate(certificate):
    rebuilt = convex_family_lyapunov_certificate(
        certificate["vertices"],
        certificate["source_metric"],
        target_metric=certificate["target_metric"],
        gain_squared=certificate["gain_squared_upper"],
    )
    if rebuilt != certificate:
        raise ValueError("convex-family Lyapunov certificate does not replay")
    return True


def graph_window_certificate(edges, window_length):
    """Enumerate every admissible graph path of a fixed length exactly."""

    if (
        not isinstance(window_length, int)
        or isinstance(window_length, bool)
        or window_length < 1
    ):
        raise ValueError("window_length must be a positive integer")
    if not isinstance(edges, (list, tuple)) or not edges:
        raise ValueError("the path graph must contain at least one edge")
    normalized = []
    identifiers = set()
    for edge in edges:
        if not isinstance(edge, dict) or set(edge) != {
            "edge_id", "source", "target", "gain_upper",
        }:
            raise ValueError("graph edges have missing or unknown fields")
        edge_id = str(edge["edge_id"])
        source = str(edge["source"])
        target = str(edge["target"])
        if not edge_id or not source or not target or edge_id in identifiers:
            raise ValueError("graph edge identifiers must be unique and nonempty")
        gain = as_fraction(edge["gain_upper"])
        if not 0 <= gain:
            raise ValueError("edge gains must be nonnegative")
        identifiers.add(edge_id)
        normalized.append({
            "edge_id": edge_id,
            "source": source,
            "target": target,
            "gain_upper": gain,
        })
    sources = {edge["source"] for edge in normalized}
    targets = {edge["target"] for edge in normalized}
    if not targets.issubset(sources):
        missing = ", ".join(sorted(targets - sources))
        raise ValueError("path graph has a terminal cell: " + missing)
    paths = [([edge["edge_id"]], edge["target"], edge["gain_upper"])
             for edge in normalized]
    for _ in range(1, window_length):
        extended = []
        for identifiers_path, target, path_product in paths:
            for edge in normalized:
                if edge["source"] == target:
                    extended.append((
                        [*identifiers_path, edge["edge_id"]],
                        edge["target"],
                        path_product * edge["gain_upper"],
                    ))
        if not extended:
            raise ValueError("path graph has no complete registered window")
        paths = extended
    maximum = max(path_product for _, _, path_product in paths)
    if maximum >= 1:
        raise ValueError("an admissible graph window is not contractive")
    return {
        "schema_version": "pldr-graph-window-contraction-v1",
        "window_length": window_length,
        "edges": [{
            **{name: edge[name] for name in ("edge_id", "source", "target")},
            "gain_upper": rational_object(edge["gain_upper"]),
        } for edge in normalized],
        "admissible_path_count": len(paths),
        "path_products": [{
            "edge_ids": identifiers_path,
            "gain_product_upper": rational_object(path_product),
        } for identifiers_path, _, path_product in paths],
        "maximum_window_gain": rational_object(maximum),
        "strictly_contracting": True,
        "derivation": "exact_enumeration_of_every_admissible_graph_path",
    }


def check_graph_window_certificate(certificate):
    rebuilt = graph_window_certificate(
        certificate["edges"], certificate["window_length"])
    if rebuilt != certificate:
        raise ValueError("graph-window certificate does not replay exactly")
    return True
