"""Numerical kernels for fixed-registry row-map observable margins.

The kernels in this module are intentionally model independent.  Live
instrumentation owns tensors and source digests; this module owns the finite
graph, exact gate/shape energy ledger, affine block recursion, descriptive
attribution, and occupied-segment power coefficients consumed by an
independent analyzer.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np


def _array(value, name, *, ndim=None):
    result = np.asarray(value, dtype=np.float64)
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have dimension {ndim}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _scalar(value, name):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def digest_object(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _squared_distances(points):
    points = _array(points, "points", ndim=2)
    if len(points) < 2:
        raise ValueError("a graph needs at least two vertices")
    norms = np.sum(points * points, axis=1)
    distances = norms[:, None] + norms[None, :] - 2.0 * points @ points.T
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, np.inf)
    return distances


def deterministic_knn_mst_graph(points, *, neighbors=8):
    """Return a symmetric k-NN graph augmented by a deterministic MST.

    Nearest-neighbor ties are broken by vertex index.  Prim's algorithm uses
    the same tie rule, so byte-identical input produces byte-identical edges.
    Duplicate undirected edges are merged and all retained edges receive the
    uniform normalization ``1 / edge_count``.
    """

    points = _array(points, "points", ndim=2)
    vertex_count = len(points)
    neighbors = int(neighbors)
    if neighbors < 1:
        raise ValueError("neighbors must be positive")
    neighbor_count = min(neighbors, vertex_count - 1)
    distance = _squared_distances(points)
    indices = np.arange(vertex_count)
    edges = set()
    for source in range(vertex_count):
        order = np.lexsort((indices, distance[source]))
        for target in order[:neighbor_count]:
            left, right = sorted((source, int(target)))
            edges.add((left, right))

    # Deterministic Prim tree on the complete Euclidean graph.
    selected = np.zeros(vertex_count, dtype=bool)
    selected[0] = True
    best = distance[0].copy()
    parent = np.zeros(vertex_count, dtype=np.int64)
    best[0] = np.inf
    for _ in range(vertex_count - 1):
        candidates = np.flatnonzero(~selected)
        target = min(candidates, key=lambda item: (best[item], int(item)))
        source = int(parent[target])
        edges.add(tuple(sorted((source, int(target)))))
        selected[target] = True
        for candidate in np.flatnonzero(~selected):
            candidate_distance = distance[target, candidate]
            if (
                candidate_distance < best[candidate]
                or (
                    candidate_distance == best[candidate]
                    and target < parent[candidate]
                )
            ):
                best[candidate] = candidate_distance
                parent[candidate] = target

    edge_array = np.asarray(sorted(edges), dtype=np.int64)
    weights = np.full(len(edge_array), 1.0 / len(edge_array), dtype=np.float64)
    graph = {
        "rule": "symmetric-8nn-plus-euclidean-mst-uniform-v1",
        "neighbors": neighbors,
        "vertex_count": vertex_count,
        "edges": edge_array,
        "weights": weights,
    }
    graph["graph_sha256"] = graph_digest(graph)
    return graph


def graph_digest(graph):
    edges = np.asarray(graph["edges"], dtype=np.int64)
    weights = np.asarray(graph["weights"], dtype=np.float64)
    payload = {
        "rule": str(graph["rule"]),
        "neighbors": int(graph["neighbors"]),
        "vertex_count": int(graph["vertex_count"]),
        "edges": edges.tolist(),
        "weights_hex": [float(value).hex() for value in weights],
    }
    return digest_object(payload)


def pack_union_graph(graph):
    """Return the canonical sufficient representation of one union graph."""

    edge = np.asarray(graph["edges"], dtype=np.int64)
    weight = _array(graph["weights"], "weights", ndim=1)
    vertex_count = int(graph["vertex_count"])
    graph_laplacian(vertex_count, edge, weight)
    digest = graph_digest(graph)
    if graph.get("graph_sha256") not in {None, digest}:
        raise ValueError("union graph digest does not replay")
    return {
        "rule": str(graph["rule"]),
        "neighbors": int(graph["neighbors"]),
        "vertex_count": vertex_count,
        "edges": edge.copy(),
        "weights": weight.copy(),
        "union_graph_sha256": digest,
    }


def unpack_union_graph(
    vertex_count, edges, weights, *, rule, neighbors, expected_sha256=None,
):
    """Validate and rebuild one connected locked union graph."""

    graph = {
        "rule": str(rule),
        "neighbors": int(neighbors),
        "vertex_count": int(vertex_count),
        "edges": np.asarray(edges, dtype=np.int64).copy(),
        "weights": _array(weights, "weights", ndim=1).copy(),
    }
    graph_laplacian(graph["vertex_count"], graph["edges"], graph["weights"])
    graph["graph_sha256"] = graph_digest(graph)
    if expected_sha256 is not None and graph["graph_sha256"] != str(
        expected_sha256
    ):
        raise ValueError("locked union graph digest does not replay")
    return graph


def shape_from_edge_contrasts(vertex_count, edges, contrasts):
    """Reconstruct row differences, up to translation, from a connected graph.

    Each stored contrast has the registered orientation ``shape[left] -
    shape[right]``. A deterministic spanning traversal reconstructs anchored
    rows and every non-tree edge then supplies an independent cycle residual.
    """

    vertex_count = int(vertex_count)
    edges = np.asarray(edges, dtype=np.int64)
    contrasts = _array(contrasts, "contrasts", ndim=2)
    if edges.shape != (len(contrasts), 2) or vertex_count < 2:
        raise ValueError("edge contrasts and graph dimensions disagree")
    adjacency = [[] for _ in range(vertex_count)]
    for edge_index, (left, right) in enumerate(edges):
        if left < 0 or right < 0 or left >= vertex_count or right >= vertex_count:
            raise ValueError("edge contrast leaves the union vertex registry")
        adjacency[int(left)].append((int(right), edge_index, -1.0))
        adjacency[int(right)].append((int(left), edge_index, 1.0))
    rows = np.zeros((vertex_count, contrasts.shape[1]), dtype=np.float64)
    visited = np.zeros(vertex_count, dtype=bool)
    visited[0] = True
    queue = [0]
    while queue:
        source = queue.pop(0)
        for target, edge_index, sign in adjacency[source]:
            if visited[target]:
                continue
            # sign=-1 for left->right and +1 for right->left.
            rows[target] = rows[source] + sign * contrasts[edge_index]
            visited[target] = True
            queue.append(target)
    if not np.all(visited):
        raise ValueError("edge contrasts use a disconnected graph")
    replay = rows[edges[:, 0]] - rows[edges[:, 1]]
    residual = float(np.linalg.norm(replay - contrasts))
    scale = max(float(np.linalg.norm(replay)), float(np.linalg.norm(contrasts)), 1e-300)
    return {
        "anchored_rows": rows,
        "relative_cycle_residual": residual / scale,
    }


def graph_laplacian(vertex_count, edges, weights):
    vertex_count = int(vertex_count)
    edges = np.asarray(edges, dtype=np.int64)
    weights = _array(weights, "weights", ndim=1)
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(weights):
        raise ValueError("edges and weights disagree")
    if vertex_count < 2 or np.any(weights <= 0.0):
        raise ValueError("a weighted graph needs positive weights")
    if np.any(edges < 0) or np.any(edges >= vertex_count):
        raise ValueError("an edge leaves the vertex set")
    laplacian = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    for (left, right), weight in zip(edges, weights, strict=True):
        if left == right:
            raise ValueError("self edges are forbidden")
        laplacian[left, left] += weight
        laplacian[right, right] += weight
        laplacian[left, right] -= weight
        laplacian[right, left] -= weight
    eigenvalues = np.linalg.eigvalsh(laplacian)
    scale = max(float(eigenvalues[-1]), 1.0)
    connected = int(np.count_nonzero(eigenvalues <= 1e-11 * scale)) == 1
    if not connected:
        raise ValueError("the registry graph is disconnected")
    return laplacian


def effective_resistance_constant(vertex_count, edges, weights):
    """Compute ``max_{r,s} R_eff(r,s)`` from the graph pseudoinverse."""

    laplacian = graph_laplacian(vertex_count, edges, weights)
    inverse = np.linalg.pinv(laplacian, hermitian=True, rcond=1e-12)
    diagonal = np.diag(inverse)
    resistance = diagonal[:, None] + diagonal[None, :] - 2.0 * inverse
    np.maximum(resistance, 0.0, out=resistance)
    return {
        "effective_resistance_max": float(np.max(resistance)),
        "resistance_matrix": resistance,
        "laplacian": laplacian,
    }


def graph_energy(gamma, normalized_shape, edges, weights):
    gamma = _array(gamma, "gamma", ndim=1)
    shape = _array(normalized_shape, "normalized_shape", ndim=2)
    edges = np.asarray(edges, dtype=np.int64)
    weights = _array(weights, "weights", ndim=1)
    if shape.shape[1] != len(gamma):
        raise ValueError("gate and normalized-shape dimensions disagree")
    if edges.ndim != 2 or edges.shape[1] != 2 or len(edges) != len(weights):
        raise ValueError("edges and weights disagree")
    contrast = shape[edges[:, 0]] - shape[edges[:, 1]]
    q = np.sum(weights[:, None] * contrast * contrast, axis=0)
    energy = float(np.sum(q * gamma * gamma))
    row_map = shape * gamma
    pairwise = row_map[:, None, :] - row_map[None, :, :]
    diameter = float(np.max(np.linalg.norm(pairwise, axis=2)))
    return {"q": q, "energy": energy, "vertex_diameter": diameter}


def effective_resistance_bound(energy, resistance_max):
    energy = _scalar(energy, "energy")
    resistance_max = _scalar(resistance_max, "resistance_max")
    if energy < 0.0 or resistance_max < 0.0:
        raise ValueError("energy and resistance must be nonnegative")
    return math.sqrt(resistance_max * energy)


def shape_secant(gamma_next, shape, shape_increment, edges, weights):
    """Exact signed linear shape gain and nonnegative quadratic charge.

    For ``Psi(x) = sum_e w_e ||diag(gamma_next) Delta_e x||^2``, the
    returned values satisfy

    ``Psi(x+h) - Psi(x) = -signed_linear_gain + quadratic_charge``.
    """

    gamma_next = _array(gamma_next, "gamma_next", ndim=1)
    shape = _array(shape, "shape", ndim=2)
    increment = _array(shape_increment, "shape_increment", ndim=2)
    edges = np.asarray(edges, dtype=np.int64)
    weights = _array(weights, "weights", ndim=1)
    if shape.shape != increment.shape or shape.shape[1] != len(gamma_next):
        raise ValueError("shape secant dimensions disagree")
    source_edge = shape[edges[:, 0]] - shape[edges[:, 1]]
    increment_edge = increment[edges[:, 0]] - increment[edges[:, 1]]
    gamma_square = gamma_next * gamma_next
    linear_work = 2.0 * float(np.sum(
        weights[:, None] * source_edge * increment_edge * gamma_square,
    ))
    quadratic_charge = float(np.sum(
        weights[:, None] * increment_edge * increment_edge * gamma_square,
    ))
    return {
        "signed_linear_gain": -linear_work,
        "linear_work": linear_work,
        "quadratic_charge": quadratic_charge,
        "shape_work": linear_work + quadratic_charge,
    }


def shape_metric_secant(gamma_next, q, q_next, linear_q, quadratic_q):
    """Compact replay of :func:`shape_secant` from coordinate ledgers."""

    gamma_next = _array(gamma_next, "gamma_next", ndim=1)
    q = _array(q, "q", ndim=1)
    q_next = _array(q_next, "q_next", ndim=1)
    linear_q = _array(linear_q, "linear_q", ndim=1)
    quadratic_q = _array(quadratic_q, "quadratic_q", ndim=1)
    if not (
        gamma_next.shape == q.shape == q_next.shape
        == linear_q.shape == quadratic_q.shape
    ):
        raise ValueError("compact shape ledgers disagree")
    if np.any(q < 0.0) or np.any(q_next < 0.0) or np.any(quadratic_q < -1e-13):
        raise ValueError("shape metrics and quadratic charge must be nonnegative")
    reconstruction = q + linear_q + quadratic_q
    residual = float(np.linalg.norm(q_next - reconstruction))
    scale = max(float(np.linalg.norm(q_next)), float(np.linalg.norm(reconstruction)), 1e-300)
    linear_work = float(np.sum(gamma_next * gamma_next * linear_q))
    quadratic_charge = float(np.sum(gamma_next * gamma_next * quadratic_q))
    return {
        "signed_linear_gain": -linear_work,
        "linear_work": linear_work,
        "quadratic_charge": quadratic_charge,
        "shape_work": linear_work + quadratic_charge,
        "relative_reconstruction_residual": residual / scale,
    }


def native_roundoff_charge(shadow_gamma_next, executed_gamma_next, q_next):
    """Measure native-successor work against the float64 shadow successor."""

    shadow = _array(shadow_gamma_next, "shadow_gamma_next", ndim=1)
    executed = _array(executed_gamma_next, "executed_gamma_next", ndim=1)
    q_next = _array(q_next, "q_next", ndim=1)
    if not (shadow.shape == executed.shape == q_next.shape) or np.any(q_next < 0.0):
        raise ValueError("native-roundoff dimensions or metric are invalid")
    delta = executed - shadow
    linear_work = 2.0 * float(np.sum(q_next * shadow * delta))
    quadratic_work = float(np.sum(q_next * delta * delta))
    signed_work = linear_work + quadratic_work
    direct_work = float(np.sum(q_next * (executed * executed - shadow * shadow)))
    scale = max(abs(signed_work), abs(direct_work), 1e-300)
    return {
        "linear_work": linear_work,
        "quadratic_work": quadratic_work,
        "signed_work": signed_work,
        "floating_point_charge": abs(signed_work),
        "relative_reconstruction_residual": abs(signed_work - direct_work) / scale,
    }


def bridge_sector_replay(gamma, q, adaptive_direction, sector_directions):
    """Replay the seven executed bridge sectors and their energy gains."""

    gamma = _array(gamma, "gamma", ndim=1)
    q = _array(q, "q", ndim=1)
    direction = _array(adaptive_direction, "adaptive_direction", ndim=1)
    sectors = _array(sector_directions, "sector_directions", ndim=2)
    if sectors.shape != (7, len(gamma)) or not (
        gamma.shape == q.shape == direction.shape
    ) or np.any(q < 0.0):
        raise ValueError("bridge-sector ledger dimensions are invalid")
    names = (
        "fisher", "signed_jet", "force", "nonlinear", "clip",
        "preconditioner", "lag",
    )
    replay = np.sum(sectors, axis=0)
    residual = float(np.linalg.norm(direction - replay))
    scale = max(float(np.linalg.norm(direction)), float(np.linalg.norm(replay)), 1e-300)
    gains = {
        name: 2.0 * float(np.sum(q * gamma * sector))
        for name, sector in zip(names, sectors, strict=True)
    }
    return {
        "sector_order": names,
        "replayed_direction": replay,
        "sector_gain_without_learning_rate": gains,
        "aggregate_gain_without_learning_rate": float(sum(gains.values())),
        "relative_direction_residual": residual / scale,
    }


def certified_bridge_margin(learning_rate, sector_gain_without_learning_rate):
    """Apply the registered signed-credit and absolute-charge bridge rule."""

    eta = _scalar(learning_rate, "learning_rate")
    if eta <= 0.0:
        raise ValueError("learning rate must be positive")
    names = (
        "fisher", "signed_jet", "force", "nonlinear", "clip",
        "preconditioner", "lag",
    )
    if set(sector_gain_without_learning_rate) != set(names):
        raise ValueError("bridge-sector gain dictionary is incomplete")
    gains = {
        name: _scalar(sector_gain_without_learning_rate[name], name)
        for name in names
    }
    signed_credit = eta * (gains["fisher"] + gains["signed_jet"])
    absolute_charge = eta * sum(abs(gains[name]) for name in names[2:])
    return {
        "fisher_signed_gain": eta * gains["fisher"],
        "signed_jet_signed_gain": eta * gains["signed_jet"],
        "remainder_absolute_charge": absolute_charge,
        "certified_adaptive_gain": signed_credit - absolute_charge,
        "aggregate_adaptive_gain": eta * sum(gains.values()),
    }


def executed_source_margin(
    gamma, q, adaptive_direction, *, learning_rate, weight_decay,
    signed_shape_gain, shape_quadratic_charge, floating_point_charge=0.0,
    decay_mask=None,
):
    """Exact one-step source-owned margin for the implemented AdamW gate."""

    gamma = _array(gamma, "gamma", ndim=1)
    q = _array(q, "q", ndim=1)
    direction = _array(adaptive_direction, "adaptive_direction", ndim=1)
    if not (gamma.shape == q.shape == direction.shape) or np.any(q < 0.0):
        raise ValueError("source-margin dimensions or metric are invalid")
    eta = _scalar(learning_rate, "learning_rate")
    decay = _scalar(weight_decay, "weight_decay")
    if eta <= 0.0 or decay < 0.0:
        raise ValueError("learning rate and decay are outside their domain")
    mask = (
        np.ones_like(gamma) if decay_mask is None
        else _array(decay_mask, "decay_mask", ndim=1)
    )
    if mask.shape != gamma.shape or np.any((mask != 0.0) & (mask != 1.0)):
        raise ValueError("decay mask must be binary and match the gate")
    energy = float(np.sum(q * gamma * gamma))
    gate_source = decay * mask * gamma + direction
    decay_dissipation = 2.0 * eta * decay * float(np.sum(q * mask * gamma * gamma))
    adaptive_gain = 2.0 * eta * float(np.sum(q * gamma * direction))
    finite_step_charge = eta * eta * float(np.sum(q * gate_source * gate_source))
    shape_gain = _scalar(signed_shape_gain, "signed_shape_gain")
    shape_charge = _scalar(shape_quadratic_charge, "shape_quadratic_charge")
    roundoff_charge = _scalar(floating_point_charge, "floating_point_charge")
    if shape_charge < -1e-13 or roundoff_charge < 0.0:
        raise ValueError("registered charges must be nonnegative")
    numerator = (
        decay_dissipation + adaptive_gain + shape_gain
        - finite_step_charge - shape_charge - roundoff_charge
    )
    relative_margin = numerator / energy if energy > 0.0 else math.nan
    gamma_next = gamma - eta * gate_source
    predicted_next = energy - (
        decay_dissipation + adaptive_gain - finite_step_charge
    )
    return {
        "energy": energy,
        "gamma_next": gamma_next,
        "decay_dissipation": decay_dissipation,
        "adaptive_gain": adaptive_gain,
        "finite_step_charge": finite_step_charge,
        "signed_shape_gain": shape_gain,
        "shape_quadratic_charge": shape_charge,
        "floating_point_charge": roundoff_charge,
        "decay_mask": mask,
        "margin_numerator": numerator,
        "relative_margin": relative_margin,
        "fixed_shape_energy_next": predicted_next,
        "energy_next_from_margin": energy - numerator,
    }


def block_affine_convolution(initial_energy, factors, forcing):
    """Replay an ordered nonautonomous affine energy recursion."""

    initial = _scalar(initial_energy, "initial_energy")
    factors = _array(factors, "factors", ndim=1)
    forcing = _array(forcing, "forcing", ndim=1)
    if factors.shape != forcing.shape:
        raise ValueError("affine factors and forcing disagree")
    if initial < 0.0 or np.any(factors < 0.0) or np.any(forcing < 0.0):
        raise ValueError("affine recursion data must be nonnegative")
    bound = initial
    trajectory = [bound]
    for factor, charge in zip(factors, forcing, strict=True):
        bound = float(factor * bound + charge)
        trajectory.append(bound)
    product = float(np.prod(factors)) if len(factors) else 1.0
    convolution = trajectory[-1] - product * initial
    return {
        "factor_product": product,
        "forcing_convolution": float(convolution),
        "endpoint_bound": float(trajectory[-1]),
        "trajectory_bound": np.asarray(trajectory, dtype=np.float64),
    }


def lifted_affine_convolution(initial, operators, forcing):
    """Replay an ordered affine recursion on a finite-dimensional lift."""

    state = _array(initial, "initial", ndim=1)
    initial_state = state.copy()
    operators = _array(operators, "operators", ndim=3)
    forcing = _array(forcing, "forcing", ndim=2)
    if (
        operators.shape[0] != forcing.shape[0]
        or operators.shape[1:] != (len(state), len(state))
        or forcing.shape[1] != len(state)
    ):
        raise ValueError("lifted affine recursion dimensions disagree")
    trajectory = [state.copy()]
    product = np.eye(len(state), dtype=np.float64)
    for operator, charge in zip(operators, forcing, strict=True):
        state = operator @ state + charge
        product = operator @ product
        trajectory.append(state.copy())
    homogeneous = product @ initial_state
    return {
        "operator_product": product,
        "forcing_convolution": state - homogeneous,
        "endpoint": state,
        "trajectory": np.asarray(trajectory, dtype=np.float64),
    }


def gate_first_and_shapley(gamma, gamma_next, q, q_next):
    gamma = _array(gamma, "gamma", ndim=1)
    gamma_next = _array(gamma_next, "gamma_next", ndim=1)
    q = _array(q, "q", ndim=1)
    q_next = _array(q_next, "q_next", ndim=1)
    if not (gamma.shape == gamma_next.shape == q.shape == q_next.shape):
        raise ValueError("attribution dimensions disagree")

    def energy(gate, metric):
        return float(np.sum(metric * gate * gate))

    f00 = energy(gamma, q)
    f10 = energy(gamma_next, q)
    f01 = energy(gamma, q_next)
    f11 = energy(gamma_next, q_next)
    gate_first_gate = f10 - f00
    gate_first_shape = f11 - f10
    shapley_gate = 0.5 * ((f10 - f00) + (f11 - f01))
    shapley_shape = 0.5 * ((f01 - f00) + (f11 - f10))
    return {
        "energy_before": f00,
        "energy_after": f11,
        "gate_first_gate": gate_first_gate,
        "gate_first_shape": gate_first_shape,
        "shapley_gate": shapley_gate,
        "shapley_shape": shapley_shape,
        "gate_first_residual": f11 - f00 - gate_first_gate - gate_first_shape,
        "shapley_residual": f11 - f00 - shapley_gate - shapley_shape,
    }


def three_factor_score_secant(
    query_reference, query_candidate, generator_reference,
    generator_candidate, key_reference, key_candidate,
):
    """Exact chronological Q/G/K telescoping identity for pre-mask scores."""

    query_reference = _array(query_reference, "query_reference", ndim=2)
    query_candidate = _array(query_candidate, "query_candidate", ndim=2)
    generator_reference = _array(
        generator_reference, "generator_reference", ndim=2)
    generator_candidate = _array(
        generator_candidate, "generator_candidate", ndim=2)
    key_reference = _array(key_reference, "key_reference", ndim=2)
    key_candidate = _array(key_candidate, "key_candidate", ndim=2)
    if not (
        query_reference.shape == query_candidate.shape
        and key_reference.shape == key_candidate.shape
        and generator_reference.shape == generator_candidate.shape
        and generator_reference.shape[0] == generator_reference.shape[1]
        and query_reference.shape[1] == generator_reference.shape[0]
        and key_reference.shape[1] == generator_reference.shape[1]
    ):
        raise ValueError("three-factor score endpoint dimensions disagree")
    dimension = generator_reference.shape[0]
    scale = math.sqrt(dimension)
    reference = (
        query_reference @ generator_reference @ key_reference.T / scale)
    candidate = (
        query_candidate @ generator_candidate @ key_candidate.T / scale)
    query_term = (
        (query_candidate - query_reference)
        @ generator_reference @ key_reference.T / scale)
    generator_term = (
        query_candidate @ (generator_candidate - generator_reference)
        @ key_reference.T / scale)
    key_term = (
        query_candidate @ generator_candidate
        @ (key_candidate - key_reference).T / scale)
    prediction = query_term + generator_term + key_term
    realized = candidate - reference
    denominator = max(
        float(np.linalg.norm(realized)),
        float(np.linalg.norm(prediction)), 1e-300)
    return {
        "reference_score": reference,
        "candidate_score": candidate,
        "query_term": query_term,
        "generator_term": generator_term,
        "key_term": key_term,
        "predicted_difference": prediction,
        "realized_difference": realized,
        "identity_residual": float(
            np.linalg.norm(realized - prediction)) / denominator,
    }


def occupied_power_coefficient(left, right, exponent):
    """Secant and occupied-segment derivative bounds for ``x**exponent``.

    Bases must be strictly positive.  Real exponents, including negative and
    zero exponents, are supported without a floor-wide extrapolation.
    """

    left = _scalar(left, "left")
    right = _scalar(right, "right")
    exponent = _scalar(exponent, "exponent")
    if left <= 0.0 or right <= 0.0:
        raise ValueError("occupied power bases must be strictly positive")
    if left == right:
        coefficient = exponent * left ** (exponent - 1.0)
    else:
        coefficient = (right ** exponent - left ** exponent) / (right - left)
    low_base, high_base = sorted((left, right))
    endpoint_derivatives = [
        exponent * low_base ** (exponent - 1.0),
        exponent * high_base ** (exponent - 1.0),
    ]
    return {
        "coefficient": float(coefficient),
        "derivative_lower": float(min(endpoint_derivatives)),
        "derivative_upper": float(max(endpoint_derivatives)),
        "occupied_minimum": low_base,
        "occupied_maximum": high_base,
        "reconstruction_residual": float(
            right ** exponent - left ** exponent
            - coefficient * (right - left)
        ),
    }
