import math

import numpy as np

from analysis.analyze_causal_confirmation import analyze_gate_route
from confirm.causal_row_map import (
    adamw_gate_route_envelope,
    decompose_linear_routes,
    geometric_convolution,
    iterate_affine_diameter_bound,
    pair_energy_enclosure,
    stream_registry_prediction,
    unordered_pair_chunks,
)
from confirm.composite_plga import (
    plga_transfer,
    restricted_operator_coefficients,
    row_center,
    row_variance_identity,
)
from confirm.postnorm_geometry import (
    backward_path_metrics,
    chord_action_residual,
    compose_factors,
    layernorm_chord_operator,
    verify_path_metric,
)


def test_pair_energy_enclosure_covers_random_remainders():
    rng = np.random.default_rng(4001)
    source = rng.normal(size=(64, 7))
    directional = 0.1 * rng.normal(size=(64, 7))
    remainder = 0.01 * rng.normal(size=(64, 7))
    radius = np.linalg.norm(remainder, axis=1)
    result = pair_energy_enclosure(source, directional, radius)
    realized = np.einsum(
        "pd,pd->p",
        source + directional + remainder,
        source + directional + remainder,
    )
    increment = realized - result["source_energy"]
    assert np.all(result["lower_increment"] <= increment)
    assert np.all(increment <= result["upper_increment"])


def test_registry_prediction_certifies_both_directions_and_zero_state():
    rows = np.asarray([[1.0, 0.0], [0.0, 0.0], [-0.5, 0.25]])
    radius = np.zeros(3)
    contraction = stream_registry_prediction(
        rows, -0.2 * rows, radius, pair_chunk_size=2,
        successor_rows=0.8 * rows,
    )
    assert contraction["decision"] == "CONTRACTION_CERTIFIED"
    assert contraction["coverage"]["all_pair_enclosures_hold"]
    assert contraction["source_diameter_pair"] == [0, 2]
    expansion = stream_registry_prediction(
        rows, 0.2 * rows, radius, pair_chunk_size=2,
        successor_rows=1.2 * rows,
    )
    assert expansion["decision"] == "REEXPANSION_CERTIFIED"
    assert expansion["coverage"]["all_pair_enclosures_hold"]
    zero = stream_registry_prediction(
        np.zeros((3, 2)), np.zeros((3, 2)), radius,
    )
    assert zero["decision"] == "UNRESOLVED"
    assert zero["diameter_ratio_upper"] is None



def test_arithmetic_source_interval_controls_strict_decisions():
    rows = np.asarray([[1.0, 0.0], [0.0, 0.0], [-0.5, 0.25]])
    radius = np.zeros(3)
    certified = stream_registry_prediction(
        rows, -0.2 * rows, radius, successor_rows=0.8 * rows)
    assert certified["source_diameter_lower_squared"] <= (
        certified["source_diameter_squared"])
    assert certified["source_diameter_squared"] <= (
        certified["source_diameter_upper_squared"])
    assert certified["arithmetic"]["source_interval_used_for_decision"]
    assert certified["coverage"]["realized_energy_interval_charged"]
    unresolved = stream_registry_prediction(
        rows,
        -0.2 * rows,
        radius,
        extra_energy_charge=10.0,
        successor_rows=0.8 * rows,
    )
    assert unresolved["decision"] == "UNRESOLVED"


def test_unordered_pair_stream_is_complete_and_lexicographic():
    chunks = list(unordered_pair_chunks(7, 4))
    pairs = np.concatenate(chunks, axis=0)
    expected = np.asarray([
        (left, right)
        for left in range(6)
        for right in range(left + 1, 7)
    ])
    assert np.array_equal(pairs, expected)


def test_route_attribution_retains_total_quadratic_cross_terms():
    rng = np.random.default_rng(4002)
    source = rng.normal(size=(9, 4))
    gate = rng.normal(size=(9, 4))
    shape = rng.normal(size=(9, 4))
    total = gate + shape
    result = decompose_linear_routes(
        source, {"gate": gate, "shape": shape}, total)
    expected = np.einsum("pd,pd->p", total, total)
    assert np.allclose(
        result["total_directional_energy_with_cross_terms"], expected)
    assert result["maximum_jvp_reconstruction_relative_residual"] < 1e-15


def test_affine_block_and_gate_route_envelopes():
    envelope = iterate_affine_diameter_bound(
        4.0, np.asarray([0.5, 0.5]), np.asarray([0.0, 0.25]))
    assert np.allclose(envelope, [4.0, 1.0, 0.5])
    direct = sum(0.8 ** (5 - r) * 0.7 ** r for r in range(6))
    assert math.isclose(geometric_convolution(0.8, 0.7, 5), direct)
    equal = 6 * 0.8 ** 5
    assert math.isclose(geometric_convolution(0.8, 0.8, 5), equal)
    route = adamw_gate_route_envelope(
        initial_gate=1.0,
        initial_first_moment=0.2,
        gradient_envelope=0.1,
        gradient_rate=0.8,
        beta1=0.9,
        learning_rate_upper=7.5e-4,
        adam_epsilon=1e-5,
        bias_correction_floor=0.1,
        decay_multiplier_upper=0.999925,
        horizon=12,
    )
    assert route["gate_upper"].shape == (13,)
    assert np.all(route["innovation_upper"] >= 0.0)

    source = np.asarray([1.2, -0.7, 0.3])
    history = np.asarray([[0.2, -0.1, 0.4]])
    tail = np.asarray([0.03, -0.08, 0.11])
    second_before = np.asarray([0.4, 0.2, 0.7])
    beta1 = 0.9
    beta2 = 0.95
    epsilon = 1.0e-5
    step = 1001
    first_after = beta1 * tail + (1.0 - beta1) * history[0]
    second_after = (
        beta2 * second_before + (1.0 - beta2) * history[0] ** 2)
    direction = (
        first_after / (1.0 - beta1 ** step)
        / (np.sqrt(second_after / (1.0 - beta2 ** step)) + epsilon)
    )
    predicted = 0.999925 * source - 7.5e-4 * direction
    replay = analyze_gate_route(
        source,
        predicted,
        history,
        tail,
        second_before,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        learning_rate=7.5e-4,
        weight_decay=0.1,
        decay_mask=np.ones(3),
        optimizer_step_after=step,
    )
    assert replay["qualified"]
    assert replay["decay_multiplier_minimum"] == 0.999925
    assert replay["maximum_gate_replay_absolute_residual"] == 0.0


def test_layernorm_analytic_chord_and_path_metric():
    rng = np.random.default_rng(4003)
    left = rng.normal(size=8)
    right = rng.normal(size=8)
    gamma = rng.uniform(0.3, 1.7, size=8)
    beta = rng.normal(size=8)
    chord = layernorm_chord_operator(
        left, right, gamma=gamma, epsilon=1e-6)
    assert chord_action_residual(
        left, right, chord, gamma=gamma, beta=beta, epsilon=1e-6
    ) < 2e-14
    diagonal = layernorm_chord_operator(
        left, left, gamma=gamma, epsilon=1e-6)
    assert np.isfinite(diagonal).all()

    factors = [
        np.diag(np.linspace(0.5, 0.8, 4)),
        np.diag(np.linspace(0.4, 0.7, 4)),
        np.diag(np.linspace(0.6, 0.9, 4)),
    ]
    product = compose_factors(factors)
    assert np.allclose(product, factors[2] @ factors[1] @ factors[0])
    metrics, gains = backward_path_metrics(factors)
    certificate = verify_path_metric(factors, metrics, gains)
    assert certificate["quadratic_inequalities_hold"]
    assert certificate["physical_euclidean_coefficient"] < 1.0


def test_composite_plga_identity_variance_and_restricted_operators():
    rng = np.random.default_rng(4004)
    dimension = 5
    value = rng.normal(scale=0.2, size=(dimension, dimension))
    centered, reference = row_center(value)
    assert np.allclose(np.sum(centered, axis=0), 0.0, atol=1e-15)
    weight = rng.normal(scale=0.15, size=value.shape)
    bias = rng.normal(scale=0.1, size=value.shape)
    exponent = rng.uniform(-0.25, 0.75, size=value.shape)
    coupling = rng.normal(scale=0.15, size=value.shape)
    coupling_bias = rng.normal(scale=0.1, size=value.shape)
    transfer = plga_transfer(
        value, reference, weight, bias, exponent, coupling, coupling_bias)
    assert transfer["identity_relative_residual"] < 2e-12
    variance = row_variance_identity(value)
    assert variance["identity_relative_residual"] < 2e-15
    assert (
        variance["centered_frobenius_squared"]
        <= variance["diameter_upper_squared"] * (1.0 + 1e-14)
    )
    query = rng.normal(size=(7, dimension))
    key = rng.normal(size=(7, dimension))
    coefficients = restricted_operator_coefficients(
        composite_secant_value=transfer["composite_secant"],
        weight=weight,
        coupling=coupling,
        query=query,
        key=key,
        iterations=80,
    )
    assert all(math.isfinite(value) and value >= 0.0
               for value in coefficients.values())
