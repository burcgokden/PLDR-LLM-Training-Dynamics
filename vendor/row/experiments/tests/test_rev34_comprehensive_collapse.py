import math

import numpy as np
import pytest

from confirm.comprehensive_live_producers import (
    _intervention_statistics as producer_intervention_statistics,
    _locked_application_certificate,
    _scalar_live_adamw_operator,
)
from analysis.comprehensive_confirmation_analysis import (
    _intervention_statistics as analyzer_intervention_statistics,
)

from confirm.comprehensive_collapse import (
    adamw_successor_jacobian,
    finite_stencil_stack,
    joint_row_direction_cover,
    full_cross_entropy_response,
    metric_gain,
    oriented_integral_bound,
    pairwise_softmax_energy,
    relative_curvature_edge,
    robust_path_gains,
    scheduled_path_metrics,
    selected_pair_frame,
    softmax_fisher,
    stencil_direct_bound,
    variable_radius_envelope,
    vector_bernstein_radius,
)


def test_pairwise_softmax_identity_and_selected_lower_frame():
    probability = np.asarray([0.2, 0.3, 0.5])
    vector = np.asarray([1.2, -0.7, 0.4])
    expected = vector @ softmax_fisher(probability) @ vector
    assert pairwise_softmax_energy(probability, vector) == pytest.approx(expected)

    jacobians = np.asarray([
        [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
        [[0.0, 1.0], [1.0, 0.0], [-1.0, -1.0]],
    ])
    probabilities = np.asarray([
        [0.2, 0.3, 0.5],
        [0.4, 0.1, 0.5],
    ])
    edges = [[(0, 1), (1, 2)], [(0, 2), (0, 1)]]
    lower_products = [[0.05, 0.12], [0.18, 0.03]]
    frame = selected_pair_frame(
        jacobians, probabilities, edges,
        lower_products=lower_products)
    assert np.linalg.eigvalsh(frame["lower"])[0] > 0
    assert np.linalg.eigvalsh(frame["exact"] - frame["lower"])[0] > -1e-12


def test_selected_pair_frame_rejects_unsealed_probability_overclaim():
    with pytest.raises(ValueError, match="probability-product"):
        selected_pair_frame(
            np.asarray([[[1.0], [0.0]]]),
            np.asarray([[0.25, 0.75]]),
            [[(0, 1)]],
            lower_products=[[0.2]],
        )


def test_true_cross_entropy_response_keeps_signed_second_jet():
    probability = np.asarray([[0.6, 0.4]])
    target = np.asarray([[1.0, 0.0]])
    jacobian = np.asarray([[[1.0], [-1.0]]])
    second = np.asarray([[[[2.0]], [[-1.0]]]])
    response = full_cross_entropy_response(
        jacobian, probability, target, second)
    expected_residual = (-0.4) * 2.0 + 0.4 * -1.0
    assert response["residual"][0, 0] == pytest.approx(expected_residual)
    assert response["full"][0, 0] == pytest.approx(
        response["fisher"][0, 0] + expected_residual)


def test_relative_curvature_uses_orientation_not_ambient_norm():
    fisher = np.diag([2.0, 8.0])
    residual = np.diag([-0.5, 3.0])
    result = relative_curvature_edge(2.0, residual, fisher)
    assert result["chi"] == pytest.approx(0.25)
    assert result["relative_bound_edge"] == pytest.approx(1.5)
    assert result["full_edge"] == pytest.approx(1.5)


def _adamw_successor(state, gradient_matrix, clipped_gradient, *, values):
    dimension = clipped_gradient.size
    normal = state[:dimension]
    first = state[dimension:2 * dimension]
    second = state[2 * dimension:]
    gradient = clipped_gradient + gradient_matrix @ normal
    first_next = values["beta1"] * first + (1 - values["beta1"]) * gradient
    second_next = (
        values["beta2"] * second
        + (1 - values["beta2"]) * gradient ** 2
    )
    first_hat = first_next / (1 - values["beta1"] ** values["step"])
    second_hat = second_next / (1 - values["beta2"] ** values["step"])
    update = first_hat / (np.sqrt(second_hat) + values["epsilon"])
    normal_next = (
        (1 - values["learning_rate"] * values["weight_decay"])
        * normal
        - values["learning_rate"] * update
    )
    return np.concatenate((normal_next, first_next, second_next))


def test_full_adamw_successor_jacobian_matches_centered_finite_difference():
    gradient_matrix = np.asarray([[1.3, -0.2], [0.4, 0.8]])
    gradient = np.asarray([0.25, -0.4])
    first = np.asarray([0.1, -0.2])
    second = np.asarray([0.7, 0.9])
    values = {
        "learning_rate": 3e-3,
        "beta1": 0.9,
        "beta2": 0.99,
        "epsilon": 1e-8,
        "step": 17,
        "weight_decay": 0.02,
    }
    result = adamw_successor_jacobian(
        gradient_matrix, gradient, first, second,
        learning_rate=values["learning_rate"],
        beta1=values["beta1"], beta2=values["beta2"],
        epsilon=values["epsilon"], optimizer_step=values["step"],
        weight_decay=values["weight_decay"])
    base = np.concatenate((np.zeros(2), first, second))
    direction = np.asarray([0.3, -0.7, 0.1, 0.2, -0.4, 0.5])
    step = 1e-6
    plus = _adamw_successor(
        base + step * direction, gradient_matrix, gradient, values=values)
    minus = _adamw_successor(
        base - step * direction, gradient_matrix, gradient, values=values)
    finite_difference = (plus - minus) / (2 * step)
    assert np.allclose(
        result["operator"] @ direction, finite_difference,
        rtol=2e-7, atol=2e-9)

    scalar = adamw_successor_jacobian(
        np.asarray([[gradient_matrix[0, 0]]]), gradient[:1], first[:1],
        second[:1], learning_rate=values["learning_rate"],
        beta1=values["beta1"], beta2=values["beta2"],
        epsilon=values["epsilon"], optimizer_step=values["step"],
        weight_decay=values["weight_decay"])
    live_scalar = _scalar_live_adamw_operator(
        gradient_matrix[0, 0], gradient[0], first[0], second[0],
        learning_rate=values["learning_rate"], beta1=values["beta1"],
        beta2=values["beta2"], epsilon=values["epsilon"],
        optimizer_step=values["step"], weight_decay=values["weight_decay"],
        clip_derivative=1.0, decay_mask=1.0)
    assert np.allclose(scalar["operator"], live_scalar, atol=1e-12, rtol=1e-12)


def test_scheduled_metrics_telescope_and_robust_gain_charges_motion():
    operators = [
        np.asarray([[1.2, 0.5], [0.0, 0.3]]),
        np.asarray([[0.2, -0.1], [0.4, 0.7]]),
    ]
    schedule = scheduled_path_metrics(operators)
    assert len(schedule["metrics"]) == 3
    for index, operator in enumerate(operators):
        gain = metric_gain(
            operator, schedule["metrics"][index],
            schedule["metrics"][index + 1])
        assert gain == pytest.approx(schedule["gains"][index])
        assert gain < 1
    live = [operators[0] + 0.01 * np.eye(2), operators[1]]
    robust = robust_path_gains(operators, live, schedule["metrics"])
    assert robust[0, 2] >= robust[0, 0]
    assert robust[1, 1] == pytest.approx(0.0, abs=1e-15)


def test_variable_radius_allows_transient_expansion_and_contracting_block():
    gains = np.asarray([1.15, 0.45])
    quadratic = np.asarray([0.02, 0.02])
    forces = np.asarray([0.01, 0.01])
    radii = np.asarray([0.5, 0.60, 0.30])
    result = variable_radius_envelope(
        0.4, gains, quadratic, forces, radii)
    assert result["effective_gains"][0] > 1
    assert np.prod(result["effective_gains"]) < 1
    assert np.all(result["envelope"] <= radii + 1e-12)
    assert result["initial_inside"]
    assert result["forward_invariant"]
    assert result["envelope_inside"]

    failed = variable_radius_envelope(
        0.4, gains, quadratic, forces, np.asarray([0.5, 0.50, 0.20]),
        require_invariant=False)
    assert not failed["forward_invariant"]
    with pytest.raises(ValueError, match="not forward invariant"):
        variable_radius_envelope(
            0.4, gains, quadratic, forces,
            np.asarray([0.5, 0.50, 0.20]))


def test_finite_stencil_and_cover_bridge_for_quadratic_row_map():
    matrix = np.asarray([[2.0, 0.0], [0.0, -1.0]])

    def phi(row):
        return matrix @ row + np.asarray([0.5 * row[0] ** 2, 0.0])

    rows = np.asarray([[0.0, 0.0], [0.2, -0.1]])
    directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    stack = finite_stencil_stack(phi, rows, directions, 0.01)
    assert stack.shape == (4,)
    bound = stencil_direct_bound(
        np.max(np.abs(stack)), 1.0, 0.01, 0.1, 0.02, 1e-8)
    expected = (np.max(np.abs(stack)) + 1e-8 + 0.005 + 0.02) / 0.9
    assert bound == pytest.approx(expected)


def test_joint_row_direction_cover_uses_one_registered_pair_per_query():
    registered_rows = np.asarray([[0.0, 0.0], [1.0, 0.0]])
    registered_directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    query_rows = np.asarray([[0.1, 0.0], [0.9, 0.0]])
    query_directions = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    result = joint_row_direction_cover(
        registered_rows, registered_directions, query_rows, query_directions,
        row_scale=1.0, block_size=1)
    assert result["selected_registered_index"].tolist() == [0, 1]
    assert result["maximum_row_radius"] == pytest.approx(0.1)
    assert result["maximum_direction_radius"] == pytest.approx(0.0)


def test_oriented_integral_retains_cancellation_and_margin_shape():
    result = oriented_integral_bound(
        np.asarray([[10.0, 0.0], [-10.0, 0.0]]),
        np.asarray([0.1, 0.1]),
        np.asarray([0.5, 0.5]))
    assert result["upper_norm"] == pytest.approx(0.1)
    assert result["stagewise_triangle_bound"] == pytest.approx(10.1)

    radii, center_step = _locked_application_certificate({
        "application": [
            {
                "subdivisions": 2,
                "cell_average_radii": [0.1, 0.2],
                "center_finite_difference_step": 1e-4,
            },
            {
                "subdivisions": 2,
                "cell_average_radii": [0.15, 0.12],
                "center_finite_difference_step": 1e-4,
            },
        ],
    }, 2)
    assert np.allclose(radii, [0.15, 0.2])
    assert center_step == pytest.approx(1e-4)


def test_vector_bernstein_radius_has_expected_batch_scaling():
    small = vector_bernstein_radius(2.0, 3.0, 64, 10, 0.01)
    large = vector_bernstein_radius(2.0, 3.0, 256, 10, 0.01)
    logarithm = math.log(2.0 * 11 / 0.01)
    expected = (
        math.sqrt(2.0 * 2.0 * logarithm / 256)
        + 2.0 * 3.0 * logarithm / (3.0 * 256))
    assert large < small
    assert large == pytest.approx(expected)
    assert math.isfinite(large)


def test_rate_intervention_statistics_use_the_normal_series():
    series = np.asarray([2.0, 1.0, 1.5, 0.75])
    expected_damping = math.log(0.75 / 2.0) / 3.0
    expected_phase = float(np.sum(np.arctan2(
        np.diff(series), series[1:])))
    for function in (
            producer_intervention_statistics,
            analyzer_intervention_statistics):
        damping, phase, onset = function(series)
        assert damping == pytest.approx(expected_damping)
        assert phase == pytest.approx(expected_phase)
        assert onset == 1.0
