import numpy as np
import pytest
import torch
from torch import nn
from torch.func import functional_call

from confirm.streamed_parameter_normal import (
    complete_fisher_normal_action,
    fisher_logit_action,
    flatten_tensors,
    full_loss_normal_action,
    linearized_normal_coordinates,
    normal_basis_from_dense_jacobian,
    parameter_tuple,
    selected_pair_normal_action,
    selected_pair_logit_action,
    stream_forward_jacobian,
    symmetric_lanczos,
    unflatten_tensor,
)
from confirm.comprehensive_collapse import softmax_fisher


class ToyMap(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, -0.5], [0.2, 0.8]]))
        self.bias = nn.Parameter(torch.tensor([0.1, -0.2]))

    def forward(self, row):
        return torch.tanh(self.weight @ row + self.bias)


def _functional(module, layout, row):
    def value(parameters):
        mapping = dict(zip(layout.names, parameters))
        return functional_call(module, mapping, (row,))
    return value


def test_streamed_forward_jacobian_matches_reverse_dense(tmp_path):
    module = ToyMap().double()
    layout, parameters = parameter_tuple(module)
    row = torch.tensor([0.3, -0.7], dtype=torch.float64)
    function = _functional(module, layout, row)
    path = tmp_path / "jacobian.npy"
    record = stream_forward_jacobian(
        function, parameters, layout, path, block_size=2)
    streamed = np.load(path)

    flat = flatten_tensors(parameters, layout).detach().clone()

    def flat_function(vector):
        values = unflatten_tensor(vector, layout, like=parameters)
        return function(values)

    dense = torch.autograd.functional.jacobian(flat_function, flat).numpy()
    assert record["shape"] == list(dense.shape)
    assert np.allclose(streamed, dense, rtol=1e-12, atol=1e-12)


def test_parameter_normal_basis_is_range_of_transpose():
    jacobian = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    result = normal_basis_from_dense_jacobian(jacobian)
    assert result["rank"] == 2
    assert result["range_residual"] < 1e-12
    assert result["orthogonality_residual"] < 1e-12
    assert result["left_orthogonality_residual"] < 1e-12

    parameter_displacements = np.asarray([
        [0.4, -0.2, 0.2],
        [-0.1, 0.3, 0.2],
    ])
    stencil_vectors = parameter_displacements @ jacobian.T
    recovered = linearized_normal_coordinates(
        stencil_vectors, result["left_basis"], result["singular_values"])
    expected = parameter_displacements @ result["basis"]
    assert np.allclose(recovered["coordinates"], expected, atol=1e-12)
    assert np.allclose(
        recovered["amplitudes"], np.linalg.norm(expected, axis=1),
        atol=1e-12)
    assert np.max(recovered["stencil_reconstruction_residuals"]) < 1e-12


def test_matrix_free_fisher_pair_and_true_hessian_actions():
    module = ToyMap().double()
    layout, parameters = parameter_tuple(module)
    row = torch.tensor([0.3, -0.7], dtype=torch.float64)
    logits = _functional(module, layout, row)
    normal_basis = np.eye(layout.dimension)
    coordinate = np.asarray([0.2, -0.1, 0.3, 0.4, -0.2, 0.5])
    base_logits = logits(parameters)
    probability = torch.softmax(base_logits, dim=0)

    fisher_action = complete_fisher_normal_action(
        logits, parameters, layout, normal_basis, probability, coordinate)
    pair_action = selected_pair_normal_action(
        logits, parameters, layout, normal_basis, probability,
        [(0, 1)], [float(probability[0] * probability[1])], coordinate)
    assert np.allclose(fisher_action, pair_action, rtol=1e-10, atol=1e-11)

    target = torch.tensor([1.0, 0.0], dtype=torch.float64)

    def loss(values):
        value = logits(values)
        return torch.logsumexp(value, dim=0) - torch.dot(target, value)

    full_action = full_loss_normal_action(
        loss, parameters, layout, normal_basis, coordinate)
    assert np.isfinite(full_action).all()
    assert not np.allclose(full_action, fisher_action)


def test_lanczos_intervals_cover_dense_ritz_eigenvalues():
    matrix = np.asarray([
        [2.0, 0.3, 0.0],
        [0.3, 1.0, -0.2],
        [0.0, -0.2, 0.5],
    ])
    result = symmetric_lanczos(
        lambda vector: matrix @ vector, 3, iterations=3, seed=17)
    expected = np.linalg.eigvalsh(matrix)
    assert np.allclose(result["ritz_values"], expected, atol=1e-12)
    assert np.max(result["residuals"]) < 1e-12
    assert result["orthogonality_residual"] < 1e-12


def test_batched_vocabulary_actions_are_blockwise():
    probability = torch.tensor(
        [[0.2, 0.3, 0.5], [0.6, 0.1, 0.3]], dtype=torch.float64)
    vector = torch.tensor(
        [[1.0, -2.0, 0.5], [-1.0, 3.0, 2.0]], dtype=torch.float64)
    fisher = fisher_logit_action(probability, vector)
    expected = np.stack([
        softmax_fisher(row.detach().numpy()) @ value.detach().numpy()
        for row, value in zip(probability, vector)
    ])
    assert np.allclose(fisher.detach().numpy(), expected)

    pair = selected_pair_logit_action(
        probability, vector, [(0, 0, 2), (1, 0, 1)],
        [0.1, 0.05])
    expected_pair = torch.zeros_like(vector)
    expected_pair[0, 0] = 0.1 * (vector[0, 0] - vector[0, 2])
    expected_pair[0, 2] = -expected_pair[0, 0]
    expected_pair[1, 0] = 0.05 * (vector[1, 0] - vector[1, 1])
    expected_pair[1, 1] = -expected_pair[1, 0]
    assert torch.allclose(pair, expected_pair)
