"""Streamed parameter-normal operators for the live confirmation program.

The implementation uses forward-mode JVPs for Jacobian columns and
JVP/Fisher/VJP compositions for the complete centered-logit Gram.  Large
finite-stencil Jacobians are written directly to a NumPy memmap, so no
output-by-parameter Jacobian is retained on a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch.func import grad, jvp, vjp


@dataclass(frozen=True)
class ParameterLayout:
    names: tuple[str, ...]
    shapes: tuple[tuple[int, ...], ...]
    sizes: tuple[int, ...]

    @property
    def dimension(self):
        return sum(self.sizes)


def parameter_tuple(module, predicate=None):
    rows = [
        (name, parameter.detach())
        for name, parameter in module.named_parameters()
        if predicate is None or predicate(name, parameter)
    ]
    if not rows:
        raise ValueError("the parameter-normal registry is empty")
    names, parameters = zip(*rows)
    layout = ParameterLayout(
        names=tuple(names),
        shapes=tuple(tuple(parameter.shape) for parameter in parameters),
        sizes=tuple(parameter.numel() for parameter in parameters),
    )
    return layout, tuple(parameters)


def flatten_tensors(tensors, layout):
    if len(tensors) != len(layout.names):
        raise ValueError("tensor tuple and parameter layout disagree")
    values = []
    for tensor, shape in zip(tensors, layout.shapes):
        if tuple(tensor.shape) != shape:
            raise ValueError("a tensor shape disagrees with its parameter layout")
        values.append(tensor.reshape(-1))
    return torch.cat(values)


def unflatten_tensor(vector, layout, *, like=None):
    if vector.ndim != 1 or vector.numel() != layout.dimension:
        raise ValueError("flat tangent has the wrong dimension")
    result = []
    cursor = 0
    for index, (shape, size) in enumerate(zip(layout.shapes, layout.sizes)):
        value = vector[cursor:cursor + size].reshape(shape)
        if like is not None:
            value = value.to(device=like[index].device, dtype=like[index].dtype)
        result.append(value)
        cursor += size
    return tuple(result)


def tangent_from_normal(normal_basis, coordinate, layout, parameters):
    basis = np.asarray(normal_basis, dtype=np.float64)
    coordinate = np.asarray(coordinate, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != layout.dimension:
        raise ValueError("normal basis has the wrong parameter dimension")
    if coordinate.shape != (basis.shape[1],):
        raise ValueError("normal coordinate has the wrong dimension")
    flat = torch.from_numpy(basis @ coordinate)
    return unflatten_tensor(flat, layout, like=parameters)


def stream_forward_jacobian(
        function, parameters, layout, output_path, *, block_size=8,
        output_dtype=np.float64):
    """Write all forward-mode Jacobian columns to a CPU ``.npy`` memmap."""

    if block_size <= 0:
        raise ValueError("tangent block size must be positive")
    base_output = function(parameters)
    if not torch.is_tensor(base_output):
        raise ValueError("the streamed function must return one tensor")
    output_dimension = base_output.numel()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.lib.format.open_memmap(
        destination, mode="w+", dtype=output_dtype,
        shape=(output_dimension, layout.dimension))
    peak_tangent_bytes = 0
    for start in range(0, layout.dimension, block_size):
        stop = min(start + block_size, layout.dimension)
        flat = torch.zeros(
            (stop - start, layout.dimension),
            device=parameters[0].device, dtype=parameters[0].dtype)
        columns = torch.arange(start, stop, device=flat.device)
        flat[torch.arange(stop - start, device=flat.device), columns] = 1
        peak_tangent_bytes = max(
            peak_tangent_bytes, flat.numel() * flat.element_size())
        for local, column in enumerate(range(start, stop)):
            tangent = unflatten_tensor(flat[local], layout, like=parameters)
            _, action = jvp(function, (parameters,), (tangent,))
            matrix[:, column] = (
                action.detach().reshape(-1).double().cpu().numpy())
        matrix.flush()
    return {
        "path": str(destination),
        "shape": [output_dimension, layout.dimension],
        "tangent_block_size": int(block_size),
        "peak_tangent_bytes": int(peak_tangent_bytes),
    }


def normal_basis_from_dense_jacobian(jacobian, *, relative_tolerance=None):
    """Return an orthonormal basis of ``range(J.T)`` and rank residuals."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("finite-stencil Jacobian must be a finite matrix")
    if relative_tolerance is None:
        relative_tolerance = max(matrix.shape) * np.finfo(float).eps
    if not math.isfinite(relative_tolerance) or relative_tolerance <= 0:
        raise ValueError("rank tolerance must be finite and positive")
    left, singular, right = np.linalg.svd(matrix, full_matrices=False)
    threshold = (
        relative_tolerance * singular[0] if singular.size and singular[0] > 0
        else relative_tolerance)
    rank = int(np.sum(singular > threshold))
    basis = right[:rank].T.copy()
    left_basis = left[:, :rank].copy()
    residual = np.linalg.norm(matrix - matrix @ basis @ basis.T, ord=2)
    orthogonality = np.linalg.norm(basis.T @ basis - np.eye(rank), ord=2)
    left_orthogonality = np.linalg.norm(
        left_basis.T @ left_basis - np.eye(rank), ord=2)
    return {
        "basis": basis,
        "left_basis": left_basis,
        "rank": rank,
        "singular_values": singular,
        "rank_threshold": float(threshold),
        "range_residual": float(residual),
        "orthogonality_residual": float(orthogonality),
        "left_orthogonality_residual": float(left_orthogonality),
    }


def linearized_normal_coordinates(
        stencil_vectors, left_basis, singular_values):
    """Recover least-norm first-order normal coordinates from ``Z_h``."""

    vectors = np.asarray(stencil_vectors, dtype=np.float64)
    left = np.asarray(left_basis, dtype=np.float64)
    singular = np.asarray(singular_values, dtype=np.float64)
    if vectors.ndim != 2 or left.ndim != 2 or singular.ndim != 1:
        raise ValueError("linearized normal factors have wrong dimensions")
    if vectors.shape[1] != left.shape[0]:
        raise ValueError("stencil vectors and left singular basis disagree")
    rank = left.shape[1]
    if singular.size < rank or np.any(singular[:rank] <= 0):
        raise ValueError("retained singular values must be strictly positive")
    if not (np.isfinite(vectors).all() and np.isfinite(left).all()
            and np.isfinite(singular).all()):
        raise ValueError("linearized normal factors must be finite")
    coordinates = (vectors @ left) / singular[:rank]
    projected = (coordinates * singular[:rank]) @ left.T
    residuals = np.linalg.norm(vectors - projected, axis=1)
    return {
        "coordinates": coordinates,
        "amplitudes": np.linalg.norm(coordinates, axis=1),
        "stencil_reconstruction_residuals": residuals,
    }


def logit_jvp(logit_function, parameters, tangent):
    _, action = jvp(logit_function, (parameters,), (tangent,))
    return action


def logit_vjp(logit_function, parameters, cotangent):
    output, pullback = vjp(logit_function, parameters)
    if tuple(output.shape) != tuple(cotangent.shape):
        raise ValueError("logit cotangent has the wrong shape")
    return pullback(cotangent)[0]


def fisher_logit_action(probability, vector):
    if probability.shape != vector.shape:
        raise ValueError("probability and logit tangent shapes disagree")
    if probability.ndim not in (1, 2):
        raise ValueError("logit actions must have vocabulary or source-vocabulary shape")
    return probability * vector - probability * torch.sum(
        probability * vector, dim=-1, keepdim=probability.ndim == 2)


def complete_fisher_normal_action(
        logit_function, parameters, layout, normal_basis, probability,
        coordinate):
    tangent = tangent_from_normal(
        normal_basis, coordinate, layout, parameters)
    logit_tangent = logit_jvp(logit_function, parameters, tangent)
    fisher_tangent = fisher_logit_action(probability, logit_tangent)
    pulled = logit_vjp(logit_function, parameters, fisher_tangent)
    flat = flatten_tensors(pulled, layout).detach().double().cpu().numpy()
    return np.asarray(normal_basis, dtype=np.float64).T @ flat


def selected_pair_logit_action(probability, vector, edges, lower_products):
    if probability.shape != vector.shape:
        raise ValueError("probability and logit tangent shapes disagree")
    if len(edges) != len(lower_products):
        raise ValueError("selected edges and product bounds disagree")
    output = torch.zeros_like(vector)
    for edge, raw_bound in zip(edges, lower_products):
        bound = float(raw_bound)
        if probability.ndim == 1:
            if len(edge) != 2:
                raise ValueError("a vector logit edge needs two indices")
            left, right = edge
            realized = float(probability[left] * probability[right])
            difference = vector[left] - vector[right]
        elif probability.ndim == 2:
            if len(edge) != 3:
                raise ValueError("a source-vocabulary edge needs three indices")
            source, left, right = edge
            realized = float(
                probability[source, left] * probability[source, right])
            difference = vector[source, left] - vector[source, right]
        else:
            raise ValueError("selected pair actions require one or two dimensions")
        if bound < 0 or bound > realized * (1 + 1e-12):
            raise ValueError("selected probability-product bound is invalid")
        if probability.ndim == 1:
            output[left] += bound * difference
            output[right] -= bound * difference
        else:
            output[source, left] += bound * difference
            output[source, right] -= bound * difference
    return output


def selected_pair_normal_action(
        logit_function, parameters, layout, normal_basis, probability,
        edges, lower_products, coordinate):
    tangent = tangent_from_normal(
        normal_basis, coordinate, layout, parameters)
    logit_tangent = logit_jvp(logit_function, parameters, tangent)
    pair_tangent = selected_pair_logit_action(
        probability, logit_tangent, edges, lower_products)
    pulled = logit_vjp(logit_function, parameters, pair_tangent)
    flat = flatten_tensors(pulled, layout).detach().double().cpu().numpy()
    return np.asarray(normal_basis, dtype=np.float64).T @ flat


def full_loss_normal_action(
        loss_function, parameters, layout, normal_basis, coordinate):
    tangent = tangent_from_normal(
        normal_basis, coordinate, layout, parameters)
    gradient_function = grad(loss_function)
    _, hessian_tangent = jvp(
        gradient_function, (parameters,), (tangent,))
    flat = flatten_tensors(
        hessian_tangent, layout).detach().double().cpu().numpy()
    return np.asarray(normal_basis, dtype=np.float64).T @ flat


def symmetric_lanczos(action, dimension, *, iterations, seed):
    """Return Ritz values and residual intervals for a symmetric action."""

    if not 1 <= iterations <= dimension:
        raise ValueError("Lanczos iteration count is outside its domain")
    generator = np.random.default_rng(int(seed))
    vector = generator.normal(size=dimension)
    vector /= np.linalg.norm(vector)
    basis = []
    diagonal = []
    off_diagonal = []
    previous = np.zeros(dimension)
    beta = 0.0
    for index in range(iterations):
        basis.append(vector.copy())
        image = np.asarray(action(vector), dtype=np.float64)
        if image.shape != (dimension,) or not np.isfinite(image).all():
            raise ValueError("Lanczos action returned an invalid vector")
        work = image - beta * previous
        alpha = float(np.dot(vector, work))
        work -= alpha * vector
        for existing in basis:
            work -= np.dot(existing, work) * existing
        next_beta = float(np.linalg.norm(work))
        diagonal.append(alpha)
        if index + 1 < iterations:
            off_diagonal.append(next_beta)
        if next_beta <= 64 * np.finfo(float).eps:
            break
        previous, vector, beta = vector, work / next_beta, next_beta
    size = len(diagonal)
    tridiagonal = np.diag(diagonal)
    if size > 1:
        values = np.asarray(off_diagonal[:size - 1])
        tridiagonal += np.diag(values, 1) + np.diag(values, -1)
    ritz, vectors = np.linalg.eigh(tridiagonal)
    terminal_beta = next_beta if size == iterations else 0.0
    residuals = np.abs(terminal_beta * vectors[-1])
    return {
        "ritz_values": ritz,
        "residuals": residuals,
        "intervals": np.stack((ritz - residuals, ritz + residuals), axis=1),
        "iterations_completed": size,
        "orthogonality_residual": float(np.linalg.norm(
            np.asarray(basis) @ np.asarray(basis).T - np.eye(size), ord=2)),
    }
