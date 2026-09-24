"""Certified-grid instrumentation for the implemented PLDR row map.

The primary endpoint is an upper bound in physical row units.  Dense basis
reconstruction supplies the complete Jacobian at each cover point.  A
separate, externally justified derivative modulus transfers the grid bound
to the declared tube.  Sampled rows alone are labelled as a finite-row cover
and are never promoted to a compact-tube certificate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np
import torch


def _finite_array(value, name):
    array = np.asarray(value, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains a nonfinite value")
    return array


def _row_map_value(model, layer_index: int, row: torch.Tensor):
    """Evaluate the shared metric-learner row map of one decoder layer."""
    layers = model.decoder.dec_layers
    if not 0 <= layer_index < len(layers):
        raise IndexError("layer_index is outside the decoder")
    value = row
    for unit in layers[layer_index].mha1.reslayerAs:
        value = unit([value])
    return value


def dense_row_jacobian(model, layer_index: int, row: torch.Tensor):
    """Return the full output-by-input Jacobian using dense autograd."""
    point = row.detach().clone().requires_grad_(True)
    if point.ndim != 1:
        raise ValueError("a row-map point must be one-dimensional")
    jacobian = torch.autograd.functional.jacobian(
        lambda value: _row_map_value(model, layer_index, value),
        point,
        create_graph=False,
        strict=True,
        vectorize=False,
    )
    if jacobian.shape != (point.numel(), point.numel()):
        raise RuntimeError("row map did not return one row of equal width")
    return jacobian.detach()


def basis_row_jacobian(model, layer_index: int, row: torch.Tensor):
    """Reconstruct a full Jacobian from exactly ``d_k`` basis actions."""
    point = row.detach().clone().requires_grad_(True)
    if point.ndim != 1:
        raise ValueError("a row-map point must be one-dimensional")
    columns = []
    for index in range(point.numel()):
        tangent = torch.zeros_like(point)
        tangent[index] = 1
        _, action = torch.autograd.functional.jvp(
            lambda value: _row_map_value(model, layer_index, value),
            (point,), (tangent,), create_graph=False, strict=True,
        )
        columns.append(action.detach())
    return torch.stack(columns, dim=1)


def finite_difference_jacobian(function: Callable[[torch.Tensor], torch.Tensor],
                               row: torch.Tensor, step: float):
    """Central-difference diagnostic.  It is not a certified upper bound."""
    if not math.isfinite(step) or step <= 0:
        raise ValueError("finite-difference step must be positive and finite")
    point = row.detach().clone()
    if point.ndim != 1:
        raise ValueError("a row-map point must be one-dimensional")
    columns = []
    with torch.no_grad():
        for index in range(point.numel()):
            tangent = torch.zeros_like(point)
            tangent[index] = step
            columns.append(
                (function(point + tangent) - function(point - tangent))
                / (2.0 * step)
            )
    return torch.stack(columns, dim=1)


def greedy_finite_row_cover(rows, radius: float):
    """Deterministic cover of a finite row array in input order.

    The returned domain is explicitly ``FINITE_ROWS``.  Covering a sampled
    array does not establish a cover of the surrounding compact tube.
    """
    array = _finite_array(rows, "rows")
    if array.ndim != 2 or not len(array):
        raise ValueError("rows must be a nonempty matrix")
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("radius must be nonnegative and finite")
    centers = []
    assignment = []
    distances = []
    for row in array:
        if not centers:
            centers.append(row.copy())
        candidate = np.asarray(centers)
        dists = np.linalg.norm(candidate - row[None, :], axis=1)
        nearest = int(np.argmin(dists))
        if dists[nearest] > radius:
            centers.append(row.copy())
            nearest = len(centers) - 1
            distance = 0.0
        else:
            distance = float(dists[nearest])
        assignment.append(nearest)
        distances.append(distance)
    return {
        "domain": "FINITE_ROWS",
        "centers": np.asarray(centers),
        "assignment": np.asarray(assignment, dtype=int),
        "max_distance": float(max(distances, default=0.0)),
        "requested_radius": float(radius),
    }


@dataclass(frozen=True)
class DerivativeModulusCertificate:
    value: float
    domain: str
    method: str
    status: str = "CERTIFIED"

    def validate(self):
        if self.status != "CERTIFIED":
            raise ValueError("derivative modulus is not certified")
        if self.domain != "DECLARED_COMPACT_TUBE":
            raise ValueError("derivative modulus does not cover the tube")
        if not math.isfinite(self.value) or self.value < 0:
            raise ValueError("derivative modulus must be finite and nonnegative")
        if not self.method:
            raise ValueError("derivative modulus method is missing")


def full_matrix_cover_upper(jacobians, numerical_errors,
                            cover_radius: float,
                            modulus: DerivativeModulusCertificate):
    """Evaluate the deterministic full-matrix tube-cover theorem."""
    matrices = _finite_array(jacobians, "jacobians")
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] \
            or not len(matrices):
        raise ValueError("jacobians must have shape [points, d_k, d_k]")
    errors = _finite_array(numerical_errors, "numerical_errors")
    if errors.ndim == 0:
        errors = np.full(len(matrices), float(errors))
    if errors.shape != (len(matrices),) or (errors < 0).any():
        raise ValueError("numerical errors must be nonnegative per point")
    if not math.isfinite(cover_radius) or cover_radius < 0:
        raise ValueError("cover radius must be nonnegative and finite")
    modulus.validate()
    singular_maxima = np.linalg.svd(matrices, compute_uv=False)[:, 0]
    grid_upper = float(np.max(singular_maxima + errors))
    row_remainder = float(modulus.value * cover_radius)
    direct_upper = grid_upper + row_remainder
    return {
        "grid_upper": grid_upper,
        "row_remainder": row_remainder,
        "direct_upper": direct_upper,
        "spectral_norms": singular_maxima.tolist(),
        "numerical_errors": errors.tolist(),
        "normalizer": 1.0,
        "units": "output_row_unit_per_input_row_unit",
    }


def resource_cost(n_points: int, width: int, itemsize: int = 8):
    if n_points <= 0 or width <= 0 or itemsize <= 0:
        raise ValueError("resource dimensions must be positive")
    return {
        "jacobian_actions": int(n_points * width),
        "dense_svd_cubic_units": int(n_points * width ** 3),
        "jacobian_storage_bytes": int(n_points * width * width * itemsize),
        "points": int(n_points),
        "width": int(width),
    }


def _generic_dense_jacobian(function, point):
    value = point.detach().clone().requires_grad_(True)
    return torch.autograd.functional.jacobian(
        function, value, vectorize=False, strict=True).detach()


def _generic_basis_jacobian(function, point):
    value = point.detach().clone().requires_grad_(True)
    columns = []
    for index in range(value.numel()):
        tangent = torch.zeros_like(value)
        tangent[index] = 1
        _, action = torch.autograd.functional.jvp(
            function, (value,), (tangent,), strict=True)
        columns.append(action.detach())
    return torch.stack(columns, dim=1)


def qualify_synthetic(device="cpu"):
    """Run exact linear, planted quadratic, and small PLDR fixtures."""
    torch.manual_seed(41017)
    dtype = torch.float64
    width = 4
    matrix = torch.randn(width, width, dtype=dtype, device=device)
    bias = torch.randn(width, dtype=dtype, device=device)
    point = torch.randn(width, dtype=dtype, device=device)

    linear = lambda value: matrix @ value + bias
    dense_linear = _generic_dense_jacobian(linear, point)
    basis_linear = _generic_basis_jacobian(linear, point)
    linear_error = float(torch.linalg.matrix_norm(
        dense_linear - matrix, ord=2))
    basis_error = float(torch.linalg.matrix_norm(
        dense_linear - basis_linear, ord=2))

    quad_weight = torch.tensor([0.2, -0.4, 0.1, 0.3],
                               dtype=dtype, device=device)
    quadratic = lambda value: matrix @ value + quad_weight * value.square()
    dense_quadratic = _generic_dense_jacobian(quadratic, point)
    exact_quadratic = matrix + torch.diag(2 * quad_weight * point)
    quadratic_error = float(torch.linalg.matrix_norm(
        dense_quadratic - exact_quadratic, ord=2))
    planted_modulus = float(2 * torch.max(torch.abs(quad_weight)))

    joint_theta = torch.tensor(
        [0.7, -0.2], dtype=dtype, device=device)
    joint_row = torch.tensor(
        [0.3, -0.4, 0.5], dtype=dtype, device=device)

    def joint_map(theta_value, row_value):
        return (
            theta_value[0] * row_value ** 3
            + theta_value[1] * row_value ** 2
        )

    mixed = torch.func.jacfwd(
        torch.func.jacfwd(joint_map, argnums=1), argnums=0
    )(joint_theta, joint_row)
    third = torch.func.jacfwd(
        torch.func.jacfwd(
            torch.func.jacfwd(joint_map, argnums=1), argnums=1
        ),
        argnums=1,
    )(joint_theta, joint_row)
    exact_mixed = torch.zeros_like(mixed)
    exact_third = torch.zeros_like(third)
    for index in range(joint_row.numel()):
        exact_mixed[index, index, 0] = 3.0 * joint_row[index] ** 2
        exact_mixed[index, index, 1] = 2.0 * joint_row[index]
        exact_third[index, index, index, index] = 6.0 * joint_theta[0]
    mixed_derivative_error = float(torch.max(torch.abs(
        mixed - exact_mixed)))
    third_derivative_error = float(torch.max(torch.abs(
        third - exact_third)))

    from pldr_model_v510 import PLDR_Model
    model = PLDR_Model(
        num_layers=1, d_model=8, num_heads=2, dff=16,
        input_vocab_size=32, A_dff=8, num_reslayerA=1,
        num_denseA=1, max_seq_len=16, device=device,
    ).to(dtype=dtype, device=device)
    row = torch.randn(width, dtype=dtype, device=device)
    dense_pldr = dense_row_jacobian(model, 0, row)
    basis_pldr = basis_row_jacobian(model, 0, row)
    function = lambda value: _row_map_value(model, 0, value)
    fd_coarse = finite_difference_jacobian(function, row, 2e-5)
    fd_fine = finite_difference_jacobian(function, row, 1e-5)
    pldr_basis_error = float(torch.linalg.matrix_norm(
        dense_pldr - basis_pldr, ord=2))
    pldr_fd_error = float(torch.linalg.matrix_norm(
        dense_pldr - fd_fine, ord=2))
    pldr_fd_change = float(torch.linalg.matrix_norm(
        fd_fine - fd_coarse, ord=2))

    tolerance = 2e-8
    passed = all(value <= tolerance for value in (
        linear_error, basis_error, quadratic_error,
        mixed_derivative_error, third_derivative_error,
        pldr_basis_error,
    )) and pldr_fd_error <= 2e-5 and pldr_fd_change <= 3e-5
    return {
        "status": "PASS" if passed else "FAIL",
        "dtype": str(dtype),
        "device": str(device),
        "linear_dense_error": linear_error,
        "linear_basis_error": basis_error,

        "quadratic_dense_error": quadratic_error,
        "quadratic_modulus_exact": planted_modulus,
        "mixed_parameter_row_derivative_error": mixed_derivative_error,
        "third_row_derivative_error": third_derivative_error,
        "pldr_basis_error": pldr_basis_error,
        "pldr_finite_difference_error": pldr_fd_error,
        "pldr_finite_difference_change": pldr_fd_change,
    }
def _spectral_norm(tensor):
    return float(torch.linalg.matrix_norm(
        tensor.detach().to(dtype=torch.float64, device="cpu"), ord=2))


def _vector_norm(tensor):
    return float(torch.linalg.vector_norm(
        tensor.detach().to(dtype=torch.float64, device="cpu")))


def analytic_row_map_modulus(model, layer_index: int,
                             input_norm_upper: float):
    """Bound the derivative modulus on a Euclidean input ball.

    The bound follows the implemented composition exactly.  For SiLU we use
    the global inequalities |silu(x)| <= |x|, |silu'(x)| <= 1 + 1/e,
    |silu''(x)| <= 1/2 + 1/e, and
    |silu'''(x)| <= 3/4 + 1/e. Each GLU and residual LayerNorm is
    propagated through third order. The returned Hessian bound is a
    Lipschitz constant for the row-map derivative, while the third-order
    bound controls the finite transport of that derivative.
    """
    radius = float(input_norm_upper)
    if not math.isfinite(radius) or radius < 0:
        raise ValueError("input_norm_upper must be nonnegative and finite")
    layers = model.decoder.dec_layers
    if not 0 <= layer_index < len(layers):
        raise IndexError("layer_index is outside the decoder")
    units = layers[layer_index].mha1.reslayerAs
    current_bound = radius
    derivative_bound = 1.0
    hessian_bound = 0.0
    third_bound = 0.0
    unit_rows = []
    silu_first = 1.0 + 1.0 / math.e
    silu_second = 0.5 + 1.0 / math.e
    silu_third = 0.75 + 1.0 / math.e

    for unit_index, unit in enumerate(units):
        inner_bound = current_bound
        inner_derivative = 1.0
        inner_hessian = 0.0
        inner_third = 0.0
        dense_rows = []
        for dense_index, dense in enumerate(unit.denseAs):
            activation_name = getattr(dense.activation, "__name__", "")
            if activation_name not in {"silu", "SiLU"}:
                raise ValueError(
                    "analytic modulus currently requires SiLU activation")
            w1 = _spectral_norm(dense.gluw1.weight)
            w2 = _spectral_norm(dense.gluw2.weight)
            w3 = _spectral_norm(dense.gluw3.weight)
            b1 = _vector_norm(dense.gluw1.bias)
            b2 = _vector_norm(dense.gluw2.bias)
            b3 = _vector_norm(dense.gluw3.bias)
            u_bound = w1 * inner_bound + b1
            v_bound = w2 * inner_bound + b2
            local_derivative = w3 * (
                v_bound * silu_first * w1 + u_bound * w2)
            local_hessian = w3 * (
                v_bound * silu_second * w1 ** 2
                + 2.0 * silu_first * w1 * w2
            )
            local_third = w3 * (
                v_bound * silu_third * w1 ** 3
                + 3.0 * silu_second * w1 ** 2 * w2
            )
            output_bound = w3 * u_bound * v_bound + b3
            next_third = (
                local_third * inner_derivative ** 3
                + 3.0 * local_hessian
                * inner_derivative * inner_hessian
                + local_derivative * inner_third
            )
            next_hessian = (
                local_hessian * inner_derivative ** 2
                + local_derivative * inner_hessian
            )
            next_derivative = local_derivative * inner_derivative
            dense_rows.append({
                "index": dense_index,
                "input_norm_upper": inner_bound,
                "output_norm_upper": output_bound,
                "derivative_upper": local_derivative,
                "hessian_upper": local_hessian,
                "third_derivative_upper": local_third,
            })
            inner_bound = output_bound
            inner_derivative = next_derivative
            inner_hessian = next_hessian
            inner_third = next_third

        residual_bound = current_bound + inner_bound
        residual_derivative = 1.0 + inner_derivative
        residual_hessian = inner_hessian
        residual_third = inner_third
        layernorm = unit.layernormA
        dimension = int(layernorm.normalized_shape[0])
        epsilon = float(layernorm.eps)
        if epsilon <= 0.0:
            raise ValueError("LayerNorm epsilon must be positive")
        if layernorm.elementwise_affine:
            gamma_inf = float(torch.max(torch.abs(
                layernorm.weight.detach())).to(device="cpu"))
            beta_norm = _vector_norm(layernorm.bias)
        else:
            gamma_inf = 1.0
            beta_norm = 0.0
        layernorm_derivative = gamma_inf / math.sqrt(epsilon)
        layernorm_hessian = (
            6.0 * gamma_inf / (math.sqrt(dimension) * epsilon))
        layernorm_third = (
            36.0 * gamma_inf
            / (dimension * epsilon ** 1.5)
        )
        local_derivative = layernorm_derivative * residual_derivative
        local_hessian = (
            layernorm_hessian * residual_derivative ** 2
            + layernorm_derivative * residual_hessian
        )
        local_third = (
            layernorm_third * residual_derivative ** 3
            + 3.0 * layernorm_hessian
            * residual_derivative * residual_hessian
            + layernorm_derivative * residual_third
        )
        output_bound = gamma_inf * math.sqrt(dimension) + beta_norm
        next_third = (
            local_third * derivative_bound ** 3
            + 3.0 * local_hessian * derivative_bound * hessian_bound
            + local_derivative * third_bound
        )
        next_hessian = (
            local_hessian * derivative_bound ** 2
            + local_derivative * hessian_bound
        )
        next_derivative = local_derivative * derivative_bound
        unit_rows.append({
            "index": unit_index,
            "input_norm_upper": current_bound,
            "pre_layernorm_norm_upper": residual_bound,
            "output_norm_upper": output_bound,
            "derivative_upper": local_derivative,
            "hessian_upper": local_hessian,
            "third_derivative_upper": local_third,
            "dense_layers": dense_rows,
        })
        current_bound = output_bound
        derivative_bound = next_derivative
        hessian_bound = next_hessian
        third_bound = next_third

    if not all(math.isfinite(value) for value in (
        current_bound, derivative_bound, hessian_bound, third_bound,
    )):
        raise OverflowError("analytic row-map modulus overflowed")
    certificate = DerivativeModulusCertificate(
        value=hessian_bound,
        domain="DECLARED_COMPACT_TUBE",
        method="analytic_silu_glu_layernorm_composition",
    )
    certificate.validate()
    return {
        "certificate": certificate,
        "input_norm_upper": radius,
        "output_norm_upper": current_bound,
        "derivative_upper": derivative_bound,
        "derivative_modulus": hessian_bound,
        "third_derivative_upper": third_bound,
        "silu_first_derivative_upper": silu_first,
        "silu_second_derivative_upper": silu_second,
        "silu_third_derivative_upper": silu_third,
        "units": unit_rows,
    }
