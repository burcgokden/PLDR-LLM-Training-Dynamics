"""Numerical guards for the corrected mathematical statements of the paper:
the flip coefficient and its sign criterion, the projected-seminorm
order-parameter bound, the endpoint two-cycle of the zero-tilt quartic
trap, the DAG Hessian indefiniteness, and the compensation identity."""

import math

import torch


# ---- quartic normal form helpers ------------------------------------------

def quartic_ustar(delta, lam0, gam):
    """Positive root of lam0 u + gam u^3 = delta (Newton)."""
    u = delta / lam0
    for _ in range(200):
        f = lam0 * u + gam * u ** 3 - delta
        u -= f / (lam0 + 3 * gam * u ** 2)
    assert abs(lam0 * u + gam * u ** 3 - delta) < 1e-14
    return u


def flip_c1_closed_form(delta, lam0, gam):
    u = quartic_ustar(delta, lam0, gam)
    lam = lam0 + 3 * gam * u ** 2
    return 36 * gam ** 2 * u ** 2 / lam ** 2 - 2 * gam / lam, u, lam


def flip_c1_numeric(delta, lam0, gam):
    """c1 = (1/4) g''(u*)^2 + (1/6) g'''(u*) at eta_c = 2/lambda*, with
    derivatives of g(u) = u - eta f'(u) by central finite differences."""
    u, lam = flip_c1_closed_form(delta, lam0, gam)[1:]
    eta = 2.0 / lam

    def g(x):
        return x - eta * (-delta + lam0 * x + gam * x ** 3)

    h = 1e-4
    g2 = (g(u + h) - 2 * g(u) + g(u - h)) / h ** 2
    g3 = (g(u + 2 * h) - 2 * g(u + h) + 2 * g(u - h) - g(u - 2 * h)) \
        / (2 * h ** 3)
    return 0.25 * g2 ** 2 + g3 / 6.0


def test_flip_coefficient_closed_form_matches_numeric():
    for delta, lam0, gam in [(0.01, 1.0, 1.0), (2.0, 1.0, 1.0),
                             (0.5, 0.3, 2.0)]:
        c_closed = flip_c1_closed_form(delta, lam0, gam)[0]
        c_num = flip_c1_numeric(delta, lam0, gam)
        assert abs(c_closed - c_num) < 1e-3 * max(1.0, abs(c_closed))


def test_flip_sign_criterion_and_subcritical_regime():
    # criterion: c1 > 0 iff 15 gam u*^2 > lam0
    for delta, lam0, gam in [(0.01, 1.0, 1.0), (2.0, 1.0, 1.0),
                             (0.3, 1.0, 5.0), (0.05, 2.0, 0.1)]:
        c1, u, lam = flip_c1_closed_form(delta, lam0, gam)
        assert (c1 > 0) == (15 * gam * u ** 2 > lam0)
    # the small-tilt quartic flip is SUBCRITICAL (reviewer-verified case)
    c1, _, _ = flip_c1_closed_form(0.01, 1.0, 1.0)
    assert c1 < 0
    # the corrected sign disagrees with the old printed formula there
    u = quartic_ustar(0.01, 1.0, 1.0)
    lam = 1.0 + 3 * u ** 2
    old_printed = 0.5 * (6 * u / lam) ** 2 + 2 / lam
    assert old_printed > 0 > c1


def test_projected_seminorm_bound_and_counterexample():
    torch.manual_seed(0)
    d, k = 6, 3  # fluctuations supported on a k-dim subspace
    Q, _ = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64))
    V = Q[:, :k]                        # visited span
    P = V @ V.T
    xi = (V @ torch.randn(k, 500, dtype=torch.float64))  # samples in V
    Sigma = xi @ xi.T / 500
    sig2 = torch.linalg.eigvalsh(V.T @ Sigma @ V).min().item()
    assert sig2 > 0
    for seed in range(5):
        g = torch.Generator().manual_seed(seed)
        J = torch.randn(d, d, generator=g, dtype=torch.float64)
        lhs = torch.trace(J @ Sigma @ J.T).item()
        proj = torch.linalg.matrix_norm(J @ P, "fro").item() ** 2
        full = torch.linalg.matrix_norm(J, "fro").item() ** 2
        assert lhs >= sig2 * proj - 1e-9          # corrected bound holds
    # counterexample to the FULL-norm bound: J annihilates the span
    W = Q[:, k:]
    J0 = torch.randn(d, d - k, dtype=torch.float64) @ W.T  # J0 P = 0
    lhs0 = torch.trace(J0 @ Sigma @ J0.T).item()
    assert abs(lhs0) < 1e-9
    assert torch.linalg.matrix_norm(J0, "fro").item() > 0.1


def test_zero_tilt_trap_strict_contraction_and_endpoint_two_cycle():
    lam0, gam, r = 1.0, 1.0, 1.5

    def g(u, eta):
        return u - eta * (lam0 * u + gam * u ** 3)

    # endpoint eta s(r) = 2 with s(u) = lam0 + gam u^2 (one-step curvature
    # along [0, u]): exact two-cycle r <-> -r
    eta_c = 2.0 / (lam0 + gam * r ** 2)
    assert abs(g(r, eta_c) + r) < 1e-12
    assert abs(g(-r, eta_c) - r) < 1e-12
    # strict inequality: strict decrease of |u| everywhere in (0, r]
    eta = 0.95 * eta_c
    u = r
    for _ in range(200):
        un = g(u, eta)
        assert abs(un) < abs(u) or u == 0.0
        u = un
    assert abs(u) < 1e-3


def test_dag_hessian_indefinite():
    """D(M) = log tr exp(M o M) has a negative Hessian eigenvalue at the
    off-diagonal family M(b, c), refuting the PSD claim."""

    def D(b, c):
        M = torch.tensor([[0.0, b], [c, 0.0]], dtype=torch.float64)
        return torch.log(torch.trace(torch.linalg.matrix_exp(M * M)))

    q = 0.5
    h = 1e-4
    s = 1.0 / math.sqrt(2.0)
    # second derivative along the unit direction (1, -1)/sqrt(2)
    d2 = (D(q + s * h, q - s * h) - 2 * D(q, q)
          + D(q - s * h, q + s * h)) / h ** 2
    assert abs(d2.item() + math.tanh(q ** 2)) < 1e-5
    assert d2.item() < 0


def test_compensation_identity():
    """b_a' = b_a + a [ (iswiglu(W Abar + b)+eps)^P - (iswiglu(W Abar' + b)
    + eps)^P ] makes the head map equal at Abar' and Abar (exact algebra of
    the compensation lemma), for the actual iSwiGLU nonlinearity."""
    torch.manual_seed(1)

    def iswiglu(x):
        # reference iSwiGLU: x * silu(x) = x^2 sigmoid(x) >= 0
        return x * torch.nn.functional.silu(x)

    def head(Abar, W, b, P, a, ba, eps=1e-9):
        Ap = (iswiglu(W @ Abar + b) + eps) ** P
        return a @ Ap + ba

    d = 4
    W, b, a, ba = (torch.randn(d, d, dtype=torch.float64) for _ in range(4))
    P = torch.rand(d, d, dtype=torch.float64) * 2 - 1
    A1 = torch.randn(d, d, dtype=torch.float64)
    A2 = torch.randn(d, d, dtype=torch.float64)
    eps = 1e-9
    Ap1 = (iswiglu(W @ A1 + b) + eps) ** P
    Ap2 = (iswiglu(W @ A2 + b) + eps) ** P
    ba2 = ba + a @ (Ap1 - Ap2)
    out1 = head(A1, W, b, P, a, ba)
    out2 = head(A2, W, b, P, a, ba2)
    assert torch.allclose(out1, out2, atol=1e-12)
