"""Lanczos core against known spectra, including the near-symmetric case
where plain power iteration fails, the preconditioned operator, and the
residual-based early stop."""

import torch

import instrument


def dense_matvec(H, shapes):
    """Wrap a dense symmetric matrix as a list-of-tensors matvec, splitting
    the flat vector across several template tensors (exercises the
    list-handling path)."""

    def mv(v):
        flat = torch.cat([t.reshape(-1) for t in v])
        out = H @ flat
        res, i = [], 0
        for s in shapes:
            n = int(torch.tensor(s).prod())
            res.append(out[i : i + n].reshape(s))
            i += n
        return res

    return mv


def make_symmetric(evals, seed=0):
    n = len(evals)
    g = torch.Generator().manual_seed(seed)
    Q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
    return Q @ torch.diag(torch.tensor(evals, dtype=torch.float64)) @ Q.T


def template(shapes):
    return [torch.zeros(s, dtype=torch.float64) for s in shapes]


def test_extremal_eigenvalues_near_symmetric_spectrum():
    evals = [-10.0, -9.98] + [0.1 * k for k in range(20)] + [9.99, 10.02]
    H = make_symmetric(evals, seed=1)
    shapes = [(6,), (3, 4), (2, 3)]  # 6 + 12 + 6 = 24 = len(evals)
    r = instrument.lanczos_extremal(dense_matvec(H, shapes),
                                    template(shapes), n_iter=24, tol=0.0)
    assert abs(r["lam_max"] - 10.02) < 1e-6
    assert abs(r["lam_min"] + 10.0) < 1e-6


def test_residual_bound_and_early_stop():
    # gap-heavy spectrum: extremal Ritz pairs converge long before the
    # Krylov space exhausts the dimension
    evals = [200.0, -150.0] + [0.01 * k for k in range(38)]
    H = make_symmetric(evals, seed=2)
    shapes = [(40,)]
    r = instrument.lanczos_extremal(dense_matvec(H, shapes),
                                    template(shapes), n_iter=40, tol=1e-6)
    # early stop triggered well before full dimension
    assert r["iters"] < 40
    assert abs(r["lam_max"] - 200.0) < 1e-3
    assert abs(r["lam_min"] + 150.0) < 1e-3
    # residual bound is honest relative to the spectral scale
    assert r["resid_max"] < 1e-4 * 200.0


def test_preconditioned_operator_matches_direct_eig():
    evals = [1.0, 2.0, 5.0, -3.0, 0.5, 4.0]
    H = make_symmetric(evals, seed=3)
    d = torch.tensor([0.5, 1.0, 2.0, 4.0, 0.25, 1.5], dtype=torch.float64)
    Dinvs = torch.diag(1.0 / d.sqrt())
    Hp = Dinvs @ H @ Dinvs
    true = torch.linalg.eigvalsh(Hp)
    shapes = [(6,)]

    def mv(v):
        flat = v[0]
        pre = flat / d.sqrt()
        out = (H @ pre) / d.sqrt()
        return [out]

    r = instrument.lanczos_extremal(mv, template(shapes), n_iter=6, tol=0.0)
    assert abs(r["lam_max"] - true[-1].item()) < 1e-8
    assert abs(r["lam_min"] - true[0].item()) < 1e-8


def test_invariant_subspace_deflation_is_handled():
    # start vector confined to an invariant subspace: beta -> 0 break path
    H = torch.diag(torch.tensor([3.0, 3.0, 3.0], dtype=torch.float64))
    shapes = [(3,)]
    r = instrument.lanczos_extremal(dense_matvec(H, shapes),
                                    template(shapes), n_iter=10, tol=0.0)
    assert abs(r["lam_max"] - 3.0) < 1e-9
    assert r["resid_max"] < 1e-9
