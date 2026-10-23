"""Math-core validation — §10 tests 1-7, 9, 10.

These are the tests that catch normalization drift (3, 4, 9) and prove the
Gram-trick / window-space shortcuts agree with the literal n-space definitions.
"""

from __future__ import annotations

import torch

from cumobs import (
    CumulantConfig,
    build_intermediates,
    compute_all,
    total_scalar_cumulants,
)
from conftest import center, kappa2, kappa3, kappa4_connected

TOL = 1e-9
EIG_TOL = 1e-6  # asymmetric eig is less precise than eigh


def _cfg(**kw):
    base = dict(window_size=12, top_k=3, order=4, dtype=torch.float64)
    base.update(kw)
    return CumulantConfig(**base)


# --- test 1: Gram vs direct covariance spectrum --------------------------


def test_gram_vs_direct_spectrum():
    torch.manual_seed(1)
    m, n = 10, 30
    cfg = _cfg(window_size=m)
    B = torch.randn(m, n, dtype=torch.float64)
    meas = compute_all(B, cfg)
    spec = meas.vectors["covariance_spectrum"]

    D = m - cfg.ddof
    B_c = center(B)
    C = B_c.t() @ B_c / D
    direct = torch.linalg.eigvalsh(C).clamp(min=0).sort(descending=True).values

    assert spec.shape == (n,)
    assert torch.allclose(spec, direct, atol=TOL)


# --- test 2: contracted spectra (Gram) vs direct -------------------------


def test_contracted_spectra_vs_direct():
    torch.manual_seed(2)
    m, n = 10, 25
    cfg = _cfg(window_size=m, materialize_matrices=True)
    B = torch.randn(m, n, dtype=torch.float64)
    si = build_intermediates(B, cfg)
    meas = compute_all(B, cfg)

    # direct symmetric eig of the materialized matrices
    m3_direct = torch.linalg.eigvalsh(si.M3).sort(descending=True).values
    k_direct = torch.linalg.eigvalsh(si.K).sort(descending=True).values

    skew = meas.vectors["contracted_third_cumulant_spectrum"]
    kurt = meas.vectors["contracted_fourth_cumulant_spectrum"]

    assert torch.allclose(skew, m3_direct, atol=EIG_TOL)
    assert torch.allclose(kurt, k_direct, atol=EIG_TOL)
    assert meas.health["contracted_third_cumulant_imag_ratio"] < 1e-6
    assert meas.health["contracted_fourth_cumulant_imag_ratio"] < 1e-6


# --- test 3: full-contraction identities (brute force) -------------------


def test_full_contraction_identities():
    torch.manual_seed(3)
    m, n = 14, 5
    # the full-contraction identities are about the PLUG-IN (biased) cumulants
    cfg = _cfg(window_size=m, cumulant_estimator="biased")
    B = torch.randn(m, n, dtype=torch.float64)
    D = m - cfg.ddof
    B_c = center(B)

    meas = compute_all(B, cfg)
    k2 = kappa2(B_c, D)
    k3 = kappa3(B_c, D)
    k4 = kappa4_connected(B_c, D)

    assert abs(meas.scalars["total_second_cumulant"] - float(k2.sum())) < TOL
    assert abs(meas.scalars["total_third_cumulant"] - float(k3.sum())) < TOL
    assert abs(meas.scalars["total_fourth_cumulant"] - float(k4.sum())) < 1e-7


# --- test 3b: general higher-order scalar cumulants (§3.8) ---------------


def test_general_scalar_cumulants_recurrence():
    """kappa_p of the aggregate S via the moment-cumulant recurrence matches the
    closed-form central-moment formulas (zero-mean S), and is wired through
    compute_all as total_cumulant_{p} for p >= 5."""
    import math

    torch.manual_seed(21)
    m, n = 16, 9
    # the recurrence reproduces the PLUG-IN totals (kappa_3/4 of S), so pin biased
    cfg = _cfg(window_size=m, max_scalar_cumulant_order=6, cumulant_estimator="biased")
    D = m - cfg.ddof
    si = build_intermediates(B := torch.randn(m, n, dtype=torch.float64), cfg)
    S = si.S

    kap = total_scalar_cumulants(S, D, 6)

    # central moments of the zero-mean aggregate under the single divisor D
    def mu(k):
        return float((S**k).sum() / D)

    m2, m3, m4, m5, m6 = mu(2), mu(3), mu(4), mu(5), mu(6)

    # closed-form cumulants in terms of central moments (zero mean)
    closed = {
        1: 0.0,
        2: m2,
        3: m3,
        4: m4 - 3 * m2**2,
        5: m5 - 10 * m3 * m2,
        6: m6 - 15 * m4 * m2 - 10 * m3**2 + 30 * m2**3,
    }
    for p in range(1, 7):
        assert math.isclose(kap[p], closed[p], rel_tol=1e-9, abs_tol=1e-9)

    # the recurrence reproduces the named totals exactly (kappa_3, kappa_4 of S)
    assert abs(kap[3] - mu(3)) < TOL
    assert abs(kap[2] - mu(2)) < TOL

    # compute_all emits total_cumulant_5 / _6 (and not _3 / _4, which are named)
    meas = compute_all(B, cfg)
    assert "total_cumulant_5" in meas.scalars
    assert "total_cumulant_6" in meas.scalars
    assert "total_cumulant_3" not in meas.scalars
    assert abs(meas.scalars["total_cumulant_5"] - kap[5]) < TOL
    assert abs(meas.scalars["total_cumulant_6"] - kap[6]) < TOL
    assert abs(meas.scalars["total_third_cumulant"] - kap[3]) < TOL
    assert abs(meas.scalars["total_fourth_cumulant"] - kap[4]) < 1e-7

    # default config does NOT add the higher-order keys
    base = compute_all(B, _cfg(window_size=m))
    assert not any(k.startswith("total_cumulant_") for k in base.scalars)


# --- test 4: contracted-matrix identities --------------------------------


def test_contracted_matrix_identities():
    torch.manual_seed(4)
    m, n = 14, 5
    cfg = _cfg(window_size=m, materialize_matrices=True)
    B = torch.randn(m, n, dtype=torch.float64)
    D = m - cfg.ddof
    B_c = center(B)
    si = build_intermediates(B, cfg)

    k3 = kappa3(B_c, D)
    k4 = kappa4_connected(B_c, D)
    m3_brute = k3.sum(dim=2)               # sum_k kappa3[i,j,k]
    k_brute = k4.sum(dim=(2, 3))           # sum_{kl} kappa4[i,j,k,l]

    assert torch.allclose(si.M3, m3_brute, atol=TOL)
    assert torch.allclose(si.K, k_brute, atol=1e-7)


# --- test 5: symmetry ----------------------------------------------------


def test_materialized_matrices_symmetric():
    torch.manual_seed(5)
    cfg = _cfg(materialize_matrices=True)
    B = torch.randn(12, 18, dtype=torch.float64)
    si = build_intermediates(B, cfg)
    for M in (si.C, si.M3, si.K):
        assert torch.allclose(M, M.t(), atol=TOL)


# --- test 6: projection consistency --------------------------------------


def test_projection_consistency():
    torch.manual_seed(6)
    cfg = _cfg(materialize_matrices=True, compute_eigenvectors=True, top_k=4)
    B = torch.randn(12, 20, dtype=torch.float64)
    si = build_intermediates(B, cfg)

    # E[P_a^2] == lambda_a
    p_var = (si.P**2).sum(dim=0) / si.D
    assert torch.allclose(p_var, si.lam_top, atol=TOL)

    # ||V_a|| == 1
    norms = torch.linalg.vector_norm(si.V, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=TOL)

    # C V_a == lambda_a V_a
    lhs = si.C @ si.V
    rhs = si.V * si.lam_top
    assert torch.allclose(lhs, rhs, atol=1e-8)


# --- test 7: PR / IPR bounds ---------------------------------------------


def test_pr_bounds_and_rank1():
    torch.manual_seed(7)
    cfg = _cfg()
    B = torch.randn(12, 20, dtype=torch.float64)
    meas = compute_all(B, cfg)
    pr = meas.scalars["participation_ratio"]
    rank = meas.health["rank"]
    assert 1.0 - 1e-9 <= pr <= rank + 1e-9

    # rank-1 buffer: B[t,i] = a[t] * w[i]  ->  PR ~ 1
    a = torch.randn(12, dtype=torch.float64)
    w = torch.randn(20, dtype=torch.float64)
    B1 = torch.outer(a, w)
    pr1 = compute_all(B1, cfg).scalars["participation_ratio"]
    assert abs(pr1 - 1.0) < 1e-6


def test_ipr_one_hot_and_uniform():
    cfg = _cfg(compute_eigenvectors=True, top_k=1)
    n = 16
    a = torch.randn(12, dtype=torch.float64)

    # common-mode: B[t,:] = a[t] * 1  -> top eigvec uniform -> IPR = 1/n
    B_uniform = torch.outer(a, torch.ones(n, dtype=torch.float64))
    ipr_u = compute_all(B_uniform, cfg).vectors["inverse_participation_ratio"]
    assert abs(float(ipr_u[0]) - 1.0 / n) < 1e-6

    # one-hot: B[t,:] = a[t] * e_3  -> top eigvec = e_3 -> IPR = 1
    e = torch.zeros(n, dtype=torch.float64)
    e[3] = 1.0
    B_onehot = torch.outer(a, e)
    ipr_o = compute_all(B_onehot, cfg).vectors["inverse_participation_ratio"]
    assert abs(float(ipr_o[0]) - 1.0) < 1e-6
    # bounds on random
    iprs = compute_all(torch.randn(12, n, dtype=torch.float64), cfg).vectors[
        "inverse_participation_ratio"
    ]
    assert torch.all(iprs >= 1.0 / n - 1e-9) and torch.all(iprs <= 1.0 + 1e-9)


# --- test 9: ddof invariances --------------------------------------------


def test_ddof_invariances():
    torch.manual_seed(9)
    m, n = 12, 20
    B = torch.randn(m, n, dtype=torch.float64)
    c1 = _cfg(window_size=m, ddof=1)
    c0 = _cfg(window_size=m, ddof=0)

    s1 = compute_all(B, c1)
    s0 = compute_all(B, c0)

    # SHC scales as D^-2
    D1, D0 = m - 1, m - 0
    ratio = s1.scalars["spectral_heat_capacity"] / s0.scalars["spectral_heat_capacity"]
    assert abs(ratio - (D0 / D1) ** 2) < 1e-6

    # PR invariant
    assert abs(s1.scalars["participation_ratio"] - s0.scalars["participation_ratio"]) < 1e-9

    # fidelity / TV between two fixed windows are ddof-invariant
    from cumobs import classical_fidelity, trace_distance

    B2 = torch.randn(m, n, dtype=torch.float64)
    p1 = compute_all(B, c1).vectors["covariance_spectrum"]
    q1 = compute_all(B2, c1).vectors["covariance_spectrum"]
    p0 = compute_all(B, c0).vectors["covariance_spectrum"]
    q0 = compute_all(B2, c0).vectors["covariance_spectrum"]
    assert abs(classical_fidelity(p1, q1) - classical_fidelity(p0, q0)) < 1e-9
    assert abs(trace_distance(p1, q1) - trace_distance(p0, q0)) < 1e-9


# --- test 10: path agreement, top_k clamp, constant buffer ---------------


def test_path_agreement_m_ge_n():
    torch.manual_seed(10)
    m, n = 25, 6
    cfg = _cfg(window_size=m)
    B = torch.randn(m, n, dtype=torch.float64)
    meas = compute_all(B, cfg)             # direct path (m > n)
    assert meas.health["use_gram"] == 0.0

    # gram-on-G0 manual computation must agree on the overlapping eigenvalues
    D = m - cfg.ddof
    B_c = center(B)
    G0 = B_c @ B_c.t()
    gram = (torch.linalg.eigvalsh(G0).clamp(min=0) / D).sort(descending=True).values
    direct = meas.vectors["covariance_spectrum"]
    assert torch.allclose(direct, gram[:n], atol=1e-9)


def test_topk_clamp_to_rank():
    torch.manual_seed(11)
    # rank-3 buffer embedded in n=8 features
    m, n, r = 12, 8, 3
    base = torch.randn(m, r, dtype=torch.float64)
    mix = torch.randn(r, n, dtype=torch.float64)
    B = base @ mix
    cfg = _cfg(window_size=m, top_k=10, compute_eigenvectors=True)
    meas = compute_all(B, cfg)
    assert meas.health["n_principal"] == float(r)
    assert meas.vectors["principal_third_cumulant"].numel() == r
    assert meas.vectors["inverse_participation_ratio"].numel() == r


def test_constant_buffer_no_crash():
    cfg = _cfg(compute_eigenvectors=True, materialize_matrices=True)
    B = torch.full((12, 20), 3.5, dtype=torch.float64)
    meas = compute_all(B, cfg)
    assert meas.health["constant_window"] == 1.0
    # PR is NaN for a dead window (documented); nothing raised
    import math

    assert math.isnan(meas.scalars["participation_ratio"])
    assert abs(meas.scalars["total_third_cumulant"]) < 1e-12
