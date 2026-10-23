"""Tests for the v0.2 additions: estimator choice, rank-SHC, detrending,
spectral gap / entropy / effective rank, CKA + scale-sensitive distance,
jackknife error bars, signed-spectrum guard, and config validation."""

from __future__ import annotations

import math

import pytest
import torch

from cumobs import (
    CumulantConfig,
    build_intermediates,
    cka_distance,
    cka_similarity,
    classical_fidelity,
    compute_all,
    spectral_drift,
)
from cumobs.observables import _fourth_cumulant, _third_cumulant

torch.set_default_dtype(torch.float64)
TOL = 1e-9


def _cfg(**kw):
    base = dict(window_size=20, top_k=3, order=4, dtype=torch.float64)
    base.update(kw)
    return CumulantConfig(**base)


# --- cumulant estimators -------------------------------------------------


def test_unbiased_matches_scipy_kstat():
    stats = pytest.importorskip("scipy.stats")
    torch.manual_seed(0)
    x = torch.randn(30) * 3 + 7
    for n, fn in ((3, _third_cumulant), (4, _fourth_cumulant)):
        ours = float(fn(x, x.numel() - 1, "unbiased"))
        assert abs(ours - float(stats.kstat(x.numpy(), n))) < 1e-9


def test_unbiased_vectorized_per_column():
    stats = pytest.importorskip("scipy.stats")
    torch.manual_seed(1)
    X = torch.randn(40, 5).exp()              # skewed, per-column independent
    k4 = _fourth_cumulant(X, 39, "unbiased")
    for j in range(5):
        assert abs(float(k4[j]) - float(stats.kstat(X[:, j].numpy(), 4))) < 1e-8


def test_biased_reproduces_plugin():
    torch.manual_seed(2)
    x = torch.randn(25)
    D = 24.0
    xc = x - x.mean()
    assert abs(float(_third_cumulant(xc, D, "biased")) - float((xc**3).sum() / D)) < TOL
    b4 = float((xc**4).sum() / D - 3 * ((xc**2).sum() / D) ** 2)
    assert abs(float(_fourth_cumulant(xc, D, "biased")) - b4) < TOL


def test_estimator_keys_match_and_differ():
    torch.manual_seed(3)
    B = torch.randn(20, 6)
    sb = compute_all(B, _cfg(cumulant_estimator="biased")).scalars
    su = compute_all(B, _cfg(cumulant_estimator="unbiased")).scalars
    # same key set, generally different 4th-cumulant value at finite m
    assert "total_fourth_cumulant" in sb and "total_fourth_cumulant" in su
    assert sb["total_fourth_cumulant"] != su["total_fourth_cumulant"]


def test_unbiased_gaussian_is_unbiased_on_average():
    torch.manual_seed(4)
    cfg = _cfg(window_size=24, cumulant_estimator="unbiased")
    acc = 0.0
    R = 4000
    for _ in range(R):
        acc += compute_all(torch.randn(24, 5), cfg).scalars["total_fourth_cumulant"]
    mean = acc / R
    # Gaussian connected 4th cumulant is 0; the k-statistic is unbiased so the
    # MC mean sits near 0 (within a few SE), unlike the biased plug-in.
    assert abs(mean) < 3.0


# --- detrending ----------------------------------------------------------


def test_detrend0_is_mean_centering():
    torch.manual_seed(5)
    B = torch.randn(20, 7)
    si = build_intermediates(B, _cfg(detrend_order=0))
    assert torch.allclose(si.B_c, B - B.mean(dim=0), atol=TOL)


def test_detrend_removes_linear_drift():
    torch.manual_seed(6)
    m, n = 30, 10
    t = torch.linspace(0, 1, m).unsqueeze(1)
    B = t @ torch.randn(1, n) + 1e-3 * torch.randn(m, n)   # shared linear drift
    s0 = compute_all(B, _cfg(window_size=m, order=2, detrend_order=0)).scalars
    s1 = compute_all(B, _cfg(window_size=m, order=2, detrend_order=1, ddof=2)).scalars
    # drift dominates as a single mode at d=0; detrending collapses it
    assert s0["top_eigenvalue_fraction"] > 0.99
    assert s1["top_eigenvalue_fraction"] < 0.5
    assert s1["participation_ratio"] > 3 * s0["participation_ratio"]


# --- rank- vs full-padded SHC --------------------------------------------


def test_rank_shc_invariant_to_dead_probes():
    torch.manual_seed(7)
    m = 30
    core = torch.randn(m, 5)
    base = compute_all(core, _cfg(window_size=m, order=2, shc_mode="rank")).scalars
    padded = torch.cat([core, torch.full((m, 40), 4.2)], dim=1)
    pad_rank = compute_all(padded, _cfg(window_size=m, order=2, shc_mode="rank")).scalars
    pad_full = compute_all(padded, _cfg(window_size=m, order=2, shc_mode="full")).scalars
    # rank-SHC ignores the dead probes; full-SHC does not
    assert abs(base["spectral_heat_capacity"] - pad_rank["spectral_heat_capacity"]) < 1e-9
    assert abs(base["spectral_heat_capacity"] - pad_full["spectral_heat_capacity"]) > 1e-6


# --- spectral gap / entropy / effective rank -----------------------------


def test_spectral_entropy_isotropic_and_rank1():
    torch.manual_seed(8)
    n, m = 8, 4000
    iso = compute_all(torch.randn(m, n), _cfg(window_size=m, order=2)).scalars
    assert abs(iso["von_neumann_entropy"] - math.log(n)) < 0.05
    assert abs(iso["effective_rank"] - n) < 0.3
    assert math.isclose(iso["effective_rank"], math.exp(iso["von_neumann_entropy"]), rel_tol=1e-9)

    a = torch.randn(m, 1)
    v = torch.zeros(1, n); v[0, :4] = 1.0
    rank1 = compute_all(3 * a @ v + 0.02 * torch.randn(m, n),
                        _cfg(window_size=m, order=2)).scalars
    assert rank1["effective_rank"] < 1.2
    assert rank1["top_eigenvalue_fraction"] > 0.99
    assert rank1["spectral_gap"] > 0.0


# --- CKA + scale-sensitive distance --------------------------------------


def test_cka_self_scale_rotation():
    torch.manual_seed(9)
    n, k = 10, 3
    Q, _ = torch.linalg.qr(torch.randn(n, n))
    V1, lam1 = Q[:, :k], torch.tensor([5.0, 2.0, 1.0])
    assert abs(cka_similarity(V1, lam1, V1, lam1) - 1.0) < 1e-9
    assert abs(cka_similarity(V1, lam1, V1, 10 * lam1) - 1.0) < 1e-9   # scale-invariant
    V2 = Q[:, 1:1 + k]                                                  # rotated subspace
    sim = cka_similarity(V1, lam1, V2, lam1)
    assert 0.0 <= sim < 1.0
    assert abs(cka_distance(V1, lam1, V2, lam1) - (1.0 - sim)) < 1e-12


def test_spectral_drift_scale_sensitive():
    p = torch.tensor([5.0, 2.0, 1.0])
    assert spectral_drift(p, p) < 1e-12
    assert spectral_drift(p, 10 * p) > 0.5            # responds to scale
    assert abs(classical_fidelity(p, 10 * p) - 1.0) < 1e-9   # classical is blind


# --- jackknife error bars ------------------------------------------------


def test_jackknife_error_bars():
    torch.manual_seed(10)
    B = torch.randn(20, 6)
    meas = compute_all(B, _cfg(error_bars="jackknife"))
    se_keys = [k for k in meas.scalars if k.endswith("/se")]
    assert se_keys
    assert "total_fourth_cumulant/se" in meas.scalars
    for k in se_keys:
        v = meas.scalars[k]
        assert math.isnan(v) or v >= 0.0
    # off by default
    assert not any(k.endswith("/se") for k in compute_all(B, _cfg()).scalars)


# --- signed-spectrum guard -----------------------------------------------


def test_as_prob_warns_on_signed_spectrum():
    p = torch.tensor([1.0, -0.4, 0.2])
    q = torch.tensor([0.9, -0.3, 0.1])
    with pytest.warns(RuntimeWarning, match="negative mass"):
        classical_fidelity(p, q)


# --- config validation ---------------------------------------------------


@pytest.mark.parametrize("kw", [
    dict(cumulant_estimator="nope"),
    dict(shc_mode="nope"),
    dict(error_bars="nope"),
    dict(detrend_order=25),                       # >= window_size - 1
    dict(compute_cka_distance=True),              # needs compute_eigenvectors
    dict(distance_source="contracted_skew"),      # old name removed
])
def test_config_validation(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


def test_cka_config_ok_with_eigenvectors():
    cfg = _cfg(compute_cka_distance=True, compute_eigenvectors=True)
    assert cfg.compute_cka_distance
