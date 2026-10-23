"""Tests for the v0.3 extension batch (cumobs_extension_spec):

R1 divisor + rank cap, the standardized cumulant spectrum / normalized_shc /
log_det / Fisher zero (§1, §1c), the factorization-residual helper (§1b), the
geodesic + noise-floored distances (§2, §4), critical-slowing-down autocorrelation
(§5), cross-operator alignment (§7), the surrogate + MP/TW null model (§3), the
BBP bitmask (§6), eigenvector tracking + per-mode CSD (§7 step 2 / §5), and the
§8 consolidation trim.
"""

from __future__ import annotations

import json
import math

import pytest
import torch

from cumobs import (
    CumulantConfig,
    CumulantMonitor,
    InMemoryLogger,
    JSONLLogger,
    LoggingConfig,
    ProbeSet,
    bures_angle,
    build_intermediates,
    classical_fidelity,
    compute_all,
    factorization_residual,
    fisher_rao_distance,
    integrated_autocorr_time,
    leading_fisher_zero,
    moments_from_cumulants,
    quantum_fidelity,
    total_scalar_cumulants,
)
from cumobs.observables import _aggregate_cumulants

torch.set_default_dtype(torch.float64)
TOL = 1e-9


def _cfg(**kw):
    base = dict(window_size=20, top_k=3, order=4, dtype=torch.float64)
    base.update(kw)
    return CumulantConfig(**base)


# --- R1: divisor + rank cap ----------------------------------------------


def test_divisor_accounts_for_detrend_order():
    assert _cfg(window_size=20, ddof=1, detrend_order=0).divisor == 19
    assert _cfg(window_size=20, ddof=1, detrend_order=1).divisor == 18
    assert _cfg(window_size=20, ddof=2, detrend_order=3).divisor == 15


def test_build_intermediates_uses_corrected_divisor():
    torch.manual_seed(0)
    B = torch.randn(20, 8)
    # biased estimator stays consistent under a detrend (no R1 unbiasedness warning)
    si = build_intermediates(B, _cfg(ddof=1, detrend_order=2, cumulant_estimator="biased"))
    assert si.D == 20 - (1 + 2)


def test_rank_capped_by_window_minus_dplus1():
    torch.manual_seed(1)
    # n >> m: rank is capped by m - (detrend_order + 1), never by n (R3).
    B = torch.randn(12, 200)
    si0 = build_intermediates(B, _cfg(window_size=12, detrend_order=0))
    si1 = build_intermediates(
        B, _cfg(window_size=12, detrend_order=1, ddof=2, cumulant_estimator="biased")
    )
    assert si0.rank <= 12 - 1
    assert si1.rank <= 12 - 2


def test_unbiased_detrend_warns():
    torch.manual_seed(2)
    B = torch.randn(20, 6)
    with pytest.warns(RuntimeWarning, match="no longer unbiased"):
        compute_all(B, _cfg(detrend_order=1, ddof=2, cumulant_estimator="unbiased"))


# --- §1: standardized cumulant spectrum / normalized_shc / log_det -------


def test_standardized_spectrum_matches_definition():
    torch.manual_seed(3)
    B = torch.randn(24, 10)
    cfg = _cfg(window_size=24, cumulant_estimator="biased", max_scalar_cumulant_order=6)
    si = build_intermediates(B, cfg)
    kap = _aggregate_cumulants(si, cfg, 6)
    gamma = compute_all(B, cfg).vectors["standardized_cumulant_spectrum"]
    assert gamma.numel() == 4  # γ3..γ6
    for i, p in enumerate(range(3, 7)):
        assert abs(float(gamma[i]) - kap[p] / kap[2] ** (p / 2)) < 1e-9


def test_normalized_shc_equals_rank_over_pr_minus_one():
    torch.manual_seed(4)
    B = torch.randn(20, 12)
    m = compute_all(B, _cfg(order=2))
    pr = m.scalars["participation_ratio"]
    rank = m.health["rank"]
    assert abs(m.scalars["normalized_shc"] - (rank / pr - 1.0)) < 1e-9


def test_log_det_matches_nonzero_eigs():
    torch.manual_seed(5)
    B = torch.randn(20, 9)
    m = compute_all(B, _cfg(order=2))
    spec = m.vectors["covariance_spectrum"]
    rank = int(m.health["rank"])
    expect = float(torch.log(spec[:rank] + 1e-13).sum())
    assert abs(m.scalars["log_det"] - expect) < 1e-9


# --- §1b: factorization residuals ----------------------------------------


def test_factorization_residual_known_cases():
    kap = {1: 0.0, 2: 1.7, 3: 0.9, 4: 2.3, 5: -1.1, 6: 4.0}
    assert abs(factorization_residual(kap, 4, 2) - kap[4]) < TOL
    assert abs(factorization_residual(kap, 4, 3) - kap[4]) < TOL  # 4->3 ≡ 4->2
    assert abs(factorization_residual(kap, 6, 3) - (kap[6] + 15 * kap[4] * kap[2])) < TOL
    assert abs(
        factorization_residual(kap, 6, 2)
        - (kap[6] + 15 * kap[4] * kap[2] + 10 * kap[3] ** 2)
    ) < TOL
    assert abs(factorization_residual(kap, 5, 2) - (kap[5] + 10 * kap[3] * kap[2])) < TOL
    assert abs(factorization_residual(kap, 5, 4) - kap[5]) < TOL  # R(M,M-1)=κ_M


def test_moments_from_cumulants_inverts_recurrence():
    torch.manual_seed(6)
    # moments_from_cumulants assumes a centered variable (κ₁ = 0), as the
    # aggregate S is; feed it the cumulants of the mean-removed series.
    xc = torch.randn(50)
    xc = xc - xc.mean()
    D = 49.0
    kap = total_scalar_cumulants(xc, D, 6)
    m = moments_from_cumulants(kap, 6)
    for p in range(2, 7):
        assert abs(m[p] - float((xc**p).sum() / D)) < 1e-7
    assert m[0] == 1.0 and abs(m[1]) < 1e-12


# --- §1c: Fisher / Lee–Yang zero -----------------------------------------


def test_fisher_zero_order2_gap():
    z = leading_fisher_zero({2: 8.0}, 2, 1e-13)
    assert z is not None
    _, gap = z
    assert abs(gap - math.sqrt(2.0 / 8.0)) < 1e-9


def test_fisher_zero_dead_window_is_none():
    assert leading_fisher_zero({2: 0.0}, 4, 1e-13) is None


def test_fisher_zero_scalars_present():
    torch.manual_seed(7)
    m = compute_all(torch.randn(20, 8), _cfg())
    for k in ("leading_fisher_zero_re", "leading_fisher_zero_im",
              "leading_fisher_zero_abs", "leading_fisher_zero_arg", "fisher_zero_gap"):
        assert k in m.scalars
    # gap ≈ √(2/κ₂) to leading order
    k2 = m.scalars["total_second_cumulant"]
    assert m.scalars["fisher_zero_gap"] == pytest.approx(math.sqrt(2 / k2), rel=0.5)


# --- §2: geodesic distances ----------------------------------------------


def test_fisher_rao_identity_and_relation():
    p = torch.tensor([5.0, 2.0, 1.0])
    assert fisher_rao_distance(p, p) < 1e-9
    # 2·arccos(√fidelity)
    q = torch.tensor([4.0, 3.0, 1.0])
    f = classical_fidelity(p, q)
    assert abs(fisher_rao_distance(p, q) - 2 * math.acos(min(1.0, math.sqrt(f)))) < 1e-9


def test_bures_angle_half_fisher_rao_when_commuting():
    # diagonal (commuting) ρ, σ: bures_angle = ½ fisher_rao on the shared spectra
    spec_p = torch.tensor([0.7, 0.3])
    spec_q = torch.tensor([0.4, 0.6])
    rho, sigma = torch.diag(spec_p), torch.diag(spec_q)
    ba = bures_angle(rho, sigma)
    fr = fisher_rao_distance(spec_p, spec_q)
    assert abs(ba - 0.5 * fr) < 1e-7


def test_fisher_rao_inherits_signed_warning():
    p = torch.tensor([1.0, -0.4, 0.2])
    q = torch.tensor([0.9, -0.3, 0.1])
    with pytest.warns(RuntimeWarning, match="negative mass"):
        fisher_rao_distance(p, q)


# --- §5: critical-slowing-down autocorrelation ---------------------------


def test_integrated_autocorr_time_ar1_vs_white():
    torch.manual_seed(8)
    # AR(1) with phi=0.8 -> strong positive lag-1 autocorr, tau > 1
    phi, n = 0.8, 4000
    x = torch.zeros(n)
    eps = torch.randn(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    tau_ar, rho1 = integrated_autocorr_time(x)
    assert rho1 > 0.6
    assert tau_ar > 3.0
    # white noise -> tau ≈ 1
    tau_w, _ = integrated_autocorr_time(torch.randn(n))
    assert tau_w < 1.5


def test_autocorr_keys_and_effective_window():
    torch.manual_seed(9)
    m = compute_all(torch.randn(20, 8), _cfg(detrend_order=0))
    assert "autocorr_lag1" in m.scalars and "autocorr_time" in m.scalars
    assert "effective_window" in m.health
    assert m.health["effective_window"] <= 20 - 1 + 1e-9  # floored τ ⇒ ≤ m-(d+1)


# --- §7: cross-operator alignment ----------------------------------------


def test_align_scalar_matches_direct_frobenius_cosine():
    torch.manual_seed(10)
    B = torch.randn(14, 9)
    cfg = _cfg(window_size=14, materialize_matrices=True)
    si = build_intermediates(B, cfg)
    m = compute_all(B, cfg)

    def fro_cos(A, Bm):
        num = float((A * Bm).sum())
        return num / (float(torch.linalg.matrix_norm(A)) * float(torch.linalg.matrix_norm(Bm)))

    assert abs(m.scalars["align_cov_skew"] - fro_cos(si.C, si.M3)) < 1e-7
    assert abs(m.scalars["align_cov_kurtosis"] - fro_cos(si.C, si.K)) < 1e-7
    assert -1.0 <= m.scalars["align_cov_skew"] <= 1.0


def test_subspace_alignment_in_range():
    torch.manual_seed(11)
    m = compute_all(torch.randn(16, 12), _cfg(compute_eigenvectors=True))
    for key in ("subspace_align_cov_skew", "subspace_align_cov_kurtosis"):
        v = m.vectors[key]
        assert torch.all(v >= -1e-9) and torch.all(v <= 1.0 + 1e-9)


# --- §3: null model (surrogate + MP/TW) ----------------------------------


def _planted_rank1(seed=0, n=40, m=24, noise=0.3):
    torch.manual_seed(seed)
    a = 6.0 * torch.randn(m, 1)
    v = torch.zeros(1, n)
    v[0, :6] = torch.randn(6)
    return a @ v + noise * torch.randn(m, n)


def test_mp_null_detects_spike():
    B = _planted_rank1()
    m = compute_all(B, _cfg(window_size=24, null_model="mp"))
    assert m.scalars["mp_edge_upper"] > 0.0
    assert m.scalars["n_spikes"] >= 1.0
    assert m.scalars["spike_energy_fraction"] > 0.5
    assert "mp_bulk_ks" in m.health
    # edge_gap = λ₁ − edge is positive when a spike has detached
    spec = m.vectors["covariance_spectrum"]
    assert abs(m.scalars["edge_gap"] - (float(spec[0]) - m.scalars["mp_edge_upper"])) < 1e-9
    assert m.scalars["edge_gap"] > 0.0


def test_leading_nongaussian_order():
    from cumobs import leading_nongaussian_order
    g = torch.tensor([0.05, 0.02, 0.8, -0.1])   # γ3..γ6; γ5 (index 2) is large
    assert leading_nongaussian_order(g, band=0.1) == 5
    assert leading_nongaussian_order(g, band=10.0) is None
    # per-entry band
    band = torch.tensor([1.0, 1.0, 0.5, 1.0])
    assert leading_nongaussian_order(g, band=band) == 5


def test_surrogate_null_zscores():
    B = _planted_rank1(seed=1)
    m = compute_all(B, _cfg(window_size=24, null_model="surrogate", null_samples=40))
    # the planted mode makes the spectrum far from the phase-scrambled null
    assert m.scalars["spectral_heat_capacity/null_z"] > 2.0
    assert "spectral_heat_capacity/null_mean" in m.scalars
    assert "spectral_drift/null_mean" in m.scalars
    assert "spectral_drift/null_sd" in m.scalars


# --- §6: BBP bitmask ------------------------------------------------------


def test_bbp_mask_localizes_planted_group():
    B = _planted_rank1(seed=2)
    cfg = _cfg(window_size=24, compute_eigenvectors=True, null_model="mp", rho_keep=0.9)
    m = compute_all(B, cfg)
    masks = m.aux["active_probe_mask"]
    assert isinstance(masks, list) and len(masks) >= 1
    bits = [i for i in range(40) if (int(masks[0], 16) >> i) & 1]
    # the planted signal lives on probes 0..5
    assert all(b < 6 for b in bits)
    assert "active_probe_mask_any" in m.aux
    assert m.vectors["active_probe_count"].numel() == len(masks)


def test_bbp_absent_without_edge():
    B = _planted_rank1(seed=3)
    m = compute_all(B, _cfg(window_size=24, compute_eigenvectors=True))  # null off
    assert "active_probe_mask" not in m.aux


# --- §4: noise-floored distances (via monitor) ---------------------------


def test_noise_floored_distances_emitted():
    torch.manual_seed(12)
    n = 30
    state = {"loss": torch.zeros(n)}
    probes = ProbeSet(torch.arange(n), lambda mdl, b: state["loss"][b], name="p")
    cfg = _cfg(window_size=10, eval_interval=1, update_interval=4,
               null_model="both", null_samples=20)
    logger = InMemoryLogger()
    mon = CumulantMonitor(None, probes, cfg, logger=logger)
    for step in range(40):
        state["loss"] = torch.randn(n)
        mon.step(step)
    rows = [r for r in logger.history if "fisher_rao_distance" in r]
    assert rows
    r = rows[-1]
    assert "denoised_spectral_drift" in r
    assert "spectral_drift_snr" in r
    assert "signal_energy_drift" in r
    assert 0.0 <= r["denoised_spectral_drift"]


# --- §7 step 2 / §5: tracking + per-mode CSD -----------------------------


def test_mode_tracking_and_csd_keys():
    torch.manual_seed(13)
    n = 18
    v1 = torch.zeros(n); v1[:5] = torch.randn(5)
    v2 = torch.zeros(n); v2[9:14] = torch.randn(5)
    state = {"loss": torch.zeros(n)}
    probes = ProbeSet(torch.arange(n), lambda mdl, b: state["loss"][b], name="p")
    cfg = _cfg(window_size=10, eval_interval=1, update_interval=3, top_k=3,
               compute_eigenvectors=True, track_modes=True)
    logger = InMemoryLogger()
    mon = CumulantMonitor(None, probes, cfg, logger=logger)
    for step in range(60):
        if step < 25:
            state["loss"] = 3.0 * torch.randn(()) * v1 + 0.05 * torch.randn(n)
        else:
            state["loss"] = (2.0 * torch.randn(()) * v1
                             + 5.0 * torch.randn(()) * v2 + 0.05 * torch.randn(n))
        mon.step(step)
    assert any("crossing_detected" in r for r in logger.history)
    # the second mode turning on forces at least one tracked crossing
    assert any(r.get("crossing_detected", 0.0) == 1.0 for r in logger.history)
    assert any("principal_autocorr_lag1/mean" in r for r in logger.history)


def test_track_modes_requires_eigenvectors():
    with pytest.raises(ValueError):
        _cfg(track_modes=True, compute_eigenvectors=False)


# --- §8: consolidation trim ----------------------------------------------


def test_trim_consolidated_drops_and_keeps():
    torch.manual_seed(14)
    m = compute_all(torch.randn(20, 8), _cfg(max_scalar_cumulant_order=6))
    full = m.to_flat_dict()
    lc = LoggingConfig(trim_consolidated=True)
    dk, dp = lc.dropped()
    trimmed = m.to_flat_dict(drop_keys=dk, drop_prefixes=dp)
    assert "von_neumann_entropy" in full and "von_neumann_entropy" not in trimmed
    assert "total_cumulant_5" in full and "total_cumulant_5" not in trimmed
    assert "effective_rank" in trimmed            # kept companion
    assert "participation_ratio" in trimmed


# --- aux serialization ----------------------------------------------------


def test_aux_mask_survives_jsonl(tmp_path):
    B = _planted_rank1(seed=4)
    cfg = _cfg(window_size=24, compute_eigenvectors=True, null_model="mp")
    m = compute_all(B, cfg)
    path = tmp_path / "run.jsonl"
    logger = JSONLLogger(str(path))
    logger.log(m.to_flat_dict(), step=0)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert "active_probe_mask_any" in rows[0]
    assert isinstance(rows[0]["active_probe_mask"], list)


# --- config validation for the new fields --------------------------------


@pytest.mark.parametrize("kw", [
    dict(null_model="nope"),
    dict(null_samples=0),
    dict(rho_keep=0.0),
    dict(rho_keep=1.5),
    dict(principal_cumulant_order=3),
    dict(root_order=1),
])
def test_v3_config_validation(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)
