"""The pure-tensor observable core (§3-§4).

Every routine here is a free function of a buffer ``B in R^{m x n}`` and a
:class:`~cumobs.config.CumulantConfig`.  No model, no IO.  The whole scalar +
spectral catalog is computable in *window space* (``m x m``) without ever
allocating an ``n x n`` matrix; the ``n x n`` objects (``C``, ``M3``, ``K``) are
opt-in behind ``cfg.materialize_matrices``.

Build order follows the reuse contract of §4 exactly: :func:`build_intermediates`
runs once per measurement and produces :class:`SharedIntermediates`; the section
functions (:func:`scalar_cumulants`, :func:`spectral`, :func:`contracted`,
:func:`per_group`) read from it.  :func:`compute_all` assembles a
:class:`~cumobs.types.Measurement`.

Two estimator families are supported for the 3rd/4th cumulants
(``cfg.cumulant_estimator``): the plug-in central-moment formulas (``"biased"``)
and the Fisher k-statistics (``"unbiased"``, the default — a Gaussian window then
reads ≈ 0 instead of the ``O(1/m)`` plug-in baseline).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import replace
from math import comb

import torch
from torch import Tensor

from .config import CumulantConfig
from .types import Measurement, SharedIntermediates

# ---------------------------------------------------------------------------
# small numerical helpers
# ---------------------------------------------------------------------------


def _pad_sort_desc(vec: Tensor, n: int) -> Tensor:
    """Pad ``vec`` with zeros (or truncate) to length ``n``, sorted descending.

    For a non-negative spectrum the appended zeros land at the tail; for a
    signed (contracted) spectrum re-sorting places them between positive and
    negative eigenvalues, which is the correct full spectrum of the ``n x n``
    operator whose rank deficiency those zeros represent.
    """
    r = vec.numel()
    if r < n:
        pad = torch.zeros(n - r, dtype=vec.dtype, device=vec.device)
        vec = torch.cat([vec, pad])
    elif r > n:
        # vec is assumed already descending; keep the n largest-by-position.
        vec = vec[:n]
    return torch.sort(vec, descending=True).values


def _detrend(B: Tensor, order: int) -> Tensor:
    """Remove the per-column least-squares polynomial of degree ``order`` in time.

    ``order == 0`` is plain per-probe mean-centering (``B - mean``), the original
    behaviour.  ``order >= 1`` strips a smooth within-window trajectory drift so
    it does not masquerade as a covariance mode (the dynamical-2RDM caveat: rows
    are *consecutive* training steps, not i.i.d. samples).
    """
    if order == 0:
        return B - B.mean(dim=0)
    m = B.shape[0]
    t = torch.linspace(-1.0, 1.0, m, dtype=B.dtype, device=B.device)
    design = torch.stack([t**p for p in range(order + 1)], dim=1)  # (m, order+1)
    coef = torch.linalg.lstsq(design, B).solution                  # (order+1, n)
    return B - design @ coef


def _column_signs(M: Tensor) -> Tensor:
    """Per-column sign so the largest-magnitude entry is positive (D6).

    Zero columns map to ``+1``.
    """
    k = M.shape[1] if M.dim() == 2 else 0
    if k == 0:
        return torch.ones(0, dtype=M.dtype, device=M.device)
    idx = M.abs().argmax(dim=0)
    cols = torch.arange(k, device=M.device)
    vals = M[idx, cols]
    signs = torch.sign(vals)
    return torch.where(signs == 0, torch.ones_like(signs), signs)


# ---------------------------------------------------------------------------
# cumulant estimators (biased plug-in vs unbiased k-statistics)
# ---------------------------------------------------------------------------


def _third_cumulant(X: Tensor, D: float, estimator: str) -> Tensor:
    """Third cumulant along dim 0 of ``X`` (``(m,)`` or ``(m, k)``).

    ``"biased"``: plug-in ``Ê[X³] = Σ X³ / D``.
    ``"unbiased"``: Fisher k-statistic ``k₃ = m²/((m-1)(m-2)) · cm₃``.
    """
    if estimator == "biased":
        return (X**3).sum(dim=0) / D
    m = X.shape[0]
    if m < 3:
        return torch.full(X.shape[1:], float("nan"), dtype=X.dtype, device=X.device)
    Xc = X - X.mean(dim=0)
    cm3 = (Xc**3).mean(dim=0)
    return (m * m) / ((m - 1) * (m - 2)) * cm3


def _fourth_cumulant(X: Tensor, D: float, estimator: str) -> Tensor:
    """Connected 4th cumulant along dim 0 of ``X`` (``(m,)`` or ``(m, k)``).

    ``"biased"``: plug-in ``Ê[X⁴] − 3·Ê[X²]²`` (single divisor ``D``).
    ``"unbiased"``: k-statistic ``k₄ = m²/((m-1)(m-2)(m-3))·[(m+1)cm₄ − 3(m-1)cm₂²]``.
    """
    if estimator == "biased":
        c2 = (X**2).sum(dim=0) / D
        return (X**4).sum(dim=0) / D - 3.0 * c2**2
    m = X.shape[0]
    if m < 4:
        return torch.full(X.shape[1:], float("nan"), dtype=X.dtype, device=X.device)
    Xc = X - X.mean(dim=0)
    cm2 = (Xc**2).mean(dim=0)
    cm4 = (Xc**4).mean(dim=0)
    denom = (m - 1) * (m - 2) * (m - 3)
    return (m * m) / denom * ((m + 1) * cm4 - 3.0 * (m - 1) * cm2**2)


# ---------------------------------------------------------------------------
# spectral-variance helper (rank- vs full-padded SHC)
# ---------------------------------------------------------------------------


def _spectral_variance(lam: Tensor, rank: int, mode: str) -> float:
    """Population variance of a (descending) spectrum.

    ``"rank"``: over the ``rank`` leading (nonzero) eigenvalues — invariant to the
    number of probes / structural zeros.  ``"full"``: over the whole padded
    spectrum (legacy; ``∝ 1/n``).
    """
    if mode == "rank":
        if rank < 1:
            return float("nan")
        lz = lam[:rank]
        return float((lz * lz).mean() - lz.mean() ** 2)
    return float((lam * lam).mean() - lam.mean() ** 2)


def _shc_pr(B_sub: Tensor, D: float, cfg: CumulantConfig) -> tuple[float, float]:
    """Spectral heat capacity and participation ratio of a centered sub-buffer.

    Routes through the smaller of the Gram (``m x m``) and direct (``n_sub x
    n_sub``) matrices.  SHC honours ``cfg.shc_mode``; PR is ``(sum lam)^2 / sum
    lam^2`` (invariant to the structural zeros either way).
    """
    eps = cfg.eps
    m, n_sub = B_sub.shape
    if m <= n_sub:
        G = B_sub @ B_sub.transpose(0, 1)
        mu = torch.clamp(torch.linalg.eigvalsh(G), min=0.0)
        lam = _pad_sort_desc(mu / D, n_sub)
    else:
        C = B_sub.transpose(0, 1) @ B_sub / D
        lam = torch.sort(
            torch.clamp(torch.linalg.eigvalsh(C), min=0.0), descending=True
        ).values
    lam_max = float(lam[0]) if lam.numel() else 0.0
    struct_rank = max(0, min(m - (cfg.detrend_order + 1), n_sub))
    rank = min(int((lam > cfg.eig_threshold * lam_max).sum()), struct_rank) if lam_max > 0 else 0
    shc = _spectral_variance(lam, rank, cfg.shc_mode)
    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    pr = (s1 * s1 / s2) if s2 > eps else float("nan")
    return shc, pr


# ---------------------------------------------------------------------------
# build_intermediates  (§4 reuse contract)
# ---------------------------------------------------------------------------


def build_intermediates(
    B: Tensor,
    cfg: CumulantConfig,
    contraction: Tensor | None = None,
) -> SharedIntermediates:
    """Run the shared linear algebra once for one measurement (§4).

    Parameters
    ----------
    B:
        Buffer ``(m, n)`` (window rows x features).
    cfg:
        Pipeline configuration.
    contraction:
        Optional ``(n,)`` contraction direction for the aggregate variable
        ``S = B_c @ contraction`` (§9.1 seam).  Defaults to all-ones (sum over
        all features), which is the loss-semantics "total" direction.
    """
    B = B.to(cfg.dtype)
    m, n = B.shape
    # R1: detrending of degree d is a rank-(d+1) projection, so the single
    # unbiased divisor is D = m - (ddof + detrend_order) (reduces to m - ddof at
    # detrend_order = 0).  One global D keeps the full-contraction identities,
    # which are algebraic in (B_c, D), intact.
    D = float(m - (cfg.ddof + cfg.detrend_order))
    if D <= 0:
        raise ValueError(
            f"D = m - (ddof + detrend_order) must be > 0 "
            f"(m={m}, ddof={cfg.ddof}, detrend_order={cfg.detrend_order}). "
            "Is the window full?"
        )
    # R1: the k-statistic prefactors are derived for the centering projector
    # only; the detrend hat matrix breaks them, so orders >= 3 are no longer
    # unbiased under a polynomial detrend.  Warn (rate-limited by the warnings
    # module) so the user either fixes ddof or switches to the plug-in estimator.
    if cfg.detrend_order > 0 and cfg.order >= 3 and cfg.cumulant_estimator == "unbiased":
        warnings.warn(
            "cumulant_estimator='unbiased' with detrend_order>0: the k-statistic "
            "unbiasedness holds only for the centering projector, so orders >= 3 "
            "are no longer unbiased under the polynomial detrend. Prefer "
            "cumulant_estimator='biased' (the single-D plug-in stays consistent "
            "under any projector) or treat the 3rd/4th cumulants as relative.",
            RuntimeWarning,
            stacklevel=2,
        )
    # R1/R3: rank(C) <= min(m - (detrend_order + 1), n) regardless of n.
    struct_rank = max(0, min(m - (cfg.detrend_order + 1), n))

    # 1-3: centering / detrending and the aggregate variable ----------------
    E_L = B.mean(dim=0)                       # (n,)  reported mean (plain 1/m)
    B_c = _detrend(B, cfg.detrend_order)      # (m,n) degree-0 == B - E_L
    if contraction is None:
        S = B_c.sum(dim=1)                    # (m,)
    else:
        S = B_c @ contraction.to(B_c)
    S2 = S * S
    S3 = S2 * S
    S4 = S2 * S2
    s2 = float(S2.sum())

    use_gram = m <= n
    V: Tensor | None = None
    C_mat: Tensor | None = None

    if use_gram:
        # 4-7: Gram trick -------------------------------------------------
        G0 = B_c @ B_c.transpose(0, 1)        # (m,m)  the one O(m^2 n) product
        mu, U = torch.linalg.eigh(G0)         # ascending
        mu = torch.clamp(mu, min=0.0)
        order_idx = torch.argsort(mu, descending=True)
        mu = mu[order_idx]
        U = U[:, order_idx]
        lam = mu / D                          # (m,)
        eigvals = _pad_sort_desc(lam, n)      # (n,)
        mu_max = float(mu[0]) if mu.numel() else 0.0
        thr = cfg.eig_threshold * mu_max
        rank = min(int((mu > thr).sum()), struct_rank)
        top_k_eff = max(0, min(cfg.top_k, rank))

        mu_top = mu[:top_k_eff]
        U_top = U[:, :top_k_eff]
        sqrt_mu = torch.sqrt(mu_top)
        P = U_top * sqrt_mu                    # (m,k)
        lam_top = mu_top / D
        if cfg.compute_eigenvectors:
            denom = torch.clamp(sqrt_mu, min=cfg.eps)
            V = (B_c.transpose(0, 1) @ U_top) / denom  # (n,k)
    else:
        # m > n direct path: form C (bounded, n <= m), eigh directly -------
        G0 = None
        C_mat = B_c.transpose(0, 1) @ B_c / D  # (n,n)
        lamn, W = torch.linalg.eigh(C_mat)
        lamn = torch.clamp(lamn, min=0.0)
        order_idx = torch.argsort(lamn, descending=True)
        lamn = lamn[order_idx]
        W = W[:, order_idx]
        eigvals = lamn                         # (n,)
        lam_max = float(lamn[0]) if lamn.numel() else 0.0
        thr = cfg.eig_threshold * lam_max
        rank = min(int((lamn > thr).sum()), struct_rank)
        top_k_eff = max(0, min(cfg.top_k, rank))

        W_top = W[:, :top_k_eff]
        lam_top = lamn[:top_k_eff]
        P = B_c @ W_top                        # (m,k)  principal projections
        if cfg.compute_eigenvectors:
            V = W_top

    # sign gauge (D6): make max-abs entry of the exported eigenvector +ve.
    # When V is unavailable we gauge by P so the odd-power principal cumulants
    # (e.g. principal_third_cumulant) are still deterministic across steps.
    if cfg.sign_gauge and top_k_eff > 0:
        basis = V if V is not None else P
        signs = _column_signs(basis)
        P = P * signs
        if V is not None:
            V = V * signs

    # kurtosis contraction operator Q (§3.5); cheap m x m, always available.
    I_m = torch.eye(m, dtype=B.dtype, device=B.device)
    Q = torch.diag(S2 / D) - (s2 / D**2) * I_m - (2.0 / D**2) * torch.outer(S, S)

    # 12: opt-in n x n materialization ----------------------------------------
    M3 = K = None
    if cfg.materialize_matrices:
        if C_mat is None:
            C_mat = B_c.transpose(0, 1) @ B_c / D
        M3 = B_c.transpose(0, 1) @ ((S / D).unsqueeze(1) * B_c)
        K = B_c.transpose(0, 1) @ (Q @ B_c)
    elif not use_gram:
        # in the direct path C was needed for the eig but is not "materialized"
        # output unless requested.
        C_mat = None

    total_var = float((B_c * B_c).sum())
    health = {
        "rank": float(rank),
        "n_principal": float(top_k_eff),
        "n_nonzero": float(rank),
        "use_gram": float(use_gram),
        "constant_window": float(total_var <= cfg.eps),
    }

    return SharedIntermediates(
        m=m,
        n=n,
        D=D,
        use_gram=use_gram,
        rank=rank,
        top_k_eff=top_k_eff,
        E_L=E_L,
        B_c=B_c,
        S=S,
        S2=S2,
        S3=S3,
        S4=S4,
        s2=s2,
        eigvals=eigvals,
        lam_top=lam_top,
        P=P,
        V=V,
        Q=Q,
        G0=G0,
        C=C_mat if cfg.materialize_matrices else None,
        M3=M3,
        K=K,
        health=health,
    )


# ---------------------------------------------------------------------------
# general higher-order scalar cumulant of the aggregate (§3.8)
# ---------------------------------------------------------------------------


def total_scalar_cumulants(
    series: Tensor, D: float, max_order: int
) -> dict[int, float]:
    """General scalar cumulants ``kappa_1 .. kappa_max_order`` of a 1-D series.

    Uses the moment-cumulant recurrence (§3.8)::

        kappa_n = m_n - sum_{k=1}^{n-1} C(n-1, k-1) * kappa_k * m_{n-k}

    where ``m_k = Ê[series**k] = (1/D) sum_t series_t**k`` is the ``k``-th raw
    moment under the package's single divisor ``D`` (decision D1).

    This is the **plug-in** ladder (the same trick behind the biased totals).  It
    is used for orders >= 5 in *both* estimator modes — no closed-form
    k-statistic ships beyond order 4 (mirroring ``scipy.stats.kstat``).  High
    orders are numerically delicate (catastrophic cancellation in the recurrence),
    so ``float64`` is recommended.

    Returns a ``{p: kappa_p}`` dict for ``p = 1 .. max_order``.
    """
    if max_order < 1:
        return {}
    # raw moments m_0 .. m_max_order  (m_0 = 1)
    moments = [1.0]
    power = torch.ones_like(series)
    for _ in range(max_order):
        power = power * series
        moments.append(float(power.sum() / D))
    # recurrence: kappa_n = m_n - sum_{k=1}^{n-1} C(n-1,k-1) kappa_k m_{n-k}
    kappa: list[float] = [0.0]  # kappa_0 slot (unused)
    for n in range(1, max_order + 1):
        val = moments[n]
        for k in range(1, n):
            val -= comb(n - 1, k - 1) * kappa[k] * moments[n - k]
        kappa.append(val)
    return {p: kappa[p] for p in range(1, max_order + 1)}


# ---------------------------------------------------------------------------
# §1 standardized cumulant spectrum / §1b residuals / §1c Fisher zero
# ---------------------------------------------------------------------------


def moments_from_cumulants(kappa: dict[int, float], max_order: int) -> list[float]:
    """Raw moments ``m_0 .. m_max_order`` of a **centered** variable from its
    cumulants (the inverse of :func:`total_scalar_cumulants`'s recurrence).

    ``m_p = Σ_{k=1}^{p} C(p−1, k−1) κ_k m_{p−k}`` with ``m_0 = 1`` and ``κ_1 = 0``
    (centering), so ``m_1 = 0``, ``m_2 = κ_2``, ``m_3 = κ_3``,
    ``m_4 = κ_4 + 3κ_2²``, ….  Used by the Fisher-zero partition function (§1c).
    """
    kap = dict(kappa)
    kap[1] = 0.0
    m = [1.0]
    for p in range(1, max_order + 1):
        val = 0.0
        for k in range(1, p + 1):
            val += comb(p - 1, k - 1) * kap.get(k, 0.0) * m[p - k]
        m.append(val)
    return m


def _integer_partitions(n: int, max_part: int | None = None):
    """Yield each integer partition of ``n`` once, as a descending-size tuple."""
    if max_part is None:
        max_part = n
    if n == 0:
        yield ()
        return
    for first in range(min(n, max_part), 0, -1):
        for rest in _integer_partitions(n - first, first):
            yield (first,) + rest


def factorization_residual(cumulants: dict[int, float], M: int, N: int) -> float:
    """§1b: residual of the ``M``-point function after the most generous
    reconstruction into connected blocks of size ``≤ N``.

    ``R(M,N) = Σ_{set partitions of {1..M} with some block > N} Π_B κ_{|B|}``,
    a fixed combinatorial function of the cumulants already stored.  For a
    centered variable singleton blocks carry ``κ_1 = 0``, so any partition with a
    singleton vanishes; ``R(M, M−1) = κ_M`` is the purest case.  Examples (matching
    the spec): ``R(4,2) = κ₄``, ``R(6,3) = κ₆ + 15κ₄κ₂``,
    ``R(6,2) = κ₆ + 15κ₄κ₂ + 10κ₃²``, ``R(5,2) = κ₅ + 10κ₃κ₂``.

    ``cumulants`` is a ``{order: κ_order}`` mapping (missing orders treated as 0;
    ``κ_1`` defaults to 0).
    """
    if M < 1 or N < 1:
        raise ValueError(f"factorization_residual needs M, N >= 1; got M={M}, N={N}")
    kap = {int(k): float(v) for k, v in cumulants.items()}
    kap.setdefault(1, 0.0)
    total = 0.0
    for parts in _integer_partitions(M):
        if not any(p > N for p in parts):
            continue
        if 1 in parts:                              # singleton block ⇒ κ_1 = 0
            continue
        counts: dict[int, int] = {}
        for p in parts:
            counts[p] = counts.get(p, 0) + 1
        # number of set partitions of {1..M} with this block-size profile
        denom = 1
        for size, mult in counts.items():
            denom *= (math.factorial(size) ** mult) * math.factorial(mult)
        n_setpart = math.factorial(M) // denom
        prod = 1.0
        for p in parts:
            prod *= kap.get(p, 0.0)
        total += n_setpart * prod
    return total


def _poly_roots(coeffs_low_to_high: list[float], eps: float) -> Tensor:
    """Complex roots of ``Σ_p coeffs[p] x^p`` via a companion-matrix ``eigvals``.

    Pure-torch (no numpy) so the core stays torch-only.  Trailing ~0 leading
    coefficients are stripped to recover the true degree.
    """
    c = list(coeffs_low_to_high)
    while len(c) > 1 and abs(c[-1]) <= eps:
        c.pop()
    R = len(c) - 1
    if R < 1:
        return torch.zeros(0, dtype=torch.complex128)
    a = [c[i] / c[-1] for i in range(R)]            # monic lower coefficients
    comp = torch.zeros(R, R, dtype=torch.float64)
    if R > 1:
        comp[1:, :-1] = torch.eye(R - 1, dtype=torch.float64)
    for i in range(R):
        comp[i, -1] = -a[i]
    return torch.linalg.eigvals(comp)


def _real_axis_distance(z: complex) -> float:
    """Distance of ``z`` to the **positive** real axis: ``|Im z|`` if ``Re z ≥ 0``
    else ``|z|`` (§1c leading-root selection)."""
    if z.real >= 0.0:
        return abs(z.imag)
    return abs(z)


def leading_fisher_zero(
    kappa: dict[int, float], root_order: int, eps: float
) -> tuple[complex, float] | None:
    """§1c: leading Fisher / Lee–Yang zero of the truncated partition function.

    ``Z_P(t) = Σ_{p=0}^{root_order} m_p t^p / p!`` (``m_p`` reconstructed from the
    cumulants, ``m_0 = 1``, ``m_1 = 0``).  Rooted in the standardized variable
    ``τ = t·√κ₂`` for conditioning, then ``t = τ/√κ₂`` is recovered.  Returns the
    single leading complex ``t`` (the root nearest the positive real axis, taking
    the upper-half-plane representative of each conjugate pair) and its gap
    ``d(t)``, or ``None`` for a dead window (``κ₂ ≲ eps``).
    """
    k2 = kappa.get(2, 0.0)
    if not (k2 > eps):
        return None
    R = max(2, int(root_order))
    moments = moments_from_cumulants(kappa, R)
    sqrt_k2 = math.sqrt(k2)
    # Z in τ: coefficient of τ^p is (m_p / κ₂^{p/2}) / p!  (= 1, 0, 1/2, γ₃/6, …)
    coeffs = [moments[p] / (k2 ** (p / 2.0)) / math.factorial(p) for p in range(R + 1)]
    roots_tau = _poly_roots(coeffs, eps)
    if roots_tau.numel() == 0:
        return None
    t_roots = [complex(z) / sqrt_k2 for z in roots_tau.tolist()]
    # upper-half-plane representative of each conjugate pair
    upper = [z for z in t_roots if z.imag >= -1e-12]
    cand = upper if upper else t_roots
    leading = min(cand, key=_real_axis_distance)
    return complex(leading), _real_axis_distance(leading)


def leading_nongaussian_order(
    standardized_spectrum: Tensor, band: float | Tensor = 0.0
) -> int | None:
    """§1 reconstruct-on-demand: the smallest order ``p ≥ 3`` whose ``|γ_p|`` clears
    its null/SE ``band``, i.e. the leading non-Gaussianity.

    ``standardized_spectrum`` is ``standardized_cumulant_spectrum`` (``γ_3 …``).
    ``band`` is a scalar threshold or a per-entry tensor (e.g. the jackknife SE or
    a surrogate null band).  Returns ``None`` if every order is within its band.
    Computed on demand — not tracked."""
    g = standardized_spectrum.detach().to(torch.float64)
    for i in range(g.numel()):
        b = float(band) if isinstance(band, (int, float)) else float(band[i])
        if abs(float(g[i])) > b:
            return i + 3
    return None


def _aggregate_cumulants(
    si: SharedIntermediates, cfg: CumulantConfig, max_order: int
) -> dict[int, float]:
    """Cumulants ``κ_1 .. κ_max_order`` of the aggregate ``S`` that the package
    tracks: plug-in recurrence throughout, with orders 3/4 overridden by the
    configured named estimator (so ``γ₃``/``γ₄`` match ``total_third/fourth_cumulant``)."""
    hi = max(2, max_order)
    kap = total_scalar_cumulants(si.S, si.D, hi)
    est = cfg.cumulant_estimator
    if hi >= 3 and cfg.order >= 3:
        kap[3] = float(_third_cumulant(si.S, si.D, est))
    if hi >= 4 and cfg.order >= 4:
        kap[4] = float(_fourth_cumulant(si.S, si.D, est))
    return kap


def _root_order_eff(cfg: CumulantConfig) -> int:
    """Resolve the effective Fisher-zero truncation order (§1c)."""
    p_tot = max(cfg.order, cfg.max_scalar_cumulant_order)
    default = min(4, p_tot)
    if cfg.root_order is None:
        return max(2, default)
    eff = min(cfg.root_order, p_tot)
    if cfg.root_order > p_tot:
        warnings.warn(
            f"root_order {cfg.root_order} > max total cumulant order {p_tot}; "
            f"using {p_tot}.",
            RuntimeWarning,
            stacklevel=2,
        )
    return max(2, eff)


def _plugin_cumulant_per_column(P: Tensor, D: float, order: int) -> dict[int, Tensor]:
    """Per-column plug-in cumulants ``κ_2..κ_order`` of each column of ``P`` (m, k).

    Returns ``{p: Tensor(k,)}``.  Vectorized moment–cumulant recurrence (the
    per-mode generalization of :func:`total_scalar_cumulants`)."""
    k = P.shape[1] if P.dim() == 2 else 0
    out: dict[int, Tensor] = {}
    if k == 0 or order < 2:
        return {p: torch.zeros(k, dtype=P.dtype, device=P.device) for p in range(2, order + 1)}
    moments = [torch.ones(k, dtype=P.dtype, device=P.device)]      # m_0
    power = torch.ones_like(P)
    for _ in range(order):
        power = power * P
        moments.append(power.sum(dim=0) / D)
    kappa = [torch.zeros(k, dtype=P.dtype, device=P.device)]       # κ_0 slot
    for nn in range(1, order + 1):
        val = moments[nn].clone()
        for kk in range(1, nn):
            val = val - comb(nn - 1, kk - 1) * kappa[kk] * moments[nn - kk]
        kappa.append(val)
    return {p: kappa[p] for p in range(2, order + 1)}


def derived_cumulants(
    si: SharedIntermediates, cfg: CumulantConfig
) -> tuple[dict[str, float], dict[str, Tensor]]:
    """§1/§1c readouts that are pure transforms of the tracked cumulants:
    the standardized cumulant spectrum, the per-mode standardized/extra cumulants,
    and the Fisher / Lee–Yang zero.  Returns ``(scalars, vectors)``."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    if cfg.order < 3:
        # only the Fisher zero is available at order 2 (κ₂ alone ⇒ gap ≈ √(2/κ₂)).
        _fisher_zero_scalars(si, cfg, scalars)
        return scalars, vectors

    dtype = si.S.dtype
    P_std = max(3, cfg.max_scalar_cumulant_order)
    hi = max(P_std, _root_order_eff(cfg))
    kap = _aggregate_cumulants(si, cfg, hi)
    k2 = kap.get(2, 0.0)

    # --- standardized cumulant spectrum γ_p = κ_p / κ₂^{p/2}, p = 3..P_std -----
    denom2 = k2 if k2 > cfg.eps else float("nan")
    gamma = [kap[p] / (denom2 ** (p / 2.0)) for p in range(3, P_std + 1)]
    vectors["standardized_cumulant_spectrum"] = torch.tensor(gamma, dtype=dtype)

    # --- per-mode standardized / extra cumulants ------------------------------
    if cfg.principal_cumulant_order > 4:
        q_hi = cfg.principal_cumulant_order
    else:
        q_hi = min(cfg.order, 4)
    _principal_standardized(si, cfg, q_hi, vectors)

    # --- Fisher / Lee–Yang zero (built from the same κ) -----------------------
    _fisher_zero_scalars(si, cfg, scalars, kap=kap)
    return scalars, vectors


def _principal_standardized(
    si: SharedIntermediates,
    cfg: CumulantConfig,
    q_hi: int,
    vectors: dict[str, Tensor],
) -> None:
    """Per-mode standardized cumulants ``κ_p(P_a)/λ_a^{p/2}`` for p = 3..q_hi, plus
    the raw per-mode ``principal_cumulant_{p}`` for p ≥ 5 (the new primitive)."""
    D = si.D
    P = si.P
    est = cfg.cumulant_estimator
    lam = si.lam_top
    dtype = si.S.dtype
    eps = cfg.eps
    k = P.shape[1] if P.dim() == 2 else 0

    # plug-in per-column cumulants (only when an order needs them)
    need_plugin = q_hi > 4 or (q_hi >= 4 and cfg.order < 4)
    plugin = _plugin_cumulant_per_column(P, D, q_hi) if need_plugin else {}

    cols: list[Tensor] = []
    for p in range(3, q_hi + 1):
        if p == 3:
            kp = _third_cumulant(P, D, est)
        elif p == 4:
            kp = _fourth_cumulant(P, D, est) if cfg.order >= 4 else plugin[4]
        else:
            kp = plugin[p]
            vectors[f"principal_cumulant_{p}"] = kp        # new per-mode primitive
        cols.append(kp / (lam ** (p / 2.0) + eps))
    if cols:
        vectors["principal_standardized_cumulant"] = (
            torch.stack(cols, dim=1) if k else torch.zeros(0, len(cols), dtype=dtype)
        )


def _fisher_zero_scalars(
    si: SharedIntermediates,
    cfg: CumulantConfig,
    scalars: dict[str, float],
    kap: dict[int, float] | None = None,
) -> None:
    """Emit the flat Fisher-zero scalars (``leading_fisher_zero_re/_im/_abs/_arg``
    and ``fisher_zero_gap``); ``scalars`` is float-typed so the complex value is
    split (§1c serialization note)."""
    R = _root_order_eff(cfg)
    if kap is None:
        kap = _aggregate_cumulants(si, cfg, R)
    res = leading_fisher_zero(kap, R, cfg.eps)
    if res is None:
        for suffix in ("re", "im", "abs", "arg"):
            scalars[f"leading_fisher_zero_{suffix}"] = float("nan")
        scalars["fisher_zero_gap"] = float("nan")
        return
    z, gap = res
    scalars["leading_fisher_zero_re"] = float(z.real)
    scalars["leading_fisher_zero_im"] = float(z.imag)
    scalars["leading_fisher_zero_abs"] = float(abs(z))
    scalars["leading_fisher_zero_arg"] = float(math.atan2(z.imag, z.real))
    scalars["fisher_zero_gap"] = float(gap)


# ---------------------------------------------------------------------------
# scalar / diagonal / principal cumulants (§3.2, 3.4, 3.5)
# ---------------------------------------------------------------------------


def scalar_cumulants(
    si: SharedIntermediates, cfg: CumulantConfig
) -> tuple[dict[str, float], dict[str, Tensor]]:
    """First/total/diagonal/principal cumulants up to ``cfg.order``."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    D = si.D
    B_c = si.B_c
    est = cfg.cumulant_estimator

    # --- first cumulant --------------------------------------------------
    vectors["first_cumulant_vector"] = si.E_L
    scalars["total_first_cumulant"] = float(si.E_L.sum())

    # --- second cumulant (total) ----------------------------------------
    # Ê[S^2] = sum_{ij} kappa2[i,j]: the window-variance of the aggregate S.
    # Tied to the single divisor D (the 2RDM trace semantics); ddof=1 makes it
    # the unbiased variance, consistent with the covariance spectrum.
    scalars["total_second_cumulant"] = float(si.s2 / D)

    if cfg.order < 3:
        return scalars, vectors

    # --- third cumulant (co-skewness) -----------------------------------
    scalars["total_third_cumulant"] = float(_third_cumulant(si.S, D, est))
    vectors["diagonal_third_cumulant"] = _third_cumulant(B_c, D, est)
    vectors["principal_third_cumulant"] = _third_cumulant(si.P, D, est)

    if cfg.order < 4:
        return scalars, vectors

    # --- fourth cumulant (connected co-kurtosis) ------------------------
    scalars["total_fourth_cumulant"] = float(_fourth_cumulant(si.S, D, est))
    vectors["diagonal_fourth_cumulant"] = _fourth_cumulant(B_c, D, est)

    pck = _fourth_cumulant(si.P, D, est)
    vectors["principal_fourth_cumulant"] = pck
    # Wick/Isserlis breakdown: connected 4th cumulant relative to the Gaussian
    # prediction 3 lam^2 (lam = the eigenvalue / principal variance).  R5: this
    # equals the per-mode standardized excess kurtosis / 3 (= gamma4(P_a)/3), the
    # p=4 column of principal_standardized_cumulant — not independent of it.
    vectors["wick_breakdown_ratio"] = pck / (3.0 * si.lam_top**2 + cfg.eps)

    # --- general higher-order scalar cumulants of the aggregate S (§3.8) -----
    # kappa_p of S for p >= 5 (plug-in, both estimator modes); orders 2-4 are
    # the named totals above. Gated by cfg.max_scalar_cumulant_order.
    if cfg.max_scalar_cumulant_order > 4:
        kap = total_scalar_cumulants(si.S, D, cfg.max_scalar_cumulant_order)
        for p in range(5, cfg.max_scalar_cumulant_order + 1):
            scalars[f"total_cumulant_{p}"] = float(kap[p])

    return scalars, vectors


# ---------------------------------------------------------------------------
# spectral diagnostics of the 2RDM (§3.3)
# ---------------------------------------------------------------------------


def spectral(
    si: SharedIntermediates, cfg: CumulantConfig
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, Tensor]]:
    """Covariance spectrum, SHC, PR, gap, entropy, IPR (and ``C`` if materialized)."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    matrices: dict[str, Tensor] = {}

    lam = si.eigvals
    vectors["covariance_spectrum"] = lam

    # spectral heat capacity = variance of the spectrum (rank- or full-padded).
    # R2: this is a condensation / anisotropy order parameter, NOT a specific
    # heat — the literal heat-capacity analog is total_second_cumulant = 1ᵀC1.
    scalars["spectral_heat_capacity"] = _spectral_variance(lam, si.rank, cfg.shc_mode)

    s1 = float(lam.sum())
    s2 = float((lam * lam).sum())
    pr = (s1 * s1 / s2) if s2 > cfg.eps else float("nan")
    scalars["participation_ratio"] = pr

    # §1: normalized SHC = Var(λ)/mean(λ)^2 over nonzero λ = r/PR − 1 (exact).
    # A scale-free condensation index; a convenience transform of PR and rank
    # (the EM-section "normalize to remove the n-effect" object), not new state.
    scalars["normalized_shc"] = (
        si.rank / pr - 1.0 if (pr == pr and pr > cfg.eps and si.rank > 0) else float("nan")
    )

    # §8: log-determinant Σ log λ_i over the nonzero spectrum (paper's logged
    # grokking observable; O(rank), with an ε jitter against log(0)).
    if si.rank > 0:
        lam_nz = lam[: si.rank]
        scalars["log_det"] = float(torch.log(lam_nz + cfg.eps).sum())
    else:
        scalars["log_det"] = float("nan")

    # spectral gap and top-eigenvalue fraction: simple condensation order params.
    if lam.numel() >= 2:
        scalars["spectral_gap"] = float(lam[0] - lam[1])
    else:
        scalars["spectral_gap"] = float("nan")
    scalars["top_eigenvalue_fraction"] = (
        float(lam[0] / s1) if s1 > cfg.eps and lam.numel() else float("nan")
    )

    # von Neumann (spectral) entropy and effective rank of rho = C / tr C.
    if s1 > cfg.eps:
        p = lam / s1
        p = p[p > 0]
        H = float(-(p * torch.log(p)).sum())
        scalars["von_neumann_entropy"] = H
        scalars["effective_rank"] = math.exp(H)
    else:
        scalars["von_neumann_entropy"] = float("nan")
        scalars["effective_rank"] = float("nan")

    if si.V is not None:
        # IPR[a] = sum_i V[i,a]^4  in [1/n, 1]
        vectors["inverse_participation_ratio"] = (si.V**4).sum(dim=0)

    if cfg.materialize_matrices and si.C is not None:
        matrices["covariance"] = si.C

    return scalars, vectors, matrices


# ---------------------------------------------------------------------------
# contracted-cumulant spectra (§3.4, 3.5)
# ---------------------------------------------------------------------------


def _contracted_real_spectrum(
    operator: Tensor, symmetric: bool, n: int, eps: float
) -> tuple[Tensor, float]:
    """Real eigenvalues of a contracted operator, descending + padded.

    Returns ``(spectrum, imag_ratio)`` where ``imag_ratio = max|Im| /
    (max|Re| + eps)`` is a numerical-health metric (0 for the symmetric path).
    """
    if symmetric:
        ev = torch.linalg.eigvalsh(operator)         # real
        re = ev
        imag_ratio = 0.0
    else:
        ev = torch.linalg.eigvals(operator)          # complex
        re = ev.real
        denom = float(re.abs().max()) + eps if re.numel() else eps
        imag_ratio = float(ev.imag.abs().max()) / denom if ev.numel() else 0.0
    return _pad_sort_desc(re, n), imag_ratio


def contracted(
    si: SharedIntermediates, cfg: CumulantConfig
) -> tuple[dict[str, Tensor], dict[str, float], dict[str, Tensor]]:
    """Contracted 3rd (§3.4) and 4th (§3.5) cumulant spectra via Gram partners.

    Returns ``(vectors, health, matrices)``.
    """
    vectors: dict[str, Tensor] = {}
    health: dict[str, float] = {}
    matrices: dict[str, Tensor] = {}
    if cfg.order < 3:
        return vectors, health, matrices

    D = si.D
    n = si.n
    B_c = si.B_c
    S = si.S

    # --- third order: M3 = B_c^T diag(S/D) B_c --------------------------
    if si.use_gram:
        G3 = (S / D).unsqueeze(1) * si.G0        # diag(S/D) @ G0 (row scale)
        skew_spec, skew_imag = _contracted_real_spectrum(G3, False, n, cfg.eps)
    else:
        M3 = si.M3 if si.M3 is not None else B_c.transpose(0, 1) @ (
            (S / D).unsqueeze(1) * B_c
        )
        skew_spec, skew_imag = _contracted_real_spectrum(M3, True, n, cfg.eps)
    vectors["contracted_third_cumulant_spectrum"] = skew_spec
    health["contracted_third_cumulant_imag_ratio"] = skew_imag
    if cfg.materialize_matrices and si.M3 is not None:
        matrices["contracted_third_cumulant_matrix"] = si.M3

    if cfg.order < 4:
        _warn_imag(cfg, skew_imag, None)
        return vectors, health, matrices

    # --- fourth order: K = B_c^T Q B_c ----------------------------------
    if si.use_gram:
        G4 = si.Q @ si.G0
        kurt_spec, kurt_imag = _contracted_real_spectrum(G4, False, n, cfg.eps)
    else:
        K = si.K if si.K is not None else B_c.transpose(0, 1) @ (si.Q @ B_c)
        kurt_spec, kurt_imag = _contracted_real_spectrum(K, True, n, cfg.eps)
    vectors["contracted_fourth_cumulant_spectrum"] = kurt_spec
    health["contracted_fourth_cumulant_imag_ratio"] = kurt_imag
    if cfg.materialize_matrices and si.K is not None:
        matrices["contracted_fourth_cumulant_matrix"] = si.K

    _warn_imag(cfg, skew_imag, kurt_imag)
    return vectors, health, matrices


def _warn_imag(cfg: CumulantConfig, skew: float | None, kurt: float | None) -> None:
    for name, val in (
        ("third_cumulant", skew),
        ("fourth_cumulant", kurt),
    ):
        if val is not None and val > cfg.imag_ratio_warn:
            warnings.warn(
                f"contracted_{name}_imag_ratio={val:.2e} exceeds "
                f"{cfg.imag_ratio_warn:.0e}: likely precision loss or "
                f"near-degeneracy; consider dtype=float64.",
                RuntimeWarning,
                stacklevel=3,
            )


# ---------------------------------------------------------------------------
# §5 critical-slowing-down autocorrelation (of the aggregate S)
# ---------------------------------------------------------------------------


def _autocorr_coeffs(x: Tensor, max_lag: int) -> Tensor | None:
    """Normalized autocorrelation ``ρ_0 .. ρ_max_lag`` of a 1-D series, or ``None``
    for a dead (zero-variance) series."""
    x = x - x.mean()
    nm = x.numel()
    var = float((x * x).sum())
    if var <= 0.0 or nm < 2:
        return None
    lags = min(max_lag, nm - 1)
    out = torch.empty(lags + 1, dtype=x.dtype, device=x.device)
    for k in range(lags + 1):
        out[k] = (x[: nm - k] * x[k:]).sum() / var
    return out


def integrated_autocorr_time(x: Tensor, c: float = 5.0) -> tuple[float, float]:
    """Integrated autocorrelation time ``τ_int = 1 + 2 Σ_{k≥1} ρ_k`` of a 1-D
    series via Sokal's automatic windowing (window ``W`` = smallest lag with
    ``W ≥ c·τ_int(W)``).  Returns ``(τ_int, ρ_1)``; ``(1.0, NaN)`` for a dead series."""
    rho = _autocorr_coeffs(x, x.numel() - 1)
    if rho is None or rho.numel() < 2:
        return 1.0, float("nan")
    running = 0.0
    tau = 1.0
    for k in range(1, rho.numel()):
        running += float(rho[k])
        tau_w = 1.0 + 2.0 * running
        if k >= c * tau_w:
            tau = tau_w
            break
    else:
        tau = 1.0 + 2.0 * running
    return max(tau, 1e-6), float(rho[1])


def autocorrelation(
    si: SharedIntermediates, cfg: CumulantConfig
) -> tuple[dict[str, float], dict[str, float], float]:
    """§5: lag-1 autocorrelation and integrated autocorrelation time of the
    aggregate ``S`` (always-on, purely 1-D), plus the honest ``effective_window``
    sample count.  Returns ``(scalars, health, τ_int)`` — ``τ_int`` also feeds the
    MP/TW null (§3b) via ``m_eff``."""
    tau, rho1 = integrated_autocorr_time(si.S)
    scalars = {"autocorr_lag1": rho1, "autocorr_time": tau}
    # Honest sample size: an effectively-independent count can't exceed the row
    # count, so floor τ_int at 1 for m_eff (the raw τ is kept in autocorr_time).
    m_eff = (si.m - (cfg.detrend_order + 1)) / max(tau, 1.0)
    health = {"effective_window": m_eff}
    if tau > 10.0:
        warnings.warn(
            f"autocorr_time τ_int={tau:.1f} ≫ 1: window rows are strongly "
            f"autocorrelated; effective sample size is ~{m_eff:.1f}, not "
            f"{si.m}. RMT/Wishart calibration uses m_eff, and the jackknife SE "
            "underestimates the true error.",
            RuntimeWarning,
            stacklevel=2,
        )
    return scalars, health, tau


# ---------------------------------------------------------------------------
# §7 cross-operator alignment among {C, M3, K}
# ---------------------------------------------------------------------------


def _cos_operator(G_O: Tensor, G0: Tensor, trG0sq: float, eps: float) -> float:
    """Window-space cosine ``cos(C, O) = tr(G_O G0) / (√tr(G0²)·√tr(G_O²))`` for an
    operator ``O = B_cᵀ W B_c`` with Gram partner ``G_O = W G0`` (the ``1/D``
    cancels).  Signed, clamped to ``[-1, 1]``."""
    tr_GO_G0 = float((G_O * G0).sum())            # G0 symmetric ⇒ tr(G_O G0)
    tr_GO_sq = float((G_O * G_O.transpose(0, 1)).sum())
    denom = math.sqrt(max(trG0sq, 0.0)) * math.sqrt(max(tr_GO_sq, 0.0))
    if denom <= eps:
        return float("nan")
    return float(max(-1.0, min(1.0, tr_GO_G0 / denom)))


def alignment(
    si: SharedIntermediates, cfg: CumulantConfig
) -> tuple[dict[str, float], dict[str, Tensor]]:
    """§7: cross-operator alignment of the skew (``M3``) and kurtosis (``K``)
    structure with the variance (``C``) condensation.  Scalar cosines are
    always-on (§7a); the per-mode subspace overlap (§7b) needs eigenvectors.

    Free byproduct of the existing ``G0``/``G3``/``G4`` work — the window-space
    trace identity makes the ``1/D`` cancel.  Sold as "non-Gaussianity
    co-locating with variance condensation," **not** the paper's Gaussian-part
    ``⟨C⁽¹⁾, Δ⟩_F`` (which is the Isserlis piece ``K`` discards)."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    if cfg.order < 3:
        return scalars, vectors

    B_c = si.B_c
    G0 = si.G0 if si.G0 is not None else B_c @ B_c.transpose(0, 1)
    trG0sq = float((G0 * G0).sum())

    G3 = (si.S / si.D).unsqueeze(1) * G0           # diag(S/D) @ G0
    scalars["align_cov_skew"] = _cos_operator(G3, G0, trG0sq, cfg.eps)
    G4 = None
    if cfg.order >= 4:
        G4 = si.Q @ G0
        scalars["align_cov_kurtosis"] = _cos_operator(G4, G0, trG0sq, cfg.eps)

    if cfg.compute_eigenvectors and si.V is not None and si.top_k_eff > 0:
        _subspace_alignment(si, cfg, G3, G4, vectors)
    return scalars, vectors


def _subspace_alignment(
    si: SharedIntermediates,
    cfg: CumulantConfig,
    G3: Tensor,
    G4: Tensor | None,
    vectors: dict[str, Tensor],
) -> None:
    """§7b: per-mode overlap (cosines of the principal angles) between the
    variance subspace ``V_C`` and the recovered skew/kurtosis subspaces
    ``V_O = orthonormalize(B_cᵀ W_O)``, where ``W_O`` are the top window-space
    eigenvectors of the asymmetric Gram partner (recovered for free)."""
    k = si.top_k_eff
    V_C = si.V                                     # (n, k) orthonormal
    for name, G in (("skew", G3), ("kurtosis", G4)):
        if G is None:
            continue
        evals, evecs = torch.linalg.eig(G)         # asymmetric ⇒ general eig
        idx = torch.argsort(evals.real.abs(), descending=True)[:k]
        W = evecs[:, idx].real                      # (m, k)
        V_O = si.B_c.transpose(0, 1) @ W            # (n, k)  pushforward
        Q, _ = torch.linalg.qr(V_O)                 # orthonormalize columns
        # principal angles between span(V_C) and span(Q): singular values of
        # V_Cᵀ Q (= cos θ_i); report descending (1 ⇒ shared direction).
        sv = torch.linalg.svdvals(V_C.transpose(0, 1) @ Q).clamp(0.0, 1.0)
        vectors[f"subspace_align_cov_{name}"] = torch.sort(sv, descending=True).values


# ---------------------------------------------------------------------------
# per-group decomposition (§3.7)
# ---------------------------------------------------------------------------


def per_group(
    si: SharedIntermediates,
    cfg: CumulantConfig,
    groups: dict,
) -> tuple[dict[str, float], dict[str, Tensor]]:
    """Group total variance, group SHC/PR, and block energy (§3.7)."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    D = si.D
    diag_var = (si.B_c**2).sum(dim=0) / D       # (n,)  per-feature variance

    for label, idx in groups.items():
        idx = idx.to(si.B_c.device).long()
        scalars[f"group_total_variance/{label}"] = float(diag_var[idx].sum())

        B_sub = si.B_c[:, idx]
        shc, pr = _shc_pr(B_sub, D, cfg)
        scalars[f"group_shc/{label}"] = shc
        scalars[f"group_pr/{label}"] = pr

        if si.V is not None:
            vectors[f"block_energy/{label}"] = (si.V[idx] ** 2).sum(dim=0)

    return scalars, vectors


# ---------------------------------------------------------------------------
# jackknife error bars (§ optional)
# ---------------------------------------------------------------------------


def _jackknife_se(
    B: Tensor,
    cfg: CumulantConfig,
    groups: dict | None,
    contraction: Tensor | None,
    reference_loss: Tensor | None,
) -> dict[str, float]:
    """Leave-one-row-out jackknife standard error for every scalar observable.

    Re-runs :func:`compute_all` ``m`` times (expensive — gated by
    ``cfg.error_bars``).  SE_jack = sqrt((m-1)/m · Σ_i (θ_(i) − θ̄)²) over the
    leave-one-out estimates (NaNs dropped per key).  Returns ``{key+'/se': se}``.
    """
    # strip the per-measurement re-resampling layers so a single jackknife pass
    # doesn't trigger the surrogate null (or itself) on every leave-one-out.
    base = replace(cfg, error_bars="off", null_model="off")
    m = B.shape[0]
    vals: dict[str, list[float]] = {}
    keep = torch.ones(m, dtype=torch.bool)
    for i in range(m):
        keep[i] = False
        Bi = B[keep]
        keep[i] = True
        meas_i = compute_all(
            Bi, base, groups=groups, contraction=contraction,
            reference_loss=reference_loss,
        )
        for k, v in meas_i.scalars.items():
            vals.setdefault(k, []).append(v)

    out: dict[str, float] = {}
    for k, lst in vals.items():
        t = torch.tensor(lst, dtype=torch.float64)
        t = t[torch.isfinite(t)]
        if t.numel() < 2:
            out[f"{k}/se"] = float("nan")
            continue
        c = t.numel()
        var = ((t - t.mean()) ** 2).sum()
        out[f"{k}/se"] = float(torch.sqrt((c - 1) / c * var))
    return out


# ---------------------------------------------------------------------------
# §3 null-background model (surrogate bootstrap + Marchenko-Pastur / Tracy-Widom)
# ---------------------------------------------------------------------------


def _circular_surrogate(B_c: Tensor) -> Tensor:
    """Independent circular shift of each column of ``B_c`` (§3a).

    Destroys cross-probe phase alignment (the 2RDM signal) while preserving every
    column's marginal *and* its within-column autocorrelation, so the aspect
    ratio stays honest."""
    m, n = B_c.shape
    shifts = torch.randint(1, m, (n,), device=B_c.device)
    idx = (torch.arange(m, device=B_c.device).unsqueeze(1) - shifts.unsqueeze(0)) % m
    return torch.gather(B_c, 0, idx.long())


def _surrogate_null(
    si: SharedIntermediates,
    cfg: CumulantConfig,
    observed: dict[str, float],
    contraction: Tensor | None,
) -> dict[str, float]:
    """§3a surrogate (block-bootstrap) null: per-scalar ``null_mean`` / ``null_sd``
    / ``null_z``, plus the ``spectral_drift`` null floor between independent
    surrogates (used by the §4 noise-floored distances)."""
    base = replace(
        cfg, null_model="off", error_bars="off", per_group=False,
        compute_eigenvectors=False, materialize_matrices=False,
        compute_cka_distance=False, track_modes=False,
    )
    vals: dict[str, list[float]] = {}
    specs: list[Tensor] = []
    for _ in range(cfg.null_samples):
        sur = _circular_surrogate(si.B_c)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            meas_s = compute_all(sur, base, contraction=contraction)
        for k, v in meas_s.scalars.items():
            vals.setdefault(k, []).append(v)
        specs.append(meas_s.vectors["covariance_spectrum"])

    out: dict[str, float] = {}
    for k, lst in vals.items():
        t = torch.tensor(lst, dtype=torch.float64)
        t = t[torch.isfinite(t)]
        if t.numel() < 2:
            continue
        mean = float(t.mean())
        sd = float(t.std(unbiased=True))
        out[f"{k}/null_mean"] = mean
        out[f"{k}/null_sd"] = sd
        if k in observed:
            obs = observed[k]
            out[f"{k}/null_z"] = (
                (obs - mean) / sd if (sd > cfg.eps and obs == obs) else float("nan")
            )

    # spectral_drift null floor: drift between independent surrogate spectra.
    from .distances import spectral_drift
    drifts = [
        spectral_drift(specs[i], specs[i + 1], cfg.eps) for i in range(len(specs) - 1)
    ]
    drifts = [d for d in drifts if d == d]
    if len(drifts) >= 2:
        dt = torch.tensor(drifts, dtype=torch.float64)
        out["spectral_drift/null_mean"] = float(dt.mean())
        out["spectral_drift/null_sd"] = float(dt.std(unbiased=True))
    return out


def _mp_grid(gamma: float, sigma2: float, npts: int = 2048):
    """Grid ``(xs, cdf)`` of the Marchenko-Pastur law (ratio ``gamma``, scale
    ``sigma2``) over its bulk support, ``cdf`` normalized to 1 (the continuous
    part — we compare against the *nonzero* empirical eigenvalues)."""
    sg = math.sqrt(gamma)
    lo = sigma2 * (1.0 - sg) ** 2
    hi = sigma2 * (1.0 + sg) ** 2
    if hi <= lo:
        return None
    xs = torch.linspace(lo, hi, npts, dtype=torch.float64)
    inside = ((hi - xs) * (xs - lo)).clamp(min=0.0)
    dens = torch.sqrt(inside) / (2.0 * math.pi * gamma * sigma2 * xs.clamp(min=1e-300))
    dx = (hi - lo) / (npts - 1)
    cdf = torch.zeros(npts, dtype=torch.float64)
    cdf[1:] = torch.cumsum((dens[:-1] + dens[1:]) * 0.5 * dx, dim=0)
    total = float(cdf[-1])
    if total <= 0.0:
        return None
    return xs, cdf / total


def _mp_median(gamma: float) -> float:
    """Median of the unit-scale (σ²=1) Marchenko-Pastur bulk for ratio ``gamma``."""
    grid = _mp_grid(gamma, 1.0)
    if grid is None:
        return float("nan")
    xs, cdf = grid
    idx = int(torch.searchsorted(cdf, torch.tensor(0.5, dtype=torch.float64)))
    if idx <= 0:
        return float(xs[0])
    if idx >= xs.numel():
        return float(xs[-1])
    c0, c1 = float(cdf[idx - 1]), float(cdf[idx])
    x0, x1 = float(xs[idx - 1]), float(xs[idx])
    if c1 == c0:
        return x0
    return x0 + (0.5 - c0) / (c1 - c0) * (x1 - x0)


def _mp_ks(bulk: Tensor, gamma: float, sigma2: float) -> float:
    """KS distance between the empirical CDF of the sub-edge (bulk) eigenvalues and
    the Marchenko-Pastur CDF (null goodness-of-fit; large ⇒ MP inadequate)."""
    if bulk.numel() == 0:
        return float("nan")
    grid = _mp_grid(gamma, sigma2)
    if grid is None:
        return float("nan")
    xs, cdf = grid
    b = torch.sort(bulk.to(torch.float64)).values
    idx = torch.searchsorted(xs, b).clamp(1, xs.numel() - 1)
    x0, x1 = xs[idx - 1], xs[idx]
    y0, y1 = cdf[idx - 1], cdf[idx]
    t = (b - x0) / (x1 - x0).clamp(min=1e-300)
    fmp = (y0 + t * (y1 - y0)).clamp(0.0, 1.0)
    fmp = torch.where(b <= xs[0], torch.zeros_like(fmp), fmp)
    fmp = torch.where(b >= xs[-1], torch.ones_like(fmp), fmp)
    nb = b.numel()
    f_over = torch.arange(1, nb + 1, dtype=torch.float64) / nb
    f_under = torch.arange(0, nb, dtype=torch.float64) / nb
    return float(torch.maximum((f_over - fmp).abs(), (fmp - f_under).abs()).max())


def _mp_null(
    si: SharedIntermediates, cfg: CumulantConfig, tau_int: float
) -> tuple[dict[str, float], dict[str, float]]:
    """§3b Marchenko-Pastur / Tracy-Widom null: the bulk edge ``mp_edge_upper``
    (BBP detachment threshold), the TW-gated spike count ``n_spikes``, the
    detached-energy fraction, and the bulk goodness-of-fit ``mp_bulk_ks``.

    Uses the **effective** sample count ``m_eff = (m − (detrend_order+1)) / τ_int``
    (autocorrelated rows), and a robust noise level ``σ̂²`` matched on the median."""
    scalars: dict[str, float] = {}
    health: dict[str, float] = {}
    nan = float("nan")
    lam = si.eigvals.to(torch.float64)
    rank = si.rank
    tau = max(tau_int, 1.0)
    m_eff = (si.m - (cfg.detrend_order + 1)) / tau
    if rank < 2 or m_eff <= 0.0:
        scalars.update(mp_edge_upper=nan, n_spikes=nan, spike_energy_fraction=nan)
        health["mp_bulk_ks"] = nan
        return scalars, health

    gamma = si.n / m_eff
    nz = lam[:rank]
    emp_median = float(nz.median())
    med_unit = _mp_median(gamma)
    if not (med_unit > 0.0) or emp_median != emp_median:
        scalars.update(mp_edge_upper=nan, n_spikes=nan, spike_energy_fraction=nan)
        health["mp_bulk_ks"] = nan
        return scalars, health

    sigma2 = emp_median / med_unit
    sg = math.sqrt(gamma)
    edge = sigma2 * (1.0 + sg) ** 2
    # Tracy-Widom fluctuation scale of the largest eigenvalue (∝ σ²·m_eff^(−2/3)).
    tw_sd = sigma2 * (1.0 + sg) * ((1.0 + 1.0 / sg) ** (1.0 / 3.0)) * m_eff ** (-2.0 / 3.0)
    thresh = edge + 2.0 * tw_sd                      # a couple of TW sd above edge

    total = float(lam.sum())
    scalars["mp_edge_upper"] = edge
    # §8: the re-grounded condensation gap — the leading mode's separation from
    # the *noise edge* (preferred over the bare spectral_gap once MP is on).
    scalars["edge_gap"] = float(nz[0]) - edge
    scalars["n_spikes"] = float(int((nz > thresh).sum()))
    spike_mass = float(nz[nz > edge].sum())
    scalars["spike_energy_fraction"] = spike_mass / total if total > cfg.eps else nan
    health["mp_sigma2"] = sigma2
    health["mp_gamma"] = gamma
    health["mp_bulk_ks"] = _mp_ks(nz[nz <= edge], gamma, sigma2)
    return scalars, health


def null_model(
    si: SharedIntermediates,
    cfg: CumulantConfig,
    tau_int: float,
    observed: dict[str, float],
    contraction: Tensor | None = None,
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, float]]:
    """§3 dispatch: surrogate z-scores and/or the MP/TW edge.  Returns
    ``(scalars, vectors, health)``."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    health: dict[str, float] = {}
    if cfg.null_model in ("surrogate", "both"):
        scalars.update(_surrogate_null(si, cfg, observed, contraction))
    if cfg.null_model in ("mp", "both"):
        sc, h = _mp_null(si, cfg, tau_int)
        scalars.update(sc)
        health.update(h)
    return scalars, vectors, health


# ---------------------------------------------------------------------------
# §6 BBP spike-detachment bitmask (which probes participate, as an integer)
# ---------------------------------------------------------------------------


def bbp_bitmask(
    si: SharedIntermediates, cfg: CumulantConfig, mp_edge_upper: float
) -> tuple[dict[str, float], dict[str, Tensor], dict[str, object]]:
    """§6: per detached mode (``λ_a > mp_edge_upper``), the integer support mask
    of participating probes — bit ``i`` set ⇔ probe ``i`` is in the smallest set
    capturing ``rho_keep`` of the eigenvector's energy.

    The mask is a Python big-int serialized as a hex string in ``aux`` (the
    ``scalars`` bucket is float).  Squared eigenvector entries ⇒ the D6 sign gauge
    is irrelevant."""
    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    aux: dict[str, object] = {}
    if si.V is None or si.top_k_eff == 0 or not (mp_edge_upper == mp_edge_upper):
        return scalars, vectors, aux

    masks: list[str] = []
    counts: list[int] = []
    mask_any = 0
    for a in range(si.top_k_eff):
        if float(si.lam_top[a]) <= mp_edge_upper:
            continue
        v2 = si.V[:, a] ** 2
        order = torch.argsort(v2, descending=True)
        cum = torch.cumsum(v2[order], dim=0)
        total = float(v2.sum())
        keep_n = int((cum < cfg.rho_keep * total).sum()) + 1
        keep_n = min(keep_n, v2.numel())
        support = order[:keep_n].tolist()
        mask = 0
        for i in support:
            mask |= 1 << int(i)
        masks.append(hex(mask))
        counts.append(len(support))
        mask_any |= mask

    aux["active_probe_mask"] = masks
    aux["active_probe_mask_any"] = hex(mask_any)
    vectors["active_probe_count"] = torch.tensor(counts, dtype=torch.float64)
    return scalars, vectors, aux


# ---------------------------------------------------------------------------
# top-level assembly
# ---------------------------------------------------------------------------


def compute_all(
    B: Tensor,
    cfg: CumulantConfig,
    groups: dict | None = None,
    step: int = 0,
    contraction: Tensor | None = None,
    reference_loss: Tensor | None = None,
) -> Measurement:
    """Compute the full observable catalog for buffer ``B`` (§4 → §3).

    ``groups`` maps a label to a ``LongTensor`` of column indices (from
    :meth:`cumobs.probes.ProbeSet.groups`).  ``reference_loss`` (optional) is the
    converged-model loss ``ell(theta*)`` enabling the ``excess_loss`` byproduct
    (§9.4).  Distances are **not** computed here (they need a reference
    measurement); the monitor adds them.
    """
    si = build_intermediates(B, cfg, contraction=contraction)

    scalars: dict[str, float] = {}
    vectors: dict[str, Tensor] = {}
    matrices: dict[str, Tensor] = {}
    health: dict[str, float] = dict(si.health)
    aux: dict[str, object] = {}

    s_sc, s_vec = scalar_cumulants(si, cfg)
    scalars.update(s_sc)
    vectors.update(s_vec)

    sp_sc, sp_vec, sp_mat = spectral(si, cfg)
    scalars.update(sp_sc)
    vectors.update(sp_vec)
    matrices.update(sp_mat)

    # §1/§1c: standardized cumulant spectrum, per-mode standardized cumulants,
    # Fisher / Lee–Yang zero (pure transforms of the tracked cumulants).
    d_sc, d_vec = derived_cumulants(si, cfg)
    scalars.update(d_sc)
    vectors.update(d_vec)

    if cfg.order >= 3:
        c_vec, c_health, c_mat = contracted(si, cfg)
        vectors.update(c_vec)
        health.update(c_health)
        matrices.update(c_mat)

    # §5: critical-slowing-down autocorrelation of the aggregate S (always-on).
    a_sc, a_health, tau_int = autocorrelation(si, cfg)
    scalars.update(a_sc)
    health.update(a_health)

    # §7: cross-operator alignment among {C, M3, K} (scalar always-on, subspace
    # when eigenvectors are available).
    if cfg.order >= 3:
        al_sc, al_vec = alignment(si, cfg)
        scalars.update(al_sc)
        vectors.update(al_vec)

    # §3: null-background model (surrogate z-scores and/or MP/TW edge).
    if cfg.null_model != "off":
        n_sc, n_vec, n_health = null_model(
            si, cfg, tau_int, dict(scalars), contraction=contraction,
        )
        scalars.update(n_sc)
        vectors.update(n_vec)
        health.update(n_health)

    # §6: BBP spike-detachment bitmask (needs eigenvectors + the MP edge).
    if cfg.compute_eigenvectors and ("mp_edge_upper" in scalars):
        b_sc, b_vec, b_aux = bbp_bitmask(si, cfg, scalars["mp_edge_upper"])
        scalars.update(b_sc)
        vectors.update(b_vec)
        aux.update(b_aux)

    if cfg.per_group and groups:
        g_sc, g_vec = per_group(si, cfg, groups)
        scalars.update(g_sc)
        vectors.update(g_vec)

    # §9.4 excess-loss byproduct (reuses E_L); beta absorbed by the caller.
    if reference_loss is not None:
        scalars["excess_loss"] = float(
            (si.E_L - reference_loss.to(si.E_L)).sum()
        )

    # eigen-factors for the CKA distance (not logged; consumed by the monitor).
    factors: dict[str, Tensor] = {}
    if cfg.compute_eigenvectors and si.V is not None:
        factors["V"] = si.V
        factors["lam_top"] = si.lam_top

    # optional jackknife error bars on the scalar observables.
    if cfg.error_bars == "jackknife":
        scalars.update(
            _jackknife_se(B, cfg, groups, contraction, reference_loss)
        )

    return Measurement(
        step=step,
        scalars=scalars,
        vectors=vectors,
        matrices=matrices,
        health=health,
        factors=factors,
        aux=aux,
    )
