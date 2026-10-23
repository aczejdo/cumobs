"""Distances between successive 2RDM measurements (§3.6, decision D2).

Two flavors ship in v1:

* **Classical** (spectrum-only).  Each spectrum ``lambda`` is normalized to a
  probability vector ``p = lambda / sum(lambda)`` and the distance is computed
  spectrum-to-spectrum.  These are eigenvector-*blind*: two states with the same
  spectrum but rotated eigenvectors score identical.
* **Quantum** (density-matrix).  On ``rho = C / tr C`` the Uhlmann fidelity and
  trace norm are eigenvector-*sensitive* and cost ``O(n^3)``; they require the
  materialized covariance matrices.

All functions return Python floats.  Degenerate inputs (zero total mass) return
``NaN`` so the monitor can skip them rather than divide by zero (§8.5).
"""

from __future__ import annotations

import math
import warnings

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# classical (spectrum-only)
# ---------------------------------------------------------------------------


def _as_prob(x: Tensor, eps: float, warn_signed: bool = True) -> Tensor | None:
    """Normalize a non-negative spectrum to a probability vector; clip negatives.

    Returns ``None`` if the total mass is ~0 (dead / constant window).

    The classical fidelity / trace distance treat a spectrum as a probability
    distribution, which is only valid for a **non-negative** (PSD) spectrum.  The
    *contracted* skew / kurtosis spectra are signed; clipping their negative half
    silently discards information.  When the negative mass is non-trivial this
    emits a (rate-limited) ``RuntimeWarning`` rather than failing quietly — use
    the covariance source, the quantum distances, or the CKA distance instead.
    """
    x = x.detach().to(torch.float64)
    if warn_signed:
        abs_total = float(x.abs().sum())
        neg_mass = float(x.clamp(max=0.0).abs().sum())
        if abs_total > eps and neg_mass / abs_total > 1e-6:
            warnings.warn(
                f"spectrum has significant negative mass ({neg_mass / abs_total:.1%}): "
                "classical fidelity / trace distance clip negatives and are only "
                "meaningful for the PSD covariance source. Prefer "
                "distance_source='covariance', the quantum distances, or the CKA "
                "distance for signed (contracted) spectra.",
                RuntimeWarning,
                stacklevel=3,
            )
    x = torch.clamp(x, min=0.0)
    total = float(x.sum())
    if total <= eps:
        return None
    return x / total


def classical_fidelity(p: Tensor, q: Tensor, eps: float = 1e-8) -> float:
    """Classical (Bhattacharyya) fidelity ``(sum_i sqrt(p_i q_i))^2`` in ``[0,1]``.

    ``p`` and ``q`` are raw covariance spectra; they are clipped and normalized
    to probability vectors internally.  ``=1`` iff ``p == q``.  Spectra are
    compared element-wise in their given (descending) order — eigenvector
    identity is not used (the D2 eigenvector-blindness caveat).
    """
    pp = _as_prob(p, eps)
    qq = _as_prob(q, eps)
    if pp is None or qq is None:
        return float("nan")
    pp, qq = _align(pp, qq)
    bc = torch.sqrt(pp * qq).sum()
    return float((bc * bc).clamp(0.0, 1.0))


def trace_distance(p: Tensor, q: Tensor, eps: float = 1e-8) -> float:
    """Classical trace distance = total variation ``0.5 sum_i |p_i - q_i|``.

    In ``[0, 1]``; ``=0`` iff ``p == q``.
    """
    pp = _as_prob(p, eps)
    qq = _as_prob(q, eps)
    if pp is None or qq is None:
        return float("nan")
    pp, qq = _align(pp, qq)
    return float(0.5 * (pp - qq).abs().sum().clamp(0.0, 1.0))


def _align(p: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
    """Zero-pad the shorter probability vector so both have equal length."""
    if p.numel() == q.numel():
        return p, q
    n = max(p.numel(), q.numel())
    pad = lambda v: torch.cat(
        [v, torch.zeros(n - v.numel(), dtype=v.dtype, device=v.device)]
    )
    return pad(p), pad(q)


def spectral_drift(p: Tensor, q: Tensor, eps: float = 1e-8) -> float:
    """Scale-**sensitive** relative L2 distance between two raw spectra.

    ``||p − q||₂ / max(||p||₂, ||q||₂, eps)`` on the un-normalized (descending,
    zero-padded) spectra — ``0`` iff identical, in ``[0, ~√2]``.  Unlike
    :func:`classical_fidelity` / :func:`trace_distance` (which normalize to
    probability vectors and are therefore **blind to overall scale**), this
    responds to a pure change in total variance — e.g. a window whose spectrum
    keeps its shape but grows 10× scores ≈ 0.9 here but fidelity 1.0 there.
    """
    p = p.detach().to(torch.float64)
    q = q.detach().to(torch.float64)
    p, q = _align(p, q)
    num = float(torch.linalg.vector_norm(p - q))
    den = max(
        float(torch.linalg.vector_norm(p)),
        float(torch.linalg.vector_norm(q)),
        eps,
    )
    return num / den


# ---------------------------------------------------------------------------
# §2 geodesic distances (metric forms of the two fidelities)
# ---------------------------------------------------------------------------


def fisher_rao_distance(p: Tensor, q: Tensor, eps: float = 1e-8) -> float:
    """Fisher–Rao geodesic on the simplex of normalized spectra (§2).

    ``2·arccos(√classical_fidelity)`` ∈ ``[0, π]`` — a proper metric (respects the
    triangle inequality), so the preferred default over the bare
    :func:`classical_fidelity` (which is recoverable from it).  Inherits the
    clip-negative-mass-and-warn behaviour on signed (contracted) sources (R6),
    since it is a transform of :func:`classical_fidelity`."""
    f = classical_fidelity(p, q, eps)
    if f != f:                                       # NaN (dead window)
        return float("nan")
    bc = math.sqrt(min(max(f, 0.0), 1.0))
    return float(2.0 * math.acos(min(max(bc, 0.0), 1.0)))


def bures_distance(rho: Tensor, sigma: Tensor, eps: float = 1e-8) -> float:
    """Bures distance ``√(2(1 − √quantum_fidelity))`` ∈ ``[0, √2]`` (§2).

    The geometry whose geodesic is the Uhlmann fidelity; the eigenvector-sensitive
    operator metric, preferred over the bare :func:`quantum_fidelity`."""
    qf = quantum_fidelity(rho, sigma, eps)
    if qf != qf:
        return float("nan")
    root = math.sqrt(min(max(qf, 0.0), 1.0))
    return float(math.sqrt(max(0.0, 2.0 * (1.0 - root))))


def bures_angle(rho: Tensor, sigma: Tensor, eps: float = 1e-8) -> float:
    """Bures angle ``arccos(√quantum_fidelity)`` ∈ ``[0, π/2]`` (§2).

    The operator analogue of :func:`fisher_rao_distance`; when ``ρ, σ`` commute,
    ``quantum_fidelity → classical_fidelity`` and ``bures_angle =
    ½·fisher_rao_distance``."""
    qf = quantum_fidelity(rho, sigma, eps)
    if qf != qf:
        return float("nan")
    root = math.sqrt(min(max(qf, 0.0), 1.0))
    return float(math.acos(min(max(root, 0.0), 1.0)))


# ---------------------------------------------------------------------------
# §4 noise-floored distances (zero point = "indistinguishable from noise")
# ---------------------------------------------------------------------------


def denoised_spectral_drift(drift: float, null_mean: float, eps: float = 1e-8) -> float:
    """``max(0, (drift − μ₀)/(1 − μ₀))`` — the drift with its surrogate null floor
    subtracted and renormalized so an identical-pair limit maps 1→1 (§4).
    ``μ₀`` is the null mean of :func:`spectral_drift` between independent
    surrogates."""
    if drift != drift or null_mean != null_mean:
        return float("nan")
    denom = 1.0 - null_mean
    if abs(denom) <= eps:
        return float("nan")
    return float(max(0.0, (drift - null_mean) / denom))


def spectral_drift_snr(
    drift: float, null_mean: float, null_sd: float, eps: float = 1e-8
) -> float:
    """``(drift − μ₀)/s₀`` — the drift in units of its surrogate null sd (§4)."""
    if drift != drift or null_mean != null_mean or null_sd != null_sd:
        return float("nan")
    if null_sd <= eps:
        return float("nan")
    return float((drift - null_mean) / null_sd)


def signal_energy_drift(
    p: Tensor, q: Tensor, edge: float, eps: float = 1e-8
) -> float:
    """:func:`spectral_drift` on the **signal** spectra ``max(0, λ − edge)`` only,
    so the sub-edge (MP bulk) contributes exactly zero (§4).  ``edge`` is the
    MP upper edge ``mp_edge_upper`` of the current measurement."""
    if edge != edge:
        return float("nan")
    ps = (p.detach().to(torch.float64) - edge).clamp(min=0.0)
    qs = (q.detach().to(torch.float64) - edge).clamp(min=0.0)
    return spectral_drift(ps, qs, eps)


# ---------------------------------------------------------------------------
# CKA (representation-similarity on the top-k eigen-factors)
# ---------------------------------------------------------------------------


def cka_similarity(
    V1: Tensor, lam1: Tensor, V2: Tensor, lam2: Tensor, eps: float = 1e-8
) -> float:
    """Linear-CKA similarity between two rank-``k`` covariance factorizations.

    Each covariance is ``C = V diag(lam) Vᵀ`` with ``V`` (``n × k``) orthonormal
    columns (the top-``k`` eigen-factors).  Linear CKA is the Frobenius cosine

        CKA = ⟨C₁,C₂⟩_F / (‖C₁‖_F ‖C₂‖_F)
            = Σ_{ab} lam1_a lam2_b (V1ᵀ V2)_{ab}²  / (‖lam1‖₂ ‖lam2‖₂),

    in ``[0, 1]`` (``=1`` iff the two covariances are proportional).  It is
    **eigenvector-sensitive** (it sees a subspace rotation, which the classical
    spectrum distances do not) and costs ``O(n k²)`` — far cheaper than the
    ``O(n³)`` quantum distances.  Returns ``NaN`` for a dead factor.
    """
    V1 = V1.detach().to(torch.float64)
    V2 = V2.detach().to(torch.float64)
    lam1 = lam1.detach().to(torch.float64)
    lam2 = lam2.detach().to(torch.float64)
    if lam1.numel() == 0 or lam2.numel() == 0:
        return float("nan")
    n1 = float(torch.linalg.vector_norm(lam1))
    n2 = float(torch.linalg.vector_norm(lam2))
    if n1 <= eps or n2 <= eps:
        return float("nan")
    M = V1.transpose(0, 1) @ V2                       # (k1, k2)
    num = float((lam1.unsqueeze(1) * lam2.unsqueeze(0) * (M * M)).sum())
    return min(max(num / (n1 * n2), 0.0), 1.0)


def cka_distance(
    V1: Tensor, lam1: Tensor, V2: Tensor, lam2: Tensor, eps: float = 1e-8
) -> float:
    """``1 − cka_similarity`` — an eigenvector-sensitive representation drift in ``[0,1]``."""
    sim = cka_similarity(V1, lam1, V2, lam2, eps)
    return float("nan") if sim != sim else 1.0 - sim


# ---------------------------------------------------------------------------
# quantum (density-matrix; eigenvector-sensitive)
# ---------------------------------------------------------------------------


def _as_density(C: Tensor, eps: float) -> Tensor | None:
    """Normalize a PSD covariance matrix to a unit-trace density matrix."""
    C = C.detach().to(torch.float64)
    C = 0.5 * (C + C.transpose(-1, -2))      # symmetrize against round-off
    tr = float(torch.diagonal(C).sum())
    if tr <= eps:
        return None
    return C / tr


def quantum_fidelity(rho: Tensor, sigma: Tensor, eps: float = 1e-8) -> float:
    """Uhlmann fidelity ``(tr sqrt(sqrt(rho) sigma sqrt(rho)))^2`` in ``[0,1]``.

    ``rho`` / ``sigma`` are raw covariance matrices; they are symmetrized and
    normalized to unit-trace density matrices internally.  Eigenvector-sensitive
    (decision D2): rotated eigenvectors lower the fidelity even at equal spectra.
    """
    r = _as_density(rho, eps)
    s = _as_density(sigma, eps)
    if r is None or s is None:
        return float("nan")
    # sqrt(rho) via symmetric eigendecomposition
    evals, evecs = torch.linalg.eigh(r)
    sqrt_r = (evecs * torch.clamp(evals, min=0.0).sqrt()) @ evecs.transpose(-1, -2)
    inner = sqrt_r @ s @ sqrt_r
    inner = 0.5 * (inner + inner.transpose(-1, -2))
    w = torch.clamp(torch.linalg.eigvalsh(inner), min=0.0)
    f = float(w.sqrt().sum() ** 2)
    return min(max(f, 0.0), 1.0)


def quantum_trace_distance(rho: Tensor, sigma: Tensor, eps: float = 1e-8) -> float:
    """Quantum trace distance ``0.5 ||rho - sigma||_1`` in ``[0,1]``.

    The Schatten-1 norm of the (Hermitian) difference is the sum of absolute
    eigenvalues.  ``=0`` iff ``rho == sigma``.
    """
    r = _as_density(rho, eps)
    s = _as_density(sigma, eps)
    if r is None or s is None:
        return float("nan")
    diff = r - s
    diff = 0.5 * (diff + diff.transpose(-1, -2))
    evals = torch.linalg.eigvalsh(diff)
    return float(0.5 * evals.abs().sum().clamp(0.0, 1.0))
