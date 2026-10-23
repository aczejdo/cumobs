"""Example 05 — the non-Gaussianity profile and the Fisher / Lee-Yang zero (§1, §1c).

The v0.3 batch adds the *order profile of non-Gaussianity* as a single readout:

    standardized_cumulant_spectrum = [γ₃, γ₄, γ₅, …],   γ_p = κ_p(S) / κ₂(S)^{p/2}

zero at every order for a Gaussian window; the first nonzero `γ_p` is the leading
non-Gaussianity (odd p = asymmetry, even p = tail/peakedness). The same aggregate
cumulants drive the **Fisher / Lee-Yang zero**: the leading complex root of the
truncated partition function `Z_P(t)`, whose gap → 0 near a transition and whose
*phase* is switched on by skew.

This script contrasts a (near-)Gaussian window with a deliberately skewed one,
prints the standardized spectrum and Fisher zero for both, and demonstrates the
`factorization_residual` and `leading_nongaussian_order` helpers. Pure math core
— no model, no monitor.

Run::

    python examples/05_nongaussianity_and_fisher.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import (
    CumulantConfig,
    build_intermediates,
    compute_all,
    factorization_residual,
    leading_nongaussian_order,
    total_scalar_cumulants,
)


def make_buffer(kind: str, m: int = 200, n: int = 30, seed: int = 0) -> torch.Tensor:
    """A window whose aggregate `S` is Gaussian or skewed via a common-mode factor.

    `B[t,:] = a[t]·v1 + noise`, so the aggregate `S = B_c·1` mostly tracks `a`.
    `kind="gaussian"` draws `a ~ N(0,1)`; `kind="skewed"` draws a shifted χ²₃
    (right-skewed). The per-probe noise is Gaussian either way.
    """
    g = torch.Generator().manual_seed(seed)
    v1 = torch.ones(n, dtype=torch.float64) / math.sqrt(n)
    if kind == "gaussian":
        a = torch.randn(m, generator=g, dtype=torch.float64)
    else:  # right-skewed common-mode amplitude (mean-removed χ²₃)
        z = torch.randn(m, 3, generator=g, dtype=torch.float64)
        a = (z * z).sum(dim=1) - 3.0
    noise = 0.1 * torch.randn(m, n, generator=g, dtype=torch.float64)
    return torch.outer(a, v1) + noise


def report(kind: str, cfg: CumulantConfig) -> None:
    B = make_buffer(kind)
    meas = compute_all(B, cfg)
    gamma = meas.vectors["standardized_cumulant_spectrum"]
    print(f"== {kind} window ==")
    labels = [f"g{p}" for p in range(3, 3 + gamma.numel())]   # g3=skew, g4=kurtosis
    print("  standardized cumulant spectrum (g_p = kappa_p / kappa_2^(p/2)):")
    print("    " + "  ".join(f"{lab}={float(v):+.3f}" for lab, v in zip(labels, gamma)))
    # leading non-Gaussianity: smallest order whose |g_p| clears a band
    band = 0.3
    order = leading_nongaussian_order(gamma, band=band)
    desc = {3: "asymmetry (skew)", 4: "tails (kurtosis)"}.get(order, "higher order")
    print(f"  leading_nongaussian_order(|g|>{band}) = {order}"
          + (f"  -> {desc}" if order else "  -> Gaussian-like"))
    # Fisher / Lee-Yang zero: gap (criticality) + phase (shape information)
    gap = meas.scalars["fisher_zero_gap"]
    re = meas.scalars["leading_fisher_zero_re"]
    im = meas.scalars["leading_fisher_zero_im"]
    arg_deg = math.degrees(meas.scalars["leading_fisher_zero_arg"])
    k2 = meas.scalars["total_second_cumulant"]
    print(f"  fisher_zero_gap = {gap:.4f}   (leading order ~ sqrt(2/k2) = {math.sqrt(2 / k2):.4f})")
    print(f"  leading_fisher_zero = {re:+.4f}{im:+.4f}i   phase = {arg_deg:+.1f} deg"
          "   (phase encodes shape; gap encodes criticality)\n")


def fisher_criticality_demo() -> None:
    """The Fisher gap is a criticality readout: it shrinks as the variance grows.

    Scaling a window's amplitude by `s` scales kappa_2 by s^2, so the gap
    (~ sqrt(2/kappa_2)) shrinks ~ 1/s -- the partition-function signature of an
    approaching transition (the leading zero pinching toward the real axis).
    """
    print("== Fisher-zero gap as a criticality signature (1c) ==")
    base = make_buffer("gaussian")
    cfg = CumulantConfig(window_size=base.shape[0], order=4,
                         cumulant_estimator="biased", dtype=torch.float64)
    print(f"  {'scale':>6} {'total_2nd_cumulant':>20} {'fisher_zero_gap':>17}")
    for scale in (1.0, 2.0, 4.0, 8.0):
        meas = compute_all(scale * base, cfg)
        k2 = meas.scalars["total_second_cumulant"]
        gap = meas.scalars["fisher_zero_gap"]
        print(f"  {scale:>6.1f} {k2:>20.2f} {gap:>17.5f}")
    print("  (variance ~ scale^2 grows, gap ~ 1/scale shrinks -> 0 = approaching a transition)\n")


def factorization_demo() -> None:
    """`factorization_residual(kappa, M, N)` reconstructs M->N residuals from stored kappa."""
    print("== factorization residuals (1b) -- reconstructed, never tracked ==")
    B = make_buffer("skewed")
    cfg = CumulantConfig(window_size=B.shape[0], order=4, max_scalar_cumulant_order=6)
    si = build_intermediates(B, cfg)
    kap = total_scalar_cumulants(si.S, si.D, 6)        # kappa_1..kappa_6 of the aggregate
    print("  aggregate cumulants k2..k6:",
          "  ".join(f"k{p}={kap[p]:+.3g}" for p in range(2, 7)))
    checks = [
        ("R(4,2)", 4, 2, kap[4], "k4"),
        ("R(5,4)", 5, 4, kap[5], "k5  (= kappa_M for N=M-1)"),
        ("R(6,3)", 6, 3, kap[6] + 15 * kap[4] * kap[2], "k6 + 15*k4*k2"),
        ("R(6,2)", 6, 2, kap[6] + 15 * kap[4] * kap[2] + 10 * kap[3] ** 2, "k6+15*k4*k2+10*k3^2"),
    ]
    for name, M, N, expect, formula in checks:
        got = factorization_residual(kap, M, N)
        ok = "ok" if abs(got - expect) < 1e-6 else "MISMATCH"
        print(f"  {name} = {got:+.4g}   (= {formula})   [{ok}]")
    print()


def main() -> None:
    # g3 (skew) and g4 (excess kurtosis) are the interpretable headline; higher
    # orders extend the vector but cancel badly at finite m (use float64 + large m).
    cfg = CumulantConfig(window_size=200, order=4, max_scalar_cumulant_order=4,
                         cumulant_estimator="biased", dtype=torch.float64)
    print("Standardized cumulant spectrum is ~ 0 at every order for a Gaussian\n"
          "window; the skew turns on g3 (the leading non-Gaussianity). The Fisher\n"
          "zero built from the same cumulants tracks criticality (its gap) and\n"
          "shape (its phase).\n")
    report("gaussian", cfg)
    report("skewed", cfg)
    fisher_criticality_demo()
    factorization_demo()


if __name__ == "__main__":
    main()
