"""Example 04 — what the *quantum* distances see that nothing else does.

A spectrum is a lossy summary of a covariance: it keeps the eigen*values* and
throws away the eigen*vectors*. Most of the catalog (SHC, PR, entropy, the
classical fidelity / trace distance, even the scale-sensitive `spectral_drift`)
is a function of the spectrum alone. The **quantum** distances and the **CKA**
distance are the only observables here that see the eigen*basis* — so they are
the only ones that can detect a change in which probes co-fluctuate when the
spectrum stays put.

We drive a small system through three qualitatively different changes and watch
which observables react. The "system": `n = 12` probes split into two halves
`A = {0..5}` and `B = {6..11}`. Its fluctuations have a covariance (the 2RDM)

    C(α, θ, λ₂) = α² · [ λ₁ · u(θ)u(θ)ᵀ + σ² I ]  +  λ₂ · w wᵀ ,
    u(θ) = cosθ · u_A + sinθ · u_B           (a unit "dominant mode")

with `u_A`/`u_B` the uniform direction on half A/B and `w` a contrast mode on B.
Think of it as a fluctuating field with one dominant mode plus an isotropic
background. We sample windows from `C` and feed them through the usual pipeline.

The three regimes, each compared against the **baseline** (mode on A, α=1):

1. **Scale** (α: 1→3) — the fluctuations get *louder*; eigenvectors and the
   *shape* of the spectrum are unchanged, only its overall magnitude grows.
2. **Isospectral rotation** (θ: 0→π/2) — the dominant mode *migrates from half A
   to half B*. The eigenvalues are **exactly unchanged** (u(θ) stays unit norm);
   only the eigenvector rotates.
3. **Reshape** (λ₂: 0→8) — a genuine *second* mode grows on B: the spectrum
   itself changes (rank/shape), not just its basis.

Two ways to run it:

    python examples/04_quantum_isospectral_rotation.py          # the regime table
    python examples/04_quantum_isospectral_rotation.py --plot   # + a trackable
                                                                #   training plot

The `--plot` mode runs the same changes *continuously* through a real
sliding-window `CumulantMonitor` (as you would during training) and plots a
three-signal **decomposition** of the change plus the per-group localization —
saving `04_quantum_isospectral_rotation.png`. (Needs `matplotlib`; takes ~1 min.)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import (
    CumulantConfig,
    CumulantMonitor,
    InMemoryLogger,
    ProbeSet,
    classical_fidelity,
    cka_distance,
    compute_all,
    quantum_fidelity,
    spectral_drift,
    trace_distance,
)

torch.set_default_dtype(torch.float64)

N = 12
A = np.arange(0, 6)
B = np.arange(6, 12)
LAM1, SIG2 = 10.0, 0.1

_uA = np.zeros(N); _uA[A] = 1.0; _uA /= np.linalg.norm(_uA)
_uB = np.zeros(N); _uB[B] = 1.0; _uB /= np.linalg.norm(_uB)
_w = np.zeros(N); _w[B] = [1, -1, 1, -1, 1, -1.0]; _w /= np.linalg.norm(_w)


def covariance(alpha: float, theta: float, lam2: float) -> np.ndarray:
    """The true 2RDM for the current regime knobs."""
    u = math.cos(theta) * _uA + math.sin(theta) * _uB
    C = alpha**2 * (LAM1 * np.outer(u, u) + SIG2 * np.eye(N))
    return C + lam2 * np.outer(_w, _w)


def sample_window(C: np.ndarray, m: int, rng: np.random.Generator) -> torch.Tensor:
    """Draw an (m, n) window of fluctuations with covariance C."""
    L = np.linalg.cholesky(C)
    return torch.tensor(rng.standard_normal((m, N)) @ L.T)


# ---------------------------------------------------------------------------
# Part 1 — the discrete regime table (the conceptual core)
# ---------------------------------------------------------------------------

def regime_table() -> None:
    groups = {"A": torch.tensor(A), "B": torch.tensor(B)}
    cfg = CumulantConfig(
        window_size=2000, order=2,
        materialize_matrices=True,    # quantum distances need C
        compute_eigenvectors=True,    # CKA + block energy need V
        per_group=True, top_k=2,
    )
    rng = np.random.default_rng(0)

    ref = compute_all(sample_window(covariance(1, 0, 0), 6000, rng), cfg, groups=groups)
    r_spec = ref.vectors["covariance_spectrum"]
    r_C = ref.matrices["covariance"]
    r_V, r_lam = ref.factors["V"], ref.factors["lam_top"]

    schedule = [
        ("0  baseline (mode on A)", 1, 0.0, 0),
        ("0  steady (noise floor)", 1, 0.0, 0),
        ("1  scale x2", 2, 0.0, 0),
        ("1  scale x3", 3, 0.0, 0),
        ("2  rotate 45deg", 1, math.pi / 4, 0),
        ("2  rotate 90 (A->B)", 1, math.pi / 2, 0),
        ("3  reshape lam2=4", 1, math.pi / 2, 4),
        ("3  reshape lam2=8", 1, math.pi / 2, 8),
    ]

    print("Each regime is compared against the baseline measurement.\n")
    print("                          |---- spectrum-only ----|  |-- basis-aware --|")
    print(f"{'regime':25s}{'SHC':>9}{'PR':>6}{'Hvn':>6}{'cfid':>7}{'sdrift':>8}"
          f"{'  ':>2}{'qfid':>7}{'cka':>7}{'blkA':>7}{'blkB':>6}")
    print("-" * 92)
    for label, alpha, theta, lam2 in schedule:
        C = covariance(alpha, theta, lam2)
        acc: dict[str, float] = defaultdict(float)
        R, m = 5, 2000
        for _ in range(R):
            meas = compute_all(sample_window(C, m, rng), cfg, groups=groups)
            spec = meas.vectors["covariance_spectrum"]
            acc["SHC"] += meas.scalars["spectral_heat_capacity"]
            acc["PR"] += meas.scalars["participation_ratio"]
            acc["Hvn"] += meas.scalars["von_neumann_entropy"]
            acc["cfid"] += classical_fidelity(r_spec, spec)
            acc["sdrift"] += spectral_drift(r_spec, spec)
            acc["qfid"] += quantum_fidelity(r_C, meas.matrices["covariance"])
            acc["cka"] += cka_distance(r_V, r_lam, meas.factors["V"], meas.factors["lam_top"])
            acc["blkA"] += float(meas.vectors["block_energy/A"][0])
            acc["blkB"] += float(meas.vectors["block_energy/B"][0])
        d = {k: v / R for k, v in acc.items()}
        print(f"{label:25s}{d['SHC']:9.2f}{d['PR']:6.2f}{d['Hvn']:6.2f}"
              f"{d['cfid']:7.3f}{d['sdrift']:8.3f}{'  ':>2}{d['qfid']:7.3f}"
              f"{d['cka']:7.3f}{d['blkA']:7.2f}{d['blkB']:6.2f}")

    print("""
Reading the table (each row vs. the baseline 'mode on A'):

  SCALE (regime 1) -- the fluctuations get louder, nothing reorganizes:
     SHC and spectral_drift jump; classical fidelity, PR, entropy, the quantum
     distance and CKA are all ~unchanged (they are scale-invariant).

  ISOSPECTRAL ROTATION (regime 2) -- the punchline:
     The dominant mode migrates from half A to half B, but the eigenVALUES never
     move. So SHC, PR, entropy, classical_fidelity (=1.0) AND spectral_drift
     (~noise floor) ALL say 'steady state'. Only the basis-aware observables
     react: quantum_fidelity collapses, CKA distance -> 1, block energy hands
     off A:1->0, B:0->1.  >>> invisible to every spectral summary. <<<

  RESHAPE (regime 3) -- a genuine new mode appears: the spectrum itself changes,
     so EVERYTHING moves (PR 1.2->2.2, entropy up, classical_fidelity drops).

Sensitivity ladder (what each observable can detect):

                              scale     isospectral-rotation   spectral-reshape
    classical fid / TV        blind          blind                 detects
    PR / entropy / top-frac   blind          blind                 detects
    spectral_heat_capacity    DETECTS        blind                 detects
    spectral_drift            DETECTS        blind                 detects
    quantum fid / CKA         blind          DETECTS               detects
    block_energy (per-group)  blind          DETECTS               detects
""")


# ---------------------------------------------------------------------------
# Part 2 — the same changes as a continuous training trajectory (+ plot)
# ---------------------------------------------------------------------------

# regime bands on the (continuous) trajectory, in training steps
SCALE_BAND = (250, 520)
ROT_BAND = (580, 800)
RESHAPE_BAND = (860, 1040)
T_TRAJ = 1040
W_TRAJ = 100


def _smooth(t: float, t0: float, t1: float) -> float:
    x = min(1.0, max(0.0, (t - t0) / (t1 - t0)))
    return x * x * (3 - 2 * x)


def _knobs(t: int) -> tuple[float, float, float]:
    alpha = 1 + 1.5 * (_smooth(t, 250, 400) - _smooth(t, 400, 520))   # up then back
    theta = (math.pi / 2) * _smooth(t, *ROT_BAND)
    lam2 = 8.0 * _smooth(t, *RESHAPE_BAND)
    return alpha, theta, lam2


def _run_one(seed: int) -> dict[int, dict]:
    """One sliding-window monitor run over the continuous schedule."""
    rng = np.random.default_rng(seed)
    state = {"loss": torch.zeros(N)}
    probes = ProbeSet(
        torch.arange(N), lambda m, b: state["loss"][b],
        group_labels=["A"] * 6 + ["B"] * 6, name="p",
    )
    cfg = CumulantConfig(
        window_size=W_TRAJ, eval_interval=1, update_interval=1, order=2, top_k=1,
        materialize_matrices=True, compute_eigenvectors=True, per_group=True,
        compute_cka_distance=True, distance_flavor="both",
        distance_reference="baseline",
    )
    mon = CumulantMonitor(None, probes, cfg, logger=InMemoryLogger())
    rows: dict[int, dict] = {}
    for t in range(T_TRAJ):
        alpha, theta, lam2 = _knobs(t)
        L = np.linalg.cholesky(covariance(alpha, theta, lam2))
        state["loss"] = torch.tensor(L @ rng.standard_normal(N))
        meas = mon.step(t)
        if meas is None or "quantum_trace_distance" not in meas.scalars:
            continue
        s = meas.scalars
        rows[t] = dict(
            sdrift=s["spectral_drift"],
            ctd=s["trace_distance"],                       # classical (spectrum-shape)
            reorg=s["quantum_trace_distance"] - s["trace_distance"],  # basis-only
            cka=s["cka_distance"],
            erank=s["effective_rank"],
            bA=float(meas.vectors["block_energy/A"][0]),
            bB=float(meas.vectors["block_energy/B"][0]),
        )
    return rows


def trajectory(n_seeds: int = 6) -> dict:
    """Average the trajectory metrics over a few probe replicas (denoise)."""
    runs = [_run_one(s) for s in range(n_seeds)]
    steps = sorted(set.intersection(*[set(r) for r in runs]))
    keys = ["sdrift", "ctd", "reorg", "cka", "erank", "bA", "bB"]
    out = {"steps": np.array(steps)}
    for k in keys:
        out[k] = np.array([float(np.mean([r[t][k] for r in runs])) for t in steps])
    return out


def make_plot(tj: dict, path: str) -> None:
    import matplotlib.pyplot as plt

    steps = tj["steps"]
    bands = [(SCALE_BAND, "amplitude ↑", "#fde0c0"),
             (ROT_BAND, "basis rotation A→B", "#cfe8cf"),
             (RESHAPE_BAND, "new mode", "#d6dcf5")]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 7))

    for (lo, hi), label, color in bands:
        for ax in (ax1, ax2):
            ax.axvspan(lo, hi, color=color, alpha=0.7, lw=0)
        ax1.text((lo + hi) / 2, 0.97, label, ha="center", va="top",
                 transform=ax1.get_xaxis_transform(), fontsize=9, color="#444")

    # Panel 1: decompose the change into three (trackable) scalar signals.
    ax1.plot(steps, tj["sdrift"], color="#d2691e", lw=2, label="spectral_drift  (amplitude)")
    ax1.plot(steps, tj["ctd"], color="#2e7d32", lw=2, label="trace_distance  (spectrum shape)")
    ax1.plot(steps, tj["reorg"], color="#1f4fd6", lw=2.4,
             label="quantum − classical TD  (basis reorganization)")
    ax1.plot(steps, tj["cka"], color="#1f4fd6", lw=1.2, ls=":", alpha=0.8,
             label="cka_distance  (cheap basis proxy)")
    ax1.set_ylabel("distance vs. baseline")
    ax1.set_title("Decomposing a change: each signal fires for a different kind of change")
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax1.set_ylim(-0.03, 1.05)

    # Panel 2: where the structure went (block energy) + how complex (eff. rank).
    ax2.plot(steps, tj["bA"], color="#c62828", lw=2, label="block_energy / A")
    ax2.plot(steps, tj["bB"], color="#6a1b9a", lw=2, label="block_energy / B")
    ax2.set_ylabel("top-mode block energy")
    ax2.set_xlabel("training step")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(loc="center left", fontsize=8, framealpha=0.9)
    ax2b = ax2.twinx()
    ax2b.plot(steps, tj["erank"], color="#555", lw=1.6, ls="--", label="effective_rank")
    ax2b.set_ylabel("effective_rank")
    ax2b.legend(loc="center right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    print(f"saved {path}")


def trajectory_summary(tj: dict) -> None:
    print("\nContinuous trajectory (sliding-window monitor), key signals by regime:")
    print(f"{'step':>6}{'sdrift':>9}{'classTD':>9}{'reorg':>8}{'cka':>7}"
          f"{'erank':>8}{'blkA':>7}{'blkB':>7}")
    for t in tj["steps"][::80]:
        i = int(np.where(tj["steps"] == t)[0][0])
        print(f"{int(t):>6}{tj['sdrift'][i]:>9.3f}{tj['ctd'][i]:>9.3f}"
              f"{tj['reorg'][i]:>8.3f}{tj['cka'][i]:>7.3f}{tj['erank'][i]:>8.2f}"
              f"{tj['bA'][i]:>7.2f}{tj['bB'][i]:>7.2f}")
    print("""
The interpretable, trackable decomposition (all three are plain logged scalars):
  * spectral_drift                 fires on AMPLITUDE changes (scale band only).
  * trace_distance  (classical)    fires on SPECTRUM-SHAPE changes (reshape band).
  * quantum_TD - classical_TD      isolates BASIS reorganization -- it is the only
                                   one that lights up in the rotation band, where
                                   the loss-fluctuation spectrum never moves.
  (cka_distance tracks the same basis signal at O(n k^2) instead of O(n^3).)
Panel 2 shows WHERE: the top mode hands off A->B during the rotation, and
effective_rank climbs only when the genuine new mode appears.""")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true",
                        help="also run the continuous trajectory and save a figure")
    args = parser.parse_args()

    regime_table()

    if args.plot:
        print("Running the continuous training trajectory (sliding-window monitor)...")
        tj = trajectory(n_seeds=6)
        trajectory_summary(tj)
        out_png = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "04_quantum_isospectral_rotation.png",
        )
        try:
            make_plot(tj, out_png)
        except ImportError:
            print("\n(matplotlib not available; printed the summary above instead)")


if __name__ == "__main__":
    main()
