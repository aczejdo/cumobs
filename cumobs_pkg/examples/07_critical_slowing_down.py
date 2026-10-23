"""Example 07 — critical slowing down, mode tracking, and the geodesic distance (§5, §7, §2).

The paper's early-warning story is "diverging variance **AND** rising
autocorrelation." Spectral heat capacity is the variance half; the v0.3 batch
adds the autocorrelation half — a genuine new *leading* indicator:

    autocorr_lag1, autocorr_time   (of the aggregate S, always-on)

As a system approaches a transition its dominant mode relaxes more slowly, so the
window's lag-1 autocorrelation rises *before* (or together with) the variance
spike. This script drives a synthetic 2RDM through three regimes and watches the
monitor:

    1. fast iid noise            -> low SHC, low autocorr
    2. approaching transition    -> an AR(1) mode whose persistence phi ramps up
                                    => autocorr rises and variance grows (CSD)
    3. a second mode switches on  -> eigenvector tracking flags a crossing

Alongside it logs the **geodesic** `fisher_rao_distance` (§2, the metric form of
the classical fidelity) and the **alignment** `align_cov_skew` (§7).

Run::

    python examples/07_critical_slowing_down.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import CumulantConfig, CumulantMonitor, InMemoryLogger, ProbeSet


def run() -> InMemoryLogger:
    torch.manual_seed(0)
    n, n_a = 24, 8
    v1 = torch.zeros(n, dtype=torch.float64)
    v1[:n_a] = torch.randn(n_a, dtype=torch.float64)
    v1 /= torch.linalg.vector_norm(v1)
    v2 = torch.zeros(n, dtype=torch.float64)
    v2[n_a:2 * n_a] = torch.randn(n_a, dtype=torch.float64)   # disjoint second mode
    v2 /= torch.linalg.vector_norm(v2)

    state = {"loss": torch.zeros(n, dtype=torch.float64)}
    probes = ProbeSet(torch.arange(n), lambda m, idx: state["loss"][idx], name="p")
    cfg = CumulantConfig(
        window_size=12, eval_interval=1, update_interval=4, top_k=2, order=4,
        compute_eigenvectors=True, track_modes=True,
        distance_flavor="classical", dtype=torch.float64,
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(None, probes, cfg, logger=logger)

    amp = 0.0
    for step in range(120):
        if step < 36:                       # 1. fast iid noise
            state["loss"] = 0.2 * torch.randn(n, dtype=torch.float64)
        elif step < 84:                     # 2. AR(1) mode, persistence ramps up
            phi = 0.2 + 0.75 * (step - 36) / 48.0          # 0.2 -> ~0.95
            amp = phi * amp + math.sqrt(1.0 - phi * phi) * float(torch.randn(()))
            state["loss"] = 1.5 * amp * v1 + 0.1 * torch.randn(n, dtype=torch.float64)
        else:                               # 3. dominance alternates -> crossings
            # every 12 steps the larger mode switches v1<->v2; the tracked mode
            # identity reshuffles, so crossing_detected fires at each handoff.
            big, small = (v2, v1) if ((step - 84) // 12) % 2 == 0 else (v1, v2)
            a_big = 3.0 * float(torch.randn(()))
            a_small = 0.6 * float(torch.randn(()))
            state["loss"] = a_big * big + a_small * small + 0.1 * torch.randn(n, dtype=torch.float64)
        monitor.step(step)
    return logger


def main() -> None:
    logger = run()
    rows = [r for r in logger.history if "spectral_heat_capacity" in r]

    print("Regimes: [noise] steps<36   [approach] 36-83   [two-mode] 84+\n")
    hdr = (f"  {'step':>4} {'regime':>9} {'SHC':>9} {'ac_lag1':>8} {'ac_time':>8} "
           f"{'eff_win':>8} {'fisher_rao':>11} {'align_skew':>11} {'cross':>6}")
    print(hdr)
    for r in rows:
        s = r["step"]
        regime = "noise" if s < 36 else ("approach" if s < 84 else "two-mode")
        fr = r.get("fisher_rao_distance", float("nan"))
        print(f"  {s:>4} {regime:>9} {r['spectral_heat_capacity']:>9.3g} "
              f"{r['autocorr_lag1']:>8.3f} {r['autocorr_time']:>8.2f} "
              f"{r['health/effective_window']:>8.2f} {fr:>11.4f} "
              f"{r.get('align_cov_skew', float('nan')):>11.3f} "
              f"{r.get('crossing_detected', 0.0):>6.0f}")

    # summarize the CSD lead: mean lag-1 autocorr per regime
    def mean_ac(lo, hi):
        v = [r["autocorr_lag1"] for r in rows if lo <= r["step"] < hi]
        return sum(v) / len(v) if v else float("nan")
    print("\n  critical slowing down: mean autocorr_lag1 rises toward the transition")
    print(f"    noise    (steps <36): {mean_ac(0, 36):+.3f}")
    print(f"    approach (36-83):     {mean_ac(36, 84):+.3f}   <- rising = early warning")
    n_cross = sum(1 for r in rows if r.get("crossing_detected", 0.0) == 1.0)
    print(f"  eigenvector tracking flagged {n_cross} crossing(s) "
          "(mode identity reshuffles when the 2nd mode dominates)")


if __name__ == "__main__":
    main()
