"""Example 06 — the null-background detector and the BBP participation bitmask (§3, §6).

A bare spectral heat-capacity spike answers "did *something* change"; it does not
answer "is this distinguishable from no co-fluctuation." The v0.3 null model
closes that gap:

* **Surrogate (block bootstrap)** — phase-scramble each probe's column
  (independent circular shifts), re-run the spectral path, and score every scalar
  as a z-distance from the no-transition baseline: `<scalar>/null_z`.
* **Marchenko-Pastur / Tracy-Widom** — the analytic bulk edge `mp_edge_upper`;
  eigenvalues above it (`n_spikes`) are genuine detached modes (the BBP
  criterion), carrying `spike_energy_fraction` of the variance.

Once a mode has detached, the **BBP bitmask** (§6) encodes *which probes* it acts
on as a single decodable integer — bit `i` ⇔ probe `i` participates — which, with
group labels, localizes the mode to a probe group.

This script contrasts a planted rank-1 window with a pure-noise window, then
decodes the bitmask back to the planted group.

Run::

    python examples/06_null_model_and_bbp.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import CumulantConfig, CumulantMonitor, InMemoryLogger, ProbeSet, compute_all

N, N_A, M = 40, 6, 24
LABELS = ["A"] * N_A + ["B"] * (N - N_A)            # planted signal lives on group A


def make_buffer(kind: str, seed: int = 0) -> torch.Tensor:
    """`kind="signal"` plants a strong rank-1 mode on group A; `"noise"` is iid."""
    g = torch.Generator().manual_seed(seed)
    if kind == "noise":
        return torch.randn(M, N, generator=g, dtype=torch.float64)
    v1 = torch.zeros(N, dtype=torch.float64)
    v1[:N_A] = torch.randn(N_A, generator=g, dtype=torch.float64)
    a = 6.0 * torch.randn(M, 1, generator=g, dtype=torch.float64)
    return a @ v1.unsqueeze(0) + 0.3 * torch.randn(M, N, generator=g, dtype=torch.float64)


def detector_row(kind: str, cfg: CumulantConfig) -> None:
    meas = compute_all(make_buffer(kind), cfg)
    shc_z = meas.scalars.get("spectral_heat_capacity/null_z", float("nan"))
    pr_z = meas.scalars.get("participation_ratio/null_z", float("nan"))
    edge = meas.scalars["mp_edge_upper"]
    nsp = meas.scalars["n_spikes"]
    frac = meas.scalars["spike_energy_fraction"]
    print(f"  {kind:>7} | SHC null_z={shc_z:+7.2f}  PR null_z={pr_z:+7.2f} | "
          f"mp_edge={edge:7.3f}  n_spikes={nsp:.0f}  spike_energy={frac:.3f}")


def decode_bitmask(cfg: CumulantConfig) -> None:
    print("\n== BBP bitmask (6): which probes does each detached mode act on ==")
    meas = compute_all(make_buffer("signal"), cfg)
    masks = meas.aux.get("active_probe_mask", [])
    counts = meas.vectors.get("active_probe_count")
    if not masks:
        print("  (no detached modes)")
        return
    for a, hexmask in enumerate(masks):
        bits = [i for i in range(N) if (int(hexmask, 16) >> i) & 1]
        groups = sorted({LABELS[i] for i in bits})
        size = int(counts[a]) if counts is not None else len(bits)
        print(f"  mode {a}: mask={hexmask}  |support|={size}  probes={bits}")
        print(f"          -> groups {groups}   (planted on group A = probes 0..{N_A - 1})")
    print(f"  union (active_probe_mask_any) = {meas.aux.get('active_probe_mask_any')}")


def noise_floored_distance_demo() -> None:
    """§4: a distance whose zero point is "indistinguishable from noise".

    The monitor subtracts the surrogate null floor `mu0` (mean drift between
    independent surrogates) from the raw `spectral_drift`: `denoised_spectral_drift`
    sits at ~0 while only noise reshuffles the window, then jumps when a real
    mode switches on.
    """
    print("\n== noise-floored distance (4): drift minus the null floor ==")
    torch.manual_seed(1)
    v1 = torch.zeros(N, dtype=torch.float64)
    v1[:N_A] = torch.randn(N_A, dtype=torch.float64)
    state = {"loss": torch.zeros(N, dtype=torch.float64)}
    probes = ProbeSet(torch.arange(N), lambda m, idx: state["loss"][idx], name="p")
    cfg = CumulantConfig(
        window_size=10, eval_interval=1, update_interval=5,
        null_model="surrogate", null_samples=30, distance_source="covariance",
        dtype=torch.float64,
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(None, probes, cfg, logger=logger)
    print(f"  {'step':>4} {'regime':>8} {'spectral_drift':>15} {'denoised':>10} {'snr':>8}")
    for step in range(40):
        regime = "noise" if step < 20 else "signal"
        if step < 20:
            state["loss"] = 0.3 * torch.randn(N, dtype=torch.float64)
        else:
            state["loss"] = 4.0 * float(torch.randn(())) * v1 + 0.3 * torch.randn(N, dtype=torch.float64)
        meas = monitor.step(step)
        if meas is not None and "denoised_spectral_drift" in meas.scalars:
            print(f"  {step:>4} {regime:>8} {meas.scalars['spectral_drift']:>15.4f} "
                  f"{meas.scalars['denoised_spectral_drift']:>10.4f} "
                  f"{meas.scalars['spectral_drift_snr']:>8.2f}")
    print("  (raw drift is nonzero even for noise; denoised ~0 until the real mode turns on)")


def main() -> None:
    # seed the global RNG so the surrogate circular shifts (and thus the null_z
    # scores) are reproducible run-to-run.
    torch.manual_seed(0)
    cfg = CumulantConfig(
        window_size=M, top_k=3, order=4, compute_eigenvectors=True,
        null_model="both", null_samples=60, rho_keep=0.9, dtype=torch.float64,
    )
    print("A planted rank-1 mode is highly significant vs the phase-scrambled null\n"
          "and detaches above the MP edge; pure noise is not.\n")
    print("== null-background detector (3): signal vs noise ==")
    detector_row("signal", cfg)
    detector_row("noise", cfg)
    decode_bitmask(cfg)
    noise_floored_distance_demo()


if __name__ == "__main__":
    main()
