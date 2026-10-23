"""Reproduce the Fig. 1 synthetic transition with the cumobs monitor.

We inject a per-sample loss ``l(t) = a(t) * v1 + eps`` where the planted
eigenvector ``v1`` is supported on a single probe group ("A"). Before the
transition the loss is isotropic low-amplitude noise; after it, the strong
rank-1 mode turns on. The monitor should show:

* the spectral heat capacity (SHC) spiking across the transition, and
* the top eigenvector localizing on group A (high block energy).

This is the §10 test-12 behavioral acceptance check, runnable as a script.
Optionally plots if matplotlib is available (``--plot``).

Usage::

    python examples/fig1_transition.py [--plot]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

# allow `python examples/fig1_transition.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import (
    CumulantConfig,
    CumulantMonitor,
    InMemoryLogger,
    LoggingConfig,
    ProbeSet,
)


def run(seed: int = 0, n_steps: int = 80, transition: int = 25):
    torch.manual_seed(seed)
    n, n_a = 24, 8
    labels = ["A"] * n_a + ["B"] * (n - n_a)

    # planted eigenvector, supported only on group A
    v1 = torch.zeros(n, dtype=torch.float64)
    v1[:n_a] = torch.randn(n_a, dtype=torch.float64)
    v1 = v1 / torch.linalg.vector_norm(v1)

    # the "model" here is just a shared mutable loss state the train loop drives
    state = {"loss": torch.zeros(n, dtype=torch.float64)}

    def loss_fn(model, batch_idx):
        return state["loss"][batch_idx]

    probes = ProbeSet(torch.arange(n), loss_fn, group_labels=labels, name="probe")
    cfg = CumulantConfig(
        window_size=10,
        eval_interval=1,
        update_interval=5,
        top_k=1,
        order=4,
        compute_eigenvectors=True,
        per_group=True,
        distance_flavor="classical",
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(
        model=None,
        probes=probes,
        cfg=cfg,
        logger=logger,
        logging_config=LoggingConfig(vector_policy="summary"),
    )

    steps, shc, energy_a, energy_b = [], [], [], []
    for step in range(n_steps):
        if step < transition:
            state["loss"] = 0.01 * torch.randn(n, dtype=torch.float64)
        else:
            a = 5.0 * torch.randn((), dtype=torch.float64)
            state["loss"] = a * v1 + 0.01 * torch.randn(n, dtype=torch.float64)

        meas = monitor.step(step)
        if meas is not None:
            steps.append(step)
            shc.append(meas.scalars["spectral_heat_capacity"])
            energy_a.append(float(meas.vectors["block_energy/A"].sum()))
            energy_b.append(float(meas.vectors["block_energy/B"].sum()))

    return dict(steps=steps, shc=shc, energy_a=energy_a, energy_b=energy_b,
                transition=transition, logger=logger)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    out = run()
    print(f"{'step':>5} {'SHC':>12} {'blockE[A]':>10} {'blockE[B]':>10}")
    for s, h, ea, eb in zip(out["steps"], out["shc"], out["energy_a"], out["energy_b"]):
        marker = "  <- transition" if s == out["transition"] else ""
        print(f"{s:>5} {h:>12.4e} {ea:>10.3f} {eb:>10.3f}{marker}")

    pre = [h for s, h in zip(out["steps"], out["shc"]) if s < out["transition"]]
    post = [h for s, h in zip(out["steps"], out["shc"]) if s >= out["transition"] + 10]
    if pre and post:
        print(f"\nSHC pre-transition (max):  {max(pre):.3e}")
        print(f"SHC post-transition (max): {max(post):.3e}")
        print(f"spike factor: {max(post) / max(pre):.1f}x")
        print(f"final block energy on planted group A: {out['energy_a'][-1]:.3f}")

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n(matplotlib not available; skipping plot)")
            return
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
        ax1.semilogy(out["steps"], out["shc"], "o-")
        ax1.axvline(out["transition"], ls="--", c="k", alpha=0.5)
        ax1.set_ylabel("spectral heat capacity")
        ax2.plot(out["steps"], out["energy_a"], "o-", label="block energy A (planted)")
        ax2.plot(out["steps"], out["energy_b"], "s-", label="block energy B")
        ax2.axvline(out["transition"], ls="--", c="k", alpha=0.5)
        ax2.set_xlabel("training step")
        ax2.set_ylabel("top-eigvec block energy")
        ax2.legend()
        fig.tight_layout()
        fig.savefig("fig1_transition.png", dpi=120)
        print("\nsaved fig1_transition.png")


if __name__ == "__main__":
    main()
