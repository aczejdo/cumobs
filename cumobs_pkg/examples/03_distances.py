"""Example 03 — distances between measurements (transition detection).

Distances compare a measurement's 2RDM against a reference one, flagging *when*
the loss-covariance structure changes. This example covers all three ideas:

1. The classical distances (fidelity, trace distance) on bare spectra, including
   the identity and two-point closed form.
2. Classical vs quantum eigenvector-sensitivity (decision D2): two states with
   the *same spectrum* but rotated eigenvectors are identical to the classical
   distance and distinct to the quantum one.
3. The monitor computing distances across a synthetic transition, so the trace
   distance spikes exactly when a new mode switches on.

Run::

    python examples/03_distances.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import (
    CumulantConfig,
    CumulantMonitor,
    InMemoryLogger,
    ProbeSet,
    classical_fidelity,
    quantum_fidelity,
    quantum_trace_distance,
    trace_distance,
)


def part1_classical_basics() -> None:
    print("== 1. classical distances on spectra ==")
    p = torch.tensor([0.7, 0.3], dtype=torch.float64)
    q = torch.tensor([0.25, 0.75], dtype=torch.float64)

    # identity: a spectrum against itself
    print(f"  F(p, p)            = {classical_fidelity(p, p):.4f}   (== 1)")
    print(f"  T(p, p)            = {trace_distance(p, p):.4f}   (== 0)")

    # two-point closed form
    f_expected = (math.sqrt(0.7 * 0.25) + math.sqrt(0.3 * 0.75)) ** 2
    print(f"  F(p, q)            = {classical_fidelity(p, q):.4f}   "
          f"(closed form {f_expected:.4f})")
    print(f"  T(p, q)            = {trace_distance(p, q):.4f}   "
          f"(= |0.7 - 0.25| = 0.45)")

    # raw covariance spectra (unnormalized) are normalized internally
    lam_a = torch.tensor([10.0, 1.0, 0.1, 0.0], dtype=torch.float64)
    lam_b = torch.tensor([8.0, 3.0, 0.2, 0.0], dtype=torch.float64)
    print(f"  F(lam_a, lam_b)    = {classical_fidelity(lam_a, lam_b):.4f}   "
          "(spectra are clipped + normalized for you)")


def part2_quantum_sensitivity() -> None:
    print("\n== 2. classical is eigenvector-blind, quantum is not (D2) ==")
    spec = torch.tensor([0.7, 0.3], dtype=torch.float64)
    rho = torch.diag(spec)
    theta = 0.6                                    # rotate the eigenbasis
    R = torch.tensor(
        [[math.cos(theta), -math.sin(theta)],
         [math.sin(theta), math.cos(theta)]],
        dtype=torch.float64,
    )
    sigma = R @ rho @ R.t()                        # same spectrum, rotated vectors

    print(f"  classical_fidelity(spec, spec) = {classical_fidelity(spec, spec):.4f}"
          "   (same spectrum -> identical)")
    print(f"  quantum_fidelity(rho, sigma)   = {quantum_fidelity(rho, sigma):.4f}"
          "   (< 1: sees the rotation)")
    print(f"  quantum_trace_distance         = {quantum_trace_distance(rho, sigma):.4f}"
          "   (> 0: sees the rotation)")


def part3_monitor_transition() -> None:
    print("\n== 3. monitor: trace distance spikes at a transition ==")
    torch.manual_seed(0)
    n, n_a = 24, 8
    v1 = torch.zeros(n, dtype=torch.float64)
    v1[:n_a] = torch.randn(n_a, dtype=torch.float64)
    v1 = v1 / torch.linalg.vector_norm(v1)

    state = {"loss": torch.zeros(n, dtype=torch.float64)}
    probes = ProbeSet(torch.arange(n), lambda m, idx: state["loss"][idx], name="p")
    cfg = CumulantConfig(
        window_size=10, eval_interval=1, update_interval=5,
        top_k=1, distance_flavor="classical", distance_reference="previous",
    )
    logger = InMemoryLogger()
    monitor = CumulantMonitor(None, probes, cfg, logger=logger)

    transition = 25
    print(f"  {'step':>5} {'fidelity':>9} {'trace_dist':>11}")
    for step in range(60):
        if step < transition:
            state["loss"] = 0.01 * torch.randn(n, dtype=torch.float64)
        else:
            a = 5.0 * torch.randn((), dtype=torch.float64)
            state["loss"] = a * v1 + 0.01 * torch.randn(n, dtype=torch.float64)
        meas = monitor.step(step)
        if meas is not None and "trace_distance" in meas.scalars:
            marker = "  <- transition window" if abs(step - transition) <= 5 else ""
            print(f"  {step:>5} {meas.scalars['classical_fidelity']:>9.4f} "
                  f"{meas.scalars['trace_distance']:>11.4f}{marker}")
    print("  (fidelity drops / trace distance jumps as the rank-1 mode turns on)")


def main() -> None:
    part1_classical_basics()
    part2_quantum_sensitivity()
    part3_monitor_transition()


if __name__ == "__main__":
    main()
