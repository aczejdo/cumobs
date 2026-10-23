"""Example 01 — the pure observable core, no model and no monitor.

The shortest path into `cumobs`: build a buffer ``B in R^{m x n}`` by hand and
call :func:`cumobs.compute_all` on it. This is exactly what the monitor does per
measurement, minus the scheduling — ideal for notebook / offline analysis of
losses you have already logged.

We plant a controllable structure into the buffer so the printed numbers are
interpretable:

    B[t, :] = a[t] * v1  +  noise          (a strong rank-1 mode on group "A")

so you should see a large leading covariance eigenvalue, a participation ratio
near 1 (one dominant mode), and — with ``compute_eigenvectors=True`` — the top
eigenvector's block energy concentrated on group A.

Run::

    python examples/01_minimal_observables.py
"""

from __future__ import annotations

import os
import sys

import torch

# allow `python examples/01_minimal_observables.py` from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cumobs import CumulantConfig, compute_all


def make_buffer(m: int = 20, n: int = 40, n_a: int = 12, seed: int = 0):
    """A window with one planted rank-1 mode supported on the first ``n_a`` probes."""
    torch.manual_seed(seed)
    # planted eigenvector v1, nonzero only on group "A" (first n_a probes)
    v1 = torch.zeros(n, dtype=torch.float64)
    v1[:n_a] = torch.randn(n_a, dtype=torch.float64)
    v1 = v1 / torch.linalg.vector_norm(v1)

    a = 3.0 * torch.randn(m, dtype=torch.float64)          # per-row amplitude
    noise = 0.05 * torch.randn(m, n, dtype=torch.float64)
    B = torch.outer(a, v1) + noise                          # (m, n)

    labels = ["A"] * n_a + ["B"] * (n - n_a)
    groups = _groups_from_labels(labels)
    return B, groups


def _groups_from_labels(labels):
    """label -> LongTensor of column indices (what ProbeSet.groups() returns)."""
    out: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        out.setdefault(lab, []).append(i)
    return {lab: torch.tensor(idx, dtype=torch.long) for lab, idx in out.items()}


def main() -> None:
    B, groups = make_buffer()
    m, n = B.shape

    # Turn on everything so the full catalog is populated.
    cfg = CumulantConfig(
        window_size=m,
        order=4,
        max_scalar_cumulant_order=6,  # -> total_cumulant_5, total_cumulant_6 (§3.8)
        top_k=3,
        compute_eigenvectors=True,   # -> IPR, block energy
        materialize_matrices=True,   # -> C, M3, K in meas.matrices
        per_group=True,              # -> group_* observables
        dtype=torch.float64,
    )

    meas = compute_all(B, cfg, groups=groups)

    print(f"buffer B: m={m} rows (window) x n={n} cols (probes)\n")

    print("== scalars ==")
    for key in sorted(meas.scalars):
        print(f"  {key:<32} {meas.scalars[key]:+.6g}")

    print("\n== vectors (showing first few entries) ==")
    for key in sorted(meas.vectors):
        vec = meas.vectors[key]
        flat = vec.flatten()                     # some keys are 2-D, e.g. (top_k, q-2)
        head = ", ".join(f"{x:+.4g}" for x in flat[:4].tolist())
        more = " ..." if flat.numel() > 4 else ""
        print(f"  {key:<32} shape={tuple(vec.shape)}  [{head}{more}]")

    print("\n== matrices (opt-in, materialize_matrices=True) ==")
    for key in sorted(meas.matrices):
        print(f"  {key:<32} shape={tuple(meas.matrices[key].shape)}")

    print("\n== health ==")
    for key in sorted(meas.health):
        print(f"  {key:<32} {meas.health[key]:.4g}")

    # --- a couple of interpreted readouts -------------------------------------
    spec = meas.vectors["covariance_spectrum"]
    pr = meas.scalars["participation_ratio"]
    print("\n== interpretation ==")
    print(f"  leading eigenvalue lambda_1   = {spec[0]:.4g}")
    print(f"  2nd eigenvalue lambda_2       = {spec[1]:.4g}   (>> smaller => rank-1)")
    print(f"  participation ratio           = {pr:.3f}   (near 1 => one dominant mode)")
    be_a = float(meas.vectors["block_energy/A"][0])
    be_b = float(meas.vectors["block_energy/B"][0])
    print(f"  block energy of v1 on group A = {be_a:.3f}")
    print(f"  block energy of v1 on group B = {be_b:.3f}   (mode localizes on A)")


if __name__ == "__main__":
    main()
