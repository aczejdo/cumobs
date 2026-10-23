"""Shared helpers for the cumobs test suite.

Brute-force cumulant tensors used by the full-contraction / contracted-matrix
identity tests (§10 tests 3-4).  These deliberately materialize the dense
``n^k`` co-moment tensors at small ``n`` so they can check the window-space
shortcuts against the literal definitions in §3.4-3.5.
"""

from __future__ import annotations

import torch


def center(B: torch.Tensor) -> torch.Tensor:
    """Per-feature (column) mean centering, the 1/m mean used by the package."""
    return B - B.mean(dim=0)


def kappa2(B_c: torch.Tensor, D: float) -> torch.Tensor:
    """Second cumulant (central moment) tensor ``kappa2[i,j] = E[d_i d_j]``."""
    return torch.einsum("ti,tj->ij", B_c, B_c) / D


def kappa3(B_c: torch.Tensor, D: float) -> torch.Tensor:
    """Third cumulant ``kappa3[i,j,k] = E[d_i d_j d_k]`` (no subtraction)."""
    return torch.einsum("ti,tj,tk->ijk", B_c, B_c, B_c) / D


def kappa4_connected(B_c: torch.Tensor, D: float) -> torch.Tensor:
    """Connected 4th cumulant (Wick/Isserlis subtraction), §3.5."""
    raw = torch.einsum("ti,tj,tk,tl->ijkl", B_c, B_c, B_c, B_c) / D
    k2 = kappa2(B_c, D)
    wick = (
        torch.einsum("ij,kl->ijkl", k2, k2)
        + torch.einsum("ik,jl->ijkl", k2, k2)
        + torch.einsum("il,jk->ijkl", k2, k2)
    )
    return raw - wick
